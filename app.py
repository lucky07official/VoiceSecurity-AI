import os, io, json, time, hmac, hashlib, secrets, sqlite3, smtplib, mimetypes, uuid
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

APP_NAME = "VoiceGuard AI"
DB_PATH = os.getenv("VOICEGUARD_DB", "voiceguard.db")
DETECTOR_ID = "garystafford/wav2vec2-deepfake-voice-detector"
TARGET_SR = 16000
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_DURATION_SEC = int(os.getenv("MAX_DURATION_SEC", "300"))
st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at 15% 0%,#153252 0%,#0a1323 44%,#080f1b 100%);color:#eaf2ff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:#f4f8ff!important}
.block-container{max-width:1500px;padding-top:1.4rem;padding-bottom:3rem}
section[data-testid="stSidebar"]{background:#0b1627}
.hero,.card{border:1px solid rgba(66,217,232,.25);border-radius:18px;padding:22px;background:linear-gradient(120deg,rgba(20,54,82,.92),rgba(16,29,49,.92));margin-bottom:18px}
.kicker{color:#42d9e8;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;font-size:.75rem}
.metric{background:#13243a;border:1px solid #29415c;border-radius:14px;padding:16px}.stDataFrame{border:1px solid #29415c;border-radius:12px}.stDownloadButton button{border-radius:10px}
div.stButton>button{background:linear-gradient(90deg,#087f91,#13a7ad);color:white;border:0;border-radius:10px;font-weight:700}
</style>
""", unsafe_allow_html=True)

def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
          id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
          phone TEXT DEFAULT '', password_hash TEXT NOT NULL, role TEXT DEFAULT 'user',
          verified INTEGER DEFAULT 0, is_blocked INTEGER DEFAULT 0, two_fa INTEGER DEFAULT 0,
          created_at TEXT NOT NULL, last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS otp_codes(
          id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL, purpose TEXT NOT NULL,
          code_hash TEXT NOT NULL, expires_at REAL NOT NULL, attempts INTEGER DEFAULT 0,
          created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS login_events(
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, email TEXT, event TEXT,
          ip TEXT DEFAULT 'Unavailable', user_agent TEXT DEFAULT 'Unavailable', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analyses(
          id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, filename TEXT, prediction TEXT,
          real_score REAL, fake_score REAL, metrics_json TEXT, sha256 TEXT,
          created_at TEXT NOT NULL
        );
        """)
        # Admin is provisioned from environment only; no hard-coded default password.
        admin_email = os.getenv("ADMIN_EMAIL", "").strip().lower()
        admin_password = os.getenv("ADMIN_PASSWORD", "")
        if admin_email and admin_password and not c.execute("SELECT 1 FROM users WHERE email=?", (admin_email,)).fetchone():
            c.execute("INSERT INTO users(name,email,password_hash,role,verified,created_at) VALUES(?,?,?,?,?,?)",
                      ("Administrator", admin_email, hash_password(admin_password), "admin", 1, now()))

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310000)
    return salt.hex() + "$" + digest.hex()

def verify_password(password, stored):
    try:
        salt_hex, digest_hex = stored.split("$", 1)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 310000).hex()
        return hmac.compare_digest(candidate, digest_hex)
    except Exception:
        return False

def send_email(to, subject, body):
    host=os.getenv("SMTP_HOST"); port=int(os.getenv("SMTP_PORT","587"))
    username=os.getenv("SMTP_USERNAME"); password=os.getenv("SMTP_PASSWORD")
    sender=os.getenv("SMTP_FROM", username or "")
    if not all([host, username, password, sender]):
        return False, "Email is not configured. Add SMTP settings in the .env file."
    msg=EmailMessage(); msg["Subject"]=subject; msg["From"]=sender; msg["To"]=to; msg.set_content(body)
    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls(); server.login(username,password); server.send_message(msg)
        return True, "Email sent."
    except Exception as e:
        return False, f"Email delivery failed: {e}"

def issue_otp(email, purpose):
    with db() as c:
        recent=c.execute("SELECT created_at FROM otp_codes WHERE email=? AND purpose=? ORDER BY id DESC LIMIT 1",(email,purpose)).fetchone()
        if recent and time.time()-recent["created_at"] < 60:
            return False, "Please wait 60 seconds before requesting another code."
        code=f"{secrets.randbelow(1000000):06d}"
        digest=hashlib.sha256((code+os.getenv("OTP_SECRET","change-this-secret")).encode()).hexdigest()
        c.execute("DELETE FROM otp_codes WHERE email=? AND purpose=?", (email,purpose))
        c.execute("INSERT INTO otp_codes(email,purpose,code_hash,expires_at,created_at) VALUES(?,?,?,?,?)",
                  (email,purpose,digest,time.time()+600,time.time()))
    support_email = os.getenv("SUPPORT_EMAIL", os.getenv("SMTP_USERNAME", ""))
    email_body = f"""Hello,

Your VoiceGuard AI {purpose.replace("_", " ").title()} OTP is: {code}
This code is valid for 10 minutes only. Please do not share it with anyone.

Important notes:
1. Enter this OTP only on the official VoiceGuard AI website.
2. VoiceGuard support will never ask you to share your OTP or password.
3. If the code expires, request a fresh OTP from the website.
4. If you did not request this code, ignore this email and secure your account.
5. Check your Inbox, Spam and Promotions folders if you cannot find future emails.
6. Never forward this message or share its verification code with anyone.
7. For login, signup or password-reset issues, contact our support team.
8. Email support: {support_email or "Please configure SUPPORT_EMAIL in Streamlit Secrets"}.

Regards,
VoiceGuard AI Support Team
"""
    purpose_label = purpose.replace("_", " ").title()
    ok,msg=send_email(email, f"Your VoiceGuard AI {purpose_label} OTP", email_body)
    if not ok:
        # Do not leave an unusable OTP behind when SMTP delivery fails.
        with db() as c:
            c.execute("DELETE FROM otp_codes WHERE email=? AND purpose=?", (email,purpose))
    return ok,msg

def verify_otp(email,purpose,code):
    with db() as c:
        row=c.execute("SELECT * FROM otp_codes WHERE email=? AND purpose=? ORDER BY id DESC LIMIT 1",(email,purpose)).fetchone()
        if not row or row["expires_at"] < time.time() or row["attempts"] >= 5:
            return False
        digest=hashlib.sha256((code+os.getenv("OTP_SECRET","change-this-secret")).encode()).hexdigest()
        if hmac.compare_digest(digest,row["code_hash"]):
            c.execute("DELETE FROM otp_codes WHERE email=? AND purpose=?", (email,purpose))
            return True
        c.execute("UPDATE otp_codes SET attempts=attempts+1 WHERE id=?", (row["id"],))
        return False

def log_event(user_id,email,event):
    try:
        headers = st.context.headers
        forwarded = headers.get("X-Forwarded-For", "")
        ip = forwarded.split(",")[0].strip() if forwarded else headers.get("X-Real-IP", "Unavailable")
        ua = headers.get("User-Agent", "Unavailable")
    except Exception:
        ip, ua = "Unavailable", "Unavailable"
    with db() as c:
        c.execute("INSERT INTO login_events(user_id,email,event,ip,user_agent,created_at) VALUES(?,?,?,?,?,?)",(user_id,email,event,ip,ua,now()))

def current_user():
    if "user_id" not in st.session_state: return None
    with db() as c:
        return c.execute("SELECT * FROM users WHERE id=?", (st.session_state.user_id,)).fetchone()

@st.cache_resource(show_spinner="Loading pretrained voice detector (first run downloads model)...")
def load_detector():
    extractor=AutoFeatureExtractor.from_pretrained(DETECTOR_ID)
    model=AutoModelForAudioClassification.from_pretrained(DETECTOR_ID)
    model.eval()
    return extractor,model

def decode_audio(raw, suffix):
    import tempfile
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as f:
            f.write(raw); path=f.name
        audio,sr=librosa.load(path,sr=TARGET_SR,mono=True)
    finally:
        if path and os.path.exists(path): os.remove(path)
    if audio.size==0: raise ValueError("Audio is empty or unsupported.")
    duration=len(audio)/sr
    if duration>MAX_DURATION_SEC: raise ValueError(f"Audio exceeds {MAX_DURATION_SEC} seconds.")
    return audio.astype(np.float32),sr

def classify(audio,sr):
    extractor,model=load_detector()
    inputs=extractor(audio,sampling_rate=sr,return_tensors="pt",padding=True)
    with torch.inference_mode(): probs=torch.softmax(model(**inputs).logits,dim=-1)[0].cpu().numpy()
    labels={int(k):str(v) for k,v in (model.config.id2label or {}).items()}
    scores={labels.get(i,f"class_{i}"):float(p) for i,p in enumerate(probs)}
    fake=next((v for k,v in scores.items() if "fake" in k.lower() or "spoof" in k.lower()),None)
    real=next((v for k,v in scores.items() if "real" in k.lower() or "bonafide" in k.lower()),None)
    return scores,max(scores,key=scores.get),real,fake

def metrics(audio,sr):
    return {"Duration (seconds)":round(len(audio)/sr,2),"Sample rate (Hz)":sr,
      "RMS energy":round(float(np.sqrt(np.mean(audio**2))),5),
      "Peak amplitude":round(float(np.max(np.abs(audio))),5),
      "Zero-crossing rate":round(float(np.mean(librosa.feature.zero_crossing_rate(y=audio))),5),
      "Spectral centroid (Hz)":round(float(np.mean(librosa.feature.spectral_centroid(y=audio,sr=sr))),2)}

def make_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    b=io.BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4); styles=getSampleStyleSheet()
    rows=[["Analysis ID",result["id"]],["File",result["filename"]],["Timestamp",result["created_at"]],["Prediction",result["prediction"]]]
    if result["real_score"] is not None: rows += [["Real raw score",f'{result["real_score"]*100:.2f}%'],["Fake raw score",f'{result["fake_score"]*100:.2f}%']]
    rows += [[k,str(v)] for k,v in result["metrics"].items()]
    table=Table(rows,colWidths=[170,330]); table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.4,colors.grey),("PADDING",(0,0),(-1,-1),8)]))
    doc.build([Paragraph("VoiceGuard AI — Analysis Report",styles["Title"]),Spacer(1,18),table,Spacer(1,18),
      Paragraph("Important: model scores are raw outputs, not calibrated probabilities. This result is not proof of authenticity, identity, or fraud. False positives and false negatives are possible.",styles["BodyText"])])
    return b.getvalue()

def auth_ui():
    st.markdown('<div class="hero"><div class="kicker">Audio authenticity screening platform</div><h1>🛡️ VoiceGuard AI</h1><p>AI voice-clone and deepfake audio screening workspace.</p></div>',unsafe_allow_html=True)
    st.info("🔒 Email OTP verification is required for every login. New accounts must verify their email before accessing VoiceGuard.")
    tabs=st.tabs(["Login","Create account","Forgot password"])
    with tabs[0]:
        with st.form("login_form"):
            email=st.text_input("Gmail / Email").strip().lower()
            password=st.text_input("Password",type="password")
            submit=st.form_submit_button("Login",use_container_width=True)
        if submit:
            with db() as c: user=c.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
            if not user or not verify_password(password,user["password_hash"]):
                log_event(None,email,"LOGIN_FAILED"); st.error("Invalid email or password.")
            elif user["is_blocked"]:
                log_event(user["id"],email,"BLOCKED_LOGIN_ATTEMPT"); st.error("This account is blocked. Contact the administrator.")
            elif not user["verified"]:
                st.session_state.pending_email=email; st.session_state.pending_purpose="verify"
                st.warning("Email not verified. Request a verification code in Create account.")
            else:
                ok,msg=issue_otp(email,"login")
                if ok: st.session_state.login_otp_email=email; st.success("Login OTP sent to your email.")
                else: st.error(msg)
        if st.session_state.get("login_otp_email"):
            st.markdown("**Two-step verification**")
            st.caption(f"Login OTP sent to {st.session_state.login_otp_email}. Check Inbox, Spam and Promotions.")
            otp=st.text_input("6-digit email code",key="login_otp")
            login_col,resend_login_col=st.columns(2)
            with login_col:
                verify_login=st.button("Verify and login",use_container_width=True)
            with resend_login_col:
                resend_login=st.button("Resend login OTP",use_container_width=True)
            if verify_login:
                email=st.session_state.login_otp_email
                if verify_otp(email,"login",otp):
                    with db() as c: user=c.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
                    if not user or user["is_blocked"]:
                        st.error("This account is blocked or unavailable.")
                    else:
                        st.session_state.user_id=user["id"]
                        with db() as c: c.execute("UPDATE users SET last_login=? WHERE id=?",(now(),user["id"]))
                        log_event(user["id"],email,"LOGIN_SUCCESS_OTP")
                        del st.session_state.login_otp_email
                        st.rerun()
                else:
                    st.error("OTP is invalid, expired, or has reached the 5-attempt limit. Request a new OTP.")
            if resend_login:
                ok,msg=issue_otp(st.session_state.login_otp_email,"login")
                if ok: st.success("A new login OTP has been sent.")
                else: st.error(msg)
    with tabs[1]:
        with st.form("signup_form"):
            name=st.text_input("Full name"); email=st.text_input("Email address").strip().lower()
            phone=st.text_input("Contact number (optional)")
            pw=st.text_input("Password",type="password"); confirm=st.text_input("Confirm password",type="password")
            create=st.form_submit_button("Create account & send OTP",use_container_width=True)
        if create:
            if not name.strip() or "@" not in email: st.error("Enter a valid name and email.")
            elif len(pw)<10 or not any(x.isdigit() for x in pw) or not any(x.isalpha() for x in pw): st.error("Use at least 10 characters with letters and numbers.")
            elif pw!=confirm: st.error("Passwords do not match.")
            else:
                with db() as c:
                    existing=c.execute("SELECT id,verified,is_blocked FROM users WHERE email=?",(email,)).fetchone()

                if existing and existing["verified"]:
                    st.error("This email is already registered. Please log in or use Forgot password.")
                elif existing and existing["is_blocked"]:
                    st.error("This account is blocked. Please contact the administrator.")
                elif not all(os.getenv(k) for k in ["SMTP_HOST","SMTP_USERNAME","SMTP_PASSWORD"]):
                    st.error("Email OTP is not configured. Add SMTP settings in Streamlit Secrets.")
                else:
                    if not existing:
                        # Create an unverified account. If the user closes the page,
                        # the next signup attempt can resume verification for this email.
                        with db() as c:
                            c.execute(
                                "INSERT INTO users(name,email,phone,password_hash,created_at) VALUES(?,?,?,?,?)",
                                (name.strip(),email,phone.strip(),hash_password(pw),now())
                            )
                    else:
                        # Resume an abandoned, unverified registration.
                        with db() as c:
                            c.execute(
                                "UPDATE users SET name=?,phone=?,password_hash=? WHERE email=? AND verified=0",
                                (name.strip(),phone.strip(),hash_password(pw),email)
                            )

                    ok,msg=issue_otp(email,"signup")
                    if ok:
                        st.session_state.signup_otp_email=email
                        st.success("Verification code sent. Enter the OTP below to finish creating your account.")
                    else:
                        st.error(f"Could not send signup OTP: {msg}")
        if st.session_state.get("signup_otp_email"):
            st.caption(f"Verification is pending for {st.session_state.signup_otp_email}. Check Inbox, Spam and Promotions.")
            code=st.text_input("Enter 6-digit email verification OTP",key="signup_otp")
            verify_col,resend_col=st.columns(2)
            with verify_col:
                verify_signup=st.button("Verify email",use_container_width=True)
            with resend_col:
                resend_signup=st.button("Resend signup OTP",use_container_width=True)
            if verify_signup:
                email=st.session_state.signup_otp_email
                if verify_otp(email,"signup",code):
                    with db() as c: c.execute("UPDATE users SET verified=1 WHERE email=?",(email,))
                    log_event(None,email,"SIGNUP_VERIFIED")
                    del st.session_state.signup_otp_email
                    st.success("Email verified. Your account is ready. Please log in.")
                else:
                    st.error("OTP is invalid, expired, or has reached the 5-attempt limit. Request a new OTP.")
            if resend_signup:
                ok,msg=issue_otp(st.session_state.signup_otp_email,"signup")
                if ok: st.success("A new signup OTP has been sent.")
                else: st.error(msg)
    with tabs[2]:
        email=st.text_input("Account email",key="reset_email_input").strip().lower()
        if st.button("Send password reset OTP"):
            with db() as c:
                user=c.execute("SELECT id,verified,is_blocked FROM users WHERE email=?",(email,)).fetchone()
            if not email or "@" not in email:
                st.error("Please enter a valid account email.")
            elif not user:
                st.info("If the account exists, a reset email can be sent.")
            elif not user["verified"]:
                st.error("This account's email is not verified. Finish signup verification first.")
            elif user["is_blocked"]:
                st.error("This account is blocked. Please contact the administrator.")
            else:
                ok,msg=issue_otp(email,"reset")
                if ok:
                    st.session_state.reset_email_target=email
                    st.success("Password reset OTP sent. Check Inbox, Spam and Promotions.")
                else:
                    st.error(f"Could not send reset OTP: {msg}")
        if st.session_state.get("reset_email_target"):
            st.caption(f"Password reset requested for {st.session_state.reset_email_target}. OTP expires in 10 minutes.")
            code=st.text_input("Enter 6-digit reset OTP",key="reset_code")
            newpw=st.text_input("New password (10+ characters, letters and numbers)",type="password",key="new_pw")
            reset_col,resend_reset_col=st.columns(2)
            with reset_col:
                reset_clicked=st.button("Reset password",use_container_width=True)
            with resend_reset_col:
                resend_reset=st.button("Resend reset OTP",use_container_width=True)
            if reset_clicked:
                if len(newpw)<10 or not any(x.isdigit() for x in newpw) or not any(x.isalpha() for x in newpw):
                    st.error("Use at least 10 characters with both letters and numbers.")
                elif verify_otp(st.session_state.reset_email_target,"reset",code):
                    with db() as c:
                        c.execute("UPDATE users SET password_hash=? WHERE email=?",
                                  (hash_password(newpw),st.session_state.reset_email_target))
                    del st.session_state.reset_email
                    st.success("Password updated successfully. Please log in with your new password.")
                else:
                    st.error("Reset OTP is invalid, expired, or has reached the 5-attempt limit. Request a new OTP.")
            if resend_reset:
                ok,msg=issue_otp(st.session_state.reset_email_target,"reset")
                if ok: st.success("A new password reset OTP has been sent.")
                else: st.error(msg)

def user_app(user):
    with st.sidebar:
        st.markdown("## 🛡️ VoiceGuard AI"); st.caption(f"Signed in: {user['name']}")
        st.caption("Audio screening workspace · Secure session")
        page=st.radio("Navigation",["Dashboard","Analyze audio","History","Account settings"])
        if st.button("Secure logout",use_container_width=True):
            log_event(user["id"],user["email"],"LOGOUT"); st.session_state.clear(); st.rerun()
    if page=="Dashboard":
        st.markdown(f"## Welcome, {user['name']}")
        with db() as c:
            total=c.execute("SELECT COUNT(*) n FROM analyses WHERE user_id=?",(user["id"],)).fetchone()["n"]
            recent=c.execute("SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT 5",(user["id"],)).fetchall()
        a,b,d=st.columns(3)
        for col,label,value in [(a,"Analyses",total),(b,"Email verified","Yes" if user["verified"] else "No"),(d,"Two-step login","On" if user["two_fa"] else "Off")]:
            col.markdown(f'<div class="metric"><div>{label}</div><h2>{value}</h2></div>',unsafe_allow_html=True)
        st.subheader("Recent activity")
        st.dataframe(pd.DataFrame([dict(x) for x in recent]),use_container_width=True,hide_index=True)
    elif page=="Analyze audio":
        st.markdown('<div class="hero"><div class="kicker">Audio authenticity screening platform · Version 3.1</div><h1>🛡️ VoiceGuard AI</h1><p>AI-powered voice clone and deepfake audio screening assistant.</p><span class="kicker">Wav2Vec2 detector　·　Audio forensics　·　Privacy-aware workflow</span></div>',unsafe_allow_html=True)
        st.subheader("🎙️ Upload, record & listen")
        st.write("Add a recording, listen to it immediately, then run AI screening. Playback is available before and after analysis.")
        left,right=st.columns(2)
        with left:
            files=st.file_uploader("Choose audio files",type=["mp3","wav","m4a","flac","ogg","aac"],accept_multiple_files=True)
        with right:
            recorded=st.audio_input("Record a voice sample")
        candidates=[]
        if files:
            for f in files: candidates.append((f.name,f.getvalue()))
        if recorded is not None: candidates.append(("Recorded_voice.wav",recorded.getvalue()))
        if candidates:
            st.markdown("#### 🔊 Audio preview · Listen before analysis")
            for name,raw in candidates:
                with st.container(border=True):
                    st.caption(name); st.audio(raw)
        run=st.button("🚀 Analyze selected audio",disabled=not candidates,use_container_width=True)
        if run:
            for filename,raw in candidates:
                if len(raw)>MAX_UPLOAD_MB*1024*1024:
                    st.error(f"{filename}: exceeds {MAX_UPLOAD_MB} MB."); continue
                try:
                    with st.spinner(f"Analyzing {filename}… first run may download the AI model"):
                        audio,sr=decode_audio(raw,Path(filename).suffix or ".wav")
                        scores,pred,real,fake=classify(audio,sr); met=metrics(audio,sr)
                    rid=str(uuid.uuid4()); created=now(); digest=hashlib.sha256(raw).hexdigest()
                    with db() as c:
                        c.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?,?,?)",(rid,user["id"],filename,pred,real,fake,json.dumps(met),digest,created))
                    result={"id":rid,"filename":filename,"prediction":pred,"real_score":real,"fake_score":fake,"metrics":met,"created_at":created,"scores":scores}
                    st.success(f"Analysis complete: {filename}")
                    a,b,d=st.columns(3)
                    a.metric("REAL score · raw",f"{real*100:.2f}%" if real is not None else "N/A")
                    b.metric("FAKE score · raw",f"{fake*100:.2f}%" if fake is not None else "N/A")
                    d.metric("Top model label",pred)
                    st.markdown("**Model class scores (raw)**")
                    st.bar_chart(pd.DataFrame({"Raw score":scores}))
                    with st.expander("📊 Audio waveform and spectrogram",expanded=True):
                        fig,ax=plt.subplots(figsize=(12,2.5)); librosa.display.waveshow(audio,sr=sr,ax=ax); ax.set_title("Waveform"); ax.set_xlabel("Time (seconds)"); st.pyplot(fig); plt.close(fig)
                        spec=librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=96); spec_db=librosa.power_to_db(spec,ref=np.max)
                        fig2,ax2=plt.subplots(figsize=(12,3)); img=librosa.display.specshow(spec_db,sr=sr,x_axis="time",y_axis="mel",ax=ax2); ax2.set_title("Mel spectrogram"); fig2.colorbar(img,ax=ax2,format="%+2.0f dB"); st.pyplot(fig2); plt.close(fig2)
                    st.markdown("**Acoustic metrics**"); st.dataframe(pd.DataFrame([met]),use_container_width=True,hide_index=True)
                    c1,c2,c3=st.columns(3)
                    c1.download_button("⬇️ Download PDF",make_pdf(result),file_name=f"VoiceGuard_{rid[:8]}.pdf",mime="application/pdf",key=f"pdf{rid}")
                    c2.download_button("⬇️ Download JSON",json.dumps(result,indent=2),file_name=f"VoiceGuard_{rid[:8]}.json",mime="application/json",key=f"json{rid}")
                    c3.download_button("⬇️ Download CSV",pd.DataFrame([{"analysis_id":rid,"filename":filename,"timestamp":created,"prediction":pred,"real_score":real,"fake_score":fake,**met}]).to_csv(index=False),file_name=f"VoiceGuard_{rid[:8]}.csv",mime="text/csv",key=f"csv{rid}")
                except Exception as e: st.error(f"Could not analyze {filename}: {e}")
        with st.expander("🔐 Privacy, interpretation & model limitations"):
            st.write("Uploaded audio is processed for analysis and is not stored by this app. Analysis metadata, scores and acoustic metrics are saved to your account history. Only upload recordings you have permission to use. The detector can make mistakes.")
    elif page=="History":
        st.subheader("Analysis history")
        with db() as c: rows=c.execute("SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC",(user["id"],)).fetchall()
        df=pd.DataFrame([dict(r) for r in rows])
        if not df.empty:
            st.dataframe(df.drop(columns=["metrics_json","sha256"],errors="ignore"),use_container_width=True,hide_index=True)
            st.download_button("⬇️ Export history CSV",df.drop(columns=["metrics_json","sha256"],errors="ignore").to_csv(index=False),"voiceguard_history.csv","text/csv")
            selected=st.selectbox("Delete an analysis record",[""]+[r["id"] for r in rows])
            if selected and st.button("Delete selected record"):
                with db() as c: c.execute("DELETE FROM analyses WHERE id=? AND user_id=?",(selected,user["id"]))
                st.success("Analysis record deleted."); st.rerun()
        else: st.info("No analyses yet.")
    elif page=="Account settings":
        st.subheader("Profile and security")
        st.write(f"**Name:** {user['name']}  \n**Email:** {user['email']}  \n**Contact:** {user['phone'] or 'Not provided'}")
        enabled=st.toggle("Require email OTP at every login (2FA)",value=bool(user["two_fa"]))
        if st.button("Save security settings"):
            with db() as c: c.execute("UPDATE users SET two_fa=? WHERE id=?",(int(enabled),user["id"]))
            st.success("Settings saved."); st.rerun()
        with st.form("profile_form"):
            name=st.text_input("Update name",value=user["name"]); phone=st.text_input("Update contact",value=user["phone"] or "")
            if st.form_submit_button("Update profile"):
                with db() as c: c.execute("UPDATE users SET name=?,phone=? WHERE id=?",(name.strip(),phone.strip(),user["id"]))
                st.success("Profile updated."); st.rerun()

def admin_app(admin):
    with st.sidebar:
        st.markdown("## 🛡️ Admin Console")
        page=st.radio("Admin navigation",["Overview","Users","Login activity","Analysis records"])
        if st.button("Admin logout"): log_event(admin["id"],admin["email"],"ADMIN_LOGOUT"); st.session_state.clear(); st.rerun()
    if page=="Overview":
        with db() as c:
            users=c.execute("SELECT COUNT(*) n FROM users WHERE role='user'").fetchone()["n"]
            blocked=c.execute("SELECT COUNT(*) n FROM users WHERE is_blocked=1").fetchone()["n"]
            logs=c.execute("SELECT COUNT(*) n FROM login_events").fetchone()["n"]
            analyses=c.execute("SELECT COUNT(*) n FROM analyses").fetchone()["n"]
        st.title("Admin overview")
        cols=st.columns(4)
        for col,label,val in zip(cols,["Users","Blocked accounts","Login events","Analyses"],[users,blocked,logs,analyses]): col.metric(label,val)
        st.info("Administrators can view account/contact details and security audit metadata. Passwords are never displayed; only salted password hashes are stored.")
        st.subheader("Latest login activity")
        with db() as c: latest=c.execute("SELECT email,event,ip,user_agent,created_at FROM login_events ORDER BY id DESC LIMIT 10").fetchall()
        st.dataframe(pd.DataFrame([dict(r) for r in latest]),use_container_width=True,hide_index=True)
    elif page=="Users":
        st.title("User management")
        with db() as c: rows=c.execute("SELECT id,name,email,phone,role,verified,is_blocked,two_fa,created_at,last_login FROM users ORDER BY id DESC").fetchall()
        st.dataframe(pd.DataFrame([dict(r) for r in rows]),use_container_width=True,hide_index=True)
        uid=st.number_input("User ID",min_value=1,step=1)
        action=st.selectbox("Action",["Block user","Unblock user","Delete user"])
        if st.button("Apply user action"):
            with db() as c:
                if action=="Block user": c.execute("UPDATE users SET is_blocked=1 WHERE id=? AND role!='admin'",(uid,))
                elif action=="Unblock user": c.execute("UPDATE users SET is_blocked=0 WHERE id=? AND role!='admin'",(uid,))
                else: c.execute("DELETE FROM users WHERE id=? AND role!='admin'",(uid,))
            st.success("Action applied (administrator accounts are protected)."); st.rerun()
    elif page=="Login activity":
        st.title("Login and security activity")
        with db() as c: rows=c.execute("SELECT id,user_id,email,event,ip,user_agent,created_at FROM login_events ORDER BY id DESC LIMIT 1000").fetchall()
        audit=pd.DataFrame([dict(r) for r in rows])
        if not audit.empty:
            search=st.text_input("Search by email or event")
            if search: audit=audit[audit.astype(str).apply(lambda col: col.str.contains(search,case=False,na=False)).any(axis=1)]
            st.dataframe(audit,use_container_width=True,hide_index=True)
            st.download_button("⬇️ Download audit CSV",audit.to_csv(index=False),"voiceguard_login_activity.csv","text/csv")
        else: st.info("No login activity recorded yet.")
    elif page=="Analysis records":
        st.title("Platform analysis records")
        with db() as c: rows=c.execute("SELECT id,user_id,filename,prediction,real_score,fake_score,created_at FROM analyses ORDER BY created_at DESC LIMIT 1000").fetchall()
        st.dataframe(pd.DataFrame([dict(r) for r in rows]),use_container_width=True,hide_index=True)

init_db()
if len(os.getenv("OTP_SECRET", "")) < 32 or os.getenv("OTP_SECRET", "").startswith("change-this"):
    st.caption("Security setup: configure a unique OTP_SECRET (32+ characters) in Streamlit Secrets before production use.")
user=current_user()
if not user:
    auth_ui()
elif user["is_blocked"]:
    st.error("This account is blocked."); st.session_state.clear()
elif user["role"]=="admin":
    admin_app(user)
else:
    user_app(user)

st.markdown("---")
st.caption("VoiceGuard AI · Academic prototype. Analyze only recordings you have permission to use. Detection results can be wrong.")
