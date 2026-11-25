import os, io, json, time, hmac, hashlib, secrets, sqlite3, smtplib, mimetypes, uuid
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
import streamlit as st


def config_value(name, default=""):
    """Read deployment secrets first (Streamlit Cloud), then local .env."""
    try:
        value = st.secrets.get(name, None)
        if value is not None and str(value).strip() != "":
            return str(value)
    except Exception:
        pass
    value = os.getenv(name, default)
    return default if value is None else value
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

APP_NAME = "VoiceGuard AI"
DB_PATH = config_value("VOICEGUARD_DB", "voiceguard.db")
DETECTOR_ID = "garystafford/wav2vec2-deepfake-voice-detector"
TARGET_SR = 16000
MAX_UPLOAD_MB = int(config_value("MAX_UPLOAD_MB", "25"))
MAX_DURATION_SEC = int(config_value("MAX_DURATION_SEC", "300"))
st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at 15% 0%,#153252 0%,#0a1323 44%,#080f1b 100%);color:#eaf2ff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:#f4f8ff!important}
.block-container{max-width:1400px;padding-top:2rem}
section[data-testid="stSidebar"]{background:#0b1627}
.hero,.card{border:1px solid rgba(66,217,232,.25);border-radius:18px;padding:22px;background:linear-gradient(120deg,rgba(20,54,82,.92),rgba(16,29,49,.92));margin-bottom:18px}
.kicker{color:#42d9e8;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;font-size:.75rem}
.metric{background:#13243a;border:1px solid #29415c;border-radius:14px;padding:16px}
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
        admin_email = config_value("ADMIN_EMAIL", "").strip().lower()
        admin_password = config_value("ADMIN_PASSWORD", "")
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
    host=config_value("SMTP_HOST"); port=int(config_value("SMTP_PORT","587"))
    username=config_value("SMTP_USERNAME"); password=config_value("SMTP_PASSWORD")
    sender=config_value("SMTP_FROM", username or "")
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
    otp_secret = config_value("OTP_SECRET", "")
    if len(otp_secret) < 32:
        return False, "OTP_SECRET missing/too short. Add a unique 32+ character secret in Streamlit Secrets."
    with db() as c:
        recent=c.execute("SELECT created_at FROM otp_codes WHERE email=? AND purpose=? ORDER BY id DESC LIMIT 1",(email,purpose)).fetchone()
        if recent and time.time()-recent["created_at"] < 60:
            return False, "Please wait 60 seconds before requesting another code."
        code=f"{secrets.randbelow(1000000):06d}"
        digest=hashlib.sha256((code+otp_secret).encode()).hexdigest()
        c.execute("DELETE FROM otp_codes WHERE email=? AND purpose=?", (email,purpose))
        c.execute("INSERT INTO otp_codes(email,purpose,code_hash,expires_at,created_at) VALUES(?,?,?,?,?)",
                  (email,purpose,digest,time.time()+600,time.time()))
    ok,msg=send_email(email, f"Your VoiceGuard {purpose} code", f"Your verification code is {code}. It expires in 10 minutes. Do not share it.")
    return ok,msg

def verify_otp(email,purpose,code):
    otp_secret = config_value("OTP_SECRET", "")
    if len(otp_secret) < 32:
        return False
    with db() as c:
        row=c.execute("SELECT * FROM otp_codes WHERE email=? AND purpose=? ORDER BY id DESC LIMIT 1",(email,purpose)).fetchone()
        if not row or row["expires_at"] < time.time() or row["attempts"] >= 5:
            return False
        digest=hashlib.sha256((code+otp_secret).encode()).hexdigest()
        if hmac.compare_digest(digest,row["code_hash"]):
            c.execute("DELETE FROM otp_codes WHERE email=? AND purpose=?", (email,purpose))
            return True
        c.execute("UPDATE otp_codes SET attempts=attempts+1 WHERE id=?", (row["id"],))
        return False

def log_event(user_id,email,event):
    with db() as c:
        c.execute("INSERT INTO login_events(user_id,email,event,created_at) VALUES(?,?,?,?)",(user_id,email,event,now()))

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
            elif user["two_fa"]:
                ok,msg=issue_otp(email,"login")
                if ok: st.session_state.login_otp_email=email; st.success("Login OTP sent to your email.")
                else: st.error(msg)
            else:
                st.session_state.user_id=user["id"]
                with db() as c: c.execute("UPDATE users SET last_login=? WHERE id=?",(now(),user["id"]))
                log_event(user["id"],email,"LOGIN_SUCCESS"); st.rerun()
        if st.session_state.get("login_otp_email"):
            st.markdown("**Two-step verification**")
            otp=st.text_input("6-digit email code",key="login_otp")
            if st.button("Verify and login"):
                email=st.session_state.login_otp_email
                if verify_otp(email,"login",otp):
                    with db() as c: user=c.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
                    st.session_state.user_id=user["id"]; log_event(user["id"],email,"LOGIN_SUCCESS_2FA")
                    del st.session_state.login_otp_email; st.rerun()
                else: st.error("Invalid or expired code.")
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
                with db() as c: exists=c.execute("SELECT 1 FROM users WHERE email=?",(email,)).fetchone()
                if exists: st.error("An account with this email already exists.")
                else:
                    ok,msg=send_email(email,"Verify your VoiceGuard account","Your account verification code will be sent after account creation.")
                    # Insert only after mail service is configured; issue actual OTP below.
                    if not config_value("SMTP_HOST"): st.error("Email OTP is not configured. Set SMTP settings in .env first.")
                    else:
                        with db() as c:
                            c.execute("INSERT INTO users(name,email,phone,password_hash,created_at) VALUES(?,?,?,?,?)",
                                      (name.strip(),email,phone.strip(),hash_password(pw),now()))
                        ok,msg=issue_otp(email,"signup")
                        if ok: st.session_state.signup_otp_email=email; st.success("Account created. Check your email for OTP.")
                        else: st.error(msg)
        if st.session_state.get("signup_otp_email"):
            code=st.text_input("Enter email verification OTP",key="signup_otp")
            if st.button("Verify email"):
                email=st.session_state.signup_otp_email
                if verify_otp(email,"signup",code):
                    with db() as c: c.execute("UPDATE users SET verified=1 WHERE email=?",(email,))
                    log_event(None,email,"SIGNUP_VERIFIED"); del st.session_state.signup_otp_email
                    st.success("Email verified. You can now log in.")
                else: st.error("Invalid or expired OTP.")
    with tabs[2]:
        email=st.text_input("Account email",key="reset_email").strip().lower()
        if st.button("Send password reset OTP"):
            with db() as c: user=c.execute("SELECT id FROM users WHERE email=?",(email,)).fetchone()
            if user:
                ok,msg=issue_otp(email,"reset")
                if ok: st.session_state.reset_email=email; st.success("Reset code sent.")
                else: st.error(msg)
            else: st.info("If the account exists, a reset email can be sent.")
        if st.session_state.get("reset_email"):
            code=st.text_input("Reset OTP",key="reset_code")
            newpw=st.text_input("New password (10+ characters)",type="password",key="new_pw")
            if st.button("Reset password"):
                if len(newpw)<10: st.error("Password must be at least 10 characters.")
                elif verify_otp(st.session_state.reset_email,"reset",code):
                    with db() as c: c.execute("UPDATE users SET password_hash=? WHERE email=?",(hash_password(newpw),st.session_state.reset_email))
                    del st.session_state.reset_email; st.success("Password updated. Please log in.")
                else: st.error("Invalid or expired OTP.")

def user_app(user):
    with st.sidebar:
        st.markdown("## 🛡️ VoiceGuard AI"); st.caption(f"Signed in: {user['name']}")
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
        st.subheader("🎙️ Upload and analyze audio")
        files=st.file_uploader("MP3, WAV, M4A, FLAC, OGG or AAC",type=["mp3","wav","m4a","flac","ogg","aac"],accept_multiple_files=True)
        if files:
            for f in files: st.audio(f.getvalue(),format=mimetypes.guess_type(f.name)[0] or "audio/wav")
        if st.button("Run AI voice screening",disabled=not files,use_container_width=True):
            for f in files:
                raw=f.getvalue()
                if len(raw)>MAX_UPLOAD_MB*1024*1024: st.error(f"{f.name}: exceeds {MAX_UPLOAD_MB} MB."); continue
                try:
                    audio,sr=decode_audio(raw,Path(f.name).suffix or ".audio")
                    scores,pred,real,fake=classify(audio,sr); met=metrics(audio,sr)
                    rid=str(uuid.uuid4()); created=now(); digest=hashlib.sha256(raw).hexdigest()
                    with db() as c:
                        c.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?,?,?)",(rid,user["id"],f.name,pred,real,fake,json.dumps(met),digest,created))
                    result={"id":rid,"filename":f.name,"prediction":pred,"real_score":real,"fake_score":fake,"metrics":met,"created_at":created}
                    st.success(f"Analysis complete: {f.name}")
                    x,y,z=st.columns(3); x.metric("Model label",pred)
                    if real is not None: y.metric("Real raw score",f"{real*100:.2f}%"); z.metric("Fake raw score",f"{fake*100:.2f}%")
                    st.bar_chart(pd.DataFrame({"Raw score":scores}))
                    st.json(met)
                    st.download_button("Download PDF report",make_pdf(result),file_name=f"VoiceGuard_{rid[:8]}.pdf",mime="application/pdf",key=f"pdf{rid}")
                    st.caption("Model scores are raw outputs, not calibrated probabilities. This is a screening signal, not proof.")
                except Exception as e: st.error(f"Could not analyze {f.name}: {e}")
    elif page=="History":
        st.subheader("Analysis history")
        with db() as c: rows=c.execute("SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC",(user["id"],)).fetchall()
        df=pd.DataFrame([dict(r) for r in rows])
        if not df.empty:
            st.dataframe(df.drop(columns=["metrics_json","sha256"],errors="ignore"),use_container_width=True,hide_index=True)
            st.download_button("Export history CSV",df.to_csv(index=False),"voiceguard_history.csv","text/csv")
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
        st.info("Administrators can view account/contact and audit metadata. Passwords are never displayed; only salted password hashes are stored.")
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
        st.dataframe(pd.DataFrame([dict(r) for r in rows]),use_container_width=True,hide_index=True)
        if rows: st.download_button("Download audit CSV",pd.DataFrame([dict(r) for r in rows]).to_csv(index=False),"voiceguard_login_activity.csv","text/csv")
    elif page=="Analysis records":
        st.title("Platform analysis records")
        with db() as c: rows=c.execute("SELECT id,user_id,filename,prediction,real_score,fake_score,created_at FROM analyses ORDER BY created_at DESC LIMIT 1000").fetchall()
        st.dataframe(pd.DataFrame([dict(r) for r in rows]),use_container_width=True,hide_index=True)

init_db()
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
