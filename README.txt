VOICEGUARD AI — USER WEBSITE + ADMIN PORTAL

FEATURES
- Dark navy/cyan VoiceGuard interface inspired by the supplied screenshots.
- Account creation with Gmail OTP verification.
- Email OTP is required at every login.
- Forgot-password OTP flow.
- User dashboard, profile, account settings and analysis history.
- Audio upload and microphone recording, playback, AI voice screening.
- Model score chart, waveform, Mel spectrogram and acoustic metrics.
- PDF, JSON and CSV report downloads; history CSV export.
- Admin portal with user directory, block/unblock/delete controls, login/security audit, analysis records and CSV export.
- Passwords stored as salted PBKDF2 hashes; OTP codes are hashed and expire after 10 minutes.

WINDOWS LOCAL SETUP
1. Install Python 3.10–3.12 and enable “Add Python to PATH”.
2. Open this folder in File Explorer, click the address bar, type cmd and press Enter.
3. Run:
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
4. Copy .env.example to .env and fill in Gmail SMTP and admin settings.
5. Run: streamlit run app.py
6. Open the localhost address shown by Streamlit (usually http://localhost:8501).

GMAIL OTP SETUP
- Enable 2-Step Verification on the Gmail account used to send OTPs.
- Create a Google App Password in Google Account security settings.
- Put the App Password in SMTP_PASSWORD (not your normal Gmail password).
- SMTP_USERNAME and SMTP_FROM should be the sending Gmail address.
- Set OTP_SECRET to a long, unique random value (32+ characters).

ADMIN LOGIN
- Use ADMIN_EMAIL and ADMIN_PASSWORD from .env.
- Admin is created on first startup. If it already exists, changing .env does not reset its password.
- Open the app and sign in with the admin email/password. Admin login also requires an email OTP sent to that admin email, so SMTP must work.

STREAMLIT CLOUD DEPLOYMENT
- Push app.py and requirements.txt to a private/public GitHub repository as appropriate.
- Deploy the repository in Streamlit Community Cloud.
- In App settings → Secrets, add the same settings in TOML format, e.g.:
  SMTP_HOST = "smtp.gmail.com"
  SMTP_PORT = 587
  SMTP_USERNAME = "yourgmail@gmail.com"
  SMTP_PASSWORD = "your-app-password"
  SMTP_FROM = "yourgmail@gmail.com"
  OTP_SECRET = "your-long-random-secret"
  ADMIN_EMAIL = "admin@example.com"
  ADMIN_PASSWORD = "your-strong-admin-password"
- Do not commit .env, secrets, or a database containing user information.

IMPORTANT LIMITATIONS / PRODUCTION
- Streamlit Community Cloud local files/database may not be durable across app restarts or redeployments. Use managed PostgreSQL for persistent production user data.
- Login IP and browser user-agent are recorded only when hosting proxy headers provide them; they may show Unavailable in local runs.
- Uploaded audio is processed in memory and is not saved as a recording; analysis metadata is stored.
- First analysis downloads the pretrained model and needs internet access; it can take several minutes.
- Model scores are raw outputs, not calibrated probabilities. They are not proof of authenticity, identity or fraud. False positives/negatives are possible.
- Before public production, add HTTPS, CSRF/session hardening, rate limiting, retention policy, backups, privacy notice/consent, and managed persistent storage. Never publish SMTP credentials or admin secrets.
