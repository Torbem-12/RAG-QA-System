# ============================================================
# app.py  —  FMOAU RAG Student Assistant
# Father Moses Orshio Adasu University, Makurdi
# Author : Bem Thomas | Supervisor: Dr. Stephen Akuma
# ============================================================

import traceback
import sys
import os, json, hashlib, tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
try:
    from rag_chain import ask
except Exception as _import_err:
    import streamlit as st
    st.error(f"STARTUP ERROR: {_import_err}")
    st.code(traceback.format_exc())
    st.stop()

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
WHISPER_MODEL = "whisper-large-v3-turbo"

DATA_DIR    = Path("data")
USERS_FILE  = DATA_DIR / "users.json"
HISTORY_DIR = DATA_DIR / "histories"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FMOAU Student Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS  — single source of truth, no conflicting rules
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family:'Inter',sans-serif; box-sizing:border-box; }

/* ── 1. Hide Streamlit chrome — keep hamburger ── */
#MainMenu, footer { visibility:hidden; }
[data-testid="stDecoration"]     { display:none !important; }
[data-testid="stStatusWidget"]   { display:none !important; }
[data-testid="stToolbarActions"] { display:none !important; }
[data-testid="stAppToolbar"]     { display:none !important; }
.stDeployButton                  { display:none !important; }
/* Hide Deploy button specifically */
.stDeployButton { display:none !important; }
button[kind="header"][data-testid="baseButton-header"] { display:none !important; }

/* ── 2. Header bar — always dark, hamburger always white ── */
header[data-testid="stHeader"] {
    background: #0f2744 !important;
    opacity: 1 !important; visibility: visible !important;
}
header[data-testid="stHeader"] button {
    opacity: 1 !important; visibility: visible !important;
    color: #fff !important;
}
header[data-testid="stHeader"] button svg {
    opacity: 1 !important; fill: #fff !important; stroke: #fff !important;
}
[data-testid="collapsedControl"] {
    display: flex !important; opacity: 1 !important;
    visibility: visible !important;
    background: rgba(255,255,255,.2) !important;
    border-radius: 7px !important;
    padding: 4px 6px !important; margin: 4px 6px !important;
}
[data-testid="collapsedControl"] svg {
    fill: #fff !important; stroke: #fff !important; opacity: 1 !important;
}
/* Prevent Streamlit from fading the button when not hovered */
header[data-testid="stHeader"] button { opacity: 1 !important; }

/* ── 3. Page background & spacing ── */
[data-testid="stAppViewContainer"] { background:#f0f4f9 !important; }
.block-container {
    padding-top: 4rem !important;
    padding-bottom: 6rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
}

/* ── 4. Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f2744,#1a3c5e) !important;
}
[data-testid="stSidebar"] * { color:#e8f0fe !important; }
[data-testid="stSidebarNav"] { display:none !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,.1) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    border-radius: 8px !important; width: 100% !important;
    font-size: .81rem !important; padding: .42rem .8rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,.22) !important;
}

/* ── 5. Top banner ── */
.top-header {
    background: linear-gradient(135deg,#0f2744,#1a6fc4);
    padding: .85rem 1.4rem; margin-bottom: .6rem;
    border-radius: 12px; box-shadow: 0 4px 14px rgba(15,39,68,.22);
}
.top-header h1 { color:#fff; font-size:1.05rem; font-weight:700; margin:0; }
.top-header p  { color:#9ec5f5; font-size:.72rem; margin:.08rem 0 0; }

/* ── 6. Welcome card ── */
.welcome-card {
    max-width:560px; margin:3rem auto; text-align:center;
    background:#fff; border-radius:16px; padding:2rem;
    box-shadow:0 4px 20px rgba(0,0,0,.08); border-top:3px solid #1a6fc4;
}
.welcome-card h2 { color:#0f2744; font-size:1.15rem; margin-bottom:.5rem; }
.welcome-card p  { color:#5a6a7e; font-size:.86rem; line-height:1.65; }

/* ── 7. Chat bubbles ── */
.user-row {
    display:flex !important; justify-content:flex-end !important;
    margin:.45rem 0 .2rem !important; padding-right:.5rem !important;
}
.user-bubble {
    background:linear-gradient(135deg,#1a6fc4,#0f2744) !important;
    color:#fff !important; padding:.65rem .9rem !important;
    border-radius:16px 16px 3px 16px !important;
    max-width:65% !important; font-size:.87rem !important;
    line-height:1.55 !important;
    box-shadow:0 2px 8px rgba(26,111,196,.25) !important;
    word-wrap:break-word !important;
}
.bot-row {
    display:flex !important; align-items:flex-start !important;
    gap:.4rem !important; margin:.2rem 0 .1rem !important;
    padding-left:.3rem !important;
}
.bot-av {
    width:28px !important; height:28px !important; flex-shrink:0 !important;
    background:linear-gradient(135deg,#1a6fc4,#0f2744) !important;
    border-radius:50% !important;
    display:flex !important; align-items:center !important;
    justify-content:center !important; font-size:.82rem !important;
}
.bot-bubble {
    background:#fff !important; color:#1a1a2e !important;
    padding:.65rem .9rem !important;
    border-radius:3px 16px 16px 16px !important;
    max-width:74% !important; font-size:.87rem !important;
    line-height:1.6 !important;
    box-shadow:0 2px 8px rgba(0,0,0,.07) !important;
    border-top:3px solid #1a6fc4 !important;
    word-wrap:break-word !important;
}
.source-block { margin:.1rem 0 .55rem 2.3rem !important; }
.src-line {
    background:#f0f7ff; border-left:3px solid #1a6fc4;
    border-radius:0 6px 6px 0; padding:.18rem .6rem;
    margin:.15rem 0; font-size:.72rem; color:#1a3c5e;
}
.badge { display:inline-block; padding:1px 6px; border-radius:8px;
         font-size:.65rem; font-weight:600; margin-right:3px; }
.badge-local    { background:#d4edda; color:#155724; }
.badge-uni      { background:#cce5ff; color:#004085; }
.badge-academic { background:#fff3cd; color:#856404; }

/* ── 8. Sidebar history ── */
.hist-row {
    display: flex; align-items: center;
    background: rgba(255,255,255,.07);
    border-left: 3px solid rgba(255,255,255,.18);
    border-radius: 8px; padding: .45rem .6rem;
    margin: .1rem 0; gap: .3rem; cursor:pointer;
    transition: background .15s;
}
.hist-row:hover { background: rgba(255,255,255,.14) !important; }
.hist-row.active { border-left-color: #1a6fc4 !important; }
.hist-text { flex:1; min-width:0; overflow:hidden; }
.hist-title {
    font-size:.77rem; font-weight:500; color:#d4e8fc;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.hist-time { font-size:.61rem; color:#7aaed4; margin-top:.03rem; }

/* ── Audio input widget — slim and clean ── */
[data-testid="stAudioInput"] > div {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: .2rem .5rem !important;
    box-shadow: none !important;
    min-height: 0 !important;
}
[data-testid="stAudioInput"] label { display:none !important; }

/* ── 9. Input bar — Claude style ── */
/* Fixed bottom wrapper */
.input-bar-outer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 999;
    background: #f9fafb;
    border-top: 1px solid #e5e7eb;
    padding: .6rem 1rem .75rem;
}
/* Centered pill container */
.input-bar-pill {
    max-width: 720px;
    margin: 0 auto;
    background: #ffffff;
    border: 1.5px solid #d1d5db;
    border-radius: 28px;
    box-shadow: 0 1px 6px rgba(0,0,0,.08);
    display: flex;
    align-items: center;
    padding: .3rem .35rem .3rem .6rem;
    gap: .2rem;
}
/* Text input inside pill — borderless, transparent */
.input-bar-pill .stTextInput { flex:1 !important; min-width:0 !important; }
.input-bar-pill .stTextInput > div { width:100% !important; }
.input-bar-pill .stTextInput input {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: .95rem !important;
    color: #111827 !important;
    padding: .4rem .5rem !important;
    height: 44px !important;
    width: 100% !important;
}
.input-bar-pill .stTextInput input::placeholder {
    color: #9ca3af !important;
    font-size: .93rem !important;
}
/* Mic button — col 2 */
.input-bar-pill div[data-testid="column"]:nth-child(2) .stButton > button {
    background: none !important;
    border: none !important;
    border-radius: 50% !important;
    width: 36px !important; height: 36px !important;
    min-width: 36px !important; padding: 0 !important;
    font-size: 1rem !important;
    color: #6b7280 !important;
    box-shadow: none !important;
}
.input-bar-pill div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: #f3f4f6 !important;
}
/* Send button — col 3, dark circle */
.input-bar-pill div[data-testid="column"]:nth-child(3) .stButton > button {
    background: #1e293b !important;
    border: none !important;
    border-radius: 50% !important;
    width: 36px !important; height: 36px !important;
    min-width: 36px !important; padding: 0 !important;
    font-size: .95rem !important;
    color: #ffffff !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.2) !important;
}
.input-bar-pill div[data-testid="column"]:nth-child(3) .stButton > button:hover {
    background: #334155 !important;
}

/* ✓ Send and ✕ Cancel — slim pill buttons */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #16a34a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 20px !important;
    height: 34px !important;
    font-size: .85rem !important;
    font-weight: 700 !important;
    padding: 0 .5rem !important;
    box-shadow: none !important;
    width: 100% !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: #15803d !important;
}
div[data-testid="column"]:nth-child(3) .stButton > button {
    background: #f3f4f6 !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 20px !important;
    height: 34px !important;
    font-size: .85rem !important;
    font-weight: 700 !important;
    padding: 0 .5rem !important;
    box-shadow: none !important;
    width: 100% !important;
}
div[data-testid="column"]:nth-child(3) .stButton > button:hover {
    background: #e5e7eb !important;
}

/* ── 10. Login card ── */
div[data-testid="column"]:nth-child(2) > div[data-testid="stVerticalBlock"] {
    background:#fff !important; border-radius:20px !important;
    padding:2rem 2.2rem !important; margin-top:1.5rem !important;
    box-shadow:0 4px 8px rgba(15,39,68,.08),
               0 12px 28px rgba(15,39,68,.14) !important;
    border-top:3px solid #1a6fc4 !important;
}
div[data-testid="column"]:nth-child(2) .stFormSubmitButton > button,
div[data-testid="column"]:nth-child(2) .stButton > button {
    background:linear-gradient(135deg,#1a6fc4,#0f2744) !important;
    color:#fff !important; border-radius:10px !important;
    font-weight:600 !important; border:none !important; width:100% !important;
}

@media(max-width:768px){
    .user-bubble { max-width:85% !important; }
    .bot-bubble  { max-width:88% !important; }
}

/* Hide only the bridge audio input and text bridge */
input[placeholder=""] { display:none !important; }

/* Hide mic toggle button completely but keep clickable */
button[title="Toggle voice recorder"] {
    position:fixed !important;
    bottom:-200px !important; left:-200px !important;
    width:1px !important; height:1px !important;
    opacity:0 !important; pointer-events:auto !important;
}

/* Voice recorder bar buttons */
.voice-done-btn .stButton > button {
    background:#16a34a !important; color:#fff !important;
    border:none !important; border-radius:50% !important;
    width:34px !important; height:34px !important;
    font-size:.85rem !important; font-weight:700 !important;
}
.voice-cancel-btn .stButton > button {
    background:none !important; color:#6b7280 !important;
    border:1.5px solid #d1d5db !important;
    border-radius:50% !important;
    width:34px !important; height:34px !important;
}

/* Remove bottom padding since pill is in iframe */
.block-container { padding-bottom: 4rem !important; }

/* Input bar outer wrapper */
.input-bar-outer {
    position: fixed; bottom:0; left:0; right:0;
    z-index: 9999;
    background: #f9fafb;
    border-top: 1px solid #e5e7eb;
    padding: .5rem 1rem .65rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SECURITY QUESTIONS
# ============================================================
SECURITY_QUESTIONS = [
    "What is the name of your primary school?",
    "What is your mother's maiden name?",
    "What was the name of your first pet?",
    "What is the name of the street you grew up on?",
    "What was your childhood nickname?",
    "What is the name of your favourite teacher in secondary school?",
    "In what city were you born?",
]

# ============================================================
# HELPERS
# ============================================================
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def load_users():
    if not USERS_FILE.exists(): return {}
    try:
        t = USERS_FILE.read_text(encoding="utf-8").strip()
        return json.loads(t) if t else {}
    except Exception:
        return {}

def save_users(u): USERS_FILE.write_text(json.dumps(u, indent=2, ensure_ascii=False), encoding='utf-8')

def register_user(username, password, sec_q, sec_a):
    u = load_users()
    if username in u: return False, "Username already exists."
    u[username] = {
        "password": hash_pw(password),
        "security_question": sec_q,
        "security_answer": hash_pw(sec_a.strip().lower()),
        "created": str(datetime.now()),
    }
    save_users(u); return True, "Account created! Please sign in."

def verify_user(username, password):
    u = load_users()
    return username in u and u[username]["password"] == hash_pw(password)

def get_security_question(username):
    return load_users().get(username, {}).get("security_question")

def verify_security_answer(username, answer):
    stored = load_users().get(username, {}).get("security_answer", "")
    return stored == hash_pw(answer.strip().lower())

def reset_password(username, new_pw):
    u = load_users(); u[username]["password"] = hash_pw(new_pw); save_users(u)

def sessions_path(username): return HISTORY_DIR / f"{username}_sessions.json"

def load_sessions(username):
    p = sessions_path(username)
    if not p.exists(): return []
    try:
        t = p.read_text(encoding="utf-8").strip()
        return json.loads(t) if t else []
    except Exception:
        try:
            # Try reading with different encoding as fallback
            t = p.read_text(encoding="cp1252").strip()
            return json.loads(t) if t else []
        except Exception:
            return []

def save_sessions(username, sessions):
    """Save sessions safely with UTF-8 encoding to handle all characters."""
    try:
        data = json.dumps(sessions, indent=2, ensure_ascii=False)
        sessions_path(username).write_text(data, encoding="utf-8")
    except Exception as e:
        # Fallback: save with ASCII-safe encoding
        data = json.dumps(sessions, indent=2, ensure_ascii=True)
        sessions_path(username).write_text(data, encoding="utf-8")

def save_current_session(username, session_id, messages):
    if not messages: return
    sessions = load_sessions(username)
    for s in sessions:
        if s["id"] == session_id:
            s["messages"] = messages
            save_sessions(username, sessions); return
    first_q = next((m["content"] for m in messages if m["role"]=="user"), "Chat")
    # Strip non-printable characters from title to prevent encoding issues
    safe_title = "".join(c for c in first_q if c.isprintable())
    title = safe_title[:48] + ("…" if len(safe_title) > 48 else "")
    sessions.insert(0, {
        "id": session_id, "title": title,
        "time": datetime.now().strftime("%d %b, %I:%M %p"),
        "messages": messages,
    })
    save_sessions(username, sessions)

# ============================================================
# SESSION STATE
# ============================================================
_defaults = {
    "logged_in": False, "username": "", "messages": [], "session_id": "",
    "show_register": False, "show_reset": False,
    "reset_step": 1, "reset_username": "",
    "rename_session": None,
    "last_audio_hash": None,
}
for k, v in _defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ============================================================
# GROQ HELPERS
# ============================================================
def groq_client(): return Groq(api_key=GROQ_API_KEY)

def transcribe_audio(audio_bytes):
    # Write to temp file with proper extension for Groq
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as af:
            r = groq_client().audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=("audio.wav", af, "audio/wav"),
                language="en")
        return r.text.strip()
    finally:
        try: os.unlink(tmp_path)
        except: pass

def badge_html(rtype):
    m = {
        "local":                  ("Local PDF", "badge-local"),
        "semantic":               ("Local PDF", "badge-local"),
        "bm25":                   ("Local PDF", "badge-local"),
        "uni_web":                ("Uni Web",   "badge-uni"),
        "web — university/general":("Uni Web",  "badge-uni"),
        "academic":               ("Academic",  "badge-academic"),
        "web — academic":         ("Academic",  "badge-academic"),
    }
    label, cls = m.get(rtype, ("Web", "badge-academic"))
    return f'<span class="badge {cls}">{label}</span>'

# ============================================================
# PROCESS & RESPOND
# ============================================================
def process_and_respond(question):
    with st.spinner("🔍 Searching…"):
        try:
            result  = ask(question, use_web=True)
            answer  = result.get("answer", "Sorry, I could not find an answer.")
            sources = result.get("sources", [])
        except Exception as e:
            answer = f"An error occurred: {e}"; sources = []
    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources, "time": datetime.now().strftime("%I:%M %p"),
    })
    if st.session_state.logged_in:
        save_current_session(st.session_state.username,
                             st.session_state.session_id,
                             st.session_state.messages)

# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background:#dde4ee !important; }
    .block-container { padding-top:2rem !important; }
    </style>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:.5rem 0 .8rem;">
            <span style="font-size:2.6rem;">🎓</span>
            <h2 style="color:#0f2744;font-size:1.3rem;font-weight:700;
                       margin:.2rem 0 .1rem;">FMOAU Student Assistant</h2>
            <p style="color:#5a6a7e;font-size:.78rem;margin:0 0 .3rem;">
                Father Moses Orshio Adasu University, Makurdi</p>
            <hr style="border:none;border-top:1px solid #dde5ef;
                       margin:.5rem 0 .7rem;">
        </div>""", unsafe_allow_html=True)

        if not st.session_state.show_register and not st.session_state.show_reset:
            st.markdown("<p style='font-weight:600;color:#0f2744;font-size:.92rem;"
                        "margin-bottom:.3rem;'>Sign In</p>", unsafe_allow_html=True)
            with st.form("login_form"):
                uname = st.text_input("u", placeholder="Username",
                                      label_visibility="collapsed")
                pw    = st.text_input("p", type="password",
                                      placeholder="Password",
                                      label_visibility="collapsed")
                sub   = st.form_submit_button("Sign In →",
                                              use_container_width=True)
            if sub:
                if verify_user(uname, pw):
                    st.session_state.logged_in  = True
                    st.session_state.username   = uname
                    st.session_state.messages   = []
                    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
                    st.rerun()
                else: st.error("Incorrect username or password.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Forgot password?", use_container_width=True,
                             key="go_reset"):
                    st.session_state.show_reset = True
                    st.session_state.reset_step = 1; st.rerun()
            with c2:
                if st.button("Create Account", use_container_width=True,
                             key="go_reg"):
                    st.session_state.show_register = True; st.rerun()

        elif st.session_state.show_register:
            st.markdown("<p style='font-weight:600;color:#0f2744;font-size:.92rem;"
                        "margin-bottom:.3rem;'>Create Account</p>",
                        unsafe_allow_html=True)
            with st.form("reg_form"):
                nu  = st.text_input("nu", placeholder="Choose a username",
                                    label_visibility="collapsed")
                np1 = st.text_input("np1", type="password",
                                    placeholder="Choose a password (min 6 chars)",
                                    label_visibility="collapsed")
                np2 = st.text_input("np2", type="password",
                                    placeholder="Confirm password",
                                    label_visibility="collapsed")
                st.markdown("<p style='font-size:.76rem;color:#5a6a7e;"
                            "margin:.3rem 0 .1rem;'>Security Question</p>",
                            unsafe_allow_html=True)
                sq  = st.selectbox("sq", SECURITY_QUESTIONS,
                                   label_visibility="collapsed")
                sa  = st.text_input("sa",
                                    placeholder="Your answer (not case-sensitive)",
                                    label_visibility="collapsed")
                sub2 = st.form_submit_button("Register →",
                                             use_container_width=True)
            if sub2:
                if not nu or not np1:   st.error("Username and password required.")
                elif np1 != np2:        st.error("Passwords do not match.")
                elif len(np1) < 6:      st.error("Password must be at least 6 chars.")
                elif not sa.strip():    st.error("Please answer the security question.")
                else:
                    ok, msg = register_user(nu, np1, sq, sa)
                    if ok:
                        st.success(msg)
                        st.session_state.show_register = False; st.rerun()
                    else: st.error(msg)
            if st.button("← Back to Sign In", use_container_width=True,
                         key="back1"):
                st.session_state.show_register = False; st.rerun()

        else:
            step = st.session_state.reset_step
            st.markdown(f"<p style='font-weight:600;color:#0f2744;font-size:.92rem;"
                        f"margin-bottom:.2rem;'>🔑 Reset Password — Step {step}/3</p>",
                        unsafe_allow_html=True)
            if step == 1:
                with st.form("rs1"):
                    ru = st.text_input("ru", placeholder="Enter your username",
                                       label_visibility="collapsed")
                    s1 = st.form_submit_button("Continue →",
                                               use_container_width=True)
                if s1:
                    if get_security_question(ru) is None:
                        st.error("Username not found.")
                    else:
                        st.session_state.reset_username = ru
                        st.session_state.reset_step = 2; st.rerun()
            elif step == 2:
                q = get_security_question(st.session_state.reset_username)
                st.markdown(f"<p style='font-size:.83rem;color:#0f2744;"
                            f"font-weight:500;margin-bottom:.4rem;'>❓ {q}</p>",
                            unsafe_allow_html=True)
                with st.form("rs2"):
                    ra = st.text_input("ra", placeholder="Your answer",
                                       label_visibility="collapsed")
                    s2 = st.form_submit_button("Verify →",
                                               use_container_width=True)
                if s2:
                    if verify_security_answer(st.session_state.reset_username, ra):
                        st.session_state.reset_step = 3; st.rerun()
                    else: st.error("Incorrect answer.")
            elif step == 3:
                with st.form("rs3"):
                    rp1 = st.text_input("rp1", type="password",
                                        placeholder="New password",
                                        label_visibility="collapsed")
                    rp2 = st.text_input("rp2", type="password",
                                        placeholder="Confirm new password",
                                        label_visibility="collapsed")
                    s3  = st.form_submit_button("Save Password →",
                                                use_container_width=True)
                if s3:
                    if len(rp1) < 6:  st.error("Min 6 characters.")
                    elif rp1 != rp2:  st.error("Passwords do not match.")
                    else:
                        reset_password(st.session_state.reset_username, rp1)
                        st.success("✅ Password reset! Please sign in.")
                        st.session_state.show_reset = False
                        st.session_state.reset_step = 1; st.rerun()
            if st.button("← Back to Sign In", use_container_width=True,
                         key="back2"):
                st.session_state.show_reset = False
                st.session_state.reset_step = 1; st.rerun()

# ============================================================
# MAIN APP
# ============================================================
def show_main_app():

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:.3rem 0 .7rem;">
            <div style="font-size:1.7rem;">🎓</div>
            <div style="font-size:.9rem;font-weight:700;color:#fff;">
                FMOAU Assistant</div>
            <div style="font-size:.67rem;color:#7aaed4;">
                Signed in as
                <b style="color:#9ec5f5;">{st.session_state.username}</b>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button("＋  New Chat", use_container_width=True, key="new_chat"):
            if st.session_state.messages:
                save_current_session(st.session_state.username,
                                     st.session_state.session_id,
                                     st.session_state.messages)
            st.session_state.messages   = []
            st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
            st.rerun()

        st.markdown('<hr style="border-color:rgba(255,255,255,.12);margin:.4rem 0;">',
                    unsafe_allow_html=True)

        # ── CHAT HISTORY ─────────────────────────────────────
        all_sessions = load_sessions(st.session_state.username)
        if all_sessions:
            st.markdown('<div style="font-size:.67rem;color:#7aaed4;'
                        'text-transform:uppercase;letter-spacing:.06em;'
                        'margin-bottom:.35rem;">Chat History</div>',
                        unsafe_allow_html=True)

            for sess in all_sessions[:25]:
                sid        = sess["id"]
                is_current = sid == st.session_state.session_id
                border_col = "#1a6fc4" if is_current else "rgba(255,255,255,.18)"

                # Rename mode
                if st.session_state.rename_session == sid:
                    new_t = st.text_input("rt", value=sess["title"],
                                          key=f"ri_{sid}",
                                          label_visibility="collapsed")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        if st.button("Save", key=f"rs_{sid}",
                                     use_container_width=True):
                            s2 = load_sessions(st.session_state.username)
                            for s in s2:
                                if s["id"] == sid: s["title"] = new_t; break
                            save_sessions(st.session_state.username, s2)
                            st.session_state.rename_session = None; st.rerun()
                    with rc2:
                        if st.button("Cancel", key=f"rc_{sid}",
                                     use_container_width=True):
                            st.session_state.rename_session = None; st.rerun()
                    continue

                # Normal display — title + date, small action buttons
                st.markdown(f"""
                <div class="hist-row {'active' if is_current else ''}"
                     style="border-left-color:{border_col};">
                    <div class="hist-text">
                        <div class="hist-title">{sess["title"]}</div>
                        <div class="hist-time">{sess["time"]}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Action buttons on ONE row: [Open (wide)] [✏️] [🗑]
                b1, b2, b3 = st.columns([5, 1, 1])
                with b1:
                    if st.button(f"Open", key=f"lo_{sid}",
                                 use_container_width=True,
                                 help=sess["title"]):
                        st.session_state.messages   = sess["messages"]
                        st.session_state.session_id = sid; st.rerun()
                with b2:
                    if st.button("✏️", key=f"rn_{sid}",
                                 help="Rename", use_container_width=True):
                        st.session_state.rename_session = sid; st.rerun()
                with b3:
                    if st.button("🗑", key=f"dl_{sid}",
                                 help="Delete", use_container_width=True):
                        slist = load_sessions(st.session_state.username)
                        slist = [s for s in slist if s["id"] != sid]
                        save_sessions(st.session_state.username, slist)
                        if st.session_state.session_id == sid:
                            st.session_state.messages   = []
                            st.session_state.session_id = (
                                datetime.now().strftime("%Y%m%d%H%M%S"))
                        st.rerun()
        else:
            st.markdown('<div style="font-size:.75rem;color:#7aaed4;'
                        'text-align:center;padding:.7rem 0;">No chats yet</div>',
                        unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(255,255,255,.12);margin:.4rem 0;">',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.7rem;color:#7aaed4;line-height:1.9;">
        <b style="color:#9ec5f5;">I can help with:</b><br>
        🏫 University policies &nbsp; 📅 Academic calendar<br>
        💰 Fees &amp; payments &nbsp;&nbsp; 📝 Exam information<br>
        📖 Course content &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 🔬 Assignment help<br>
        🎤 Voice questions
        </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(255,255,255,.12);margin:.4rem 0;">',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.65rem;color:#5a8ab0;line-height:1.6;">
        ⚠️ For official information, contact the university office directly.<br><br>
        <b style="color:#8ab4d8;">Bem Thomas</b><br>
        Dept. of Mathematics &amp; Computer Science<br>
        Father Moses Orshio Adasu University
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔒 Sign Out", use_container_width=True, key="sign_out"):
            if st.session_state.messages:
                save_current_session(st.session_state.username,
                                     st.session_state.session_id,
                                     st.session_state.messages)
            for k, v in _defaults.items(): st.session_state[k] = v
            st.rerun()

    # ── TOP BANNER ───────────────────────────────────────────
    st.markdown("""
    <div class="top-header">
        <h1>🎓 Father Moses Orshio Adasu University</h1>
        <p>Student Information &amp; Academic Assistant — Makurdi</p>
    </div>""", unsafe_allow_html=True)

    # ── CHAT MESSAGES ─────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <h2>👋 Welcome to FMOAU Student Assistant</h2>
            <p>Ask about university policies, fees, exams, academic content,
            and much more.<br><br>
            Type your question below, or tap the mic button to speak.</p>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-row">'
                    f'<div class="user-bubble">{msg["content"]}</div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                body = msg["content"].replace("\n", "<br>")
                st.markdown(
                    f'<div class="bot-row">'
                    f'<div class="bot-av">🎓</div>'
                    f'<div class="bot-bubble">{body}</div>'
                    f'</div>', unsafe_allow_html=True)
                sources = msg.get("sources", [])
                if sources:
                    # Determine primary source type from top result
                    pdf_sources = [s for s in sources
                                   if s.get("retrieval_type") in
                                   ("semantic", "bm25", "local")]
                    web_sources = [s for s in sources
                                   if s.get("retrieval_type") in
                                   ("uni_web", "academic",
                                    "web — university/general",
                                    "web — academic")]

                    # Show only the source the answer came from
                    if pdf_sources and web_sources:
                        # Both used — show top PDF pages and top web source
                        pages = list(dict.fromkeys(
                            [str(s.get("page","")) for s in pdf_sources
                             if s.get("page","") and s.get("page","") != "web"]
                        ))[:3]
                        page_str = ", ".join(pages) if pages else ""
                        web_url  = web_sources[0].get("source","")
                        domain   = web_url.split("/")[2] if web_url.startswith("http") else web_url
                        src_txt  = f"Local PDF (Pages {page_str}) · {domain}" if page_str                                    else f"Local PDF · {domain}"
                    elif pdf_sources:
                        pages = list(dict.fromkeys(
                            [str(s.get("page","")) for s in pdf_sources
                             if s.get("page","") and s.get("page","") != "web"]
                        ))[:3]
                        page_str = ", ".join(pages) if pages else ""
                        fname    = pdf_sources[0].get("source","Document")
                        src_txt  = f"{fname} — Pages {page_str}" if page_str else fname
                    elif web_sources:
                        web_url = web_sources[0].get("source","")
                        domain  = web_url.split("/")[2] if web_url.startswith("http") else web_url
                        src_txt = domain
                    else:
                        src_txt = None

                    if src_txt:
                        st.markdown(
                            f'<div class="source-block">' 
                            f'<div class="src-line">'
                            f'<span class="badge badge-local">Source</span> '
                            f'{src_txt}</div></div>',
                            unsafe_allow_html=True
                        )

    # ── INPUT BAR ─────────────────────────────────────────────
    # Voice recorder shown above input when mic is clicked
    show_voice = st.session_state.get("show_voice", False)
    if show_voice:
        rc, dc, xc = st.columns([8, 1, 1])
        with rc:
            audio_rec = st.audio_input("rec",
                key=f"ar_{len(st.session_state.messages)}",
                label_visibility="collapsed")
        with dc:
            if st.button("✓", key="vdone", help="Send recording",
                         use_container_width=True):
                if audio_rec is not None:
                    try:
                        ab = audio_rec.getvalue()
                        if ab:
                            ah = hashlib.md5(ab).hexdigest()
                            if st.session_state.get("last_audio_hash") != ah:
                                st.session_state["last_audio_hash"] = ah
                                st.session_state["show_voice"] = False
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": "[Transcribing voice...]",
                                    "time": datetime.now().strftime("%I:%M %p"),
                                })
                                with st.spinner("Transcribing..."):
                                    q = transcribe_audio(ab)
                                if q:
                                    st.session_state.messages[-1]["content"] = f"[Voice] {q}"
                                    process_and_respond(q)
                                    st.rerun()
                    except Exception as e:
                        st.error(f"Voice error: {e}")
                else:
                    st.warning("Please record first.")
        with xc:
            if st.button("✕", key="vcancel", help="Cancel",
                         use_container_width=True):
                st.session_state["show_voice"] = False
                st.rerun()

    # ── Input pill ───────────────────────────────────────────
    st.markdown('<div class="input-bar-outer"><div class="input-bar-pill">',
                unsafe_allow_html=True)
    c_text, c_mic, c_send = st.columns([14, 1, 1])
    with c_text:
        input_key  = f"ti_{len(st.session_state.messages)}"
        user_input = st.text_input("msg", key=input_key,
            placeholder="Write a message…",
            label_visibility="collapsed")
    with c_mic:
        mic_pressed = st.button("🎙", key="btn_mic",
                                help="Voice question",
                                use_container_width=True)
    with c_send:
        send_pressed = st.button("➤", key="btn_send",
                                 help="Send",
                                 use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if mic_pressed:
        st.session_state["show_voice"] = not st.session_state.get("show_voice", False)
        st.rerun()

    if send_pressed and user_input.strip():
        q = user_input.strip()
        st.session_state.messages.append({
            "role": "user", "content": q,
            "time": datetime.now().strftime("%I:%M %p"),
        })
        process_and_respond(q)
        st.rerun()
# ============================================================
# ROUTER
# ============================================================
if st.session_state.logged_in:
    show_main_app()
else:
    show_login()
