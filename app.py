import numpy as np
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(
    page_title="Buddy — HR Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

:root {
    --cream: #FAF7F2;
    --white: #FFFFFF;
    --sage: #6B8F71;
    --sage-light: #E8F0E9;
    --sage-mid: #A8C5AD;
    --amber: #C17D0E;
    --amber-light: #FEF3E2;
    --rose: #B85460;
    --rose-light: #FDEAEC;
    --charcoal: #2C2C2C;
    --mid: #6B6B6B;
    --border: #E8E4DF;
    --shadow: 0 2px 12px rgba(44,44,44,0.07);
}

html, body, [data-testid="stApp"] {
    background-color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--charcoal) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--white) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
.main .block-container { padding: 2rem 2rem !important; max-width: 100% !important; }

/* Sidebar */
.sb-brand { font-family: 'DM Serif Display',serif; font-size: 1.15rem; color: var(--sage); }
.sb-sub { font-size: 0.65rem; color: var(--mid); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }
.sb-div { border: none; border-top: 1px solid var(--border); margin: 0.25rem 0; }
.sb-lbl { font-size: 0.67rem; font-weight: 600; color: var(--mid); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
.avatar { width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: 700; }
.badge { display: inline-block; padding: 0.18rem 0.65rem; border-radius: 99px; font-size: 0.7rem; font-weight: 600; }
.badge-on { background: var(--sage-light); color: var(--sage); }
.badge-risk { background: var(--amber-light); color: var(--amber); }
.badge-over { background: var(--rose-light); color: var(--rose); }

/* Task card */
.task-card {
    background: var(--white); border-radius: 14px; padding: 1.4rem 1.6rem;
    border: 1px solid var(--border); box-shadow: var(--shadow); margin-bottom: 1.25rem;
}
.task-card-title {
    font-family: 'DM Serif Display', serif; font-size: 1.1rem;
    color: var(--charcoal); margin: 0 0 1rem;
}
.task-item { display: flex; align-items: center; gap: 8px; padding: 0.4rem 0; border-bottom: 1px solid var(--border); }
.task-item:last-child { border-bottom: none; }
.task-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.task-done-text { color: #C5C0BA; font-size: 0.85rem; text-decoration: line-through; }
.task-todo-text { color: var(--charcoal); font-size: 0.85rem; }

/* Greeting */
.greeting-card {
    background: linear-gradient(135deg, #E8F0E9 0%, #F5EFE6 100%);
    border-radius: 16px; padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;
    border: 1px solid var(--sage-mid); position: relative; overflow: hidden;
}
.greeting-card::after {
    content: '🌿'; position: absolute; right: 1.5rem; top: 1rem;
    font-size: 2.5rem; opacity: 0.18;
}
.greet-name { font-family: 'DM Serif Display', serif; font-size: 1.6rem; margin: 0 0 0.2rem; color: var(--charcoal); }
.greet-sub { font-size: 0.78rem; color: var(--mid); margin: 0 0 0.9rem; }
.greet-nudge {
    background: white; border-radius: 10px; padding: 0.75rem 1rem;
    font-size: 0.875rem; color: var(--charcoal); border-left: 3px solid var(--sage); line-height: 1.6;
}

/* Chat bubbles */
.chat-user {
    background: var(--sage); color: white;
    border-radius: 16px 16px 4px 16px; padding: 0.8rem 1.1rem;
    margin: 0.5rem 0 0.5rem 12%; font-size: 0.875rem; line-height: 1.5;
}
.buddy-label {
    font-size: 0.65rem; font-weight: 600; color: var(--sage);
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.chat-buddy {
    background: var(--white); color: var(--charcoal);
    border-radius: 16px 16px 16px 4px; padding: 0.9rem 1.1rem;
    margin: 0.2rem 12% 0.5rem 0; font-size: 0.875rem; line-height: 1.65;
    box-shadow: var(--shadow); border: 1px solid var(--border);
}
.source-tag {
    display: inline-block; background: var(--sage-light); color: var(--sage);
    border-radius: 5px; padding: 0.12rem 0.5rem; font-size: 0.65rem;
    font-weight: 500; margin-top: 0.45rem;
}

/* Input */
[data-testid="stTextInput"] input {
    border-radius: 12px !important; border: 2px solid var(--border) !important;
    padding: 0.75rem 1rem !important; font-size: 0.92rem !important;
    background: var(--white) !important; color: var(--charcoal) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(107,143,113,0.1) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #C0BDB8 !important; }

/* Progress */
[data-testid="stProgress"] > div {
    background: var(--border) !important; border-radius: 99px !important; height: 6px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--sage), var(--sage-mid)) !important; border-radius: 99px !important;
}

/* Buttons */
[data-testid="stButton"] button {
    background: var(--white) !important; border: 1px solid var(--border) !important;
    color: var(--mid) !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.8rem !important;
}
[data-testid="stButton"] button:hover {
    border-color: var(--sage) !important; color: var(--sage) !important;
    background: var(--sage-light) !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label { font-size: 0.85rem !important; color: var(--charcoal) !important; }

/* Radio */
[data-testid="stRadio"] label { font-size: 0.8rem !important; color: var(--charcoal) !important; }
[data-testid="stRadio"] > div { gap: 6px !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important; border: 1px solid var(--border) !important;
    background: var(--cream) !important; color: var(--charcoal) !important;
}

/* Expander */
[data-testid="stExpander"] { border-radius: 10px !important; border: 1px solid var(--border) !important; }
[data-testid="stExpander"] summary { font-size: 0.78rem !important; color: var(--mid) !important; }

/* Manager table */
.mgr-row {
    background: var(--white); border-radius: 12px; padding: 0.9rem 1.2rem;
    border: 1px solid var(--border); margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 1rem;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────
HR_DOCS = """
EMPLOYEE HANDBOOK

PTO Policy: Employees receive 20 days of Paid Time Off per year. Requests must be submitted via the HR Portal at least 5 business days in advance.

BGV (Background Verification): Must be completed before Day 1. US employees use HireRight. UK employees use Credence. India employees use AuthBridge.

Tools & Access: Slack for communication, Asana for project management, Figma for design, Gemini for AI tasks, Google Workspace for documents and email.

IT Support: Email it@company.com for hardware issues, software access, or VPN setup. Laptops are assigned on Day 1.

Mandatory Training (first 14 days):
1. Harassment Policy — complete online via LMS
2. AI Ethics Training — complete online via LMS
3. Tool Onboarding — Slack, Asana, Figma setup sessions

1-on-1s: Weekly check-ins every Friday with your direct manager. First 1-on-1 should happen within the first 3 days.

Onboarding Stages: Day 1-3 Intake and IT Setup. Day 4-7 Orientation. Day 8-14 Tool Training. Day 15-30 First Project. Day 30 Check-in. Day 90 Review.

Leave Policy: Sick leave is 10 days per year separate from PTO. Parental leave is 16 weeks fully paid. Bereavement leave is 5 days.

Performance Reviews: Twice yearly in June and December. Self-assessments due 2 weeks before review.

Expense Policy: Submit within 30 days. Meals up to $75 per person. Travel booked via Concur.

Compliance: All employees must complete Harassment Policy, AI Ethics, and Tool Onboarding within the first 14 days. Failure triggers an HR flag after 30 days.
"""

EMPLOYEES = {
    "Anushka Naidu": {
        "role": "UX Strategist", "day": 5, "stage": "Tool Training",
        "status": "At Risk", "initials": "AN", "color": "#6B8F71",
        "todos": ["AI Ethics Training", "Asana Onboarding", "First Project kickoff"],
        "done": ["Laptop setup", "Slack access", "Team intro meeting"],
        "greeting": "You're on Day 5 — right in Tool Training week. AI Ethics is due in 9 days. Let's get Asana set up before the week's out.",
        "dept": "Design",
    },
    "Marcus Lee": {
        "role": "Copywriter", "day": 12, "stage": "Tool Training",
        "status": "On Track", "initials": "ML", "color": "#7B6FA0",
        "todos": ["Gemini 101 training", "30-day check-in prep"],
        "done": ["Harassment Policy", "AI Ethics", "Slack & Asana", "First brief"],
        "greeting": "Day 12 and looking great! Two items left before your 30-day check-in. You've knocked out all mandatory training already.",
        "dept": "Marketing",
    },
    "Priya Sharma": {
        "role": "Account Manager", "day": 3, "stage": "Orientation",
        "status": "On Track", "initials": "PS", "color": "#C0616B",
        "todos": ["Meet the client team", "Complete Harassment Policy", "IT setup finalisation"],
        "done": ["Laptop & badge", "Day 1 welcome session"],
        "greeting": "Welcome to Day 3, Priya! Right on schedule. Meeting the client team is your next milestone this week.",
        "dept": "Sales",
    },
    "Jordan Kim": {
        "role": "Designer", "day": 18, "stage": "30-Day Check-in",
        "status": "Overdue", "initials": "JK", "color": "#D4820A",
        "todos": ["Schedule 30-day check-in with manager"],
        "done": ["All mandatory training", "First project delivered", "Tool onboarding"],
        "greeting": "Almost there, Jordan! One thing left — your 30-day check-in hasn't been scheduled yet. Reach out to your manager today.",
        "dept": "Design",
    },
}

DEPT_COMPLIANCE = {
    "Design": {"pct": 52, "employees": 8, "flagged": 4},
    "Marketing": {"pct": 78, "employees": 12, "flagged": 2},
    "Sales": {"pct": 91, "employees": 15, "flagged": 1},
    "Engineering": {"pct": 45, "employees": 20, "flagged": 11},
    "Operations": {"pct": 83, "employees": 6, "flagged": 1},
    "Finance": {"pct": 96, "employees": 9, "flagged": 0},
}

MOODS = {
    "🌟 Energised": {"label": "energised", "color": "#6B8F71", "bg": "#E8F0E9"},
    "😊 Good":      {"label": "good",       "color": "#7B6FA0", "bg": "#EEEDF8"},
    "😐 Okay":      {"label": "okay",       "color": "#D4820A", "bg": "#FEF3E2"},
    "😔 Drained":   {"label": "drained",    "color": "#B85460", "bg": "#FDEAEC"},
}

def badge_cls(s):
    return {"On Track":"badge-on","At Risk":"badge-risk","Overdue":"badge-over"}.get(s,"badge-on")

def heatmap_color(p):
    return "#6B8F71" if p >= 85 else "#D4820A" if p >= 60 else "#C0616B"

def cosine_sim(qv, matrix):
    dots = matrix @ qv
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(qv)
    return dots / (norms + 1e-10)

def build_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vecs = np.array(model.embed_documents(chunks))
    return chunks, vecs, model

def query_db(chunks, matrix, model, q, n=3):
    qv = np.array(model.embed_query(q))
    scores = cosine_sim(qv, matrix)
    top = np.argsort(scores)[::-1][:n]
    return [chunks[i] for i in top], [round(float(scores[i]), 2) for i in top]

def ask_buddy(question, emp_name, emp, chunks, vecs, model_emb, api_key, mood_label=""):
    top_chunks, scores = query_db(chunks, vecs, model_emb, question)
    context = "\n\n".join(top_chunks)
    mood_note = ""
    if mood_label in ["drained", "okay"]:
        mood_note = "The employee is feeling drained. Be gentle and brief."
    elif mood_label == "energised":
        mood_note = "The employee is energised. Be upbeat and proactive."
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"""You are Buddy, a warm HR assistant.
Employee: {emp_name}, {emp['role']}, Day {emp['day']}, {emp['stage']}, {emp['status']}.
Pending: {', '.join(emp['todos'])}. Done: {', '.join(emp['done'])}. {mood_note}
Handbook: {context}
Question: {question}
Answer in 2-4 warm sentences. No bullet lists."""}]
    )
    return resp.choices[0].message.content, scores[0] > 0.3, top_chunks

# ── SESSION STATE ──────────────────────────────────────────────────
if "chunks" not in st.session_state:
    with st.spinner(""):
        c, v, m = build_db(HR_DOCS)
        st.session_state.chunks, st.session_state.vecs, st.session_state.model_emb = c, v, m

for k, v in [("history", []), ("user", "Anushka Naidu"), ("mood", None),
             ("completed", None), ("celebrate", False), ("last_query", ""),
             ("view", "Employee")]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.completed is None:
    st.session_state.completed = {n: [] for n in EMPLOYEES}

if "GROQ_API_KEY" not in st.secrets:
    st.error("Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()
api_key = st.secrets["GROQ_API_KEY"]

# ── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 1.2rem 0.7rem;">
        <div class="sb-brand">🤝 Buddy</div>
        <div class="sb-sub">HR & Compliance Assistant</div>
    </div>
    <hr class="sb-div">
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0.9rem 1.1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="sb-lbl">Logged in as</div>', unsafe_allow_html=True)
    selected = st.selectbox("", list(EMPLOYEES.keys()),
                            index=list(EMPLOYEES.keys()).index(st.session_state.user),
                            label_visibility="collapsed")
    if selected != st.session_state.user:
        st.session_state.user = selected
        st.session_state.history = []
        st.session_state.last_query = ""
        st.session_state.mood = None
        st.rerun()

    emp = EMPLOYEES[st.session_state.user]
    user_done = st.session_state.completed[st.session_state.user]
    total = len(emp["todos"]) + len(emp["done"])
    done_count = len(emp["done"]) + len([t for t in user_done if t in emp["todos"]])
    pct = int((done_count / total) * 100) if total else 0

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-top:0.75rem;">
        <div class="avatar" style="background:{emp['color']}22;color:{emp['color']};">{emp['initials']}</div>
        <div>
            <div style="font-weight:600;font-size:0.9rem;">{st.session_state.user}</div>
            <div style="font-size:0.73rem;color:var(--mid);">{emp['role']} · Day {emp['day']}</div>
        </div>
    </div>
    <span class="badge {badge_cls(emp['status'])}" style="margin-top:0.5rem;display:inline-block;">{emp['status']}</span>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:0.6rem;"></div>', unsafe_allow_html=True)
    st.progress(pct / 100)
    st.markdown(f'<div style="font-size:0.68rem;color:var(--mid);margin-top:0.2rem;">{pct}% complete</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-div"><div style="padding:0.75rem 1.1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="sb-lbl">How are you feeling?</div>', unsafe_allow_html=True)
    mood_keys = list(MOODS.keys())
    mood_idx = mood_keys.index(st.session_state.mood) if st.session_state.mood in mood_keys else None
    mood_pick = st.radio("", mood_keys, index=mood_idx, label_visibility="collapsed")
    if mood_pick and mood_pick != st.session_state.mood:
        st.session_state.mood = mood_pick
        st.rerun()
    if st.session_state.mood and st.session_state.mood in MOODS:
        md = MOODS[st.session_state.mood]
        st.markdown(f'<div style="font-size:0.72rem;color:{md["color"]};margin-top:4px;">Buddy will keep that in mind 💙</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-div"><div style="padding:0.75rem 1.1rem 1.2rem;">', unsafe_allow_html=True)
    view = st.radio("", ["My Dashboard", "Manager View"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ── BALLOONS ──────────────────────────────────────────────────────
if st.session_state.celebrate:
    st.balloons()
    st.session_state.celebrate = False

emp = EMPLOYEES[st.session_state.user]
first = st.session_state.user.split()[0]
mood_label = MOODS.get(st.session_state.mood, {}).get("label", "")

# ── MANAGER VIEW ──────────────────────────────────────────────────
if view == "Manager View":
    st.markdown(f'<div style="font-family:DM Serif Display,serif;font-size:1.6rem;margin-bottom:0.25rem;">Manager Overview</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.82rem;color:var(--mid);margin-bottom:1.5rem;">Onboarding and compliance status across your team</div>', unsafe_allow_html=True)

    on_t = sum(1 for e in EMPLOYEES.values() if e['status']=='On Track')
    at_r = sum(1 for e in EMPLOYEES.values() if e['status']=='At Risk')
    ovd  = sum(1 for e in EMPLOYEES.values() if e['status']=='Overdue')
    avg  = int(sum(e['progress'] if 'progress' not in e else 0 for e in EMPLOYEES.values()) / len(EMPLOYEES))

    c1,c2,c3,c4 = st.columns(4)
    for col, num, lbl, color in [
        (c1, len(EMPLOYEES), "Total Employees", "#2C2C2C"),
        (c2, f"{int(sum([50,60,40,85])/4)}%", "Avg Progress", "#2C2C2C"),
        (c3, at_r+ovd, "Needs Attention", "#B85460"),
        (c4, on_t, "On Track", "#6B8F71"),
    ]:
        with col:
            st.markdown(f"""<div style="background:var(--white);border:1px solid var(--border);
                border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;">
                <div style="font-size:0.65rem;color:var(--mid);text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:0.3rem;">{lbl}</div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:{color};">{num}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.82rem;font-weight:600;color:var(--mid);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem;">Employee Progress</div>', unsafe_allow_html=True)
    for name, data in EMPLOYEES.items():
        col1, col2, col3 = st.columns([3,3,1.5])
        with col1:
            st.markdown(f"""<div style="display:flex;align-items:center;gap:9px;padding:0.4rem 0;">
                <div style="width:32px;height:32px;border-radius:50%;background:{data['color']}22;
                    color:{data['color']};display:flex;align-items:center;justify-content:center;
                    font-size:0.8rem;font-weight:700;flex-shrink:0;">{data['initials']}</div>
                <div><div style="font-weight:600;font-size:0.88rem;">{name}</div>
                    <div style="font-size:0.72rem;color:var(--mid);">{data['role']} · Day {data['day']}</div>
                </div></div>""", unsafe_allow_html=True)
        with col2:
            progress_val = {"Anushka Naidu":50,"Marcus Lee":60,"Priya Sharma":40,"Jordan Kim":85}.get(name,50)
            st.markdown('<div style="padding-top:0.55rem;"></div>', unsafe_allow_html=True)
            st.progress(progress_val/100)
            st.markdown(f'<div style="font-size:0.68rem;color:var(--mid);margin-top:-0.2rem;">{progress_val}% · {data["stage"]}</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div style="padding-top:0.5rem;"><span class="badge {badge_cls(data["status"])}">{data["status"]}</span></div>', unsafe_allow_html=True)
            if data['status'] in ['At Risk','Overdue']:
                if st.button(f"Nudge", key=f"nudge_{name}"):
                    st.toast(f"Nudge sent to {name.split()[0]}! 👋", icon="✅")
        st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:0.1rem 0;">', unsafe_allow_html=True)

    st.markdown('<br><div style="font-size:0.82rem;font-weight:600;color:var(--mid);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem;">Compliance by Department</div>', unsafe_allow_html=True)
    dept_items = list(DEPT_COMPLIANCE.items())
    for row in [dept_items[:3], dept_items[3:]]:
        cols = st.columns(3)
        for col, (dept, data) in zip(cols, row):
            color = heatmap_color(data['pct'])
            with col:
                st.markdown(f"""<div style="background:var(--white);border:1px solid var(--border);
                    border-radius:10px;padding:0.9rem 1rem;margin-bottom:0.6rem;">
                    <div style="font-weight:600;font-size:0.85rem;margin-bottom:0.4rem;">{dept}</div>
                    <div style="height:5px;background:var(--border);border-radius:99px;overflow:hidden;margin-bottom:0.35rem;">
                        <div style="width:{data['pct']}%;height:100%;background:{color};border-radius:99px;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="font-size:0.72rem;color:{color};font-weight:600;">{data['pct']}% compliant</span>
                        <span style="font-size:0.68rem;color:var(--mid);">{data['employees']} people</span>
                    </div>
                    {"" if not data['flagged'] else f'<div style="font-size:0.68rem;color:#B85460;margin-top:0.3rem;">⚠ {data["flagged"]} flagged</div>'}
                </div>""", unsafe_allow_html=True)

# ── EMPLOYEE DASHBOARD ─────────────────────────────────────────────
else:
    # Greeting
    st.markdown(f"""
    <div class="greeting-card">
        <p class="greet-name">Hey, {first} 👋</p>
        <p class="greet-sub">Day {emp['day']} · {emp['stage']}</p>
        <div class="greet-nudge">💬 {emp['greeting']}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.mood and st.session_state.mood in MOODS:
        md = MOODS[st.session_state.mood]
        st.markdown(f"""<div style="background:{md['bg']};border-radius:8px;padding:0.5rem 0.9rem;
            margin-bottom:1.25rem;border-left:3px solid {md['color']};">
            <span style="font-size:0.8rem;color:{md['color']};font-weight:500;">
                {st.session_state.mood} — Buddy is tuned in to how you're feeling today 💙
            </span></div>""", unsafe_allow_html=True)

    # Two column layout: tasks left, buddy chat right
    col_tasks, col_chat = st.columns([1, 1.4], gap="large")

    # ── TASKS COLUMN ──────────────────────────────────────────────
    with col_tasks:
        st.markdown('<div style="font-family:\'DM Serif Display\',serif;font-size:1.2rem;margin-bottom:1rem;">Your Onboarding Tasks</div>', unsafe_allow_html=True)

        user_done = st.session_state.completed[st.session_state.user]

        # Pending tasks
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.67rem;font-weight:600;color:#E8612A;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;">To Do</div>', unsafe_allow_html=True)

        newly_checked = None
        if emp["todos"]:
            for task in emp["todos"]:
                already = task in user_done
                checked = st.checkbox(task, value=already, key=f"t_{st.session_state.user}_{task}")
                if checked and not already:
                    newly_checked = task
                    user_done.append(task)
        else:
            st.markdown('<div style="font-size:0.85rem;color:var(--mid);padding:0.5rem 0;">All tasks complete! 🎉</div>', unsafe_allow_html=True)

        if newly_checked:
            st.session_state.celebrate = True
            st.session_state.history.append({
                "role": "assistant",
                "content": f"🎉 You just completed **{newly_checked}**! Great work — your progress is growing.",
                "source": False, "chunks": [],
            })
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Completed tasks
        all_done = emp["done"] + [t for t in user_done if t in emp["todos"]]
        if all_done:
            st.markdown('<div class="task-card" style="margin-top:0.75rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.67rem;font-weight:600;color:var(--sage);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;">Done ✓</div>', unsafe_allow_html=True)
            for item in all_done:
                st.markdown(f'<div class="task-item"><div class="task-dot" style="background:var(--sage-mid);"></div><span class="task-done-text">{item}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Progress summary
        total = len(emp["todos"]) + len(emp["done"])
        done_count = len(emp["done"]) + len([t for t in user_done if t in emp["todos"]])
        pct = int((done_count / total) * 100) if total else 0
        st.markdown(f'<div style="margin-top:1rem;font-size:0.78rem;color:var(--mid);">{done_count} of {total} tasks complete</div>', unsafe_allow_html=True)
        st.progress(pct / 100)

    # ── BUDDY CHAT COLUMN ─────────────────────────────────────────
    with col_chat:
        st.markdown('<div style="font-family:\'DM Serif Display\',serif;font-size:1.2rem;margin-bottom:1rem;">Ask Buddy anything</div>', unsafe_allow_html=True)

        # Chat history
        if st.session_state.history:
            for msg in st.session_state.history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    src = '<div class="source-tag">📄 Employee Handbook</div>' if msg.get("source") else ""
                    st.markdown(f'<div class="buddy-label">🤝 Buddy</div><div class="chat-buddy">{msg["content"]}{src}</div>', unsafe_allow_html=True)
                    if msg.get("chunks"):
                        with st.expander("🔍 View source from handbook"):
                            for j, chunk in enumerate(msg["chunks"][:2]):
                                st.markdown(f'<div style="background:#FAF7F2;border-left:3px solid #6B8F71;border-radius:0 8px 8px 0;padding:0.7rem 0.9rem;margin-bottom:0.5rem;font-size:0.8rem;line-height:1.6;"><span style="font-size:0.62rem;font-weight:700;color:#6B8F71;text-transform:uppercase;letter-spacing:0.06em;">Source {j+1}</span><br><br>{chunk}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.82rem;color:var(--mid);margin-bottom:0.75rem;">Pick a question or type your own:</div>', unsafe_allow_html=True)
            chips = ["What's the PTO policy?", "What do I need this week?", "How do I submit expenses?", "When's my 30-day check-in?"]
            c1, c2 = st.columns(2)
            for i, chip in enumerate(chips):
                col = c1 if i % 2 == 0 else c2
                with col:
                    if st.button(chip, key=f"chip_{i}", use_container_width=True):
                        if chip != st.session_state.last_query:
                            st.session_state.last_query = chip
                            st.session_state.history.append({"role": "user", "content": chip})
                            with st.spinner(""):
                                reply, has_source, chunks = ask_buddy(chip, st.session_state.user, emp, st.session_state.chunks, st.session_state.vecs, st.session_state.model_emb, api_key, mood_label)
                            st.session_state.history.append({"role": "assistant", "content": reply, "source": has_source, "chunks": chunks})
                            st.rerun()

        # Input
        col_q, col_prep = st.columns([3, 1.2])
        with col_q:
            query = st.text_input("", placeholder=f"Ask Buddy something…", label_visibility="collapsed")
        with col_prep:
            prep = st.button("📋 1-on-1 Prep", use_container_width=True)

        if prep:
            with st.spinner(""):
                client = Groq(api_key=api_key)
                r = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": f"""You are Buddy. Write a short friendly 1-on-1 prep message for {first} to send their manager.
Role: {emp['role']}, Day {emp['day']}, {emp['stage']}, {emp['status']}.
Done: {', '.join(emp['done'])}. Pending: {', '.join(emp['todos'])}.
3-4 sentences. Warm tone. Sign off with just '{first}'."""}]
                )
            st.session_state.history.append({"role": "user", "content": "📋 1-on-1 prep"})
            st.session_state.history.append({"role": "assistant", "content": r.choices[0].message.content, "source": False, "chunks": []})
            st.rerun()

        if query and query.strip() and query != st.session_state.last_query:
            st.session_state.last_query = query
            st.session_state.history.append({"role": "user", "content": query})
            with st.spinner(""):
                reply, has_source, chunks = ask_buddy(query, st.session_state.user, emp, st.session_state.chunks, st.session_state.vecs, st.session_state.model_emb, api_key, mood_label)
            st.session_state.history.append({"role": "assistant", "content": reply, "source": has_source, "chunks": chunks})
            st.rerun()

        if st.session_state.history:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear chat", type="secondary"):
                    st.session_state.history = []
                    st.session_state.last_query = ""
                    st.rerun()
            with c2:
                if st.button("Reset mood", type="secondary"):
                    st.session_state.mood = None
                    st.rerun()
