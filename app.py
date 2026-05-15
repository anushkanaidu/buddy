import os
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

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
.main .block-container { padding: 0 !important; max-width: 100% !important; }

h1,h2,h3 { font-family: 'DM Serif Display', serif !important; }

/* Breadcrumb */
.breadcrumb {
    background: var(--white);
    border-bottom: 1px solid var(--border);
    padding: 0.6rem 2rem;
    display: flex; align-items: center; gap: 0;
    overflow-x: auto; flex-wrap: nowrap;
}
.bc-stage {
    font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; padding: 0.25rem 0.6rem; border-radius: 4px;
    white-space: nowrap;
}
.bc-stage.past { color: #CCC; }
.bc-stage.active { color: var(--sage); background: var(--sage-light); }
.bc-stage.future { color: #DDD; }
.bc-arrow { color: #DDD; margin: 0 3px; font-size: 0.75rem; }

/* KPI tiles */
.kpi-wrap { padding: 1rem 2rem 0; display: grid; grid-template-columns: repeat(4,1fr); gap: 0.75rem; }
.kpi-tile {
    background: var(--white); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.9rem 1rem;
}
.kpi-lbl { font-size: 0.62rem; font-weight: 600; color: var(--mid); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.kpi-val { font-family: 'DM Serif Display', serif; font-size: 1.4rem; color: var(--charcoal); line-height: 1; }

/* Greeting */
.greeting-card {
    background: linear-gradient(135deg, #E8F0E9 0%, #F5EFE6 100%);
    border-radius: 16px; padding: 1.5rem 1.75rem; margin-bottom: 1.25rem;
    border: 1px solid var(--sage-mid); position: relative; overflow: hidden;
}
.greeting-card::after {
    content: '🌿'; position: absolute; right: 1.5rem; top: 1rem;
    font-size: 2.5rem; opacity: 0.2;
}
.greet-name { font-family: 'DM Serif Display',serif; font-size: 1.5rem; margin: 0 0 0.2rem; color: var(--charcoal); }
.greet-day { font-size: 0.78rem; color: var(--mid); margin: 0 0 0.85rem; }
.greet-nudge {
    background: white; border-radius: 8px; padding: 0.65rem 0.9rem;
    font-size: 0.85rem; color: var(--charcoal); border-left: 3px solid var(--sage); line-height: 1.55;
}

/* Chat bubbles */
.chat-user {
    background: var(--sage); color: white;
    border-radius: 16px 16px 4px 16px; padding: 0.8rem 1.1rem;
    margin: 0.5rem 0 0.5rem 18%; font-size: 0.88rem; line-height: 1.5;
}
.chat-buddy {
    background: var(--white); color: var(--charcoal);
    border-radius: 16px 16px 16px 4px; padding: 0.9rem 1.1rem;
    margin: 0.5rem 18% 0.5rem 0; font-size: 0.88rem; line-height: 1.6;
    box-shadow: var(--shadow); border: 1px solid var(--border);
    position: relative;
}
.buddy-label {
    font-size: 0.65rem; font-weight: 600; color: var(--sage);
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.verified-badge {
    position: absolute; top: 0.75rem; right: 0.75rem;
    background: var(--sage-light); color: var(--sage);
    border: 1px solid var(--sage-mid); border-radius: 4px;
    padding: 0.1rem 0.5rem; font-size: 0.6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.source-tag {
    display: inline-block; background: var(--sage-light); color: var(--sage);
    border-radius: 5px; padding: 0.12rem 0.45rem; font-size: 0.65rem;
    font-weight: 500; margin-top: 0.45rem; margin-right: 0.25rem;
}

/* Input */
[data-testid="stTextInput"] input {
    border-radius: 10px !important; border: 2px solid var(--border) !important;
    padding: 0.7rem 1rem !important; font-family: 'DM Sans',sans-serif !important;
    font-size: 0.92rem !important; background: var(--white) !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(107,143,113,0.1) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #CCC !important; }

/* Progress */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--sage), var(--sage-mid)) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: var(--border) !important; border-radius: 99px !important; height: 6px !important;
}

/* Buttons */
[data-testid="stButton"] button {
    background: var(--white) !important; border: 1px solid var(--border) !important;
    color: var(--mid) !important; border-radius: 8px !important;
    font-family: 'DM Sans',sans-serif !important; font-size: 0.8rem !important;
}
[data-testid="stButton"] button:hover {
    border-color: var(--sage) !important; color: var(--sage) !important;
    background: var(--sage-light) !important;
}

/* Sidebar */
.sb-logo { padding: 1.4rem 1.2rem 0.7rem; }
.sb-logo-name { font-family: 'DM Serif Display',serif; font-size: 1.1rem; color: var(--sage); }
.sb-logo-sub { font-size: 0.65rem; color: var(--mid); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 1px; }
.sb-divider { border: none; border-top: 1px solid var(--border); margin: 0.2rem 0; }
.sb-section { padding: 0.9rem 1.1rem; }
.sb-label { font-size: 0.67rem; font-weight: 600; color: var(--mid); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.45rem; }
.avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
.profile-name { font-weight: 600; font-size: 0.88rem; color: var(--charcoal); margin: 0; }
.profile-role { font-size: 0.73rem; color: var(--mid); margin: 0.1rem 0 0; }
.badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 99px; font-size: 0.68rem; font-weight: 600; }
.badge-on { background: var(--sage-light); color: var(--sage); }
.badge-risk { background: var(--amber-light); color: var(--amber); }
.badge-over { background: var(--rose-light); color: var(--rose); }
.todo-item { display: flex; align-items: center; gap: 6px; padding: 0.35rem 0; font-size: 0.8rem; color: var(--charcoal); border-bottom: 1px solid var(--border); }
.todo-item:last-child { border-bottom: none; }
.dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

/* Checkbox */
[data-testid="stCheckbox"] label { font-size: 0.8rem !important; color: var(--charcoal) !important; }

/* Radio */
[data-testid="stRadio"] label { font-size: 0.8rem !important; }

/* Metric cards */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.5rem; }
.metric-card {
    background: var(--white); border-radius: 12px; padding: 1.1rem 1.3rem;
    border: 1px solid var(--border); box-shadow: var(--shadow); text-align: center;
}
.metric-num { font-family: 'DM Serif Display',serif; font-size: 1.9rem; color: var(--charcoal); line-height: 1; margin: 0; }
.metric-lbl { font-size: 0.67rem; color: var(--mid); margin-top: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

/* Heatmap */
.heatmap-card {
    background: var(--white); border-radius: 10px; padding: 0.9rem 1rem;
    border: 1px solid var(--border);
}
.heatmap-dept { font-weight: 600; font-size: 0.82rem; color: var(--charcoal); margin-bottom: 0.4rem; }
.heatmap-bar-track { height: 6px; background: var(--border); border-radius: 99px; overflow: hidden; margin-bottom: 0.3rem; }
.heatmap-bar-fill { height: 100%; border-radius: 99px; }

/* Section headers */
.section-title { font-family: 'DM Serif Display',serif; font-size: 1.25rem; color: var(--charcoal); margin: 0 0 0.65rem; }
.section-sub { font-size: 0.8rem; color: var(--mid); margin: -0.4rem 0 1.1rem; }

/* Suggestion chips */
.suggestion-chip {
    display: inline-block; background: var(--white); border: 1px solid var(--border);
    border-radius: 99px; padding: 0.3rem 0.8rem; font-size: 0.78rem; color: var(--mid);
    margin: 0 0.3rem 0.4rem 0;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important; border: 1px solid var(--border) !important;
    background: var(--cream) !important;
}

/* Expander */
[data-testid="stExpander"] { border-radius: 10px !important; border: 1px solid var(--border) !important; }

/* Mood radio override */
[data-testid="stRadio"] > div { gap: 6px !important; flex-wrap: wrap !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
hr { border-color: var(--border) !important; margin: 6px 0 !important; }
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
        "progress": 20, "status": "At Risk", "initials": "AN", "color": "#6B8F71",
        "region": "US", "bgv": "Cleared",
        "todos": ["AI Ethics Training", "Asana Onboarding", "First Project kickoff"],
        "done": ["Laptop setup", "Slack access", "Team intro meeting"],
        "greeting": "You're on Day 5 — right in Tool Training. AI Ethics is due in 9 days. Let's get Asana set up before the week's out.",
        "dept": "Design",
    },
    "Marcus Lee": {
        "role": "Copywriter", "day": 12, "stage": "Tool Training",
        "progress": 60, "status": "On Track", "initials": "ML", "color": "#7B6FA0",
        "region": "UK", "bgv": "Cleared",
        "todos": ["Gemini 101 training", "30-day check-in prep"],
        "done": ["Harassment Policy", "AI Ethics", "Slack & Asana", "First brief"],
        "greeting": "Day 12 and looking great! Two items left before your 30-day check-in. You've already knocked out all mandatory training.",
        "dept": "Marketing",
    },
    "Priya Sharma": {
        "role": "Account Manager", "day": 3, "stage": "Orientation",
        "progress": 40, "status": "On Track", "initials": "PS", "color": "#C0616B",
        "region": "IN", "bgv": "Cleared",
        "todos": ["Meet the client team", "Complete Harassment Policy", "IT setup finalisation"],
        "done": ["Laptop & badge", "Day 1 welcome session"],
        "greeting": "Welcome to Day 3, Priya! Right on schedule. Meeting the client team is your next milestone this week.",
        "dept": "Sales",
    },
    "Jordan Kim": {
        "role": "Designer", "day": 18, "stage": "30-Day Check-in",
        "progress": 85, "status": "Overdue", "initials": "JK", "color": "#D4820A",
        "region": "US", "bgv": "Cleared",
        "todos": ["Schedule 30-day check-in with manager"],
        "done": ["All mandatory training", "First project delivered", "Tool onboarding"],
        "greeting": "Almost there, Jordan. One thing left — your 30-day check-in. Reach out to your manager today to lock it in.",
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

STAGES = ["Intake", "IT Setup", "Orientation", "Tool Training", "First Project", "30-Day", "BGV", "Exit"]
STAGE_MAP = {"Intake": 0, "IT Setup": 1, "Orientation": 2, "Tool Training": 3,
             "First Project": 4, "30-Day Check-in": 5, "BGV": 6, "Exit": 7}

MOODS = {
    "🌟 Energised": {"label": "energised", "color": "#6B8F71", "bg": "#E8F0E9"},
    "😊 Good":      {"label": "good",       "color": "#7B6FA0", "bg": "#EEEDF8"},
    "😐 Okay":      {"label": "okay",        "color": "#D4820A", "bg": "#FEF3E2"},
    "😔 Drained":   {"label": "drained",     "color": "#B85460", "bg": "#FDEAEC"},
}

def heatmap_color(pct):
    if pct >= 85: return "#6B8F71"
    if pct >= 60: return "#D4820A"
    return "#C0616B"

def badge_cls(status):
    return {"On Track": "badge-on", "At Risk": "badge-risk", "Overdue": "badge-over"}.get(status, "badge-on")

def cosine_similarity(qv, matrix):
    dots = matrix @ qv
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(qv)
    return dots / (norms + 1e-10)

def build_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vecs = np.array(model.embed_documents(chunks))
    return chunks, vecs, model

def query_db(chunks, matrix, model, query, n=3):
    qv = np.array(model.embed_query(query))
    scores = cosine_similarity(qv, matrix)
    top = np.argsort(scores)[::-1][:n]
    return [chunks[i] for i in top], [round(float(scores[i]), 2) for i in top]

# ── SESSION STATE ─────────────────────────────────────────────────
if "chunks" not in st.session_state:
    with st.spinner(""):
        st.session_state.chunks, st.session_state.vecs, st.session_state.model = build_vector_db(HR_DOCS)
if "history" not in st.session_state:
    st.session_state.history = []
if "user" not in st.session_state:
    st.session_state.user = "Anushka Naidu"
if "mood" not in st.session_state:
    st.session_state.mood = None
if "completed_tasks" not in st.session_state:
    st.session_state.completed_tasks = {n: [] for n in EMPLOYEES}
if "celebrate" not in st.session_state:
    st.session_state.celebrate = False

if "GROQ_API_KEY" not in st.secrets:
    st.error("Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()
api_key = st.secrets["GROQ_API_KEY"]

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-name">🤝 Buddy</div>
        <div class="sb-logo-sub">HR & Compliance Assistant</div>
    </div>
    <hr class="sb-divider">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Logged in as</div>', unsafe_allow_html=True)
    selected = st.selectbox("", list(EMPLOYEES.keys()),
                            index=list(EMPLOYEES.keys()).index(st.session_state.user),
                            label_visibility="collapsed")
    if selected != st.session_state.user:
        st.session_state.user = selected
        st.session_state.history = []
        st.session_state.mood = None
        st.rerun()

    emp = EMPLOYEES[st.session_state.user]
    user_done = st.session_state.completed_tasks[st.session_state.user]
    total_t = len(emp['todos']) + len(emp['done'])
    newly_done = len([t for t in user_done if t in emp['todos']])
    done_t = len(emp['done']) + newly_done
    live_pct = int((done_t / total_t) * 100) if total_t else emp['progress']

    st.markdown(f"""
    <div class="avatar" style="background:{emp['color']}22;color:{emp['color']};">{emp['initials']}</div>
    <p class="profile-name">{st.session_state.user}</p>
    <p class="profile-role">{emp['role']} · Day {emp['day']}</p>
    """, unsafe_allow_html=True)
    st.markdown('<div style="margin:0.5rem 0 0.2rem;"></div>', unsafe_allow_html=True)
    st.progress(live_pct / 100)
    st.markdown(f'<p style="font-size:0.68rem;color:var(--mid);margin-top:0.2rem;">{live_pct}% · {emp["stage"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge {badge_cls(emp["status"])}" style="margin-top:0.3rem;display:inline-block;">{emp["status"]}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Live Tasks</div>', unsafe_allow_html=True)

    newly_checked = None
    for task in emp['todos']:
        already = task in user_done
        checked = st.checkbox(task, value=already, key=f"task_{st.session_state.user}_{task}")
        if checked and not already:
            newly_checked = task
            user_done.append(task)

    if newly_checked:
        st.session_state.celebrate = True
        st.session_state.history.append({
            "role": "assistant",
            "content": f"🎉 **{newly_checked}** marked complete! Your progress is growing. Keep going.",
            "sources": [], "chunks": [],
        })
        st.rerun()

    if emp['done']:
        st.markdown('<div class="sb-label" style="margin-top:0.65rem;">Done ✓</div>', unsafe_allow_html=True)
        for item in emp['done']:
            st.markdown(f'<div class="todo-item"><div class="dot" style="background:#6B8F71;"></div><span style="color:var(--mid);">{item}</span></div>', unsafe_allow_html=True)
        for item in user_done:
            if item not in emp['done']:
                st.markdown(f'<div class="todo-item"><div class="dot" style="background:#6B8F71;"></div><span style="color:var(--mid);">{item}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Energy today</div>', unsafe_allow_html=True)
    mood_keys = list(MOODS.keys())
    mood_idx = mood_keys.index(st.session_state.mood) if st.session_state.mood in mood_keys else None
    mood_pick = st.radio("", mood_keys, index=mood_idx, horizontal=True, label_visibility="collapsed")
    if mood_pick and mood_pick != st.session_state.mood:
        st.session_state.mood = mood_pick
        st.rerun()
    if st.session_state.mood and st.session_state.mood in MOODS:
        mdata = MOODS[st.session_state.mood]
        st.markdown(f'<div style="font-size:0.72rem;color:{mdata["color"]};margin-top:4px;">Feeling {MOODS[st.session_state.mood]["label"]} — Buddy knows 💙</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div style="padding:0.6rem 1.1rem 1.2rem;">', unsafe_allow_html=True)
    view = st.radio("", ["Employee Portal", "HR Manager View"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ── BALLOONS ──────────────────────────────────────────────────────
if st.session_state.celebrate:
    st.balloons()
    st.session_state.celebrate = False

emp = EMPLOYEES[st.session_state.user]

# ── MANAGER VIEW ──────────────────────────────────────────────────
if view == "HR Manager View":
    st.markdown('<div style="padding:1.5rem 2rem;">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Manager Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Compliance and onboarding status across your team</p>', unsafe_allow_html=True)

    on_track = sum(1 for e in EMPLOYEES.values() if e['status'] == 'On Track')
    at_risk  = sum(1 for e in EMPLOYEES.values() if e['status'] == 'At Risk')
    overdue  = sum(1 for e in EMPLOYEES.values() if e['status'] == 'Overdue')
    avg_prog = int(sum(e['progress'] for e in EMPLOYEES.values()) / len(EMPLOYEES))

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><p class="metric-num">{len(EMPLOYEES)}</p><p class="metric-lbl">Total Employees</p></div>
        <div class="metric-card"><p class="metric-num">{avg_prog}%</p><p class="metric-lbl">Avg Progress</p></div>
        <div class="metric-card" style="border-color:#FDEAEC;"><p class="metric-num" style="color:#B85460;">{at_risk + overdue}</p><p class="metric-lbl">Needs Attention</p></div>
        <div class="metric-card" style="border-color:#E8F0E9;"><p class="metric-num" style="color:#6B8F71;">{on_track}</p><p class="metric-lbl">On Track</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title" style="font-size:1.05rem;margin-bottom:0.5rem;">Employee Progress</p>', unsafe_allow_html=True)
    for name, data in EMPLOYEES.items():
        col1, col2, col3 = st.columns([3, 3, 1.8])
        with col1:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:0.5rem 0;">
                <div style="width:32px;height:32px;border-radius:50%;background:{data['color']}22;
                    color:{data['color']};display:flex;align-items:center;justify-content:center;
                    font-size:0.82rem;font-weight:700;flex-shrink:0;">{data['initials']}</div>
                <div>
                    <div style="font-weight:600;font-size:0.88rem;">{name}</div>
                    <div style="font-size:0.72rem;color:var(--mid);">{data['role']} · Day {data['day']}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="padding-top:0.6rem;"></div>', unsafe_allow_html=True)
            st.progress(data['progress'] / 100)
            st.markdown(f'<p style="font-size:0.68rem;color:var(--mid);margin-top:-0.3rem;">{data["progress"]}% · {data["stage"]}</p>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div style="padding-top:0.5rem;"><span class="badge {badge_cls(data["status"])}">{data["status"]}</span></div>', unsafe_allow_html=True)
            if data['status'] in ['At Risk', 'Overdue']:
                if st.button(f"Nudge {name.split()[0]}", key=f"nudge_{name}"):
                    st.toast(f"Nudge sent to {name.split()[0]}! 👋", icon="✅")
        st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:0.1rem 0;">', unsafe_allow_html=True)

    st.markdown('<br><p class="section-title" style="font-size:1.05rem;margin-bottom:0.4rem;">Compliance by Department</p>', unsafe_allow_html=True)
    dept_items = list(DEPT_COMPLIANCE.items())
    for row in [dept_items[:3], dept_items[3:]]:
        cols = st.columns(3)
        for col, (dept, data) in zip(cols, row):
            color = heatmap_color(data['pct'])
            with col:
                st.markdown(f"""
                <div class="heatmap-card">
                    <div class="heatmap-dept">{dept}</div>
                    <div class="heatmap-bar-track">
                        <div class="heatmap-bar-fill" style="width:{data['pct']}%;background:{color};"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:0.72rem;color:{color};font-weight:600;">{data['pct']}% compliant</span>
                        <span style="font-size:0.67rem;color:var(--mid);">{data['employees']} employees</span>
                    </div>
                    {"" if not data['flagged'] else f'<div style="font-size:0.68rem;color:#B85460;margin-top:0.3rem;">⚠ {data["flagged"]} flagged</div>'}
                </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── EMPLOYEE PORTAL ───────────────────────────────────────────────
else:
    first = st.session_state.user.split()[0]

    # Breadcrumb
    current_idx = STAGE_MAP.get(emp['stage'], 3)
    stages_html = ""
    for i, s in enumerate(STAGES):
        if i < current_idx:
            cls = "past"
        elif i == current_idx:
            cls = "active"
        else:
            cls = "future"
        arrow = '<span class="bc-arrow">›</span>' if i < len(STAGES)-1 else ""
        stages_html += f'<span class="bc-stage {cls}">{s}</span>{arrow}'
    st.markdown(f'<div class="breadcrumb">{stages_html}</div>', unsafe_allow_html=True)

    # KPI tiles
    user_done = st.session_state.completed_tasks[st.session_state.user]
    total_t = len(emp['todos']) + len(emp['done'])
    done_t = len(emp['done']) + len([t for t in user_done if t in emp['todos']])
    live_pct = int((done_t / total_t) * 100) if total_t else emp['progress']

    kpi_color = {"On Track": "#6B8F71", "At Risk": "#C17D0E", "Overdue": "#B85460"}[emp['status']]
    st.markdown(f"""
    <div class="kpi-wrap">
        <div class="kpi-tile">
            <div class="kpi-lbl">Progress</div>
            <div class="kpi-val" style="color:{kpi_color};">{live_pct}%</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">BGV Status</div>
            <div class="kpi-val" style="color:#6B8F71;">{emp['bgv']}</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">Tenure Day</div>
            <div class="kpi-val">D-{emp['day']}</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">Region</div>
            <div class="kpi-val">{emp['region']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:1rem 2rem;">', unsafe_allow_html=True)

    # Greeting
    st.markdown(f"""
    <div class="greeting-card">
        <p class="greet-name">Hey, {first} 👋</p>
        <p class="greet-day">Day {emp['day']} · {emp['stage']}</p>
        <div class="greet-nudge">💬 {emp['greeting']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Mood banner
    if st.session_state.mood and st.session_state.mood in MOODS:
        mdata = MOODS[st.session_state.mood]
        st.markdown(f"""
        <div style="background:{mdata['bg']};border-radius:8px;padding:0.55rem 0.9rem;
                    margin-bottom:1rem;border-left:3px solid {mdata['color']};">
            <span style="font-size:0.8rem;color:{mdata['color']};font-weight:500;">
                {st.session_state.mood} — Buddy is in {mdata['label']} mode for you today
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Chat section
    st.markdown('<p class="section-title" style="font-size:1.1rem;margin-bottom:0.25rem;">Ask Buddy anything</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:var(--mid);margin-bottom:0.85rem;">HR policies, onboarding steps, tool access, leave — Buddy reads the handbook so you don\'t have to.</p>', unsafe_allow_html=True)

    if st.session_state.history:
        for msg in st.session_state.history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                src_html = "".join([f'<span class="source-tag">📄 {s}</span>' for s in msg.get('sources', [])])
                verified = '<span class="verified-badge">Verified Policy</span>' if msg.get('sources') else ''
                st.markdown(f'<div class="buddy-label">🤝 Buddy</div><div class="chat-buddy">{verified}{msg["content"]}{("<br>" + src_html) if src_html else ""}</div>', unsafe_allow_html=True)
                if msg.get('chunks'):
                    with st.expander("🔍 View source from handbook", expanded=False):
                        for j, chunk in enumerate(msg['chunks'][:2]):
                            st.markdown(f'<div style="background:#FAF7F2;border-left:3px solid #6B8F71;border-radius:0 8px 8px 0;padding:0.7rem 0.9rem;margin-bottom:0.5rem;font-size:0.8rem;color:#2C2C2C;line-height:1.6;"><span style="font-size:0.62rem;font-weight:700;color:#6B8F71;text-transform:uppercase;letter-spacing:0.06em;">Source {j+1} — Employee Handbook</span><br><br>{chunk}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom:0.85rem;">
            <span class="suggestion-chip">What's the PTO policy?</span>
            <span class="suggestion-chip">What do I need this week?</span>
            <span class="suggestion-chip">How do I submit expenses?</span>
            <span class="suggestion-chip">When's my 30-day check-in?</span>
        </div>
        """, unsafe_allow_html=True)

    col_input, col_prep = st.columns([4, 1.4])
    with col_input:
        query = st.text_input("", placeholder=f"Ask Buddy something, {first}…", label_visibility="collapsed")
    with col_prep:
        prep_clicked = st.button("📋 Prep my 1-on-1", use_container_width=True)

    if prep_clicked:
        with st.spinner(""):
            client = Groq(api_key=api_key)
            mood_label = MOODS.get(st.session_state.mood, {}).get("label", "not specified") if st.session_state.mood else "not specified"
            r = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"""You are Buddy, helping {first} prep for their Friday 1-on-1.
Employee: {st.session_state.user}, {emp['role']}, Day {emp['day']}, {emp['stage']}, {emp['status']}
Done: {', '.join(emp['done'])}. Pending: {', '.join(emp['todos'])}.
Mood: {mood_label}.
Write 3 short warm paragraphs they can copy-paste to their manager. Sign off as {first}."""}]
            )
        st.session_state.history.append({"role": "user", "content": "📋 Generate my 1-on-1 prep"})
        st.session_state.history.append({"role": "assistant", "content": r.choices[0].message.content, "sources": [], "chunks": []})
        st.rerun()

    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.spinner(""):
            top_chunks, scores = query_db(st.session_state.chunks, st.session_state.vecs, st.session_state.model, query)
            context = "\n\n".join(top_chunks)
            mood_note = ""
            if st.session_state.mood:
                ml = MOODS.get(st.session_state.mood, {}).get("label", "")
                if ml in ["drained", "okay"]:
                    mood_note = "Employee is feeling drained. Be gentle, brief, one thing at a time."
                elif ml == "energised":
                    mood_note = "Employee is energised. Be upbeat and proactive."
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"""You are Buddy, a warm HR assistant.
Employee: {st.session_state.user}, {emp['role']}, Day {emp['day']}, {emp['stage']}, {emp['status']}
Pending: {', '.join(emp['todos'])}. Done: {', '.join(emp['done'])}. {mood_note}
Handbook: {context}
Question: {query}
Answer in 2-4 sentences. Warm, specific, no bullet lists."""}]
            )
            buddy_reply = resp.choices[0].message.content
            sources = ["Employee Handbook"] if scores[0] > 0.3 else []
        st.session_state.history.append({"role": "assistant", "content": buddy_reply, "sources": sources, "chunks": top_chunks})
        st.rerun()

    if st.session_state.history:
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("Clear conversation", type="secondary"):
                st.session_state.history = []
                st.rerun()
        with c2:
            if st.button("Reset mood", type="secondary"):
                st.session_state.mood = None
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
