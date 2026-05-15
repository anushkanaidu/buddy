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
    --shadow-hover: 0 6px 24px rgba(44,44,44,0.13);
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
.main .block-container { padding: 2rem 2.5rem !important; max-width: 960px !important; }

h1,h2,h3 { font-family: 'DM Serif Display', serif !important; }

/* Greeting */
.greeting-card {
    background: linear-gradient(135deg, #E8F0E9 0%, #F5EFE6 100%);
    border-radius: 20px; padding: 1.75rem 2rem; margin-bottom: 1.5rem;
    border: 1px solid var(--sage-mid); position: relative; overflow: hidden;
}
.greeting-card::after {
    content: '🌿'; position: absolute; right: 1.5rem; top: 1rem;
    font-size: 3rem; opacity: 0.2;
}
.greet-name { font-family: 'DM Serif Display',serif; font-size: 1.65rem; margin: 0 0 0.25rem; color: var(--charcoal); }
.greet-day { font-size: 0.82rem; color: var(--mid); margin: 0 0 1rem; }
.greet-nudge {
    background: white; border-radius: 10px; padding: 0.75rem 1rem;
    font-size: 0.875rem; color: var(--charcoal); border-left: 3px solid var(--sage);
    line-height: 1.55;
}

/* Chat bubbles */
.chat-user {
    background: var(--sage); color: white;
    border-radius: 18px 18px 4px 18px; padding: 0.875rem 1.25rem;
    margin: 0.5rem 0 0.5rem 18%; font-size: 0.9rem; line-height: 1.5;
}
.chat-buddy {
    background: var(--white); color: var(--charcoal);
    border-radius: 18px 18px 18px 4px; padding: 1rem 1.25rem;
    margin: 0.5rem 18% 0.5rem 0; font-size: 0.9rem; line-height: 1.6;
    box-shadow: var(--shadow); border: 1px solid var(--border);
}
.buddy-label {
    font-size: 0.72rem; font-weight: 600; color: var(--sage);
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.35rem;
}
.source-tag {
    display: inline-block; background: var(--sage-light); color: var(--sage);
    border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.7rem;
    font-weight: 500; margin-top: 0.5rem; margin-right: 0.3rem;
}

/* Input */
[data-testid="stTextInput"] input {
    border-radius: 12px !important; border: 2px solid var(--border) !important;
    padding: 0.75rem 1rem !important; font-family: 'DM Sans',sans-serif !important;
    font-size: 0.95rem !important; background: var(--white) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(107,143,113,0.12) !important;
}

/* Progress */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--sage), var(--sage-mid)) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: var(--border) !important; border-radius: 99px !important; height: 7px !important;
}

/* Sidebar */
.sb-logo { padding: 1.5rem 1.25rem 0.75rem; }
.sb-logo-name { font-family: 'DM Serif Display',serif; font-size: 1.15rem; color: var(--sage); }
.sb-logo-sub { font-size: 0.68rem; color: var(--mid); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 1px; }
.sb-divider { border: none; border-top: 1px solid var(--border); margin: 0.25rem 0; }
.sb-section { padding: 1rem 1.25rem; }
.sb-label { font-size: 0.7rem; font-weight: 600; color: var(--mid); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
.avatar { width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: 600; margin-bottom: 0.6rem; }
.profile-name { font-weight: 600; font-size: 0.9rem; color: var(--charcoal); margin: 0; }
.profile-role { font-size: 0.76rem; color: var(--mid); margin: 0.1rem 0 0; }
.badge { display: inline-block; padding: 0.18rem 0.6rem; border-radius: 99px; font-size: 0.72rem; font-weight: 600; }
.badge-on { background: var(--sage-light); color: var(--sage); }
.badge-risk { background: var(--amber-light); color: var(--amber); }
.badge-over { background: var(--rose-light); color: var(--rose); }
.todo-item { display: flex; align-items: center; gap: 6px; padding: 0.38rem 0; font-size: 0.82rem; color: var(--charcoal); border-bottom: 1px solid var(--border); }
.todo-item:last-child { border-bottom: none; }
.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

/* Metric cards */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.85rem; margin-bottom: 1.75rem; }
.metric-card {
    background: var(--white); border-radius: 14px; padding: 1.25rem 1.5rem;
    border: 1px solid var(--border); box-shadow: var(--shadow); text-align: center;
}
.metric-num { font-family: 'DM Serif Display',serif; font-size: 2.1rem; color: var(--charcoal); line-height: 1; margin: 0; }
.metric-lbl { font-size: 0.72rem; color: var(--mid); margin-top: 0.35rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }

/* Employee rows */
.emp-row {
    background: var(--white); border-radius: 14px; padding: 1rem 1.25rem;
    margin-bottom: 0.65rem; border: 1px solid var(--border); box-shadow: var(--shadow);
    display: flex; align-items: center; gap: 1rem;
}
.emp-avatar { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.82rem; font-weight: 600; flex-shrink: 0; }
.emp-info { flex: 1; min-width: 0; }
.emp-name { font-weight: 600; font-size: 0.9rem; color: var(--charcoal); margin: 0; }
.emp-meta { font-size: 0.75rem; color: var(--mid); margin: 0.1rem 0 0; }
.emp-bar { flex: 1.2; }
.emp-bar-track { height: 6px; background: var(--border); border-radius: 99px; overflow: hidden; margin-bottom: 3px; }
.emp-bar-fill { height: 100%; border-radius: 99px; }
.emp-pct { font-size: 0.72rem; color: var(--mid); }
.emp-actions { display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0; }

/* Heatmap */
.heatmap-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 1rem; }
.heatmap-card {
    background: var(--white); border-radius: 12px; padding: 1rem 1.1rem;
    border: 1px solid var(--border);
}
.heatmap-dept { font-weight: 600; font-size: 0.85rem; color: var(--charcoal); margin-bottom: 0.4rem; }
.heatmap-bar-track { height: 8px; background: var(--border); border-radius: 99px; overflow: hidden; margin-bottom: 0.35rem; }
.heatmap-bar-fill { height: 100%; border-radius: 99px; }
.heatmap-pct { font-size: 0.75rem; color: var(--mid); }

/* Section headers */
.section-title { font-family: 'DM Serif Display',serif; font-size: 1.3rem; color: var(--charcoal); margin: 0 0 0.75rem; }
.section-sub { font-size: 0.83rem; color: var(--mid); margin: -0.5rem 0 1.25rem; }

/* Suggestions */
.suggestion-chip {
    display: inline-block; background: var(--white); border: 1px solid var(--border);
    border-radius: 99px; padding: 0.35rem 0.9rem; font-size: 0.8rem; color: var(--mid);
    margin: 0 0.35rem 0.5rem 0; cursor: pointer;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important; border: 2px solid var(--border) !important;
    background: var(--cream) !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────
hr_docs = """
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

Onboarding Stages:
- Day 1-3: Intake and IT Setup (laptop, accounts, badges)
- Day 4-7: Orientation (team intros, culture, values)
- Day 8-14: Tool Training (Slack, Asana, Figma, Gemini)
- Day 15-30: First Project (shadow team, take ownership of a small task)
- Day 30: 30-Day Check-in with manager
- Day 90: 90-Day Review with HR

Leave Policy: Sick leave is separate from PTO — 10 days per year. Parental leave is 16 weeks fully paid. Bereavement leave is 5 days.

Performance Reviews: Twice yearly in June and December. Self-assessments due 2 weeks before review.

Expense Policy: Submit within 30 days of expense. Meals up to $75 per person. Travel booked via Concur.

Compliance Requirements: All employees must complete Harassment Policy, AI Ethics, and Tool Onboarding within the first 14 days. Failure to complete within 30 days triggers an HR flag.
"""

EMPLOYEES = {
    "Anushka Naidu": {
        "role": "UX Strategist", "day": 5, "stage": "Tool Training",
        "progress": 20, "status": "At Risk", "initials": "AN", "color": "#6B8F71",
        "todos": ["AI Ethics Training", "Asana Onboarding", "First Project kickoff"],
        "done": ["Laptop setup", "Slack access", "Team intro meeting"],
        "greeting": "You're on Day 5 — right in the middle of Tool Training. Let's make sure you're set up on Asana and Figma before the week's out. Your AI Ethics training is also due in 9 days!",
        "dept": "Design"
    },
    "Marcus Lee": {
        "role": "Copywriter", "day": 12, "stage": "Tool Training",
        "progress": 60, "status": "On Track", "initials": "ML", "color": "#7B6FA0",
        "todos": ["Gemini 101 training", "30-day check-in prep"],
        "done": ["Harassment Policy", "AI Ethics", "Slack & Asana", "First brief"],
        "greeting": "Day 12 and you're looking great! Just two items left before your 30-day check-in. You've knocked out all the mandatory training — that's a big deal.",
        "dept": "Marketing"
    },
    "Priya Sharma": {
        "role": "Account Manager", "day": 3, "stage": "Orientation",
        "progress": 40, "status": "On Track", "initials": "PS", "color": "#C0616B",
        "todos": ["Meet the client team", "Complete Harassment Policy", "IT setup finalisation"],
        "done": ["Laptop & badge", "Day 1 welcome session"],
        "greeting": "Welcome to Day 3, Priya! You're right on schedule. Your next milestone is meeting the client team and finishing IT setup — both should happen this week.",
        "dept": "Sales"
    },
    "Jordan Kim": {
        "role": "Designer", "day": 18, "stage": "30-Day Check-in",
        "progress": 85, "status": "Overdue", "initials": "JK", "color": "#D4820A",
        "todos": ["Schedule 30-day check-in with manager"],
        "done": ["All mandatory training", "First project delivered", "Tool onboarding"],
        "greeting": "Almost there, Jordan! You've completed everything except one thing — your 30-day check-in still hasn't been scheduled. Reach out to your manager today to lock it in.",
        "dept": "Design"
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

def heatmap_color(pct):
    if pct >= 85: return "#6B8F71"
    if pct >= 60: return "#D4820A"
    return "#C0616B"

def cosine_similarity(query_vec, matrix):
    dots = matrix @ query_vec
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    return dots / (norms + 1e-10)

def build_vector_db(text_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text_data)
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vecs = np.array(model.embed_documents(chunks))
    return chunks, vecs, model

def query_db(chunks, matrix, model, query, n=3):
    qv = np.array(model.embed_query(query))
    scores = cosine_similarity(qv, matrix)
    top = np.argsort(scores)[::-1][:n]
    return [chunks[i] for i in top], [round(float(scores[i]), 2) for i in top]

# ── INIT ──────────────────────────────────────────────────────────
if "chunks" not in st.session_state:
    with st.spinner("Buddy is waking up…"):
        st.session_state.chunks, st.session_state.vecs, st.session_state.model = build_vector_db(hr_docs)

if "history" not in st.session_state:
    st.session_state.history = []

if "user" not in st.session_state:
    st.session_state.user = "Anushka Naidu"

if "GROQ_API_KEY" not in st.secrets:
    st.error("Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]

# Track completed tasks per user + celebration flag
if "completed_tasks" not in st.session_state:
    st.session_state.completed_tasks = {name: [] for name in EMPLOYEES}
if "celebrate" not in st.session_state:
    st.session_state.celebrate = False

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
        st.rerun()

    emp = EMPLOYEES[st.session_state.user]
    badge_map = {"On Track": "badge-on", "At Risk": "badge-risk", "Overdue": "badge-over"}

    st.markdown(f"""
    <div class="avatar" style="background:{emp['color']}22;color:{emp['color']};">{emp['initials']}</div>
    <p class="profile-name">{st.session_state.user}</p>
    <p class="profile-role">{emp['role']} · Day {emp['day']}</p>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin:0.6rem 0 0.25rem;"></div>', unsafe_allow_html=True)
    user_completed_preview = st.session_state.completed_tasks.get(st.session_state.user, [])
    total_tasks_preview = len(emp['todos']) + len(emp['done'])
    tasks_done_preview = len(emp['done']) + len(user_completed_preview)
    display_progress = int((tasks_done_preview / total_tasks_preview) * 100) if total_tasks_preview > 0 else emp['progress']
    st.progress(display_progress / 100)
    st.markdown(f'<p style="font-size:0.72rem;color:var(--mid);margin-top:0.2rem;">{display_progress}% · {emp["stage"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge {badge_map[emp["status"]]}" style="margin-top:0.4rem;display:inline-block;">{emp["status"]}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Live Tasks</div>', unsafe_allow_html=True)

    user_completed = st.session_state.completed_tasks[st.session_state.user]
    all_todos = emp['todos']
    newly_checked = None

    for task in all_todos:
        already_done = task in user_completed
        checked = st.checkbox(task, value=already_done, key=f"task_{st.session_state.user}_{task}")
        if checked and not already_done:
            newly_checked = task
            user_completed.append(task)

    if newly_checked:
        st.session_state.celebrate = newly_checked
        st.session_state.history.append({
            "role": "assistant",
            "content": f"🎉 You just completed **{newly_checked}**! That's a big deal — your progress is growing. Keep going, you're doing great!",
            "sources": []
        })
        st.rerun()

    remaining = [t for t in all_todos if t not in user_completed]
    total_tasks = len(all_todos) + len(emp['done'])
    tasks_done = len(emp['done']) + len(user_completed)
    live_progress = int((tasks_done / total_tasks) * 100) if total_tasks > 0 else emp['progress']

    if emp['done']:
        st.markdown('<div class="sb-label" style="margin-top:0.75rem;">Done ✓</div>', unsafe_allow_html=True)
        for item in emp['done']:
            st.markdown(f'<div class="todo-item"><div class="dot" style="background:#6B8F71;"></div><span style="color:var(--mid);">{item}</span></div>', unsafe_allow_html=True)
        for item in user_completed:
            if item not in emp['done']:
                st.markdown(f'<div class="todo-item"><div class="dot" style="background:#6B8F71;"></div><span style="color:var(--mid);">{item}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div style="padding:0.75rem 1.25rem 1.5rem;">', unsafe_allow_html=True)
    view = st.radio("View", ["Employee Portal", "HR Manager View"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────
if view == "HR Manager View":
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

    st.markdown('<p class="section-title" style="font-size:1.1rem;margin-bottom:0.5rem;">Employee Progress</p>', unsafe_allow_html=True)

    for name, data in EMPLOYEES.items():
        bc = {"On Track": "#6B8F71", "At Risk": "#D4820A", "Overdue": "#C0616B"}[data['status']]
        badge_cls = badge_map[data['status']]
        col1, col2, col3 = st.columns([3, 3, 1.8])

        with col1:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:0.5rem 0;">
                <div class="emp-avatar" style="background:{data['color']}22;color:{data['color']};">{data['initials']}</div>
                <div>
                    <div style="font-weight:600;font-size:0.9rem;">{name}</div>
                    <div style="font-size:0.75rem;color:var(--mid);">{data['role']} · Day {data['day']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div style="padding-top:0.7rem;"></div>', unsafe_allow_html=True)
            st.progress(data['progress'] / 100)
            st.markdown(f'<p style="font-size:0.72rem;color:var(--mid);margin-top:-0.4rem;">{data["progress"]}% · {data["stage"]}</p>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div style="padding-top:0.55rem;display:flex;flex-direction:column;gap:6px;">', unsafe_allow_html=True)
            st.markdown(f'<span class="badge {badge_cls}">{data["status"]}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if data['status'] in ['At Risk', 'Overdue']:
                if st.button(f"Nudge {name.split()[0]}", key=f"nudge_{name}"):
                    st.toast(f"Nudge sent to {name.split()[0]}! 👋", icon="✅")

        st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:0.15rem 0;">', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size:1.1rem;margin-bottom:0.4rem;">Compliance by Department</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub" style="margin-bottom:0.75rem;">Percentage of employees with all mandatory training complete</p>', unsafe_allow_html=True)

    dept_items = list(DEPT_COMPLIANCE.items())
    rows = [dept_items[i:i+3] for i in range(0, len(dept_items), 3)]
    for row in rows:
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
                        <span class="heatmap-pct" style="color:{color};font-weight:600;">{data['pct']}% compliant</span>
                        <span style="font-size:0.7rem;color:var(--mid);">{data['employees']} employees</span>
                    </div>
                    {"" if data['flagged'] == 0 else f'<div style="font-size:0.72rem;color:#B85460;margin-top:0.35rem;">⚠ {data[chr(102)+"lagged"]} flagged</div>'}
                </div>
                """, unsafe_allow_html=True)

else:
    first = st.session_state.user.split()[0]

    # Trigger balloons if task was just completed
    if st.session_state.celebrate:
        st.balloons()
        st.session_state.celebrate = False

    st.markdown(f"""
    <div class="greeting-card">
        <p class="greet-name">Hey, {first} 👋</p>
        <p class="greet-day">Day {emp['day']} · {emp['stage']}</p>
        <div class="greet-nudge">💬 {emp['greeting']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── MOOD CHECK ────────────────────────────────────────────────
    if "mood" not in st.session_state:
        st.session_state.mood = None

    MOODS = {
        "🌟 Energised": {"label": "energised", "color": "#6B8F71", "bg": "#E8F0E9"},
        "😊 Good":      {"label": "good",       "color": "#7B6FA0", "bg": "#EEEDF8"},
        "😐 Okay":      {"label": "okay",        "color": "#D4820A", "bg": "#FEF3E2"},
        "😔 Drained":   {"label": "drained",     "color": "#B85460", "bg": "#FDEAEC"},
    }

    if st.session_state.mood is None:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:1.1rem 1.25rem;
                    border:1px solid var(--border);margin-bottom:1.25rem;">
            <p style="font-size:0.85rem;font-weight:500;color:var(--charcoal);margin:0 0 0.65rem;">
                How are you feeling today, <strong>{first}</strong>?
            </p>
        """.replace("{first}", first), unsafe_allow_html=True)

        mood_cols = st.columns(4)
        for col, (emoji_label, mdata) in zip(mood_cols, MOODS.items()):
            with col:
                if st.button(emoji_label, key=f"mood_{emoji_label}", use_container_width=True):
                    st.session_state.mood = mdata['label']
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        mood_data = next(m for m in MOODS.values() if m['label'] == st.session_state.mood)
        st.markdown(f"""
        <div style="background:{mood_data['bg']};border-radius:10px;padding:0.6rem 1rem;
                    margin-bottom:1.25rem;display:flex;align-items:center;justify-content:space-between;">
            <span style="font-size:0.82rem;color:{mood_data['color']};font-weight:500;">
                Feeling {st.session_state.mood} today — Buddy will keep that in mind 💙
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── CHAT ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title" style="font-size:1.15rem;margin-bottom:0.3rem;">Ask Buddy anything</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.83rem;color:var(--mid);margin-bottom:1rem;">HR policies, onboarding steps, tool access, leave — Buddy reads the handbook so you don\'t have to.</p>', unsafe_allow_html=True)

    if st.session_state.history:
        for i, msg in enumerate(st.session_state.history):
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                sources_html = "".join([f'<span class="source-tag">📄 {s}</span>' for s in msg.get('sources', [])])
                st.markdown(f'<div class="buddy-label">🤝 Buddy</div><div class="chat-buddy">{msg["content"]}{("<br>" + sources_html) if sources_html else ""}</div>', unsafe_allow_html=True)
                if msg.get('chunks'):
                    with st.expander('🔍 View source from handbook', expanded=False):
                        for j, chunk in enumerate(msg['chunks'][:2]):
                            st.markdown(f'<div style="background:#FAF7F2;border-left:3px solid #6B8F71;border-radius:0 8px 8px 0;padding:0.75rem 1rem;margin-bottom:0.5rem;font-size:0.82rem;color:#2C2C2C;line-height:1.6;"><span style="font-size:0.68rem;font-weight:600;color:#6B8F71;text-transform:uppercase;letter-spacing:0.05em;">Source {j+1} — Employee Handbook</span><br><br>{chunk}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom:1rem;">
            <span class="suggestion-chip">What's the PTO policy?</span>
            <span class="suggestion-chip">What do I need this week?</span>
            <span class="suggestion-chip">How do I submit expenses?</span>
            <span class="suggestion-chip">When's my 30-day check-in?</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 1-ON-1 PREP BUTTON ────────────────────────────────────────
    col_input, col_oneon = st.columns([4, 1.4])
    with col_input:
        query = st.text_input("", placeholder=f"Ask Buddy something, {first}…", label_visibility="collapsed")
    with col_oneon:
        prep_clicked = st.button("📋 Prep my 1-on-1", use_container_width=True)

    if prep_clicked:
        with st.spinner("Drafting your 1-on-1 summary…"):
            client = Groq(api_key=api_key)
            prep_prompt = f"""You are Buddy, an HR assistant helping {first} prepare for their weekly Friday 1-on-1 with their manager.

Employee context:
- Name: {st.session_state.user}, Role: {emp['role']}
- Day {emp['day']} of onboarding, Stage: {emp['stage']}
- Completed this week: {', '.join(emp['done'])}
- Still pending: {', '.join(emp['todos'])}
- Current status: {emp['status']}
- Energy today: {st.session_state.mood or 'not specified'}

Write a short, friendly 1-on-1 prep summary they can copy-paste to their manager.
Include: what went well, what's still in progress, one thing they need support with.
Keep it warm, honest, 3-4 short paragraphs. Sign off as if from {first}."""

            prep_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prep_prompt}]
            )
            prep_text = prep_response.choices[0].message.content

        st.session_state.history.append({"role": "user", "content": "📋 Generate my 1-on-1 prep summary"})
        st.session_state.history.append({"role": "assistant", "content": prep_text, "sources": []})
        st.rerun()

    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.spinner(""):
            top_chunks, scores = query_db(
                st.session_state.chunks, st.session_state.vecs,
                st.session_state.model, query
            )
            context = "\n\n".join(top_chunks)
            client = Groq(api_key=api_key)

            mood_instruction = ""
            if st.session_state.mood in ["drained", "okay"]:
                mood_instruction = f"The employee has indicated they are feeling {st.session_state.mood} today. Be extra gentle, brief, and supportive. Don't overwhelm them with tasks."
            elif st.session_state.mood == "energised":
                mood_instruction = "The employee is feeling energised today. You can be upbeat and proactive — suggest what they could get ahead on."

            prompt = f"""You are Buddy, a warm and knowledgeable HR assistant.
You know this employee personally:
- Name: {st.session_state.user}
- Role: {emp['role']}
- Day {emp['day']} of onboarding, currently in: {emp['stage']}
- Status: {emp['status']}
- Still to complete: {', '.join(emp['todos'])}
- Already done: {', '.join(emp['done'])}
{mood_instruction}

HR Handbook context:
{context}

Employee question: {query}

Respond warmly and specifically. Reference their personal situation when relevant.
2-4 sentences max. Write naturally, no bullet lists."""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            buddy_reply = response.choices[0].message.content
            sources = ["Employee Handbook"] if scores[0] > 0.3 else []

        st.session_state.history.append({"role": "assistant", "content": buddy_reply, "sources": sources, "chunks": top_chunks})
        st.rerun()

    if st.session_state.history:
        col_clear, col_mood_reset = st.columns([2, 1])
        with col_clear:
            if st.button("Clear conversation", type="secondary"):
                st.session_state.history = []
                st.rerun()
        with col_mood_reset:
            if st.button("Reset mood", type="secondary"):
                st.session_state.mood = None
                st.rerun()
