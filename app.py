import numpy as np
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(
    page_title="Buddy — HR Control Tower",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --obsidian: #050505;
    --obsidian-2: #0A0A0A;
    --obsidian-3: #111111;
    --obsidian-4: #161616;
    --obsidian-5: #1E1E1E;
    --border: #1A1A1A;
    --border-2: #242424;
    --crimson: #E11D48;
    --crimson-dim: rgba(225,29,72,0.08);
    --crimson-border: rgba(225,29,72,0.2);
    --slate: #94A3B8;
    --slate-dim: #64748B;
    --white: #FFFFFF;
    --white-90: rgba(255,255,255,0.9);
    --white-60: rgba(255,255,255,0.6);
    --white-20: rgba(255,255,255,0.2);
    --white-10: rgba(255,255,255,0.1);
    --white-05: rgba(255,255,255,0.05);
    --green: #10B981;
    --green-dim: rgba(16,185,129,0.08);
    --amber: #F59E0B;
    --amber-dim: rgba(245,158,11,0.08);
    --glass: rgba(255,255,255,0.03);
    --glass-border: rgba(255,255,255,0.06);
}

html, body, [data-testid="stApp"] {
    background: var(--obsidian) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--white) !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--obsidian-2) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 240px !important;
    max-width: 240px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--obsidian-4) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 6px !important;
    color: var(--white) !important;
    font-size: 12px !important;
}

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] input {
    background: var(--obsidian-4) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 6px !important;
    color: var(--white) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    caret-color: var(--crimson) !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--crimson) !important;
    box-shadow: 0 0 0 2px rgba(225,29,72,0.12) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--slate-dim) !important; }

/* ── BUTTONS ── */
[data-testid="stButton"] button {
    background: var(--obsidian-4) !important;
    border: 1px solid var(--border-2) !important;
    color: var(--slate) !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] button:hover {
    border-color: var(--border-2) !important;
    color: var(--white) !important;
    background: var(--obsidian-5) !important;
}

/* ── PROGRESS ── */
[data-testid="stProgress"] > div {
    background: var(--obsidian-5) !important;
    height: 3px !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div > div {
    background: var(--crimson) !important;
    border-radius: 99px !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label {
    color: var(--slate) !important;
    font-size: 11px !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── RADIO ── */
[data-testid="stRadio"] label {
    color: var(--slate) !important;
    font-size: 11px !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: var(--slate-dim) !important;
    font-size: 10px !important;
    letter-spacing: 0.05em !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 99px; }

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; margin: 6px 0 !important; }

/* ── TOAST ── */
[data-testid="stToast"] {
    background: var(--obsidian-3) !important;
    border: 1px solid var(--border-2) !important;
    color: var(--white) !important;
}
</style>
""", unsafe_allow_html=True)

# ── DATA ─────────────────────────────────────────────────────────────────────
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
        "progress": 20, "status": "At Risk", "initials": "AN",
        "region": "US", "bgv": "Cleared",
        "todos": ["AI Ethics Training", "Asana Onboarding", "First Project kickoff"],
        "done": ["Laptop setup", "Slack access", "Team intro meeting"],
        "greeting": "You're on Day 5. AI Ethics is due in 9 days — let's close that gap.",
        "next_milestone": "AI Ethics Training", "days_to_milestone": 9,
        "dept": "Design",
    },
    "Marcus Lee": {
        "role": "Copywriter", "day": 12, "stage": "Tool Training",
        "progress": 60, "status": "On Track", "initials": "ML",
        "region": "UK", "bgv": "Cleared",
        "todos": ["Gemini 101 training", "30-day check-in prep"],
        "done": ["Harassment Policy", "AI Ethics", "Slack & Asana", "First brief"],
        "greeting": "Day 12 — strong progress. Two items to close before your 30-day check-in.",
        "next_milestone": "30-Day Check-in", "days_to_milestone": 18,
        "dept": "Marketing",
    },
    "Priya Sharma": {
        "role": "Account Manager", "day": 3, "stage": "Orientation",
        "progress": 40, "status": "On Track", "initials": "PS",
        "region": "IN", "bgv": "Cleared",
        "todos": ["Meet the client team", "Complete Harassment Policy", "IT setup"],
        "done": ["Laptop & badge", "Day 1 welcome session"],
        "greeting": "Welcome to Day 3, Priya. Orientation week — meeting the client team is next.",
        "next_milestone": "Harassment Policy", "days_to_milestone": 11,
        "dept": "Sales",
    },
    "Jordan Kim": {
        "role": "Designer", "day": 18, "stage": "30-Day Check-in",
        "progress": 85, "status": "Overdue", "initials": "JK",
        "region": "US", "bgv": "Cleared",
        "todos": ["Schedule 30-day check-in"],
        "done": ["All mandatory training", "First project delivered", "Tool onboarding"],
        "greeting": "One item remaining, Jordan. Schedule that check-in today.",
        "next_milestone": "30-Day Check-in", "days_to_milestone": -2,
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

STAGES = ["Intake", "IT Setup", "Orientation", "Tool Training", "First Project", "30-Day", "BGV", "Offboard", "Exit"]

MOODS = {
    "🌟": {"label": "energised", "gradient": "radial-gradient(ellipse at top left, rgba(16,185,129,0.04) 0%, transparent 60%)"},
    "😊": {"label": "good", "gradient": "radial-gradient(ellipse at top left, rgba(99,102,241,0.04) 0%, transparent 60%)"},
    "😐": {"label": "okay", "gradient": "radial-gradient(ellipse at top left, rgba(245,158,11,0.04) 0%, transparent 60%)"},
    "😔": {"label": "drained", "gradient": "radial-gradient(ellipse at top left, rgba(100,116,139,0.06) 0%, transparent 60%)"},
}

def status_colors(s):
    return {
        "On Track":  ("#10B981", "rgba(16,185,129,0.08)",  "rgba(16,185,129,0.2)"),
        "At Risk":   ("#E11D48", "rgba(225,29,72,0.08)",   "rgba(225,29,72,0.2)"),
        "Overdue":   ("#F59E0B", "rgba(245,158,11,0.08)",  "rgba(245,158,11,0.2)"),
    }.get(s, ("#94A3B8", "rgba(148,163,184,0.08)", "rgba(148,163,184,0.2)"))

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

def query_db(chunks, matrix, model, query, n=3):
    qv = np.array(model.embed_query(query))
    scores = cosine_sim(qv, matrix)
    top = np.argsort(scores)[::-1][:n]
    return [chunks[i] for i in top], [round(float(scores[i]), 3) for i in top]

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [
    ("chunks", None), ("history", []), ("user", "Anushka Naidu"),
    ("mood", "😊"), ("completed", None), ("celebrate", False),
    ("active_source", None), ("view", "Employee"),
]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.completed is None:
    st.session_state.completed = {n: [] for n in EMPLOYEES}

if st.session_state.chunks is None:
    with st.spinner(""):
        c, v, m = build_db(HR_DOCS)
        st.session_state.chunks = c
        st.session_state.vecs = v
        st.session_state.model_emb = m

if "GROQ_API_KEY" not in st.secrets:
    st.error("Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]
emp = EMPLOYEES[st.session_state.user]
sc, sc_bg, sc_border = status_colors(emp["status"])
mood_gradient = MOODS.get(st.session_state.mood, MOODS["😊"])["gradient"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 18px 14px;">
        <div style="font-family:'Playfair Display',serif;font-size:18px;font-weight:600;
                    color:#fff;letter-spacing:-0.3px;">Buddy</div>
        <div style="font-size:8.5px;color:#333;text-transform:uppercase;
                    letter-spacing:0.14em;margin-top:2px;">HR Control Tower</div>
    </div>
    <div style="height:1px;background:#1A1A1A;margin:0 0 2px;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:12px 16px 6px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:8px;font-weight:600;color:#333;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:6px;">Operator</div>', unsafe_allow_html=True)
    selected = st.selectbox("", list(EMPLOYEES.keys()),
                            index=list(EMPLOYEES.keys()).index(st.session_state.user),
                            label_visibility="collapsed")
    if selected != st.session_state.user:
        st.session_state.user = selected
        st.session_state.history = []
        st.session_state.active_source = None
        emp = EMPLOYEES[selected]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    emp = EMPLOYEES[st.session_state.user]
    sc, sc_bg, sc_border = status_colors(emp["status"])
    user_done = st.session_state.completed[st.session_state.user]
    total_t = len(emp["todos"]) + len(emp["done"])
    done_t = len(emp["done"]) + len([t for t in user_done if t in emp["todos"]])
    live_pct = int((done_t / total_t) * 100) if total_t else emp["progress"]

    st.markdown(f"""
    <div style="margin:0 12px 10px;background:var(--glass);border:1px solid var(--glass-border);
                border-radius:10px;padding:14px;backdrop-filter:blur(8px);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="width:36px;height:36px;border-radius:50%;background:{sc_bg};
                        border:1px solid {sc_border};display:flex;align-items:center;
                        justify-content:center;font-size:11px;font-weight:700;color:{sc};
                        font-family:'JetBrains Mono',monospace;flex-shrink:0;">{emp['initials']}</div>
            <div>
                <div style="font-size:12px;font-weight:600;color:#fff;">{st.session_state.user}</div>
                <div style="font-size:10px;color:#333;">{emp['role']}</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
            <div style="background:var(--obsidian-3);border:1px solid var(--border);
                        border-radius:6px;padding:8px 10px;">
                <div style="font-size:8px;color:#333;text-transform:uppercase;
                            letter-spacing:0.1em;margin-bottom:3px;">Day</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                            font-weight:500;color:#fff;">{emp['day']}</div>
            </div>
            <div style="background:var(--obsidian-3);border:1px solid var(--border);
                        border-radius:6px;padding:8px 10px;">
                <div style="font-size:8px;color:#333;text-transform:uppercase;
                            letter-spacing:0.1em;margin-bottom:3px;">Region</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                            font-weight:500;color:#fff;">{emp['region']}</div>
            </div>
        </div>
        <div style="font-size:8px;color:#333;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:5px;">Progress</div>
        <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:4px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:22px;
                         font-weight:500;color:#fff;">{live_pct}%</span>
            <span style="font-size:9px;color:#333;">{emp['stage']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(live_pct / 100)

    st.markdown(f"""
    <div style="margin:8px 12px 0;">
        <span style="background:{sc_bg};border:1px solid {sc_border};color:{sc};
                     border-radius:4px;padding:3px 8px;font-size:9px;font-weight:600;
                     letter-spacing:0.06em;text-transform:uppercase;">{emp['status']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#1A1A1A;margin:12px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div style="padding:0 16px;">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:8px;font-weight:600;color:#E11D48;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px;">⚡ Action Queue</div>', unsafe_allow_html=True)

    newly_checked = None
    for task in emp["todos"]:
        already = task in user_done
        checked = st.checkbox(task, value=already, key=f"t_{st.session_state.user}_{task}")
        if checked and not already:
            newly_checked = task
            user_done.append(task)

    if newly_checked:
        st.session_state.celebrate = True
        st.session_state.history.append({
            "role": "assistant",
            "content": f"⬡ **{newly_checked}** marked complete. Progress updated. Keep going.",
            "sources": [], "chunks": [],
        })
        st.rerun()

    if emp["done"]:
        st.markdown('<div style="font-size:8px;color:#1E1E1E;text-transform:uppercase;letter-spacing:0.12em;margin:10px 0 6px;">Cleared</div>', unsafe_allow_html=True)
        for item in emp["done"][:3]:
            st.markdown(f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;"><div style="width:5px;height:5px;border-radius:50%;background:#1E1E1E;flex-shrink:0;"></div><span style="font-size:10px;color:#1E1E1E;text-decoration:line-through;">{item}</span></div>', unsafe_allow_html=True)
        for item in user_done:
            if item not in emp["done"]:
                st.markdown(f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;"><div style="width:5px;height:5px;border-radius:50%;background:#E11D48;flex-shrink:0;"></div><span style="font-size:10px;color:#333;text-decoration:line-through;">{item}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#1A1A1A;margin:12px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div style="padding:0 16px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:8px;color:#333;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">Signal</div>', unsafe_allow_html=True)
    mood_keys = list(MOODS.keys())
    mood_idx = mood_keys.index(st.session_state.mood) if st.session_state.mood in mood_keys else 1
    mood_pick = st.radio("", mood_keys, index=mood_idx,
                         horizontal=True, label_visibility="collapsed")
    if mood_pick != st.session_state.mood:
        st.session_state.mood = mood_pick
        st.rerun()
    mood_label = MOODS[st.session_state.mood]["label"]
    st.markdown(f'<div style="font-size:9px;color:#333;margin-top:4px;">Operating at — {mood_label}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#1A1A1A;margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="padding:0 16px 16px;">', unsafe_allow_html=True)
    view = st.radio("", ["Employee", "Manager"], horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ── BALLOONS ──────────────────────────────────────────────────────────────────
if st.session_state.celebrate:
    st.balloons()
    st.session_state.celebrate = False

# ── MAIN ──────────────────────────────────────────────────────────────────────
emp = EMPLOYEES[st.session_state.user]
sc, sc_bg, sc_border = status_colors(emp["status"])
mood_gradient = MOODS.get(st.session_state.mood, MOODS["😊"])["gradient"]

if view == "Manager":
    st.markdown(f"""
    <div style="padding:28px 32px;background:{mood_gradient};">
        <div style="font-size:9px;color:#E11D48;text-transform:uppercase;letter-spacing:0.16em;
                    margin-bottom:8px;font-weight:600;">⬡ MANAGER · OVERSIGHT</div>
        <div style="font-family:'Playfair Display',serif;font-size:26px;font-weight:500;
                    color:#fff;letter-spacing:-0.5px;margin-bottom:2px;">Control Overview</div>
        <div style="font-size:12px;color:#333;">Compliance and onboarding intelligence across your organisation</div>
    </div>
    """, unsafe_allow_html=True)

    on_t = sum(1 for e in EMPLOYEES.values() if e["status"] == "On Track")
    at_r = sum(1 for e in EMPLOYEES.values() if e["status"] == "At Risk")
    ovd = sum(1 for e in EMPLOYEES.values() if e["status"] == "Overdue")
    avg = int(sum(e["progress"] for e in EMPLOYEES.values()) / len(EMPLOYEES))

    st.markdown('<div style="padding:0 32px 20px;">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, color in zip(
        [c1, c2, c3, c4],
        [len(EMPLOYEES), f"{avg}%", at_r + ovd, on_t],
        ["TOTAL OPERATORS", "AVG PROGRESS", "NEEDS ATTENTION", "ON TRACK"],
        ["#fff", "#fff", "#E11D48", "#10B981"]
    ):
        with col:
            st.markdown(f"""
            <div style="background:var(--obsidian-3);border:1px solid var(--border);
                        border-radius:8px;padding:14px 16px;">
                <div style="font-size:7.5px;color:#333;text-transform:uppercase;
                            letter-spacing:0.14em;margin-bottom:6px;font-weight:600;">{lbl}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:26px;
                            font-weight:500;color:{color};letter-spacing:-0.5px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:8px;color:#333;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:12px;font-weight:600;">Operator Registry</div>', unsafe_allow_html=True)

    for name, data in EMPLOYEES.items():
        dsc, dsc_bg, dsc_border = status_colors(data["status"])
        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;">
                <div style="width:30px;height:30px;border-radius:50%;background:{dsc_bg};
                            border:1px solid {dsc_border};display:flex;align-items:center;
                            justify-content:center;font-size:9px;font-weight:700;color:{dsc};
                            font-family:'JetBrains Mono',monospace;flex-shrink:0;">{data['initials']}</div>
                <div>
                    <div style="font-size:12px;font-weight:500;color:#fff;">{name}</div>
                    <div style="font-size:10px;color:#333;">{data['role']} · Day {data['day']}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="padding-top:10px;"></div>', unsafe_allow_html=True)
            st.progress(data["progress"] / 100)
            st.markdown(f'<div style="font-size:9px;color:#333;margin-top:-4px;font-family:JetBrains Mono,monospace;">{data["progress"]}% · {data["stage"]}</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div style="padding-top:10px;"><span style="background:{dsc_bg};border:1px solid {dsc_border};color:{dsc};border-radius:4px;padding:3px 8px;font-size:9px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;">{data["status"]}</span></div>', unsafe_allow_html=True)
            if data["status"] in ["At Risk", "Overdue"]:
                if st.button(f"Nudge", key=f"nudge_{name}"):
                    st.toast(f"Signal sent to {name.split()[0]}", icon="⬡")
        st.markdown('<div style="height:1px;background:#111;margin:2px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<br><div style="font-size:8px;color:#333;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:12px;font-weight:600;">Compliance Matrix</div>', unsafe_allow_html=True)
    dept_items = list(DEPT_COMPLIANCE.items())
    for row_items in [dept_items[:3], dept_items[3:]]:
        cols = st.columns(3)
        for col, (dept, data) in zip(cols, row_items):
            color = "#10B981" if data["pct"] >= 85 else "#F59E0B" if data["pct"] >= 60 else "#E11D48"
            with col:
                st.markdown(f"""
                <div style="background:var(--obsidian-3);border:1px solid var(--border);
                            border-radius:8px;padding:12px 14px;margin-bottom:8px;">
                    <div style="font-size:10px;font-weight:600;color:#fff;margin-bottom:6px;">{dept}</div>
                    <div style="height:2px;background:#111;border-radius:99px;overflow:hidden;margin-bottom:5px;">
                        <div style="width:{data['pct']}%;height:100%;background:{color};border-radius:99px;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                     font-weight:500;color:{color};">{data['pct']}%</span>
                        <span style="font-size:9px;color:#333;">{data['employees']} ops</span>
                    </div>
                    {"" if not data['flagged'] else f'<div style="font-size:9px;color:#E11D48;margin-top:4px;">⚡ {data["flagged"]} flagged</div>'}
                </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    first = st.session_state.user.split()[0]

    # ── LIFECYCLE BREADCRUMB ──────────────────────────────────────────────────
    current_stage = emp["stage"]
    stage_map = {"Intake": 0, "IT Setup": 1, "Orientation": 2, "Tool Training": 3,
                 "First Project": 4, "30-Day Check-in": 5, "BGV": 6, "Offboard": 7, "Exit": 8}
    current_idx = stage_map.get(current_stage, 3)

    stages_html = ""
    for i, s in enumerate(STAGES):
        if i < current_idx:
            color, bg, weight = "#333", "transparent", "400"
        elif i == current_idx:
            color, bg, weight = "#E11D48", "rgba(225,29,72,0.08)", "600"
        else:
            color, bg, weight = "#1E1E1E", "transparent", "400"
        arrow = '<span style="color:#1A1A1A;margin:0 4px;font-size:10px;">›</span>' if i < len(STAGES)-1 else ""
        stages_html += f'<span style="font-size:8px;font-weight:{weight};color:{color};background:{bg};padding:3px 8px;border-radius:4px;text-transform:uppercase;letter-spacing:0.1em;white-space:nowrap;">{s}</span>{arrow}'

    st.markdown(f"""
    <div style="background:var(--obsidian-2);border-bottom:1px solid var(--border);
                padding:10px 28px;display:flex;align-items:center;gap:0;overflow-x:auto;flex-wrap:nowrap;">
        {stages_html}
    </div>
    """, unsafe_allow_html=True)

    # ── KPI ROW ───────────────────────────────────────────────────────────────
    user_done = st.session_state.completed[st.session_state.user]
    total_t = len(emp["todos"]) + len(emp["done"])
    done_t = len(emp["done"]) + len([t for t in user_done if t in emp["todos"]])
    live_pct = int((done_t / total_t) * 100) if total_t else emp["progress"]

    st.markdown('<div style="padding:16px 28px 0;">', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        ("PROGRESS", f"{live_pct}%", sc),
        ("BGV STATUS", emp["bgv"], "#10B981"),
        ("TENURE DAY", f"D-{emp['day']}", "#94A3B8"),
        ("REGION", emp["region"], "#94A3B8"),
    ]
    for col, (lbl, val, color) in zip([k1, k2, k3, k4], kpi_data):
        with col:
            st.markdown(f"""
            <div style="background:var(--obsidian-3);border:1px solid var(--border);
                        border-radius:8px;padding:12px 14px;">
                <div style="font-size:7.5px;color:#333;text-transform:uppercase;
                            letter-spacing:0.14em;margin-bottom:5px;font-weight:600;">{lbl}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:20px;
                            font-weight:500;color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── GREETING ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:16px 28px 12px;background:{mood_gradient};">
        <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
            <div style="font-family:'Playfair Display',serif;font-size:22px;font-weight:500;
                        color:#fff;letter-spacing:-0.3px;">Hello, {first}.</div>
            <span style="background:{sc_bg};border:1px solid {sc_border};color:{sc};
                         border-radius:4px;padding:2px 8px;font-size:9px;font-weight:600;
                         letter-spacing:0.06em;text-transform:uppercase;">{emp['status']}</span>
        </div>
        <div style="font-size:12px;color:#333;border-left:2px solid #E11D48;
                    padding-left:10px;line-height:1.6;">{emp['greeting']}</div>
    </div>
    <div style="height:1px;background:#111;"></div>
    """, unsafe_allow_html=True)

    # ── MAIN BENTO: 2:1 SPLIT ─────────────────────────────────────────────────
    st.markdown('<div style="padding:16px 28px;">', unsafe_allow_html=True)
    col_chat, col_intel = st.columns([2, 1], gap="medium")

    with col_chat:
        st.markdown(f"""
        <div style="background:var(--obsidian-2);border:1px solid var(--border);
                    border-radius:10px;overflow:hidden;">
            <div style="padding:10px 14px;border-bottom:1px solid var(--border);
                        display:flex;align-items:center;justify-content:space-between;">
                <div style="font-size:8px;color:#333;text-transform:uppercase;
                            letter-spacing:0.14em;font-weight:600;">⬡ COMMAND TERMINAL</div>
                <div style="font-size:8px;color:#1E1E1E;font-family:'JetBrains Mono',monospace;">
                    {len(st.session_state.history)//2} exchanges
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chat messages
        if st.session_state.history:
            for msg in st.session_state.history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background:var(--obsidian-4);border:1px solid var(--border);
                                border-radius:8px 8px 2px 8px;padding:10px 14px;
                                margin:8px 0 8px 20%;font-size:12px;color:#94A3B8;line-height:1.55;">
                        {msg["content"]}
                    </div>""", unsafe_allow_html=True)
                else:
                    src_tags = "".join([f'<span style="background:rgba(225,29,72,0.08);color:#E11D48;border:1px solid rgba(225,29,72,0.2);border-radius:4px;padding:1px 6px;font-size:8px;font-weight:600;margin-right:4px;letter-spacing:0.06em;">{s}</span>' for s in msg.get("sources", [])])
                    st.markdown(f"""
                    <div style="margin:4px 0 4px 0;">
                        <div style="font-size:8px;color:#333;text-transform:uppercase;
                                    letter-spacing:0.12em;margin-bottom:5px;font-weight:600;">⬡ BUDDY</div>
                        <div style="background:var(--glass);border:1px solid var(--glass-border);
                                    border-radius:2px 8px 8px 8px;padding:12px 14px;
                                    margin-right:10%;backdrop-filter:blur(8px);">
                            <div style="font-size:12px;color:#94A3B8;line-height:1.7;">
                                {msg["content"]}
                            </div>
                            {('<div style="margin-top:8px;">' + src_tags + '</div>') if src_tags else ''}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    if msg.get("chunks"):
                        with st.expander("⬡ SOURCE INTEGRITY — view retrieved context", expanded=False):
                            for j, chunk in enumerate(msg["chunks"][:2]):
                                st.markdown(f"""
                                <div style="background:var(--obsidian-4);border-left:2px solid #E11D48;
                                            border-radius:0 6px 6px 0;padding:10px 12px;
                                            margin-bottom:8px;font-size:10px;color:#333;
                                            line-height:1.7;font-family:'JetBrains Mono',monospace;">
                                    <div style="font-size:8px;color:#E11D48;text-transform:uppercase;
                                                letter-spacing:0.1em;margin-bottom:6px;font-weight:600;">
                                        CHUNK {j+1} · COSINE SCORE: {msg.get('scores', [0,0])[j] if j < len(msg.get('scores',[])) else '—'}
                                    </div>
                                    {chunk}
                                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding:20px 14px;">
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                    {"".join([f'<span style="background:var(--obsidian-4);border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:10px;color:#333;cursor:pointer;">{q}</span>' for q in ["PTO policy?", "What's due this week?", "Expense submission?", "When's my 30-day check-in?"]])}
                </div>
            </div>""", unsafe_allow_html=True)

        # Input
        col_q, col_prep = st.columns([3, 1])
        with col_q:
            query = st.text_input("", placeholder=f"Query the system, {first}…",
                                  label_visibility="collapsed")
        with col_prep:
            prep = st.button("⬡ 1-on-1 Brief", use_container_width=True)

        if prep:
            with st.spinner(""):
                client = Groq(api_key=api_key)
                r = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": f"""You are Buddy, an HR assistant.
Generate a concise 1-on-1 prep brief for {first}.
Role: {emp['role']}, Day {emp['day']}, Stage: {emp['stage']}, Status: {emp['status']}
Done: {', '.join(emp['done'])}. Pending: {', '.join(emp['todos'])}.
Mood: {MOODS.get(st.session_state.mood, {}).get('label', 'neutral')}.
Write 3 short paragraphs. Professional but warm. Sign off as {first}."""}]
                )
            st.session_state.history.append({"role": "user", "content": "⬡ Generate 1-on-1 brief"})
            st.session_state.history.append({"role": "assistant", "content": r.choices[0].message.content, "sources": [], "chunks": [], "scores": []})
            st.rerun()

        if query:
            st.session_state.history.append({"role": "user", "content": query})
            with st.spinner(""):
                top_chunks, scores = query_db(st.session_state.chunks, st.session_state.vecs, st.session_state.model_emb, query)
                context = "\n\n".join(top_chunks)
                q_lower = query.lower()
                for kw in ["pto", "leave", "vacation", "training", "ethics", "tool", "asana", "bgv", "1-on-1", "expense", "onboard"]:
                    if kw in q_lower:
                        st.session_state.active_source = kw
                        break
                mood_note = ""
                m = MOODS.get(st.session_state.mood, {}).get("label", "")
                if m in ["drained", "okay"]:
                    mood_note = "Employee is feeling drained. Be concise and supportive."
                elif m == "energised":
                    mood_note = "Employee is energised. Be direct and proactive."
                client = Groq(api_key=api_key)
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": f"""You are Buddy, an HR intelligence assistant.
Employee: {st.session_state.user}, {emp['role']}, Day {emp['day']}, {emp['stage']}, {emp['status']}
Pending: {', '.join(emp['todos'])}. Done: {', '.join(emp['done'])}. {mood_note}
Handbook: {context}
Question: {query}
Answer in 2-3 sentences. Be precise, warm, and specific to this employee."""}]
                )
                buddy_reply = resp.choices[0].message.content
                sources = ["Employee Handbook"] if scores[0] > 0.3 else []
            st.session_state.history.append({"role": "assistant", "content": buddy_reply, "sources": sources, "chunks": top_chunks, "scores": scores})
            st.rerun()

        if st.session_state.history:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear terminal", type="secondary"):
                    st.session_state.history = []
                    st.session_state.active_source = None
                    st.rerun()
            with c2:
                if st.button("Reset signal", type="secondary"):
                    st.session_state.mood = "😊"
                    st.rerun()

    # ── INTELLIGENCE FEED ─────────────────────────────────────────────────────
    with col_intel:
        # Active RAG source
        active = st.session_state.active_source
        st.markdown(f"""
        <div style="background:var(--obsidian-2);border:1px solid var(--border);
                    border-radius:10px;padding:14px;margin-bottom:10px;">
            <div style="font-size:8px;color:#333;text-transform:uppercase;
                        letter-spacing:0.14em;font-weight:600;margin-bottom:10px;">⬡ LIVE SOURCE</div>
            <div style="font-size:9px;color:#1E1E1E;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">PTO Policy</div>
            <div style="font-size:10px;color:#1E1E1E;line-height:1.6;margin-bottom:4px;">20 days/year. 10 sick days separate.</div>
            {"<div style='background:rgba(225,29,72,0.06);border-left:2px solid #E11D48;padding:5px 8px;font-size:10px;color:#555;border-radius:0 4px 4px 0;'>Submit via HR Portal — 5 days advance.</div>" if active and "pto" in active.lower() else "<div style='font-size:10px;color:#1A1A1A;padding:4px 0;'>Submit via HR Portal — 5 days advance.</div>"}
            <div style="font-size:9px;color:#1E1E1E;text-transform:uppercase;letter-spacing:0.1em;margin:8px 0 3px;">Training</div>
            {"<div style='background:rgba(225,29,72,0.06);border-left:2px solid #E11D48;padding:5px 8px;font-size:10px;color:#555;border-radius:0 4px 4px 0;'>Harassment + AI Ethics + Tools — 14 days.</div>" if active and "training" in active.lower() else "<div style='font-size:10px;color:#1A1A1A;'>Harassment + AI Ethics + Tools — 14 days.</div>"}
            <div style="font-size:9px;color:#1E1E1E;text-transform:uppercase;letter-spacing:0.1em;margin:8px 0 3px;">1-on-1s</div>
            {"<div style='background:rgba(225,29,72,0.06);border-left:2px solid #E11D48;padding:5px 8px;font-size:10px;color:#555;border-radius:0 4px 4px 0;'>Every Friday. First within 3 days.</div>" if active and "1-on-1" in active.lower() else "<div style='font-size:10px;color:#1A1A1A;'>Every Friday. First within 3 days.</div>"}
            {('<div style="margin-top:8px;font-size:8px;color:#E11D48;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">↑ Source active for last query</div>') if active else ''}
        </div>
        """, unsafe_allow_html=True)

        # Next milestone
        days = emp["days_to_milestone"]
        milestone_color = "#E11D48" if days <= 0 else "#F59E0B" if days <= 5 else "#10B981"
        days_label = f"OVERDUE {abs(days)}d" if days <= 0 else f"{days}d remaining"
        st.markdown(f"""
        <div style="background:var(--obsidian-2);border:1px solid {'rgba(225,29,72,0.3)' if days <= 0 else ('rgba(245,158,11,0.3)' if days <= 5 else '#1A1A1A')};
                    border-radius:10px;padding:14px;margin-bottom:10px;">
            <div style="font-size:8px;color:#333;text-transform:uppercase;
                        letter-spacing:0.14em;font-weight:600;margin-bottom:8px;">⬡ NEXT MILESTONE</div>
            <div style="font-size:13px;font-weight:500;color:#fff;margin-bottom:4px;">{emp['next_milestone']}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{milestone_color};
                        font-weight:500;">{days_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Critical blocker tile
        if emp["status"] in ["At Risk", "Overdue"]:
            blocker_color = "#E11D48" if emp["status"] == "At Risk" else "#F59E0B"
            st.markdown(f"""
            <div style="background:rgba(225,29,72,0.04);border:1px solid rgba(225,29,72,0.2);
                        border-radius:10px;padding:14px;margin-bottom:10px;">
                <div style="font-size:8px;color:#E11D48;text-transform:uppercase;
                            letter-spacing:0.14em;font-weight:600;margin-bottom:8px;">⚡ CRITICAL BLOCKER</div>
                {"".join([f'<div style="display:flex;align-items:center;gap:6px;padding:4px 0;border-bottom:1px solid rgba(225,29,72,0.1);"><div style="width:4px;height:4px;border-radius:50%;background:#E11D48;flex-shrink:0;"></div><span style="font-size:10px;color:#94A3B8;">{t}</span></div>' for t in emp["todos"]])}
            </div>
            """, unsafe_allow_html=True)

        # Mood indicator
        mood_label = MOODS.get(st.session_state.mood, {}).get("label", "neutral")
        st.markdown(f"""
        <div style="background:var(--obsidian-2);border:1px solid var(--border);
                    border-radius:10px;padding:14px;">
            <div style="font-size:8px;color:#333;text-transform:uppercase;
                        letter-spacing:0.14em;font-weight:600;margin-bottom:8px;">⬡ SIGNAL</div>
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="font-size:22px;">{st.session_state.mood}</div>
                <div>
                    <div style="font-size:11px;font-weight:500;color:#fff;text-transform:capitalize;">{mood_label}</div>
                    <div style="font-size:9px;color:#333;">Buddy adapts to your state</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
