import numpy as np
import streamlit as st
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# ── Data ──────────────────────────────────────────────────────────────────────

hr_docs = """
EMPLOYEE HANDBOOK — v2.1  (Updated: May 2026)

PTO Policy: Employees receive 20 days of Paid Time Off per year. Requests must be submitted via the HR Portal at portal.company.com. Unused PTO can be carried over up to 5 days per year.

BGV: Background verification must be completed before Day 1. US employees use HireRight, UK employees use Credence. BGV typically takes 3–5 business days.

Tools: Slack (team communication), Asana (project management), Figma (design), Gemini (AI assistant), Google Workspace (documents and email). Access is provisioned by IT on Day 0.

IT Support: Email itsupport@company.com or raise a ticket in the IT Portal for any hardware or access issues. Response time is within 4 business hours.

Mandatory Training: All employees must complete three trainings within the first 14 days: (1) Harassment Prevention Policy, (2) AI Ethics & Responsible Use, (3) Tool Onboarding. Completion is tracked automatically in the HR Portal.

1-on-1s: Weekly every Friday with your manager. First 1-on-1 happens on Day 5. Meetings are 30 minutes and focus on onboarding progress, blockers, and questions.

Wellness: Employees receive a $50/month wellness stipend, redeemable for gym memberships, meditation apps, or mental health services. Submit receipts via the HR Portal.

Sync-Off Hours: Company encourages a daily 30-minute Sync-Off break. No meetings are scheduled between 12:00–12:30 PM. Employees are encouraged to step away from screens.

Onboarding Stages: (1) Intake & Paperwork, (2) IT Setup, (3) Orientation, (4) Meet Your Team, (5) Mandatory Training, (6) First Project, (7) 30-Day Check-in, (8) 90-Day Review.

Remote Work: Employees may work remotely up to 3 days per week after completing the first 30 days. Requests are managed through the HR Portal.

Holidays: 12 paid public holidays per year. The holiday calendar is published in January and available on the company intranet.

Payroll: Payroll is processed bi-monthly on the 1st and 15th of each month. Contact payroll@company.com for any queries. Direct deposit takes 1–2 business days.

Security Training: All employees must complete Security Awareness Training within their first 14 days. Failure to complete by Day 14 triggers an automatic compliance flag.
"""

ONBOARDING_STAGES = [
    ("Intake",        "Day 0",     "1"),
    ("IT Setup",      "Day 0–1",   "2"),
    ("Orientation",   "Day 1–2",   "3"),
    ("Meet Team",     "Day 2–4",   "4"),
    ("Training",      "Day 5–14",  "5"),
    ("First Project", "Day 15–29", "6"),
    ("30-Day",        "Day 30",    "7"),
    ("90-Day",        "Day 90",    "8"),
]

STAGE_INDEX = {
    "Intake & Paperwork": 0,
    "IT Setup": 1,
    "Orientation": 2,
    "Meet Your Team": 3,
    "Mandatory Training": 4,
    "First Project": 5,
    "30-Day Check-in": 6,
    "90-Day Review": 7,
}

# FIX: progress % now matches actual task counts; Jordan day corrected to 28
EMPLOYEES = {
    "Anushka Naidu": {
        "role": "UX Strategist", "day": 5, "stage": "Mandatory Training",
        "progress": 50,   # 3 done / 6 total = 50%
        "status": "At Risk", "region": "US", "bgv": "Pending",
        "done": ["Laptop Setup", "Slack Onboarding", "Met the Team"],
        "todo": ["AI Ethics Training", "Tool Onboarding", "First Project Kick-off"],
        "deadline_warning": "AI Ethics Training is due by Day 14 — you have 9 days left.",
    },
    "Marcus Lee": {
        "role": "Copywriter", "day": 12, "stage": "First Project",
        "progress": 67,   # 4 done / 6 total = 67%
        "status": "On Track", "region": "UK", "bgv": "Verified",
        "done": ["Laptop Setup", "Slack Onboarding", "Met the Team", "Harassment Training"],
        "todo": ["AI Ethics Training", "First Project Kick-off"],
        "deadline_warning": "AI Ethics Training is due by Day 14 — only 2 days left!",
    },
    "Priya Sharma": {
        "role": "Account Manager", "day": 8, "stage": "Mandatory Training",
        "progress": 71,   # 5 done / 7 total = 71%
        "status": "On Track", "region": "US", "bgv": "Verified",
        "done": ["Laptop Setup", "Slack Onboarding", "Met the Team", "Harassment Training", "Tool Onboarding"],
        "todo": ["AI Ethics Training", "First Project Kick-off"],
        "deadline_warning": "AI Ethics Training is due by Day 14 — you have 6 days left.",
    },
    "Jordan Kim": {
        "role": "Designer", "day": 31, "stage": "30-Day Check-in",  # Day 31 — overdue for 30-day check-in
        "progress": 88,   # 7 done / 8 total = 88%
        "status": "Overdue", "region": "US", "bgv": "Verified",
        "done": ["Laptop Setup", "Slack Onboarding", "Met the Team", "Harassment Training",
                 "Tool Onboarding", "AI Ethics Training", "First Project Kick-off"],
        "todo": ["30-Day Check-in Meeting"],
        "deadline_warning": "Your 30-Day Check-in is overdue. Book it with your manager immediately.",
    },
}

NUDGES = {
    "Anushka Naidu": "Your Friday 1-on-1 is coming up — tackle AI Ethics first, it unlocks your first project.",
    "Marcus Lee": "You're doing great! AI Ethics is the last training piece — knock it out today.",
    "Priya Sharma": "Nice progress! AI Ethics is next — should take about an hour.",
    "Jordan Kim": "One item left: your 30-Day Check-in. Book it with your manager today.",
}

STARTERS = {
    "Anushka Naidu": [
        "How many PTO days do I have?",
        "What trainings are pending for me?",
        "Explain the remote work policy.",
        "Who do I contact for payroll?",
    ],
    "Marcus Lee": [
        "How many PTO days do I have?",
        "What's left in my onboarding?",
        "When is my next 1-on-1?",
        "Who do I contact for IT support?",
    ],
    "Priya Sharma": [
        "How many PTO days do I have?",
        "What onboarding tasks are pending?",
        "Explain the remote work policy.",
        "What is the wellness stipend?",
    ],
    "Jordan Kim": [
        "What do I need to complete onboarding?",
        "How many PTO days do I have?",
        "Explain the remote work policy.",
        "Who do I contact for payroll?",
    ],
}

RECOMMENDATIONS = {
    "pto":      ["How do I submit a PTO request?", "Can I carry over unused PTO?", "What's the holiday calendar?"],
    "training": ["What happens if I miss the training deadline?", "How long does AI Ethics training take?", "Where do I access the training portal?"],
    "bgv":      ["Who processes my background check?", "What do I do if BGV is delayed?", "Can I start before BGV is complete?"],
    "payroll":  ["When is my first paycheck?", "How do I update my bank details?", "Who do I contact if my pay is wrong?"],
    "remote":   ["When can I start working remotely?", "How do I request remote work days?", "Are there remote work guidelines?"],
    "1-on-1":   ["What should I prepare for my 1-on-1?", "What if I need to reschedule my Friday meeting?", "How long are 1-on-1s?"],
    "wellness": ["How do I claim my wellness stipend?", "What counts as a wellness expense?", "When does the stipend reset?"],
    "tools":    ["How do I get access to Figma?", "What's Asana used for here?", "Who do I contact if I can't log in to Slack?"],
    "default":  ["What are my next onboarding steps?", "When is my next 1-on-1?", "How many PTO days do I have?"],
}

# FIX: source section names for inline display — keyed by topic
SOURCE_SECTIONS = {
    "pto":      "PTO Policy",
    "training": "Mandatory Training",
    "bgv":      "BGV",
    "payroll":  "Payroll",
    "remote":   "Remote Work",
    "1-on-1":   "1-on-1s",
    "wellness": "Wellness",
    "tools":    "Tools & Access",
    "default":  "Employee Handbook",
}

def get_topic(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["pto", "leave", "vacation", "time off", "days off"]): return "pto"
    if any(w in q for w in ["training", "ethics", "harassment", "course", "security"]): return "training"
    if any(w in q for w in ["bgv", "background", "verification"]): return "bgv"
    if any(w in q for w in ["payroll", "pay", "salary", "paycheck", "bank"]): return "payroll"
    if any(w in q for w in ["remote", "wfh", "work from home", "hybrid"]): return "remote"
    if any(w in q for w in ["1-on-1", "one on one", "manager", "friday", "meeting"]): return "1-on-1"
    if any(w in q for w in ["wellness", "stipend", "gym", "meditation", "health"]): return "wellness"
    if any(w in q for w in ["slack", "asana", "figma", "tool", "access", "it"]): return "tools"
    return "default"

def get_recommendations(query: str) -> list:
    return RECOMMENDATIONS[get_topic(query)]

def get_source_section(query: str, top_chunks: list = None) -> str:
    """Detect source section from retrieved chunk content first, fall back to query keyword."""
    if top_chunks:
        combined = " ".join(top_chunks).lower()
        # Check chunk content for section headers (more accurate than query keyword)
        if "pto" in combined or "paid time off" in combined or "vacation" in combined:
            return SOURCE_SECTIONS["pto"]
        if "mandatory training" in combined or "ai ethics" in combined or "harassment" in combined:
            return SOURCE_SECTIONS["training"]
        if "bgv" in combined or "background verification" in combined:
            return SOURCE_SECTIONS["bgv"]
        if "payroll" in combined or "direct deposit" in combined:
            return SOURCE_SECTIONS["payroll"]
        if "remote work" in combined or "work remotely" in combined:
            return SOURCE_SECTIONS["remote"]
        if "1-on-1" in combined or "friday" in combined or "weekly" in combined:
            return SOURCE_SECTIONS["1-on-1"]
        if "wellness" in combined or "stipend" in combined:
            return SOURCE_SECTIONS["wellness"]
        if "slack" in combined or "asana" in combined or "figma" in combined:
            return SOURCE_SECTIONS["tools"]
    # Fall back to query keyword if chunk scan inconclusive
    return SOURCE_SECTIONS[get_topic(query)]

BADGE = {
    "On Track": "background:#d1fae5; color:#065f46;",
    "At Risk":  "background:#fef3c7; color:#92400e;",
    "Overdue":  "background:#fee2e2; color:#991b1b;",
}

# ── RAG ───────────────────────────────────────────────────────────────────────

def cosine_similarity(query_vec, matrix):
    dots = matrix @ query_vec
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    return dots / (norms + 1e-10)

@st.cache_resource(show_spinner=False)
def build_vector_db(text_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text_data)
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    matrix = np.array(model.embed_documents(chunks))
    return chunks, matrix, model

def query_vector_db(chunks, matrix, model, query, n=3):
    qvec = np.array(model.embed_query(query))
    scores = cosine_similarity(qvec, matrix)
    top = np.argsort(scores)[::-1][:n]
    return [chunks[i] for i in top]

def now_str() -> str:
    """Return current time as e.g. '7:03 PM' — works on Mac, Linux, and Windows"""
    return datetime.now().strftime("%I:%M %p").lstrip("0")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Buddy — HR Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --bg: #111111;
    --panel: #171717;
    --panel-soft: #1c1c1c;
    --border: #2a2a2a;
    --text: #f3f4f6;
    --muted: #a3a3a3;
    --subtle: #737373;
    --accent: #6b9e78;
    --accent-soft: rgba(107,158,120,0.16);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container,
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        radial-gradient(circle at top left, rgba(107,158,120,0.10), transparent 30%),
        radial-gradient(circle at bottom right, rgba(230,190,120,0.05), transparent 28%),
        var(--bg) !important;
}

html, body, p, div, span, label,
[class*="css"], .stMarkdown, li {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text);
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden; height: 0; }

.block-container {
    padding: 2rem 2.5rem 7rem !important;
    max-width: 100% !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] > div { padding-top: 1.1rem !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSelectbox"] > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stRadio { margin-top: .55rem !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.875rem !important; }
[data-testid="stSidebar"] .stCheckbox label { font-size: 0.82rem !important; }
[data-testid="stSidebar"] input[type="checkbox"] { accent-color: var(--accent) !important; }

/* Buttons / chips */
.stButton > button {
    background-color: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.42rem 0.75rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background-color: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: #d8f3df !important;
}

/* Progress bar */
.stProgress > div > div > div > div { background-color: var(--accent) !important; }
.stProgress > div > div > div { background-color: #2a2a2a !important; }

/* Expander */
[data-testid="stExpander"] {
    background-color: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-top: 0.4rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: #b9e2c2 !important;
    font-weight: 500 !important;
}

/* Spinner */
[data-testid="stSpinner"] p {
    font-size: 0.85rem !important;
    color: var(--muted) !important;
}

/* Timeline */
.timeline-wrap {
    display: flex;
    align-items: flex-start;
    gap: 0;
    overflow-x: auto;
    padding-bottom: 0.15rem;
}
.timeline-step {
    flex: 1;
    min-width: 78px;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
}
.timeline-step:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 14px;
    left: calc(50% + 14px);
    width: calc(100% - 28px);
    height: 2px;
    z-index: 0;
}
.timeline-step.done::after { background: var(--accent); }
.timeline-step.active::after,
.timeline-step.upcoming::after { background: #333333; }

.timeline-dot {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.74rem;
    position: relative; z-index: 1; flex-shrink: 0;
}
.timeline-dot.done { background:var(--accent); border:2px solid var(--accent); color:white; font-size:0.65rem; }
.timeline-dot.active { background:#111111; border:2.5px solid var(--accent); color:#d8f3df; font-size:0.78rem; box-shadow:0 0 0 4px rgba(107,158,120,0.14); }
.timeline-dot.upcoming { background:#171717; border:2px solid #3a3a3a; color:#a3a3a3; font-size:0.74rem; }

.timeline-label {
    margin-top: 0.38rem; font-size: 0.63rem; font-weight: 600;
    text-align: center; line-height: 1.25;
}
.timeline-label.done { color: #9dc9a6; }
.timeline-label.active { color: var(--text); }
.timeline-label.upcoming { color: #6f6f6f; }

.timeline-sub { font-size:0.58rem; color:#666; text-align:center; margin-top:0.08rem; }
.timeline-sub.active { color:#9a9a9a; }

/* Native Streamlit chat — keep it simple and aligned */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.35rem 0 !important;
}
[data-testid="stChatMessageContent"] {
    max-width: 760px !important;
}
[data-testid="stChatInput"] {
    background: rgba(17,17,17,0.96) !important;
    border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInput"] textarea {
    background-color: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #777 !important; }

/* Pill label */
.pill-label {
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #777;
    font-weight: 700;
    margin-bottom: 0.35rem;
    margin-top: 0.9rem;
    display: block;
}

/* Keep Markdown text readable */
h1, h2, h3, h4 { color: var(--text) !important; }
small, .caption { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

# Master task registry — persists ALL employees' checkbox states across switches
if "master_tasks" not in st.session_state:
    st.session_state.master_tasks = {
        name: {task: False for task in data["todo"]}
        for name, data in EMPLOYEES.items()
    }

# Per-employee chat history — each employee keeps their own conversation
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {name: [] for name in EMPLOYEES}
if "all_source_chunks" not in st.session_state:
    st.session_state.all_source_chunks = {name: {} for name in EMPLOYEES}
if "all_source_sections" not in st.session_state:
    st.session_state.all_source_sections = {name: {} for name in EMPLOYEES}
if "all_timestamps" not in st.session_state:
    st.session_state.all_timestamps = {name: {} for name in EMPLOYEES}
# Per-employee recommendations and last query — prevents cross-employee bleed
if "all_recommendations" not in st.session_state:
    st.session_state.all_recommendations = {name: [] for name in EMPLOYEES}
if "all_last_query" not in st.session_state:
    st.session_state.all_last_query = {name: "" for name in EMPLOYEES}

for key, val in [
    ("_active_employee", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Secrets ───────────────────────────────────────────────────────────────────

if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]

# ── Load RAG ──────────────────────────────────────────────────────────────────

with st.spinner("Loading knowledge base…"):
    chunks, emb_matrix, emb_model = build_vector_db(hr_docs)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:0.2rem 0 1.2rem;">
        <span style="font-size:1.4rem;">🤝</span>
        <span style="font-family:'DM Serif Display',serif; font-size:1.15rem; margin-left:0.4rem;">Buddy</span>
        <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#8a8a8a; margin-top:0.1rem;">HR Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Static, believable metadata
    st.markdown("""
    <div style="background:#171717; border:1px solid #2a2a2a; border-radius:8px; padding:0.6rem 0.75rem; margin-bottom:1rem;">
        <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.3rem;">
            <span style="width:7px; height:7px; background:#22c55e; border-radius:50%; display:inline-block; flex-shrink:0;"></span>
            <span style="font-size:0.75rem; font-weight:600; color:#f3f4f6;">All systems operational</span>
        </div>
        <div style="font-size:0.7rem; color:#a8a8a8; line-height:1.7;">
            HR Handbook v2.1 · 12 policies indexed<br>
            Knowledge base updated May 2026<br>
            Responses grounded in verified documents
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#8a8a8a; font-weight:600; margin-bottom:0.35rem;'>Viewing as</div>", unsafe_allow_html=True)
    selected_name = st.selectbox("Employee", list(EMPLOYEES.keys()), label_visibility="collapsed")
    emp = EMPLOYEES[selected_name]

    if st.session_state["_active_employee"] != selected_name:
        st.session_state["_active_employee"] = selected_name

    badge_css = BADGE.get(emp["status"], "background:#eee; color:#333;")
    st.markdown(f"""
    <div style="margin:0.8rem 0 0.9rem;">
        <div style="font-weight:600; font-size:0.98rem;">{selected_name}</div>
        <div style="font-size:0.8rem; color:#a3a3a3; margin-top:0.12rem;">{emp['role']} · Day {emp['day']} · {emp['region']}</div>
        <div style="margin-top:0.45rem;">
            <span style="display:inline-block; padding:2px 9px; border-radius:100px; font-size:0.68rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; {badge_css}">{emp['status']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    checked_count = sum(1 for v in st.session_state.master_tasks[selected_name].values() if v)
    total_tasks = len(emp["done"]) + len(emp["todo"])
    live_done = len(emp["done"]) + checked_count
    live_progress = int((live_done / total_tasks) * 100) if total_tasks > 0 else emp["progress"]

    st.markdown(f"""<div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#8a8a8a; font-weight:600; margin-bottom:0.25rem;">
        Onboarding <span style="float:right; text-transform:none; letter-spacing:0; font-size:0.78rem; color:#f3f4f6; font-weight:600;">{live_progress}%</span>
    </div>""", unsafe_allow_html=True)
    st.progress(live_progress / 100)

    st.markdown(f"""
    <div style="display:flex; gap:0.5rem; margin:0.75rem 0 0.9rem;">
        <div style="flex:1; background:#171717; border:1px solid #2a2a2a; border-radius:8px; padding:0.5rem 0.65rem;">
            <div style="font-size:0.62rem; text-transform:uppercase; letter-spacing:0.08em; color:#777; font-weight:600;">Stage</div>
            <div style="font-size:0.78rem; font-weight:600; margin-top:0.08rem;">{emp['stage']}</div>
        </div>
        <div style="flex:1; background:#171717; border:1px solid #2a2a2a; border-radius:8px; padding:0.5rem 0.65rem;">
            <div style="font-size:0.62rem; text-transform:uppercase; letter-spacing:0.08em; color:#777; font-weight:600;">BGV</div>
            <div style="font-size:0.78rem; font-weight:600; margin-top:0.08rem; color:{'#2d7a4f' if emp['bgv']=='Verified' else '#b45309'};">{emp['bgv']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if emp["done"]:
        st.markdown("<div style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#8a8a8a; font-weight:600; margin-bottom:0.25rem;'>✓ Completed</div>", unsafe_allow_html=True)
        for item in emp["done"]:
            st.markdown(f"<div style='font-size:0.8rem; color:#777; padding:2px 0; text-decoration:line-through;'>{item}</div>", unsafe_allow_html=True)

    if emp["todo"]:
        st.markdown("<div style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#8a8a8a; font-weight:600; margin:0.65rem 0 0.3rem;'>◦ To do</div>", unsafe_allow_html=True)
        all_done_before = all(st.session_state.master_tasks[selected_name].get(t, False) for t in emp["todo"])
        for task in emp["todo"]:
            checked = st.checkbox(task, value=st.session_state.master_tasks[selected_name].get(task, False), key=f"cb_{selected_name}_{task}")
            st.session_state.master_tasks[selected_name][task] = checked
        all_done_after = all(st.session_state.master_tasks[selected_name].get(t, False) for t in emp["todo"])
        if all_done_after and not all_done_before:
            st.balloons()
            st.success("All tasks complete! 🎉")

    st.markdown("<hr style='border:none; border-top:1px solid #2a2a2a; margin:1rem 0 0.6rem;'>", unsafe_allow_html=True)
    view_mode = st.radio("View", ["Employee Portal", "HR Manager View"], label_visibility="collapsed")
    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.all_chats[selected_name] = []
        st.session_state.all_source_chunks[selected_name] = {}
        st.session_state.all_source_sections[selected_name] = {}
        st.session_state.all_timestamps[selected_name] = {}
        st.session_state.all_recommendations[selected_name] = []
        st.session_state.all_last_query[selected_name] = ""
        st.rerun()

# ── Manager View ──────────────────────────────────────────────────────────────

if view_mode == "HR Manager View":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:'DM Serif Display',serif; font-size:1.9rem; line-height:1.2;">Manager Dashboard</div>
        <div style="font-size:0.83rem; color:#a8a8a8; margin-top:0.25rem;">Onboarding overview · All employees</div>
    </div>
    """, unsafe_allow_html=True)

    def get_live_progress(name, e):
        tasks = st.session_state.master_tasks.get(name, {})
        live_done_mgr = len(e["done"]) + sum(1 for v in tasks.values() if v)
        total_tasks_mgr = len(e["done"]) + len(e["todo"])
        return int((live_done_mgr / total_tasks_mgr) * 100) if total_tasks_mgr else e["progress"]

    total    = len(EMPLOYEES)
    avg_prog = int(sum(get_live_progress(n, e) for n, e in EMPLOYEES.items()) / total)
    at_risk  = sum(1 for e in EMPLOYEES.values() if e["status"] == "At Risk")
    overdue  = sum(1 for e in EMPLOYEES.values() if e["status"] == "Overdue")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, sub in [
        (c1, total,         "Total",       "employees"),
        (c2, f"{avg_prog}%","Avg Progress", "across all"),
        (c3, at_risk,       "At Risk",      "need attention"),
        (c4, overdue,       "Overdue",      "action required"),
    ]:
        col.markdown(f"""
        <div style="background:#171717; border:1px solid #2a2a2a; border-radius:12px; padding:1.2rem; text-align:center; box-shadow:none;">
            <div style="font-family:'DM Serif Display',serif; font-size:2rem; line-height:1;">{val}</div>
            <div style="font-size:0.75rem; font-weight:600; margin-top:0.3rem;">{label}</div>
            <div style="font-size:0.7rem; color:#777;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#8a8a8a; font-weight:600; margin-bottom:0.7rem;'>Employees</div>", unsafe_allow_html=True)

    for name, e in EMPLOYEES.items():
        bc = BADGE.get(e["status"], "background:#eee;")
        display_progress = get_live_progress(name, e)
        with st.expander(f"{name}  ·  {e['role']}  ·  Day {e['day']}"):
            col1, col2, col3 = st.columns([5, 2, 1])
            with col1:
                st.progress(display_progress / 100, text=f"{display_progress}% complete")
                if e["todo"]:
                    st.markdown(f"<div style='font-size:0.78rem; color:#a8a8a8; margin-top:0.15rem;'>Pending: {', '.join(e['todo'])}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span style='display:inline-block; padding:2px 9px; border-radius:100px; font-size:0.68rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; {bc}'>{e['status']}</span>", unsafe_allow_html=True)
            with col3:
                if e["status"] in ["At Risk", "Overdue"]:
                    if st.button("Nudge", key=f"nudge_{name}"):
                        st.toast(f"Nudge sent to {name}! 👋")

# ── Employee Portal ───────────────────────────────────────────────────────────

else:
    first_name  = selected_name.split()[0]
    status_icon = {"At Risk": "⚠️", "Overdue": "🔴", "On Track": "✅"}.get(emp["status"], "")
    nudge       = NUDGES.get(selected_name, "You're doing great — keep it up!")
    starters    = STARTERS.get(selected_name, STARTERS["Anushka Naidu"])

    # ── Hero — FIX: more vertical breathing room ──────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:1.8rem; padding-top:0.5rem;">
        <div style="font-family:'DM Serif Display',serif; font-size:2.4rem; line-height:1.1; color:#f3f4f6;">Buddy</div>
        <div style="font-size:1.05rem; color:#c9c9c9; font-weight:500; margin-top:0.35rem;">Your AI workplace assistant</div>
        <div style="font-size:0.85rem; color:#8a8a8a; margin-top:0.3rem; max-width:500px; line-height:1.55;">
            Get instant answers on onboarding, HR policies, compliance, and workplace tools.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Onboarding Timeline ───────────────────────────────────────────────────
    current_stage_idx = STAGE_INDEX.get(emp["stage"], 0)
    dots_html = ""
    for i, (label, day_range, icon) in enumerate(ONBOARDING_STAGES):
        if i < current_stage_idx:
            state, dot_content = "done", "✓"
        elif i == current_stage_idx:
            state, dot_content = "active", icon
        else:
            state, dot_content = "upcoming", icon

        dots_html += f"""
        <div class="timeline-step {state}">
            <div class="timeline-dot {state}">{dot_content}</div>
            <div class="timeline-label {state}">{label}</div>
            <div class="timeline-sub {state}">{day_range}</div>
        </div>"""

    st.markdown(f"""
    <div style="background:#171717; border:1px solid #2a2a2a; border-radius:14px; padding:1rem 1.4rem 1.1rem; margin-bottom:1.2rem; box-shadow:0 12px 28px rgba(0,0,0,0.25);">
        <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#8a8a8a; font-weight:600; margin-bottom:0.75rem;">
            Onboarding Journey &nbsp;·&nbsp; <span style="color:#6b9e78; text-transform:none; letter-spacing:0;">{emp['stage']}</span>
        </div>
        <div class="timeline-wrap">{dots_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Greeting card ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#171717; border:1px solid #2a2a2a; border-radius:14px; padding:1.2rem 1.5rem; margin-bottom:1.2rem; box-shadow:0 12px 28px rgba(0,0,0,0.25);">
        <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; line-height:1.25;">
            Hey, {first_name}. {status_icon}
        </div>
        <div style="font-size:0.8rem; color:#a8a8a8; margin-top:0.25rem;">
            Day {emp['day']} &nbsp;·&nbsp; {emp['stage']} &nbsp;·&nbsp; {live_progress}% complete
        </div>
        <div style="margin-top:0.7rem; border-left:3px solid #6b9e78; padding-left:0.8rem; font-size:0.87rem; color:#d7d7d7; font-style:italic; line-height:1.5;">
            {nudge}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Deadline warning ──────────────────────────────────────────────────────
    if emp.get("deadline_warning"):
        warn_color  = "#fca5a5" if emp["status"] == "Overdue" else "#fbbf24"
        warn_bg     = "rgba(248,113,113,0.08)" if emp["status"] == "Overdue" else "rgba(251,191,36,0.08)"
        warn_border = "rgba(248,113,113,0.35)" if emp["status"] == "Overdue" else "rgba(251,191,36,0.35)"
        st.markdown(f"""
        <div style="background:{warn_bg}; border:1px solid {warn_border}; border-radius:10px; padding:0.75rem 1rem; margin-bottom:1.1rem; display:flex; align-items:flex-start; gap:0.6rem;">
            <span style="font-size:1rem; flex-shrink:0;">{'🔴' if emp['status']=='Overdue' else '⚠️'}</span>
            <div>
                <div style="font-size:0.82rem; font-weight:600; color:{warn_color};">Action required</div>
                <div style="font-size:0.82rem; color:{warn_color}; margin-top:0.1rem;">{emp['deadline_warning']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Convenience aliases for current employee's chat store
    chat_history    = st.session_state.all_chats[selected_name]
    source_chunks   = st.session_state.all_source_chunks[selected_name]
    source_sections = st.session_state.all_source_sections[selected_name]
    msg_timestamps  = st.session_state.all_timestamps[selected_name]
    # Per-employee query and recommendations
    last_query      = st.session_state.all_last_query[selected_name]
    recommendations = st.session_state.all_recommendations[selected_name]

    # ── Preloaded welcome message ─────────────────────────────────────────────
    if len(chat_history) == 0:
        pending_str = ", ".join(emp["todo"]) if emp["todo"] else "nothing — you're all caught up!"
        welcome = (
            f"Hi {first_name} 👋  I can help with onboarding, HR policies, PTO, "
            f"compliance training, and your internal tools. "
            f"You still need to complete: **{pending_str}**. What can I help you with today?"
        )
        chat_history.append({"role": "assistant", "content": welcome})
        msg_timestamps[0] = now_str()

    # ── Chat history ──────────────────────────────────────────────────────────
    for i, msg in enumerate(chat_history):
        ts = msg_timestamps.get(i, "")
        role = "user" if msg["role"] == "user" else "assistant"

        with st.chat_message(role):
            st.caption(("You" if role == "user" else "Buddy") + (f" · {ts}" if ts else ""))
            st.markdown(msg["content"])

            if role == "assistant":
                source_section = source_sections.get(i, "")
                if i > 0 and source_section:
                    st.caption(f"Source: HR Handbook v2.1 → {source_section} · Confidence: High")

                if i > 0 and i in source_chunks:
                    with st.expander("View full source excerpt"):
                        for chunk in source_chunks[i]:
                            st.markdown(chunk)

    # ── AI Recommendations — FIX: "Suggested follow-ups" label, pill styling ─
    if recommendations:
        st.markdown("<span class='pill-label'>Suggested follow-ups</span>", unsafe_allow_html=True)
        rec_cols = st.columns(len(recommendations))
        for i, (col, rq) in enumerate(zip(rec_cols, recommendations)):
            if col.button(rq, key=f"rec_{selected_name}_{i}_{rq[:15]}", use_container_width=True):
                st.session_state["_chip_q"] = rq
                st.session_state.all_last_query[selected_name] = ""  # reset so same Q can re-fire
                st.rerun()

    # ── Chat input — auto-clears after submit ────────────────────────────────
    query = st.chat_input("Ask anything about HR, policies, onboarding…")

    # ── Conversation starters ─────────────────────────────────────────────────
    if starters:
        st.markdown("<span class='pill-label' style='margin-top:0.4rem;'>Try asking</span>", unsafe_allow_html=True)
        starter_cols = st.columns(len(starters))
        for i, (col, q) in enumerate(zip(starter_cols, starters)):
            if col.button(q, key=f"starter_{selected_name}_{i}", use_container_width=True):
                st.session_state["_chip_q"] = q
                st.session_state.all_last_query[selected_name] = ""  # reset so same Q can re-fire
                st.rerun()

    chip_q  = st.session_state.pop("_chip_q", None)
    active_q = chip_q or query

    # ── Process query ─────────────────────────────────────────────────────────
    if active_q and active_q != last_query:
        st.session_state.all_last_query[selected_name] = active_q

        with st.spinner("Buddy is thinking…"):
            try:
                top_chunks = query_vector_db(chunks, emb_matrix, emb_model, active_q)
                context    = "\n".join(top_chunks)
                source_sec = get_source_section(active_q, top_chunks)

                emp_ctx = f"""Employee: {selected_name} ({emp['role']})
Day: {emp['day']} | Stage: {emp['stage']} | Status: {emp['status']}
Completed: {', '.join(emp['done']) if emp['done'] else 'None yet'}
Still to complete: {', '.join(emp['todo']) if emp['todo'] else 'All done!'}
BGV: {emp['bgv']} | Region: {emp['region']}
Upcoming deadline: {emp.get('deadline_warning', 'None')}"""

                prompt = f"""You are Buddy, a warm HR assistant. You know who you're talking to — use their first name ({first_name}) and their specific situation.
Answer accurately from the HR handbook below. Be friendly and concise (2–4 sentences max).
After answering, if the employee has any pending tasks, naturally remind them of the most urgent one in a single sentence starting with "Also —" or "Reminder:".
If you don't know the answer, say so honestly and suggest contacting HR.

Employee context:
{emp_ctx}

HR Handbook:
{context}

Question: {active_q}

Buddy:"""

                client = Groq(api_key=api_key)
                resp   = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = resp.choices[0].message.content
                ts     = now_str()

                # Append to this employee's chat store
                user_idx  = len(chat_history)
                chat_history.append({"role": "user", "content": active_q})
                msg_timestamps[user_idx] = ts

                buddy_idx = len(chat_history)
                chat_history.append({"role": "assistant", "content": answer})
                msg_timestamps[buddy_idx]  = ts
                source_chunks[buddy_idx]   = top_chunks
                source_sections[buddy_idx] = source_sec
                st.session_state.all_recommendations[selected_name] = get_recommendations(active_q)

            except Exception as e:
                st.error(f"Buddy couldn't respond right now — please try again. ({type(e).__name__})")
                # Reset last_query so user can retry the same question
                st.session_state.all_last_query[selected_name] = ""

        st.rerun()
