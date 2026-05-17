import numpy as np
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# -----------------------------
# Data
# -----------------------------
hr_docs = """
EMPLOYEE HANDBOOK

PTO Policy: Employees receive 20 days of Paid Time Off per year. Requests must be submitted via the HR Portal.

BGV: Background verification must be completed before Day 1. US employees use HireRight, UK employees use Credence.

Tools: Slack, Asana, Figma, Gemini, Google Workspace.

IT Support: Email the IT team for any hardware or access issues.

Mandatory Training: All employees must complete Harassment Policy, AI Ethics, and Tool Onboarding within first 14 days.

1-on-1s: Weekly every Friday with your manager.

Onboarding Stages: Intake, IT Setup, Orientation, Meet Your Team, Training, First Project, 30-Day Check-in, 90-Day Review.
"""

employees_stats = [
    {"name": "Anushka Naidu", "role": "UX Strategist", "progress": 20, "status": "At Risk"},
    {"name": "Marcus Lee", "role": "Copywriter", "progress": 60, "status": "On Track"},
    {"name": "Priya Sharma", "role": "Account Manager", "progress": 40, "status": "On Track"},
    {"name": "Jordan Kim", "role": "Designer", "progress": 85, "status": "Overdue"},
]

task_list = [
    "Complete AI Ethics Training",
    "Finish Tool Onboarding",
    "Prepare for first 1-on-1",
]

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Buddy", page_icon="🤝", layout="wide")

# -----------------------------
# CSS polish
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(107,161,118,0.20), transparent 28%),
        radial-gradient(circle at bottom right, rgba(96,165,250,0.12), transparent 30%),
        linear-gradient(135deg, #0f1117 0%, #12151f 45%, #171a24 100%) !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 900px;
    padding-top: 4rem;
}

/* cleaner sidebar */
[data-testid="stSidebar"] {
    background: rgba(24, 26, 36, 0.96) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #f4f4f5;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem !important;
}

.sidebar-logo {
    padding: 0.3rem 0 1rem 0;
}

.sidebar-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.1rem;
}

.sidebar-subtitle {
    color: #a1a1aa;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.sidebar-card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 0.95rem;
    margin: 1rem 0;
}

.sidebar-name {
    color: white;
    font-weight: 700;
    font-size: 0.98rem;
}

.sidebar-meta {
    color: #a1a1aa;
    font-size: 0.82rem;
    margin-top: 0.15rem;
}

.sidebar-small-label {
    color: #a1a1aa;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.35rem;
}

.pending-card {
    background: rgba(107,161,118,0.13);
    border: 1px solid rgba(107,161,118,0.30);
    border-radius: 16px;
    padding: 0.85rem 0.95rem;
    margin-top: 0.8rem;
}

.pending-title {
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.45rem;
}

.pending-item {
    color: #d4d4d8;
    font-size: 0.86rem;
    margin: 0.28rem 0;
}

h1 {
    color: white !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}

h2, h3 {
    color: white !important;
}

p, label, span {
    color: #e5e7eb;
}

.stCaptionContainer {
    color: #a1a1aa !important;
}

.hero-card {
    padding: 1.5rem 1.7rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.09);
    margin-bottom: 1.2rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.18);
}

.task-card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    margin-bottom: 1rem;
}

.task-title {
    color: white;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.task-line {
    color: #d4d4d8;
    font-size: 0.92rem;
    margin: 0.25rem 0;
}

.response-card {
    background: #ffffff;
    color: #111827;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    margin-top: 1rem;
    line-height: 1.6;
    box-shadow: 0 14px 35px rgba(0,0,0,0.22);
}

.response-card * {
    color: #111827 !important;
}

.source-card {
    padding: 0.75rem 0.9rem;
    border-left: 4px solid #6ba176;
    background: rgba(107,161,118,0.12);
    border-radius: 10px;
    margin-top: 0.8rem;
    font-size: 0.85rem;
}

.stTextInput input {
    background: rgba(40,42,54,0.95) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
}

.stTextInput input:focus {
    border-color: #6ba176 !important;
    box-shadow: 0 0 0 3px rgba(107,161,118,0.18) !important;
}

.stButton > button {
    border-radius: 999px !important;
    background: rgba(255,255,255,0.07) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    transition: 0.15s ease-in-out;
}

.stButton > button:hover {
    border-color: #6ba176 !important;
    background: rgba(107,161,118,0.20) !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.055);
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.09);
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 14px !important;
}

/* make default info/warning boxes less loud in sidebar */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# RAG functions
# -----------------------------
def cosine_similarity(query_vec, matrix):
    dots = matrix @ query_vec
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    return dots / (norms + 1e-10)

@st.cache_resource(show_spinner=False)
def build_vector_db(text_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text_data)
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedded_chunks = np.array(embeddings_model.embed_documents(chunks))
    return chunks, embedded_chunks, embeddings_model

def query_vector_db(chunks, embeddings_matrix, embeddings_model, query, n_results=3):
    query_vec = np.array(embeddings_model.embed_query(query))
    scores = cosine_similarity(query_vec, embeddings_matrix)
    top_indices = np.argsort(scores)[::-1][:n_results]
    return [chunks[i] for i in top_indices]

def get_followups(query):
    q = query.lower()
    if "pto" in q or "leave" in q:
        return ["How do I submit PTO?", "Can unused PTO carry over?", "Where is the HR Portal?"]
    if "training" in q or "ethics" in q:
        return ["What happens if I miss training?", "How long do I have to complete it?", "Where do I access training?"]
    if "tool" in q or "slack" in q or "figma" in q:
        return ["Who gives me tool access?", "What is Asana used for?", "Who do I contact for IT issues?"]
    if "1-on-1" in q or "manager" in q:
        return ["What should I prepare?", "How often are 1-on-1s?", "What should I ask my manager?"]
    return ["What are my next onboarding steps?", "What trainings are pending?", "Who do I contact for IT help?"]

# -----------------------------
# Secrets
# -----------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("Add GROQ_API_KEY to Streamlit Secrets!")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]

# -----------------------------
# Session state
# -----------------------------
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-title">🤝 Buddy</div>
        <div class="sidebar-subtitle">HR Assistant</div>
    </div>

    <div class="sidebar-card">
        <div class="sidebar-name">Anushka Naidu</div>
        <div class="sidebar-meta">Data Science Student · Day 1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-small-label">Onboarding progress</div>', unsafe_allow_html=True)
    st.progress(20)

    st.markdown("""
    <div class="pending-card">
        <div class="pending-title">Pending tasks</div>
        <div class="pending-item">• AI Ethics Training</div>
        <div class="pending-item">• Tool Onboarding</div>
        <div class="pending-item">• First Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sidebar-small-label">View</div>', unsafe_allow_html=True)
    view_mode = st.radio(
        "View",
        ["Employee Portal", "HR Manager View"],
        label_visibility="collapsed"
    )

# -----------------------------
# Employee Portal
# -----------------------------
st.markdown("""
<div class="hero-card">
    <h1>🤝 Buddy</h1>
    <p style="font-size:1.05rem; font-weight:600; margin-bottom:0.25rem;">Your AI workplace assistant</p>
    <p style="color:#a1a1aa;">Ask me anything about HR policies, onboarding, tools, training, or PTO.</p>
</div>
""", unsafe_allow_html=True)

# Today's priorities
st.markdown("""
<div class="task-card">
    <div class="task-title">Today's priorities</div>
    <div class="task-line">☐ Complete AI Ethics Training</div>
    <div class="task-line">☐ Finish Tool Onboarding</div>
    <div class="task-line">☐ Prepare for first 1-on-1</div>
</div>
""", unsafe_allow_html=True)

if "chunks" not in st.session_state:
    with st.spinner("Buddy is loading the HR handbook..."):
        st.session_state.chunks, st.session_state.embeddings_matrix, st.session_state.embeddings_model = build_vector_db(hr_docs)

starters = [
    "How many PTO days do I have?",
    "What trainings are pending?",
    "What tools do I need?",
    "When are 1-on-1s?",
]

st.caption("Try asking:")
cols = st.columns(4)
for i, question in enumerate(starters):
    if cols[i].button(question, use_container_width=True, key=f"starter_{i}"):
        st.session_state.selected_question = question

# Quick actions
st.caption("Quick actions:")
action_cols = st.columns(4)
for i, action in enumerate(quick_actions):
    if action_cols[i].button(action, use_container_width=True, key=f"action_{i}"):
        if action == "Request PTO":
            st.session_state.selected_question = "How do I request PTO?"
        elif action == "Contact IT":
            st.session_state.selected_question = "Who do I contact for IT support?"
        elif action == "View onboarding checklist":
            st.session_state.selected_question = "What are my onboarding steps?"
        else:
            st.session_state.selected_question = "What should I prepare for my first 1-on-1?"

query = st.text_input(
    "Ask Buddy something:",
    value=st.session_state.selected_question,
    placeholder="Example: What trainings do I need to complete?"
)

if query:
    with st.spinner("Buddy is reviewing HR documents..."):
        try:
            top_chunks = query_vector_db(
                st.session_state.chunks,
                st.session_state.embeddings_matrix,
                st.session_state.embeddings_model,
                query,
            )
            context = "\n".join(top_chunks)

            client = Groq(api_key=api_key)
            prompt = f"""You are Buddy, a warm and encouraging HR and compliance assistant.
Use the HR context below to answer the employee question accurately.
If you do not know the answer, say so honestly and suggest they reach out to their HR team.
Keep the response concise, professional, warm, and direct. Avoid overly long supportive language. Maximum 3 short paragraphs.

HR Context:
{context}

Employee Question: {query}

Buddy Response:"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.choices[0].message.content
            st.session_state.last_answer = answer
            st.session_state.last_chunks = top_chunks
            st.session_state.last_followups = get_followups(query)

        except Exception as e:
            st.error(f"Buddy could not respond right now. Please try again. ({type(e).__name__})")

if st.session_state.last_answer:
    st.markdown("### Buddy says:")
    st.markdown(f'<div class="response-card">{st.session_state.last_answer}</div>', unsafe_allow_html=True)

    with st.expander("View source used"):
        if st.session_state.last_chunks:
            st.markdown(f'<div class="source-card">{st.session_state.last_chunks[0]}</div>', unsafe_allow_html=True)

if st.session_state.last_followups:
    st.caption("Suggested follow-ups:")
    follow_cols = st.columns(3)
    for i, followup in enumerate(st.session_state.last_followups):
        if follow_cols[i].button(followup, use_container_width=True, key=f"followup_{i}_{followup[:8]}"):
            st.session_state.selected_question = followup
            st.rerun()
