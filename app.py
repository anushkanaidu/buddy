import html
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

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Buddy", page_icon="🤝", layout="wide")

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(107,161,118,0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(96,165,250,0.11), transparent 30%),
        linear-gradient(135deg, #0f1117 0%, #12151f 45%, #171a24 100%) !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 880px;
    padding-top: 4rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(24, 26, 36, 0.96) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #f4f4f5;
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

.progress-text {
    color: #d4d4d8;
    font-size: 0.82rem;
    margin-top: 0.35rem;
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

/* Main */
h1 {
    color: white !important;
    font-size: 2.45rem !important;
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
    padding: 1.35rem 1.55rem;
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 1.05rem;
    box-shadow: 0 24px 60px rgba(0,0,0,0.22);
    backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -60px;
    top: -70px;
    background: radial-gradient(circle, rgba(107,161,118,0.25), transparent 65%);
}

.center-hero {
    text-align: left;
}

.hero-title {
    color: white;
    font-size: 2.05rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    margin-bottom: 0.45rem;
    position: relative;
}

.hero-subtitle {
    color: #f4f4f5;
    font-size: 1rem;
    font-weight: 650;
    margin-bottom: 0.28rem;
    position: relative;
}

.hero-copy {
    color: #a1a1aa;
    font-size: 0.95rem;
}

.focus-card {
    padding: 0.95rem 1rem;
    border-radius: 18px;
    background: rgba(107,161,118,0.085);
    border: 1px solid rgba(107,161,118,0.20);
    margin-bottom: 1rem;
    backdrop-filter: blur(14px);
}

.status-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
    position: relative;
}

.status-pill {
    background: rgba(107,161,118,0.14);
    border: 1px solid rgba(107,161,118,0.28);
    color: #dff5e5;
    padding: 0.28rem 0.62rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.focus-pills {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}

.focus-pill {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.10);
    color: #f4f4f5;
    border-radius: 999px;
    padding: 0.35rem 0.65rem;
    font-size: 0.84rem;
}

.focus-title {
    color: white;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
}

.focus-line {
    color: #f4f4f5;
    font-size: 0.88rem;
}

.response-card {
    background: rgba(248,250,252,0.95);
    color: #111827;
    padding: 0.9rem 1rem;
    border-radius: 16px;
    margin-top: 1.1rem;
    line-height: 1.55;
    box-shadow: 0 14px 35px rgba(0,0,0,0.22);
    max-width: 92%;
}

.source-line {
    color: #a1a1aa;
    font-size: 0.82rem;
    margin-top: 0.55rem;
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
    min-height: 2.45rem;
    font-weight: 600 !important;
}

.stButton > button:hover {
    border-color: #6ba176 !important;
    background: rgba(107,161,118,0.20) !important;
    transform: translateY(-1px);
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text_data)
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    matrix = np.array(model.embed_documents(chunks))
    return chunks, matrix, model

def query_vector_db(chunks, matrix, model, query, n_results=3):
    query_vec = np.array(model.embed_query(query))
    scores = cosine_similarity(query_vec, matrix)
    top_indices = np.argsort(scores)[::-1][:n_results]
    return [chunks[i] for i in top_indices]

def source_label(query, chunks):
    """Choose a clean source label. Query intent gets priority so labels don't get confused by mixed retrieved chunks."""
    q = query.lower()
    chunk_text = " ".join(chunks).lower()

    if any(word in q for word in ["1-on-1", "one on one", "manager", "friday", "meeting"]):
        return "1-on-1s section"
    if any(word in q for word in ["pto", "paid time off", "leave", "vacation"]):
        return "PTO Policy section"
    if any(word in q for word in ["training", "ethics", "harassment", "tool onboarding"]):
        return "Mandatory Training section"
    if any(word in q for word in ["slack", "asana", "figma", "gemini", "workspace", "tool"]):
        return "Tools section"
    if any(word in q for word in ["it support", "hardware", "access"]):
        return "IT Support section"
    if any(word in q for word in ["bgv", "background", "verification"]):
        return "Background Verification section"

    # fallback to retrieved context only if the query is broad
    if "1-on-1" in chunk_text or "friday" in chunk_text:
        return "1-on-1s section"
    if "pto" in chunk_text or "paid time off" in chunk_text:
        return "PTO Policy section"
    if "mandatory training" in chunk_text or "ai ethics" in chunk_text:
        return "Mandatory Training section"
    if "slack" in chunk_text or "asana" in chunk_text or "figma" in chunk_text:
        return "Tools section"
    if "bgv" in chunk_text or "background verification" in chunk_text:
        return "Background Verification section"

    return "Onboarding Policy section"

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

if "last_source" not in st.session_state:
    st.session_state.last_source = ""

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">🤝 Buddy</div>
    <div class="sidebar-subtitle">HR Assistant</div>

    <div class="sidebar-card">
        <div class="sidebar-name">Anushka Naidu</div>
        <div class="sidebar-meta">Data Science Student · Day 1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-small-label">Onboarding progress</div>', unsafe_allow_html=True)
    st.progress(20)
    st.markdown('<div class="progress-text">20% complete</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pending-card">
        <div class="pending-title">Pending tasks</div>
        <div class="pending-item">AI Ethics Training</div>
        <div class="pending-item">Tool Onboarding</div>
        <div class="pending-item">First Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Built with RAG + Groq")

# -----------------------------
# Main app
# -----------------------------
st.markdown("""
<div class="hero-card center-hero">
    <div class="hero-title">Good to see you, Anushka 👋</div>
    <div class="hero-subtitle">Buddy is ready to help with onboarding and HR questions.</div>
    <div class="hero-copy">Ask about policies, tools, training, PTO, or your next onboarding step.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="focus-card">
    <div class="focus-title">Current onboarding focus</div>
    <div class="focus-pills">
        <div class="focus-pill">AI Ethics Training</div>
        <div class="focus-pill">Tool Onboarding</div>
        <div class="focus-pill">First Project</div>
    </div>
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
cols = st.columns(len(starters))
for i, question in enumerate(starters):
    if cols[i].button(question, use_container_width=True, key=f"starter_{i}"):
        st.session_state.selected_question = question
        st.rerun()

query = st.text_input(
    "Ask Buddy",
    value=st.session_state.selected_question,
    placeholder="Ask about PTO, onboarding, training..."
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
            prompt = f"""You are Buddy, a warm HR and onboarding assistant helping Anushka.
Use the HR context below to answer accurately.
Start responses naturally with "Hi Anushka," when appropriate.
Keep the response concise, professional, warm, and direct.
Maximum 4-6 concise lines.
Prefer bullet points when helpful.
Avoid long explanations.
Do not use overly long greetings or emotional language.
If you do not know the answer, say so honestly and suggest reaching out to HR.

HR Context:
{context}

Employee Question: {query}

Buddy Response:"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
            )

            st.session_state.last_answer = response.choices[0].message.content
            st.session_state.last_source = source_label(query, top_chunks)

        except Exception as e:
            st.error(f"Buddy could not respond right now. Please try again. ({type(e).__name__})")

if st.session_state.last_answer:
    safe_answer = html.escape(st.session_state.last_answer).replace("\n", "<br>")
    safe_source = html.escape(st.session_state.last_source or "Employee Handbook")
    st.markdown(f'<div class="response-card">{safe_answer}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="source-line">Found in: {safe_source}</div>', unsafe_allow_html=True)
