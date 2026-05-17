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

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Buddy", page_icon="🤝", layout="wide")

# -----------------------------
# Simple polish only
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0f1117;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 900px;
    padding-top: 4rem;
}

[data-testid="stSidebar"] {
    background: #24252f;
}

[data-testid="stSidebar"] * {
    color: #f4f4f5;
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
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.2rem;
}

.response-card {
    background: white;
    color: #111827;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    margin-top: 1rem;
    line-height: 1.6;
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
    background: #282a36 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}

.stButton > button {
    border-radius: 999px !important;
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}

.stButton > button:hover {
    border-color: #6ba176 !important;
    background: rgba(107,161,118,0.18) !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
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

# -----------------------------
# Secrets
# -----------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("Add GROQ_API_KEY to Streamlit Secrets!")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Logged in as")
    st.info("Anushka Naidu - Data Science Student - Day 1 of onboarding")
    st.progress(20, text="Onboarding: 20% complete")
    st.warning("Still to complete: AI Ethics Training, Tool Onboarding, First Project")
    st.divider()
    view_mode = st.radio("Switch View:", ["Employee Portal", "HR Manager View"])

# -----------------------------
# Manager View
# -----------------------------
if view_mode == "HR Manager View":
    st.title("📈 Manager Insights")
    st.caption("Quick overview of employee onboarding progress.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Onboarding Avg", "51%")
    col2.metric("At Risk", "1")
    col3.metric("Overdue Tasks", "4")

    st.subheader("Employee Progress")
    for emp in employees_stats:
        with st.expander(f"{emp['name']} — {emp['role']}"):
            c1, c2 = st.columns([3, 1])
            c1.progress(emp["progress"] / 100, text=f"{emp['progress']}% complete")
            if emp["status"] in ["At Risk", "Overdue"]:
                if c2.button(f"Nudge {emp['name'].split()[0]}", key=emp["name"]):
                    st.toast(f"Nudge sent to {emp['name']}!", icon="👋")
            else:
                c2.success(emp["status"])

# -----------------------------
# Employee Portal
# -----------------------------
else:
    st.markdown("""
    <div class="hero-card">
        <h1>🤝 Buddy</h1>
        <p style="font-size:1.05rem; font-weight:600; margin-bottom:0.25rem;">Your AI workplace assistant</p>
        <p style="color:#a1a1aa;">Ask me anything about HR policies, onboarding, tools, training, or PTO.</p>
    </div>
    """, unsafe_allow_html=True)

    if "chunks" not in st.session_state:
        with st.spinner("Buddy is loading..."):
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
        if cols[i].button(question, use_container_width=True):
            st.session_state["selected_question"] = question

    query = st.text_input(
        "Ask Buddy something:",
        value=st.session_state.pop("selected_question", ""),
        placeholder="Example: What trainings do I need to complete?"
    )

    if query:
        with st.spinner("Buddy is thinking..."):
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
Be friendly, specific, and concise.

HR Context:
{context}

Employee Question: {query}

Buddy Response:"""

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                )

                answer = response.choices[0].message.content

                st.markdown("### Buddy says:")
                st.markdown(f'<div class="response-card">{answer}</div>', unsafe_allow_html=True)

                with st.expander("View source used"):
                    st.markdown(f'<div class="source-card">{top_chunks[0]}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Buddy could not respond right now. Please try again. ({type(e).__name__})")
