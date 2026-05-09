import os
import numpy as np
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

hr_docs = """
EMPLOYEE HANDBOOK

PTO Policy: Employees receive 20 days of Paid Time Off per year. Requests must be submitted via the HR Portal.

BGV: Must be completed before Day 1. US employees use HireRight, UK employees use Credence.

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
    {"name": "Jordan Kim", "role": "Designer", "progress": 85, "status": "Overdue"}
]

def cosine_similarity(query_vec, matrix):
    dots = matrix @ query_vec
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    return dots / (norms + 1e-10)

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

st.set_page_config(page_title="Buddy", page_icon="🤝")

with st.sidebar:
    st.header("Logged in as")
    st.info("Anushka Naidu - Data Science Student - Day 1 of onboarding")
    st.progress(20, text="Onboarding: 20% complete")
    st.warning("Still to complete: AI Ethics Training, Tool Onboarding, First Project")
    st.divider()
    view_mode = st.radio("Switch View:", ["Employee Portal", "HR Manager View"])

if view_mode == "HR Manager View":
    st.title("📈 Manager Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Onboarding Avg", "51%")
    col2.metric("At Risk", "1")
    col3.metric("Overdue Tasks", "4")
    st.subheader("Employee Progress")
    for emp in employees_stats:
        with st.expander(f"{emp['name']} — {emp['role']}"):
            c1, c2 = st.columns([3, 1])
            c1.progress(emp['progress'])
            if emp['status'] in ["At Risk", "Overdue"]:
                if c2.button(f"Nudge {emp['name'].split()[0]}", key=emp['name']):
                    st.toast(f"Nudge sent to {emp['name']}!", icon="👋")
            else:
                c2.success(emp['status'])

else:
    st.title("🤝 Buddy — HR & Compliance Assistant")
    st.caption("Ask me anything about HR policies and onboarding.")
    if "chunks" not in st.session_state:
        with st.spinner("Buddy is loading..."):
            st.session_state.chunks, st.session_state.embeddings_matrix, st.session_state.embeddings_model = build_vector_db(hr_docs)
            st.success("Buddy is ready!")
    query = st.text_input("Ask Buddy something:")
    if query:
        with st.spinner("Buddy is thinking..."):
            top_chunks = query_vector_db(
                st.session_state.chunks,
                st.session_state.embeddings_matrix,
                st.session_state.embeddings_model,
                query
            )
            context = "\n".join(top_chunks)
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
            prompt = f"""You are Buddy, a warm and encouraging HR and compliance assistant.
Use the HR context below to answer the employee question accurately.
If you do not know the answer, say so honestly and suggest they reach out to their HR team.
Be friendly and specific like a helpful colleague not a robot.

HR Context:
{context}

Employee Question: {query}

Buddy Response:"""
            response = llm.invoke([HumanMessage(content=prompt)])
            st.markdown("### Buddy says:")
            st.write(response.content)
