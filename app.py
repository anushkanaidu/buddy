import os
import streamlit as st
import chromadb
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

def build_vector_db(text_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text_data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedded_chunks = embeddings.embed_documents(chunks)
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("hr_policies")
    collection.add(
        documents=chunks,
        embeddings=embedded_chunks,
        ids=[str(i) for i in range(len(chunks))]
    )
    return collection, embeddings

st.set_page_config(page_title="Buddy", page_icon="🤝")
st.title("🤝 Buddy — HR & Compliance Assistant")
st.caption("Ask me anything about HR policies and onboarding.")

if "collection" not in st.session_state:
    with st.spinner("Buddy is loading — takes 30 seconds first time..."):
        st.session_state.collection, st.session_state.embeddings = build_vector_db(hr_docs)
        st.success("Buddy is ready!")

with st.sidebar:
    st.header("Logged in as")
    st.info("Anushka Naidu - Data Science Student - Day 1 of onboarding")
    st.progress(20, text="Onboarding: 20% complete")
    st.warning("Still to complete: AI Ethics Training, Tool Onboarding, First Project")

query = st.text_input("Ask Buddy something:")

if query:
    with st.spinner("Buddy is thinking..."):
        query_embedding = st.session_state.embeddings.embed_query(query)
        results = st.session_state.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        context = "\n".join(results["documents"][0])
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
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