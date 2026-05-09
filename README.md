# Buddy — HR & Compliance AI Assistant
## Project Log & Build Notes

---

## What is Buddy?

Buddy is a RAG-powered HR and Compliance AI assistant. It reads real company documents — handbooks, policies, onboarding guides — and answers employee questions accurately from them. Unlike a normal chatbot that makes things up, Buddy only answers from what's actually in the documents.

**Who it's for:**
- New employees going through onboarding
- Existing employees with HR or compliance questions
- HR managers tracking who has completed what

**The core experience:**
- Employee asks a question in plain English
- Buddy searches the HR documents for relevant information
- Buddy responds warmly and accurately — like a knowledgeable colleague, not a robot
- A sidebar shows the employee's onboarding progress and what they still need to complete

---

## How RAG works in Buddy

**Phase 1 — Indexing (done once at startup):**
1. HR documents are split into small chunks (~500 tokens each)
2. Each chunk is converted into a vector (a list of numbers representing its meaning) using an embedding model
3. These vectors are stored in ChromaDB — a vector database

**Phase 2 — Every time a user asks a question:**
1. The question is also converted into a vector
2. ChromaDB finds the 3 most similar chunks (semantic search)
3. Those chunks are passed to Gemini as context
4. Gemini writes a warm, accurate answer grounded in the real documents

---

## What we built — step by step

### Step 1 — Installed all libraries
```bash
pip install langchain-google-genai chromadb langchain pypdf streamlit
pip install langchain-community langchain-text-splitters
pip install sentence-transformers
pip install langchain-huggingface
```

### Step 2 — Created the project folder
```bash
mkdir ~/Desktop/buddy
cd ~/Desktop/buddy
touch app.py
touch .env
```

### Step 3 — Added the Gemini API key to .env
```
GOOGLE_API_KEY=your_key_here
```

### Step 4 — Wrote the app.py code
Core components:
- HR handbook hardcoded as a string (dummy data for now)
- `build_vector_db()` function to chunk, embed, and store documents
- Streamlit UI with a chat input and sidebar
- RAG pipeline: embed query → search ChromaDB → pass context to Gemini → display response

---

## What didn't work and why

### Problem 1 — Wrong import paths (LangChain restructured)
LangChain moved modules into separate packages in newer versions.

| Old (broken) | New (working) |
|---|---|
| `from langchain.text_splitter import ...` | `from langchain_text_splitters import ...` |
| `from langchain_community.embeddings import HuggingFaceEmbeddings` | `from langchain_huggingface import HuggingFaceEmbeddings` |
| `from langchain.chains import RetrievalQA` | Replaced with direct similarity search |

### Problem 2 — Google embedding model not found (404)
Both `models/text-embedding-004` and `models/embedding-001` returned 404 errors because:
- The newer langchain-google-genai library uses API v1beta
- Some Google embedding models are only on v1 (stable)

**Solution:** Switched to a free local embedding model — `all-MiniLM-L6-v2` from HuggingFace via `sentence-transformers`. No API key needed, runs on your Mac, works perfectly.

### Problem 3 — ChromaDB `KeyError: 'ephemeral'`
The newer version of ChromaDB (1.5.9) changed its internal API. The LangChain wrapper used the old API.

**Solution:** Used ChromaDB's native `EphemeralClient()` directly instead of going through LangChain's wrapper.

### Problem 4 — ChromaDB collection already exists
On rerun, `create_collection()` failed because the collection already existed.

**Solution:** Changed to `get_or_create_collection()`.

### Problem 5 — Gemini model names (404)
- `gemini-1.5-flash` → 404 (old model, removed from v1beta API)
- `gemini-2.0-flash` → quota exhausted on free tier
- `gemini-2.5-flash` → works ✅

**Solution:** Claude Code diagnosed this and updated the model name automatically.

### Problem 6 — Permissions error installing Claude Code
```bash
npm install -g @anthropic-ai/claude-code  # failed
sudo npm install -g @anthropic-ai/claude-code  # worked
```

### How Claude Code helped
When the errors kept stacking, we installed Claude Code — Anthropic's terminal-based coding agent. It:
- Read the app.py file automatically
- Checked installed package versions
- Diagnosed both remaining issues (HuggingFace deprecation + Gemini model name)
- Fixed the code directly in the file
- Confirmed the fix worked

---

## Final working stack

| Component | Tool | Why |
|---|---|---|
| Frontend | Streamlit | Python-native, easy to deploy |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free, local, no API needed |
| Vector database | ChromaDB EphemeralClient | In-memory, no setup required |
| LLM | Gemini 2.5 Flash via Google API | Free tier, fast responses |
| Environment | Python 3.13 + Miniconda | Already installed |
| API key management | python-dotenv + .env file | Keeps key out of code |

---

## What Buddy can do right now

- Answer any question from the HR handbook accurately
- Respond warmly and encouragingly like a real buddy
- Show the employee's onboarding progress in the sidebar
- Flag what's still incomplete

**Test questions that work:**
- "What is the PTO policy?"
- "What do I need to complete in my first 14 days?"
- "How does BGV work?"
- "When are my 1-on-1s with my manager?"

---

## Next steps

### This week
- [ ] Push to GitHub with a clean README
- [ ] Deploy on Streamlit Cloud (free) so it has a live URL
- [ ] Add the live URL to your resume and LinkedIn

### Week 2 — Make it smarter
- [ ] Allow PDF upload so any HR document can be loaded
- [ ] Add multiple employee profiles (not just Anushka)
- [ ] Make compliance checklist interactive (checkboxes that update progress)
- [ ] Add a proactive daily check-in message based on onboarding day

### Week 3 — Make it agentic
- [ ] Add a manager view showing all employees and their progress
- [ ] Add a "Send Nudge" button that triggers Buddy to message an overdue employee
- [ ] Connect to a real database instead of in-memory ChromaDB

### Long term (the full Buddy vision)
- [ ] Slack integration — Buddy lives where employees already work
- [ ] Real-time progress tracking via Asana/Google Workspace webhooks
- [ ] Persistent memory so Buddy remembers past conversations
- [ ] Multi-company support with separate document stores per organisation

---

## How to run Buddy locally

```bash
cd ~/Desktop/buddy
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## What this project proves on your resume

- You can build a RAG pipeline from scratch — chunking, embedding, vector search, LLM response
- You understand the full AI engineering stack — not just calling an API
- You can debug complex dependency and API errors across multiple libraries
- You shipped something that actually works and answers questions accurately
- You used Claude Code, an agentic AI tool, to diagnose and fix production errors
