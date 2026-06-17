"""
app.py — RAG Document Q&A for Hugging Face Spaces

Each browser session gets completely isolated services (own ChromaDB instance).
Uploading and querying is scoped to the current session only.
Refreshing the page starts completely fresh.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.conf-HIGH   { background:#d4edda; color:#155724; padding:3px 10px; border-radius:6px; font-weight:600; font-size:13px; }
.conf-MEDIUM { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:6px; font-weight:600; font-size:13px; }
.conf-LOW    { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:6px; font-weight:600; font-size:13px; }
.answer-box  { border:1px solid rgba(128,128,128,0.3); border-radius:8px; padding:18px; margin:10px 0; }
.src-card    { border-left:3px solid #6c757d; padding:8px 12px; margin:4px 0; border-radius:4px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ── Env setup ─────────────────────────────────────────────────────────────────
# Works on Streamlit Community Cloud (st.secrets), Hugging Face Spaces (env var),
# and local development (.env / shell env var) — whichever is available first.
def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


GROQ_API_KEY = _get_secret("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error(
        "⚠️ GROQ_API_KEY is not set.\n\n"
        "- On Streamlit Community Cloud: App settings → Secrets → add `GROQ_API_KEY = \"...\"`\n"
        "- On Hugging Face Spaces: Space Settings → Secrets → add `GROQ_API_KEY`"
    )
    st.stop()

os.environ["GROQ_API_KEY"]          = GROQ_API_KEY
os.environ.setdefault("EMBEDDING_MODEL",        "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "rag_docs")
os.environ.setdefault("LLM_MODEL",              "llama-3.3-70b-versatile")
os.environ.setdefault("LLM_TEMPERATURE",        "0.1")
os.environ.setdefault("LLM_MAX_TOKENS",         "1024")
os.environ.setdefault("RETRIEVAL_K",            "5")
os.environ.setdefault("SEMANTIC_WEIGHT",        "0.6")
os.environ.setdefault("BM25_WEIGHT",            "0.4")
os.environ.setdefault("CHUNK_SIZE",             "512")
os.environ.setdefault("CHUNK_OVERLAP",          "64")

# ── Session initialisation ────────────────────────────────────────────────────
# We store service instances in session_state.
# Each session creates brand-new instances with their own EphemeralClient.
# This means User A and User B never share any data.

if "vs" not in st.session_state:
    from backend.services.vectorstore import VectorStoreService
    from backend.services.retriever   import HybridRetrieverService
    from backend.services.qa_chain    import QAChainService

    vs = VectorStoreService()
    st.session_state.vs      = vs
    st.session_state.hr      = HybridRetrieverService(vs)
    st.session_state.qa      = QAChainService()
    st.session_state.sources = []   # list of indexed filenames this session

vs     = st.session_state.vs
hr     = st.session_state.hr
qa_svc = st.session_state.qa

# ── Actions ───────────────────────────────────────────────────────────────────

def upload_files(files):
    from backend.services.ingestion import ingest_file
    results, total = [], 0
    for f in files:
        if f.name in st.session_state.sources:
            results.append({"name": f.name, "ok": True, "skip": True, "n": 0})
            continue
        suffix = Path(f.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            chunks = ingest_file(tmp_path)
            for c in chunks:
                c.metadata["source"] = f.name
            vs.add_documents(chunks)
            st.session_state.sources.append(f.name)
            results.append({"name": f.name, "ok": True, "skip": False, "n": len(chunks)})
            total += len(chunks)
        except Exception as e:
            results.append({"name": f.name, "ok": False, "skip": False, "n": 0, "err": str(e)})
        finally:
            os.unlink(tmp_path)
    if total > 0:
        try:
            hr.rebuild()
        except Exception:
            pass
    return results, total


def clear_all():
    vs.clear()
    hr.reset()
    st.session_state.sources = []


def ask(question: str):
    return qa_svc.answer(question, hr)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Doc Q&A")
    st.caption("Hybrid Search · Cited Answers · Groq / Llama 3.3")
    st.divider()

    n = vs.count()
    st.metric("Chunks indexed", n)
    st.divider()

    st.subheader("Upload Documents")
    files = st.file_uploader(
        "PDF · DOCX · TXT · Markdown",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if files and st.button("📤 Index Documents", use_container_width=True):
        with st.spinner("Indexing..."):
            results, total = upload_files(files)
        for r in results:
            if r.get("skip"):
                st.info(f"⏭️ `{r['name']}` already indexed")
            elif r["ok"]:
                st.success(f"✅ `{r['name']}` — {r['n']} chunks")
            else:
                st.error(f"❌ `{r['name']}`: {r.get('err','unknown error')}")
        if total:
            st.rerun()

    st.divider()
    st.subheader("Indexed this session")
    srcs = st.session_state.sources
    if srcs:
        for s in srcs:
            st.markdown(f"- `{s}`")
        if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
            clear_all()
            st.rerun()
    else:
        st.info("No documents yet. Upload to get started.")

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · Streamlit")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Ask your documents")
st.markdown(
    "Upload documents in the sidebar, then ask any question below. "
    "Every session starts fresh — your documents are never shared with anyone else."
)

if n > 0:
    st.markdown("**Quick questions:**")
    q_examples = ["What are the main topics?", "Summarise the key findings.", "What conclusions are drawn?"]
    cols = st.columns(3)
    for col, q in zip(cols, q_examples):
        if col.button(q, use_container_width=True):
            st.session_state["prefill"] = q

prefill  = st.session_state.pop("prefill", "")
question = st.text_area(
    "Question",
    value=prefill,
    placeholder="e.g. What does the document say about data privacy?",
    height=80,
    label_visibility="collapsed",
)

col_ask, _ = st.columns([1, 5])
clicked = col_ask.button("🔍 Ask", type="primary", use_container_width=True)

if clicked:
    if not question.strip():
        st.warning("Please type a question first.")
    elif n == 0:
        st.warning("No documents indexed yet — please upload at least one.")
    else:
        with st.spinner("Thinking..."):
            try:
                resp  = ask(question.strip())
                error = None
            except Exception as e:
                resp  = None
                error = str(e)

        if error:
            st.error(f"Error: {error}")
        elif resp:
            st.divider()
            badge = f'<span class="conf-{resp.confidence}">{resp.confidence} CONFIDENCE</span>'
            st.markdown(f"### Answer &nbsp; {badge}", unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{resp.answer}</div>', unsafe_allow_html=True)

            if resp.sources:
                st.markdown(f"#### Sources ({len(resp.sources)})")
                for src in resp.sources:
                    pg = f" · page {src.page}" if src.page is not None else ""
                    with st.expander(f"[Source {src.index}] {src.source}{pg}"):
                        st.markdown(f'<div class="src-card">{src.content[:300]}…</div>',
                                    unsafe_allow_html=True)
