# app.py
"""Module 7: Streamlit interface for the Domain-Specific RAG Chatbot."""

import os

import streamlit as st
from dotenv import load_dotenv

# Local development: read GOOGLE_API_KEY from .env
load_dotenv()

# Streamlit Cloud: read GOOGLE_API_KEY from st.secrets (no .env file there).
try:
    if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass  # no secrets file configured - that is fine locally

from document_loader import MAX_FILE_MB, extract_text_from_pdfs  # noqa: E402
from rag_pipeline import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    MissingAPIKeyError,
    generate_answer,
)
from vector_store import (  # noqa: E402
    create_vector_store,
    delete_saved_index,
    load_vector_store,
    save_vector_store,
    saved_index_exists,
)

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📄", layout="wide")

# Gemini model options shown in the sidebar dropdown (newest first).
GEMINI_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-1.5-flash"]

# Default slider values (must match module defaults so a fresh load is consistent).
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "vector_store": None,   # FAISS index for the current documents
        "indexed_files": [],    # [{"name", "pages", "empty_pages"}]
        "skipped_files": [],    # [{"name", "reason"}]
        "total_chunks": 0,      # chunk count from the last build
        "chat_history": [],     # [{"role", "content", "sources"}]
        "uploader_key": 0,      # changing this resets the file uploader widget
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# ---------------------------------------------------------------------------
# Button callbacks
# ---------------------------------------------------------------------------
def clear_chat() -> None:
    st.session_state.chat_history = []


def clear_documents() -> None:
    """Remove the indexed documents (and the saved index on disk), the chat, and the uploads."""
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.session_state.skipped_files = []
    st.session_state.total_chunks = 0
    st.session_state.chat_history = []
    st.session_state.uploader_key += 1
    delete_saved_index()


def process_documents(uploaded_files, persist: bool, chunk_size: int, chunk_overlap: int) -> None:
    """Extract -> chunk -> embed -> index. Replaces any previously indexed documents."""
    with st.spinner("Extracting text, creating embeddings and building the FAISS index..."):
        documents, loaded, skipped = extract_text_from_pdfs(uploaded_files)
        st.session_state.skipped_files = skipped

        if not documents:
            st.error("None of the selected files could be used. See the reasons below.")
            return

        try:
            store, chunk_count = create_vector_store(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as exc:
            st.error(f"Could not build the vector store ({type(exc).__name__}): {exc}")
            return

        st.session_state.vector_store = store
        st.session_state.indexed_files = loaded
        st.session_state.total_chunks = chunk_count
        st.session_state.chat_history = []  # new documents -> start a fresh conversation

        if persist:
            try:
                save_vector_store(store, loaded)
            except Exception as exc:
                st.warning(f"The documents were indexed but the index could not be saved: {exc}")

    st.success(f"Indexed {len(loaded)} PDF file(s). You can now ask questions.")


def load_saved_index() -> None:
    try:
        store, files = load_vector_store()
    except Exception as exc:
        st.error(f"Could not load the saved index ({type(exc).__name__}): {exc}")
        return
    if store is None:
        st.warning("No saved index was found.")
        return
    st.session_state.vector_store = store
    st.session_state.indexed_files = files
    st.session_state.skipped_files = []
    st.session_state.chat_history = []
    st.success("Saved index loaded.")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_sources(sources) -> None:
    """Display answer with source document and page."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src['source']}** - page {src['page']}  |  relevance {src['score']:.0%}")
            st.caption(src["snippet"])


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources"))


# ---------------------------------------------------------------------------
# Sidebar: document upload (Module 1) and controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Document Operations")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"PDF only, up to {MAX_FILE_MB} MB per file.",
    )
    st.caption(f"PDF only - max {MAX_FILE_MB}MB per file. Do not upload confidential documents without permission.")

    # Show uploaded file names and sizes before processing.
    if uploaded_files:
        st.subheader("Selected files")
        for uf in uploaded_files:
            size_kb = uf.size / 1024
            if size_kb >= 1024:
                st.write(f"📄 **{uf.name}** ({size_kb / 1024:.1f} MB)")
            else:
                st.write(f"📄 **{uf.name}** ({size_kb:.0f} KB)")

    # -------------------------------------------------------------------
    # RAG Parameters (MUST be defined before the Process Documents button)
    # -------------------------------------------------------------------
    st.divider()
    st.subheader("RAG Parameters")

    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
        help="Number of characters per text chunk. Larger chunks retain more context but may dilute relevance.",
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=300,
        value=DEFAULT_CHUNK_OVERLAP,
        step=10,
        help="Overlap between consecutive chunks to preserve context across boundaries.",
    )
    top_k = st.slider(
        "Chunks to Retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        step=1,
        help="How many of the most relevant chunks to pass to the LLM.",
    )

    # -------------------------------------------------------------------
    # Persistence & Process button
    # -------------------------------------------------------------------
    persist = st.checkbox(
        "Save index on this computer for reuse",
        value=False,
        help=(
            "Stores the index in vector_store/saved_index/. Leave OFF on shared or "
            "deployed servers, because the saved index contains text from your documents."
        ),
    )

    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please choose at least one PDF file first.")
        else:
            process_documents(uploaded_files, persist, chunk_size, chunk_overlap)

    if saved_index_exists():
        st.button("Load saved index", on_click=load_saved_index)

    # Show which documents are active, and which were rejected.
    if st.session_state.indexed_files:
        st.subheader("Indexed documents")
        for info in st.session_state.indexed_files:
            details = f"{info.get('pages', '?')} pages"
            if info.get("empty_pages"):
                details += f", {info['empty_pages']} empty page(s) skipped"
            st.write(f"📄 **{info['name']}** ({details})")
    for item in st.session_state.skipped_files:
        st.warning(f"Skipped **{item['name']}**: {item['reason']}")

    # -------------------------------------------------------------------
    # Model & API Settings
    # -------------------------------------------------------------------
    st.divider()
    st.subheader("⚙️ Model & API Settings")

    api_key_env = os.getenv("GOOGLE_API_KEY", "")
    if api_key_env:
        st.success("🔒 Gemini API Key: Configured")
    else:
        st.warning("⚠️ Gemini API: Not configured")

    api_key_override = st.text_input(
        "Override Gemini API Key (Optional)",
        type="password",
        placeholder="Paste a key to override .env/secrets",
        help="Optional. If provided, this key is used instead of the one in .env or Streamlit secrets.",
    )
    effective_api_key = api_key_override.strip() or None

    selected_model = st.selectbox(
        "Gemini Model",
        options=GEMINI_MODELS,
        index=0,
        help="Choose the Gemini model for answer generation.",
    )

    st.caption("Embedding Model: sentence-transformers/all-MiniLM-L6-v2")

    # -------------------------------------------------------------------
    # Bottom Controls
    # -------------------------------------------------------------------
    st.divider()
    col_a, col_b = st.columns(2)
    col_a.button("Clear Chat", on_click=clear_chat)
    col_b.button(
        "Reset Documents & Knowledge Base",
        on_click=clear_documents,
        help="Removes the indexed documents, the chat and any saved index. Upload new files to replace them.",
    )


# ---------------------------------------------------------------------------
# Main page: chat
# ---------------------------------------------------------------------------
st.title("📄 Domain-Specific RAG Chatbot")
st.subheader("Grounded PDF Question Answering with Page-Level Source Attribution")
st.warning(
    "⚠️ Responsible AI Disclaimer: AI-generated responses are based on retrieved "
    "document passages. Please verify all critical or high-stakes information "
    "against original official documents."
)

# Architecture expander
with st.expander("ℹ️ How this RAG System Works (PDF Guidance Architecture)"):
    st.markdown(
        """
**1. PDF Parsing & Validation** — Uploaded PDFs are validated by extension, magic bytes (`%PDF-`), and a 200 MB size limit. Text is extracted page-by-page using `pypdf`, preserving the source filename and page number as metadata.

**2. Recursive Chunking** — Extracted text is split into overlapping chunks using `RecursiveCharacterTextSplitter` (configurable size & overlap). This keeps passages focused while preserving context across boundaries.

**3. Embedding & FAISS Indexing** — Each chunk is converted into a 384-dimensional dense vector via `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index for fast similarity search.

**4. Similarity Search & Retrieval** — When a question is asked, it is embedded and compared against all chunk vectors using cosine similarity (derived from FAISS L2 distance). The top-k most relevant chunks are retrieved, and chunks below the relevance threshold (0.20) are filtered out.

**5. Gemini LLM Generation** — Filtered chunks are injected into a strict guardrail prompt inside `<context>` tags. Google's Gemini model generates an answer **only** from the supplied context, citing the source document and page number. If no relevant chunks are found, the bot refuses to answer rather than fabricate facts.
        """
    )

# Live metric dashboard
if st.session_state.vector_store is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed Documents", len(st.session_state.indexed_files))
    col2.metric("Total Text Chunks", st.session_state.total_chunks)
    col3.metric("Vector Database", "FAISS (Local Disk)")

if not os.getenv("GOOGLE_API_KEY") and not effective_api_key:
    st.error(
        "GOOGLE_API_KEY is not set. Add it to a .env file (local) or to Streamlit "
        "secrets (cloud), or paste it in the sidebar under Model & API Settings."
    )

if st.session_state.vector_store is None:
    st.write("👈 Upload one or more PDFs in the sidebar and click **Process Documents** to begin.")

for message in st.session_state.chat_history:
    render_message(message)

query = st.chat_input("Ask a question based on the uploaded documents...")

if query:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process at least one PDF file first.")
    else:
        user_message = {"role": "user", "content": query}
        render_message(user_message)

        result = None
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    result = generate_answer(
                        query,
                        st.session_state.vector_store,
                        k=top_k,
                        chat_history=st.session_state.chat_history,
                        model=selected_model,
                        api_key=effective_api_key,
                    )
                except MissingAPIKeyError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Could not generate an answer ({type(exc).__name__}): {str(exc)[:300]}")

            if result is not None:
                st.markdown(result["answer"])
                render_sources(result["sources"])

        if result is not None:
            st.session_state.chat_history.append(user_message)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
