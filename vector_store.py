# vector_store.py
"""Module 4: Embeddings and FAISS vector store creation, persistence, and loading."""

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Graceful imports for FAISS
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain_community.vectorstores.faiss import FAISS

# Graceful imports for HuggingFaceEmbeddings (langchain-huggingface / langchain-community)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Module 3 chunking logic is imported from document_loader
from document_loader import CHUNK_OVERLAP, CHUNK_SIZE, split_documents

# The guidance calls this model "all-MiniLM-L6-v2" (full Hugging Face id below).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Where the index is saved when the user chooses to keep it (see the sidebar checkbox).
INDEX_DIR = Path(__file__).resolve().parent / "vector_store" / "saved_index"
MANIFEST_NAME = "manifest.json"


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Load the embedding model once per process (cached for performance).

    Default: Hugging Face sentence-transformers/all-MiniLM-L6-v2 (CPU, normalised).
    Fallback: GoogleGenerativeAIEmbeddings if HuggingFace fails or configured.
    Embeddings are normalised to length 1 so FAISS L2 distance maps directly to
    cosine similarity.
    """
    use_gemini = os.getenv("USE_GEMINI_EMBEDDINGS", "false").lower() in ("true", "1")
    if use_gemini:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        # Graceful fallback to Gemini embeddings if sentence-transformers has issues
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        raise RuntimeError(
            f"Failed to load HuggingFace embedding model ({EMBEDDING_MODEL}): {exc}"
        ) from exc


def create_vector_store(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[FAISS, int]:
    """Chunk the documents, embed every chunk and build a FAISS index.

    Returns (FAISS_index, total_chunk_count) so the UI can show the metric.
    """
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No text chunks could be created from the documents.")
    store = FAISS.from_documents(chunks, get_embeddings())
    return store, len(chunks)


# ---------------------------------------------------------------------------
# Optional persistence ("Save the vector store locally if the application must reuse it")
# ---------------------------------------------------------------------------

def saved_index_exists(path: Path = INDEX_DIR) -> bool:
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def save_vector_store(store: FAISS, files_info: List[Dict], path: Path = INDEX_DIR) -> None:
    """Save the FAISS index plus a small manifest listing the indexed files."""
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    manifest = {"files": files_info}
    (path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_vector_store(path: Path = INDEX_DIR) -> Tuple[Optional[FAISS], List[Dict]]:
    """Load a previously saved index. Returns (None, []) if nothing is saved."""
    if not saved_index_exists(path):
        return None, []
    # FAISS.load_local uses pickle. That is only safe for files YOU created, which is
    # the case here (the app writes this folder itself). Never load an index from
    # an untrusted source.
    store = FAISS.load_local(
        str(path), get_embeddings(), allow_dangerous_deserialization=True
    )
    files: List[Dict] = []
    manifest_file = path / MANIFEST_NAME
    if manifest_file.exists():
        try:
            files = json.loads(manifest_file.read_text(encoding="utf-8")).get("files", [])
        except (OSError, ValueError):
            files = []
    return store, files


def delete_saved_index(path: Path = INDEX_DIR) -> None:
    shutil.rmtree(path, ignore_errors=True)
