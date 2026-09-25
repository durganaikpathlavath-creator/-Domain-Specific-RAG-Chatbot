# rag_pipeline.py
"""Module 5 (retrieval) and Module 6 (answer generation)."""

import os
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from prompt import CONDENSE_QUESTION_PROMPT, FALLBACK_MESSAGE, RAG_PROMPT, REFUSAL_MARKER

# Module 5: default retrieve count (overridable from the UI slider).
DEFAULT_TOP_K = 4

# How many past chat turns (user+assistant pairs) to feed into the follow-up
# rewrite step. Keeping this small keeps the extra LLM call cheap and on-topic.
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))

# Chunks whose cosine similarity to the question is below this value are treated as
# "not relevant". If NO chunk passes, the bot refuses without calling the LLM at all.
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.20"))

# Default model — the UI dropdown can override this at runtime.
DEFAULT_MODEL = "gemini-3.6-flash"


class MissingAPIKeyError(RuntimeError):
    """GOOGLE_API_KEY is not configured."""


def get_llm(model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Build a Gemini LLM client.

    ``api_key`` (if provided) overrides the environment / secrets key so the
    user can paste a personal key in the sidebar.  ``model`` is the model name
    chosen in the sidebar dropdown.
    """
    resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY is not set. Add it to your .env file (local) or to "
            "Streamlit secrets (cloud), or paste it in the sidebar, then restart."
        )
    return ChatGoogleGenerativeAI(model=model, google_api_key=resolved_key)


def _similarity_from_distance(squared_l2: float) -> float:
    """Convert a FAISS distance into cosine similarity.

    FAISS IndexFlatL2 returns the SQUARED L2 distance. For unit-length vectors
    (our embeddings are normalised) squared_L2 = 2 - 2*cosine, so
    cosine = 1 - squared_L2 / 2.
    """
    return max(0.0, min(1.0, 1.0 - squared_l2 / 2.0))


def retrieve(question: str, vector_store, k: int = DEFAULT_TOP_K) -> List[Dict]:
    """Return the k most similar chunks, best first, with a 0-1 relevance score."""
    results = vector_store.similarity_search_with_score(question, k=k)
    hits = []
    for doc, distance in results:
        hits.append(
            {
                "source": doc.metadata.get("source", "Unknown document"),
                "page": doc.metadata.get("page", "N/A"),
                "score": _similarity_from_distance(float(distance)),
                "text": doc.page_content,
            }
        )
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def _build_context(hits: List[Dict]) -> str:
    parts = []
    for hit in hits:
        parts.append(f"[Source: {hit['source']} | Page: {hit['page']}]\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


def _unique_sources(hits: List[Dict]) -> List[Dict]:
    """One entry per (document, page), keeping the best-scoring chunk of that page."""
    best: Dict[tuple, Dict] = {}
    for hit in hits:
        key = (hit["source"], hit["page"])
        if key not in best or hit["score"] > best[key]["score"]:
            best[key] = hit
    ordered = sorted(best.values(), key=lambda h: h["score"], reverse=True)
    return [
        {
            "source": h["source"],
            "page": h["page"],
            "score": h["score"],
            "snippet": (h["text"][:300] + "...") if len(h["text"]) > 300 else h["text"],
        }
        for h in ordered
    ]


def _message_text(message) -> str:
    """Get plain text from a model reply (content can be a string or a list of blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _to_lc_messages(chat_history: Optional[List[Dict]]) -> List:
    """Convert the app's [{"role", "content", ...}, ...] history into LangChain
    message objects, keeping only the most recent turns."""
    if not chat_history:
        return []
    trimmed = chat_history[-2 * MAX_HISTORY_TURNS :]
    messages = []
    for turn in trimmed:
        content = turn.get("content", "")
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def condense_question(
    question: str,
    chat_history: Optional[List[Dict]],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> str:
    """Rewrite a follow-up question into a standalone one using recent history.

    Falls back to the original question if there is no history, if an error
    occurs, or if the rewrite comes back empty.
    """
    lc_history = _to_lc_messages(chat_history)
    if not lc_history:
        return question

    try:
        llm = get_llm(model=model, api_key=api_key)
        chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
        rewritten = chain.invoke({"chat_history": lc_history, "question": question}).strip()
        return rewritten or question
    except Exception:
        return question


def generate_answer(
    question: str,
    vector_store,
    k: int = DEFAULT_TOP_K,
    chat_history: Optional[List[Dict]] = None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> Dict:
    """Answer a question using only the indexed documents.

    ``model`` and ``api_key`` come from the sidebar dropdown / override field.
    ``k`` is the top-k slider value.  ``chat_history`` is the app's list of past
    {"role", "content"} turns (optional).

    Returns {"answer": str, "sources": [...], "refused": bool}.
    `sources` is empty when the bot refuses.
    """
    question = question.strip()
    standalone_question = condense_question(question, chat_history, model=model, api_key=api_key)

    hits = retrieve(standalone_question, vector_store, k)
    relevant = [h for h in hits if h["score"] >= MIN_RELEVANCE]

    # Nothing in the documents is even close to the question -> refuse, no LLM call.
    if not relevant:
        return {"answer": FALLBACK_MESSAGE, "sources": [], "refused": True}

    # Module 6: Return clean string replies using StrOutputParser()
    llm = get_llm(model=model, api_key=api_key)
    chain = RAG_PROMPT | llm | StrOutputParser()
    raw_answer = chain.invoke({"context": _build_context(relevant), "question": standalone_question})
    answer = (raw_answer if isinstance(raw_answer, str) else _message_text(raw_answer)).strip() or FALLBACK_MESSAGE

    refused = REFUSAL_MARKER in answer.lower()
    return {
        "answer": answer,
        "sources": [] if refused else _unique_sources(relevant),
        "refused": refused,
    }
