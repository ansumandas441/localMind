"""RAG query: embed question, retrieve from ChromaDB, generate answer via Ollama."""
import ollama
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from localmind.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    OLLAMA_BASE_URL,
    TOP_K,
)


def _get_collection():
    """Get ChromaDB collection with Ollama embedding function."""
    import chromadb
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = OllamaEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        url=OLLAMA_BASE_URL.rstrip("/") + "/api/embeddings",
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "localMind document chunks"},
    )


def retrieve(question: str, top_k: int = TOP_K) -> list[tuple[str, dict, float]]:
    """
    Embed the question and return top-k similar chunks from ChromaDB.
    Returns list of (document_text, metadata, distance).
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0] or []
    metas = results["metadatas"][0] or []
    dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
    return list(zip(docs, metas, dists))


def build_prompt(question: str, chunks: list[tuple[str, dict, float]]) -> str:
    """Build a single prompt with context and question."""
    if not chunks:
        return (
            "No relevant documents were found. Answer based on your general knowledge "
            "and say you have no document context.\n\nQuestion: " + question
        )
    context = "\n\n---\n\n".join(
        f"[Source: {m.get('source', 'unknown')}" + (f", page {m.get('page')}" if m.get("page") else "") + "]\n" + text
        for text, m, _ in chunks
    )
    return (
        "Use the following context from the user's documents to answer the question. "
        "If the context does not contain enough information, say so.\n\n"
        "Context:\n" + context + "\n\nQuestion: " + question
    )


def ask(question: str, top_k: int = TOP_K, stream: bool = False):
    """
    Run RAG: retrieve chunks, build prompt, call Ollama chat.
    If stream=True, yields response chunks; else returns full response and sources.
    """
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)
    client = ollama.Client(host=OLLAMA_BASE_URL)

    if stream:
        stream_gen = client.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], stream=True)
        return stream_gen, chunks
    response = client.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    message = getattr(response, "message", response) if not isinstance(response, dict) else response.get("message") or {}
    answer = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "") or ""
    return answer, chunks
