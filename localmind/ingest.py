"""Document ingestion: load PDFs/text, chunk, embed via Ollama, store in ChromaDB."""
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pypdf import PdfReader

from localmind.config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks (recursive-style: try paragraph, then line, then space)."""
    if not text or not text.strip():
        return []
    separators = ["\n\n", "\n", ". ", " "]
    chunks = [text]
    for sep in separators:
        new_chunks = []
        for c in chunks:
            if len(c) <= chunk_size:
                new_chunks.append(c)
                continue
            parts = c.split(sep) if sep != " " else re.split(r"\s+", c)
            current = ""
            for i, part in enumerate(parts):
                add = part if sep == " " else (part + (sep if i < len(parts) - 1 else ""))
                if len(current) + len(add) <= chunk_size:
                    current += add
                else:
                    if current:
                        new_chunks.append(current.strip())
                    # start next chunk with overlap
                    overlap_start = "" if overlap <= 0 else current[-overlap:] if len(current) >= overlap else current
                    current = overlap_start + add
            if current.strip():
                new_chunks.append(current.strip())
        chunks = new_chunks
    return [c for c in chunks if c.strip()]


def load_pdf(path: Path) -> list[tuple[str, dict]]:
    """Load a PDF file; return list of (page_text, {source, page})."""
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            out.append((text, {"source": path.name, "page": i + 1}))
    return out


def load_text(path: Path) -> list[tuple[str, dict]]:
    """Load a plain text file; return list of (content, {source})."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [(text, {"source": path.name})]


def load_document(path: Path) -> list[tuple[str, dict]]:
    """Load one document (PDF or .txt); return list of (text, metadata)."""
    path = Path(path)
    if not path.exists():
        return []
    suf = path.suffix.lower()
    if suf == ".pdf":
        return load_pdf(path)
    if suf in (".txt", ".md", ".text"):
        return load_text(path)
    return []


def get_chroma_client_and_collection():
    """Return ChromaDB persistent client and the docs collection (with Ollama embeddings)."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = OllamaEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        url=OLLAMA_BASE_URL.rstrip("/") + "/api/embeddings",
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "localMind document chunks"},
    )
    return client, collection


def ingest_paths(paths: list[Path]) -> tuple[int, list[str]]:
    """
    Ingest a list of file/folder paths into ChromaDB.
    Returns (number of chunks added, list of error messages).
    """
    client, collection = get_chroma_client_and_collection()
    errors = []
    all_ids: list[str] = []
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []

    def add_file(file_path: Path) -> None:
        nonlocal all_ids, all_chunks, all_metadatas
        items = load_document(file_path)
        if not items:
            if file_path.suffix.lower() in (".pdf", ".txt", ".md"):
                errors.append(f"Empty or unreadable: {file_path}")
            return
        base_name = file_path.stem
        idx = 0
        for text, meta in items:
            for chunk in _chunk_text(text):
                doc_id = f"{base_name}_{idx}"
                all_ids.append(doc_id)
                all_chunks.append(chunk)
                all_metadatas.append(meta)
                idx += 1

    for p in paths:
        p = Path(p).resolve()
        if not p.exists():
            errors.append(f"Not found: {p}")
            continue
        if p.is_file():
            add_file(p)
        else:
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md", ".text"):
                    add_file(f)

    if not all_chunks:
        return 0, errors

    # ChromaDB expects no embedding_fn when adding with precomputed; we use the collection's
    # embedding function by adding documents (collection will call the embedding function).
    collection.add(ids=all_ids, documents=all_chunks, metadatas=all_metadatas)
    return len(all_ids), errors


def ingest_files_in_memory(file_contents: list[tuple[str, bytes, str]], filenames: list[str]) -> tuple[int, list[str]]:
    """
    Ingest files from in-memory content (e.g. Streamlit uploads).
    file_contents: list of (filename, raw_bytes, inferred_extension).
    filenames: display names for metadata.
    Returns (chunks added, errors).
    """
    import tempfile
    client, collection = get_chroma_client_and_collection()
    errors = []
    all_ids: list[str] = []
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []

    for (name, raw, ext), display_name in zip(file_contents, filenames):
        with tempfile.NamedTemporaryFile(suffix=ext or ".txt", delete=False) as tmp:
            tmp.write(raw)
            tmp.flush()
            tmp_path = Path(tmp.name)
        try:
            items = load_document(tmp_path)
        except Exception as e:
            errors.append(f"{display_name}: {e}")
            tmp_path.unlink(missing_ok=True)
            continue
        tmp_path.unlink(missing_ok=True)
        if not items:
            errors.append(f"Empty or unreadable: {display_name}")
            continue
        base_name = Path(display_name).stem
        idx = 0
        for text, meta in items:
            # override source to uploaded name
            meta = {**meta, "source": display_name}
            for chunk in _chunk_text(text):
                doc_id = f"{base_name}_{idx}_{id(raw)}"
                all_ids.append(doc_id)
                all_chunks.append(chunk)
                all_metadatas.append(meta)
                idx += 1
    if not all_chunks:
        return 0, errors
    collection.add(ids=all_ids, documents=all_chunks, metadatas=all_metadatas)
    return len(all_chunks), errors
