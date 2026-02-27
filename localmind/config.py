"""Configuration for localMind RAG. Uses env vars with defaults."""
import os
from pathlib import Path

# Paths (env or defaults)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LOCALMIND_DATA_DIR", str(BASE_DIR / "data")))
CHROMA_PATH = Path(os.environ.get("LOCALMIND_CHROMA_PATH", str(DATA_DIR / "chroma")))
DOCUMENTS_DIR = Path(os.environ.get("LOCALMIND_DOCUMENTS_DIR", str(DATA_DIR / "documents")))

# Models
EMBEDDING_MODEL = os.environ.get("LOCALMIND_EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("LOCALMIND_CHAT_MODEL", "llama3.1:8b")

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Chunking
CHUNK_SIZE = int(os.environ.get("LOCALMIND_CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("LOCALMIND_CHUNK_OVERLAP", "200"))

# Retrieval
TOP_K = int(os.environ.get("LOCALMIND_TOP_K", "5"))

# Chroma collection name
COLLECTION_NAME = os.environ.get("LOCALMIND_COLLECTION", "localmind_docs")
