# localMind

Local RAG (Retrieval-Augmented Generation) using **Ollama** for the LLM and embeddings, **ChromaDB** for the vector store, and a **Streamlit** UI. All runs on your machine; no cloud or API keys.

## Prerequisites

1. **Ollama** — [install](https://ollama.com) and start it (e.g. `ollama serve`).
2. Pull the embedding and chat models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.1:8b
   ```
   You can use other chat models (e.g. `mistral`, `mistral-nemo:12b`) and set `LOCALMIND_CHAT_MODEL` (see Configuration).

## Install

```bash
cd /path/to/localMind
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run streamlit_app.py
```

Then:

1. In the sidebar: **upload PDF/txt/md files** and/or enter a **folder path** to scan.
2. Click **Embed documents** to chunk and index them into ChromaDB.
3. In the main area, **ask questions**; answers are generated from your documents via Ollama.

## Configuration

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALMIND_DATA_DIR` | `./data` | Base data directory |
| `LOCALMIND_CHROMA_PATH` | `./data/chroma` | ChromaDB persistence path |
| `LOCALMIND_DOCUMENTS_DIR` | `./data/documents` | Default documents folder |
| `LOCALMIND_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LOCALMIND_CHAT_MODEL` | `llama3.1:8b` | Ollama chat model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LOCALMIND_CHUNK_SIZE` | `1024` | Max characters per chunk |
| `LOCALMIND_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `LOCALMIND_TOP_K` | `5` | Number of chunks to retrieve per query |
| `LOCALMIND_COLLECTION` | `localmind_docs` | ChromaDB collection name |

## Project layout

- `localmind/config.py` — configuration (paths, models, chunk/retrieval settings)
- `localmind/ingest.py` — load PDF/text, chunk, embed with Ollama, store in ChromaDB
- `localmind/rag.py` — embed query, retrieve from ChromaDB, generate answer with Ollama
- `streamlit_app.py` — Streamlit UI: upload, embed, chat
- `data/` — ChromaDB data (and optional documents folder)

## Ingest from CLI (optional)

You can ingest a folder from the command line:

```bash
python -c "
from pathlib import Path
from localmind.ingest import ingest_paths
n, errs = ingest_paths([Path('path/to/your/documents')])
print(f'Added {n} chunks. Errors: {errs}')
"
```

## License

Use and modify as you like.
