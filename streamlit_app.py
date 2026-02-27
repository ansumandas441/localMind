"""Streamlit UI for localMind: upload documents, embed, and chat with RAG."""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from localmind.ingest import get_chroma_client_and_collection, ingest_paths, ingest_files_in_memory
from localmind.rag import ask


def main():
    st.set_page_config(page_title="localMind", page_icon="🧠", layout="centered")
    st.title("localMind")
    st.caption("Local RAG with Ollama — your documents, your machine.")

    # Sidebar: documents and embed
    with st.sidebar:
        st.header("Documents")
        st.markdown("Add PDFs or text files, then click **Embed** to index them.")

        # File uploader
        uploaded = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        # Optional: folder path (user types path)
        folder_path = st.text_input(
            "Or folder path to scan (optional)",
            placeholder="/path/to/documents",
            help="Absolute or relative path to a folder containing PDF/txt/md files.",
        )

        embed_clicked = st.button("Embed documents", type="primary")

        if embed_clicked:
            chunks_added = 0
            errors = []
            if uploaded:
                file_contents = []
                names = []
                for f in uploaded:
                    raw = f.read()
                    ext = Path(f.name).suffix.lower()
                    file_contents.append((f.name, raw, ext))
                    names.append(f.name)
                if file_contents:
                    n, errs = ingest_files_in_memory(file_contents, names)
                    chunks_added += n
                    errors.extend(errs)
            if folder_path:
                p = Path(folder_path).expanduser().resolve()
                n, errs = ingest_paths([p])
                chunks_added += n
                errors.extend(errs)
            if not uploaded and not folder_path:
                st.warning("Upload files or enter a folder path, then click Embed.")
            else:
                if chunks_added:
                    st.success(f"Indexed **{chunks_added}** chunks.")
                for e in errors:
                    st.error(e)

        st.divider()
        # Show collection count if available
        try:
            _, col = get_chroma_client_and_collection()
            count = col.count()
            st.metric("Chunks in index", count)
        except Exception:
            st.metric("Chunks in index", 0)

    # Main: chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for (text, meta, _) in msg["sources"]:
                        src = meta.get("source", "?")
                        page = meta.get("page")
                        label = f"{src}" + (f" (page {page})" if page else "")
                        st.caption(label)
                        st.text(text[:300] + ("..." if len(text) > 300 else ""))

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = ask(prompt)
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for (text, meta, _) in sources:
                                src = meta.get("source", "?")
                                page = meta.get("page")
                                label = f"{src}" + (f" (page {page})" if page else "")
                                st.caption(label)
                                st.text(text[:300] + ("..." if len(text) > 300 else ""))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(str(e))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                        "sources": None,
                    })


if __name__ == "__main__":
    main()
