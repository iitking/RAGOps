"""Build and populate vector store index from raw document corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunking import RecursiveTextChunker
from app.ingestion.cleaning import deduplicate_chunks
from app.ingestion.loaders import DirectoryLoader, load_document
from app.retrieval.retriever import Retriever


def build_index(input_path: Path, chunk_size: int, chunk_overlap: int) -> int:
    """Load documents from input path, chunk, and index into vector store."""
    if input_path.is_dir():
        docs = DirectoryLoader(input_path).load()
    elif input_path.exists():
        docs = [load_document(input_path)]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not docs:
        print(f"No valid documents found in {input_path}")
        return 0

    print(f"Loaded {len(docs)} documents. Chunking with size={chunk_size}, overlap={chunk_overlap}...")
    chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    unique_chunks = deduplicate_chunks(chunks)
    print(f"Generated {len(unique_chunks)} unique chunks.")

    retriever = Retriever()
    indexed = retriever.index_chunks(unique_chunks)
    total_in_store = retriever.vector_store.count()
    print(f"Successfully indexed {indexed} chunks into vector store. Total records: {total_in_store}")
    return indexed


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index documents into RAGOps vector store.")
    parser.add_argument("--input", default=str(settings.raw_data_dir), help="Path to raw document file or folder")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap, help="Chunk overlap in characters")
    args = parser.parse_args()

    input_path = Path(args.input)
    build_index(input_path, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
