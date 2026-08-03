"""Retriever module connecting embeddings and vector store."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.ingestion.chunking import Chunk
from app.retrieval.embeddings import BaseEmbedding, get_embedding_model
from app.retrieval.vector_store import (
    BaseVectorStore,
    VectorRecord,
    get_vector_store,
)


@dataclass
class RetrieverResult:
    """Retrieved chunk with relevance score and metadata."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class Retriever:
    """Manages document chunk indexing and semantic query retrieval."""

    def __init__(
        self,
        embedding_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.settings = get_settings()
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or get_vector_store()

    def index_chunks(self, chunks: List[Chunk]) -> int:
        """Embed and upsert chunks into the vector store."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self.embedding_model.embed_batch(texts)

        records: List[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors):
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        **chunk.metadata,
                    },
                )
            )

        return self.vector_store.upsert(records)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> tuple[List[RetrieverResult], float]:
        """Retrieve top-k relevant chunks for a query.

        Returns tuple of (results, latency_ms).
        """
        start_time = time.perf_counter()

        k = top_k if top_k is not None else self.settings.top_k
        threshold = score_threshold if score_threshold is not None else self.settings.min_similarity

        query_vector = self.embedding_model.embed_text(query)
        hits = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
            score_threshold=threshold,
        )

        results: List[RetrieverResult] = []
        for hit in hits:
            payload = hit.payload
            results.append(
                RetrieverResult(
                    chunk_id=payload.get("chunk_id", hit.id),
                    document_id=payload.get("document_id", "unknown"),
                    text=payload.get("text", ""),
                    score=hit.score,
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return results, latency_ms
