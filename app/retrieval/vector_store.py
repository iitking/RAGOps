"""Vector store abstractions and implementations for RAGOps.

Supports Qdrant vector store and a lightweight In-Memory vector store.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings


@dataclass
class VectorRecord:
    """Record stored in the vector database."""
    id: str
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result returned from a vector similarity search."""
    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract interface for vector databases."""

    @abstractmethod
    def upsert(self, records: List[VectorRecord]) -> int:
        """Insert or update vector records. Returns count of upserted items."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search nearest neighbors by cosine similarity."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """Delete records by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all records from store."""
        pass


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class InMemoryVectorStore(BaseVectorStore):
    """Thread-safe, zero-dependency in-memory vector store."""

    def __init__(self) -> None:
        self._records: Dict[str, VectorRecord] = {}

    def upsert(self, records: List[VectorRecord]) -> int:
        for r in records:
            self._records[r.id] = r
        return len(records)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        scored: List[SearchResult] = []
        for r in self._records.values():
            sim = _cosine_similarity(query_vector, r.vector)
            if sim >= score_threshold:
                scored.append(SearchResult(id=r.id, score=sim, payload=r.payload))

        # Sort descending by similarity score
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def delete(self, ids: List[str]) -> int:
        count = 0
        for i in ids:
            if i in self._records:
                del self._records[i]
                count += 1
        return count

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector database wrapper."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "ragops_documents",
        vector_size: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            if url == ":memory:" or not url:
                self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(url=url, api_key=api_key)

            # Ensure collection exists
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}")

    def upsert(self, records: List[VectorRecord]) -> int:
        from qdrant_client.http import models

        points = [
            models.PointStruct(
                id=r.id if r.id.isdigit() else abs(hash(r.id)) % (2**63 - 1),
                vector=r.vector,
                payload={"orig_id": r.id, **r.payload},
            )
            for r in records
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(records)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0.0 else None,
        )
        return [
            SearchResult(
                id=hit.payload.get("orig_id", str(hit.id)),
                score=float(hit.score),
                payload=hit.payload or {},
            )
            for hit in hits
        ]

    def delete(self, ids: List[str]) -> int:
        from qdrant_client.http import models

        point_ids = [int(i) if i.isdigit() else abs(hash(i)) % (2**63 - 1) for i in ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )
        return len(ids)

    def count(self) -> int:
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count or 0

    def clear(self) -> None:
        from qdrant_client.http import models
        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )


_shared_in_memory_store = InMemoryVectorStore()


def get_vector_store() -> BaseVectorStore:
    """Return the configured vector store instance."""
    settings = get_settings()

    if settings.vector_store_type == "qdrant":
        try:
            return QdrantVectorStore(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection_name=settings.qdrant_collection,
                vector_size=settings.embedding_dimension,
            )
        except Exception:
            # Fallback to shared in-memory vector store
            return _shared_in_memory_store

    return _shared_in_memory_store
