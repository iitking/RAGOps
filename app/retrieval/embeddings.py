"""Embedding generation for RAGOps.

Supports Sentence Transformers, OpenAI hosted embeddings, and a deterministic
hash-based dense embedding model for fast offline testing and CI.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from app.config import get_settings


class BaseEmbedding(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a float vector."""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into float vectors."""
        return [self.embed_text(t) for t in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class DeterministicHashEmbedding(BaseEmbedding):
    """Deterministic, lightweight dense embedding generator for offline testing and CI.

    Maps text tokens into an N-dimensional space using MD5 feature hashing with
    subword n-grams and L2 normalization. Retains semantic keyword similarity without
    requiring model weights or network access.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if not tokens:
            return vec

        # Token hashing
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign

            # Bigrams for local phrase structure
            if len(token) > 3:
                for j in range(len(token) - 2):
                    sub = token[j:j+3]
                    sh = int(hashlib.md5(sub.encode("utf-8")).hexdigest(), 16)
                    s_idx = sh % self._dim
                    s_sign = 0.5 if ((sh >> 8) & 1) else -0.5
                    vec[s_idx] += s_sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec


class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformers embedding model (e.g., all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            raise ImportError(
                f"Failed to load SentenceTransformer model '{model_name}': {e}. "
                "Ensure sentence-transformers is installed."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI hosted embedding model (e.g. text-embedding-3-small)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or get_settings().llm_api_key
        self._dim = 1536

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"model": self.model, "input": text}
            resp = httpx.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding request failed: {e}")


def get_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseEmbedding:
    """Factory function to instantiate the configured embedding model."""
    settings = get_settings()
    provider = provider or settings.embedding_provider
    model_name = model_name or settings.embedding_model_name

    if provider == "sentence-transformers":
        try:
            return SentenceTransformerEmbedding(model_name=model_name)
        except Exception:
            # Graceful fallback to deterministic hash model if library/weights not available
            return DeterministicHashEmbedding(dimension=settings.embedding_dimension)
    elif provider == "openai":
        return OpenAIEmbedding(model=model_name, api_key=settings.llm_api_key)
    else:
        return DeterministicHashEmbedding(dimension=settings.embedding_dimension)
