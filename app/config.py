"""Configuration settings for RAGOps.

Supports loading from environment variables and .env files, with safe defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        app_env: str = Field(default="development", alias="APP_ENV")
        host: str = Field(default="0.0.0.0", alias="HOST")
        port: int = Field(default=8000, alias="PORT")
        debug: bool = Field(default=False, alias="DEBUG")

        # Paths
        data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
        raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
        eval_data_dir: Path = Field(default=Path("data/eval"), alias="EVAL_DATA_DIR")
        reports_dir: Path = Field(default=Path("reports"), alias="REPORTS_DIR")

        # Vector store: "inmemory" or "qdrant"
        vector_store_type: str = Field(default="inmemory", alias="VECTOR_STORE_TYPE")
        qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
        qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
        qdrant_collection: str = Field(default="ragops_documents", alias="QDRANT_COLLECTION")

        # Embeddings: "hash" (offline/test), "sentence-transformers", "openai"
        embedding_provider: str = Field(default="hash", alias="EMBEDDING_PROVIDER")
        embedding_model_name: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME")
        embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

        # LLM Generation: "mock" (deterministic offline), "openai", "anthropic"
        llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
        llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
        llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
        llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")

        # RAG pipeline parameters
        chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
        chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
        top_k: int = Field(default=3, alias="TOP_K")
        min_similarity: float = Field(default=0.0, alias="MIN_SIMILARITY")

        # CI Quality Gate Thresholds
        gate_min_context_recall: float = Field(default=0.90, alias="GATE_MIN_CONTEXT_RECALL")
        gate_min_context_precision: float = Field(default=0.70, alias="GATE_MIN_CONTEXT_PRECISION")
        gate_min_faithfulness: float = Field(default=0.85, alias="GATE_MIN_FAITHFULNESS")
        gate_min_answer_relevancy: float = Field(default=0.80, alias="GATE_MIN_ANSWER_RELEVANCY")
        gate_max_hallucination_rate: float = Field(default=0.02, alias="GATE_MAX_HALLUCINATION_RATE")

        # Observability / Tracing
        enable_local_tracing: bool = Field(default=True, alias="ENABLE_LOCAL_TRACING")
        local_traces_path: Path = Field(default=Path("reports/traces.jsonl"), alias="LOCAL_TRACES_PATH")
        langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
        langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

except ImportError:
    # Fallback when pydantic_settings is not installed
    class Settings:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.app_env = os.getenv("APP_ENV", "development")
            self.host = os.getenv("HOST", "0.0.0.0")
            self.port = int(os.getenv("PORT", "8000"))
            self.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

            self.data_dir = Path(os.getenv("DATA_DIR", "data"))
            self.raw_data_dir = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
            self.eval_data_dir = Path(os.getenv("EVAL_DATA_DIR", "data/eval"))
            self.reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))

            self.vector_store_type = os.getenv("VECTOR_STORE_TYPE", "inmemory")
            self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
            self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "ragops_documents")

            self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "hash")
            self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
            self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))

            self.llm_provider = os.getenv("LLM_PROVIDER", "mock")
            self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            self.llm_api_key = os.getenv("LLM_API_KEY", None)
            self.llm_base_url = os.getenv("LLM_BASE_URL", None)

            self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
            self.top_k = int(os.getenv("TOP_K", "3"))
            self.min_similarity = float(os.getenv("MIN_SIMILARITY", "0.0"))

            self.gate_min_context_recall = float(os.getenv("GATE_MIN_CONTEXT_RECALL", "0.90"))
            self.gate_min_context_precision = float(os.getenv("GATE_MIN_CONTEXT_PRECISION", "0.70"))
            self.gate_min_faithfulness = float(os.getenv("GATE_MIN_FAITHFULNESS", "0.85"))
            self.gate_min_answer_relevancy = float(os.getenv("GATE_MIN_ANSWER_RELEVANCY", "0.80"))
            self.gate_max_hallucination_rate = float(os.getenv("GATE_MAX_HALLUCINATION_RATE", "0.02"))

            self.enable_local_tracing = os.getenv("ENABLE_LOCAL_TRACING", "true").lower() in ("true", "1", "yes")
            self.local_traces_path = Path(os.getenv("LOCAL_TRACES_PATH", "reports/traces.jsonl"))
            self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", None)
            self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", None)
            self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
