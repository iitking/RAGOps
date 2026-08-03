"""FastAPI application exposing ingestion, query, feedback, and evaluation endpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self) -> Dict[str, Any]:
            return self.__dict__
        def dict(self) -> Dict[str, Any]:
            return self.__dict__

    def Field(default: Any = None, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return default

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    # If FastAPI is not installed, mock classes to allow imports
    class FastAPI:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def include_router(self, *args: Any, **kwargs: Any) -> None: pass
        def add_middleware(self, *args: Any, **kwargs: Any) -> None: pass
        def post(self, *args: Any, **kwargs: Any):
            def d(f): return f
            return d
        def get(self, *args: Any, **kwargs: Any):
            def d(f): return f
            return d

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

from app.config import get_settings
from app.evaluation.run_eval import ensure_index_populated, run_evaluation
from app.feedback.routes import router as feedback_router
from app.generation.answer import Citation, GenerationResult, LLMGenerator
from app.ingestion.chunking import Chunk, RecursiveTextChunker
from app.ingestion.loaders import DirectoryLoader, Document, load_document
from app.observability.tracing import TraceEvent, get_trace_collector
from app.retrieval.retriever import Retriever


# Pydantic Request / Response models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question to answer")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override top-k retrieved chunks")


class CitationModel(BaseModel):
    document_id: str
    chunk_id: str
    source: str
    section: str
    text_snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationModel]
    trace_id: str
    model: str
    prompt_version: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class IngestRequest(BaseModel):
    path: Optional[str] = Field(default=None, description="Path to raw document file or folder")
    document_id: Optional[str] = Field(default=None, description="Explicit document ID")
    content: Optional[str] = Field(default=None, description="Inline raw content to index directly")


class EvaluateRequest(BaseModel):
    dataset_path: Optional[str] = Field(default="data/eval/test.jsonl")
    output_path: Optional[str] = Field(default="reports/eval.json")
    regress: bool = Field(default=False, description="Simulate regressed pipeline to test gate failure")


# App initialization
app = FastAPI(
    title="RAGOps API",
    version="0.1.0",
    description="Enterprise RAG with Automated Evaluation & Tracing",
)

try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

app.include_router(feedback_router)

# Lazy singletons
_retriever: Optional[Retriever] = None
_generator: Optional[LLMGenerator] = None


def get_pipeline() -> tuple[Retriever, LLMGenerator]:
    global _retriever, _generator
    if _retriever is None:
        _retriever = Retriever()
        ensure_index_populated(_retriever)
    if _generator is None:
        _generator = LLMGenerator()
    return _retriever, _generator


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health and readiness probe."""
    settings = get_settings()
    retriever, _ = get_pipeline()
    return {
        "status": "healthy",
        "vector_store_type": settings.vector_store_type,
        "indexed_chunks": retriever.vector_store.count(),
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.llm_provider,
    }


@app.get("/metrics", tags=["Observability"])
def operational_metrics() -> Dict[str, Any]:
    """Expose operational latency, token, and feedback metrics."""
    collector = get_trace_collector()
    return collector.get_metrics_summary()


@app.post("/ingest", tags=["Ingestion"])
def ingest_documents(payload: IngestRequest) -> Dict[str, Any]:
    """Ingest documents or raw text into the vector store."""
    settings = get_settings()
    retriever, _ = get_pipeline()
    chunker = RecursiveTextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    docs: List[Document] = []
    if payload.content:
        doc_id = payload.document_id or f"doc-custom-{int(time.time())}"
        docs.append(Document(id=doc_id, content=payload.content, metadata={"source": "api_payload"}))
    elif payload.path:
        target = Path(payload.path)
        if target.is_dir():
            docs = DirectoryLoader(target).load()
        elif target.exists():
            docs = [load_document(target)]
        else:
            raise HTTPException(status_code=404, detail=f"Target path not found: {payload.path}")
    else:
        # Default to configured raw_data_dir
        if settings.raw_data_dir.exists():
            docs = DirectoryLoader(settings.raw_data_dir).load()

    if not docs:
        return {"status": "no_documents_found", "indexed_chunks": 0}

    chunks = chunker.chunk_documents(docs)
    count = retriever.index_chunks(chunks)

    return {
        "status": "success",
        "documents_ingested": len(docs),
        "chunks_indexed": count,
        "total_store_count": retriever.vector_store.count(),
    }


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def query_rag(payload: QueryRequest) -> QueryResponse:
    """Run semantic retrieval, generate grounded answer with citations, and record trace."""
    start_total = time.perf_counter()
    retriever, generator = get_pipeline()

    # 1. Retrieve evidence chunks
    results, retrieval_latency = retriever.retrieve(payload.question, top_k=payload.top_k)

    # 2. Generate answer
    gen_result = generator.generate(payload.question, results)
    total_latency = (time.perf_counter() - start_total) * 1000.0

    # 3. Create and record trace event
    citations_data = [c.to_dict() for c in gen_result.citations]
    retrieved_data = [r.to_dict() for r in results]

    event = TraceEvent(
        endpoint="/query",
        query=payload.question,
        retrieved_chunks=retrieved_data,
        answer=gen_result.answer,
        citations=citations_data,
        model=gen_result.model,
        prompt_version=gen_result.prompt_version,
        retrieval_latency_ms=round(retrieval_latency, 2),
        generation_latency_ms=round(gen_result.generation_latency_ms, 2),
        total_latency_ms=round(total_latency, 2),
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
        total_tokens=gen_result.total_tokens,
    )

    collector = get_trace_collector()
    collector.record_trace(event)

    return QueryResponse(
        answer=gen_result.answer,
        citations=[
            CitationModel(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                source=c.source,
                section=c.section,
                text_snippet=c.text_snippet,
            )
            for c in gen_result.citations
        ],
        trace_id=event.trace_id,
        model=gen_result.model,
        prompt_version=gen_result.prompt_version,
        retrieval_latency_ms=round(retrieval_latency, 2),
        generation_latency_ms=round(gen_result.generation_latency_ms, 2),
        total_latency_ms=round(total_latency, 2),
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
        total_tokens=gen_result.total_tokens,
    )


@app.post("/evaluate", tags=["Evaluation"])
def trigger_evaluation(payload: EvaluateRequest) -> Dict[str, Any]:
    """Trigger an authorized evaluation job across test benchmark dataset."""
    d_path = Path(payload.dataset_path or "data/eval/test.jsonl")
    o_path = Path(payload.output_path or "reports/eval.json")

    report = run_evaluation(d_path, o_path, regress=payload.regress)
    return {
        "status": "completed",
        "report_path": str(o_path),
        "metrics": report["metrics"],
    }


def start() -> None:
    """Entry point for running uvicorn server."""
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.api:app", host=settings.host, port=settings.port, reload=settings.debug)


if __name__ == "__main__":
    start()
