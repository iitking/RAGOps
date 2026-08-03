"""Observability, request tracing, and feedback linkage."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings


@dataclass
class TraceEvent:
    """Structured trace capturing end-to-end execution of a RAG query."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    endpoint: str = "/query"
    query: str = ""
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    model: str = "mock-generator"
    prompt_version: str = "v1.0"
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    feedback: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceCollector:
    """In-memory and file-persisted trace collector with optional Langfuse sync."""

    def __init__(self, log_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.log_path = log_path or settings.local_traces_path
        self._traces: Dict[str, TraceEvent] = {}
        self.langfuse_client = None

        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse
                self.langfuse_client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Langfuse: {e}")

    def record_trace(self, event: TraceEvent) -> None:
        """Record trace event to in-memory store and JSONL trace log."""
        self._traces[event.trace_id] = event

        # Ensure directory exists
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write trace to {self.log_path}: {e}")

        # Send to Langfuse if configured
        if self.langfuse_client:
            try:
                trace = self.langfuse_client.trace(
                    id=event.trace_id,
                    name=event.endpoint,
                    input={"query": event.query},
                    output={"answer": event.answer, "citations": event.citations},
                    metadata={
                        "model": event.model,
                        "prompt_version": event.prompt_version,
                        "retrieval_latency_ms": event.retrieval_latency_ms,
                        "generation_latency_ms": event.generation_latency_ms,
                    },
                )
                trace.generation(
                    name="answer_generation",
                    model=event.model,
                    usage={
                        "prompt_tokens": event.prompt_tokens,
                        "completion_tokens": event.completion_tokens,
                        "total_tokens": event.total_tokens,
                    },
                )
            except Exception as e:
                print(f"Warning: Failed to send trace to Langfuse: {e}")

    def add_feedback(
        self,
        trace_id: str,
        rating: int,
        comment: Optional[str] = None,
    ) -> bool:
        """Attach user feedback to an existing trace."""
        fb_dict = {
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if trace_id in self._traces:
            self._traces[trace_id].feedback = fb_dict

        # Forward to Langfuse
        if self.langfuse_client:
            try:
                self.langfuse_client.score(
                    trace_id=trace_id,
                    name="user_feedback",
                    value=rating,
                    comment=comment,
                )
            except Exception as e:
                print(f"Warning: Failed to send feedback to Langfuse: {e}")

        return trace_id in self._traces

    def get_trace(self, trace_id: str) -> Optional[TraceEvent]:
        return self._traces.get(trace_id)

    def get_all_traces(self) -> List[TraceEvent]:
        return list(self._traces.values())

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Compute operational metrics: latency percentiles, feedback, and tokens."""
        traces = list(self._traces.values())
        count = len(traces)
        if count == 0:
            return {
                "total_requests": 0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "total_tokens": 0,
                "positive_feedback_count": 0,
                "negative_feedback_count": 0,
                "feedback_ratio": 0.0,
            }

        latencies = sorted(t.total_latency_ms for t in traces)
        p50 = latencies[int(count * 0.50)]
        p95 = latencies[min(int(count * 0.95), count - 1)]
        tokens = sum(t.total_tokens for t in traces)

        pos_fb = sum(1 for t in traces if t.feedback and t.feedback.get("rating", 0) > 0)
        neg_fb = sum(1 for t in traces if t.feedback and t.feedback.get("rating", 0) < 0)
        total_fb = pos_fb + neg_fb
        ratio = (pos_fb / total_fb) if total_fb > 0 else 1.0

        return {
            "total_requests": count,
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "total_tokens": tokens,
            "positive_feedback_count": pos_fb,
            "negative_feedback_count": neg_fb,
            "feedback_ratio": round(ratio, 3),
        }


_global_trace_collector = TraceCollector()


def get_trace_collector() -> TraceCollector:
    """Return the global trace collector instance."""
    return _global_trace_collector
