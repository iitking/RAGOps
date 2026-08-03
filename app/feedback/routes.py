"""User feedback collection endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional
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
    from fastapi import APIRouter, HTTPException, status
except ImportError:
    # Minimal fallback mock router for standalone operation
    class APIRouter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.routes: list = []
        def post(self, *args: Any, **kwargs: Any):
            def decorator(f): return f
            return decorator
        def get(self, *args: Any, **kwargs: Any):
            def decorator(f): return f
            return decorator

from app.observability.tracing import get_trace_collector


router = APIRouter(prefix="", tags=["Feedback"])


class FeedbackRequest(BaseModel):
    """Schema for thumbs up / thumbs down feedback."""
    request_id: str = Field(..., description="Trace/Request ID to associate with this feedback")
    rating: int = Field(..., ge=-1, le=1, description="Rating score: 1 for thumbs-up, -1 for thumbs-down")
    comment: Optional[str] = Field(default=None, max_length=1000, description="Optional qualitative feedback comment")


@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest) -> Dict[str, Any]:
    """Record user feedback tied to a specific request/trace ID."""
    collector = get_trace_collector()
    success = collector.add_feedback(
        trace_id=payload.request_id,
        rating=payload.rating,
        comment=payload.comment,
    )

    return {
        "status": "recorded",
        "request_id": payload.request_id,
        "rating": payload.rating,
        "trace_found": success,
    }


@router.get("/feedback/summary")
def get_feedback_summary() -> Dict[str, Any]:
    """Retrieve aggregate summary of collected feedback."""
    collector = get_trace_collector()
    return collector.get_metrics_summary()
