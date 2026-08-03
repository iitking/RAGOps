"""Unit and integration tests for API endpoints and quality gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.api import (
    IngestRequest,
    QueryRequest,
    app,
    health_check,
    ingest_documents,
    operational_metrics,
    query_rag,
)
from app.feedback.routes import FeedbackRequest, submit_feedback
from app.evaluation.quality_gates import check_quality_gates
from app.observability.tracing import get_trace_collector


class TestAPI(unittest.TestCase):
    """Test API functionality and quality gates enforcement."""

    def setUp(self) -> None:
        self.collector = get_trace_collector()

    def test_health_check(self) -> None:
        res = health_check()
        self.assertEqual(res["status"], "healthy")
        self.assertIn("vector_store_type", res)
        self.assertIn("indexed_chunks", res)

    def test_ingest_and_query_flow(self) -> None:
        # Ingest custom content
        ingest_payload = IngestRequest(
            document_id="doc-security-1",
            content="All employees must report security incidents within 2 hours. Passwords must be at least 16 characters.",
        )
        ingest_res = ingest_documents(ingest_payload)
        self.assertEqual(ingest_res["status"], "success")
        self.assertGreater(ingest_res["chunks_indexed"], 0)

        # Query the ingested content
        query_payload = QueryRequest(
            question="Within how many hours must employees report security incidents?",
            top_k=2,
        )
        query_res = query_rag(query_payload)

        self.assertIsNotNone(query_res.answer)
        self.assertIn("2 hours", query_res.answer)
        self.assertGreater(len(query_res.citations), 0)
        self.assertIsNotNone(query_res.trace_id)
        self.assertGreater(query_res.total_latency_ms, 0.0)

        # Submit feedback for the query
        fb_payload = FeedbackRequest(
            request_id=query_res.trace_id,
            rating=1,
            comment="Accurate and fast response",
        )
        fb_res = submit_feedback(fb_payload)
        self.assertEqual(fb_res["status"], "recorded")
        self.assertTrue(fb_res["trace_found"])

        # Check metrics endpoint
        metrics = operational_metrics()
        self.assertGreaterEqual(metrics["total_requests"], 1)
        self.assertGreaterEqual(metrics["positive_feedback_count"], 1)

    def test_quality_gates_pass(self) -> None:
        passing_metrics = {
            "context_recall": 0.95,
            "context_precision": 0.85,
            "faithfulness": 0.95,
            "hallucination_rate": 0.01,
            "answer_relevancy": 0.90,
        }
        passed, failures, checks = check_quality_gates(passing_metrics)
        self.assertTrue(passed)
        self.assertEqual(len(failures), 0)

    def test_quality_gates_fail_on_regression(self) -> None:
        # Regressed metrics violating quality gates
        regressed_metrics = {
            "context_recall": 0.60,       # Below 0.90 gate
            "context_precision": 0.50,    # Below 0.70 gate
            "faithfulness": 0.50,         # Below 0.85 gate
            "hallucination_rate": 0.25,   # Above 0.02 gate
            "answer_relevancy": 0.65,     # Below 0.80 gate
        }
        passed, failures, checks = check_quality_gates(regressed_metrics)
        self.assertFalse(passed)
        self.assertGreater(len(failures), 0)
        # Check specific failure messages
        reasons = " ".join(failures)
        self.assertIn("Context Recall", reasons)
        self.assertIn("Hallucination Rate", reasons)


if __name__ == "__main__":
    unittest.main()
