"""Unit tests for retrieval pipeline and evaluation metrics."""

from __future__ import annotations

import math
import unittest

from app.evaluation.metrics import (
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_hallucination_rate,
    compute_mrr,
    compute_retrieval_recall_at_k,
)
from app.ingestion.chunking import Chunk
from app.retrieval.embeddings import DeterministicHashEmbedding
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import InMemoryVectorStore, VectorRecord


class TestRetrieval(unittest.TestCase):
    """Test cases for embeddings, vector store, and retriever."""

    def test_embeddings_dimension_and_norm(self) -> None:
        model = DeterministicHashEmbedding(dimension=384)
        self.assertEqual(model.dimension, 384)

        vec = model.embed_text("Employees must submit the form within 10 business days.")
        self.assertEqual(len(vec), 384)

        norm = math.sqrt(sum(x * x for x in vec))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_in_memory_vector_store(self) -> None:
        store = InMemoryVectorStore()
        store.clear()

        r1 = VectorRecord(id="1", vector=[1.0, 0.0, 0.0], payload={"text": "Apple"})
        r2 = VectorRecord(id="2", vector=[0.0, 1.0, 0.0], payload={"text": "Banana"})
        r3 = VectorRecord(id="3", vector=[0.9, 0.1, 0.0], payload={"text": "Pear"})

        store.upsert([r1, r2, r3])
        self.assertEqual(store.count(), 3)

        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "1")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)
        self.assertEqual(results[1].id, "3")

        store.delete(["2"])
        self.assertEqual(store.count(), 2)

    def test_retriever_pipeline(self) -> None:
        store = InMemoryVectorStore()
        embed = DeterministicHashEmbedding(dimension=384)
        retriever = Retriever(embedding_model=embed, vector_store=store)

        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="doc-hr",
                text="The form must be submitted within 10 business days.",
                metadata={"section": "Grievance"},
            ),
            Chunk(
                chunk_id="c2",
                document_id="doc-hr",
                text="Employees receive a health and wellness stipend of $1200 per year.",
                metadata={"section": "Benefits"},
            ),
        ]
        retriever.index_chunks(chunks)

        hits, latency = retriever.retrieve("timeline to submit grievance form", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "c1")
        self.assertGreater(latency, 0.0)

    def test_evaluation_metrics_perfect_case(self) -> None:
        ref_context = ["Employees must submit the form within 10 business days."]
        ret_context = ["Employees must submit the form within 10 business days."]
        answer = "The form must be submitted within 10 business days."

        recall = compute_context_recall(ret_context, ref_context)
        precision = compute_context_precision(ret_context, ref_context)
        faith = compute_faithfulness(answer, ret_context)
        hallucination = compute_hallucination_rate(faith)
        relevancy = compute_answer_relevancy("How long to submit the form?", answer)

        self.assertGreaterEqual(recall, 0.90)
        self.assertGreaterEqual(precision, 0.80)
        self.assertGreaterEqual(faith, 0.90)
        self.assertLessEqual(hallucination, 0.02)
        self.assertGreaterEqual(relevancy, 0.80)

    def test_unanswerable_metrics(self) -> None:
        ref_context = []
        ret_context = ["General information about company equipment and facilities."]
        safe_answer = "Based on the provided documents, I do not have enough information to answer this question."

        recall = compute_context_recall(ret_context, ref_context)
        faith = compute_faithfulness(safe_answer, ret_context)
        hallucination = compute_hallucination_rate(faith)

        self.assertEqual(recall, 1.0)
        self.assertEqual(faith, 1.0)
        self.assertEqual(hallucination, 0.0)

    def test_retrieval_ranking_metrics(self) -> None:
        retrieved = ["doc-1", "doc-2", "doc-3", "doc-4"]
        relevant = ["doc-2"]

        recall_3 = compute_retrieval_recall_at_k(retrieved, relevant, k=3)
        recall_1 = compute_retrieval_recall_at_k(retrieved, relevant, k=1)
        mrr = compute_mrr(retrieved, relevant)

        self.assertEqual(recall_3, 1.0)
        self.assertEqual(recall_1, 0.0)
        self.assertEqual(mrr, 0.5)  # Rank 2 => 1/2 = 0.5


if __name__ == "__main__":
    unittest.main()
