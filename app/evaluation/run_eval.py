"""RAG evaluation runner.

Executes benchmark dataset against RAG pipeline, computes metrics,
and exports comprehensive machine-readable evaluation reports.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.evaluation.metrics import aggregate_metrics, evaluate_sample
from app.generation.answer import LLMGenerator
from app.ingestion.chunking import RecursiveTextChunker
from app.ingestion.loaders import DirectoryLoader, load_document
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import get_vector_store


def ensure_index_populated(retriever: Retriever) -> None:
    """Ensure vector store has documents indexed if empty."""
    if retriever.vector_store.count() > 0:
        return

    settings = get_settings()
    raw_dir = settings.raw_data_dir
    docs = []

    if raw_dir.exists():
        docs = DirectoryLoader(raw_dir).load()

    if not docs:
        # Fallback default policy if raw_dir has no files
        from app.ingestion.loaders import Document
        docs = [
            Document(
                id="hr-policy-v3",
                content=(
                    "Acme Corp HR Policy v3. "
                    "Employees must submit the completed form within 10 business days of the incident occurring. "
                    "Core working hours are between 10:00 AM and 4:00 PM local time. "
                    "The company provides an annual health and wellness stipend of $1,200 per employee. "
                    "Reimbursement receipts must be submitted within 30 days of purchase through the expense portal. "
                    "Unused PTO up to a maximum of five days may be rolled over into the subsequent calendar year. "
                    "Lost or stolen devices must be reported to the IT Security Desk within 2 hours of discovery."
                ),
                metadata={"source": "hr_policy.txt"},
            )
        ]

    chunker = RecursiveTextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    retriever.index_chunks(chunks)


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    regress: bool = False,
) -> Dict[str, Any]:
    """Execute evaluation harness across test dataset."""
    settings = get_settings()
    retriever = Retriever()
    ensure_index_populated(retriever)
    generator = LLMGenerator()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    # Read JSONL cases
    cases: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if not cases:
        raise ValueError(f"No evaluation cases found in {dataset_path}")

    sample_results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for case in cases:
        question = case.get("question", "")
        start_t = time.perf_counter()

        if regress:
            # Deliberately regressed pipeline for CI gate failure demonstration:
            # empty retrieval (top_k=0) and forced hallucinated answer!
            retrieved_chunks = []
            retrieval_ms = 0.0
            gen_res = generator.generate(question, [])
            # Inject ungrounded claims to trigger hallucination failure
            answer = "The employee has unlimited days and there are no requirements whatsoever."
        else:
            retrieved_chunks, retrieval_ms = retriever.retrieve(question, top_k=settings.top_k)
            gen_res = generator.generate(question, retrieved_chunks)
            answer = gen_res.answer

        total_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(total_ms)

        sample_eval = evaluate_sample(
            case=case,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
        )
        sample_eval["retrieval_latency_ms"] = round(retrieval_ms, 2)
        sample_eval["total_latency_ms"] = round(total_ms, 2)
        sample_results.append(sample_eval)

    # Compute aggregate metrics
    aggregated = aggregate_metrics(sample_results)
    sorted_latencies = sorted(latencies)
    p50_lat = sorted_latencies[int(len(sorted_latencies) * 0.50)]
    p95_lat = sorted_latencies[min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)]

    aggregated["latency_p50_ms"] = round(p50_lat, 2)
    aggregated["latency_p95_ms"] = round(p95_lat, 2)

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_file": str(dataset_path),
            "total_cases": len(cases),
            "regressed_mode": regress,
            "pipeline_config": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k": 0 if regress else settings.top_k,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model_name,
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
            },
        },
        "metrics": aggregated,
        "sample_breakdown": sample_results,
    }

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG benchmark evaluation.")
    parser.add_argument("--dataset", default="data/eval/test.jsonl", help="Path to evaluation JSONL dataset")
    parser.add_argument("--output", default="reports/eval.json", help="Path to write evaluation report")
    parser.add_argument("--regress", action="store_true", help="Deliberately simulate a regressed pipeline to test CI failure")
    args = parser.parse_args()

    dataset_p = Path(args.dataset)
    output_p = Path(args.output)

    print(f"Executing RAG benchmark on {dataset_p} (regress={args.regress})...")
    report = run_evaluation(dataset_p, output_p, regress=args.regress)

    metrics = report["metrics"]
    print("\n" + "=" * 60)
    print(" " * 18 + "RAG EVALUATION REPORT")
    print("=" * 60)
    print(f" Context Recall:       {metrics.get('context_recall', 0):.4f}")
    print(f" Context Precision:    {metrics.get('context_precision', 0):.4f}")
    print(f" Faithfulness:         {metrics.get('faithfulness', 0):.4f}")
    print(f" Hallucination Rate:   {metrics.get('hallucination_rate', 0):.4f}")
    print(f" Answer Relevancy:     {metrics.get('answer_relevancy', 0):.4f}")
    print(f" Latency (p50 / p95):  {metrics.get('latency_p50_ms', 0):.1f}ms / {metrics.get('latency_p95_ms', 0):.1f}ms")
    print("=" * 60)
    print(f"Report exported to: {output_p}\n")


if __name__ == "__main__":
    main()
