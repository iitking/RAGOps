"""Evaluation metrics for RAGOps: Context Recall, Context Precision,

Faithfulness, Answer Relevancy, Retrieval Recall@k, MRR, and Hallucination Rate.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set

from app.retrieval.retriever import RetrieverResult


def _tokenize(text: str) -> Set[str]:
    """Extract clean lowercased alphanumeric tokens, excluding common stopwords."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "is", "are", "was", "were", "be", "been", "this", "that",
        "it", "by", "as", "from", "at", "has", "have", "had", "do", "does",
        "how", "what", "where", "when", "why", "who", "which", "long", "much", "many",
    }
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - stopwords


def _token_overlap_ratio(text_a: str, text_b: str) -> float:
    """Calculate token intersection ratio relative to text_a with subword prefix support."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a:
        return 1.0 if not tokens_b else 0.0

    hits = 0
    for ta in tokens_a:
        # Exact match or prefix match (e.g., submit vs submitted)
        if any(ta == tb or (len(ta) >= 4 and len(tb) >= 4 and (ta.startswith(tb[:4]) or tb.startswith(ta[:4]))) for tb in tokens_b):
            hits += 1

    return hits / len(tokens_a)


def compute_context_recall(
    retrieved_contexts: List[str],
    reference_contexts: List[str],
) -> float:
    """Compute Context Recall: fraction of reference evidence found in retrieved chunks."""
    if not reference_contexts:
        # For explicitly unanswerable test cases
        return 1.0

    combined_retrieved = " ".join(retrieved_contexts)
    if not combined_retrieved.strip():
        return 0.0

    scores = []
    for ref in reference_contexts:
        if not ref.strip():
            continue
        ratio = _token_overlap_ratio(ref, combined_retrieved)
        scores.append(min(1.0, ratio * 1.2))  # allow minor syntactic variation

    return sum(scores) / len(scores) if scores else 1.0


def compute_context_precision(
    retrieved_contexts: List[str],
    reference_contexts: List[str],
) -> float:
    """Compute Context Precision (Mean Average Precision over ranked retrieved chunks)."""
    if not retrieved_contexts:
        return 0.0
    if not reference_contexts:
        return 1.0

    combined_reference = " ".join(reference_contexts)
    relevant_hits = 0
    precision_sum = 0.0

    for rank, context in enumerate(retrieved_contexts, 1):
        ratio = _token_overlap_ratio(context, combined_reference)
        # Threshold for relevance
        is_relevant = ratio >= 0.20
        if is_relevant:
            relevant_hits += 1
            precision_at_k = relevant_hits / rank
            precision_sum += precision_at_k

    if relevant_hits == 0:
        return 0.0

    return precision_sum / relevant_hits


def compute_faithfulness(
    answer: str,
    retrieved_contexts: List[str],
) -> float:
    """Compute Faithfulness: proportion of claims in answer supported by context."""
    if not answer.strip():
        return 0.0

    # If the answer acknowledges lack of information, it is completely faithful
    refusal_markers = [
        "not have enough information",
        "not enough information",
        "insufficient information",
        "do not have enough",
        "no information",
    ]
    if any(m in answer.lower() for m in refusal_markers):
        return 1.0

    combined_context = " ".join(retrieved_contexts).lower()
    if not combined_context.strip():
        return 0.0

    # Extract sentences/claims from answer (stripping citations)
    clean_ans = re.sub(r"\[Doc:[^\]]+\]", "", answer).strip()
    claims = [c.strip() for c in re.split(r"(?<=[.!?])\s+", clean_ans) if len(c.strip()) > 10]

    if not claims:
        return 1.0

    supported_count = 0
    for claim in claims:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            continue
        context_tokens = _tokenize(combined_context)
        overlap = len(claim_tokens.intersection(context_tokens))
        ratio = overlap / len(claim_tokens)
        if ratio >= 0.50:
            supported_count += 1

    return supported_count / len(claims) if claims else 1.0


def compute_hallucination_rate(faithfulness: float) -> float:
    """Hallucination rate computed as the unsupported claim proportion (1 - faithfulness)."""
    return max(0.0, round(1.0 - faithfulness, 4))


def compute_answer_relevancy(question: str, answer: str) -> float:
    """Compute Answer Relevancy: semantic relevance of the answer to question."""
    if not answer.strip():
        return 0.0

    refusal_markers = [
        "not have enough information",
        "not enough information",
        "insufficient information",
        "do not have enough",
    ]
    if any(m in answer.lower() for m in refusal_markers):
        return 0.90  # Appropriate refusal score

    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)

    if not q_tokens:
        return 1.0

    hits = 0
    for qt in q_tokens:
        if any(qt == at or (len(qt) >= 4 and len(at) >= 4 and (qt.startswith(at[:4]) or at.startswith(qt[:4]))) for at in a_tokens):
            hits += 1

    ratio = hits / len(q_tokens)
    # Scale score into [0.0, 1.0] range
    return min(1.0, max(0.0, ratio * 1.25))


def compute_retrieval_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 3,
) -> float:
    """Recall@k retrieval metric based on ground truth document or chunk IDs."""
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """Mean Reciprocal Rank (MRR) for first relevant hit."""
    if not relevant_ids:
        return 1.0
    target_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in target_set:
            return 1.0 / rank
    return 0.0


def evaluate_sample(
    case: Dict[str, Any],
    retrieved_chunks: List[RetrieverResult],
    answer: str,
) -> Dict[str, Any]:
    """Evaluate a single test case and return individual metric scores."""
    reference_contexts = case.get("reference_contexts", [])
    retrieved_contexts = [r.text for r in retrieved_chunks]
    retrieved_ids = [r.chunk_id for r in retrieved_chunks]
    source_doc = case.get("source_document_id")
    relevant_ids = [source_doc] if source_doc else []

    c_recall = compute_context_recall(retrieved_contexts, reference_contexts)
    c_prec = compute_context_precision(retrieved_contexts, reference_contexts)
    faith = compute_faithfulness(answer, retrieved_contexts)
    hallucination = compute_hallucination_rate(faith)
    relevancy = compute_answer_relevancy(case.get("question", ""), answer)

    return {
        "case_id": case.get("id"),
        "question": case.get("question"),
        "difficulty": case.get("difficulty", "standard"),
        "context_recall": round(c_recall, 4),
        "context_precision": round(c_prec, 4),
        "faithfulness": round(faith, 4),
        "hallucination_rate": round(hallucination, 4),
        "answer_relevancy": round(relevancy, 4),
        "retrieved_count": len(retrieved_chunks),
        "answer": answer,
    }


def aggregate_metrics(sample_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute dataset-level mean for each metric."""
    if not sample_results:
        return {
            "context_recall": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "hallucination_rate": 0.0,
            "answer_relevancy": 0.0,
            "total_samples": 0.0,
        }

    keys = ["context_recall", "context_precision", "faithfulness", "hallucination_rate", "answer_relevancy"]
    aggregated: Dict[str, float] = {}

    for k in keys:
        vals = [r[k] for r in sample_results if k in r]
        aggregated[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

    aggregated["total_samples"] = len(sample_results)
    return aggregated
