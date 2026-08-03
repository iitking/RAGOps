"""CI/CD Quality Gate enforcement for RAGOps.

Reads machine-readable evaluation reports, verifies metrics against configured
quality gates, and blocks deployment/PR merge on quality regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.config import get_settings


def check_quality_gates(
    metrics: Dict[str, Any],
    min_context_recall: float = 0.90,
    min_context_precision: float = 0.70,
    min_faithfulness: float = 0.85,
    min_answer_relevancy: float = 0.80,
    max_hallucination_rate: float = 0.02,
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """Verify evaluation metrics against defined quality gate thresholds."""
    failures: List[str] = []
    checks: List[Dict[str, Any]] = []

    # 1. Context Recall
    actual_cr = metrics.get("context_recall", 0.0)
    cr_pass = actual_cr >= min_context_recall
    checks.append({
        "metric": "Context Recall",
        "actual": actual_cr,
        "operator": ">=",
        "threshold": min_context_recall,
        "status": "PASS" if cr_pass else "FAIL",
    })
    if not cr_pass:
        failures.append(f"Context Recall ({actual_cr:.4f}) is below minimum threshold ({min_context_recall:.4f})")

    # 2. Context Precision
    actual_cp = metrics.get("context_precision", 0.0)
    cp_pass = actual_cp >= min_context_precision
    checks.append({
        "metric": "Context Precision",
        "actual": actual_cp,
        "operator": ">=",
        "threshold": min_context_precision,
        "status": "PASS" if cp_pass else "FAIL",
    })
    if not cp_pass:
        failures.append(f"Context Precision ({actual_cp:.4f}) is below minimum threshold ({min_context_precision:.4f})")

    # 3. Faithfulness
    actual_f = metrics.get("faithfulness", 0.0)
    f_pass = actual_f >= min_faithfulness
    checks.append({
        "metric": "Faithfulness",
        "actual": actual_f,
        "operator": ">=",
        "threshold": min_faithfulness,
        "status": "PASS" if f_pass else "FAIL",
    })
    if not f_pass:
        failures.append(f"Faithfulness ({actual_f:.4f}) is below minimum threshold ({min_faithfulness:.4f})")

    # 4. Hallucination Rate
    actual_hr = metrics.get("hallucination_rate", 1.0)
    hr_pass = actual_hr <= max_hallucination_rate
    checks.append({
        "metric": "Hallucination Rate",
        "actual": actual_hr,
        "operator": "<=",
        "threshold": max_hallucination_rate,
        "status": "PASS" if hr_pass else "FAIL",
    })
    if not hr_pass:
        failures.append(f"Hallucination Rate ({actual_hr:.4f}) exceeds maximum threshold ({max_hallucination_rate:.4f})")

    # 5. Answer Relevancy
    actual_ar = metrics.get("answer_relevancy", 0.0)
    ar_pass = actual_ar >= min_answer_relevancy
    checks.append({
        "metric": "Answer Relevancy",
        "actual": actual_ar,
        "operator": ">=",
        "threshold": min_answer_relevancy,
        "status": "PASS" if ar_pass else "FAIL",
    })
    if not ar_pass:
        failures.append(f"Answer Relevancy ({actual_ar:.4f}) is below minimum threshold ({min_answer_relevancy:.4f})")

    passed = len(failures) == 0
    return passed, failures, checks


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Enforce CI quality gates on evaluation report.")
    parser.add_argument("--report", default="reports/eval.json", help="Path to evaluation report JSON")
    parser.add_argument("--min-recall", type=float, default=settings.gate_min_context_recall)
    parser.add_argument("--min-precision", type=float, default=settings.gate_min_context_precision)
    parser.add_argument("--min-faithfulness", type=float, default=settings.gate_min_faithfulness)
    parser.add_argument("--min-relevancy", type=float, default=settings.gate_min_answer_relevancy)
    parser.add_argument("--max-hallucination", type=float, default=settings.gate_max_hallucination_rate)
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: Evaluation report not found at {report_path}", file=sys.stderr)
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    passed, failures, checks = check_quality_gates(
        metrics=metrics,
        min_context_recall=args.min_recall,
        min_context_precision=args.min_precision,
        min_faithfulness=args.min_faithfulness,
        min_answer_relevancy=args.min_relevancy,
        max_hallucination_rate=args.max_hallucination,
    )

    print("\n" + "=" * 70)
    print(" " * 22 + "CI QUALITY GATES VERIFICATION")
    print("=" * 70)
    print(f" {'METRIC':<22} | {'ACTUAL':<10} | {'CRITERIA':<12} | {'STATUS':<8}")
    print("-" * 70)

    for c in checks:
        criteria_str = f"{c['operator']} {c['threshold']:.2f}"
        actual_str = f"{c['actual']:.4f}"
        status_color = c["status"]
        print(f" {c['metric']:<22} | {actual_str:<10} | {criteria_str:<12} | {status_color:<8}")

    print("=" * 70)

    if passed:
        print("\nSUCCESS: All quality gates PASSED! Pull request / build approved.\n")
        sys.exit(0)
    else:
        print("\nFAILURE: RAG evaluation failed quality gate checks:")
        for f in failures:
            print(f" - {f}")
        print("\nDeployment blocked by CI/CD policy.\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
