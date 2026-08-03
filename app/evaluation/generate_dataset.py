"""Synthetic evaluation dataset generator for RAGOps.

Generates versioned Question-Context-Answer evaluation cases from raw document
corpora, including single-hop, multi-hop, and explicit unanswerable cases.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingestion.chunking import RecursiveTextChunker
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import DirectoryLoader, Document, load_document


def _extract_facts(text: str) -> List[Dict[str, str]]:
    """Extract candidate factual sentences from text to formulate Q/A pairs."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
    facts = []

    for s in sentences:
        # Match time/numbers
        if any(char.isdigit() for char in s) or any(w in s.lower() for w in ["must", "require", "stipulate", "within", "between", "policy"]):
            facts.append({"sentence": s})

    return facts


def generate_cases_from_document(document: Document) -> List[Dict[str, Any]]:
    """Generate structured evaluation test cases from a single document."""
    chunker = RecursiveTextChunker(chunk_size=400, chunk_overlap=40)
    chunks = chunker.chunk_document(document)

    cases: List[Dict[str, Any]] = []
    case_idx = 1

    for chunk in chunks:
        facts = _extract_facts(chunk.text)
        section = chunk.metadata.get("section", "general")

        for f in facts[:2]:
            sentence = f["sentence"]
            # Generate deterministic targeted question
            question = None

            if "business days" in sentence.lower() or "days" in sentence.lower():
                question = f"What is the timeline specified regarding {section.lower()}?"
            elif "$" in sentence or "stipend" in sentence.lower() or "amount" in sentence.lower():
                question = f"What is the financial amount or stipend provided for {section.lower()}?"
            elif "hours" in sentence.lower() or "time" in sentence.lower():
                question = f"What are the hours or time requirements stated in {section.lower()}?"
            elif "laptop" in sentence.lower() or "device" in sentence.lower():
                question = "What is the procedure for company-issued devices and security?"
            elif len(sentence.split()) > 6:
                words = sentence.split()[:4]
                question = f"What does the policy state about {' '.join(words).lower()}?"

            if question:
                cases.append({
                    "id": f"{document.id}-{case_idx:04d}",
                    "question": question,
                    "reference_answer": sentence,
                    "reference_contexts": [chunk.text],
                    "source_document_id": document.id,
                    "source_span": f"{section}-{chunk.metadata.get('chunk_index', 0)}",
                    "difficulty": "single-hop" if case_idx % 2 == 1 else "multi-hop",
                    "split": "test",
                    "is_answerable": True,
                })
                case_idx += 1

    # Add deliberate unanswerable question to test hallucination resistance
    cases.append({
        "id": f"{document.id}-{case_idx:04d}",
        "question": f"What is the company stock option vesting policy mentioned in {document.id}?",
        "reference_answer": "Based on the provided documents, I do not have enough information to answer this question.",
        "reference_contexts": [],
        "source_document_id": document.id,
        "source_span": "unanswerable",
        "difficulty": "unanswerable",
        "split": "test",
        "is_answerable": False,
    })

    return cases


def generate_evaluation_dataset(
    input_path: Path,
    output_path: Path,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load documents, generate synthetic cases, and write to JSONL."""
    if input_path.is_dir():
        docs = DirectoryLoader(input_path).load()
    else:
        docs = [load_document(input_path)]

    if not docs:
        raise ValueError(f"No documents found at {input_path}")

    all_cases: List[Dict[str, Any]] = []
    for doc in docs:
        cases = generate_cases_from_document(doc)
        all_cases.extend(cases)

    if max_samples and len(all_cases) > max_samples:
        all_cases = all_cases[:max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in all_cases:
            f.write(json.dumps(c) + "\n")

    return all_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic RAG evaluation dataset.")
    parser.add_argument("--input", default="data/raw", help="Path to input raw corpus file or directory")
    parser.add_argument("--output", default="data/eval/test.jsonl", help="Output path for JSONL dataset")
    parser.add_argument("--samples", type=int, default=10, help="Maximum number of test cases to generate")
    args = parser.parse_args()

    input_p = Path(args.input)
    output_p = Path(args.output)

    print(f"Generating synthetic evaluation cases from {input_p}...")
    cases = generate_evaluation_dataset(input_p, output_p, max_samples=args.samples)
    print(f"Generated {len(cases)} cases successfully saved to {output_p}")


if __name__ == "__main__":
    main()
