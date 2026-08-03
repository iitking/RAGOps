"""Text cleaning, normalization, and deduplication for RAGOps."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, List, Set, TypeVar

T = TypeVar("T")


def clean_text(text: str) -> str:
    """Normalize and clean raw document text.

    - Normalizes Unicode characters via NFKC
    - Replaces weird whitespace (non-breaking spaces, zero-width spaces)
    - Removes unprintable control characters
    - Normalizes excessive newlines
    - Strips trailing/leading spaces from lines
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Replace zero-width spaces and non-breaking spaces
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")

    # Strip non-printable characters except \n and \t
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")

    # Strip trailing whitespace on each line and collapse multiple inline spaces
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def content_hash(text: str) -> str:
    """Compute a normalized SHA-256 hash of text."""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_chunks(chunks: List[Any]) -> List[Any]:
    """Deduplicate chunk objects based on their text content hash.

    Preserves the first occurrence of each unique chunk.
    """
    seen_hashes: Set[str] = set()
    unique_chunks: List[Any] = []

    for chunk in chunks:
        # Support chunk objects with .text or dicts with ['text']
        text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
        h = content_hash(text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_chunks.append(chunk)

    return unique_chunks
