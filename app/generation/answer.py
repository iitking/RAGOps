"""LLM answer generation with citation extraction and token tracking."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.generation.prompts import PROMPT_VERSION, assemble_rag_prompt
from app.retrieval.retriever import RetrieverResult


@dataclass
class Citation:
    """Represents a cited source in the generated answer."""
    document_id: str
    chunk_id: str
    source: str
    section: str
    text_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.section,
            "text_snippet": self.text_snippet,
        }


@dataclass
class GenerationResult:
    """Contains generated answer, citations, and operational metadata."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    model: str = "mock-generator"
    prompt_version: str = PROMPT_VERSION
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "model": self.model,
            "prompt_version": self.prompt_version,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "generation_latency_ms": self.generation_latency_ms,
        }


def _estimate_tokens(text: str) -> int:
    """Fast approximation of token count (~4 characters per token)."""
    return max(1, len(text) // 4)


class LLMGenerator:
    """Generates grounded responses with source citations."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.base_url = settings.llm_base_url

    def generate(
        self,
        question: str,
        results: List[RetrieverResult],
    ) -> GenerationResult:
        start_time = time.perf_counter()
        sys_prompt, user_prompt = assemble_rag_prompt(question, results)
        prompt_tokens = _estimate_tokens(sys_prompt + user_prompt)

        # Build citations from retrieved results
        citations: List[Citation] = []
        for r in results:
            citations.append(
                Citation(
                    document_id=r.document_id,
                    chunk_id=r.chunk_id,
                    source=r.metadata.get("source", r.document_id),
                    section=r.metadata.get("section", "general"),
                    text_snippet=r.text[:150] + ("..." if len(r.text) > 150 else ""),
                )
            )

        if self.provider == "openai" and self.api_key:
            answer = self._call_openai(sys_prompt, user_prompt)
        elif self.provider == "anthropic" and self.api_key:
            answer = self._call_anthropic(sys_prompt, user_prompt)
        else:
            answer = self._generate_mock(question, results)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        completion_tokens = _estimate_tokens(answer)

        return GenerationResult(
            answer=answer,
            citations=citations if "not enough information" not in answer.lower() else [],
            model=self.model if self.api_key else f"mock-{self.model}",
            prompt_version=PROMPT_VERSION,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            generation_latency_ms=latency_ms,
        )

    def _call_openai(self, sys_prompt: str, user_prompt: str) -> str:
        import httpx
        url = self.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_anthropic(self, sys_prompt: str, user_prompt: str) -> str:
        import httpx
        url = self.base_url or "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    def _generate_mock(self, question: str, results: List[RetrieverResult]) -> str:
        """Deterministic rule-grounded answer generation for offline CI and testing."""
        if not results:
            return "Based on the provided documents, I do not have enough information to answer this question."

        q_lower = question.lower()
        q_words = set(re.findall(r"[a-z0-9]+", q_lower)) - {
            "what", "when", "how", "why", "where", "is", "are", "the", "a", "an", "to", "for", "in", "of", "do", "does"
        }

        # Check if any retrieved chunk shares substantial query tokens
        best_match = None
        best_overlap = 0

        for r in results:
            text_lower = r.text.lower()
            overlap = sum(1 for w in q_words if w in text_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = r

        # Threshold: if less than 2 keywords match or no overlap, declare insufficient info
        if best_overlap < 2 or not best_match:
            return "Based on the provided documents, I do not have enough information to answer this question."

        # Clean and rank sentences by query keyword overlap
        cleaned_text = re.sub(r"^#+\s+[^\n]+\n?", "", best_match.text, flags=re.MULTILINE)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_text) if s.strip()]

        def score_sentence(s: str) -> int:
            s_low = s.lower()
            return sum(1 for w in q_words if (w in s_low or any(w.startswith(tok[:4]) for tok in s_low.split() if len(tok) >= 4)))

        # Sort sentences descending by match score
        ranked_sentences = sorted(sentences, key=score_sentence, reverse=True)
        selected = [s for s in ranked_sentences if score_sentence(s) > 0][:2]
        if not selected:
            selected = sentences[:2]

        section = best_match.metadata.get("section", "Section")
        doc_id = best_match.document_id
        grounded_text = " ".join(selected)

        return f"{grounded_text} [Doc: {doc_id}, Section: {section}]"
