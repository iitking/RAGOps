"""Chunking strategies for RAGOps ingestion pipeline.

Implements hierarchical recursive text splitting while preserving source spans,
document metadata, and section headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import Document


@dataclass
class Chunk:
    """Represents a discrete text chunk with full lineage metadata."""
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "metadata": self.metadata,
        }


class RecursiveTextChunker:
    """Recursive chunker that splits along natural boundaries:

    1. Section headers (Markdown headers #, ##, etc.)
    2. Paragraphs (\n\n)
    3. Sentences (. , ! , ? )
    4. Words (space)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a Document into Chunks with metadata and character spans."""
        cleaned = clean_text(document.content)
        if not cleaned:
            return []

        # Identify sections if present
        sections = self._split_by_sections(cleaned)
        chunks: List[Chunk] = []
        chunk_idx = 0

        for section_title, section_text, section_start in sections:
            section_chunks = self._chunk_text(
                text=section_text,
                base_offset=section_start,
                doc_id=document.id,
                source=document.metadata.get("source", document.id),
                section_title=section_title,
                start_index=chunk_idx,
            )
            chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)

        # Update total_chunks in metadata for each chunk
        total = len(chunks)
        for c in chunks:
            c.metadata["total_chunks"] = total

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split a list of documents into chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _split_by_sections(self, text: str) -> List[tuple[str, str, int]]:
        """Split text by Markdown section headings if available, retaining offsets."""
        header_pattern = re.compile(r"(^|\n)(#{1,4}\s+[^\n]+)", re.MULTILINE)
        matches = list(header_pattern.finditer(text))

        if not matches:
            return [("general", text, 0)]

        sections: List[tuple[str, str, int]] = []

        # Preamble before first header
        first_start = matches[0].start()
        if first_start > 0:
            preamble = text[:first_start].strip()
            if preamble:
                sections.append(("preamble", preamble, 0))

        for i, match in enumerate(matches):
            header_text = match.group(2).strip().lstrip("#").strip()
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_content = text[start_pos:end_pos].strip()
            if section_content:
                sections.append((header_text, section_content, start_pos))

        return sections

    def _chunk_text(
        self,
        text: str,
        base_offset: int,
        doc_id: str,
        source: str,
        section_title: str,
        start_index: int,
    ) -> List[Chunk]:
        """Splits arbitrary text into chunks respecting chunk_size and chunk_overlap."""
        if len(text) <= self.chunk_size:
            chunk_id = f"{doc_id}-chunk-{start_index}"
            return [
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=text,
                    metadata={
                        "source": source,
                        "section": section_title,
                        "chunk_index": start_index,
                        "start_char": base_offset,
                        "end_char": base_offset + len(text),
                    },
                )
            ]

        # Break text recursively
        step = self.chunk_size - self.chunk_overlap
        chunks: List[Chunk] = []
        pos = 0
        idx = start_index

        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            chunk_text = text[pos:end]

            # Try to snap to sentence or paragraph boundary if not at end of text
            if end < len(text):
                last_para = chunk_text.rfind("\n\n")
                last_period = chunk_text.rfind(". ")
                if last_para > int(self.chunk_size * 0.5):
                    end = pos + last_para + 2
                    chunk_text = text[pos:end]
                elif last_period > int(self.chunk_size * 0.5):
                    end = pos + last_period + 2
                    chunk_text = text[pos:end]

            clean_chunk = chunk_text.strip()
            if clean_chunk:
                chunk_id = f"{doc_id}-chunk-{idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        text=clean_chunk,
                        metadata={
                            "source": source,
                            "section": section_title,
                            "chunk_index": idx,
                            "start_char": base_offset + pos,
                            "end_char": base_offset + end,
                        },
                    )
                )
                idx += 1

            if end >= len(text):
                break
            pos += step

        return chunks
