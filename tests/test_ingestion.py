"""Unit tests for ingestion pipeline: loading, cleaning, and chunking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ingestion.chunking import Chunk, RecursiveTextChunker
from app.ingestion.cleaning import clean_text, deduplicate_chunks
from app.ingestion.loaders import Document, TextLoader, load_document


class TestIngestion(unittest.TestCase):
    """Test cases for document ingestion."""

    def test_clean_text_normalization(self) -> None:
        raw = "  Hello \u200bWorld!  \n\n\n\nThis is a \xa0test.\t\n  "
        cleaned = clean_text(raw)
        self.assertNotIn("\u200b", cleaned)
        self.assertNotIn("\xa0", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Hello World!", cleaned)
        self.assertIn("This is a test.", cleaned)

    def test_deduplicate_chunks(self) -> None:
        chunks = [
            Chunk(chunk_id="c1", document_id="d1", text="Duplicate content"),
            Chunk(chunk_id="c2", document_id="d1", text="Duplicate content"),
            Chunk(chunk_id="c3", document_id="d1", text="Unique content"),
        ]
        unique = deduplicate_chunks(chunks)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0].chunk_id, "c1")
        self.assertEqual(unique[1].chunk_id, "c3")

    def test_text_loader(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
            f.write("# Document ID: test-doc-01\nSample policy body text.")
            f_path = Path(f.name)

        try:
            doc = load_document(f_path)
            self.assertEqual(doc.id, "test-doc-01")
            self.assertIn("Sample policy body text.", doc.content)
            self.assertEqual(doc.metadata["file_type"], "txt")
        finally:
            if f_path.exists():
                f_path.unlink()

    def test_recursive_chunker_basic(self) -> None:
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        doc = Document(
            id="doc-test",
            content="First sentence of the document. Second sentence with important details. Third sentence for testing.",
            metadata={"source": "test.txt"},
        )
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(c.document_id, "doc-test")
            self.assertLessEqual(len(c.text), 120)
            self.assertIn("chunk_index", c.metadata)
            self.assertIn("start_char", c.metadata)

    def test_chunker_with_sections(self) -> None:
        chunker = RecursiveTextChunker(chunk_size=300, chunk_overlap=30)
        content = (
            "## Section 1: Intro\n"
            "Here is the introduction to company policies.\n\n"
            "## Section 2: Details\n"
            "Here are the specific details and operational procedures."
        )
        doc = Document(id="doc-sections", content=content, metadata={"source": "sections.md"})
        chunks = chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 2)
        sections = [c.metadata.get("section") for c in chunks]
        self.assertIn("Section 1: Intro", sections)
        self.assertIn("Section 2: Details", sections)


if __name__ == "__main__":
    unittest.main()
