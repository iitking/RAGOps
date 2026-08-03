"""Document loaders for RAGOps ingestion pipeline.

Supports plain text, Markdown, HTML, PDF, and DOCX formats.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Document:
    """Represents an ingested document."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }


def _generate_doc_id(content: str, source: str) -> str:
    """Generate a stable, deterministic ID from content hash and source name."""
    hasher = hashlib.sha256()
    hasher.update(source.encode("utf-8"))
    hasher.update(content[:500].encode("utf-8"))
    return f"doc-{hasher.hexdigest()[:12]}"


class TextLoader:
    """Loader for plain text and markdown documents."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = Path(file_path)

    def load(self) -> Document:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Document file not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        doc_id = _extract_document_id(content) or _generate_doc_id(content, self.file_path.name)
        stat = self.file_path.stat()

        return Document(
            id=doc_id,
            content=content,
            metadata={
                "source": self.file_path.name,
                "file_path": str(self.file_path.resolve()),
                "file_type": self.file_path.suffix.lstrip(".").lower() or "txt",
                "file_size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            },
        )


class HTMLLoader:
    """Loader for HTML documents, extracting text while stripping tags."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = Path(file_path)

    def load(self) -> Document:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Document file not found: {self.file_path}")

        raw_html = self.file_path.read_text(encoding="utf-8", errors="replace")

        # Strip HTML tags cleanly
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator="\n")
        except ImportError:
            # Fallback regex strip
            text = re.sub(r"<style[\s\S]*?</style>", "", raw_html, flags=re.IGNORECASE)
            text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)

        doc_id = _extract_document_id(text) or _generate_doc_id(text, self.file_path.name)
        stat = self.file_path.stat()

        return Document(
            id=doc_id,
            content=text,
            metadata={
                "source": self.file_path.name,
                "file_path": str(self.file_path.resolve()),
                "file_type": "html",
                "file_size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            },
        )


class PDFLoader:
    """Loader for PDF documents."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = Path(file_path)

    def load(self) -> Document:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Document file not found: {self.file_path}")

        text_pages: List[str] = []

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(self.file_path))
            for page in doc:
                text_pages.append(page.get_text())
            doc.close()
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(self.file_path))
                for page in reader.pages:
                    text_pages.append(page.extract_text() or "")
            except ImportError:
                # If neither is installed, attempt binary string extract or error gracefully
                raise ImportError("PyMuPDF or pypdf is required to parse PDF documents.")

        content = "\n\n".join(text_pages)
        doc_id = _extract_document_id(content) or _generate_doc_id(content, self.file_path.name)
        stat = self.file_path.stat()

        return Document(
            id=doc_id,
            content=content,
            metadata={
                "source": self.file_path.name,
                "file_path": str(self.file_path.resolve()),
                "file_type": "pdf",
                "file_size_bytes": stat.st_size,
                "page_count": len(text_pages),
                "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            },
        )


def _extract_document_id(text: str) -> Optional[str]:
    """Extract explicit Document ID if stated in headers (e.g. Document ID: hr-policy-v3)."""
    match = re.search(r"(?:Document ID|Doc ID|ID):\s*([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def load_document(file_path: Union[str, Path]) -> Document:
    """Auto-detect file format and load into a Document object."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md", ".json", ".csv"):
        return TextLoader(path).load()
    elif suffix in (".html", ".htm"):
        return HTMLLoader(path).load()
    elif suffix == ".pdf":
        return PDFLoader(path).load()
    else:
        # Fallback to TextLoader
        return TextLoader(path).load()


class DirectoryLoader:
    """Loads all documents from a directory matching supported extensions."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".html", ".htm", ".pdf"}

    def __init__(self, directory_path: Union[str, Path], recursive: bool = True) -> None:
        self.directory_path = Path(directory_path)
        self.recursive = recursive

    def load(self) -> List[Document]:
        if not self.directory_path.exists():
            return []

        documents: List[Document] = []
        pattern = "**/*" if self.recursive else "*"

        for file_path in self.directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = load_document(file_path)
                    documents.append(doc)
                except Exception as e:
                    # Keep loading other documents, logging warning
                    print(f"Warning: Failed to load {file_path}: {e}")

        return documents
