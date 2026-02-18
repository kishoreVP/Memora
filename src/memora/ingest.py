import logging
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from memora.config import settings

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_text(text)


def collect_files(path: Path, recursive: bool = False) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED else []
    pattern = "**/*" if recursive else "*"
    return [f for f in path.glob(pattern) if f.is_file() and f.suffix.lower() in SUPPORTED]
