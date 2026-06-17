"""
Ingestion — load PDF/DOCX/TXT and split into chunks.
"""

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

from backend.core import logger

LOADERS = {
    ".pdf":  PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
}

CHUNK_SIZE    = int(__import__("os").environ.get("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(__import__("os").environ.get("CHUNK_OVERLAP", 64))


def ingest_file(file_path: str) -> List[Document]:
    path = Path(file_path)
    ext  = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {list(LOADERS)}")

    docs = LOADERS[ext](str(path)).load()
    for doc in docs:
        doc.metadata.setdefault("source", path.name)
        doc.metadata["file_type"] = ext.lstrip(".")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(f"'{path.name}' → {len(chunks)} chunks")
    return chunks
