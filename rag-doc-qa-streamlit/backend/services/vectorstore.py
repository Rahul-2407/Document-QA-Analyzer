"""
VectorStore — per-session in-memory ChromaDB (EphemeralClient).
No disk, no tenant errors, no cross-session leakage.
"""

import os
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from backend.core import logger

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "rag_documents")
RETRIEVAL_K     = int(os.environ.get("RETRIEVAL_K", 5))


class VectorStoreService:
    """
    One instance per user session.
    Uses EphemeralClient so each instance has its own isolated in-memory DB.
    """

    def __init__(self):
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        # Each instance creates its own independent in-memory ChromaDB
        self._client = chromadb.EphemeralClient()
        self._client.get_or_create_collection(COLLECTION_NAME)
        self._store: Optional[Chroma] = None
        logger.info("VectorStoreService initialised (EphemeralClient).")

    # ── Embeddings ──────────────────────────────────────────────────────────

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    # ── Store ───────────────────────────────────────────────────────────────

    def _get_store(self) -> Chroma:
        # Verify collection still exists; recreate if deleted
        try:
            self._client.get_collection(COLLECTION_NAME)
        except Exception:
            logger.warning("Collection missing — recreating.")
            self._client.get_or_create_collection(COLLECTION_NAME)
            self._store = None

        if self._store is None:
            self._store = Chroma(
                client=self._client,
                collection_name=COLLECTION_NAME,
                embedding_function=self._get_embeddings(),
            )
        return self._store

    # ── Public API ──────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> List[str]:
        ids = self._get_store().add_documents(documents)
        logger.info(f"Indexed {len(ids)} chunks.")
        return ids

    def get_retriever(self, k: int = RETRIEVAL_K) -> VectorStoreRetriever:
        return self._get_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def get_all_documents(self) -> List[Document]:
        raw = self._get_store()._collection.get(include=["documents", "metadatas"])
        return [
            Document(page_content=c, metadata=m or {})
            for c, m in zip(raw["documents"], raw["metadatas"])
        ]

    def count(self) -> int:
        try:
            return self._get_store()._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._client.get_or_create_collection(COLLECTION_NAME)
        self._store = None
        logger.info("Collection cleared.")
