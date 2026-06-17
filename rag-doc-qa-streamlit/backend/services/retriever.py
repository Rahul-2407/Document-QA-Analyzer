"""
Hybrid retriever — BM25 + semantic with RRF fusion.
One instance per session, tied to a VectorStoreService instance.
"""

import os
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from backend.core import logger

RETRIEVAL_K     = int(os.environ.get("RETRIEVAL_K", 5))
SEMANTIC_WEIGHT = float(os.environ.get("SEMANTIC_WEIGHT", 0.6))
BM25_WEIGHT     = float(os.environ.get("BM25_WEIGHT", 0.4))


class HybridRetrieverService:

    def __init__(self, vectorstore_service):
        self._vs = vectorstore_service
        self._retriever: Optional[EnsembleRetriever] = None

    def _build(self) -> EnsembleRetriever:
        docs = self._vs.get_all_documents()
        if not docs:
            raise RuntimeError("No documents indexed yet.")

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = RETRIEVAL_K

        semantic = self._vs.get_retriever(k=RETRIEVAL_K)

        self._retriever = EnsembleRetriever(
            retrievers=[semantic, bm25],
            weights=[SEMANTIC_WEIGHT, BM25_WEIGHT],
        )
        logger.info(f"Hybrid retriever built from {len(docs)} docs.")
        return self._retriever

    def rebuild(self) -> None:
        self._retriever = None
        self._build()

    def retrieve(self, query: str) -> List[Document]:
        if self._retriever is None:
            self._build()
        results = self._retriever.invoke(query)
        logger.info(f"Retrieved {len(results)} chunks.")
        return results

    def reset(self) -> None:
        self._retriever = None
