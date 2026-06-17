"""
QA Chain — retrieve → build context → LLM → parse citations + confidence.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from backend.core import logger

LLM_MODEL       = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS  = int(os.environ.get("LLM_MAX_TOKENS", 1024))
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")


@dataclass
class SourceChunk:
    index: int
    content: str
    source: str
    page: Optional[int] = None


@dataclass
class QAResponse:
    question: str
    answer: str
    confidence: str
    sources: List[SourceChunk] = field(default_factory=list)


SYSTEM_PROMPT = """You are a precise document Q&A assistant. Answer using ONLY the context below.

Rules:
- If context is insufficient, say: "I could not find enough information in the uploaded documents."
- Cite sources inline as [Source N].
- End your answer with exactly: CONFIDENCE: <HIGH|MEDIUM|LOW>
  HIGH = context directly answers the question
  MEDIUM = partial answer or inference required
  LOW = barely relevant

Context:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}"),
])


class QAChainService:

    def __init__(self):
        self._llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

    def _format_context(self, chunks: List[Document]):
        source_chunks = []
        parts = []
        for i, doc in enumerate(chunks, 1):
            meta   = doc.metadata
            source = meta.get("source", "Unknown")
            page   = meta.get("page", None)
            source_chunks.append(SourceChunk(index=i, content=doc.page_content, source=source, page=page))
            page_label = f", page {page}" if page is not None else ""
            parts.append(f"[Source {i}] ({source}{page_label})\n{doc.page_content}")
        return "\n\n---\n\n".join(parts), source_chunks

    def _parse_confidence(self, raw: str):
        match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", raw, re.IGNORECASE)
        if match:
            return raw[:match.start()].strip(), match.group(1).upper()
        return raw.strip(), "MEDIUM"

    def answer(self, question: str, retriever_service) -> QAResponse:
        try:
            chunks = retriever_service.retrieve(question)
        except RuntimeError as e:
            return QAResponse(question=question, answer=str(e), confidence="LOW")

        if not chunks:
            return QAResponse(
                question=question,
                answer="No relevant content found in the uploaded documents.",
                confidence="LOW",
            )

        context_str, source_chunks = self._format_context(chunks)
        chain = QA_PROMPT | self._llm | StrOutputParser()
        raw = chain.invoke({"context": context_str, "question": question})
        answer, confidence = self._parse_confidence(raw)

        logger.info(f"Answer generated. Confidence: {confidence}")
        return QAResponse(
            question=question,
            answer=answer,
            confidence=confidence,
            sources=source_chunks,
        )
