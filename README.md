# RAG Document Q&A System
 
A production-grade Retrieval-Augmented Generation (RAG) application that lets users upload documents — PDF, DOCX, TXT, or Markdown — and ask questions about them in natural language. The system retrieves the most relevant sections of the document and generates answers grounded in that content, complete with source citations and a confidence score.
 
🔗 **Live demo:** [huggingface.co/spaces/Rahul-2407/rag-doc-qa](https://huggingface.co/spaces/Rahul-2407/rag-doc-qa)
 
---
 
## How it works
 
When a document is uploaded, it's loaded, split into overlapping chunks, converted into vector embeddings using a local sentence-transformers model, and stored in ChromaDB. When a question is asked, two retrievers run in parallel — a semantic retriever that searches by meaning, and a BM25 retriever that searches by keyword — and their results are merged using Reciprocal Rank Fusion. The most relevant chunks are passed to Llama 3.3 70B via Groq, which generates an answer citing its sources and assessing its own confidence as HIGH, MEDIUM, or LOW.
 
```
INDEXING PIPELINE
Document → Loader → Recursive Chunker → Embeddings → ChromaDB
 
QUERY PIPELINE
Question → Semantic Retriever ─┐
        └→ BM25 Retriever ─────┴→ RRF Fusion → Context Builder → LLM → Answer + Citations + Confidence
```
 
## What makes it production-grade
 
Most RAG demos stop at a basic semantic search loop. This project adds:
 
- **Hybrid retrieval** for better accuracy on both conceptual and keyword-based questions
- **Inline source citations** so every claim is traceable back to a specific document and page
- **LLM-assessed confidence scoring** so users know when to verify externally
- **Modular service architecture** with dependency injection for testability
- **Session isolation** for safe multi-user deployment
## Tech stack
 
| Layer | Technology |
|---|---|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, no API key) |
| Vector store | ChromaDB |
| Hybrid retrieval | LangChain EnsembleRetriever (BM25 + semantic, RRF fusion) |
| Orchestration | LangChain |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Observability | LangSmith |
| Evaluation | RAGAS |
 
---
