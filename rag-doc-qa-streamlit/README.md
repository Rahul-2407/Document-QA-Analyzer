---
title: RAG Document Q&A
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.37.0
app_file: app.py
pinned: false
---

# RAG Document Q&A System

Upload any PDF, DOCX, or TXT and ask questions in natural language.

- Hybrid BM25 + semantic search (RRF fusion)
- Source citations per answer
- LLM confidence scoring (HIGH / MEDIUM / LOW)
- Each session is isolated — fresh start on every visit

**Stack:** LangChain · ChromaDB · Groq (Llama 3.3) · sentence-transformers · Streamlit
