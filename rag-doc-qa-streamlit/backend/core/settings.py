"""
backend/core/settings.py
Central config — reads from .env file via pydantic-settings.
Import `settings` anywhere in the project instead of calling os.getenv directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    llm_model: str = Field("llama-3.3-70b-versatile", alias="LLM_MODEL")
    llm_temperature: float = Field(0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(1024, alias="LLM_MAX_TOKENS")

    # Embeddings
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    # ChromaDB
    chroma_persist_dir: str = Field("./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("rag_documents", alias="CHROMA_COLLECTION_NAME")

    # Retrieval
    retrieval_k: int = Field(5, alias="RETRIEVAL_K")
    semantic_weight: float = Field(0.6, alias="SEMANTIC_WEIGHT")
    bm25_weight: float = Field(0.4, alias="BM25_WEIGHT")

    # Chunking
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")

    # LangSmith
    langchain_tracing_v2: bool = Field(False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field("", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field("rag-doc-qa", alias="LANGCHAIN_PROJECT")

    # FastAPI
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    upload_dir: str = Field("./data/uploads", alias="UPLOAD_DIR")
    max_upload_size_mb: int = Field(50, alias="MAX_UPLOAD_SIZE_MB")

    # Streamlit
    api_base_url: str = Field("http://localhost:8000", alias="API_BASE_URL")


# Single shared instance — import this everywhere
settings = Settings()
