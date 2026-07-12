"""Exobrain configuration — all via environment variables."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # LLM provider — any OpenAI-compatible API
    llm_api_key: str = field(default_factory=lambda: os.getenv("EXOBRAIN_LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("EXOBRAIN_LLM_BASE_URL", "https://api.deepseek.com"))
    llm_model: str = field(default_factory=lambda: os.getenv("EXOBRAIN_LLM_MODEL", "deepseek-chat"))

    # Server
    host: str = field(default_factory=lambda: os.getenv("EXOBRAIN_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("EXOBRAIN_PORT", "8080")))

    # RAG
    rag_index_path: str = field(default_factory=lambda: os.getenv("EXOBRAIN_RAG_INDEX", ""))
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("EXOBRAIN_RAG_TOP_K", "3")))

    # CORS origins (comma-separated)
    cors_origins: list[str] = field(default_factory=lambda: os.getenv("EXOBRAIN_CORS_ORIGINS", "*").split(","))

    # Rate limiting (requests per minute per IP, 0 = disabled)
    rate_limit_rpm: int = field(default_factory=lambda: int(os.getenv("EXOBRAIN_RATE_LIMIT_RPM", "10")))


config = Config()
