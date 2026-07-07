from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    #  LLM
    openai_api_key: str = Field(..., description="OpenAI API key.")
    openai_model_default: str = Field("gpt-4o-mini", description="Default model name.")

    #  RAG (PGVector store + OpenAI embeddings)
    database_url: str | None = Field(
        None,
        description="Async Postgres URL, e.g. postgresql+psycopg://user:pw@host:5432/db. "
        "Unset → NullRetriever (RAG disabled).",
    )
    embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model.")
    embedding_dim: int = Field(1536, description="Embedding vector length (model-dependent).")
    rag_collection: str = Field("mascan", description="PGVector collection name.")
    rag_vision_model: str = Field(
        "gpt-4o", description="Vision model for reading/captioning figures."
    )
    vision_base_url: str | None = Field(
        None,
        description="OpenAI-compatible endpoint for the vision model "
        "(e.g. Ollama http://localhost:11434/v1). Unset → OpenAI.",
    )
    vision_api_key: str | None = Field(
        None, description="Key for vision_base_url. Unset → reuse openai_api_key."
    )
    rag_image_dir: str = Field(
        "rag_images", description="Directory where figures extracted from uploaded PDFs are saved."
    )
    rag_max_retries: int = Field(
        1, description="Max self-correction (CRAG) rewrite-retries on the full retrieval path."
    )
    chunk_size: int = 1000
    chunk_overlap: int = 150


    #  Tools
    firecrawl_api_key: str | None = Field(
        None, description="Firecrawl API key (not needed for a self-hosted instance)."
    )
    firecrawl_api_url: str | None = Field(
        None, description="Base URL of a self-hosted Firecrawl."
    )

    #  Social agent — X/Twitter cookie secrets (kept in env, not config.yaml)
    twitter_auth_token: str | None = Field(
        None, description="X/Twitter auth_token cookie for twitter-cli in-process auth."
    )
    twitter_ct0: str | None = Field(
        None, description="X/Twitter ct0 (CSRF) cookie for twitter-cli in-process auth."
    )

    # Technological agent — Semantic Scholar API key and URL (optional, but recommended)
    semantic_scholar_api_key: str | None = Field(
        None, description="Semantic Scholar API key (optional, but recommended for better rate limits)."
    )
    semantic_scholar_api_url: str | None = Field(
        None, description="Semantic Scholar API URL."
    )

    #  App
    log_level: str = Field("INFO", description="Logging level.")
    environment: str = Field("development", description="dev | prod.")


settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global settings
    if settings is None:
        settings = Settings()
    return settings
