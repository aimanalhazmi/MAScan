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

    #  Tools
    firecrawl_api_key: str = Field(..., description="Firecrawl API key.")

    #  Social agent — X/Twitter cookie secrets (kept in env, not config.yaml)
    twitter_auth_token: str | None = Field(
        None, description="X/Twitter auth_token cookie for twitter-cli in-process auth."
    )
    twitter_ct0: str | None = Field(
        None, description="X/Twitter ct0 (CSRF) cookie for twitter-cli in-process auth."
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
