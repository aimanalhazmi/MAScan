from functools import lru_cache

from langchain_openai import ChatOpenAI

from mascan.core.settings import get_settings


@lru_cache(maxsize=8)
def get_chat_model(
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance for the given parameters"""
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.openai_model_default,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.openai_api_key,
    )