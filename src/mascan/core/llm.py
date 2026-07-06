from functools import lru_cache

from langchain_openai import ChatOpenAI

from mascan.core.settings import get_settings


@lru_cache(maxsize=8)
def get_chat_model(
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance for the given parameters.

    base_url/api_key let this point at any OpenAI-compatible endpoint (Ollama,
    OpenRouter, Groq, vLLM); unset → OpenAI with the configured key.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.openai_model_default,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        api_key=api_key or settings.openai_api_key,
    )


def get_vision_model(temperature: float = 0.2) -> ChatOpenAI:
    """Vision model for reading/captioning figures.

    Defaults to OpenAI (rag_vision_model).
    Or set VISION_BASE_URL to use a self-hosted OpenAI-compatible endpoint.
    """
    s = get_settings()
    return get_chat_model(
        model=s.rag_vision_model,
        temperature=temperature,
        base_url=s.vision_base_url or None,
        api_key=s.vision_api_key or s.openai_api_key,
    )
