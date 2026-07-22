from typing import Any, ClassVar

from newsdataapi import NewsDataApiClient
from pydantic import BaseModel, Field

from mascan.contracts import ToolResult
from mascan.core.settings import get_settings
from mascan.tools.base import BaseTool


class NewsDataSearchInput(BaseModel):
    query: str = Field(description="Political or geopolitical news query.")
    country: str | None = None
    language: str | None = None
    category: str = "politics"
    size: int = Field(10, description="Requested articles; values are clamped to 1–10.")


class NewsDataSearchTool(BaseTool):
    name = "news_api"

    description = "Search latest political and geopolitical news articles using NewsData.io."
    input_schema: ClassVar[type[BaseModel] | None] = NewsDataSearchInput
    MAX_RESULTS: ClassVar[int] = 10
    MAX_DESCRIPTION_CHARS: ClassVar[int] = 1_000

    def run(
        self,
        query: str,
        country: str | None = None,
        language: str | None = None,
        category: str = "politics",
        size: int = 10,
    ) -> ToolResult[Any]:
        """
        Search latest political news.

        Args:
            query: Search query.
            country: Optional country code.
            language: Optional language code.
            category: News category.
            size: Number of articles to return.
        """

        try:
            bounded_size = max(1, min(size, self.MAX_RESULTS))
            settings = get_settings()

            api = NewsDataApiClient(apikey=settings.news_api_key or "")

            params = {
                "q": query,
                "category": category,
                "size": bounded_size,
            }

            if country:
                params["country"] = country

            if language:
                params["language"] = language

            response = api.news_api(**params)  # type: ignore[attr-defined]

            raw_articles = response.get("results", [])
            articles = []
            text_truncated = False

            for item in raw_articles[: self.MAX_RESULTS]:
                description = item.get("description")
                if isinstance(description, str) and len(description) > self.MAX_DESCRIPTION_CHARS:
                    text_truncated = True
                articles.append(
                    {
                        "title": item.get("title"),
                        "description": self.truncate_text(
                            description,
                            self.MAX_DESCRIPTION_CHARS,
                        ),
                        "source": item.get("source_id"),
                        "url": item.get("link"),
                        "published_at": item.get("pubDate"),
                    }
                )

            data = {
                "query": query,
                "country": country,
                "language": language,
                "category": category,
                "article_count": len(articles),
                "articles": articles,
            }

            return ToolResult(
                success=True,
                data=data,
                source=f"newsdata:{query}",
                metadata={
                    "provider": "newsdata.io",
                    "country": country,
                    "language": language,
                    "category": category,
                    "article_count": len(articles),
                    "limit_applied": (
                        size != bounded_size
                        or len(raw_articles) > self.MAX_RESULTS
                        or text_truncated
                    ),
                },
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                data=None,
                source=f"newsdata:{query}",
                error=str(exc),
                metadata={
                    "provider": "newsdata.io",
                    "country": country,
                    "language": language,
                    "category": category,
                },
            )
