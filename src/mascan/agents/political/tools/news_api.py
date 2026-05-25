from newsdataapi import NewsDataApiClient

from mascan.contracts import ToolResult
from mascan.core.settings import get_settings
from mascan.tools.base import BaseTool


class NewsDataSearchTool(BaseTool):
    name = "news_api"

    description = (
        "Search latest political and geopolitical news articles "
        "using NewsData.io."
    )

    def run(
        self,
        query: str,
        country: str | None = None,
        language: str | None = None,
        category: str = "politics",
        size: int = 10,
    ) -> ToolResult:
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
            settings = get_settings()

            api = NewsDataApiClient(
                apikey=settings.NEWSDATA_API_KEY
            )

            params = {
                "q": query,
                "category": category,
                "size": size,
            }

            if country:
                params["country"] = country

            if language:
                params["language"] = language

            response = api.news_api(**params)

            articles = []

            for item in response.get("results", []):
                articles.append(
                    {
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "source": item.get("source_id"),
                        "url": item.get("link"),
                        "published_at": item.get("pubDate"),
                        "image_url": item.get("image_url"),
                        "keywords": item.get("keywords"),
                        "country": item.get("country"),
                        "category": item.get("category"),
                        "language": item.get("language"),
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