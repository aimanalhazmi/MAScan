from datetime import date
from typing import Any

import pytest

from mascan.agents.economics.tools.market_data import WeeklyStockPricesTool
from mascan.agents.environmental.tools.world_bank import (
    WorldBankEnvironmentalIndicatorsTool,
)
from mascan.agents.legal.tools.eur_lex import EurLexTool
from mascan.agents.legal.tools.federal_register import FederalRegisterTool
from mascan.agents.political.tools.news_api import NewsDataSearchTool
from mascan.agents.social.tools.reddit_api import RedditSearchTool
from mascan.agents.social.tools.world_bank import WorldBankSocialIndicatorsTool
from mascan.agents.social.tools.x_api import XSearchTool
from mascan.agents.technological.tools.scholar import ScholarSearchTool
from mascan.tools.common.rag_search import RagSearchTool
from mascan.tools.common.web_search import WebSearchTool


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (WebSearchTool(api_key="test"), {"query": "q", "max_results": 1_000}),
        (RagSearchTool(), {"query": "q", "k": 1_000}),
        (NewsDataSearchTool(), {"query": "q", "size": 1_000}),
        (ScholarSearchTool(), {"query": "q", "max_results": 1_000}),
        (FederalRegisterTool(), {"query": "q", "per_page": 1_000}),
        (EurLexTool(), {"query": "q", "limit": 1_000}),
        (RedditSearchTool(), {"query": "q", "limit": 1_000}),
        (XSearchTool(), {"query": "q", "max_results": 1_000}),
    ],
)
def test_input_schemas_allow_oversized_counts_for_runtime_clamping(
    tool: Any,
    arguments: dict[str, Any],
) -> None:
    assert tool.input_schema is not None
    validated = tool.input_schema.model_validate(arguments)

    assert validated is not None


def test_news_clamps_oversized_call_without_failing(mocker: Any) -> None:
    articles = [
        {
            "title": f"Article {index}",
            "description": "x" * 1_500,
            "source_id": "wire",
            "link": f"https://news.example/{index}",
            "pubDate": "2026-07-15",
        }
        for index in range(12)
    ]
    client = mocker.patch(
        "mascan.agents.political.tools.news_api.NewsDataApiClient"
    ).return_value
    client.news_api.return_value = {"results": articles}
    mocker.patch(
        "mascan.agents.political.tools.news_api.get_settings"
    ).return_value.NEWSDATA_API_KEY = "test"

    result = NewsDataSearchTool().run(query="policy", size=50)

    assert result.success
    assert len(result.data["articles"]) == 10
    assert all(len(article["description"]) <= 1_000 for article in result.data["articles"])
    assert result.metadata["limit_applied"] is True
    client.news_api.assert_called_once_with(q="policy", category="politics", size=10)


def test_scholar_clamps_results_and_abstracts(mocker: Any) -> None:
    tool = ScholarSearchTool()
    search = mocker.patch.object(
        tool,
        "search_literature",
        return_value=[
            {
                "title": f"Paper {index}",
                "abstract": "x" * 2_500,
                "url": f"https://scholar.example/{index}",
            }
            for index in range(7)
        ],
    )

    result = tool.run(query="AI", max_results=100)

    assert result.success
    assert len(result.data) == 5
    assert all(len(paper["abstract"]) <= 2_000 for paper in result.data)
    assert result.metadata["limit_applied"] is True
    search.assert_called_once_with(
        query="AI",
        max_results=5,
        year_from=None,
        year_to=None,
    )


def test_federal_register_clamps_documents_and_abstracts(mocker: Any) -> None:
    response = mocker.Mock()
    response.json.return_value = {
        "count": 12,
        "results": [
            {
                "title": f"Rule {index}",
                "abstract": "x" * 2_500,
                "html_url": f"https://federal.example/{index}",
            }
            for index in range(12)
        ],
    }
    http_get = mocker.patch(
        "mascan.agents.legal.tools.federal_register.http_get",
        return_value=response,
    )

    result = FederalRegisterTool().run(query="privacy", per_page=1_000)

    assert result.success
    assert len(result.data["documents"]) == 10
    assert all(len(doc["abstract"]) <= 2_000 for doc in result.data["documents"])
    assert result.metadata["limit_applied"] is True
    assert http_get.call_args.kwargs["params"]["per_page"] == 10


def test_eur_lex_clamps_documents_and_titles(mocker: Any) -> None:
    response = mocker.Mock()
    response.json.return_value = {
        "results": {
            "bindings": [
                {
                    "celex": {"value": str(index)},
                    "title": {"value": "x" * 600},
                    "date": {"value": "2026-07-15"},
                }
                for index in range(12)
            ]
        }
    }
    http_get = mocker.patch(
        "mascan.agents.legal.tools.eur_lex.http_get",
        return_value=response,
    )

    result = EurLexTool().run(query="privacy", limit=50)

    assert result.success
    assert len(result.data["documents"]) == 10
    assert all(len(doc["title"]) <= 500 for doc in result.data["documents"])
    assert result.metadata["limit_applied"] is True
    assert "LIMIT 10" in http_get.call_args.kwargs["params"]["query"]


def test_reddit_clamps_posts_and_snippets_without_failing(mocker: Any) -> None:
    post = type(
        "Post",
        (),
        {
            "to_dict": lambda self: {
                "id": "1",
                "title": "Discussion",
                "selftext": "x" * 1_500,
            }
        },
    )()
    listing = type("Listing", (), {"items": [post] * 12})()
    mocker.patch("rdt_cli.auth.load_credential", return_value=object())
    client = mocker.MagicMock()
    client.__enter__.return_value.search.return_value = {"raw": "payload"}
    mocker.patch("rdt_cli.client.RedditClient", return_value=client)
    mocker.patch("rdt_cli.parser.parse_listing", return_value=listing)

    result = RedditSearchTool().run(query="market", limit=50)

    assert result.success
    assert len(result.data) == 10
    assert all(len(post["snippet"]) <= 1_000 for post in result.data)
    assert result.metadata["limit_applied"] is True
    client.__enter__.return_value.search.assert_called_once_with(
        "market",
        sort="relevance",
        time_filter="month",
        limit=10,
    )


def test_x_clamps_posts_and_text_without_failing(mocker: Any) -> None:
    settings = mocker.patch(
        "mascan.agents.social.tools.x_api.get_settings"
    ).return_value
    settings.twitter_auth_token = "token"
    settings.twitter_ct0 = "csrf"
    client = mocker.patch("twitter_cli.client.TwitterClient").return_value
    client.fetch_search.return_value = [object()] * 12
    mocker.patch(
        "twitter_cli.serialization.tweets_to_data",
        return_value=[
            {"id": str(index), "text": "x" * 1_500}
            for index in range(12)
        ],
    )

    result = XSearchTool().run(query="market", max_results=50)

    assert result.success
    assert len(result.data) == 10
    assert all(len(post["text"]) <= 1_000 for post in result.data)
    assert result.metadata["limit_applied"] is True
    client.fetch_search.assert_called_once_with("market", count=10, product="Top")


def world_bank_response() -> Any:
    payload = [
        {},
        [
            {
                "indicator": {"value": "Metric"},
                "country": {"value": "Place"},
                "date": "2025",
                "value": 1,
            }
        ],
    ]
    return type("Response", (), {"json": lambda self: payload})()


def test_social_world_bank_clamps_country_indicator_matrix(mocker: Any) -> None:
    http_get = mocker.patch(
        "mascan.agents.social.tools.world_bank.http_get",
        return_value=world_bank_response(),
    )

    result = WorldBankSocialIndicatorsTool().run(
        country_codes=["A", "B", "C", "D"],
        indicators=[str(index) for index in range(8)],
    )

    assert result.success
    assert len(result.data) == 18
    assert http_get.call_count == 18
    assert result.metadata["limit_applied"] is True


def test_environmental_world_bank_clamps_country_indicator_matrix(mocker: Any) -> None:
    http_get = mocker.patch(
        "mascan.agents.environmental.tools.world_bank.http_get",
        return_value=world_bank_response(),
    )

    result = WorldBankEnvironmentalIndicatorsTool().run(
        country_codes=["ARG", "BRA", "CHN", "DEU"],
        indicators=[str(index) for index in range(12)],
    )

    assert result.success
    # countries clamped 4 -> 3, indicators clamped 12 -> 8 => 24 records
    assert len(result.data) == 24
    assert http_get.call_count == 24
    assert result.metadata["limit_applied"] is True


def test_stock_history_clamps_range_and_rows_without_failing(mocker: Any) -> None:
    tool = WeeklyStockPricesTool()
    prices = [{"date": str(index)} for index in range(110)]
    fetch = mocker.patch.object(
        tool,
        "get_stock_prices",
        return_value={
            "ticker": "TEST",
            "start_date": "2010-01-01",
            "end_date": "2026-01-01",
            "weekly_prices": prices,
        },
    )

    result = tool.run(
        ticker="TEST",
        start_date="2010-01-01",
        end_date="2026-01-01",
    )

    assert result.success
    assert len(result.data["weekly_prices"]) == 104
    assert result.data["weekly_prices"][0] == prices[6]
    assert result.metadata["limit_applied"] is True
    applied = fetch.call_args.kwargs
    assert (date.fromisoformat(applied["end_date"]) - date.fromisoformat(applied["start_date"])).days == 730
