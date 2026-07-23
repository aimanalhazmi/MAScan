import json
from typing import Any

from langchain_core.messages import ToolMessage

from mascan.agents.registry import agent_registry
from mascan.agents.social.agent import SocialAgent, SocialEvidencePlan
from mascan.agents.social.tools.reddit_api import RedditSearchTool
from mascan.agents.social.tools.world_bank import WorldBankSocialIndicatorsTool
from mascan.agents.social.tools.x_api import XSearchTool
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult


def test_social_agent_is_registered() -> None:
    import mascan.agents.social  # noqa: F401

    assert "social" in agent_registry.all_names()


def test_social_agent_loads_reddit_world_bank_x_and_web_tools() -> None:
    import mascan.agents.social  # noqa: F401

    agent = SocialAgent()

    assert "reddit_search" in agent.tools
    assert "world_bank_social_indicators" in agent.tools
    assert "x_search" in agent.tools
    assert "web_search" in agent.tools


def test_social_agent_run_returns_report(mocker: Any) -> None:
    import mascan.agents.social  # noqa: F401

    agent = SocialAgent()
    deterministic_outputs = {
        "web_search": ToolResult(
            success=True,
            data=[
                {
                    "title": "Survey: Americans concerned about EV battery disposal",
                    "url": "https://example.com/ev-battery-recycling",
                    "markdown": "Consumers are concerned about EV battery disposal.",
                }
            ],
            source="web_search:firecrawl",
            metadata={
                "query": "consumer sentiment around EV battery recycling",
                "count": 1,
            },
        ),
        "world_bank_social_indicators": ToolResult(
            success=True,
            data=[
                {
                    "indicator_code": "SP.POP.TOTL",
                    "indicator_name": "Population, total",
                    "country_code": "WLD",
                    "country_name": "World",
                    "date": "2025",
                    "value": 8000000000,
                }
            ],
            source="world_bank:social_indicators",
            metadata={
                "provider": "World Bank Indicators API",
                "country_code": "WLD",
                "country_codes": ["WLD"],
                "indicator_count": 1,
                "source_urls": ["https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL"],
            },
        ),
    }

    llm_sources = [
        Source(
            name="r/electricvehicles: EV battery recycling concerns",
            url="https://reddit.com/r/electricvehicles/comments/abc123",
            metadata={"tool": "reddit_search"},
        )
    ]
    # The report's Sources are harvested from the ReAct result's tool messages
    # (via collect_sources -> sources_from_react), so the react_result must carry
    # the reddit post link the assertions expect.
    react_result = {
        "messages": [
            ToolMessage(
                content=json.dumps(
                    [
                        {
                            "id": "abc123",
                            "title": "r/electricvehicles: EV battery recycling concerns",
                            "url": "https://reddit.com/r/electricvehicles/comments/abc123",
                        }
                    ]
                ),
                name="reddit_search",
                tool_call_id="a",
            ),
        ]
    }
    mocker.patch.object(
        agent,
        "gather_deterministic",
        return_value=deterministic_outputs,
    )
    mocker.patch.object(
        agent,
        "run_react_agent",
        return_value=(
            react_result,
            "Social sentiment findings",
            ["reddit_search"],
            llm_sources,
        ),
    )

    report = agent.run(tasks=["consumer sentiment around EV battery recycling"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "social"
    assert report.tasks == ["consumer sentiment around EV battery recycling"]
    assert report.findings == "Social sentiment findings"
    assert report.metadata["mode"] == "mixed"
    # Social has no config.always_call_tools; World Bank is gathered via the
    # agent's overridden gather_deterministic, so this metadata list is empty.
    assert report.metadata["deterministic_tools"] == []
    assert report.metadata["llm_chosen_tools"] == ["reddit_search"]
    assert "## Social Analysis" in report.rendered_markdown
    # The LLM-chosen tools render in their own section, not mixed into Sources.
    # (world_bank is prepended via default_display_tools, so don't assume adjacency.)
    assert "**Tools the LLM chose to call:**" in report.rendered_markdown
    assert "- reddit_search" in report.rendered_markdown
    # Sources are real article-level links, not tool names.
    assert [source.url for source in report.sources] == [
        "https://example.com/ev-battery-recycling",
        "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL",
        "https://reddit.com/r/electricvehicles/comments/abc123",
    ]
    assert "Survey: Americans concerned about EV battery disposal" in report.rendered_markdown
    assert "https://reddit.com/r/electricvehicles/comments/abc123" in report.rendered_markdown
    assert "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL" in (
        report.rendered_markdown
    )


def test_social_extract_llm_sources_pulls_post_links() -> None:
    import json

    from langchain_core.messages import AIMessage, ToolMessage

    from mascan.agents.sources import sources_from_react

    reddit_payload = json.dumps(
        [{"id": "abc", "title": "t", "url": "https://reddit.com/r/x/comments/abc"}]
    )
    x_payload = json.dumps([{"id": "1", "text": "hi", "url": "https://x.com/u/status/1"}])
    result = {
        "messages": [
            AIMessage(content="thinking"),
            ToolMessage(content=reddit_payload, name="reddit_search", tool_call_id="a"),
            ToolMessage(content=x_payload, name="x_search", tool_call_id="b"),
        ]
    }

    sources = sources_from_react(result)
    by_url = {s.url: s for s in sources}

    assert by_url["https://reddit.com/r/x/comments/abc"].name == "t"
    assert by_url["https://reddit.com/r/x/comments/abc"].metadata["tool"] == "reddit_search"
    assert by_url["https://x.com/u/status/1"].name == "hi"
    assert by_url["https://x.com/u/status/1"].metadata["tool"] == "x_search"


def test_social_agent_constrains_evidence_plan() -> None:
    plan = SocialEvidencePlan(
        country_codes=["deu", "USA", "DEU", "CHN"],
        web_queries=[" germany ev recycling sentiment ", "battery recycling risk", "x", "y"],
    )

    constrained = SocialAgent.constrain_evidence_plan(plan)

    assert constrained.country_codes == ["DEU", "USA", "CHN"]
    assert constrained.web_queries == [
        "germany ev recycling sentiment",
        "battery recycling risk",
        "x",
    ]


def test_social_gather_deterministic_skips_reddit_and_x(mocker: Any) -> None:
    """Reddit/X are now LLM-decided, so deterministic gathering must not call them."""
    agent = SocialAgent()
    web_search = mocker.Mock()
    world_bank = mocker.Mock()
    reddit = mocker.Mock()
    x_search = mocker.Mock()
    web_search.run.return_value = ToolResult(success=True, data=[], source="web_search:test")
    world_bank.run.return_value = ToolResult(
        success=True,
        data=[],
        source="world_bank:social_indicators",
    )
    # gather_deterministic rebuilds self.tools from always_call_tools + optional_tools,
    # so inject the mocks there rather than on agent.tools.
    agent.always_call_tools = {}
    agent.optional_tools = {
        "web_search": web_search,
        "world_bank_social_indicators": world_bank,
        "reddit_search": reddit,
        "x_search": x_search,
    }
    mocker.patch.object(
        agent,
        "plan_evidence",
        return_value=SocialEvidencePlan(
            country_codes=["DEU", "USA"],
            web_queries=[
                "germany ev battery recycling consumer sentiment",
                "recycling social risk",
            ],
        ),
    )

    outputs = agent.gather_deterministic(["Analyze Germany and United States labour trends"])

    # Only World Bank is gathered deterministically; web_search/reddit/x are LLM-decided.
    world_bank.run.assert_called_once_with(country_codes=["DEU", "USA"])
    web_search.run.assert_not_called()
    reddit.run.assert_not_called()
    x_search.run.assert_not_called()
    assert list(outputs) == ["world_bank_social_indicators"]


def test_social_get_optional_tools_respects_flags() -> None:
    import mascan.agents.social  # noqa: F401

    agent = SocialAgent()
    agent.config.options = {"enable_reddit": True, "enable_x": False}
    optional = agent.get_optional_tools()

    # web_search and world_bank are always offered; only x_search is gated off here.
    assert {tool.name for tool in optional} == {
        "web_search",
        "world_bank_social_indicators",
        "reddit_search",
    }


def test_x_search_without_tokens_returns_failure(mocker: Any) -> None:
    settings = mocker.patch("mascan.agents.social.tools.x_api.get_settings").return_value
    settings.twitter_auth_token = None
    settings.twitter_ct0 = None

    result = XSearchTool().run(query="EV battery recycling")

    assert not result.success
    assert result.source == "x:twitter_cli_search"
    assert "TWITTER_AUTH_TOKEN" in result.error
    assert result.metadata["platform"] == "x"
    assert result.metadata["provider"] == "twitter-cli"
    assert result.metadata["query"] == "EV battery recycling"
    assert result.metadata["count"] == 0


def test_reddit_search_without_credential_returns_failure(mocker: Any) -> None:
    mocker.patch("rdt_cli.auth.load_credential", return_value=None)

    result = RedditSearchTool().run(query="EV battery recycling", limit=5)

    assert not result.success
    assert result.source == "reddit:rdt_cli_search"
    assert "rdt login" in result.error
    assert result.metadata["count"] == 0


def test_reddit_search_formats_listing(mocker: Any) -> None:
    post = type(
        "Post",
        (),
        {
            "to_dict": lambda self: {
                "id": "abc123",
                "title": "Battery recycling concerns",
                "subreddit": "electricvehicles",
                "score": 10,
                "num_comments": 3,
                "permalink": "/r/electricvehicles/comments/abc123",
                "url": "",
                "author": "someone",
                "created_utc": 1.0,
                "selftext": "People discuss recycling options.",
            }
        },
    )()
    listing = type("ListingPage", (), {"items": [post]})()

    mocker.patch("rdt_cli.auth.load_credential", return_value=object())
    client_cm = mocker.MagicMock()
    client_cm.__enter__.return_value.search.return_value = {"raw": "payload"}
    mocker.patch("rdt_cli.client.RedditClient", return_value=client_cm)
    mocker.patch("rdt_cli.parser.parse_listing", return_value=listing)

    result = RedditSearchTool().run(query="EV battery recycling", limit=5)

    assert result.success
    assert result.source == "reddit:rdt_cli_search"
    assert result.metadata["provider"] == "rdt-cli"
    assert result.metadata["count"] == 1
    assert result.data is not None
    assert result.data[0]["title"] == "Battery recycling concerns"
    assert result.data[0]["subreddit"] == "electricvehicles"
    assert result.data[0]["url"] == "https://reddit.com/r/electricvehicles/comments/abc123"
    # The full raw post object is no longer attached (context-overflow guard).
    assert "raw" not in result.data[0]


def test_reddit_search_truncates_long_selftext(mocker: Any) -> None:
    from mascan.agents.social.tools.reddit_api import RedditSearchTool

    long_text = "z" * (RedditSearchTool.MAX_SNIPPET_CHARS + 500)
    post = type(
        "Post",
        (),
        {"to_dict": lambda self: {"id": "x", "title": "t", "selftext": long_text}},
    )()
    listing = type("ListingPage", (), {"items": [post]})()

    mocker.patch("rdt_cli.auth.load_credential", return_value=object())
    client_cm = mocker.MagicMock()
    client_cm.__enter__.return_value.search.return_value = {"raw": "payload"}
    mocker.patch("rdt_cli.client.RedditClient", return_value=client_cm)
    mocker.patch("rdt_cli.parser.parse_listing", return_value=listing)

    result = RedditSearchTool().run(query="anything", limit=1)

    snippet = result.data[0]["snippet"]
    assert snippet.endswith(" […]")
    assert len(snippet) <= RedditSearchTool.MAX_SNIPPET_CHARS + len(" […]")


def test_x_search_formats_tweets(mocker: Any) -> None:
    settings = mocker.patch("mascan.agents.social.tools.x_api.get_settings").return_value
    settings.twitter_auth_token = "token"
    settings.twitter_ct0 = "csrf"

    client = mocker.patch("twitter_cli.client.TwitterClient").return_value
    client.fetch_search.return_value = ["tweet-obj"]
    mocker.patch(
        "twitter_cli.serialization.tweets_to_data",
        return_value=[
            {
                "id": "123",
                "text": "EV battery recycling is improving",
                "author": {"screenName": "analyst"},
                "createdAtISO": "2026-06-05T00:00:00Z",
                "metrics": {"likes": 5},
            }
        ],
    )

    result = XSearchTool().run(query="EV battery recycling", max_results=5)

    assert result.success
    assert result.source == "x:twitter_cli_search"
    assert result.metadata["provider"] == "twitter-cli"
    assert result.metadata["count"] == 1
    assert result.data is not None
    assert result.data[0]["text"] == "EV battery recycling is improving"
    assert result.data[0]["author"] == "analyst"
    assert result.data[0]["url"] == "https://x.com/analyst/status/123"
    # The full raw tweet object is no longer attached (context-overflow guard).
    assert "raw" not in result.data[0]


def test_world_bank_social_indicators_formats_latest_values(mocker: Any) -> None:
    payload = [
        {"page": 1, "pages": 1},
        [
            {
                "indicator": {"id": "SP.POP.TOTL", "value": "Population, total"},
                "country": {"id": "WLD", "value": "World"},
                "date": "2025",
                "value": 8000000000,
                "unit": "",
            }
        ],
    ]
    response = type("Response", (), {"json": lambda self: payload})()
    mocker.patch("mascan.agents.social.tools.world_bank.http_get", return_value=response)

    result = WorldBankSocialIndicatorsTool().run(
        country_code="WLD",
        indicators=["SP.POP.TOTL"],
    )

    assert result.success
    assert result.source == "world_bank:social_indicators"
    assert result.metadata["provider"] == "World Bank Indicators API"
    assert result.metadata["indicator_count"] == 1
    assert result.data is not None
    assert result.data[0]["indicator_code"] == "SP.POP.TOTL"
    assert result.data[0]["country_name"] == "World"
    assert result.data[0]["value"] == 8000000000
    assert result.data[0]["api_url"] == (
        "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL?format=json&per_page=5"
    )
    assert result.data[0]["url"] == (
        "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=1W"
    )
    assert result.metadata["source_urls"] == [
        "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=1W"
    ]


def test_world_bank_social_indicators_supports_multiple_countries_and_indicators(
    mocker: Any,
) -> None:
    payload = [
        {"page": 1, "pages": 1},
        [
            {
                "indicator": {"id": "SP.POP.TOTL", "value": "Population, total"},
                "country": {"id": "DEU", "value": "Germany"},
                "date": "2025",
                "value": 1,
                "unit": "",
            }
        ],
    ]
    response = type("Response", (), {"json": lambda self: payload})()
    mock_get = mocker.patch("mascan.agents.social.tools.world_bank.http_get", return_value=response)

    result = WorldBankSocialIndicatorsTool().run(
        country_codes=["DEU", "USA"],
        indicators=["SP.POP.TOTL", "SL.UEM.TOTL.ZS"],
    )

    assert result.success
    assert result.metadata["country_code"] is None
    assert result.metadata["country_codes"] == ["DEU", "USA"]
    assert result.metadata["indicator_count"] == 4
    assert result.data is not None
    assert len(result.data) == 4
    assert len(result.metadata["source_urls"]) == 4
    assert mock_get.call_count == 4
