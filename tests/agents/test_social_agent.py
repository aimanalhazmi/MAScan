from typing import Any

from mascan.agents.registry import agent_registry
from mascan.agents.social.agent import (
    ALWAYS_CALL_TOOLS,
    REDDIT_RESULTS_PER_QUERY,
    WEB_RESULTS_PER_QUERY,
    X_RESULTS_PER_QUERY,
    SocialAgent,
    SocialEvidencePlan,
)
from mascan.agents.social.tools.reddit_api import RedditSearchTool
from mascan.agents.social.tools.world_bank import WorldBankSocialIndicatorsTool
from mascan.agents.social.tools.x_api import XSearchTool
from mascan.contracts.reports import AgentReport
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


def test_social_always_calls_web_search_and_world_bank() -> None:
    assert ALWAYS_CALL_TOOLS == ("web_search", "world_bank_social_indicators")


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
                "source_urls": [
                    "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL"
                ],
            },
        ),
    }

    mocker.patch.object(
        agent,
        "gather_deterministic",
        return_value=deterministic_outputs,
    )
    mocker.patch.object(
        agent,
        "run_react_agent",
        return_value=("Social sentiment findings", ["reddit_search"]),
    )

    report = agent.run(tasks=["consumer sentiment around EV battery recycling"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "social"
    assert report.tasks == ["consumer sentiment around EV battery recycling"]
    assert report.findings == "Social sentiment findings"
    assert report.metadata["mode"] == "mixed"
    assert report.metadata["deterministic_tools"] == [
        "web_search",
        "world_bank_social_indicators",
    ]
    assert "## Social Analysis" in report.rendered_markdown
    assert [source.name for source in report.sources] == [
        "web_search:firecrawl",
        "world_bank:social_indicators",
        "reddit_search",
    ]
    assert report.sources[0].metadata["count"] == 1
    assert "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL" in (
        report.rendered_markdown
    )


def test_social_agent_constrains_evidence_plan() -> None:
    plan = SocialEvidencePlan(
        country_codes=["deu", "USA", "DEU", "CHN"],
        web_queries=[" germany ev recycling sentiment ", "battery recycling risk", "x", "y"],
        reddit_queries=["reddit one", "reddit two", "reddit three", "reddit four"],
        x_queries=["x one", "x one", "x two", "x three"],
    )

    constrained = SocialAgent.constrain_evidence_plan(plan)

    assert constrained.country_codes == ["DEU", "USA", "CHN"]
    assert constrained.web_queries == [
        "germany ev recycling sentiment",
        "battery recycling risk",
        "x",
    ]
    assert constrained.reddit_queries == ["reddit one", "reddit two", "reddit three"]
    assert constrained.x_queries == ["x one", "x two", "x three"]


def test_social_agent_uses_planned_queries_and_countries(mocker: Any) -> None:
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
    reddit.run.return_value = ToolResult(success=True, data=[], source="reddit:test")
    x_search.run.return_value = ToolResult(success=True, data=[], source="x:test")
    agent.tools = {
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
            web_queries=["germany ev battery recycling consumer sentiment", "recycling social risk"],
            reddit_queries=["EV battery recycling Germany"],
            x_queries=["EV battery recycling Germany sentiment"],
        ),
    )

    agent.gather_deterministic(["Analyze Germany and United States labour trends"])

    world_bank.run.assert_called_once_with(country_codes=["DEU", "USA"])
    assert web_search.run.call_count == 2
    web_search.run.assert_any_call(
        query="germany ev battery recycling consumer sentiment",
        max_results=WEB_RESULTS_PER_QUERY,
    )
    reddit.run.assert_called_once_with(
        query="EV battery recycling Germany",
        limit=REDDIT_RESULTS_PER_QUERY,
    )
    x_search.run.assert_called_once_with(
        query="EV battery recycling Germany sentiment",
        max_results=X_RESULTS_PER_QUERY,
    )


def test_x_search_without_cli_returns_failure(mocker: Any) -> None:
    mocker.patch("mascan.agents.social.tools.x_api.shutil.which", return_value=None)
    mocker.patch("mascan.agents.social.tools.x_api.Path.exists", return_value=False)

    result = XSearchTool().run(query="EV battery recycling")

    assert not result.success
    assert result.source == "x:twitter_cli_search"
    assert result.error == "twitter-cli is not installed or not on PATH."
    assert result.metadata["platform"] == "x"
    assert result.metadata["provider"] == "twitter-cli"
    assert result.metadata["query"] == "EV battery recycling"
    assert result.metadata["count"] == 0


def test_reddit_search_uses_rdt_cli_json(mocker: Any) -> None:
    completed = type(
        "Completed",
        (),
        {
            "stdout": (
                '[{"id":"abc123","title":"Battery recycling concerns",'
                '"subreddit":"r/electricvehicles","score":10,"comments":3,'
                '"url":"https://reddit.com/r/electricvehicles/comments/abc123",'
                '"snippet":"People discuss recycling options."}]'
            ),
            "stderr": "",
        },
    )()
    mocker.patch("mascan.agents.social.tools.reddit_api.shutil.which", return_value="/bin/rdt")
    mock_run = mocker.patch(
        "mascan.agents.social.tools.reddit_api.subprocess.run",
        return_value=completed,
    )

    result = RedditSearchTool().run(query="EV battery recycling", limit=5)

    assert result.success
    assert result.source == "reddit:rdt_cli_search"
    assert result.metadata["provider"] == "rdt-cli"
    assert result.metadata["count"] == 1
    assert result.data is not None
    assert result.data[0]["title"] == "Battery recycling concerns"
    assert result.data[0]["subreddit"] == "r/electricvehicles"
    mock_run.assert_called_once()


def test_x_search_uses_twitter_cli_json(mocker: Any) -> None:
    completed = type(
        "Completed",
        (),
        {
            "stdout": (
                '[{"id":"123","text":"EV battery recycling is improving",'
                '"username":"analyst","created_at":"2026-06-05",'
                '"url":"https://x.com/analyst/status/123"}]'
            ),
            "stderr": "",
        },
    )()
    mocker.patch("mascan.agents.social.tools.x_api.shutil.which", return_value="/bin/twitter")
    mock_run = mocker.patch(
        "mascan.agents.social.tools.x_api.subprocess.run",
        return_value=completed,
    )

    result = XSearchTool().run(query="EV battery recycling", max_results=5)

    assert result.success
    assert result.source == "x:twitter_cli_search"
    assert result.metadata["provider"] == "twitter-cli"
    assert result.metadata["count"] == 1
    assert result.data is not None
    assert result.data[0]["text"] == "EV battery recycling is improving"
    assert result.data[0]["author"] == "analyst"
    mock_run.assert_called_once()


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
        "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL"
    )
    assert result.metadata["source_urls"] == [
        "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL"
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
