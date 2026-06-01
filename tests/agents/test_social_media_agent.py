from typing import Any

from mascan.agents.registry import agent_registry
from mascan.agents.social_media.agent import ALWAYS_CALL_TOOLS, SocialMediaAgent
from mascan.agents.social_media.tools.x_api import XSearchTool
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult


def test_social_media_agent_is_registered() -> None:
    import mascan.agents.social_media  # noqa: F401

    assert "social_media" in agent_registry.all_names()


def test_social_media_agent_loads_reddit_x_and_web_tools() -> None:
    import mascan.agents.social_media  # noqa: F401

    agent = SocialMediaAgent()

    assert "reddit_search" in agent.tools
    assert "x_search" in agent.tools
    assert "web_search" in agent.tools


def test_social_media_always_calls_only_reddit() -> None:
    assert ALWAYS_CALL_TOOLS == ("reddit_search",)


def test_social_media_agent_run_returns_report(mocker: Any) -> None:
    import mascan.agents.social_media  # noqa: F401

    agent = SocialMediaAgent()
    deterministic_outputs = {
        "reddit_search": ToolResult(
            success=True,
            data=[
                {
                    "title": "EV battery recycling discussion",
                    "subreddit": "r/electricvehicles",
                    "score": 42,
                    "comments": 12,
                    "url": "https://www.reddit.com/r/electricvehicles/example",
                    "snippet": "Users discuss recycling concerns.",
                }
            ],
            source="reddit:search",
            metadata={
                "platform": "reddit",
                "provider": "reddit_public_json",
                "query": "consumer sentiment around EV battery recycling",
                "count": 1,
            },
        )
    }

    mocker.patch.object(
        agent,
        "gather_deterministic",
        return_value=deterministic_outputs,
    )
    mocker.patch.object(
        agent,
        "run_react_agent",
        return_value=("Social sentiment findings", ["web_search"]),
    )

    report = agent.run(tasks=["consumer sentiment around EV battery recycling"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "social_media"
    assert report.tasks == ["consumer sentiment around EV battery recycling"]
    assert report.findings == "Social sentiment findings"
    assert report.metadata["mode"] == "mixed"
    assert report.metadata["deterministic_tools"] == ["reddit_search"]
    assert "## Social Media Analysis" in report.rendered_markdown
    assert [source.name for source in report.sources] == ["reddit:search", "web_search"]
    assert report.sources[0].metadata["platform"] == "reddit"
    assert report.sources[0].metadata["count"] == 1


def test_x_search_without_token_returns_failure(mocker: Any) -> None:
    mocker.patch("mascan.agents.social_media.tools.x_api.os.getenv", return_value=None)

    result = XSearchTool().run(query="EV battery recycling")

    assert not result.success
    assert result.source == "x:recent_search"
    assert result.error == "X_BEARER_TOKEN is not configured."
    assert result.metadata["platform"] == "x"
    assert result.metadata["provider"] == "x_api_v2"
    assert result.metadata["query"] == "EV battery recycling"
    assert result.metadata["count"] == 0
