from typing import Any

from mascan.agents.environmental.agent import EnvironmentalAgent
from mascan.agents.environmental.tools.world_bank import (
    WorldBankEnvironmentalIndicatorsTool,
)
from mascan.agents.registry import agent_registry
from mascan.contracts.reports import AgentReport
from mascan.tools.registry import tool_registry


def test_environmental_agent_is_registered() -> None:
    import mascan.agents.environmental  # noqa: F401

    assert "environmental" in agent_registry.all_names()


def test_environmental_world_bank_tool_is_registered_and_loaded() -> None:
    import mascan.agents.environmental  # noqa: F401

    assert isinstance(
        tool_registry.get("world_bank_environmental_indicators"),
        WorldBankEnvironmentalIndicatorsTool,
    )
    agent = EnvironmentalAgent()
    assert "world_bank_environmental_indicators" in agent.tools


def test_world_bank_environmental_indicators_formats_latest_values(mocker: Any) -> None:
    payload = [
        {"page": 1, "pages": 1},
        [
            {
                "indicator": {
                    "id": "AG.LND.FRST.ZS",
                    "value": "Forest area (% of land area)",
                },
                "country": {"id": "BR", "value": "Brazil"},
                "date": "2025",
                "value": None,
                "unit": "",
            },
            {
                "indicator": {
                    "id": "AG.LND.FRST.ZS",
                    "value": "Forest area (% of land area)",
                },
                "country": {"id": "BR", "value": "Brazil"},
                "date": "2022",
                "value": 59.4,
                "unit": "",
            },
        ],
    ]
    response = type("Response", (), {"json": lambda self: payload})()
    mocker.patch(
        "mascan.agents.environmental.tools.world_bank.http_get",
        return_value=response,
    )

    result = WorldBankEnvironmentalIndicatorsTool().run(
        country_code="Brazil",
        indicators=["AG.LND.FRST.ZS"],
    )

    assert result.success
    assert result.source == "world_bank:environmental_indicators"
    assert result.metadata["provider"] == "World Bank Indicators API"
    assert result.metadata["indicator_count"] == 1
    assert result.data is not None
    # plain country name resolves to ISO3 code
    assert result.data[0]["country_code"] == "BRA"
    assert result.data[0]["country_name"] == "Brazil"
    # skips the null 2025 value and picks the latest non-empty observation
    assert result.data[0]["date"] == "2022"
    assert result.data[0]["value"] == 59.4
    assert result.data[0]["api_url"] == (
        "https://api.worldbank.org/v2/country/BRA/indicator/AG.LND.FRST.ZS"
        "?format=json&per_page=60"
    )
    assert result.data[0]["url"] == (
        "https://data.worldbank.org/indicator/AG.LND.FRST.ZS?locations=BR"
    )


def test_world_bank_environmental_indicators_uses_default_set(mocker: Any) -> None:
    payload = [
        {"page": 1, "pages": 1},
        [
            {
                "indicator": {"id": "X", "value": "X"},
                "country": {"id": "WLD", "value": "World"},
                "date": "2024",
                "value": 1,
                "unit": "",
            }
        ],
    ]
    response = type("Response", (), {"json": lambda self: payload})()
    mock_get = mocker.patch(
        "mascan.agents.environmental.tools.world_bank.http_get",
        return_value=response,
    )

    result = WorldBankEnvironmentalIndicatorsTool().run()

    assert result.success
    # four baseline indicators: forest, water stress, CO2 per capita, CO2 total
    assert mock_get.call_count == 4
    assert result.metadata["indicator_count"] == 4


def test_environmental_agent_run_returns_report(mocker: Any) -> None:
    agent = EnvironmentalAgent()

    fake_result = {
        "messages": [
            type(
                "AIMessage",
                (),
                {"content": "Environmental risk findings", "tool_calls": []},
            )()
        ]
    }
    fake_agent = mocker.Mock()
    fake_agent.invoke.return_value = fake_result

    mocker.patch("mascan.agents.environmental.agent.get_chat_model")
    mocker.patch(
        "mascan.agents.environmental.agent.create_agent",
        return_value=fake_agent,
    )

    report = agent.run(tasks=["water stress in German manufacturing regions"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "environmental"
    assert report.tasks == ["water stress in German manufacturing regions"]
    assert report.findings == "Environmental risk findings"
    assert report.metadata["mode"] == "B — LLM-driven"
    assert report.metadata["deterministic_tools"] == []
    assert report.metadata["llm_chosen_tools"] == []
    assert "## Environmental Analysis" in report.rendered_markdown
    fake_agent.invoke.assert_called_once()
