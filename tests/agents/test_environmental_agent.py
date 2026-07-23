from typing import Any

from mascan.agents.environmental.agent import EnvironmentalAgent
from mascan.agents.environmental.tools.world_bank import (
    WorldBankEnvironmentalIndicatorsTool,
)
from mascan.agents.registry import agent_registry
from mascan.contracts.reports import AgentReport
from mascan.tools.registry import tool_registry

HTTP_GET = "mascan.agents.environmental.tools.world_bank.http_get"


def response(payload: Any) -> Any:
    return type("Response", (), {"json": lambda self: payload})()


def test_environmental_agent_is_registered() -> None:
    import mascan.agents.environmental  # noqa: F401

    assert "environmental" in agent_registry.all_names()


def test_environmental_world_bank_tool_is_registered_and_loaded() -> None:
    import mascan.agents.environmental  # noqa: F401

    assert isinstance(
        tool_registry.get("world_bank_environmental_indicators"),
        WorldBankEnvironmentalIndicatorsTool,
    )
    assert "world_bank_environmental_indicators" in EnvironmentalAgent().tools


def test_world_bank_environmental_indicators_formats_latest_values(mocker: Any) -> None:
    observation = {
        "indicator": {"id": "AG.LND.FRST.ZS", "value": "Forest area (% of land area)"},
        "country": {"id": "BR", "value": "Brazil"},
        "unit": "",
    }
    payload = [
        {"page": 1, "pages": 1},
        [
            {**observation, "date": "2025", "value": None},  # null latest -> skipped
            {**observation, "date": "2022", "value": 59.4},
        ],
    ]
    mocker.patch(HTTP_GET, return_value=response(payload))

    result = WorldBankEnvironmentalIndicatorsTool().run(
        country_code="BRA",
        indicators=["AG.LND.FRST.ZS"],
    )

    assert result.success
    assert result.source == "world_bank:environmental_indicators"
    assert result.metadata["indicator_count"] == 1
    record = result.data[0]
    assert record["country_code"] == "BRA"
    assert record["country_name"] == "Brazil"
    assert record["date"] == "2022"
    assert record["value"] == 59.4
    assert record["api_url"] == (
        "https://api.worldbank.org/v2/country/BRA/indicator/AG.LND.FRST.ZS?format=json&per_page=60"
    )
    assert record["url"] == "https://data.worldbank.org/indicator/AG.LND.FRST.ZS"


def test_world_bank_uses_default_set_and_flags_fallback(mocker: Any) -> None:
    payload = [
        {"page": 1, "pages": 1},
        [{"indicator": {"value": "X"}, "country": {"value": "World"}, "date": "2024", "value": 1}],
    ]
    mock_get = mocker.patch(HTTP_GET, return_value=response(payload))

    result = WorldBankEnvironmentalIndicatorsTool().run()

    assert result.success
    assert mock_get.call_count == 4  # baseline indicators
    assert result.metadata["indicator_count"] == 4
    assert result.metadata["country_fallback"] is True
    assert "notice" in result.data[-1]


def test_world_bank_rejects_plain_country_name(mocker: Any) -> None:
    mock_get = mocker.patch(HTTP_GET)

    result = WorldBankEnvironmentalIndicatorsTool().run(country_code="Germany")

    assert not result.success
    assert mock_get.call_count == 0
    assert "ISO-3" in result.error


def test_environmental_agent_run_returns_report(mocker: Any) -> None:
    fake_result = {
        "messages": [
            type("AIMessage", (), {"content": "Environmental risk findings", "tool_calls": []})()
        ]
    }
    fake_agent = mocker.Mock()
    # The ReAct loop is driven via agent.stream(..., stream_mode="values"), which
    # yields successive state snapshots; the last one is used as the final result.
    fake_agent.stream.return_value = [fake_result]
    mocker.patch("mascan.agents.environmental.agent.get_chat_model")
    mocker.patch("mascan.agents.environmental.agent.create_agent", return_value=fake_agent)

    report = EnvironmentalAgent().run(tasks=["water stress in German manufacturing regions"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "environmental"
    assert report.findings == "Environmental risk findings"
    assert report.metadata["mode"] == "B — LLM-driven"
    assert report.metadata["llm_chosen_tools"] == []
    assert "## Environmental Analysis" in report.rendered_markdown
    fake_agent.stream.assert_called_once()
