from typing import Any

import httpx

from mascan.agents.legal.tools.eur_lex import EurLexTool

# Minimal SPARQL
SAMPLE_PAYLOAD = {
    "head": {"vars": ["work", "celex", "title", "date"]},
    "results": {
        "bindings": [
            {
                "work": {"type": "uri", "value": "http://publications.europa.eu/resource/celex/32016R0679"},
                "celex": {"type": "literal", "value": "32016R0679"},
                "title": {
                    "type": "literal",
                    "value": "Regulation (EU) 2016/679 ... (General Data Protection Regulation)",
                },
                "date": {"type": "literal", "value": "2016-04-27"},
            }
        ]
    },
}


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def test_eur_lex_parses_results(mocker: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_http_get(url: str, params: dict[str, Any] | None = None, **_: Any) -> _FakeResponse:
        captured["url"] = url
        captured["params"] = params or {}
        return _FakeResponse(SAMPLE_PAYLOAD)

    mocker.patch("mascan.agents.legal.tools.eur_lex.http_get", side_effect=fake_http_get)

    result = EurLexTool().run(query="data protection", limit=5)

    assert result.success is True
    assert result.source == "eur_lex:data protection"
    assert result.metadata["provider"] == "eur-lex.europa.eu"

    data = result.data
    assert data["query"] == "data protection"
    assert data["returned"] == 1
    assert data["documents"] == [
        {
            "celex": "32016R0679",
            "title": "Regulation (EU) 2016/679 ... (General Data Protection Regulation)",
            "date": "2016-04-27",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679",
        }
    ]

    # JSON result format
    assert captured["url"] == EurLexTool.ENDPOINT
    sparql = captured["params"]["query"]
    assert "data protection" in sparql
    assert "bif:contains" in sparql
    assert "resource_legal_id_celex" in sparql
    assert "LIMIT 5" in sparql
    assert captured["params"]["format"] == "application/sparql-results+json"


def test_eur_lex_deduplicates_by_celex(mocker: Any) -> None:
    """CELLAR returns one row per title literal, so the same act can appear
    multiple times. The tool must return each CELEX act only once."""
    payload = {
        "results": {
            "bindings": [
                {
                    "celex": {"value": "62025TO0144"},
                    "title": {"value": "Order of the General Court ... #WS v Eu"},
                    "date": {"value": "2026-03-25"},
                },
                {
                    "celex": {"value": "62025TO0144"},
                    "title": {"value": "Order of the General Court ... #another fragment"},
                    "date": {"value": "2026-03-25"},
                },
            ]
        }
    }
    mocker.patch(
        "mascan.agents.legal.tools.eur_lex.http_get",
        return_value=_FakeResponse(payload),
    )

    result = EurLexTool().run(query="order")

    assert result.success is True
    assert result.data["returned"] == 1
    assert [d["celex"] for d in result.data["documents"]] == ["62025TO0144"]


def test_eur_lex_handles_http_error(mocker: Any) -> None:
    mocker.patch(
        "mascan.agents.legal.tools.eur_lex.http_get",
        side_effect=httpx.HTTPStatusError("boom", request=mocker.Mock(), response=mocker.Mock()),
    )

    result = EurLexTool().run(query="antitrust")

    assert result.success is False
    assert result.source == "eur_lex:antitrust"
    assert result.error
    assert result.metadata["provider"] == "eur-lex.europa.eu"


def test_eur_lex_escapes_quotes_in_query(mocker: Any) -> None:
    """A query containing a single quote must not break the SPARQL string."""
    captured: dict[str, Any] = {}

    def fake_http_get(url: str, params: dict[str, Any] | None = None, **_: Any) -> _FakeResponse:
        captured["params"] = params or {}
        return _FakeResponse({"results": {"bindings": []}})

    mocker.patch("mascan.agents.legal.tools.eur_lex.http_get", side_effect=fake_http_get)

    result = EurLexTool().run(query="workers' rights")

    assert result.success is True
    
    sparql = captured["params"]["query"]
    assert "workers" in sparql
