"""EurLexTool — search EU legislation via the public CELLAR SPARQL endpoint.

EUR-Lex / CELLAR is the EU's authoritative store of legal acts (regulations,
directives, decisions) published in the Official Journal. The CELLAR SPARQL
endpoint is public and requires no API key — the EU analogue of the U.S.
``federal_register`` tool.

Matching uses Virtuoso's full-text index (``bif:contains``) over English
expression titles, which keeps the query fast against the very large triple
store. Each hit is identified by its CELEX number, from which a stable EUR-Lex
URL is built.

Endpoint: https://publications.europa.eu/webapi/rdf/sparql
Ontology: http://publications.europa.eu/ontology/cdm# (Common Data Model)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.http_client import http_get

_ENGLISH = "<http://publications.europa.eu/resource/authority/language/ENG>"


class EurLexInput(BaseModel):
    """Arguments the LLM supplies when it chooses to call this tool."""

    query: str = Field(description="Keyword(s) to match against EU legislation titles, e.g. 'data protection'.")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of legal acts to return.")


class EurLexTool(BaseTool):
    name = "eur_lex"
    description = (
        "Search official EU legislation (regulations, directives, decisions) "
        "from EUR-Lex/CELLAR by keyword. Returns CELEX ids, titles, dates and "
        "EUR-Lex URLs. No API key required."
    )

    input_schema = EurLexInput

    ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
    RESULT_FORMAT = "application/sparql-results+json"

    def run(self, query: str, limit: int = 5, **_: Any) -> ToolResult[dict[str, Any]]:
        """Search EU legislation by keyword.

        Args:
            query: Keyword(s) matched against English legislation titles.
            limit: Maximum number of legal acts to return.
        """
        try:
            sparql = self._build_query(query, limit)
            response = http_get(
                self.ENDPOINT,
                params={"query": sparql, "format": self.RESULT_FORMAT},
            )
            payload = response.json()

            bindings = payload.get("results", {}).get("bindings", [])
            documents = self._dedupe_by_celex(self._parse(b) for b in bindings)
            data = {
                "query": query,
                "returned": len(documents),
                "documents": documents,
            }
            return ToolResult(
                success=True,
                data=data,
                source=f"eur_lex:{query}",
                metadata={
                    "provider": "eur-lex.europa.eu",
                    "returned": len(documents),
                    "limit": limit,
                },
            )
        except Exception as exc:
            self.logger.exception("eur_lex failed for query=%r", query)
            return ToolResult(
                success=False,
                source=f"eur_lex:{query}",
                error=str(exc),
                metadata={"provider": "eur-lex.europa.eu"},
            )

    def _build_query(self, query: str, limit: int) -> str:
        """Build a SPARQL query against the CDM ontology.

        Restricts to legal acts (those carrying a CELEX id) with an English
        expression, full-text matched by title and ordered newest-first.
        """
        keyword = self._escape(query)
        return (
            "PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>\n"
            "SELECT DISTINCT ?work ?celex ?title ?date WHERE {\n"
            "  ?work cdm:resource_legal_id_celex ?celex .\n"
            "  ?exp cdm:expression_belongs_to_work ?work ;\n"
            f"       cdm:expression_uses_language {_ENGLISH} ;\n"
            "       cdm:expression_title ?title .\n"
            "  OPTIONAL { ?work cdm:work_date_document ?date }\n"
            f'  ?title bif:contains "\'{keyword}\'" .\n'
            "}\n"
            "ORDER BY DESC(?date)\n"
            f"LIMIT {int(limit)}"
        )

    @staticmethod
    def _escape(query: str) -> str:
        """Neutralise quotes/backslashes so the keyword can't break the SPARQL
        string or the bif:contains literal."""
        return query.replace("\\", " ").replace('"', " ").replace("'", " ").strip()

    @staticmethod
    def _dedupe_by_celex(documents: Any) -> list[dict[str, Any]]:
        """Keep the first document per CELEX id (a single act yields one row per
        title literal in CELLAR). Order is preserved."""
        seen: set[str | None] = set()
        unique: list[dict[str, Any]] = []
        for doc in documents:
            celex = doc["celex"]
            if celex in seen:
                continue
            seen.add(celex)
            unique.append(doc)
        return unique

    @staticmethod
    def _parse(binding: dict[str, Any]) -> dict[str, Any]:
        """Flatten one SPARQL result binding into a compact document dict."""
        celex = EurLexTool._value(binding, "celex")
        return {
            "celex": celex,
            "title": EurLexTool._value(binding, "title"),
            "date": EurLexTool._value(binding, "date"),
            "url": (
                f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
                if celex
                else None
            ),
        }

    @staticmethod
    def _value(binding: dict[str, Any], key: str) -> str | None:
        cell = binding.get(key)
        return cell.get("value") if cell else None
