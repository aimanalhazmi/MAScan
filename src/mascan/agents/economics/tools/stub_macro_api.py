"""StubMacroAPITool — template for an external macro-data API tool.

Replace `fetch_impl` with a real call (FRED, World Bank, OECD) when the
teammate building Economics is ready. The shape of this tool is the
template for any external API tool in the system.
"""

from __future__ import annotations

from typing import Any

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool


class StubMacroAPITool(BaseTool):
    name = "stub_macro_api"
    description = "Fetch macroeconomic indicators (stub — replace with FRED/World Bank/OECD)."

    def run(self, query: str, **_: Any) -> ToolResult[Any]:
        try:
            data = self.fetch_impl(query=query)
            return ToolResult(
                success=True,
                data=data,
                source="stub_macro_api",
                metadata={"query": query},
            )
        except Exception as exc:
            self.logger.exception("stub_macro_api failed")
            return ToolResult(success=False, source="stub_macro_api", error=str(exc))

    def fetch_impl(self, query: str) -> dict[str, Any]:
        """Replace this with a real API call.

        Pattern teammates should follow:
          1. Read API key from settings (or constructor).
          2. Use mascan.tools.http_client.http_get for the request.
          3. Parse the response into a dict (or a Pydantic output_schema).
          4. Return the result.
        """
        return {
            "stub": True,
            "indicators": {
                "gdp_growth_yoy_pct": 1.4,
                "inflation_cpi_yoy_pct": 2.6,
                "policy_rate_pct": 4.25,
            },
            "note": f"Stub response for query={query!r}. Replace _fetch_impl with a real API.",
        }
