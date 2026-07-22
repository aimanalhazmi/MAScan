import re
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.http_client import http_get

SOURCE = "world_bank:environmental_indicators"

# Verified WDI indicator codes, grouped by the agent's six scope dimensions. Codes
# are opaque and cannot be guessed, so they live here (single source of truth) and
# are exposed to the agent via the tool description; the agent only picks which.
ENV_INDICATORS_BY_DIMENSION: dict[str, dict[str, str]] = {
    "Climate & weather": {
        "AG.LND.PRCP.MM": "Average precipitation (mm/yr)",
    },
    "Emissions (GHG)": {
        "EN.GHG.ALL.MT.CE.AR5": "Total GHG excl. LULUCF (Mt CO2e)",
        "EN.GHG.CO2.MT.CE.AR5": "CO2 total excl. LULUCF (Mt CO2e)",
        "EN.GHG.CO2.PC.CE.AR5": "CO2 per capita excl. LULUCF (t CO2e)",
        "EN.GHG.CH4.MT.CE.AR5": "Methane excl. LULUCF (Mt CO2e)",
        "EN.GHG.N2O.MT.CE.AR5": "Nitrous oxide excl. LULUCF (Mt CO2e)",
    },
    "Natural resources": {
        "ER.H2O.FWST.ZS": "Water stress (withdrawal % of resources)",
        "ER.H2O.FWTL.K3": "Freshwater withdrawals (billion m3)",
        "AG.LND.FRST.ZS": "Forest area (% of land)",
        "AG.LND.ARBL.ZS": "Arable land (% of land)",
        "AG.LND.AGRI.ZS": "Agricultural land (% of land)",
    },
    "Extreme weather & disasters": {
        "EN.CLC.MDAT.ZS": "Pop. affected by disasters (%, 1990-2009 avg)",
    },
    "Air & water quality": {
        "EN.ATM.PM25.MC.M3": "PM2.5 exposure (ug/m3)",
        "SH.H2O.BASW.ZS": "Basic drinking water (% of pop)",
    },
    "Biodiversity & land use": {
        "ER.PTD.TOTL.ZS": "Protected areas (% of territory)",
        "EN.MAM.THRD.NO": "Threatened mammal species (count)",
    },
}

ENV_INDICATORS: dict[str, str] = {
    code: name
    for group in ENV_INDICATORS_BY_DIMENSION.values()
    for code, name in group.items()
}

#  Used only when the agent requests no indicators.
DEFAULT_INDICATORS: tuple[str, ...] = (
    "AG.LND.FRST.ZS",
    "ER.H2O.FWST.ZS",
    "EN.GHG.CO2.PC.CE.AR5",
    "EN.GHG.CO2.MT.CE.AR5",
)

INDICATOR_CATALOG_TEXT = "\n".join(
    f"  {dim}: " + "; ".join(f"{c} ({n})" for c, n in group.items())
    for dim, group in ENV_INDICATORS_BY_DIMENSION.items()
)

DEFAULT_FALLBACK_COUNTRY = "WLD"
COUNTRY_CODE_PATTERN = re.compile(r"^[A-Z]{2,3}$")

FALLBACK_NOTICE = (
    "No country given; values are the World (WLD) aggregate. Re-call with an ISO-3 "
    "`country_code` if the task targets specific countries."
)


class WorldBankEnvironmentalIndicatorsInput(BaseModel):
    country_code: str | None = Field(
        None,
        description=(
            "ISO-3 code of the target country (e.g. DEU, USA, CHN). Translate any "
            "named country yourself; plain names are rejected. Omit only for a global "
            "question (returns the World aggregate)."
        ),
    )
    country_codes: list[str] | None = Field(
        None,
        description="Optional ISO-3 codes to compare countries (max 3); overrides country_code.",
    )
    indicators: list[str] | None = Field(
        None,
        description=(
            "Indicator codes from the catalog in the tool description (max 8). "
            "Defaults to a small baseline if omitted; do not invent codes."
        ),
    )


class WorldBankEnvironmentalIndicatorsTool(BaseTool):
    name = "world_bank_environmental_indicators"
    description = (
        "World Bank environmental indicators (WDI) by country. Set "
        "`country_code` (or `country_codes`, max 3) to ISO-3 codes (DEU, USA, CHN); "
        "omit only for a global (World) result. Pick `indicators` from this catalog "
        "(max 8, only these codes):\n"
        f"{INDICATOR_CATALOG_TEXT}\n"
        "Emissions are AR5. There is no deforestation series and the disaster "
        "indicator is a 1990-2009 average — use web_search for current events, "
        "sub-national or daily data, and projections."
    )
    input_schema = WorldBankEnvironmentalIndicatorsInput
    MAX_COUNTRIES: ClassVar[int] = 3
    MAX_INDICATORS: ClassVar[int] = 8

    def run(
        self,
        country_code: str | None = None,
        country_codes: list[str] | None = None,
        indicators: list[str] | None = None,
        **_: Any,
    ) -> ToolResult[list[dict[str, Any]]]:
        requested_indicators = list(indicators or DEFAULT_INDICATORS)
        raw = country_codes or ([country_code] if country_code else [])
        valid_countries, invalid_countries = self.normalize_country_codes(raw)

        if invalid_countries and not valid_countries:
            return ToolResult(
                success=False,
                data=[],
                source=SOURCE,
                error=(
                    f"Invalid ISO-3 code(s): {', '.join(invalid_countries)}. "
                    "Re-call using ISO-3 codes (e.g. DEU, USA)."
                ),
                metadata={"invalid_countries": invalid_countries},
            )

        country_fallback = not valid_countries
        countries = (valid_countries or [DEFAULT_FALLBACK_COUNTRY])[: self.MAX_COUNTRIES]
        selected = requested_indicators[: self.MAX_INDICATORS]

        results: list[dict[str, Any]] = []
        errors: list[str] = []
        for country in countries:
            for indicator in selected:
                api_url = self.api_url(country, indicator)
                try:
                    latest = self.latest_non_empty_observation(
                        http_get(api_url, timeout=15.0).json()
                    )
                except Exception as exc:
                    self.logger.warning("world_bank skipped %s/%s: %s", country, indicator, exc)
                    errors.append(f"{country}/{indicator}: {exc}")
                    continue
                if latest is None:
                    errors.append(f"{country}/{indicator}: no recent value")
                    continue
                results.append(
                    {
                        "indicator_code": indicator,
                        "indicator_name": self.indicator_name(indicator, latest),
                        "country_code": country,
                        "country_name": latest.get("country", {}).get("value"),
                        "date": latest.get("date"),
                        "value": latest.get("value"),
                        "unit": latest.get("unit") or None,
                        "source_note": latest.get("indicator", {}).get("value"),
                        "api_url": api_url,
                        "url": f"https://data.worldbank.org/indicator/{indicator}",
                    }
                )

        data: list[dict[str, Any]] = list(results)
        if country_fallback and results:
            data.append({"notice": FALLBACK_NOTICE})

        return ToolResult(
            success=bool(results),
            data=data,
            source=SOURCE,
            error="; ".join(errors) if errors and not results else None,
            metadata={
                "provider": "World Bank Indicators API",
                "country_codes": countries,
                "country_fallback": country_fallback,
                "invalid_countries": invalid_countries,
                "indicator_count": len(results),
                "failed_indicators": errors,
                "source_urls": sorted({item["url"] for item in results}),
                "limit_applied": (
                    len(requested_indicators) > self.MAX_INDICATORS
                    or len(valid_countries) > self.MAX_COUNTRIES
                ),
            },
        )

    @staticmethod
    def normalize_country_codes(
        country_codes: list[str],
    ) -> tuple[list[str], list[str]]:
        """Split inputs into valid ISO-3-shaped codes and invalid tokens (deduped)."""
        valid: list[str] = []
        invalid: list[str] = []
        for raw in country_codes:
            token = (raw or "").strip().upper()
            if not token:
                continue
            target = valid if COUNTRY_CODE_PATTERN.match(token) else invalid
            if token not in target:
                target.append(token)
        return valid, invalid

    @staticmethod
    def api_url(country: str, indicator: str) -> str:
        return (
            f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
            "?format=json&per_page=60"
        )

    @staticmethod
    def latest_non_empty_observation(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, list) or len(payload) < 2:
            return None
        observations = payload[1]
        if not isinstance(observations, list):
            return None
        for observation in observations:
            if isinstance(observation, dict) and observation.get("value") is not None:
                return observation
        return None

    @staticmethod
    def indicator_name(indicator: str, observation: dict[str, Any]) -> str:
        name = observation.get("indicator", {}).get("value")
        if isinstance(name, str) and name:
            return name
        return ENV_INDICATORS.get(indicator, indicator)
