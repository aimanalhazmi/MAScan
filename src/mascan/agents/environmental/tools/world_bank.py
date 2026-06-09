from typing import Any

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.http_client import http_get

DEFAULT_ENV_INDICATORS: dict[str, str] = {
    "AG.LND.FRST.ZS": "Forest area (% of land area)",
    "ER.H2O.FWST.ZS": (
        "Level of water stress: freshwater withdrawal as a proportion of "
        "available freshwater resources"
    ),
    "EN.GHG.CO2.PC.CE.AR5": "CO2 emissions excluding LULUCF per capita (t CO2e/capita)",
    "EN.GHG.CO2.MT.CE.AR5": "CO2 emissions (total) excluding LULUCF (Mt CO2e)",
}

COUNTRY_ALIASES: dict[str, str] = {
    "argentina": "ARG",
    "australia": "AUS",
    "brazil": "BRA",
    "canada": "CAN",
    "china": "CHN",
    "chinese": "CHN",
    "european union": "EUU",
    "eu": "EUU",
    "france": "FRA",
    "germany": "DEU",
    "deutschland": "DEU",
    "india": "IND",
    "indonesia": "IDN",
    "italy": "ITA",
    "japan": "JPN",
    "mexico": "MEX",
    "netherlands": "NLD",
    "russia": "RUS",
    "russian federation": "RUS",
    "saudi arabia": "SAU",
    "south africa": "ZAF",
    "south korea": "KOR",
    "korea": "KOR",
    "spain": "ESP",
    "turkey": "TUR",
    "turkiye": "TUR",
    "united kingdom": "GBR",
    "uk": "GBR",
    "britain": "GBR",
    "united states": "USA",
    "usa": "USA",
    "u.s.": "USA",
    "u.s.a.": "USA",
    "us": "USA",
    "world": "WLD",
    "global": "WLD",
    "worldwide": "WLD",
}


class WorldBankEnvironmentalIndicatorsInput(BaseModel):
    country_code: str = Field(
        "WLD",
        description=(
            "World Bank country code or plain country name, e.g. WLD for world, "
            "DEU/Germany, USA/United States, CHN/China, BRA/Brazil."
        ),
    )
    indicators: list[str] | None = Field(
        None,
        description=(
            "Optional World Bank indicator codes. Uses an environmental baseline set "
            "(forest coverage, water stress, CO2 emissions) if omitted."
        ),
    )
    country_codes: list[str] | None = Field(
        None,
        description=(
            "Optional list of World Bank country codes or names. "
            "Overrides country_code when provided."
        ),
    )


class WorldBankEnvironmentalIndicatorsTool(BaseTool):
    name = "world_bank_environmental_indicators"
    description = (
        "Fetch official World Bank environmental indicators by country: CO2 emissions, "
        "water stress / freshwater withdrawal, and forest coverage. Free, no API key. "
        "Use for water-stress, deforestation, or emissions signals by country."
    )
    input_schema = WorldBankEnvironmentalIndicatorsInput

    def run(
        self,
        country_code: str = "WLD",
        country_codes: list[str] | None = None,
        indicators: list[str] | None = None,
        **_: Any,
    ) -> ToolResult[list[dict[str, Any]]]:
        selected = indicators or list(DEFAULT_ENV_INDICATORS)
        countries = self._normalize_country_codes(country_codes or [country_code])
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for country in countries:
            for indicator in selected:
                api_url = self._api_url(country, indicator)
                try:
                    response = http_get(
                        api_url,
                        params={
                            "format": "json",
                            "per_page": 60,
                        },
                        timeout=15.0,
                    )
                    payload = response.json()
                    latest = self._latest_non_empty_observation(payload)
                    if latest is None:
                        errors.append(f"{country}/{indicator}: no recent value")
                        continue
                    results.append(
                        {
                            "indicator_code": indicator,
                            "indicator_name": self._indicator_name(indicator, latest),
                            "country_code": country,
                            "country_name": latest.get("country", {}).get("value"),
                            "date": latest.get("date"),
                            "value": latest.get("value"),
                            "unit": latest.get("unit") or None,
                            "source_note": latest.get("indicator", {}).get("value"),
                            "api_url": api_url,
                        }
                    )
                except Exception as exc:
                    self.logger.warning(
                        "world_bank_environmental_indicators skipped "
                        "country=%r indicator=%r: %s",
                        country,
                        indicator,
                        exc,
                    )
                    errors.append(f"{country}/{indicator}: {exc}")

        return ToolResult(
            success=bool(results),
            data=results,
            source="world_bank:environmental_indicators",
            error="; ".join(errors) if errors and not results else None,
            metadata={
                "provider": "World Bank Indicators API",
                "country_code": countries[0] if len(countries) == 1 else None,
                "country_codes": countries,
                "indicator_count": len(results),
                "failed_indicators": errors,
                "source_urls": sorted({item["api_url"] for item in results}),
            },
        )

    @staticmethod
    def _normalize_country_codes(country_codes: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in country_codes:
            token = raw.strip()
            country = COUNTRY_ALIASES.get(token.lower(), token.upper())
            if country and country not in normalized:
                normalized.append(country)
        return normalized or ["WLD"]

    @staticmethod
    def _api_url(country: str, indicator: str) -> str:
        return f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

    @staticmethod
    def _latest_non_empty_observation(payload: Any) -> dict[str, Any] | None:
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
    def _indicator_name(indicator: str, observation: dict[str, Any]) -> str:
        name = observation.get("indicator", {}).get("value")
        if isinstance(name, str) and name:
            return name
        return DEFAULT_ENV_INDICATORS.get(indicator, indicator)
