from typing import Any

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.http_client import http_get

DEFAULT_SOCIAL_INDICATORS: dict[str, str] = {
    "SP.POP.TOTL": "Population, total",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total population)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    "SE.SEC.ENRR": "School enrollment, secondary (% gross)",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force)",
    "SI.POV.GINI": "Gini index",
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


class WorldBankSocialIndicatorsInput(BaseModel):
    country_code: str = Field(
        "WLD",
        description=(
            "World Bank country code, e.g. WLD for world, DEU for Germany, "
            "USA for United States, CHN for China."
        ),
    )
    indicators: list[str] | None = Field(
        None,
        description="Optional World Bank indicator codes. Uses a social baseline set if omitted.",
    )
    country_codes: list[str] | None = Field(
        None,
        description=(
            "Optional list of World Bank country codes. Overrides country_code when provided."
        ),
    )


class WorldBankSocialIndicatorsTool(BaseTool):
    name = "world_bank_social_indicators"
    description = (
        "Fetch official World Bank social baseline indicators such as population, "
        "urbanization, life expectancy, education, unemployment, and inequality."
    )
    input_schema = WorldBankSocialIndicatorsInput

    def run(
        self,
        country_code: str = "WLD",
        country_codes: list[str] | None = None,
        indicators: list[str] | None = None,
        **_: Any,
    ) -> ToolResult[list[dict[str, Any]]]:
        selected = indicators or list(DEFAULT_SOCIAL_INDICATORS)
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
                            "per_page": 5,
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
                        "world_bank_social_indicators skipped country=%r indicator=%r: %s",
                        country,
                        indicator,
                        exc,
                    )
                    errors.append(f"{country}/{indicator}: {exc}")

        return ToolResult(
            success=bool(results),
            data=results,
            source="world_bank:social_indicators",
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
        for country_code in country_codes:
            country = country_code.strip().upper()
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
        return DEFAULT_SOCIAL_INDICATORS.get(indicator, indicator)
