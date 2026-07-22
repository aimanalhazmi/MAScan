"""Live checks for the Environmental agent's World Bank tool.

Runs the tool against the real World Bank API and
prints the outcome of each case so you can eyeball the behaviour we built:
ISO-3 handling, multi-country comparison, global fallback + notice, plain-name
rejection, mixed valid/invalid input, over-limit clamping, the stale disaster
indicator, and liveness of every catalog code.

Run:
    PYTHONPATH=src uv run python scripts/manual_env_checks.py

For LLM-level (full agent) tests, use instead:
    make run-environmental Q="your query here"
"""

from mascan.agents.environmental.tools.world_bank import (
    ENV_INDICATORS,
    WorldBankEnvironmentalIndicatorsTool,
)

tool = WorldBankEnvironmentalIndicatorsTool()


def show(title: str, result) -> None:
    meta = result.metadata
    print(f"\n=== {title} ===")
    print(
        f"success={result.success} "
        f"fallback={meta.get('country_fallback')} "
        f"invalid={meta.get('invalid_countries')} "
        f"count={meta.get('indicator_count')} "
        f"limit_applied={meta.get('limit_applied')}"
    )
    if not result.success:
        print("  error:", result.error)
    for row in result.data:
        if "notice" in row:
            print("  NOTICE:", row["notice"])
        else:
            print(f"  {row['country_code']} {row['indicator_code']} = {row['value']} ({row['date']})")


def main() -> None:
    # Single country, default baseline (4 indicators)
    show("single country DEU, default baseline", tool.run(country_code="DEU"))

    # Specific dimension (emissions) for one country
    show(
        "emissions for USA",
        tool.run(country_code="USA", indicators=["EN.GHG.ALL.MT.CE.AR5", "EN.GHG.CH4.MT.CE.AR5"]),
    )

    # Multi-country comparison
    show(
        "compare CO2 total DEU/FRA/CHN",
        tool.run(country_codes=["DEU", "FRA", "CHN"], indicators=["EN.GHG.CO2.MT.CE.AR5"]),
    )

    # No country -> World aggregate + self-correction notice
    show("no country (global fallback + notice)", tool.run())

    # Plain name -> rejected without an API call
    show("plain name rejected (Germany)", tool.run(country_code="Germany"))

    # Mixed valid + invalid -> valid used, invalid flagged in metadata
    show(
        "mixed valid+invalid (DEU, Germany)",
        tool.run(country_codes=["DEU", "Germany"], indicators=["AG.LND.FRST.ZS"]),
    )

    # Over the limits -> countries clamped to 3, indicators to 8
    show(
        "over-limit (4 countries, all 16 indicators)",
        tool.run(country_codes=["DEU", "FRA", "CHN", "USA"], indicators=list(ENV_INDICATORS)),
    )

    # Stale structural indicator (1990-2009 average)
    show("stale disaster indicator DEU", tool.run(country_code="DEU", indicators=["EN.CLC.MDAT.ZS"]))

    # Liveness of every catalog code (one call each, WLD)
    print("\n=== catalog liveness (WLD, one call per code) ===")
    for code in ENV_INDICATORS:
        result = tool.run(country_code="WLD", indicators=[code])
        ok = result.success and result.metadata["indicator_count"] == 1
        print(f"  {'OK  ' if ok else 'MISS'} {code}")


if __name__ == "__main__":
    main()
