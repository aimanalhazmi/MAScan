"""Run the 3-case market-scenario evaluation (Evonik, Volkswagen, BioNTech).

Uses the same gold pipeline as the 25-case experiment but:
- separate dataset: eval_papers/market_scenario_cases.json
- separate outputs: eval_results/market_scenarios/
- no human-calibration phase (LLM judge + paired stats only)

Examples:
  # Preview commands (free)
  python scripts/run_market_scenario_eval.py

  # Run paid pipeline
  python scripts/run_market_scenario_eval.py --execute

  # Fresh run
  python scripts/run_market_scenario_eval.py --execute --init-pricing
"""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_MANIFEST = "eval_papers/market_scenario_manifest.json"
DEFAULT_OUT_DIR = "eval_results/market_scenarios"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the 3 instructor market scenarios (MAScan vs zero-shot)."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--allow-missing-price", action="store_true")
    parser.add_argument(
        "--init-pricing",
        action="store_true",
        help="Create eval_results/market_scenarios/model_pricing.json before running.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run paid API steps. Omit for dry-run command preview.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --execute, skip steps whose outputs already exist.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="With --execute, bypass preflight gating.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_dir = base / DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.init_pricing or (
        args.execute and not (out_dir / "model_pricing.json").exists()
    ):
        init_argv = [
            sys.executable,
            str(base / "scripts" / "init_gold_pricing.py"),
            "--manifest",
            args.manifest,
            "--source-url",
            "https://openai.com/api/pricing/",
            "--captured-at",
            "2026-07-14",
            "--notes",
            "Market-scenario mini eval (3 cases).",
        ]
        if (out_dir / "model_pricing.json").exists():
            init_argv.append("--overwrite")
        subprocess.run(init_argv, cwd=base, check=True)

    eval_argv = [
        sys.executable,
        str(base / "scripts" / "run_gold_eval.py"),
        "--manifest",
        args.manifest,
        "--base-dir",
        str(base),
        "--preflight-out",
        str(out_dir / "preflight.json"),
        "--preflight-markdown-out",
        str(out_dir / "preflight.md"),
        "--trace-csv-out",
        str(out_dir / "case_trace.csv"),
    ]
    if args.judge_model:
        eval_argv += ["--judge-model", args.judge_model]
    if args.allow_missing_price:
        eval_argv.append("--allow-missing-price")
    if args.execute:
        eval_argv.append("--execute")
    if args.skip_existing:
        eval_argv.append("--skip-existing")
    if args.skip_preflight:
        eval_argv.append("--skip-preflight")

    if not args.execute:
        print(
            json.dumps(
                {
                    "manifest": args.manifest,
                    "cases": [
                        "market_evonik",
                        "market_volkswagen",
                        "market_biontech",
                    ],
                    "outputs_dir": DEFAULT_OUT_DIR,
                    "next": "python scripts/run_market_scenario_eval.py --execute --init-pricing",
                },
                indent=2,
            )
        )

    return subprocess.run(eval_argv, cwd=base).returncode


if __name__ == "__main__":
    sys.exit(main())
