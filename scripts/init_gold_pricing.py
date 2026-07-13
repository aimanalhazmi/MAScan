"""Create a manifest-derived pricing table template for gold experiments."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.costing import pricing_template_for_models
from mascan.eval.readiness import load_experiment_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize gold model pricing JSON.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Defaults to manifest.pricing_file.",
    )
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--captured-at", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = load_experiment_manifest(args.manifest)
    output = args.out or manifest.pricing_file
    if output is None:
        raise ValueError("Provide --out or set pricing_file in the manifest")

    out_path = Path(output)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Pricing file already exists: {out_path}")

    pricing = pricing_template_for_models(
        [system.model for system in manifest.systems],
        source_url=args.source_url,
        captured_at=args.captured_at,
        notes=args.notes,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(pricing.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
