"""Render the methodology/status appendix for a gold-standard experiment."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.methodology import write_methodology_appendix
from mascan.eval.readiness import (
    ReadinessReport,
    load_experiment_manifest,
    validate_experiment_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render gold methodology appendix.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory used to resolve relative paths in the manifest.",
    )
    parser.add_argument(
        "--readiness",
        default=None,
        help="Optional readiness_report.json. If omitted, readiness is computed.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = load_experiment_manifest(args.manifest)
    if args.readiness:
        readiness = ReadinessReport.model_validate_json(
            Path(args.readiness).read_text(encoding="utf-8")
        )
    else:
        readiness = validate_experiment_manifest(manifest, base_dir=args.base_dir)

    write_methodology_appendix(args.out, manifest, readiness=readiness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
