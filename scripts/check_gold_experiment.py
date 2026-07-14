"""Validate that a gold-standard experiment run has all required artifacts."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.readiness import load_experiment_manifest, validate_experiment_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Check gold experiment readiness.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory used to resolve relative paths in the manifest.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    manifest = load_experiment_manifest(args.manifest)
    report = validate_experiment_manifest(manifest, base_dir=args.base_dir)
    rendered = report.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report.is_ready else 1


if __name__ == "__main__":
    sys.exit(main())
