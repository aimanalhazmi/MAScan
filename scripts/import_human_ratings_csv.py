"""Convert one or more filled human-rating CSV files into ratings JSON."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import csv
import sys
from pathlib import Path

from mascan.eval.human_ratings import human_ratings_from_csv_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Import filled human ratings CSV.")
    parser.add_argument("--csv", nargs="+", required=True, help="Filled CSV file(s).")
    parser.add_argument("--out", required=True, help="Output human_ratings.json path.")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for csv_path in args.csv:
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)

    ratings = human_ratings_from_csv_rows(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ratings.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
