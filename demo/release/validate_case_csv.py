#!/usr/bin/env python3
"""Validate a local AbdomenNet case CSV before running tutorials."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _truthy(value: str) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y", "positive", "pos"}


def validate_case_csv(
    csv_path: Path,
    expected_cases: int | None,
    min_positive_labels: int | None,
    check_files: bool,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Case CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Case CSV is empty: {csv_path}")

    fieldnames = list(rows[0].keys())
    if len(fieldnames) < 2:
        raise ValueError("Case CSV must contain an image path column and at least one label column.")

    if expected_cases is not None and len(rows) != expected_cases:
        raise ValueError(f"Expected {expected_cases} cases, found {len(rows)} in {csv_path}.")

    image_column = fieldnames[0]
    label_columns = fieldnames[1:]
    csv_dir = csv_path.parent
    positive_labels = set()
    missing_files = []

    for row_index, row in enumerate(rows, start=1):
        image_path = Path(str(row[image_column]).strip())
        if check_files and not image_path.is_absolute():
            image_path = csv_dir / image_path
        if check_files and not image_path.exists():
            missing_files.append(str(image_path))

        row_positive_labels = [label for label in label_columns if _truthy(row.get(label, ""))]
        if not row_positive_labels:
            raise ValueError(f"Case CSV row {row_index} has no positive diagnosis label.")
        positive_labels.update(row_positive_labels)

    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing image files:\n{missing}")

    if min_positive_labels is not None and len(positive_labels) < min_positive_labels:
        raise ValueError(
            "Case CSV should span multiple positive labels; "
            f"found {len(positive_labels)} distinct positive label(s): {sorted(positive_labels)}"
        )

    print(
        "Case CSV OK: "
        f"{len(rows)} rows, {len(positive_labels)} distinct positive label(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to a local case CSV file.",
    )
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--min-positive-labels", type=int, default=None)
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Only validate CSV shape and label diversity, without checking image files.",
    )
    args = parser.parse_args()
    validate_case_csv(
        args.csv_path,
        expected_cases=args.expected_cases,
        min_positive_labels=args.min_positive_labels,
        check_files=not args.skip_file_check,
    )


if __name__ == "__main__":
    main()
