"""Reproduce Table A2-1 from the final labels and fixed split IDs.

Run with Python's standard library only. Sentiment labels are -1 (negative),
0 (neutral) and 1 (positive). Each class is shown as a count and its percentage
of the corresponding dataset, rounded to two decimal places.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
STEM = "Table_A2_1_Class_Distribution"
INPUT_COLUMNS = ("sample_id", "final_label")
CLASSES = ((-1, "Negative"), (0, "Neutral"), (1, "Positive"))
SPLITS = (
    ("train", 7000, "Training set"),
    ("validation", 2000, "Validation set"),
    ("test", 1000, "Held-out test set"),
)
OUTPUT_COLUMNS = ("Data set", "N", "Negative", "Neutral", "Positive")
CAPTION = "Table A2-1. Class distribution of the final labeled sample and fixed split"
NOTE = (
    "Entries for sentiment classes are counts (percentages). Labels are "
    "negative (−1), neutral (0) and positive (1); N is the number of observations "
    "in the corresponding dataset. The 500-per-class manual-audit sampling "
    "allocation is distinct from these final-label frequencies."
)


def read_labels(path: Path):
    labels = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        missing = set(INPUT_COLUMNS) - set(headers)
        if missing:
            raise ValueError(f"{path.name}: missing columns: {', '.join(sorted(missing))}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path.name}: duplicate column names")
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("missing or extra CSV fields")
                sample_id = row["sample_id"]
                if not sample_id.strip():
                    raise ValueError("sample_id must be nonempty")
                if sample_id in labels:
                    raise ValueError(f"duplicate sample_id: {sample_id!r}")
                label = int(row["final_label"])
                if label not in (-1, 0, 1):
                    raise ValueError("final_label must be -1, 0 or 1")
                labels[sample_id] = label
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}, CSV line {reader.line_num}: {error}") from error
    expected = sum(size for _, size, _ in SPLITS)
    if len(labels) != expected:
        raise ValueError(f"{path.name}: expected {expected} final labels, got {len(labels)}")
    return labels


def read_splits(path: Path, labels):
    splits = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(splits, dict) or set(splits) != {name for name, _, _ in SPLITS}:
        raise ValueError(f"{path.name}: expected train, validation and test arrays")
    seen = set()
    for name, expected, _ in SPLITS:
        ids = splits[name]
        if not isinstance(ids, list) or len(ids) != expected:
            raise ValueError(f"{path.name}, {name}: expected an array of {expected} sample IDs")
        if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in ids):
            raise ValueError(f"{path.name}, {name}: sample IDs must be nonempty strings")
        unique_ids = set(ids)
        if len(unique_ids) != len(ids):
            raise ValueError(f"{path.name}, {name}: duplicate sample IDs")
        if unique_ids - labels.keys():
            raise ValueError(f"{path.name}, {name}: unknown sample IDs")
        if unique_ids & seen:
            raise ValueError(f"{path.name}, {name}: splits must be disjoint")
        seen.update(unique_ids)
    if seen != labels.keys():
        raise ValueError(f"{path.name}: splits must cover every final labeled sample exactly once")
    return splits


def calculate_table(data_directory: Path):
    labels = read_labels(data_directory / "sentiment_labels.csv")
    splits = read_splits(data_directory / "split_ids.json", labels)
    datasets = [("Final labeled sample", labels)]
    datasets.extend((title, splits[name]) for name, _, title in SPLITS)
    rows, counts = [], {}
    for title, ids in datasets:
        size = len(ids)
        frequencies = Counter(labels[sample_id] for sample_id in ids)
        rows.append({
            "Data set": title,
            "N": f"{size:,}",
            **{column: f"{frequencies[label]:,} ({100 * frequencies[label] / size:.2f}%)"
               for label, column in CLASSES},
        })
        counts[title] = {"N": size, **{column: frequencies[label] for label, column in CLASSES}}
    return rows, counts


def write_outputs(rows, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{STEM}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {CAPTION}", "",
        "| " + " | ".join(OUTPUT_COLUMNS) + " |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(row[column] for column in OUTPUT_COLUMNS) + " |" for row in rows)
    lines.extend(["", f"Note: {NOTE}", ""])
    md_path = output_directory / f"{STEM}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Table A2-1 from final sentiment labels and fixed split IDs.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data-raw" / "sentiment_classification",
                        help="Directory containing sentiment_labels.csv and split_ids.json.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table A2-1 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        rows, counts = calculate_table(arguments.data_dir)
        outputs = write_outputs(rows, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Table A2-1 failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated {counts['Final labeled sample']['N']:,} final labels and "
          "the 7,000/2,000/1,000 disjoint split.")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
