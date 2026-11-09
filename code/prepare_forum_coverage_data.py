"""Prepare public aggregate inputs for Table A14 from retained forum CSV files.

Input files are processed_1.csv through processed_15.csv after basic
title-validity screening. Records are counted without URL deduplication.
Export bank/subperiod counts and separate view/comment value frequencies.
Record identifiers, publication dates and author labels remain in memory.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
import gzip
import io
import math
from pathlib import Path
import sys


PACKAGE = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PACKAGE / "data/forum_coverage"
BANK_CODES = frozenset({
    "601398", "601939", "601988", "601328", "600036", "601166", "600016",
    "600000", "000001", "600015", "601009", "002142", "601169", "601998",
})
PERIODS = (
    ("A", date(2008, 1, 1), date(2011, 12, 31)),
    ("B", date(2012, 1, 1), date(2015, 12, 31)),
    ("C", date(2016, 1, 1), date(2019, 12, 31)),
    ("D", date(2020, 1, 1), date(2023, 2, 1)),
)
COUNTS_COLUMNS = ("bank_code", "panel", "posts", "active_calendar_days", "distinct_author_labels")
PERIOD_COLUMNS = ("panel", "start_date", "end_date", "archive_start", "archive_end")
FREQUENCY_COLUMNS = ("bank_code", "panel", "metric", "value", "frequency")
OUTPUT_FILES = (
    "forum_coverage_counts.csv", "forum_coverage_periods.csv",
    "forum_interaction_frequencies.csv.gz", "forum_coverage_inventory.json",
)


@dataclass
class Coverage:
    posts: int = 0
    active_dates: set[date] = field(default_factory=set)
    author_labels: set[str] = field(default_factory=set)
    views: Counter[float] = field(default_factory=Counter)
    comments: Counter[float] = field(default_factory=Counter)


def publication_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    try:
        day = datetime.fromisoformat(value).date()
    except ValueError:
        return None
    return None if day.year == 1 else day


def count_value(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value[:-1]) * 10000 if value.endswith("万") else float(value)
    except ValueError as error:
        raise ValueError("Views and comments must be finite, nonnegative counts or blank") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError("Views and comments must be finite, nonnegative counts or blank")
    return 0.0 if number == 0 else number


def prepare(source_directory: Path, output_directory: Path) -> dict[str, int]:
    """Aggregate all retained source records, then write four public input files."""
    csv.field_size_limit(2**31 - 1)
    paths = [source_directory / f"processed_{index}.csv" for index in range(1, 16)]
    if any(not path.is_file() for path in paths):
        raise ValueError("The source directory must contain processed_1.csv through processed_15.csv")
    coverage = {(bank, panel): Coverage() for bank in BANK_CODES for panel, _, _ in PERIODS}
    inventory = {
        "archive_records": 0, "panel_records": 0, "pre_period_records": 0,
        "invalid_date_records": 0, "post_period_records": 0,
    }
    seen_ids = set()
    archive_start = archive_end = None
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            headers = reader.fieldnames or []
            required = {"id", "bank_code", "pub_date", "author", "read_num", "comments_num"}
            if len(headers) != len(set(headers)) or not required.issubset(headers):
                raise ValueError(f"Missing or duplicate coverage fields: {path.name}")
            for row in reader:
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("Missing or extra CSV fields")
                    ident = row["id"].strip()
                    if not ident or ident in seen_ids:
                        raise ValueError("Record IDs must be nonempty and unique across source files")
                    seen_ids.add(ident)
                    bank = row["bank_code"].strip().zfill(6)
                    if bank not in BANK_CODES:
                        raise ValueError("Unknown core bank code")
                    day = publication_date(row["pub_date"])
                    values = {"views": count_value(row["read_num"]),
                              "comments": count_value(row["comments_num"])}
                    inventory["archive_records"] += 1
                    if day is None:
                        inventory["invalid_date_records"] += 1
                        continue
                    archive_start = min(archive_start, day) if archive_start else day
                    archive_end = max(archive_end, day) if archive_end else day
                    if day < PERIODS[0][1]:
                        inventory["pre_period_records"] += 1
                        continue
                    if day > PERIODS[-1][2]:
                        inventory["post_period_records"] += 1
                        continue
                    panel = next(panel for panel, start, end in PERIODS if start <= day <= end)
                    statistics = coverage[bank, panel]
                    statistics.posts += 1
                    statistics.active_dates.add(day)
                    author = row["author"].strip()
                    if author:
                        statistics.author_labels.add(author)
                    for metric, value in values.items():
                        if value is not None:
                            getattr(statistics, metric)[value] += 1
                    inventory["panel_records"] += 1
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path.name}, CSV line {reader.line_num}: {error}") from error
    if archive_start is None:
        raise ValueError("The retained source records have no valid publication dates")

    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / OUTPUT_FILES[0]).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COUNTS_COLUMNS)
        for (bank, panel), statistics in sorted(coverage.items()):
            writer.writerow((bank, panel, statistics.posts, len(statistics.active_dates),
                             len(statistics.author_labels)))
    with (output_directory / OUTPUT_FILES[1]).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(PERIOD_COLUMNS)
        for panel, start, end in PERIODS:
            writer.writerow((panel, start.isoformat(), end.isoformat(),
                             archive_start.isoformat(), archive_end.isoformat()))
    with (output_directory / OUTPUT_FILES[2]).open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(FREQUENCY_COLUMNS)
                for (bank, panel), statistics in sorted(coverage.items()):
                    for metric in ("comments", "views"):
                        for value, frequency in sorted(getattr(statistics, metric).items()):
                            writer.writerow((bank, panel, metric, repr(value), frequency))
    import json
    (output_directory / OUTPUT_FILES[3]).write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Directory containing the retained processed CSV files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for the four public aggregate input files")
    args = parser.parse_args()
    try:
        output_directory = args.output_dir.resolve()
        inventory = prepare(args.source_dir.resolve(), output_directory)
    except (OSError, ValueError, csv.Error, UnicodeError) as error:
        print(f"Forum coverage preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Aggregated {inventory['archive_records']:,} retained records; "
          f"{inventory['panel_records']:,} records in the four panels.")
    for filename in OUTPUT_FILES:
        print(f"Wrote {output_directory / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
