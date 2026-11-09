"""Reproduce the four panels of Table A14 from public forum aggregates.

Read bank/subperiod counts, panel and archive date bounds, separate view/comment
value-frequency distributions, and archive inventory totals. Calendar-day
statistics use the intersection of the panel and global archive intervals.
Record-level percentiles use cumulative frequencies and linear interpolation;
valid zeros are included and missing fields have no frequency entry.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from datetime import date
import gzip
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = ROOT / "data" / "forum_coverage"
CAPTION = "Table A14. Forum Coverage of the 14 Core Banks by Subperiod"
COUNTS_FILE = "forum_coverage_counts.csv"
PERIODS_FILE = "forum_coverage_periods.csv"
FREQUENCIES_FILE = "forum_interaction_frequencies.csv.gz"
INVENTORY_FILE = "forum_coverage_inventory.json"
COUNTS_COLUMNS = (
    "bank_code", "panel", "posts", "active_calendar_days", "distinct_author_labels",
)
PERIODS_COLUMNS = (
    "panel", "start_date", "end_date", "archive_start", "archive_end",
)
FREQUENCY_COLUMNS = ("bank_code", "panel", "metric", "value", "frequency")
INVENTORY_FIELDS = (
    "archive_records", "panel_records", "pre_period_records", "invalid_date_records", "post_period_records",
)
OUTPUT_COLUMNS = (
    "Bank", "Posts", "Active calendar days", "Zero-post calendar days", "Active-day %",
    "Distinct author labels", "Views: P25 / Median / P75", "Comments: P25 / Median / P75",
)
BANKS = (
    ("ICBC", "601398"), ("CCB", "601939"), ("BOC", "601988"), ("BCM", "601328"),
    ("CMB", "600036"), ("CIB", "601166"), ("CMBC", "600016"), ("SPDB", "600000"),
    ("PAB", "000001"), ("HB", "600015"), ("BON", "601009"), ("NBCB", "002142"),
    ("BOB", "601169"), ("CNCB", "601998"),
)
PERIODS = (
    ("A", "2008–2011", date(2008, 1, 1), date(2011, 12, 31), "2008_2011"),
    ("B", "2012–2015", date(2012, 1, 1), date(2015, 12, 31), "2012_2015"),
    ("C", "2016–2019", date(2016, 1, 1), date(2019, 12, 31), "2016_2019"),
    ("D", "2020–2023-02-01", date(2020, 1, 1), date(2023, 2, 1), "2020_2023"),
)


def nonnegative_integer(raw: str, column: str) -> int:
    if not raw.isascii() or not raw.isdigit():
        raise ValueError(f"{column} must be a nonnegative integer")
    return int(raw)


def nonnegative_number(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("value must be a nonnegative finite number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be a nonnegative finite number")
    return 0.0 if value == 0 else value


def csv_rows(path: Path, columns):
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path.name}: duplicate column names")
        if set(headers) != set(columns):
            raise ValueError(f"{path.name}: expected columns: {', '.join(columns)}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}, CSV line {reader.line_num}: missing or extra CSV fields")
            yield reader.line_num, {column: row[column].strip() for column in columns}


def read_date(raw: str, column: str) -> date:
    try:
        value = date.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"{column} must use a valid YYYY-MM-DD date") from error
    if value.isoformat() != raw:
        raise ValueError(f"{column} must use a valid YYYY-MM-DD date")
    return value


def read_periods(path: Path):
    expected = {panel: (start, end) for panel, _, start, end, _ in PERIODS}
    periods, intervals = {}, {}
    archive_bounds = None
    for line, row in csv_rows(path, PERIODS_COLUMNS):
        try:
            panel = row["panel"]
            if panel not in expected:
                raise ValueError("unknown panel")
            if panel in periods:
                raise ValueError("duplicate panel")
            dates = {column: read_date(row[column], column) for column in PERIODS_COLUMNS[1:]}
            if (dates["start_date"], dates["end_date"]) != expected[panel]:
                raise ValueError("panel start_date and end_date must match the Table A14 subperiod")
            bounds = dates["archive_start"], dates["archive_end"]
            if bounds[0] > bounds[1]:
                raise ValueError("archive_start must not follow archive_end")
            if archive_bounds is not None and bounds != archive_bounds:
                raise ValueError("global archive bounds must agree across panels")
            archive_bounds = bounds
            first_date = max(dates["start_date"], bounds[0])
            last_date = min(dates["end_date"], bounds[1])
            intervals[panel] = (first_date, last_date, max(0, (last_date - first_date).days + 1))
            periods[panel] = dates
        except ValueError as error:
            raise ValueError(f"{path.name}, CSV line {line}: {error}") from error
    if set(periods) != set(expected):
        raise ValueError(f"{path.name}: requires all four Table A14 panels")
    return periods, intervals


def read_counts(path: Path, intervals):
    expected = {(panel, code) for panel, *_ in PERIODS for _, code in BANKS}
    counts = {}
    for line, row in csv_rows(path, COUNTS_COLUMNS):
        try:
            key = row["panel"], row["bank_code"]
            if key not in expected:
                raise ValueError("unknown bank_code or panel")
            if key in counts:
                raise ValueError("duplicate bank_code/panel key")
            values = {column: nonnegative_integer(row[column], column) for column in COUNTS_COLUMNS[2:]}
            posts, active, authors = (values[column] for column in COUNTS_COLUMNS[2:])
            if active > min(posts, intervals[key[0]][2]):
                raise ValueError("active_calendar_days exceeds posts or calendar days counted")
            if posts > 0 and active == 0:
                raise ValueError("positive posts require at least one active calendar day")
            if authors > posts:
                raise ValueError("distinct_author_labels exceeds posts")
            counts[key] = values
        except ValueError as error:
            raise ValueError(f"{path.name}, CSV line {line}: {error}") from error
    if set(counts) != expected:
        raise ValueError(f"{path.name}: requires one row for each of the 14 banks and four panels")
    return counts


def read_frequencies(path: Path, counts):
    frequencies = {(*key, metric): {} for key in counts for metric in ("views", "comments")}
    available = {key: 0 for key in frequencies}
    for line, row in csv_rows(path, FREQUENCY_COLUMNS):
        try:
            key = row["panel"], row["bank_code"], row["metric"]
            if key not in frequencies:
                raise ValueError("unknown bank_code, panel or metric")
            value = nonnegative_number(row["value"])
            frequency = nonnegative_integer(row["frequency"], "frequency")
            if frequency == 0:
                raise ValueError("frequency must be a positive integer")
            if value in frequencies[key]:
                raise ValueError("duplicate bank_code/panel/metric/value key")
            available[key] += frequency
            if available[key] > counts[key[:2]]["posts"]:
                raise ValueError("available-field frequencies exceed posts")
            frequencies[key][value] = frequency
        except ValueError as error:
            raise ValueError(f"{path.name}, CSV line {line}: {error}") from error
    return frequencies, available


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate inventory key: {key}")
        result[key] = value
    return result


def read_inventory(path: Path, counts):
    with path.open(encoding="utf-8-sig") as source:
        inventory = json.load(source, object_pairs_hook=unique_json_object)
    if not isinstance(inventory, dict) or set(inventory) != set(INVENTORY_FIELDS):
        raise ValueError(f"{path.name}: expected inventory fields: {', '.join(INVENTORY_FIELDS)}")
    if any(type(value) is not int or value < 0 for value in inventory.values()):
        raise ValueError(f"{path.name}: inventory fields must be nonnegative integers")
    if inventory["panel_records"] != sum(row["posts"] for row in counts.values()):
        raise ValueError(f"{path.name}: panel_records differs from the coverage counts")
    if inventory["archive_records"] != sum(inventory[field] for field in INVENTORY_FIELDS[1:]):
        raise ValueError(f"{path.name}: archive inventory does not reconcile")
    return inventory


def format_quantiles(frequencies: dict[float, int]) -> str:
    """Find floor/ceiling ranks in cumulative frequencies without expanding rows."""
    if not frequencies:
        return "NA / NA / NA"
    values, cumulative = [], []
    total = 0
    for value, frequency in sorted(frequencies.items()):
        values.append(value)
        total += frequency
        cumulative.append(total)
    quantiles = []
    for numerator in (1, 2, 3):
        lower_rank, remainder = divmod((total - 1) * numerator, 4)
        upper_rank = lower_rank + bool(remainder)
        lower = values[bisect_right(cumulative, lower_rank)]
        upper = values[bisect_right(cumulative, upper_rank)]
        quantile = lower + (upper - lower) * remainder / 4
        quantiles.append(f"{quantile:.2f}")
    return " / ".join(quantiles)


def calculate_tables(data_directory: Path = DEFAULT_DATA_DIRECTORY):
    periods, intervals = read_periods(data_directory / PERIODS_FILE)
    counts = read_counts(data_directory / COUNTS_FILE, intervals)
    frequencies, available = read_frequencies(data_directory / FREQUENCIES_FILE, counts)
    inventory = read_inventory(data_directory / INVENTORY_FILE, counts)
    summary = {
        "records": inventory["archive_records"], "panel_records": inventory["panel_records"],
        "pre_period": inventory["pre_period_records"], "invalid_dates": inventory["invalid_date_records"],
        "post_period": inventory["post_period_records"],
        "archive_start": periods["A"]["archive_start"], "archive_end": periods["A"]["archive_end"],
    }
    for metric in ("views", "comments"):
        summary[f"missing_{metric}"] = inventory["panel_records"] - sum(
            frequency for key, frequency in available.items() if key[2] == metric)
    tables = {}
    for panel, *_ in PERIODS:
        calendar_days = intervals[panel][2]
        rows = []
        for bank, code in BANKS:
            statistics = counts[panel, code]
            active_days = statistics["active_calendar_days"]
            rows.append(dict(zip(OUTPUT_COLUMNS, (
                bank, f"{statistics['posts']:,}", f"{active_days:,}",
                f"{calendar_days - active_days:,}",
                f"{100 * active_days / calendar_days:.2f}" if calendar_days else "NA",
                f"{statistics['distinct_author_labels']:,}",
                format_quantiles(frequencies[panel, code, "views"]),
                format_quantiles(frequencies[panel, code, "comments"]),
            ))))
        tables[panel] = rows
    return tables, intervals, summary


def write_outputs(tables, intervals, summary, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for panel, label, _, _, years in PERIODS:
        stem = f"Table_A14_Panel_{panel}_{years}"
        csv_path = output_directory / f"{stem}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(tables[panel])
        first_date, last_date, calendar_days = intervals[panel]
        interval_note = (
            f"This panel counts {calendar_days:,} calendar days, {first_date} to {last_date}, inclusive."
            if calendar_days else "This panel has no calendar days within the archive date range."
        )
        lines = [
            f"# {CAPTION}", "", f"## Panel {panel}. {label}", "",
            "| " + " | ".join(OUTPUT_COLUMNS) + " |",
            "| " + " | ".join("---" for _ in OUTPUT_COLUMNS) + " |",
        ]
        lines.extend("| " + " | ".join(row[column] for column in OUTPUT_COLUMNS) + " |"
                     for row in tables[panel])
        lines.extend([
            "",
            "Posts are retained archived records after basic title-validity screening, without URL deduplication.",
            "",
            "Active calendar days contain at least one retained record. Zero-post calendar days contain none. "
            "The denominator is the intersection of the subperiod and the global valid-publication-date "
            f"archive range ({summary['archive_start']} to {summary['archive_end']}). "
            + interval_note + " Active-day % equals 100 times active days divided by calendar days counted. "
            "These calendar-day statistics differ from the trading-day windows used in forecasting.",
            "",
            "Distinct author labels count nonempty display names within each bank and subperiod. "
            "Shared labels may represent different accounts; counts are not "
            "additive across periods.",
            "",
            "View and comment summaries are record-level 25th, 50th and 75th percentiles of available fields, "
            "including valid zeros. The public inputs provide bank/subperiod counts, period and archive "
            "date bounds, separate value-frequency distributions for views and comments, and archive "
            "inventory totals. Percentile ranks are located in cumulative frequencies and linearly "
            "interpolated at position (n − 1)p in the ordered distribution. "
            "NA denotes an unavailable statistic. These archival fields are descriptive and are not used "
            "in the forecasting predictors.",
            "",
        ])
        md_path = output_directory / f"{stem}.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        paths.extend((csv_path, md_path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce all four Table A14 panels from public forum aggregates.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY,
                        help="Directory containing the public coverage counts, periods, interaction frequencies and inventory; defaults to data/forum_coverage beside this script.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the four panel CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        tables, intervals, summary = calculate_tables(arguments.data_dir)
        outputs = write_outputs(tables, intervals, summary, arguments.output_dir)
    except (OSError, ValueError, OverflowError, EOFError, csv.Error, UnicodeError) as error:
        print(f"Table A14 failed: {error}", file=sys.stderr)
        return 1
    print(f"Archive: {summary['records']:,} retained records; valid publication dates "
          f"{summary['archive_start']} to {summary['archive_end']}.")
    print(f"Four panels: {summary['panel_records']:,} records; pre-2008: {summary['pre_period']:,}; "
          f"after 2023-02-01: {summary['post_period']:,}; invalid publication dates: {summary['invalid_dates']:,}.")
    print(f"Missing view fields in panels: {summary['missing_views']:,}; "
          f"missing comment fields: {summary['missing_comments']:,}.")
    for panel, _, _, _, _ in PERIODS:
        print(f"Panel {panel}: {intervals[panel][2]:,} calendar days.")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
