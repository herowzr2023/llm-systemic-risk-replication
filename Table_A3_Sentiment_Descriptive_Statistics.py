"""Reproduce Table A3 from the ICBC and banking-sector descriptive series.

Use all available observations for each indicator. Statistics are the arithmetic
mean, sample standard deviation (ddof=1), minimum and maximum, rounded to two
decimal places. No additional Python packages are required.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import math
from pathlib import Path
from statistics import mean, stdev
import sys


ROOT = Path(__file__).resolve().parent
STEM = "Table_A3_Sentiment_Descriptive_Statistics"
CAPTION = "Table A3. Descriptive Statistics of Sentiment Indicators"
SERIES = (
    ("ICBC", "icbc_daily_sentiment.csv"),
    ("Banking sector", "banking_sector_daily_sentiment.csv"),
)
DEFAULT_DATA_DIRECTORY = ROOT / "data-raw" / "sentiment_diagnostics"
DEFAULT_ICBC_FILE = ROOT / "data-raw" / "full_period" / "with_sentiment" / "industrial_and_commercial_bank_of_china_601398_daily_with_sentiment.csv"
INDICATORS = (
    ("sentiment_direction", "Sentiment direction"),
    ("attention_post_count", "Attention-based intensity"),
)
STATISTICS = ("Mean", "Std. dev.", "Min", "Max")
OUTPUT_COLUMNS = ("Index", *(f"{scope} {statistic}" for scope, _ in SERIES for statistic in STATISTICS))
MISSING_VALUES = {"", "NA", "N/A", "NaN", "nan", "NULL", "null"}


def resolve_input_paths(data_directory: Path | None = None):
    directory = data_directory if data_directory is not None else DEFAULT_DATA_DIRECTORY
    return {scope: (DEFAULT_ICBC_FILE if scope == "ICBC" and data_directory is None else directory / filename)
            for scope, filename in SERIES}


def read_observations(path: Path, columns):
    observations = []
    previous_date = None
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        missing = {"date", *columns} - set(headers)
        if missing:
            raise ValueError(f"{path.name}: missing columns: {', '.join(sorted(missing))}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path.name}: duplicate column names")
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("missing or extra CSV fields")
                observed_date = date.fromisoformat(row["date"])
                if previous_date is not None and observed_date <= previous_date:
                    raise ValueError("dates must be unique and in increasing order")
                previous_date = observed_date
                observation = {"date": observed_date}
                for column in columns:
                    raw = row[column].strip()
                    if raw in MISSING_VALUES:
                        observation[column] = None
                        continue
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError(f"{column} must contain finite numbers or missing values")
                    observation[column] = value
                observations.append(observation)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}, CSV line {reader.line_num}: {error}") from error
    return observations


def read_series(path: Path):
    observations = read_observations(path, [column for column, _ in INDICATORS])
    values = {column: [row[column] for row in observations if row[column] is not None]
              for column, _ in INDICATORS}
    for column, _ in INDICATORS:
        if len(values[column]) < 2:
            raise ValueError(f"{path.name}: {column} requires at least two nonmissing observations")
    return values, (len(observations), observations[0]["date"], observations[-1]["date"])


def calculate_table(data_directory: Path | None = None):
    series, samples = {}, {}
    for scope, path in resolve_input_paths(data_directory).items():
        series[scope], samples[scope] = read_series(path)
    rows = []
    for column, title in INDICATORS:
        row = {"Index": title}
        for scope, _ in SERIES:
            values = series[scope][column]
            statistics = (mean(values), stdev(values), min(values), max(values))
            row.update({f"{scope} {name}": f"{value:.2f}"
                        for name, value in zip(STATISTICS, statistics)})
        rows.append(row)
    return rows, samples


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
        "| " + " | ".join("---" for _ in OUTPUT_COLUMNS) + " |",
    ]
    lines.extend("| " + " | ".join(row[column] for column in OUTPUT_COLUMNS) + " |" for row in rows)
    md_path = output_directory / f"{STEM}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Table A3 from the descriptive sentiment series.")
    parser.add_argument("--data-dir", type=Path,
                        help="Read icbc_daily_sentiment.csv and banking_sector_daily_sentiment.csv from this directory; defaults use full_period for ICBC and sentiment_diagnostics for the sector.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table A3 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        rows, samples = calculate_table(arguments.data_dir)
        outputs = write_outputs(rows, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Table A3 failed: {error}", file=sys.stderr)
        return 1
    for scope, (count, first_date, last_date) in samples.items():
        print(f"{scope}: {count:,} observations, {first_date} to {last_date}.")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
