"""Reproduce Table 5 from paired forecasts and daily CoVaR history.

Python's standard library only. Convert the raw history to SR = -100 * CoVaR
before calculating origin-inclusive, 252-observation linear q90 cutoffs.
Use actual_SR from the forecast files, calculate metrics for each seed, and
average within banks before averaging equally across banks. Undefined metrics
propagate through both means.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
import io
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
BANKS = (
    ("000001", "平安银行"), ("002142", "宁波银行"), ("600000", "浦发银行"),
    ("600015", "华夏银行"), ("600016", "民生银行"), ("600036", "招商银行"),
    ("601009", "南京银行"), ("601166", "兴业银行"), ("601169", "北京银行"),
    ("601328", "交通银行"), ("601398", "工商银行"), ("601939", "建设银行"),
    ("601988", "中国银行"), ("601998", "中信银行"),
)
HISTORY_COLUMNS = {
    "000001": "ping_an_bank_covar",
    "002142": "bank_of_ningbo_covar",
    "600000": "shanghai_pudong_development_bank_covar",
    "600015": "huaxia_bank_covar",
    "600016": "china_minsheng_bank_covar",
    "600036": "china_merchants_bank_covar",
    "601009": "bank_of_nanjing_covar",
    "601166": "industrial_bank_covar",
    "601169": "bank_of_beijing_covar",
    "601328": "bank_of_communications_covar",
    "601398": "industrial_and_commercial_bank_of_china_covar",
    "601939": "china_construction_bank_covar",
    "601988": "bank_of_china_covar",
    "601998": "china_citic_bank_covar",
}
COLUMNS = (
    "bank", "bank_code", "seed", "horizon", "origin_date", "target_date",
    "group", "window_id", "actual_SR", "predicted_SR_no_text", "predicted_SR_text",
)
METRICS = ("HR MAE", "Recall", "Precision", "F1", "AP", "FPR")
SPECIFICATIONS = (("Without text", "no_text"), ("With text", "text"))
N_SEEDS = 20
N_DATES = 416
N_HISTORY = 252
STEM = "Table_05_Primary_High_Risk_Evaluation"
TITLE = "Table 5. Primary high-risk evaluation of direct forecasts: rolling 252-observation q90"


@dataclass(frozen=True)
class Forecast:
    origin: date
    target: date
    actual: float
    no_text: float
    text: float


def csv_reader(path: Path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        text = raw.decode("gb18030", errors="strict")
    return csv.DictReader(io.StringIO(text, newline=""), strict=True)


@lru_cache(maxsize=8192)
def parse_date(value: str) -> date:
    for pattern in ("%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            pass
    raise ValueError(f"invalid date: {value!r}")


def finite_number(value: str, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field}: {value!r}")
    return number


def read_forecasts(path: Path, horizon: int):
    """Check the paired panel and its one-to-one seed/group assignments."""
    reader = csv_reader(path)
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError(f"{path.name}: expected the 11 paired-forecast columns")
    names = dict(BANKS)
    banks = {code: {} for code, _ in BANKS}
    seed_groups, group_seeds, actuals = {}, {}, {}
    for row in reader:
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("missing or extra CSV fields")
            code = row["bank_code"]
            if code not in banks or row["bank"] != names[code]:
                raise ValueError("unknown bank_code or mismatched bank label")
            if not row["seed"].isdigit():
                raise ValueError("seed must be a nonnegative integer")
            seed = int(row["seed"])
            if not 0 <= seed < 2**32 or row["horizon"] != str(horizon):
                raise ValueError("seed outside uint32 range or incorrect horizon")
            group = row["group"]
            if not group.strip() or not row["window_id"].strip():
                raise ValueError("empty group or window_id label")
            identity = (code, seed)
            if identity in seed_groups and seed_groups[identity] != group:
                raise ValueError("one seed assigned to multiple groups")
            if (code, group) in group_seeds and group_seeds[code, group] != seed:
                raise ValueError("one group assigned to multiple seeds")
            seed_groups[identity], group_seeds[code, group] = group, seed
            origin, target = parse_date(row["origin_date"]), parse_date(row["target_date"])
            if origin >= target:
                raise ValueError("origin must precede target")
            record = Forecast(origin, target,
                finite_number(row["actual_SR"], "actual_SR"),
                finite_number(row["predicted_SR_no_text"], "predicted_SR_no_text"),
                finite_number(row["predicted_SR_text"], "predicted_SR_text"))
            records = banks[code].setdefault(seed, [])
            if records and (origin <= records[-1].origin or target <= records[-1].target):
                raise ValueError("origin and target dates must be strictly increasing within each seed")
            truth_key = (code, target)
            if truth_key in actuals and actuals[truth_key] != record.actual:
                raise ValueError("actual_SR differs across seeds")
            records.append(record)
            actuals[truth_key] = record.actual
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path.name}, CSV line {reader.line_num}: {error}") from error

    timing = None
    for code, seeds in banks.items():
        if len(seeds) != N_SEEDS:
            raise ValueError(f"{path.name}, {code}: expected {N_SEEDS} seeds, got {len(seeds)}")
        for seed, records in seeds.items():
            if len(records) != N_DATES:
                raise ValueError(f"{path.name}, {code}/{seed}: expected {N_DATES} dates, got {len(records)}")
            sequence = [(record.origin, record.target) for record in records]
            if timing is None:
                timing = sequence
            elif sequence != timing:
                raise ValueError(f"{path.name}, {code}/{seed}: dates do not align across seeds/banks")
    return banks, timing, actuals


def read_history(path: Path):
    """Keep the full source calendar; transform only the 14 required columns."""
    reader = csv_reader(path)
    header = reader.fieldnames or []
    columns = HISTORY_COLUMNS
    if not header or header[0] != "date" or len(header) != len(set(header)):
        raise ValueError(f"{path.name}: expected 'date' as the first header and unique columns")
    missing = set(columns.values()).difference(header)
    if missing:
        raise ValueError(f"{path.name}: missing required bank columns: {', '.join(sorted(missing))}")
    dates, histories = [], {code: [] for code, _ in BANKS}
    for row in reader:
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("missing or extra CSV fields")
            day = parse_date(row["date"])
            if dates and day <= dates[-1]:
                raise ValueError("source dates must be unique and strictly increasing")
            dates.append(day)
            for code, column in columns.items():
                # Conversion precedes the quantile because its multiplier is negative.
                value = -100 * finite_number(row[column], column)
                if not math.isfinite(value):
                    raise ValueError(f"non-finite SR after converting {column}")
                histories[code].append(value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path.name}, CSV line {reader.line_num}: {error}") from error
    if len(dates) < N_HISTORY:
        raise ValueError(f"{path.name}: fewer than {N_HISTORY} observed history dates")
    return dates, histories


def linear_q90(values: list[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def origin_cutoffs(timing, dates, histories, horizon: int):
    positions = {day: index for index, day in enumerate(dates)}
    ends = []
    for origin, target in timing:
        if origin not in positions or target not in positions:
            raise ValueError(f"h{horizon}: origin or target absent from source calendar: {origin}/{target}")
        if positions[target] - positions[origin] != horizon:
            raise ValueError(f"h{horizon}: origin-to-target trading-step mismatch at {origin}")
        end = positions[origin] + 1
        if end < N_HISTORY:
            raise ValueError(f"h{horizon}: fewer than {N_HISTORY} observed values through {origin}")
        ends.append(end)
    return {code: [linear_q90(values[end - N_HISTORY:end]) for end in ends]
            for code, values in histories.items()}


def strict_mean(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    mean = math.fsum(values) / len(values)
    if not math.isfinite(mean):
        raise ValueError("non-finite metric mean")
    return mean


def average_precision(labels: list[bool], scores: list[float]) -> float | None:
    """Noninterpolated AP; all equal scores enter the ranking together."""
    positives = sum(labels)
    if positives in (0, len(labels)):
        return None
    counts = {}
    for label, score in zip(labels, scores):
        positive, size = counts.get(score, (0, 0))
        counts[score] = (positive + int(label), size + 1)
    hits = seen = 0
    terms = []
    for score in sorted(counts, reverse=True):
        positive, size = counts[score]
        hits += positive
        seen += size
        terms.append((positive / positives) * (hits / seen))
    return math.fsum(terms)


def evaluate(records: list[Forecast], cutoffs: list[float], field: str):
    labels = [record.actual >= cutoff for record, cutoff in zip(records, cutoffs)]
    predictions = [getattr(record, field) for record in records]
    alarms = [value >= cutoff for value, cutoff in zip(predictions, cutoffs)]
    tp = sum(label and alarm for label, alarm in zip(labels, alarms))
    fp = sum(not label and alarm for label, alarm in zip(labels, alarms))
    fn = sum(labels) - tp
    tn = len(labels) - tp - fp - fn
    errors = [abs(value - record.actual)
              for record, value, label in zip(records, predictions, labels) if label]
    scores = [value - cutoff for value, cutoff in zip(predictions, cutoffs)]
    if not all(math.isfinite(value) for value in (*errors, *scores)):
        raise ValueError("non-finite forecast error or ranking score")
    return {
        "HR MAE": strict_mean(errors),
        "Recall": tp / (tp + fn) if tp + fn else None,
        "Precision": tp / (tp + fp) if tp + fp else None,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "AP": average_precision(labels, scores),
        "FPR": fp / (fp + tn) if fp + tn else None,
    }


def calculate_table(data_directory: Path, history_path: Path):
    loaded = {h: read_forecasts(data_directory / f"predictions_{h}day_20pairs.csv", h)
              for h in (1, 5)}
    if loaded[1][2] != loaded[5][2]:
        raise ValueError("actual_SR or target dates differ across horizons")
    dates, histories = read_history(history_path)
    rows, undefined_runs = [], {}
    for horizon in (1, 5):
        banks, timing, _ = loaded[horizon]
        cutoffs = origin_cutoffs(timing, dates, histories, horizon)
        for specification, field in SPECIFICATIONS:
            by_bank = {}
            counts = {metric: 0 for metric in METRICS}
            for code, seeds in banks.items():
                runs = [evaluate(seeds[seed], cutoffs[code], field) for seed in sorted(seeds)]
                by_bank[code] = {metric: strict_mean([run[metric] for run in runs]) for metric in METRICS}
                for metric in METRICS:
                    counts[metric] += sum(run[metric] is None for run in runs)
            row = {"h": horizon, "Specification": specification}
            row.update({metric: strict_mean([by_bank[code][metric] for code, _ in BANKS])
                        for metric in METRICS})
            rows.append(row)
            undefined_runs[horizon, specification] = counts
    return rows, {"calendar_dates": len(dates), "undefined_runs": undefined_runs}


def write_outputs(rows, output_directory: Path):
    display = [{key: ("NA" if value is None else f"{value:.4f}") if key in METRICS else str(value)
                for key, value in row.items()} for row in rows]
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{STEM}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["h", "Specification", *METRICS], lineterminator="\n")
        writer.writeheader()
        writer.writerows(display)
    lines = [f"# {TITLE}", "", "| h | Specification | HR MAE | Recall | Precision | F1 | AP | FPR |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines.extend("| " + " | ".join(row.values()) + " |" for row in display)
    lines.extend(["", "Note. SR = −100 × raw CoVaR. Each cutoff is the linear 90th percentile of the latest "
                  "252 observed SR values through and including the forecast origin. Actual high-risk days and "
                  "forecast alarms use ≥ the same cutoff. HR MAE uses actual high-risk days; AP ranks prediction "
                  "minus cutoff with tied scores grouped. Metrics are calculated per seed, averaged over 20 "
                  "paired seeds within each bank, then averaged equally over 14 banks. NA denotes an undefined "
                  "denominator or one-class AP in any required seed, propagated through both means.", ""])
    md_path = output_directory / f"{STEM}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    try:
        rows, diagnostics = calculate_table(ROOT / "data", ROOT / "data-raw" / "All_CoVar_no_other_diff.csv")
        outputs = write_outputs(rows, ROOT / "tables")
    except (OSError, ValueError, csv.Error, UnicodeError, OverflowError) as error:
        print(f"Table 5 failed: {error}", file=sys.stderr)
        return 1
    print("Validated 14 banks x 2 horizons x 20 paired seeds x 416 dates.")
    print(f"Source calendar: {diagnostics['calendar_dates']} dates; all h1/h5 steps match.")
    print("Cutoffs use 252 origin-inclusive observations; undefined metrics propagate strictly.")
    for output in outputs:
        print(f"Wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
