"""Reproduce Table 4 directly from the two paired-forecast CSV files.

Run with Python's standard library only. Each loss is averaged over dates
within each seed, then over seeds; predictions are never averaged before loss.
DM uses daily seed-mean loss differences, trading-gap Bartlett HAC and HLN.
Student-t probabilities use an incomplete-beta continued-fraction calculation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
import hashlib
import io
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
BANKS = (
    ("000001", "PAB"), ("002142", "NBCB"), ("600000", "SPDB"),
    ("600015", "HB"), ("600016", "CMBC"), ("600036", "CMB"),
    ("601009", "BON"), ("601166", "CIB"), ("601169", "BOB"),
    ("601328", "BCM"), ("601398", "ICBC"), ("601939", "CCB"),
    ("601988", "BOC"), ("601998", "CNCB"),
)
COLUMNS = (
    "bank", "bank_code", "seed", "horizon", "origin_date", "target_date",
    "group", "window_id", "actual_SR", "predicted_SR_no_text", "predicted_SR_text",
)
N_SEEDS = 20
N_DATES = 416
STEM = "Table_04_Bank_Level_Forecast_Results"


@dataclass(frozen=True)
class Forecast:
    origin: date
    target: date
    actual: float
    no_text: float
    text: float


@lru_cache(maxsize=2048)
def parse_date(value: str) -> date:
    for pattern in ("%m/%d/%Y", "%Y-%m-%d"):
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
    """Validate each bank/seed/date identity; group/window_id are row labels."""
    raw = path.read_bytes()
    try:
        text, encoding = raw.decode("utf-8-sig", errors="strict"), "utf-8-sig"
    except UnicodeDecodeError:
        text, encoding = raw.decode("gb18030", errors="strict"), "gb18030"
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError(f"{path.name}: expected the 11 paired-forecast columns")
    banks = {code: {} for code, _ in BANKS}
    seed_groups = {}
    group_seeds = {}
    actuals = {}
    count = 0
    for row in reader:
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("missing or extra CSV fields")
            code = row["bank_code"]
            if code not in banks:
                raise ValueError(f"unknown six-digit bank_code: {code!r}")
            if not row["seed"].isdigit():
                raise ValueError("seed must be a nonnegative integer")
            seed = int(row["seed"])
            if not 0 <= seed < 2**32 or row["horizon"] != str(horizon):
                raise ValueError("seed outside uint32 range or incorrect horizon")
            if not row["bank"].strip() or not row["group"].strip() or not row["window_id"].strip():
                raise ValueError("empty bank, group or window_id label")
            group = row["group"]
            identity = (code, seed)
            if identity in seed_groups and seed_groups[identity] != group:
                raise ValueError("one seed assigned to multiple groups")
            if (code, group) in group_seeds and group_seeds[code, group] != seed:
                raise ValueError("one group assigned to multiple seeds")
            seed_groups[identity] = group
            group_seeds[code, group] = seed
            origin, target = parse_date(row["origin_date"]), parse_date(row["target_date"])
            if origin >= target:
                raise ValueError("origin must precede target")
            forecast = Forecast(origin, target,
                finite_number(row["actual_SR"], "actual_SR"),
                finite_number(row["predicted_SR_no_text"], "predicted_SR_no_text"),
                finite_number(row["predicted_SR_text"], "predicted_SR_text"))
            records = banks[code].setdefault(seed, {})
            if target in records:
                raise ValueError("duplicate bank/seed/horizon/target identity")
            truth_key = (code, target)
            if truth_key in actuals and actuals[truth_key] != forecast.actual:
                raise ValueError("actual_SR differs across seeds")
            records[target] = forecast
            actuals[truth_key] = forecast.actual
            count += 1
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path.name}, CSV line {reader.line_num}: {error}") from error

    timing = None
    for code, seeds in banks.items():
        if len(seeds) != N_SEEDS:
            raise ValueError(f"{path.name}, {code}: expected {N_SEEDS} seeds, got {len(seeds)}")
        for seed, records in seeds.items():
            if len(records) != N_DATES:
                raise ValueError(f"{path.name}, {code}/{seed}: expected {N_DATES} dates, got {len(records)}")
            ordered = [records[target] for target in sorted(records)]
            sequence = [(record.origin, record.target) for record in ordered]
            if len({record.origin for record in ordered}) != N_DATES:
                raise ValueError(f"{path.name}, {code}/{seed}: duplicate origin dates")
            if timing is None:
                timing = sequence
            elif sequence != timing:
                raise ValueError(f"{path.name}, {code}/{seed}: dates do not align across seeds/banks")
    if count != len(BANKS) * N_SEEDS * N_DATES:
        raise ValueError(f"{path.name}: incomplete paired-forecast panel")
    return banks, timing, actuals, {
        "filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": encoding, "rows": count,
    }


def beta_fraction(a: float, b: float, x: float) -> float:
    """Continued-fraction tolerance and iteration bound."""
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 3e-30 if abs(d) < 3e-30 else d
    d = 1.0 / d
    value = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 3e-30 if abs(d) < 3e-30 else d
        c = 3e-30 if abs(c) < 3e-30 else c
        d = 1 / d
        value *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 3e-30 if abs(d) < 3e-30 else d
        c = 3e-30 if abs(c) < 3e-30 else c
        d = 1 / d
        change = d * c
        value *= change
        if abs(change - 1) < 3e-7:
            return value
    raise ValueError("Student-t incomplete-beta calculation did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * beta_fraction(a, b, x) / a
    return 1 - factor * beta_fraction(b, a, 1 - x) / b


def student_t_two_sided(statistic: float, degrees: int) -> float:
    return regularized_beta(degrees / (degrees + statistic * statistic), degrees / 2, 0.5)


def dm_test(differences: list[float], positions: list[int], horizon: int):
    n = len(differences)
    mean = math.fsum(differences) / n
    centered = [value - mean for value in differences]
    hac = math.fsum(value * value for value in centered) / n
    by_position = dict(zip(positions, centered))
    for lag in range(1, horizon):
        products = [value * by_position[position - lag]
                    for position, value in zip(positions, centered)
                    if position - lag in by_position]
        hac += 2 * (1 - lag / horizon) * math.fsum(products) / n
    variance = hac / n
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("DM test has zero, negative or non-finite HAC variance")
    factor = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = mean / math.sqrt(variance) * factor
    probability = student_t_two_sided(abs(statistic), n - 1)
    if not math.isfinite(statistic) or not 0 <= probability <= 1:
        raise ValueError("non-finite DM statistic or invalid Student-t probability")
    return statistic, probability, mean, variance, factor


def calculate_table(data_directory: Path):
    loaded = {h: read_forecasts(data_directory / f"predictions_{h}day_20pairs.csv", h)
              for h in (1, 5)}
    if loaded[1][2] != loaded[5][2]:
        raise ValueError("actual_SR or target dates differ across horizons")
    calendar = sorted({day for _, timing, _, _ in loaded.values() for pair in timing for day in pair})
    calendar_index = {day: index for index, day in enumerate(calendar)}
    for horizon, (_, timing, _, _) in loaded.items():
        for origin, target in timing:
            if calendar_index[target] - calendar_index[origin] != horizon:
                raise ValueError(f"h{horizon}: origin-to-target trading-step mismatch at {origin}")

    rows = []
    for code, abbreviation in BANKS:
        for horizon in (1, 5):
            banks, timing, _, _ = loaded[horizon]
            targets = [target for _, target in timing]
            positions = [calendar_index[origin] for origin, _ in timing]
            series = [banks[code][seed] for seed in sorted(banks[code])]
            for loss in ("MSE", "MAE"):
                no_losses, text_losses = [], []
                for records in series:
                    no_errors = [records[target].no_text - records[target].actual for target in targets]
                    text_errors = [records[target].text - records[target].actual for target in targets]
                    transform = (lambda value: value * value) if loss == "MSE" else abs
                    no_losses.append([transform(error) for error in no_errors])
                    text_losses.append([transform(error) for error in text_errors])
                no_mean = math.fsum(math.fsum(values) / N_DATES for values in no_losses) / N_SEEDS
                text_mean = math.fsum(math.fsum(values) / N_DATES for values in text_losses) / N_SEEDS
                if not all(math.isfinite(value) for value in (no_mean, text_mean)) or no_mean <= 0:
                    raise ValueError(f"{code}/h{horizon}/{loss}: invalid mean forecast losses")
                reduction = 100 * (1 - text_mean / no_mean)
                daily_difference = [math.fsum(text_losses[s][i] - no_losses[s][i]
                                             for s in range(N_SEEDS)) / N_SEEDS
                                    for i in range(N_DATES)]
                statistic, probability, difference, variance, factor = dm_test(daily_difference, positions, horizon)
                stars = "***" if probability < .01 else "**" if probability < .05 else "*" if probability < .1 else ""
                reduction_display = f"{reduction:.2f}{stars}"
                dm_display = f"{statistic:.3f}"
                p_display = "<0.001" if probability < .001 else f"{probability:.3f}"
                rows.append({
                    "bank": abbreviation, "bank_code": code, "horizon": horizon, "loss": loss,
                    "n_seeds": N_SEEDS, "n_dates": N_DATES, "mean_loss_no_text": no_mean,
                    "mean_loss_text": text_mean, "reduction_pct": reduction,
                    "dm_statistic": statistic, "raw_p": probability, "df": N_DATES - 1,
                    "mean_daily_loss_difference": difference, "hac_variance": variance,
                    "hln_factor": factor, "stars": stars, "reduction_display": reduction_display,
                    "dm_display": dm_display, "p_display": p_display,
                    "cell_display": "\n".join((reduction_display, dm_display, p_display)),
                })
    return rows, {"inputs": [loaded[h][3] for h in (1, 5)], "calendar_dates": len(calendar),
                  "calendar_start": calendar[0].isoformat(), "calendar_end": calendar[-1].isoformat()}


def write_outputs(rows, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{STEM}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    cells = {(row["bank_code"], row["horizon"], row["loss"]): row["cell_display"] for row in rows}
    lines = ["# Table 4. Primary bank-level reductions and paired tests for direct forecasts", "",
             "| Bank | h1 MSE | h1 MAE | h5 MSE | h5 MAE |",
             "| --- | --- | --- | --- | --- |"]
    for code, abbreviation in BANKS:
        values = [cells[code, h, loss].replace("\n", "<br>") for h in (1, 5) for loss in ("MSE", "MAE")]
        # Escape stars so Markdown displays their literal significance symbols.
        values = [value.replace("*", "\\*") for value in values]
        lines.append("| " + " | ".join([abbreviation, *values]) + " |")
    lines.extend(["", "Each result shows reduction (%), HLN-corrected DM, and the unadjusted two-sided p-value.",
                  "Positive reduction and negative DM favour text. Significance: * p < 0.10; ** p < 0.05; *** p < 0.01.", ""])
    md_path = output_directory / f"{STEM}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    try:
        rows, diagnostics = calculate_table(ROOT / "data")
        outputs = write_outputs(rows, ROOT / "tables")
    except (OSError, ValueError, csv.Error, UnicodeError) as error:
        print(f"Table 4 failed: {error}", file=sys.stderr)
        return 1
    print("Validated 14 banks x 2 horizons x 20 paired seeds x 416 dates.")
    print(f"Inferred trading calendar: {diagnostics['calendar_dates']} dates; all h1/h5 steps match.")
    for output in outputs:
        print(f"Wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
