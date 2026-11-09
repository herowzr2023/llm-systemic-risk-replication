"""Reproduce Table 6 from the two aggregate-forecast CSV files.

Run with Python's standard library only. MSE and MAE are averaged over
target dates within each group, then over the 20 groups. Predictions are
never averaged before computing loss. The input CSVs contain
the Informer, Persistence, AR(1), GBR and PatchTST forecasts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import hashlib
import io
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
STEM = "Table_06_Secondary_Aggregate_Losses"
N_GROUPS = 20
COLUMNS = (
    "date", "TRUE", "baseline_pred", "GradientBoostingRegressor_pred",
    "PatchTST_pred", "Persistence_pred", "AR1_pred", "group_id", "seed",
    "nominal_horizon",
)
MODELS = (
    ("Informer", "baseline_pred"),
    ("Persistence", "Persistence_pred"),
    ("AR(1)", "AR1_pred"),
    ("GBR", "GradientBoostingRegressor_pred"),
    ("PatchTST", "PatchTST_pred"),
)
OUTPUT_COLUMNS = (
    "model", "h1_mse_x1e4", "h1_mae_x1e2", "h5_mse_x1e3", "h5_mae_x1e2",
)


@dataclass(frozen=True)
class Forecast:
    actual: float
    predictions: tuple[float, ...]


@lru_cache(maxsize=2048)
def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def finite_number(value: str, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field}: {value!r}")
    return number


def read_forecasts(path: Path, horizon: int):
    """Validate group/date identities and align target dates and actuals."""
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError(f"{path.name}: expected the 10 aggregate-forecast columns")
    groups = {}
    actuals = {}
    count = 0
    for row in reader:
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("missing or extra CSV fields")
            group = int(row["group_id"])
            if group <= 0 or row["nominal_horizon"] != str(horizon):
                raise ValueError("invalid group_id or incorrect nominal_horizon")
            target = parse_date(row["date"])
            forecast = Forecast(
                finite_number(row["TRUE"], "TRUE"),
                tuple(finite_number(row[column], column) for _, column in MODELS),
            )
            records = groups.setdefault(group, {})
            if target in records:
                raise ValueError("duplicate group/date identity")
            if target in actuals and actuals[target] != forecast.actual:
                raise ValueError("TRUE differs across groups")
            records[target] = forecast
            actuals[target] = forecast.actual
            count += 1
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path.name}, CSV line {reader.line_num}: {error}") from error
    if len(groups) != N_GROUPS:
        raise ValueError(f"{path.name}: expected {N_GROUPS} groups, got {len(groups)}")
    timing = None
    for group, records in sorted(groups.items()):
        dates = sorted(records)
        if timing is None:
            timing = dates
        elif dates != timing:
            raise ValueError(f"{path.name}, group {group}: target dates do not align across groups")
    return groups, {
        "filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        "horizon": horizon, "n_groups": len(groups), "n_dates": len(timing),
        "rows": count, "date_start": timing[0].isoformat(),
        "date_end": timing[-1].isoformat(),
    }


def calculate_table(data_directory: Path):
    loaded = {
        horizon: read_forecasts(data_directory / f"aggregate_predictions_h{horizon}_20groups.csv", horizon)
        for horizon in (1, 5)
    }
    means = {}
    for horizon, (groups, _) in loaded.items():
        for index, (model, _) in enumerate(MODELS):
            group_mse, group_mae = [], []
            for _, records in sorted(groups.items()):
                errors = [record.actual - record.predictions[index]
                          for _, record in sorted(records.items())]
                group_mse.append(math.fsum(error * error for error in errors) / len(errors))
                group_mae.append(math.fsum(abs(error) for error in errors) / len(errors))
            mse = math.fsum(group_mse) / len(group_mse)
            mae = math.fsum(group_mae) / len(group_mae)
            if not all(math.isfinite(value) for value in (mse, mae)):
                raise ValueError(f"h{horizon}/{model}: non-finite mean forecast loss")
            means[horizon, model, "MSE"] = mse
            means[horizon, model, "MAE"] = mae
    rows = [{
        "model": model,
        "h1_mse_x1e4": f"{means[1, model, 'MSE'] * 1e4:.2f}",
        "h1_mae_x1e2": f"{means[1, model, 'MAE'] * 1e2:.2f}",
        "h5_mse_x1e3": f"{means[5, model, 'MSE'] * 1e3:.2f}",
        "h5_mae_x1e2": f"{means[5, model, 'MAE'] * 1e2:.2f}",
    } for model, _ in MODELS]
    return rows, {"inputs": [loaded[horizon][1] for horizon in (1, 5)], "mean_losses": means}


def write_outputs(rows, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{STEM}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Table 6. Secondary aggregate losses for direct forecasts", "",
             "| Model | h1 MSE (×10⁻⁴) | h1 MAE (×10⁻²) | h5 MSE (×10⁻³) | h5 MAE (×10⁻²) |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in OUTPUT_COLUMNS) + " |")
    lines.extend(["", "MSE and MAE are computed over target dates within each group, then averaged across the 20 groups.",
                  "Values are rounded to two decimals in the indicated units. Each horizon retains its own target-date sample.", ""])
    md_path = output_directory / f"{STEM}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Table 6 from saved aggregate forecasts.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Directory containing the h1/h5 aggregate_predictions CSV files.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table 6 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        rows, diagnostics = calculate_table(arguments.data_dir)
        outputs = write_outputs(rows, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Table 6 failed: {error}", file=sys.stderr)
        return 1
    for source in diagnostics["inputs"]:
        print(f"Validated h{source['horizon']}: {source['n_groups']} groups x "
              f"{source['n_dates']} dates ({source['rows']} rows).")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

