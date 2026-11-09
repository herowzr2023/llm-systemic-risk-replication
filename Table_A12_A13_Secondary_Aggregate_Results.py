"""Reproduce Tables A12 and A13 from the saved aggregate forecasts.

Run with Python's standard library only. Losses are computed before averaging
across groups. Table A13 uses date-level group-mean loss differences, their
zero-lag variance, and a two-sided Student-t reference with n - 1 degrees of
freedom. The same statistic is used for both forecast horizons.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

from Table_04_Bank_Level_Forecast_Results import student_t_two_sided
from Table_06_Secondary_Aggregate_Losses import MODELS, read_forecasts


ROOT = Path(__file__).resolve().parent
A12_STEM = "Table_A12_Secondary_Aggregate_Losses"
A13_STEM = "Table_A13_Secondary_Predictive_Accuracy"
A12_COLUMNS = ("h", "Loss unit", *(model for model, _ in MODELS))
A13_COLUMNS = ("h", "Loss", *(model for model, _ in MODELS[1:]))
EXPECTED_DATES = {1: 672, 5: 676}
LOSS_UNITS = {
    (1, "MSE"): (1e4, "MSE (×10⁻⁴)"),
    (1, "MAE"): (1e2, "MAE (×10⁻²)"),
    (5, "MSE"): (1e3, "MSE (×10⁻³)"),
    (5, "MAE"): (1e2, "MAE (×10⁻²)"),
}


def significance(probability: float) -> str:
    if probability < 0.01:
        return "***"
    if probability < 0.05:
        return "**"
    if probability < 0.10:
        return "*"
    return ""


def secondary_dm(differences: list[float]):
    """Compute DM from gamma0 and sqrt((n - 1) / n), for either horizon."""
    n = len(differences)
    if n < 2:
        raise ValueError("secondary DM requires at least two target dates")
    mean = math.fsum(differences) / n
    gamma0 = math.fsum((value - mean) ** 2 for value in differences) / n
    if not math.isfinite(gamma0) or gamma0 <= 0:
        raise ValueError("secondary DM requires positive finite zero-lag variance")
    statistic = mean / math.sqrt(gamma0 / n) * math.sqrt((n - 1) / n)
    probability = student_t_two_sided(abs(statistic), n - 1)
    if not math.isfinite(statistic) or not 0 <= probability <= 1:
        raise ValueError("non-finite secondary DM or invalid Student-t probability")
    return {
        "n_dates": n, "DM": statistic, "p_two_sided": probability,
        "stars": significance(probability),
        "mean_loss_differential": mean, "gamma0": gamma0, "df": n - 1,
    }


def calculate_tables(data_directory: Path):
    means, daily_losses, sources = {}, {}, []
    for horizon in (1, 5):
        groups, source = read_forecasts(
            data_directory / f"aggregate_predictions_h{horizon}_20groups.csv", horizon)
        if source["n_dates"] != EXPECTED_DATES[horizon]:
            raise ValueError(f"h{horizon}: expected {EXPECTED_DATES[horizon]} target dates, "
                             f"got {source['n_dates']}")
        sources.append(source)
        for index, (model, _) in enumerate(MODELS):
            errors = [[record.actual - record.predictions[index]
                       for _, record in sorted(records.items())]
                      for _, records in sorted(groups.items())]
            group_losses = {
                "MSE": [[error * error for error in group] for group in errors],
                "MAE": [[abs(error) for error in group] for group in errors],
            }
            for loss, losses in group_losses.items():
                mean = math.fsum(math.fsum(group) / len(group) for group in losses) / len(losses)
                if not math.isfinite(mean):
                    raise ValueError(f"h{horizon}/{model}/{loss}: non-finite mean loss")
                means[horizon, model, loss] = mean
                daily_losses[horizon, model, loss] = [
                    math.fsum(values) / len(losses) for values in zip(*losses)]

    a12_rows = []
    for horizon in (1, 5):
        for loss in ("MSE", "MAE"):
            scale, unit = LOSS_UNITS[horizon, loss]
            row = {"h": str(horizon), "Loss unit": unit}
            row.update({model: f"{means[horizon, model, loss] * scale:.2f}"
                        for model, _ in MODELS})
            a12_rows.append(row)

    a13_rows, tests = [], []
    for horizon in (1, 5):
        for loss in ("MAE", "MSE"):
            row = {"h": f"{horizon}-day", "Loss": loss}
            informer = daily_losses[horizon, "Informer", loss]
            for model, _ in MODELS[1:]:
                comparator = daily_losses[horizon, model, loss]
                test = secondary_dm([left - right for left, right in zip(informer, comparator)])
                row[model] = f"{test['DM']:.3f}{test['stars']}"
                tests.append({"horizon": horizon, "loss": loss, "comparator": model, **test})
            a13_rows.append(row)
    return a12_rows, a13_rows, {"inputs": sources, "mean_losses": means, "dm_tests": tests}


def write_table(rows, columns, stem, title, notes, output_directory: Path):
    csv_path = output_directory / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[column].replace("*", "\\*") for column in columns) + " |")
    lines.extend(["", *notes, ""])
    md_path = output_directory / f"{stem}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def write_outputs(a12_rows, a13_rows, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    a12_paths = write_table(
        a12_rows, A12_COLUMNS, A12_STEM, "Table A12. Secondary aggregate mean forecast losses",
        ["MSE and MAE are computed over target dates within each group, then averaged across the 20 groups.",
         "Values are rounded to two decimals in the indicated units. Each horizon retains its own target-date sample."],
        output_directory)
    a13_paths = write_table(
        a13_rows, A13_COLUMNS, A13_STEM, "Table A13. Secondary predictive-accuracy statistics",
        ["For each target date, losses are averaged across the 20 groups. The loss differential is Informer minus comparator.",
         "With n target dates, d̄ is the mean differential and γ₀ = Σ(dₜ − d̄)² / n. "
         "DM = d̄ / √(γ₀/n) × √((n − 1)/n). Two-sided probabilities use Student-t with n − 1 degrees of freedom.",
         "Negative DM statistics favor Informer. Stars denote unadjusted p-values: "
         "\\* p < 0.10, \\*\\* p < 0.05 and \\*\\*\\* p < 0.01. "
         "DM values are rounded to three decimals; significance uses unrounded statistics."],
        output_directory)
    return (*a12_paths, *a13_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Tables A12 and A13 from saved aggregate forecasts.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Directory containing the h1/h5 aggregate_predictions CSV files.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table A12/A13 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        a12_rows, a13_rows, diagnostics = calculate_tables(arguments.data_dir)
        outputs = write_outputs(a12_rows, a13_rows, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Tables A12/A13 failed: {error}", file=sys.stderr)
        return 1
    for source in diagnostics["inputs"]:
        print(f"Validated h{source['horizon']}: {source['n_groups']} groups x "
              f"{source['n_dates']} dates ({source['rows']} rows).")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
