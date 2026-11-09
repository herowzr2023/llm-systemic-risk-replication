"""Reproduce Tables A9, A10 and A11 from paired direct forecasts.

Usage: python Table_A9_A11_High_Risk_Robustness.py [--data-dir PATH]
       [--history PATH] [--output-dir PATH]

Use Python's standard library and the Table 5 panel readers. Raw CoVaR history
is converted to SR = -100 * raw CoVaR before constructing the latest 252
observations through each forecast origin. Metrics are calculated per seed,
averaged within banks, then averaged equally across all 14 banks; undefined
components propagate through both means.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import math
from pathlib import Path
import sys

from Table_05_Primary_High_Risk_Evaluation import (
    BANKS,
    Forecast,
    N_DATES,
    N_HISTORY,
    N_SEEDS,
    evaluate as evaluate_primary,
    read_forecasts,
    read_history,
    strict_mean,
)


ROOT = Path(__file__).resolve().parent
SPECIFICATIONS = (("no_text", "no_text"), ("text", "text"))
TABLES = {
    9: (
        "Table_A9_Supplementary_Rolling_Q90",
        "Table A9. Supplementary rolling-q90 metrics",
        ("HR-MSE", "HR-Bias", "FPR", "Precision"),
    ),
    10: (
        "Table_A10_Rolling_Q95_Robustness",
        "Table A10. Rolling-q95 robustness",
        ("HR-MAE", "Recall", "F1", "AP", "FPR"),
    ),
    11: (
        "Table_A11_ECDF_Three_Level_Robustness",
        "Table A11. Recent-history ECDF three-level robustness",
        ("Accuracy", "Macro-F1", "Ordinal MAE", "High recall", "High FPR"),
    ),
}
AGGREGATION_NOTE = (
    "Metrics are calculated within each seed, averaged over 20 paired seeds within each bank, "
    "then averaged equally over 14 banks. Any undefined required seed or bank makes the "
    "corresponding summary NA; available subsets do not replace strict means."
)
NOTES = {
    9: (
        "Note. SR = -100 × raw CoVaR. Each q90 cutoff is the linearly interpolated 90th "
        "percentile of the latest 252 observed SR values through and including the forecast "
        "origin. Actual high-risk days and forecast alarms use ≥ the same cutoff. HR-MSE "
        "and HR-Bias use realized high-risk days; signed error is forecast minus realization. "
        "FPR = FP/(FP+TN) and Precision = TP/(TP+FP). Conditional errors without realized "
        "events and rates with zero denominators are NA. " + AGGREGATION_NOTE
    ),
    10: (
        "Note. SR = -100 × raw CoVaR. Each q95 cutoff is the linearly interpolated 95th "
        "percentile of the latest 252 observed SR values through and including the forecast "
        "origin. Actual high-risk days and forecast alarms use ≥ the same cutoff. HR-MAE "
        "uses realized high-risk days. Recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN), and "
        "FPR = FP/(FP+TN). AP is noninterpolated average precision of forecast-minus-cutoff "
        "margins, with tied scores grouped. Conditional errors without realized events, rates "
        "with zero denominators, and one-class AP are NA. " + AGGREGATION_NOTE
    ),
    11: (
        "Note. SR = -100 × raw CoVaR. At each forecast origin, actual and predicted SR are "
        "ranked separately against the latest 252 observed SR values through and including "
        "that origin. The ECDF count is the number of history values ≤ SR. Counts below 84 "
        "give grade 0, counts 84–167 give grade 1, and counts 168–252 give grade 2. Accuracy "
        "is the fraction of matching grades; ordinal MAE is mean absolute grade distance. "
        "Macro-F1 averages the three count-based class F1 scores within a seed and is NA "
        "unless all three actual classes have positive support. High recall and High FPR "
        "treat grade 2 as the positive class; zero denominators are NA. " + AGGREGATION_NOTE
    ),
}


def linear_quantile(ordered: list[float], quantile: float) -> float:
    """Linearly interpolate a quantile of an ascending, nonempty history window."""
    if not ordered or not 0 <= quantile <= 1:
        raise ValueError("a quantile requires a nonempty history and a probability in [0, 1]")
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def origin_windows(timing, dates, histories, horizon: int):
    """Return sorted, origin-inclusive 252-observation SR reference windows."""
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
    return {
        code: [sorted(values[end - N_HISTORY:end]) for end in ends]
        for code, values in histories.items()
    }


def evaluate_binary(records: list[Forecast], cutoffs: list[float], field: str):
    """Evaluate binary tail events, including errors only on realized event dates."""
    if len(records) != len(cutoffs):
        raise ValueError("forecast and cutoff lengths differ")
    primary = evaluate_primary(records, cutoffs, field)
    errors = [getattr(record, field) - record.actual
              for record, cutoff in zip(records, cutoffs) if record.actual >= cutoff]
    return {
        "HR-MAE": primary["HR MAE"],
        "HR-MSE": strict_mean([error * error for error in errors]),
        "HR-Bias": strict_mean(errors),
        "Recall": primary["Recall"],
        "Precision": primary["Precision"],
        "F1": primary["F1"],
        "AP": primary["AP"],
        "FPR": primary["FPR"],
    }


def ecdf_grade(ordered_history: list[float], value: float) -> int:
    """Use the right-closed ECDF count and exact integer tercile boundaries."""
    if len(ordered_history) != N_HISTORY:
        raise ValueError(f"ECDF grading requires {N_HISTORY} history observations")
    count = bisect_right(ordered_history, value)
    return int(3 * count >= N_HISTORY) + int(3 * count >= 2 * N_HISTORY)


def evaluate_three_level(records: list[Forecast], windows: list[list[float]], field: str):
    """Evaluate count-based grade metrics within one seed."""
    if not records or len(records) != len(windows):
        raise ValueError("three-level evaluation requires aligned nonempty forecasts and histories")
    actual = [ecdf_grade(window, record.actual) for record, window in zip(records, windows)]
    predicted = [ecdf_grade(window, getattr(record, field)) for record, window in zip(records, windows)]
    confusion = [[0] * 3 for _ in range(3)]
    for reference, forecast in zip(actual, predicted):
        confusion[reference][forecast] += 1
    class_f1 = []
    for grade in range(3):
        support = sum(confusion[grade])
        predicted_support = sum(row[grade] for row in confusion)
        class_f1.append(2 * confusion[grade][grade] / (support + predicted_support)
                        if support else None)
    tp = confusion[2][2]
    positives = sum(confusion[2])
    fp = confusion[0][2] + confusion[1][2]
    negatives = len(records) - positives
    return {
        "Accuracy": sum(confusion[grade][grade] for grade in range(3)) / len(records),
        "Macro-F1": strict_mean(class_f1),
        "Ordinal MAE": strict_mean([abs(reference - forecast)
                                    for reference, forecast in zip(actual, predicted)]),
        "High recall": tp / positives if positives else None,
        "High FPR": fp / negatives if negatives else None,
    }


def aggregate_runs(banks, references, field: str, evaluator, metrics):
    """Average complete seed sets within banks, then complete banks equally."""
    by_bank = {}
    undefined = {metric: 0 for metric in metrics}
    for code, _ in BANKS:
        seeds = banks[code]
        runs = [evaluator(seeds[seed], references[code], field) for seed in sorted(seeds)]
        by_bank[code] = {metric: strict_mean([run[metric] for run in runs]) for metric in metrics}
        for metric in metrics:
            undefined[metric] += sum(run[metric] is None for run in runs)
    return ({metric: strict_mean([by_bank[code][metric] for code, _ in BANKS])
             for metric in metrics}, undefined)


def calculate_tables(data_directory: Path, history_path: Path):
    loaded = {h: read_forecasts(data_directory / f"predictions_{h}day_20pairs.csv", h)
              for h in (1, 5)}
    if loaded[1][2] != loaded[5][2]:
        raise ValueError("actual_SR or target dates differ across horizons")
    dates, histories = read_history(history_path)
    tables = {number: [] for number in TABLES}
    undefined_runs = {}
    for horizon in (1, 5):
        banks, timing, _ = loaded[horizon]
        windows = origin_windows(timing, dates, histories, horizon)
        q90 = {code: [linear_quantile(window, 0.90) for window in history]
               for code, history in windows.items()}
        q95 = {code: [linear_quantile(window, 0.95) for window in history]
               for code, history in windows.items()}
        for specification, field in SPECIFICATIONS:
            for number, references, evaluator in (
                (9, q90, evaluate_binary),
                (10, q95, evaluate_binary),
                (11, windows, evaluate_three_level),
            ):
                metrics = TABLES[number][2]
                summary, undefined = aggregate_runs(banks, references, field, evaluator, metrics)
                tables[number].append({"h": horizon, "Spec": specification, **summary})
                undefined_runs[number, horizon, specification] = undefined
    return tables, {"calendar_dates": len(dates), "undefined_runs": undefined_runs}


def write_outputs(tables, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for number, (stem, title, metrics) in TABLES.items():
        columns = ("h", "Spec", *metrics)
        display = [{key: ("NA" if value is None else f"{value:.6f}")
                    if key in metrics else str(value) for key, value in row.items()}
                   for row in tables[number]]
        csv_path = output_directory / f"{stem}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(display)
        lines = [f"# {title}", "", "| " + " | ".join(columns) + " |",
                 "| " + " | ".join("---" for _ in columns) + " |"]
        lines.extend("| " + " | ".join(row[column] for column in columns) + " |" for row in display)
        lines.extend(["", NOTES[number], ""])
        md_path = output_directory / f"{stem}.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        outputs.extend((csv_path, md_path))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Tables A9, A10 and A11 from saved paired forecasts.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Directory containing predictions_1day_20pairs.csv and predictions_5day_20pairs.csv.")
    parser.add_argument("--history", type=Path, default=ROOT / "data-raw" / "All_CoVar_no_other_diff.csv",
                        help="Full daily raw CoVaR history CSV.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table A9/A10/A11 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        tables, diagnostics = calculate_tables(arguments.data_dir, arguments.history)
        outputs = write_outputs(tables, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Tables A9/A10/A11 failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated {len(BANKS)} banks x 2 horizons x {N_SEEDS} paired seeds x {N_DATES} dates.")
    print(f"Source calendar: {diagnostics['calendar_dates']} dates; all h1/h5 steps match.")
    for (number, horizon, specification), counts in diagnostics["undefined_runs"].items():
        for metric, count in counts.items():
            if count:
                print(f"Table A{number}, h{horizon}, {specification}: {metric} is undefined in {count} runs.")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
