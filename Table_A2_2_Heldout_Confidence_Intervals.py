"""Reproduce Table A2-2 from fixed held-out sentiment predictions.

Use NumPy for 10,000 paired-case bootstrap resamples per classifier. Sampling
the empirical 3x3 reference/prediction counts preserves 1,000 cases per draw.
Report linear 2.5th and 97.5th percentiles with three-decimal rounding.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from Table_A2_Sentiment_Classification import INPUT_NAME, MODELS, read_predictions


ROOT = Path(__file__).resolve().parent
STEM = "Table_A2_2_Heldout_Confidence_Intervals"
N_BOOTSTRAP = 10000
RNG_SEED = 20261001
N_TEST = 1000
METRIC_COLUMNS = ("Accuracy", "Weighted precision", "Weighted recall", "Weighted F1")
OUTPUT_COLUMNS = ("Model", *METRIC_COLUMNS)
CAPTION = "Table A2-2. Held-out classification scores and 95% confidence intervals"
NOTE = (
    "Cells report held-out scores and marginal 95% confidence intervals: the "
    "2.5th–97.5th percentiles of 10,000 bootstrap resamples of the 1,000 paired "
    "reference and predicted labels, preserving sample size. Classifiers and "
    "finalized reference labels remain fixed. Intervals reflect held-out sampling "
    "uncertainty, not retraining or label development, and do not test model rankings."
)


def classification_scores(confusion):
    """Calculate scores for one confusion matrix or an array of bootstrap draws."""
    matrix = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diagonal(matrix, axis1=-2, axis2=-1)
    support = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)
    total = support.sum(axis=-1)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support != 0)
    f1 = np.divide(2 * true_positive, support + predicted,
                   out=np.zeros_like(true_positive), where=support + predicted != 0)
    accuracy = true_positive.sum(axis=-1) / total
    weighted_recall = (support * recall).sum(axis=-1) / total
    if not np.allclose(accuracy, weighted_recall, rtol=0.0, atol=1e-12):
        raise ValueError("weighted recall does not equal accuracy")
    return {
        "Accuracy": accuracy,
        "Weighted precision": (support * precision).sum(axis=-1) / total,
        "Weighted recall": weighted_recall,
        "Weighted F1": (support * f1).sum(axis=-1) / total,
    }


def bootstrap_scores(matrix, model_index: int):
    confusion = np.asarray(matrix, dtype=np.int64)
    size = int(confusion.sum())
    if size != N_TEST:
        raise ValueError(f"expected {N_TEST} held-out samples, got {size}")
    rng = np.random.default_rng(np.random.SeedSequence([RNG_SEED, model_index]))
    sampled = rng.multinomial(size, confusion.reshape(-1) / size,
                              size=N_BOOTSTRAP).reshape(-1, 3, 3)
    return classification_scores(confusion), classification_scores(sampled)


def calculate_table(data_directory: Path):
    matrices, row_count = read_predictions(data_directory / INPUT_NAME)
    rows, metrics = [], {}
    for model_index, model in enumerate(MODELS):
        point, replicates = bootstrap_scores(matrices["test", model], model_index)
        intervals = {
            column: np.quantile(replicates[column], [0.025, 0.975], method="linear")
            for column in METRIC_COLUMNS
        }
        rows.append({
            "Model": model,
            **{column: f"{point[column]:.3f}[{intervals[column][0]:.3f},{intervals[column][1]:.3f}]"
               for column in METRIC_COLUMNS},
        })
        metrics[model] = {
            column: {"estimate": float(point[column]), "lower": float(intervals[column][0]),
                     "upper": float(intervals[column][1])}
            for column in METRIC_COLUMNS
        }
    return rows, {"rows": row_count, "metrics": metrics}


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
    parser = argparse.ArgumentParser(description="Reproduce Table A2-2 from saved held-out sentiment predictions.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help=f"Directory containing {INPUT_NAME}.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table A2-2 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        rows, _ = calculate_table(arguments.data_dir)
        outputs = write_outputs(rows, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Table A2-2 failed: {error}", file=sys.stderr)
        return 1
    print(f"Computed {N_BOOTSTRAP:,} bootstrap resamples for each of {len(MODELS)} "
          f"classifiers ({N_TEST:,} held-out samples each).")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
