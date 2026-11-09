"""Reproduce Table A2 from the saved sentiment-classification predictions.

Run with Python's standard library only. All metrics use the three labels
-1 (negative), 0 (neutral) and 1 (positive). Weighted averages use reference
class frequencies; macro averages give each class equal weight.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
STEM = "Table_A2_Sentiment_Classification"
INPUT_NAME = "sentiment_classification_predictions.csv"
LABELS = (-1, 0, 1)
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
MODELS = (
    "Llama-3-Chinese-8B-Instruct-v3-emotion",
    "Llama-3-Chinese-8B-Instruct-v2",
    "Llama-3-Chinese-8B-Instruct-v3",
    "Llama-3-Chinese-8B-Instruct",
    "Llama-3-Chinese-8B",
    "Llama-3-8B-Instruct",
    "Llama-3-8B",
    "ROBERTa",
    "Chinese-ROBERTa-wwm-ext-large",
    "Chinese-ROBERTa-wwm-ext",
    "Chinese-BERT-wwm-ext",
    "Chinese-BERT-wwm",
)
SPLITS = (
    ("train", 7000, "Panel A. Training set (n = 7,000)"),
    ("test", 1000, "Panel B. Held-out test set (n = 1,000)"),
)
INPUT_COLUMNS = (
    "model", "split", "sample_id", "true_label", "predicted_label",
)
METRIC_COLUMNS = (
    "Accuracy", "Weighted P", "Weighted F1", "Macro P", "Macro R", "Macro F1",
)
OUTPUT_COLUMNS = ("Split", "Model", *METRIC_COLUMNS)
NOTE = (
    "P = precision; R = recall. Weighted metrics use reference-label class "
    "frequencies; macro metrics weight classes equally. ¹ In this single-label "
    "three-class setting, accuracy equals weighted recall and micro precision, "
    "recall and F1, so these values appear once. Scores use the same final "
    "pre-2012 split and three-decimal rounding. Macro and micro summaries are "
    "aggregation checks, not independent replications."
)


def read_predictions(path: Path):
    """Validate sample identities and accumulate reference-by-prediction counts."""
    matrices = {(split, model): [[0] * 3 for _ in LABELS]
                for split, _, _ in SPLITS for model in MODELS}
    sample_ids = {key: set() for key in matrices}
    references = {}
    row_count = 0
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
                model, split = row["model"], row["split"]
                if model not in MODELS:
                    raise ValueError(f"unknown model: {model!r}")
                if split not in ("train", "test"):
                    raise ValueError(f"split must be train or test, got {split!r}")
                sample_id = row["sample_id"]
                if not sample_id.strip():
                    raise ValueError("sample_id must be nonempty")
                actual = int(row["true_label"])
                predicted = int(row["predicted_label"])
                if actual not in LABELS or predicted not in LABELS:
                    raise ValueError("true_label and predicted_label must be -1, 0 or 1")
                key = (split, model)
                if sample_id in sample_ids[key]:
                    raise ValueError("duplicate model/split/sample_id identity")
                reference = (split, actual)
                if sample_id in references and references[sample_id] != reference:
                    raise ValueError("sample split or reference label differs across models")
                references[sample_id] = reference
                sample_ids[key].add(sample_id)
                matrices[key][LABEL_INDEX[actual]][LABEL_INDEX[predicted]] += 1
                row_count += 1
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}, CSV line {reader.line_num}: {error}") from error
    for split, expected, _ in SPLITS:
        reference_ids = sample_ids[split, MODELS[0]]
        for model in MODELS:
            ids = sample_ids[split, model]
            if len(ids) != expected:
                raise ValueError(f"{path.name}, {split}/{model}: expected {expected} samples, got {len(ids)}")
            if ids != reference_ids:
                raise ValueError(f"{path.name}, {split}/{model}: sample IDs do not align across models")
    return matrices, row_count


def classification_metrics(matrix):
    """Compute class metrics before applying support or equal-class weights."""
    support = [sum(row) for row in matrix]
    predicted = [sum(row[index] for row in matrix) for index in range(3)]
    total = sum(support)
    if total == 0:
        raise ValueError("cannot calculate metrics for an empty split")
    true_positive = [matrix[index][index] for index in range(3)]
    precision = [tp / count if count else 0.0
                 for tp, count in zip(true_positive, predicted)]
    recall = [tp / count if count else 0.0
              for tp, count in zip(true_positive, support)]
    f1 = [2 * tp / (actual + count) if actual + count else 0.0
          for tp, actual, count in zip(true_positive, support, predicted)]
    accuracy = sum(true_positive) / total
    weighted_recall = math.fsum(score * count for score, count in zip(recall, support)) / total
    if not math.isclose(accuracy, weighted_recall, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("weighted recall does not equal accuracy")
    return {
        "Accuracy": accuracy,
        "Weighted P": math.fsum(score * count for score, count in zip(precision, support)) / total,
        "Weighted F1": math.fsum(score * count for score, count in zip(f1, support)) / total,
        "Macro P": math.fsum(precision) / len(LABELS),
        "Macro R": math.fsum(recall) / len(LABELS),
        "Macro F1": math.fsum(f1) / len(LABELS),
        "Weighted R": weighted_recall,
    }


def calculate_table(data_directory: Path):
    matrices, row_count = read_predictions(data_directory / INPUT_NAME)
    metrics = {key: classification_metrics(matrix) for key, matrix in matrices.items()}
    rows = [{
        "Split": split,
        "Model": model,
        **{column: f"{metrics[split, model][column]:.3f}" for column in METRIC_COLUMNS},
    } for split, _, _ in SPLITS for model in MODELS]
    return rows, {"rows": row_count, "metrics": metrics, "confusion_matrices": matrices}


def write_outputs(rows, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{STEM}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Table A2. Sentiment-Classification Performance: Weighted and Macro Summaries", "",
    ]
    for split, _, heading in SPLITS:
        lines.extend([
            f"## {heading}", "",
            "| Model | Accuracy¹ | Weighted P | Weighted F1 | Macro P | Macro R | Macro F1 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in rows:
            if row["Split"] == split:
                lines.append("| " + " | ".join(row[column] for column in OUTPUT_COLUMNS[1:]) + " |")
        lines.append("")
    lines.extend([f"Note: {NOTE}", ""])
    md_path = output_directory / f"{STEM}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Table A2 from saved sentiment predictions.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help=f"Directory containing {INPUT_NAME}.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table A2 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        rows, diagnostics = calculate_table(arguments.data_dir)
        outputs = write_outputs(rows, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Table A2 failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated {len(MODELS)} models x (7,000 training + 1,000 test samples) "
          f"({diagnostics['rows']:,} rows).")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
