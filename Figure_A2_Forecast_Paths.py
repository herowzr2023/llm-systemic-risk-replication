"""Render the five-trading-day forecast paths from saved prediction results.

Pixel-identical rendering: Python 3.12.14 and Matplotlib 3.10.7.
See the Figures A1-A2 environment in README.md.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "figure_A2_predictions.csv"
OUTPUT = ROOT / "plots" / "Figure_A2_Forecast_Paths.png"
EXPECTED_ROWS = 676
MODELS = [
    ("Informer", "baseline_pred"),
    ("Persistence", "Persistence_pred"),
    ("AR(1)", "AR1_pred"),
    ("GBR", "GradientBoostingRegressor_pred"),
    ("PatchTST", "PatchTST_pred"),
]
STYLES = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "x", "+"]


def read_input(path: Path) -> pd.DataFrame:
    required = {
        "date", "TRUE", "baseline_pred", "GradientBoostingRegressor_pred",
        "PatchTST_pred", "Persistence_pred", "AR1_pred",
    }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS or not rows:
        raise ValueError(f"Expected {EXPECTED_ROWS} prediction observations")
    if not required.issubset(rows[0]):
        raise ValueError("The retained-path CSV is missing required columns")
    dates = [row["date"] for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Dates must be unique and increasing")
    numeric = ["TRUE", *(column for _, column in MODELS)]
    if any(not math.isfinite(float(row[column])) for row in rows for column in numeric):
        raise ValueError("The retained-path CSV contains a non-finite value")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column])
    return frame


def render(frame: pd.DataFrame, output: Path) -> None:
    figure = plt.figure(figsize=(14, 10.4))
    grid = figure.add_gridspec(3, 4)
    axes = [figure.add_subplot(grid[0, :2])]
    for position in [grid[0, 2:], grid[1, :2], grid[1, 2:], grid[2, 1:3]]:
        axes.append(figure.add_subplot(position, sharex=axes[0]))
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.085, top=0.96,
                           wspace=0.68, hspace=0.58)
    for index, (name, column) in enumerate(MODELS):
        axis = axes[index]
        axis.plot(frame.date, frame.TRUE, color="black", linewidth=1.3, label="Observed")
        axis.plot(frame.date, frame[column], color="black", linestyle=STYLES[index],
                  marker=MARKERS[index], markevery=max(1, len(frame) // 24),
                  markersize=2.4, linewidth=1.0, label="Forecast")
        axis.set_title(name, fontsize=12, pad=9)
        axis.set_xlabel("Target date")
        axis.set_ylabel("SR proxy")
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", rotation=30, labelsize=9)
        axis.legend(frameon=False, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    render(read_input(args.input.resolve()), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
