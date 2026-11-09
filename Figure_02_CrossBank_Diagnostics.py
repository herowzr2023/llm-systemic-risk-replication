"""Render Figure 2 from data/monthly_median_range.csv.

Pixel-identical rendering: Python 3.14.6 and Matplotlib 3.11.1.
See the Figure 2 environment in README.md.
"""

import argparse
import calendar
import csv
from datetime import datetime
from io import BytesIO
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "monthly_median_range.csv"
OUTPUT = ROOT / "plots" / "Figure_02_CrossBank_Diagnostics.png"
SERIES = [
    ("Emotional_Skew", "Sentiment direction"),
    ("Total_Count", "Attention-based intensity"),
    ("bank_CoVaR", "SR proxy"),
    ("risk_spillover", "DY spillover To"),
]


def read_summary():
    with INPUT.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"month", "metric", "median", "minimum", "maximum", "n_banks"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Monthly CSV is missing required columns")
        summary = list(reader)
    if not summary:
        raise ValueError("Monthly CSV is empty")
    keys = set()
    metrics = {metric for metric, _ in SERIES}
    months = []
    for row in summary:
        month = datetime.fromisoformat(row["month"])
        key = (row["metric"], month.strftime("%Y-%m"))
        if row["metric"] not in metrics or key in keys:
            raise ValueError(f"Unknown metric or duplicate month: {key}")
        values = [float(row[name]) for name in ("minimum", "median", "maximum")]
        if not np.isfinite(values).all() or not values[0] <= values[1] <= values[2]:
            raise ValueError(f"Invalid min/median/max: {key}")
        if int(row["n_banks"]) != 14:
            raise ValueError(f"Expected 14 banks: {key}")
        keys.add(key)
        months.append(month)
    if {row["metric"] for row in summary} != metrics:
        raise ValueError("Monthly CSV must contain all four metrics")
    dates = []
    cursor = min(months).replace(day=1)
    final = max(months).replace(day=1)
    while cursor <= final:
        dates.append(cursor.replace(day=calendar.monthrange(cursor.year, cursor.month)[1]))
        cursor = (
            cursor.replace(year=cursor.year + 1, month=1)
            if cursor.month == 12
            else cursor.replace(month=cursor.month + 1)
        )
    return summary, dates


def render(summary, dates):
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharex=True)
    line_color, band_color = "#000000", "#D9D9D9"
    for axis, (metric, title) in zip(axes.ravel(), SERIES):
        lookup = {row["month"][:7]: row for row in summary if row["metric"] == metric}
        arrays = {
            key: np.array([
                float(lookup[date.strftime("%Y-%m")][key])
                if date.strftime("%Y-%m") in lookup else np.nan
                for date in dates
            ])
            for key in ["minimum", "median", "maximum"]
        }
        axis.fill_between(dates, arrays["minimum"], arrays["maximum"], color=band_color, linewidth=0, zorder=1)
        axis.plot(dates, arrays["minimum"], color="#777777", linewidth=.5, linestyle="--", zorder=2)
        axis.plot(dates, arrays["maximum"], color="#777777", linewidth=.5, linestyle="--", zorder=2)
        axis.plot(dates, arrays["median"], color=line_color, linewidth=1.5, zorder=3)
        axis.axhline(0, color="#AAAAAA", linewidth=.5, linestyle=":", zorder=2)
        axis.set_title(title, loc="center", pad=8, fontsize=11)
        axis.set_xlabel("Date")
        axis.set_ylabel("Standardized monthly mean", labelpad=7)
        axis.xaxis.set_major_locator(mdates.YearLocator(2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.tick_params(axis="both", labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=.01)
    fig.legend(handles=[
        Line2D([0], [0], color=line_color, lw=1.5, label="Median (14 banks)"),
        Patch(facecolor=band_color, edgecolor="none", label="Min–max range"),
    ], loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.075, right=.985, bottom=.135, top=.955, wspace=.23, hspace=.31)
    with BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=600, facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return image.convert("L"), image.info.get("dpi", (600, 600))


def apply_labels(image, font_path):
    if image.mode != "L" or image.size != (7200, 4080):
        raise ValueError("Expected a 7200 x 4080 grayscale render")
    before = np.asarray(image).copy()
    corrected = image.copy()
    draw = ImageDraw.Draw(corrected)
    font = ImageFont.truetype(str(font_path), 104)
    panel_font = ImageFont.truetype(str(font_path), 100)
    allowed = np.zeros(before.shape, dtype=bool)
    label_boxes = [
        (190, 310, 283, 1500), (3876, 310, 3968, 1500),
        (262, 2208, 355, 3397), (3876, 2208, 3968, 3397),
    ]
    label = "Standardized value"
    bbox = font.getbbox(label)
    text = Image.new("L", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), 255)
    ImageDraw.Draw(text).text((4 - bbox[0], 4 - bbox[1]), label, font=font, fill=0)
    vertical = text.rotate(90, expand=True)
    for x0, y0, x1, y1 in label_boxes:
        if vertical.width > x1 - x0 or vertical.height > y1 - y0:
            raise ValueError("Label font does not fit the original label area")
        allowed[y0:y1, x0:x1] = True
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
        corrected.paste(vertical, (
            round((x0 + x1 - vertical.width) / 2),
            round((y0 + y1 - vertical.height) / 2),
        ))
    for label, (x, y) in zip(
        ["(a)", "(b)", "(c)", "(d)"],
        [(550, 28), (4160, 28), (550, 1925), (4160, 1925)],
    ):
        bbox = panel_font.getbbox(label)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if not (before[y:y + height, x:x + width] >= 250).all():
            raise ValueError("Panel label would cover existing plot content")
        draw.text((x - bbox[0], y - bbox[1]), label, font=panel_font, fill=0)
        allowed[y:y + height, x:x + width] = True
    after = np.asarray(corrected)
    if ((before != after) & ~allowed).any():
        raise ValueError("Label correction changed pixels outside the label areas")
    plot_boxes = [
        (537, 181, 3480, 1636), (4151, 181, 7080, 1636),
        (537, 2080, 3480, 3534), (4151, 2080, 7080, 3534),
    ]
    for x0, y0, x1, y1 in plot_boxes:
        if not np.array_equal(before[y0:y1, x0:x1], after[y0:y1, x0:x1]):
            raise ValueError("Label correction changed plot pixels")
    return corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--font", type=Path,
        default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "times.ttf",
        help="Times New Roman .ttf path (defaults to the Windows Fonts directory)",
    )
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("Times New Roman font was not found; supply its path with --font")
    summary, dates = read_summary()
    image, dpi = render(summary, dates)
    corrected = apply_labels(image, args.font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    corrected.save(OUTPUT, format="PNG", dpi=dpi)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
