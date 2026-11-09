"""Reproduce Tables A6 and A7 from the paired bank-level forecasts.

For each seed, calculate MSE and MAE over the 416 common target dates,
then report the equally weighted mean and sample SD across 20 seeds.
Reductions use unrounded mean losses; wins require strictly lower text loss.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import stdev
import sys

from Table_04_Bank_Level_Forecast_Results import BANKS, N_DATES, N_SEEDS, read_forecasts


ROOT = Path(__file__).resolve().parent
TABLES = {
    1: (
        "Table_A6_One_Day_Forecast_Losses",
        "Table A6. Forecast Losses for Direct One-Trading-Day-Ahead Predictions",
    ),
    5: (
        "Table_A7_Five_Day_Forecast_Losses",
        "Table A7. Forecast Losses for Direct Five-Trading-Day-Ahead Predictions",
    ),
}
OUTPUT_COLUMNS = (
    "Bank", "MSE no-text", "MSE text", "MSE Δ%", "MAE no-text", "MAE text",
    "MAE Δ%", "MSE wins", "MAE wins",
)
NOTE = (
    "Note: For each seed, losses are computed over the 416 common target dates; "
    "entries report the mean ± sample SD across 20 paired seeds (ddof = 1). "
    "MSE is reported in units of 10⁻⁴ and MAE in units of 10⁻². "
    "Δ% = 100 × (1 − mean loss_text / mean loss_no-text), computed from "
    "unrounded mean losses with no-text as the denominator. A zero denominator "
    "is reported as NA; small positive denominators and negative reductions are "
    "retained. Wins count seeds whose text loss is strictly lower than no-text "
    "loss; ties are excluded and counts are out of 20."
)


def loss_cells(no_losses: list[float], text_losses: list[float], loss: str, unit: float):
    """Summarize paired seed-level losses in the table's display units."""
    no_mean = math.fsum(no_losses) / N_SEEDS
    text_mean = math.fsum(text_losses) / N_SEEDS
    no_sd, text_sd = stdev(no_losses), stdev(text_losses)
    if not all(math.isfinite(value) for value in (no_mean, text_mean, no_sd, text_sd)):
        raise ValueError(f"{loss}: non-finite seed-level loss summary")
    reduction = None if no_mean == 0 else 100 * (1 - text_mean / no_mean)
    if reduction is not None and not math.isfinite(reduction):
        raise ValueError(f"{loss}: non-finite percentage reduction")
    wins = sum(text < no_text for no_text, text in zip(no_losses, text_losses))
    return {
        f"{loss} no-text": f"{no_mean / unit:.2f} ± {no_sd / unit:.2f}",
        f"{loss} text": f"{text_mean / unit:.2f} ± {text_sd / unit:.2f}",
        f"{loss} Δ%": "NA" if reduction is None else f"{reduction:.2f}",
        f"{loss} wins": f"{wins}/{N_SEEDS}",
    }


def calculate_tables(data_directory: Path):
    tables, sources = {}, {}
    reference_actuals = None
    for horizon in TABLES:
        banks, timing, actuals, source = read_forecasts(
            data_directory / f"predictions_{horizon}day_20pairs.csv", horizon
        )
        if reference_actuals is None:
            reference_actuals = actuals
        elif actuals != reference_actuals:
            raise ValueError("actual_SR or target dates differ across horizons")
        targets = [target for _, target in timing]
        rows = []
        for code, abbreviation in BANKS:
            no_mse, text_mse, no_mae, text_mae = [], [], [], []
            for seed in sorted(banks[code]):
                records = banks[code][seed]
                no_errors = [records[target].no_text - records[target].actual for target in targets]
                text_errors = [records[target].text - records[target].actual for target in targets]
                no_mse.append(math.fsum(error * error for error in no_errors) / N_DATES)
                text_mse.append(math.fsum(error * error for error in text_errors) / N_DATES)
                no_mae.append(math.fsum(abs(error) for error in no_errors) / N_DATES)
                text_mae.append(math.fsum(abs(error) for error in text_errors) / N_DATES)
            row = {"Bank": abbreviation}
            row.update(loss_cells(no_mse, text_mse, "MSE", 1e-4))
            row.update(loss_cells(no_mae, text_mae, "MAE", 1e-2))
            rows.append(row)
        tables[horizon], sources[horizon] = rows, source
    return tables, sources


def write_outputs(tables, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for horizon, (stem, caption) in TABLES.items():
        rows = tables[horizon]
        csv_path = output_directory / f"{stem}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            f"# {caption}", "",
            "| " + " | ".join(OUTPUT_COLUMNS) + " |",
            "| " + " | ".join("---" for _ in OUTPUT_COLUMNS) + " |",
        ]
        lines.extend("| " + " | ".join(row[column] for column in OUTPUT_COLUMNS) + " |" for row in rows)
        lines.extend(["", NOTE, ""])
        md_path = output_directory / f"{stem}.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        outputs.extend((csv_path, md_path))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Tables A6 and A7 from saved paired forecasts.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Directory containing predictions_1day_20pairs.csv and predictions_5day_20pairs.csv.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table A6/A7 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        tables, sources = calculate_tables(arguments.data_dir)
        outputs = write_outputs(tables, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Tables A6/A7 failed: {error}", file=sys.stderr)
        return 1
    for horizon, source in sources.items():
        print(f"Validated h{horizon}: {len(BANKS)} banks x {N_SEEDS} paired seeds x "
              f"{N_DATES} dates ({source['rows']} rows).")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
