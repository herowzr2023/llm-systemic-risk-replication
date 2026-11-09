"""Reproduce Table 3 from the paired bank-level prediction files.

The table uses the 20-seed mean loss ratio for each bank, HLN-corrected DM
tests with unadjusted two-sided p-values, and joint circular
forecast-date block bootstrap intervals for the median bank reduction.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
N_SEEDS = 20
N_DATES = 416
N_BANKS = 14
N_BOOTSTRAP = 5000
BLOCK_LENGTH = 10
BOOTSTRAP_SEED = 20260930
LOSSES = ("MSE", "MAE")


def load_table4():
    path = ROOT / "Table_04_Bank_Level_Forecast_Results.py"
    spec = importlib.util.spec_from_file_location("table4_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def circular_indices(rng: np.random.Generator, n_dates: int, n_draws: int, block_length: int):
    starts = rng.integers(0, n_dates, size=(n_draws, math.ceil(n_dates / block_length)))
    indices = ((starts[:, :, None] + np.arange(block_length)) % n_dates).reshape(n_draws, -1)[:, :n_dates]
    return indices


def load_panel(data_dir: Path, horizon: int, helpers):
    banks, timing, actuals, diagnostics = helpers.read_forecasts(
        data_dir / f"predictions_{horizon}day_20pairs.csv", horizon
    )
    targets = [target for _, target in timing]
    positions = list(range(N_DATES))
    bank_codes = [code for code, _ in helpers.BANKS]
    errors = np.empty((N_BANKS, N_SEEDS, 2, N_DATES), dtype=float)
    for bank_index, code in enumerate(bank_codes):
        seeds = sorted(banks[code])
        assert len(seeds) == N_SEEDS
        for seed_index, seed in enumerate(seeds):
            records = banks[code][seed]
            for date_index, target in enumerate(targets):
                record = records[target]
                errors[bank_index, seed_index, 0, date_index] = record.no_text - record.actual
                errors[bank_index, seed_index, 1, date_index] = record.text - record.actual
    return errors, positions, diagnostics


def compute_summary(data_dir: Path):
    helpers = load_table4()
    panels = {h: load_panel(data_dir, h, helpers) for h in (1, 5)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    diagnostics = []
    for horizon in (1, 5):
        errors, positions, input_diagnostics = panels[horizon]
        starts, indices = None, None
        bootstrap_weights = None
        for loss_index, loss in enumerate(LOSSES):
            transform = errors ** 2 if loss == "MSE" else np.abs(errors)
            no_loss = transform[:, :, 0, :]
            text_loss = transform[:, :, 1, :]
            no_means = no_loss.mean(axis=1).mean(axis=1)
            text_means = text_loss.mean(axis=1).mean(axis=1)
            reductions = 100.0 * (1.0 - text_means / no_means)
            dm_statistics, raw_p = [], []
            for bank_index in range(N_BANKS):
                daily_difference = (text_loss[bank_index] - no_loss[bank_index]).mean(axis=0).tolist()
                statistic, probability, *_ = helpers.dm_test(daily_difference, positions, horizon)
                dm_statistics.append(statistic)
                raw_p.append(probability)
            adjusted_p = raw_p
            if starts is None:
                starts = rng.integers(0, N_DATES, size=(N_BOOTSTRAP, math.ceil(N_DATES / BLOCK_LENGTH)))
                indices = ((starts[:, :, None] + np.arange(BLOCK_LENGTH)) % N_DATES).reshape(
                    N_BOOTSTRAP, -1
                )[:, :N_DATES]
                bootstrap_weights = np.stack(
                    [np.bincount(index, minlength=N_DATES) for index in indices]
                ).astype(float) / N_DATES
            mean_seed_date_loss = transform[:, :, loss_index, :].mean(axis=1)
            # Use the same resampled dates for both specifications.
            mean_seed_date_pair = transform[:, :, :, :].mean(axis=1)
            sampled_pair = np.stack(
                [bootstrap_weights @ mean_seed_date_pair[:, specification, :].T for specification in (0, 1)],
                axis=2,
            )
            draws = np.median(100.0 * (1.0 - sampled_pair[:, :, 1] / sampled_pair[:, :, 0]), axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
            significant_improved = sum(stat < 0 and p < 0.05 for stat, p in zip(dm_statistics, adjusted_p))
            significant_worse = sum(stat > 0 and p < 0.05 for stat, p in zip(dm_statistics, adjusted_p))
            rows.append({
                "Horizon": str(horizon), "Loss": loss,
                "Improved": f"{int(np.count_nonzero(reductions > 0))}/14",
                "DM-significant / improved": f"{significant_improved}/14",
                "DM-significant / worse": f"{significant_worse}/14",
                "Median / reduction %": f"{np.median(reductions):.2f}",
                "95% interval %": f"{lower:.2f}–{upper:.2f}",
            })
            diagnostics.append({
                "horizon": horizon, "loss": loss, "median_unrounded": float(np.median(reductions)),
                "ci_lower": float(lower), "ci_upper": float(upper),
                "dm_p": raw_p,
                "input": input_diagnostics,
            })
    return rows, diagnostics


def write_outputs(rows, diagnostics, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "Table_03_CrossBank_Summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Table 3. Primary cross-bank summary of direct forecast-loss changes", "",
        "| Horizon | Loss | Improved | DM-significant / improved | DM-significant / worse | Median / reduction % | 95% interval % |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(row.values()) + " |" for row in rows)
    lines.extend([
        "", "Each bank's reduction uses the ratio of its 20-seed mean losses.",
        "DM significance uses unadjusted two-sided p < 0.05 for each comparison.",
        "The interval is the 2.5th–97.5th percentile from 5,000 joint circular forecast-date block draws with block length 10.",
    ])
    md_path = output_dir / "Table_03_CrossBank_Summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables")
    args = parser.parse_args()
    try:
        rows, diagnostics = compute_summary(args.data_dir)
        outputs = write_outputs(rows, diagnostics, args.output_dir)
    except (OSError, ValueError, csv.Error, UnicodeError) as error:
        print(f"Table 3 failed: {error}", file=sys.stderr)
        return 1
    print("Computed Table 3 from 14 banks, 20 paired seeds, 416 forecast dates and 5,000 bootstrap draws.")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
