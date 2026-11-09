"""Train direct Informer forecasts with paired seeds.

Examples (run from the replication package):
  python code/train_informer.py --list-runs
  python code/train_informer.py --bank-code 000001 --horizon 1 --group 1 --dry-run
  python code/train_informer.py --bank-code 000001 --horizon 1 --group 1 --output-dir ../training_output
  python code/train_informer.py --export --horizon 1 --output-dir ../training_output

Without filters, training runs all 560 bank/horizon/seed pairs (1,120 fits).
For each bank and horizon, a random seed is drawn and recorded for each
experiment group. The models with and without sentiment use the same seed.
training_seeds.csv records all training seeds; this script reuses them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
CODE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CODE_DIR.parent
BANKS = {
    "000001": ("ping_an_bank", "Ping An Bank"),
    "002142": ("bank_of_ningbo", "Bank of Ningbo"),
    "600000": ("shanghai_pudong_development_bank", "Shanghai Pudong Development Bank"),
    "600015": ("huaxia_bank", "Huaxia Bank"),
    "600016": ("china_minsheng_bank", "China Minsheng Bank"),
    "600036": ("china_merchants_bank", "China Merchants Bank"),
    "601009": ("bank_of_nanjing", "Bank of Nanjing"),
    "601166": ("industrial_bank", "Industrial Bank"),
    "601169": ("bank_of_beijing", "Bank of Beijing"),
    "601328": ("bank_of_communications", "Bank of Communications"),
    "601398": ("industrial_and_commercial_bank_of_china", "Industrial and Commercial Bank of China"),
    "601939": ("china_construction_bank", "China Construction Bank"),
    "601988": ("bank_of_china", "Bank of China"),
    "601998": ("china_citic_bank", "China CITIC Bank"),
}
SEED_COLUMNS = ["bank_code", "bank_name", "horizon", "group", "seed"]
INPUT_COLUMNS = [
    "date", "systemic_risk", "dy_connectedness_from", "dy_connectedness_to",
    "dy_connectedness_net", "price_to_earnings_ratio", "price_to_book_ratio",
    "price_to_sales_ratio", "price_to_cash_flow_ratio", "liquidity_spread_3m",
    "term_spread_10y_6m", "aaa_minus_government_spread_10y", "market_volatility",
    "market_log_return", "attention_post_count", "sentiment_direction",
]
PREDICTION_COLUMNS = [
    "bank", "bank_code", "seed", "horizon", "origin_date", "target_date", "group",
    "window_id", "actual_SR", "predicted_SR_no_text", "predicted_SR_text",
]
FIXED_CONFIGURATION = {
    "model": "informer", "features": "MS", "freq": "d", "d_model": 1024,
    "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 2048, "factor": 5,
    "embed": "timeF", "dropout": 0.05, "attn": "prob", "activation": "gelu",
    "distil": True, "mix": True, "train_epochs": 50, "patience": 15,
    "batch_size": 16, "test_batch_size": 32,
    "learning_rate": 0.0001, "loss": "mse", "lradj": "type1",
    "scaler": "minmax", "inverse": True, "use_amp": False, "padding": 0,
    "num_workers": 0,
}


def configuration(horizon):
    return dict(FIXED_CONFIGURATION, seq_len=5 if horizon == 1 else 10,
                label_len=2, pred_len=1)


def seed_runs():
    with (CODE_DIR / "training_seeds.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SEED_COLUMNS:
            raise ValueError("Unexpected training_seeds.csv columns")
        runs = [dict(bank_code=r["bank_code"], bank_name=r["bank_name"],
                     horizon=int(r["horizon"]), group=int(r["group"]), seed=int(r["seed"]))
                for r in reader]
    keys = {(r["bank_code"], r["horizon"], r["group"]) for r in runs}
    expected = {(bank, horizon, group) for bank in BANKS for horizon in (1, 5)
                for group in range(1, 21)}
    if len(runs) != 560 or keys != expected:
        raise ValueError("Seed list must contain exactly 20 groups for each bank/horizon")
    for bank in BANKS:
        for horizon in (1, 5):
            seeds = [r["seed"] for r in runs if r["bank_code"] == bank and r["horizon"] == horizon]
            if len(set(seeds)) != 20 or any(not 0 <= seed < 2**32 for seed in seeds):
                raise ValueError(f"Invalid or repeated seeds: {bank}, h{horizon}")
    if any(r["bank_name"] != BANKS[r["bank_code"]][1] for r in runs):
        raise ValueError("Unexpected bank names in the seed list")
    return sorted(runs, key=lambda r: (r["bank_code"], r["horizon"], r["group"]))


def input_path(bank, variant, period="training_2012"):
    slug = BANKS[bank][0]
    return (PACKAGE_DIR / "data-raw" / period / variant /
            f"{slug}_{bank}_daily_{variant}.csv")


def read_input(bank, variant):
    with input_path(bank, variant).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = INPUT_COLUMNS if variant == "with_sentiment" else INPUT_COLUMNS[:-2]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected input columns: {bank}, {variant}")
        rows = list(reader)
    dates = [row["date"] for row in rows]
    if not dates or any(not day for day in dates) or dates != sorted(set(dates)):
        raise ValueError(f"Expected nonempty, unique increasing input dates: {bank}, {variant}")
    if any(not math.isfinite(float(row[column])) for row in rows for column in expected[1:]):
        raise ValueError(f"Non-finite input: {bank}, {variant}")
    return rows


def trading_calendar():
    # Build the trading calendar from the full-period bank-date union.
    # Dates missing from a model input still count toward
    # forecast leads and observation histories.
    dates = set()
    for bank in BANKS:
        with input_path(bank, "with_sentiment", "full_period").open(
                encoding="utf-8-sig", newline="") as handle:
            dates.update(row["date"] for row in csv.DictReader(handle))
    return sorted(dates)


def planned_windows(dates, calendar, horizon):
    seq_len = configuration(horizon)["seq_len"]
    indices = {day: i for i, day in enumerate(calendar)}
    positions = [indices[day] for day in dates]
    by_position = {position: row for row, position in enumerate(positions)}
    train_end, test_start = int(len(dates) * 0.7), len(dates) - int(len(dates) * 0.2)
    windows = {"train": [], "val": [], "test": []}
    for origin in range(seq_len - 1, len(dates)):
        position = positions[origin]
        if positions[origin - seq_len + 1:origin + 1] != list(range(position - seq_len + 1, position + 1)):
            continue
        target = by_position.get(position + horizon)
        if target is not None:
            split = "train" if target < train_end else ("val" if target < test_start else "test")
            windows[split].append((origin, target))
    return windows


def input_plan(bank, horizon, calendar):
    with_rows = read_input(bank, "with_sentiment")
    without_rows = read_input(bank, "without_sentiment")
    if len(with_rows) != len(without_rows) or any(
            {key: row[key] for key in INPUT_COLUMNS[:-2]} != other
            for row, other in zip(with_rows, without_rows)):
        raise ValueError(f"Paired input cells differ: {bank}")
    dates = [row["date"] for row in with_rows]
    windows = planned_windows(dates, calendar, horizon)
    settings = configuration(horizon)
    batch_sizes = {"train": settings["batch_size"], "val": settings["batch_size"],
                   "test": settings["test_batch_size"]}
    loader_counts = {split: len(items) // batch_sizes[split] * batch_sizes[split]
                     for split, items in windows.items()}
    if any(count == 0 for count in loader_counts.values()):
        raise ValueError(f"No complete batch in a data split: {bank}, h{horizon}")
    kept = windows["test"][:loader_counts["test"]]
    return {
        "rows": len(dates),
        "dimensions_with_without": [len(with_rows[0]) - 1, len(without_rows[0]) - 1],
        "train_end": int(len(dates) * 0.7), "test_start": len(dates) - int(len(dates) * 0.2),
        "window_counts": {split: len(items) for split, items in windows.items()},
        "loader_window_counts": loader_counts,
        "test_dates": [(dates[origin], dates[target]) for origin, target in kept],
    }


def build_args(run, variant, calendar_path, run_dir, use_gpu=False, gpu_index=0):
    path = input_path(run["bank_code"], variant)
    cols = INPUT_COLUMNS[1:] if variant == "with_sentiment" else INPUT_COLUMNS[1:-2]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return SimpleNamespace(
        **configuration(run["horizon"]), data="direct", calendar_path=str(calendar_path),
        lead=run["horizon"], root_path=str(path.parent), data_path=path.name,
        target="systemic_risk", detail_freq="d", checkpoints=str(run_dir / variant / "checkpoints"),
        enc_in=len(cols), dec_in=len(cols), c_out=1, s_layers=[3, 2, 1],
        output_attention=False, do_predict=False, cols=list(cols), itr=1,
        des=f"{run['bank_code']}_h{run['horizon']}_g{run['group']:02d}_{variant}",
        is_class=False, num_class=2, use_gpu=use_gpu,
        gpu=0 if visible not in (None, "", "-1") else gpu_index,
        use_multi_gpu=False, devices="0", device_ids=[0], path=str(path),
        predict_col="systemic_risk", seed=run["seed"],
    )


def reset_seed(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_class(torch):
    sys.path.insert(0, str(CODE_DIR / "informer"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from exp.exp_informer import Exp_Informer

    class Experiment(Exp_Informer):
        def _acquire_device(self):
            # Preserve the parent process's GPU mapping.
            if not self.args.use_gpu or not torch.cuda.is_available():
                return torch.device("cpu")
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            index = 0 if visible not in (None, "", "-1") else self.args.gpu
            return torch.device(f"cuda:{index}")

    return Experiment


def collect_predictions(experiment, np, torch):
    experiment.model.eval()
    dataset, loader = experiment._get_data(flag="test")
    expected_count = (len(dataset) // loader.batch_size * loader.batch_size
                      if loader.drop_last else len(dataset))
    rows = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dates in loader:
            outputs, truth, _ = experiment._process_one_batch(
                dataset, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dates)
            predicted = np.asarray(outputs.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            actual = np.asarray(truth.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            if predicted.shape != actual.shape:
                raise ValueError("Prediction/target shapes differ")
            if len(rows) + len(actual) > expected_count:
                raise ValueError("Test loader produced more predictions than its configured windows")
            for value, forecast in zip(actual, predicted):
                origin, target = dataset.windows[len(rows)]
                rows.append((str(dataset.data_stamp_date[origin])[:10],
                             str(dataset.data_stamp_date[target])[:10], float(value), float(forecast)))
    if len(rows) != expected_count or any(not math.isfinite(v) for row in rows for v in row[2:]):
        raise ValueError(f"Expected {expected_count} finite test predictions from the configured loader")
    return rows


def write_rows(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(run, without, with_sentiment):
    if len(without) != len(with_sentiment):
        raise ValueError("The paired prediction counts differ")
    for window, (no_text, text) in enumerate(zip(without, with_sentiment)):
        if no_text[:3] != text[:3]:
            raise ValueError("Paired origins, target dates or actual values differ")
        yield dict(bank=run["bank_name"], bank_code=run["bank_code"], seed=run["seed"],
                   horizon=run["horizon"], origin_date=text[0], target_date=text[1],
                   group=run["group"], window_id=window, actual_SR=text[2],
                   predicted_SR_no_text=no_text[3], predicted_SR_text=text[3])


def pair_path(output_dir, run):
    return (output_dir / "runs" / run["bank_code"] / f"h{run['horizon']}" /
            f"group_{run['group']:02d}" / "predictions.csv")


def export_predictions(output_dir, all_runs, horizon):
    runs = [r for r in all_runs if r["horizon"] == horizon]
    paths = [pair_path(output_dir, run) for run in runs]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Export requires all 280 pairs for h{horizon}; missing {len(missing)}")
    calendar = trading_calendar()
    plans = {bank: input_plan(bank, horizon, calendar)["test_dates"] for bank in BANKS}

    def rows():
        for run, path in zip(runs, paths):
            expected_dates = plans[run["bank_code"]]
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != PREDICTION_COLUMNS:
                    raise ValueError(f"Unexpected prediction columns: {path}")
                count = 0
                for count, row in enumerate(reader, 1):
                    if (row["bank_code"], int(row["horizon"]), int(row["group"]), int(row["seed"]),
                        int(row["window_id"])) != (run["bank_code"], horizon, run["group"], run["seed"], count - 1):
                        raise ValueError(f"Unexpected run identity: {path}")
                    if any(not math.isfinite(float(row[key])) for key in PREDICTION_COLUMNS[-3:]):
                        raise ValueError(f"Non-finite prediction: {path}")
                    if count > len(expected_dates) or (row["origin_date"], row["target_date"]) != expected_dates[count - 1]:
                        raise ValueError(f"Prediction date axis differs from the configured windows: {path}")
                    yield row
                if count != len(expected_dates):
                    raise ValueError(f"Expected {len(expected_dates)} rows: {path}")

    destination = output_dir / f"predictions_{horizon}day_20pairs.csv"
    write_rows(destination, rows())
    print(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank-code", choices=tuple(BANKS))
    parser.add_argument("--horizon", type=int, choices=(1, 5))
    parser.add_argument("--group", type=int, choices=range(1, 21))
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR.parent / "training_output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--list-runs", action="store_true", help="List selected recorded seeds; no training dependencies needed")
    actions.add_argument("--dry-run", action="store_true", help="Check inputs, dates and configuration without importing PyTorch")
    actions.add_argument("--export", action="store_true", help="Combine all 280 completed pairs per selected horizon; do not train")
    options = parser.parse_args(argv)
    all_runs = seed_runs()
    selected = [r for r in all_runs if (options.bank_code is None or r["bank_code"] == options.bank_code)
                and (options.horizon is None or r["horizon"] == options.horizon)
                and (options.group is None or r["group"] == options.group)]
    if options.list_runs:
        writer = csv.DictWriter(sys.stdout, fieldnames=SEED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
        return
    output_dir = options.output_dir.resolve()
    if output_dir == PACKAGE_DIR or PACKAGE_DIR in output_dir.parents:
        raise ValueError("Use an output directory outside the replication package")
    if options.export:
        if options.bank_code is not None or options.group is not None:
            parser.error("--export combines all banks and 20 groups; only --horizon may filter it")
        for horizon in ((options.horizon,) if options.horizon else (1, 5)):
            export_predictions(output_dir, all_runs, horizon)
        return
    calendar = trading_calendar()
    plans = {}
    for bank, horizon in sorted({(r["bank_code"], r["horizon"]) for r in selected}):
        plans[(bank, horizon)] = input_plan(bank, horizon, calendar)
    if options.dry_run:
        print(json.dumps({
            "paired_runs": len(selected), "model_fits": 2 * len(selected),
            "runs": selected, "configuration": {str(h): configuration(h) for h in sorted({r['horizon'] for r in selected})},
            "inputs": [{"bank_code": bank, "horizon": horizon,
                        **{key: value for key, value in plan.items() if key != "test_dates"},
                        "test_predictions": len(plan["test_dates"]),
                        "first": plan["test_dates"][0], "last": plan["test_dates"][-1]}
                       for (bank, horizon), plan in plans.items()],
            "output_dir": str(output_dir), "torch_imported": "torch" in sys.modules,
        }, indent=2))
        return
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Install code/requirements-training.txt in a separate training environment") from exc
    use_gpu = options.device != "cpu" and torch.cuda.is_available()
    if options.device == "cuda" and not use_gpu:
        raise RuntimeError("CUDA was requested but is not available")
    output_dir.mkdir(parents=True, exist_ok=True)
    calendar_path = output_dir / "trading_calendar.csv"
    calendar_text = "date\n" + "\n".join(calendar) + "\n"
    if calendar_path.exists():
        if calendar_path.read_text(encoding="utf-8") != calendar_text:
            raise ValueError("Existing output calendar differs from packaged raw dates")
    else:
        with calendar_path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(calendar_text)
    experiment_type = None
    for run in selected:
        destination = pair_path(output_dir, run)
        destination.parent.mkdir(parents=True, exist_ok=False)
        results = {}
        for variant in ("with_sentiment", "without_sentiment"):
            reset_seed(run["seed"], np, torch)
            if experiment_type is None:
                experiment_type = experiment_class(torch)
            args = build_args(run, variant, calendar_path, destination.parent, use_gpu, options.gpu_index)
            experiment = experiment_type(args)
            experiment.train(args.des)
            results[variant] = collect_predictions(experiment, np, torch)
            if [(r[0], r[1]) for r in results[variant]] != plans[(run["bank_code"], run["horizon"])]["test_dates"]:
                raise ValueError("Test date axis differs from the configured window rule")
            del experiment
            if use_gpu:
                torch.cuda.empty_cache()
        write_rows(destination, paired_rows(run, results["without_sentiment"], results["with_sentiment"]))
        print(destination)
    for horizon in sorted({r["horizon"] for r in selected}):
        horizon_runs = [r for r in all_runs if r["horizon"] == horizon]
        if all(pair_path(output_dir, r).is_file() for r in horizon_runs):
            export_predictions(output_dir, all_runs, horizon)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
