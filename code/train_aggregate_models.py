#!/usr/bin/env python3
"""Train the five aggregate-series models and write one prediction CSV per horizon.

The Informer implementation is reused under its Apache-2.0 license; see
informer/NOTICE and informer/LICENSE.  All other orchestration stays in this file.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
CODE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CODE_DIR.parent
DEFAULT_INPUT = PACKAGE_DIR / "data-raw" / "aggregate_daily.csv"
DEFAULT_SEEDS = CODE_DIR / "aggregate_training_seeds.csv"
DEFAULT_OUTPUT = PACKAGE_DIR.parent / "aggregate_training_output"
RAW_COLUMNS = (
    "date", "systemic_risk", "dy_connectedness_from", "dy_connectedness_to",
    "dy_connectedness_net", "price_to_earnings_ratio", "price_to_book_ratio",
    "price_to_sales_ratio", "price_to_cash_flow_ratio", "liquidity_spread_3m",
    "term_spread_10y_6m", "aaa_minus_government_spread_10y", "market_volatility",
    "market_log_return", "attention_post_count", "sentiment_direction",
)
TARGET = "systemic_risk"
MODEL_COLUMNS = (
    "baseline_pred", "GradientBoostingRegressor_pred", "PatchTST_pred",
    "Persistence_pred", "AR1_pred",
)
OUTPUT_COLUMNS = ("date", "TRUE", *MODEL_COLUMNS, "group_id", "seed", "nominal_horizon")
INFORMER_SETTINGS = {
    "model": "informer", "data": "direct", "features": "MS", "target": TARGET,
    "freq": "d", "detail_freq": "d", "d_model": 1024,
    "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 2048, "factor": 5,
    "dropout": 0.05, "attn": "prob", "embed": "timeF", "activation": "gelu",
    "distil": True, "mix": True, "padding": 0, "output_attention": False,
    "train_epochs": 50, "batch_size": 16, "patience": 15,
    "learning_rate": 1e-4, "loss": "mse", "lradj": "type1", "scaler": "minmax",
    "inverse": True, "is_class": False, "num_class": 2, "use_amp": False,
    "num_workers": 0, "use_multi_gpu": False, "s_layers": [3, 2, 1],
}
PATCHTST_SETTINGS = {
    "expected_historical_covariates": 14, "loss": "mae",
    "encoder_layers": 2, "n_heads": 4, "hidden_size": 128,
    "linear_hidden_size": 256, "dropout": 0.1, "fc_dropout": 0.1,
    "head_dropout": 0.0, "attn_dropout": 0.0, "patch_len": 3, "stride": 1,
    "revin": True, "revin_affine": False, "revin_subtract_last": True,
    "max_steps": 500, "learning_rate": 1e-4, "windows_batch_size": 32,
}


@dataclass(frozen=True)
class AggregateInput:
    path: Path
    dates: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    sha256: str

    def __len__(self):
        return len(self.dates)


def read_input(path: Path) -> AggregateInput:
    """Validate the raw observations without sorting, filling, or rescaling."""
    path = path.resolve()
    dates, values = [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise ValueError(f"Expected columns in this order: {', '.join(RAW_COLUMNS)}")
        previous = None
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None or value.strip() == "" for value in row.values()):
                raise ValueError(f"Missing or extra CSV fields at line {line_number}")
            day = date.fromisoformat(row["date"])
            if row["date"] != day.isoformat() or (previous is not None and day <= previous):
                raise ValueError(f"Dates must be unique, increasing ISO dates (line {line_number})")
            previous = day
            numeric = tuple(float(row[column]) for column in RAW_COLUMNS[1:])
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"Nonfinite numeric value at line {line_number}")
            dates.append(day.isoformat())
            values.append(numeric)
    if not values:
        raise ValueError("The input contains no observations")
    for column, series in zip(RAW_COLUMNS[1:], zip(*values)):
        training_series = series[:int(len(values) * 0.7)]
        if not training_series or max(training_series) == min(training_series):
            raise ValueError(f"The Informer training portion has no variation in {column}")
    return AggregateInput(path, tuple(dates), tuple(values), hashlib.sha256(path.read_bytes()).hexdigest())


def parse_horizon(value: str) -> int:
    value = value.lower().removeprefix("h")
    if value not in ("1", "5"):
        raise argparse.ArgumentTypeError("Choose h1 or h5 (also accepts 1 or 5)")
    return int(value)


def checked_seed(value: int) -> int:
    if not 0 <= value < 2**32:
        raise ValueError("Seeds must be integers from 0 through 4294967295")
    return value


def training_plan(args) -> dict[int, list[tuple[int, int]]]:
    """Return horizon -> (group, seed), with the supplied table as the default."""
    horizons = sorted(set(args.horizon or (1, 5)))
    groups = args.group
    if groups and (len(set(groups)) != len(groups) or min(groups) < 1):
        raise ValueError("Each --group must be a distinct positive integer")
    explicit = [args.seed] if args.seed is not None else args.seeds
    if explicit is not None:
        seeds = [checked_seed(seed) for seed in explicit]
        selected = groups or list(range(1, len(seeds) + 1))
        if len(selected) != len(seeds):
            raise ValueError("Supply one --group for each explicit seed, or omit --group")
        return {horizon: list(zip(selected, seeds)) for horizon in horizons}
    rows = {}
    with args.seed_file.resolve().open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("horizon", "group", "seed"):
            raise ValueError("The seed file must have columns horizon,group,seed")
        for row in reader:
            horizon, group = parse_horizon(row["horizon"]), int(row["group"])
            seed = checked_seed(int(row["seed"]))
            key = (horizon, group)
            if group < 1 or key in rows:
                raise ValueError(f"Invalid or duplicated seed-table key: {key}")
            rows[key] = seed
    plan = {}
    for horizon in horizons:
        selected = groups or sorted(group for h, group in rows if h == horizon)
        if not selected:
            raise ValueError(f"No groups in the seed table for h{horizon}")
        missing = [group for group in selected if (horizon, group) not in rows]
        if missing:
            raise ValueError(f"Missing h{horizon} groups in the seed table: {missing}")
        plan[horizon] = [(group, rows[horizon, group]) for group in selected]
    return plan


def window_plan(nobs: int, horizon: int) -> dict:
    """Plan terminal targets, chronological splits and complete batches."""
    if horizon not in (1, 5):
        raise ValueError("Only horizons 1 and 5 are supported")
    lags, batch = (5 if horizon == 1 else 10), INFORMER_SETTINGS["batch_size"]
    test_batch = 16 if horizon == 1 else 4
    train_rows, test_rows = int(nobs * 0.7), int(nobs * 0.2)
    validation_rows = nobs - train_rows - test_rows
    training_windows = train_rows - lags - horizon + 1
    validation_windows = validation_rows
    test_windows = test_rows
    retained = (test_windows // test_batch) * test_batch
    if min(training_windows, validation_windows) < batch or test_windows < test_batch:
        raise ValueError("Insufficient windows for the training or prediction batch size")
    first = nobs - test_rows
    return {
        "horizon": horizon, "lead": horizon, "lags": lags,
        "label_len": 2, "pred_len": 1, "test_batch_size": test_batch,
        "train_rows": train_rows, "validation_rows": validation_rows, "test_rows": test_rows,
        "training_windows": training_windows, "validation_windows": validation_windows,
        "test_windows": test_windows, "retained_test_origins": retained,
        "first_target_index": first, "last_target_index": first + retained - 1,
        "target_dates": retained, "first_origin_index": first - horizon,
        "last_origin_index": first + retained - 1 - horizon,
        "comparison_training_origins": training_windows,
    }


def evaluation_dates(data: AggregateInput, horizon: int) -> tuple[str, ...]:
    window = window_plan(len(data), horizon)
    return data.dates[window["first_target_index"]:window["last_target_index"] + 1]


def predictions_by_date(pairs) -> dict[str, float]:
    """Keep one scalar terminal forecast for each unique target date."""
    result = {}
    for day, prediction in pairs:
        prediction = float(prediction)
        if not math.isfinite(prediction):
            raise ValueError(f"Nonfinite prediction for {day}")
        if day in result:
            raise ValueError(f"Duplicate terminal prediction for {day}")
        result[day] = prediction
    return result


def seed_all(seed: int, torch=None, *, deterministic: bool = False):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Set cuDNN flags independently for each fitted model.
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def device_spec(request: str, torch) -> tuple[bool, int]:
    if request == "auto":
        return bool(torch.cuda.is_available()), 0
    if request == "cpu":
        return False, 0
    try:
        index = int(request.split(":", 1)[1])
    except (ValueError, IndexError) as exc:
        raise ValueError("--device must be auto, cpu, or cuda:N") from exc
    if not request.startswith("cuda:") or index < 0:
        raise ValueError("--device must be auto, cpu, or cuda:N")
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device is unavailable: {request}")
    return True, index


def informer_predictions(data: AggregateInput, horizon: int, seed: int, device: str) -> dict[str, float]:
    """Fit a fresh Informer, keeping the best validation weights in memory."""
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    informer_dir = str(CODE_DIR / "informer")
    if informer_dir not in sys.path:
        sys.path.insert(0, informer_dir)
    from exp.exp_informer import Exp_Informer
    from utils.tools import adjust_learning_rate

    class AggregateExperiment(Exp_Informer):
        def _acquire_device(self):
            # Respect the caller's CUDA_VISIBLE_DEVICES mapping.
            return torch.device(f"cuda:{self.args.gpu}" if self.args.use_gpu else "cpu")

        def train(self, setting=None):
            train_data, train_loader = self._get_data("train")
            validation_data, validation_loader = self._get_data("val")
            test_data, test_loader = self._get_data("test")
            optimizer, criterion = self._select_optimizer(), self._select_criterion()
            best_loss, best_state, stale = math.inf, None, 0
            for epoch in range(self.args.train_epochs):
                self.model.train()
                losses = []
                for batch in train_loader:
                    optimizer.zero_grad()
                    prediction, truth, _ = self._process_one_batch(train_data, *batch)
                    loss = criterion(prediction, truth)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Informer produced a nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                validation_loss = float(self.vali(validation_data, validation_loader, criterion))
                test_loss = float(self.vali(test_data, test_loader, criterion))
                if not math.isfinite(validation_loss):
                    raise RuntimeError("Informer produced a nonfinite validation loss")
                print(f"Informer epoch {epoch + 1}: train={sum(losses) / len(losses):.8g}, "
                      f"validation={validation_loss:.8g}, test={test_loss:.8g}", flush=True)
                if validation_loss <= best_loss:
                    best_loss, stale = validation_loss, 0
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in self.model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= self.args.patience:
                        break
                adjust_learning_rate(optimizer, epoch + 1, self.args)
            if best_state is None:
                raise RuntimeError("Informer did not retain a valid validation state")
            self.model.load_state_dict(best_state)
            return self.model

    seed_all(seed, torch)
    use_gpu, gpu = device_spec(device, torch)
    window = window_plan(len(data), horizon)
    settings = dict(INFORMER_SETTINGS)
    settings.update(root_path=str(data.path.parent), data_path=data.path.name,
                    cols=list(RAW_COLUMNS[1:]), seq_len=window["lags"],
                    label_len=window["label_len"], pred_len=1, lead=horizon,
                    test_batch_size=window["test_batch_size"],
                    calendar_path=str(data.path), enc_in=15, dec_in=15, c_out=1,
                    use_gpu=use_gpu, gpu=gpu)
    experiment = AggregateExperiment(SimpleNamespace(**settings))
    experiment.train()
    test_data, test_loader = experiment._get_data("test")
    pairs = []
    experiment.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            prediction, _, target_dates = experiment._process_one_batch(test_data, *batch)
            days = np.asarray(target_dates).reshape(-1).astype("datetime64[D]").astype(str)
            pairs.extend(zip(days.tolist(), prediction.detach().cpu().numpy().reshape(-1).tolist()))
    result = predictions_by_date(pairs)
    if set(result) != set(evaluation_dates(data, horizon)):
        raise RuntimeError("Informer returned dates inconsistent with its test-window plan")
    return result


def comparison_data(data: AggregateInput, horizon: int):
    """Scale on training rows and build 15-channel scalar-target samples."""
    import numpy as np
    raw = np.asarray(data.values, dtype=float)
    window = window_plan(len(data), horizon)
    training = raw[:window["train_rows"]]
    minimum = training.min(axis=0)
    span = training.max(axis=0) - minimum
    if np.any(span == 0):
        raise ValueError("Every training channel must vary for MinMax scaling")
    normalized = ((raw - minimum) / span).astype(np.float32)
    lags = window["lags"]
    train_indices = np.arange(lags + horizon - 1, window["train_rows"], dtype=int)
    test_indices = np.arange(window["first_target_index"], window["last_target_index"] + 1, dtype=int)
    indices = np.concatenate([train_indices, test_indices])
    sequences = np.stack([normalized[i - horizon - lags + 1:i - horizon + 1] for i in indices])
    targets = normalized[indices, :1]
    split = len(train_indices)
    return (normalized, sequences[:split], targets[:split], sequences[split:],
            test_indices, float(minimum[0]), float(span[0]))


def gradient_boosting_predictions(data: AggregateInput, horizon: int, seed: int,
                                  device: str) -> dict[str, float]:
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    seed_all(seed)
    _, x_train, y_train, x_test, indices, minimum, span = comparison_data(data, horizon)
    estimator = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0,
        loss="squared_error", random_state=seed,
    )
    estimator.fit(x_train.reshape(len(x_train), -1), y_train[:, 0])
    predictions = np.asarray(estimator.predict(x_test.reshape(len(x_test), -1))).reshape(-1) * span + minimum
    return predictions_by_date(zip((data.dates[int(i)] for i in indices), predictions))


def patchtst_model(lags: int, torch, nn):
    """Build a shared patch encoder with a joint head over 15 channels."""
    settings = PATCHTST_SETTINGS
    channel_count, target_index = len(RAW_COLUMNS) - 1, 0
    if channel_count - 1 != settings["expected_historical_covariates"]:
        raise ValueError("PatchTST requires the target and all 14 historical predictors")
    patch_len = max(1, min(settings["patch_len"], lags))
    stride = max(1, min(settings["stride"], lags))
    patch_count = 1 + (lags - patch_len) // stride
    hidden_size, n_heads = settings["hidden_size"], settings["n_heads"]

    class CovariatePatchTST(nn.Module):
        def __init__(self):
            super().__init__()
            self.target_index = target_index
            self.use_revin = settings["revin"]
            self.revin_affine = settings["revin_affine"]
            self.subtract_last = settings["revin_subtract_last"]
            if self.revin_affine:
                self.revin_weight = nn.Parameter(torch.ones(1, channel_count, 1))
                self.revin_bias = nn.Parameter(torch.zeros(1, channel_count, 1))
            self.patch_embedding = nn.Linear(patch_len, hidden_size)
            self.position_embedding = nn.Parameter(torch.zeros(1, 1, patch_count, hidden_size))
            self.channel_embedding = nn.Parameter(torch.zeros(1, channel_count, 1, hidden_size))
            self.input_dropout = nn.Dropout(settings["fc_dropout"])
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=n_heads,
                dim_feedforward=settings["linear_hidden_size"],
                dropout=max(settings["dropout"], settings["attn_dropout"]),
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=settings["encoder_layers"])
            self.head = nn.Sequential(
                nn.Dropout(settings["head_dropout"]),
                nn.Linear(channel_count * patch_count * hidden_size, 1),
            )
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.channel_embedding, mean=0.0, std=0.02)

        def forward(self, inputs):
            channels = inputs.transpose(1, 2)
            if self.use_revin:
                center = channels[:, :, -1:] if self.subtract_last else channels.mean(dim=2, keepdim=True)
                scale = torch.sqrt(channels.var(dim=2, keepdim=True, unbiased=False) + 1e-5)
                channels = (channels - center) / scale
                if self.revin_affine:
                    channels = channels * self.revin_weight + self.revin_bias
            patches = channels.unfold(dimension=2, size=patch_len, step=stride)
            tokens = self.patch_embedding(patches)
            tokens = self.input_dropout(tokens + self.position_embedding + self.channel_embedding)
            batch_size = tokens.shape[0]
            encoded = self.encoder(tokens.reshape(batch_size * channel_count, patch_count, hidden_size))
            forecast = self.head(encoded.reshape(batch_size, -1))
            if self.use_revin:
                if self.revin_affine:
                    forecast = (forecast - self.revin_bias[:, self.target_index, :]) / (
                        self.revin_weight[:, self.target_index, :] + 1e-8)
                forecast = forecast * scale[:, self.target_index, :] + center[:, self.target_index, :]
            return forecast

    return CovariatePatchTST()


def patchtst_predictions(data: AggregateInput, horizon: int, seed: int,
                         device: str) -> dict[str, float]:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    seed_all(seed, torch, deterministic=True)
    use_gpu, gpu = device_spec(device, torch)
    target_device = torch.device(f"cuda:{gpu}" if use_gpu else "cpu")
    _, x_train, y_train, x_test, indices, minimum, span = comparison_data(data, horizon)
    model = patchtst_model(window_plan(len(data), horizon)["lags"], torch, nn).to(target_device)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=PATCHTST_SETTINGS["windows_batch_size"], shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=PATCHTST_SETTINGS["learning_rate"])
    criterion = nn.L1Loss()
    model.train()
    step = 0
    while step < PATCHTST_SETTINGS["max_steps"]:
        for features, targets in loader:
            features, targets = features.to(target_device), targets.to(target_device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("PatchTST produced a nonfinite training loss")
            loss.backward()
            optimizer.step()
            step += 1
            if step == 1 or step == PATCHTST_SETTINGS["max_steps"] or step % 100 == 0:
                print(f"PatchTST step {step}: loss={float(loss.detach().cpu()):.8g}", flush=True)
            if step >= PATCHTST_SETTINGS["max_steps"]:
                break
    model.eval()
    with torch.no_grad():
        predicted = model(torch.tensor(x_test, dtype=torch.float32, device=target_device)).cpu().numpy()
    predictions = np.asarray(predicted).reshape(-1) * span + minimum
    return predictions_by_date(zip((data.dates[int(i)] for i in indices), predictions))


def simple_predictions(data: AggregateInput, horizon: int) -> dict[str, dict[str, float]]:
    import numpy as np
    window = window_plan(len(data), horizon)
    indices = range(window["first_target_index"], window["last_target_index"] + 1)
    values = np.asarray(data.values, dtype=float)[:, 0]
    training_values = values[:window["train_rows"]]
    design = np.column_stack([np.ones(len(training_values) - 1), training_values[:-1]])
    intercept, phi = np.linalg.lstsq(design, training_values[1:], rcond=None)[0]
    persistence, ar1 = [], []
    for target in indices:
        last = float(values[target - horizon])
        persistence.append((data.dates[target], last))
        for _ in range(horizon):
            last = float(intercept + phi * last)
        ar1.append((data.dates[target], last))
    return {"Persistence_pred": predictions_by_date(persistence),
            "AR1_pred": predictions_by_date(ar1)}


def group_rows(data: AggregateInput, horizon: int, group: int, seed: int,
               device: str) -> list[dict]:
    import numpy as np
    predictions = {
        "baseline_pred": informer_predictions(data, horizon, seed, device),
        "GradientBoostingRegressor_pred": gradient_boosting_predictions(data, horizon, seed, device),
        "PatchTST_pred": patchtst_predictions(data, horizon, seed, device),
        **simple_predictions(data, horizon),
    }
    # Store target values using the direct loader's float32 representation.
    truth = {day: float(np.float32(row[0])) for day, row in zip(data.dates, data.values)}
    rows = []
    for day in evaluation_dates(data, horizon):
        row = {"date": day, "TRUE": truth[day]}
        for column in MODEL_COLUMNS:
            if day not in predictions[column] or not math.isfinite(predictions[column][day]):
                raise RuntimeError(f"Missing or nonfinite {column} prediction for {day}")
            row[column] = predictions[column][day]
        row.update(group_id=group, seed=seed, nominal_horizon=horizon)
        rows.append(row)
    return rows


def output_path(output_dir: Path, horizon: int, group_count: int) -> Path:
    return output_dir.resolve() / f"aggregate_predictions_h{horizon}_{group_count}groups.csv"


def describe_plan(data: AggregateInput, plan: dict, args) -> dict:
    return {
        "input": str(data.path), "sha256": data.sha256, "observations": len(data),
        "columns": list(RAW_COLUMNS), "first_date": data.dates[0], "last_date": data.dates[-1],
        "schema": list(OUTPUT_COLUMNS), "models": list(MODEL_COLUMNS),
        "historical_channels": 15, "historical_predictors": 14,
        "sentiment_columns": ["attention_post_count", "sentiment_direction"],
        "forecast_mode": "direct_terminal_target", "scaler_fit": "first floor(0.7N) rows only",
        "calendar": "input date column; each lead counts one observed trading date",
        "truth_dtype": "float32 represented as float64 in CSV",
        "seed_source": "explicit CLI seeds" if args.seed is not None or args.seeds is not None
                       else str(args.seed_file.resolve()),
        "informer": INFORMER_SETTINGS, "patchtst": PATCHTST_SETTINGS,
        "runs": [
            {**window_plan(len(data), horizon), "first_evaluation_date": evaluation_dates(data, horizon)[0],
             "last_evaluation_date": evaluation_dates(data, horizon)[-1],
             "groups": [{"group": group, "seed": seed} for group, seed in groups],
             "output": str(output_path(args.output_dir, horizon, len(groups)))}
            for horizon, groups in plan.items()
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--horizon", action="append", type=parse_horizon,
                        help="h1 or h5; repeat to select both (default: both)")
    parser.add_argument("--group", action="append", type=int,
                        help="Select a group; repeat for multiple groups")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="One seed for one group at each selected horizon")
    seeds.add_argument("--seeds", type=int, nargs="+", help="One seed per selected group")
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing final output CSVs")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and print the plan without training or writing")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = read_input(args.input)
        plan = training_plan(args)
        description = describe_plan(data, plan, args)
        if args.dry_run:
            print(json.dumps(description, indent=2))
            return 0
        missing = [module for module in ("numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib")
                   if importlib.util.find_spec(module) is None]
        if missing:
            raise RuntimeError(f"Missing training dependencies: {', '.join(missing)}. "
                               "Install code/requirements-aggregate-training.txt")
        for horizon, groups in plan.items():
            destination = output_path(args.output_dir, horizon, len(groups))
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {destination}; use --overwrite to replace it")
        import torch
        device_spec(args.device, torch)
        for horizon, groups in plan.items():
            rows = []
            for group, seed in groups:
                print(f"Training h{horizon}, group {group}, seed {seed}", flush=True)
                rows.extend(group_rows(data, horizon, group, seed, args.device))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            destination = output_path(args.output_dir, horizon, len(groups))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {len(rows)} rows: {destination}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
