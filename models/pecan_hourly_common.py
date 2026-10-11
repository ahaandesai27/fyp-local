"""Shared hourly Pecan Street data pipeline and result format for model comparisons."""

import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SEED = 42
WINDOW_HOURS = 24
HORIZON_HOURS = 1
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
FEATURES = ["grid", "air1", "furnace1", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
RAW_FEATURES = ["grid", "air1", "furnace1"]
STEP = pd.Timedelta(hours=1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_hourly_homes(csv_path, min_coverage=0.90):
    """Read 15-minute Austin data and aggregate interval-average power hourly."""
    usecols = ["dataid", "local_15min", *[f for f in RAW_FEATURES if f in pd.read_csv(csv_path, nrows=0).columns]]
    missing = {"dataid", "local_15min", "grid"} - set(usecols)
    if missing:
        raise ValueError(f"Pecan CSV is missing required columns: {sorted(missing)}")
    raw = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    raw["local_15min"] = pd.to_datetime(raw["local_15min"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["dataid", "local_15min"]).sort_values(["dataid", "local_15min"])
    for col in RAW_FEATURES:
        if col not in raw:
            raw[col] = np.nan
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in ("air1", "furnace1"):
        raw.loc[raw[col] < 0, col] = np.nan

    homes = []
    print(f"Loaded {len(raw):,} quarter-hour rows for {raw['dataid'].nunique()} homes")
    for dataid, home in raw.groupby("dataid", sort=True):
        home = home.drop_duplicates("local_15min").set_index("local_15min").sort_index()
        coverage = {c: float(home[c].notna().mean()) for c in RAW_FEATURES}
        if coverage["grid"] < min_coverage:
            print(f"Skipping home {dataid}: grid coverage {coverage['grid']:.1%}")
            continue

        # The source values are 15-minute average power readings. A mean of
        # four quarter-hour values is the average power for the hourly period.
        # Offset bins so quarter-hour timestamps (:15/:30/:45/:00) stay in the
        # same four-reading hour instead of splitting at the UTC hour boundary.
        hourly = home[RAW_FEATURES].resample("1h", origin="start_day", offset="15min").mean()
        counts = home["grid"].resample("1h", origin="start_day", offset="15min").count()
        hourly.loc[counts < 3, "grid"] = np.nan

        # Bound clearly corrupted appliance-channel spikes from train only.
        n = len(hourly)
        train_end = int(n * TRAIN_FRAC)
        for col in ("air1", "furnace1"):
            train_values = hourly[col].iloc[:train_end].dropna()
            if len(train_values):
                lo, hi = train_values.quantile([0.001, 0.999]).to_numpy()
                hourly.loc[(hourly[col] < lo) | (hourly[col] > hi), col] = np.nan

        local = hourly.index.tz_convert("America/Chicago")
        hour = local.hour + local.minute / 60.0
        dow = local.dayofweek
        hourly["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        hourly["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        hourly["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        hourly["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        hourly.index.name = "timestamp_utc"
        hourly = hourly.reset_index()

        if len(hourly) < WINDOW_HOURS + HORIZON_HOURS + 48:
            print(f"Skipping home {dataid}: only {len(hourly)} hourly rows")
            continue
        homes.append((int(dataid), hourly))
        print(f"Home {dataid}: {len(hourly):,} hourly rows | "
              f"grid={coverage['grid']:.1%}, air1={coverage['air1']:.1%}, furnace1={coverage['furnace1']:.1%}")
    if not homes:
        raise ValueError("No homes meet the minimum coverage/length requirements.")
    return homes


def split_homes(homes):
    train, val, test = [], [], []
    for dataid, frame in homes:
        n = len(frame)
        train_end = int(n * TRAIN_FRAC)
        val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
        parts = [frame.iloc[:train_end].copy(), frame.iloc[train_end:val_end].copy(), frame.iloc[val_end:].copy()]
        for part in parts:
            for col in ("air1", "furnace1"):
                part[col] = part[col].interpolate(limit=2, limit_direction="both").fillna(0.0)
        train.append((dataid, parts[0]))
        val.append((dataid, parts[1]))
        test.append((dataid, parts[2]))
    return train, val, test


def verify_metadata(metadata_path, homes):
    if metadata_path is None:
        return
    metadata = pd.read_csv(metadata_path, skiprows=[1], usecols=["dataid", "city", "state"], low_memory=False)
    homes_table = pd.DataFrame({"dataid": [dataid for dataid, _ in homes]})
    joined = homes_table.merge(metadata, on="dataid", how="left")
    missing = joined["city"].isna().sum()
    is_austin = joined["city"].fillna("").str.lower().eq("austin")
    is_texas = joined["state"].fillna("").str.lower().isin(["texas", "tx"])
    non_austin = joined[~(is_austin & is_texas)]
    if missing:
        print(f"Metadata note: {missing} of {len(joined)} home IDs were not matched.")
    if len(non_austin):
        raise ValueError(f"Metadata says {len(non_austin)} data homes are not Austin, Texas.")
    print(f"Metadata verified: {len(joined)} matched homes are in Austin, Texas.")


class HourlyWindowDataset(Dataset):
    def __init__(self, homes, feature_scaler, delta_scaler, home_to_idx):
        xs, ys, lasts, ids, times = [], [], [], [], []
        for dataid, frame in homes:
            values = feature_scaler.transform(frame[FEATURES]).astype(np.float32)
            grid = frame["grid"].to_numpy(dtype=np.float32)
            timestamps = pd.DatetimeIndex(frame["timestamp_utc"])
            for target_i in range(WINDOW_HOURS, len(frame)):
                start = target_i - WINDOW_HOURS
                if timestamps[target_i] - timestamps[start] != STEP * WINDOW_HOURS:
                    continue
                if not np.isfinite(grid[start:target_i + 1]).all():
                    continue
                xs.append(values[start:target_i])
                ys.append(grid[target_i] - grid[target_i - 1])
                lasts.append(grid[target_i - 1])
                ids.append(home_to_idx[dataid])
                times.append(timestamps[target_i])
        if not xs:
            raise ValueError("No complete hourly windows were available in a split.")
        self.x = np.asarray(xs, dtype=np.float32)
        self.y = delta_scaler.transform(np.asarray(ys, dtype=np.float32).reshape(-1, 1)).astype(np.float32).ravel()
        self.last = np.asarray(lasts, dtype=np.float32)
        self.home_idx = np.asarray(ids, dtype=np.int64)
        self.dataid = np.asarray([list(home_to_idx.keys())[i] for i in self.home_idx], dtype=np.int64)
        self.timestamps = np.asarray(times)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (torch.from_numpy(self.x[index]), torch.tensor(self.y[index]),
                torch.tensor(self.last[index]), torch.tensor(self.home_idx[index]))


def build_loaders(homes, batch_size, seed=SEED):
    train_homes, val_homes, test_homes = split_homes(homes)
    feature_scaler = StandardScaler()
    delta_scaler = StandardScaler()
    train_rows = pd.concat([frame[FEATURES] for _, frame in train_homes], ignore_index=True)
    feature_scaler.fit(train_rows)

    deltas = []
    for _, frame in train_homes:
        grid = frame["grid"].to_numpy(dtype=np.float64)
        times = pd.DatetimeIndex(frame["timestamp_utc"])
        valid = np.isfinite(grid[:-1]) & np.isfinite(grid[1:])
        valid &= np.asarray((times[1:] - times[:-1]) == STEP)
        deltas.extend((grid[1:][valid] - grid[:-1][valid]).tolist())
    if not deltas:
        raise ValueError("No adjacent observed hourly grid values in training split.")
    delta_scaler.fit(np.asarray(deltas).reshape(-1, 1))

    home_to_idx = {dataid: idx for idx, (dataid, _) in enumerate(homes)}
    datasets = [HourlyWindowDataset(parts, feature_scaler, delta_scaler, home_to_idx)
                for parts in (train_homes, val_homes, test_homes)]
    generator = torch.Generator().manual_seed(seed)
    loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=(i == 0), num_workers=0,
                          generator=generator if i == 0 else None)
               for i, dataset in enumerate(datasets)]
    print(f"Usable windows train/validation/test: {len(datasets[0]):,}/"
          f"{len(datasets[1]):,}/{len(datasets[2]):,}")
    return loaders, datasets, feature_scaler, delta_scaler, home_to_idx


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


@torch.no_grad()
def predict(model, loader, delta_scaler, device):
    model.eval()
    pred_scaled, actual_scaled, lasts, homes, indexes = [], [], [], [], []
    for x, y, last, home_idx in loader:
        pred_scaled.append(model(x.to(device), home_idx.to(device)).cpu().numpy())
        actual_scaled.append(y.numpy())
        lasts.append(last.numpy())
        homes.append(home_idx.numpy())
    pred_scaled = np.concatenate(pred_scaled)
    actual_scaled = np.concatenate(actual_scaled)
    lasts = np.concatenate(lasts)
    actual_delta = delta_scaler.inverse_transform(actual_scaled.reshape(-1, 1)).ravel()
    pred_delta = delta_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return lasts + actual_delta, lasts + pred_delta, lasts


def train_and_save(model_name, model, loaders, datasets, feature_scaler,
                   delta_scaler, home_to_idx, args, device, output_dir):
    train_loader, val_loader, test_loader = loaders
    train_dataset, val_dataset, test_dataset = datasets
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = torch.nn.MSELoss()
    history = []
    best_rmse, best_state, stale = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _, home_idx in train_loader:
            x, y, home_idx = x.to(device), y.to(device), home_idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, home_idx), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_actual, val_pred, _ = predict(model, val_loader, delta_scaler, device)
        val_metrics = calculate_metrics(val_actual, val_pred)
        scheduler.step(val_metrics["RMSE"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_mae": val_metrics["MAE"], "val_rmse": val_metrics["RMSE"],
                        "val_r2": val_metrics["R2"], "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch:02d}/{args.epochs} | Loss {np.mean(losses):.5f} | "
              f"Val MAE {val_metrics['MAE']:.4f} | Val RMSE {val_metrics['RMSE']:.4f} | "
              f"Val R² {val_metrics['R2']:.4f}")
        if val_metrics["RMSE"] < best_rmse - 1e-6:
            best_rmse, stale = val_metrics["RMSE"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    actual, predicted, persistence = predict(model, test_loader, delta_scaler, device)
    result = {
        "model": model_name,
        "dataset": "Pecan Street Austin 15-minute data aggregated to hourly",
        "seed": args.seed,
        "interval": "60 minutes",
        "window": WINDOW_HOURS,
        "horizon": HORIZON_HOURS,
        "features": FEATURES,
        "target": "grid_delta",
        "test": calculate_metrics(actual, predicted),
        "persistence_baseline": calculate_metrics(actual, persistence),
        "best_validation_rmse": best_rmse,
        "homes": len(home_to_idx),
        "window_counts": {"train": len(train_dataset), "validation": len(val_dataset), "test": len(test_dataset)},
        "units": "source grid units (hourly average power)",
        "split": {"train_fraction": TRAIN_FRAC, "validation_fraction": VAL_FRAC, "chronological_per_home": True},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "feature_scaler": feature_scaler,
                "delta_scaler": delta_scaler, "home_to_idx": home_to_idx,
                "features": FEATURES, "window": WINDOW_HOURS, "horizon": HORIZON_HOURS,
                "model": model_name}, output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    ids = test_dataset.dataid
    timestamps = test_dataset.timestamps
    pd.DataFrame({"dataid": ids, "timestamp": timestamps, "actual": actual,
                  "prediction": predicted, "persistence": persistence,
                  "residual": predicted - actual}).to_csv(output_dir / "predictions.csv", index=False)

    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Train loss")
    ax.set(xlabel="Epoch", ylabel="MSE loss", title=f"{model_name} training loss")
    ax.legend(); fig.tight_layout(); fig.savefig(plots / "training_loss.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([r["epoch"] for r in history], [r["val_rmse"] for r in history], label="Validation RMSE")
    ax.set(xlabel="Epoch", ylabel="RMSE (source grid units)", title=f"{model_name} validation RMSE")
    ax.legend(); fig.tight_layout(); fig.savefig(plots / "validation_rmse.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([r["epoch"] for r in history], [r["val_mae"] for r in history], label="Validation MAE")
    ax.set(xlabel="Epoch", ylabel="MAE (source grid units)", title=f"{model_name} validation MAE")
    ax.legend(); fig.tight_layout(); fig.savefig(plots / "validation_mae.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    n = min(1000, len(actual))
    ax.plot(actual[:n], label="Actual"); ax.plot(predicted[:n], label="Prediction"); ax.plot(persistence[:n], label="Persistence", alpha=0.7)
    ax.set(xlabel="Test window", ylabel="Grid hourly average", title=f"{model_name} test actual vs predicted")
    ax.legend(); fig.tight_layout(); fig.savefig(plots / "test_actual_vs_pred.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(predicted - actual, bins=80)
    ax.axvline(0, linestyle="--", color="black")
    ax.set(xlabel="Prediction error", ylabel="Count", title=f"{model_name} test residuals")
    fig.tight_layout(); fig.savefig(plots / "test_residuals.png", dpi=150); plt.close(fig)
    print(f"Test: {json.dumps(result['test'])}")
    print(f"Persistence: {json.dumps(result['persistence_baseline'])}")
    print(f"Saved: {output_dir}")
    return result


def add_common_arguments(parser):
    parser.add_argument("csv", type=Path, help="15-minute Pecan Street Austin CSV")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional Pecan metadata.csv for verification")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=SEED)
