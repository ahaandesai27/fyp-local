#!/usr/bin/env python3
"""
Improved SNN load-forecasting script for the Pecan Street dataset.

Changes vs. the baseline:

1. Delta/residual target -- the network predicts the CHANGE in grid load
   for the next 15-minute step, not its absolute value. The last observed
   grid reading is added back at inference time. The baseline's readout
   had to hold an absolute level in its membrane potential across the
   whole 96-step window, which acts like a low-pass filter and is exactly
   why the AC on/off square wave got smoothed out (predicted ~2 where
   actual was ~4). Predicting the increment removes that burden.

2. Learnable membrane decay (learn_beta=True) -- one fixed beta=0.90 for
   every layer forces the same time constant onto slow trends and fast
   transients. Letting each layer learn its own beta lets it specialize.

3. Per-home embedding -- homes are pooled into one dataset/scaler but have
   different appliance baselines. A small learned embedding, concatenated
   to the input at every timestep, lets the network tell homes apart.

4. Magnitude-weighted Huber loss -- large transitions (compressor turning
   on/off) are rare compared to small near-zero deltas. Plain Huber loss
   is dominated by the common case and nudges predictions toward "no
   change". Weighting each sample by its target magnitude keeps peak
   transitions from being drowned out.

5. Gap-aware windowing -- after interpolation, remaining NaN "grid" rows
   are still dropped, which can silently create a time gap inside a
   home's series. A window that spans such a gap isn't really "24 hours
   of history". Windows are now checked against the expected elapsed time
   and skipped if they cross a gap.

6. Early stopping on validation RMSE, computed in reconstructed absolute
   grid units, instead of a fixed 30 epochs.

Usage: python train_snn_v2.py path/to/data.csv [--epochs 40] [--patience 6]
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import snntorch as snn


# ============================================================
# CONFIG
# ============================================================

SEED = 42

WINDOW = 96          # 24 hours at 15-minute resolution
HORIZON = 1          # predict next 15-minute interval
STEP_SECONDS = 15 * 60

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

BATCH_SIZE = 256
EPOCHS = 40
PATIENCE = 6         # early stopping, in epochs with no val-RMSE improvement
LR = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN1 = 64
HIDDEN2 = 32
BETA_INIT = 0.90
DROPOUT = 0.15

HOME_EMBED_DIM = 4

# Weighted Huber: weight = 1 + PEAK_WEIGHT_ALPHA * |scaled target delta|
PEAK_WEIGHT_ALPHA = 2.0

NUM_WORKERS = 4

FEATURES = [
    "grid",
    "air1",
    "furnace1",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

TARGET = "grid"

MIN_COVERAGE = 0.90

RESULTS_DIR = Path("results/snn_v2")
PLOTS_DIR = RESULTS_DIR / "plots"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATA CLEANING  (unchanged from baseline)
# ============================================================

def clean_home(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["local_15min"] = pd.to_datetime(df["local_15min"], errors="coerce", utc=True)

    df = df.dropna(subset=["local_15min"])
    df = df.sort_values("local_15min")
    df = df.drop_duplicates("local_15min")

    for col in ["grid", "air1", "furnace1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Appliance measurements should not be negative. Grid is allowed to be.
    for col in ["air1", "furnace1"]:
        df.loc[df[col] < 0, col] = np.nan

    local_time = df["local_15min"].dt.tz_convert("America/Chicago")

    hour = local_time.dt.hour + local_time.dt.minute / 60.0
    dow = local_time.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return df


def coverage(df: pd.DataFrame, col: str) -> float:
    return df[col].notna().mean()


# ============================================================
# SELECT HOMES  (unchanged from baseline)
# ============================================================

def prepare_homes(raw: pd.DataFrame):

    homes = []

    for dataid, home in raw.groupby("dataid"):

        home = clean_home(home)

        if len(home) < WINDOW + HORIZON + 100:
            continue
        if coverage(home, "air1") < MIN_COVERAGE:
            continue
        if coverage(home, "furnace1") < MIN_COVERAGE:
            continue
        if coverage(home, "grid") < MIN_COVERAGE:
            continue

        homes.append((dataid, home))

    print(f"\nHomes retained: {len(homes)}")

    for dataid, home in homes:
        print(
            f"  {dataid}: {len(home):,} rows | "
            f"air1={coverage(home, 'air1'):.1%} | "
            f"furnace1={coverage(home, 'furnace1'):.1%} | "
            f"grid={coverage(home, 'grid'):.1%}"
        )

    return homes


# ============================================================
# TRAIN-ONLY CLEANING PARAMETERS  (unchanged from baseline)
# ============================================================

def get_train_bounds(train_parts):

    combined = pd.concat(train_parts, ignore_index=True)

    bounds = {}
    for col in ["air1", "furnace1"]:
        s = combined[col].dropna()
        bounds[col] = (float(s.quantile(0.001)), float(s.quantile(0.999)))

    return bounds


def apply_cleaning_and_imputation(df, bounds):

    df = df.copy()

    for col, (lo, hi) in bounds.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    for col in ["grid", "air1", "furnace1"]:
        df[col] = df[col].interpolate(method="linear", limit=4, limit_direction="both")

    for col in ["air1", "furnace1"]:
        df[col] = df[col].fillna(0.0)

    df = df.dropna(subset=["grid"])

    return df


# ============================================================
# HOME ID LOOKUP (new)
# ============================================================

def build_home_index(homes):
    return {dataid: idx for idx, (dataid, _) in enumerate(homes)}


# ============================================================
# DATASET  (rewritten: delta target + home embedding index + gap check)
# ============================================================

class WindowDataset(Dataset):

    def __init__(self, homes, feature_scaler, delta_scaler, home_to_idx):

        self.X = []
        self.y = []            # scaled delta target
        self.last_raw = []     # unscaled grid value at the last input step
        self.home_idx = []

        expected_span = pd.Timedelta(seconds=STEP_SECONDS * WINDOW)

        for dataid, df in homes:

            x = df[FEATURES].to_numpy(dtype=np.float32)
            grid_raw = df[TARGET].to_numpy(dtype=np.float32)
            times = df["local_15min"].to_numpy()

            x_scaled = feature_scaler.transform(x)

            n = len(df)
            hidx = home_to_idx[dataid]

            for end in range(WINDOW, n - HORIZON + 1):

                start = end - WINDOW
                target_idx = end + HORIZON - 1

                if target_idx >= n:
                    break

                # Gap check: the window + the target step should span
                # exactly WINDOW * 15 minutes of real elapsed time.
                span = pd.Timestamp(times[target_idx]) - pd.Timestamp(times[start])
                if span != expected_span:
                    continue

                delta = grid_raw[target_idx] - grid_raw[end - 1]

                self.X.append(x_scaled[start:end])
                self.y.append(delta)
                self.last_raw.append(grid_raw[end - 1])
                self.home_idx.append(hidx)

        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = delta_scaler.transform(
            np.asarray(self.y, dtype=np.float32).reshape(-1, 1)
        ).reshape(-1).astype(np.float32)
        self.last_raw = np.asarray(self.last_raw, dtype=np.float32)
        self.home_idx = np.asarray(self.home_idx, dtype=np.int64)

        print(f"Dataset: {len(self.X):,} sequences (gap-skipped windows excluded)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.last_raw[idx], dtype=torch.float32),
            torch.tensor(self.home_idx[idx], dtype=torch.long),
        )


def collect_train_deltas(homes):
    """Raw (unscaled) next-step deltas from the train split, used only to
    fit the delta target scaler."""
    deltas = []
    for _, df in homes:
        grid_raw = df[TARGET].to_numpy(dtype=np.float32)
        deltas.append(grid_raw[1:] - grid_raw[:-1])
    return np.concatenate(deltas).reshape(-1, 1)


# ============================================================
# SNN  (rewritten: learnable beta, dropout, home embedding)
# ============================================================

class SNNRegressor(nn.Module):

    def __init__(
        self,
        input_size,
        num_homes,
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        beta_init=BETA_INIT,
        home_embed_dim=HOME_EMBED_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.home_embedding = nn.Embedding(num_homes, home_embed_dim)

        fused_input_size = input_size + home_embed_dim

        self.fc1 = nn.Linear(fused_input_size, hidden1)
        self.lif1 = snn.Leaky(
            beta=beta_init, learn_beta=True, spike_grad=snn.surrogate.fast_sigmoid()
        )
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(
            beta=beta_init, learn_beta=True, spike_grad=snn.surrogate.fast_sigmoid()
        )
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden2, 1)

        # Non-spiking leaky readout for regression, learnable decay.
        self.readout = snn.Leaky(
            beta=beta_init, learn_beta=True, reset_mechanism="none", output=True
        )

    def forward(self, x, home_idx):

        batch_size = x.size(0)

        home_vec = self.home_embedding(home_idx)                 # (B, E)
        home_vec = home_vec.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, E)
        x = torch.cat([x, home_vec], dim=-1)

        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        mem3 = torch.zeros(batch_size, 1, device=x.device)

        for t in range(x.size(1)):

            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.drop1(spk1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.drop2(spk2)

            cur3 = self.fc3(spk2)
            _, mem3 = self.readout(cur3, mem3)

        # Predicted (scaled) delta for the next step.
        return mem3.squeeze(-1)


# ============================================================
# TRAIN
# ============================================================

def weighted_huber(pred, target, delta=1.0, alpha=PEAK_WEIGHT_ALPHA):
    """Huber loss with per-sample weight that grows with target magnitude,
    so large transitions aren't drowned out by many small near-zero ones."""
    err = pred - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    base = 0.5 * quadratic ** 2 + delta * linear

    weight = 1.0 + alpha * target.abs()
    return (base * weight).mean()


def train_one_epoch(model, loader, optimizer, device):

    model.train()
    total_loss = 0.0

    for X, y, _, home_idx in loader:

        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        home_idx = home_idx.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(X, home_idx)
        loss = weighted_huber(pred, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)


# ============================================================
# EVALUATION  (reconstructs absolute grid values from predicted deltas)
# ============================================================

@torch.no_grad()
def predict(model, loader, delta_scaler, device):

    model.eval()

    pred_deltas = []
    actual_deltas = []
    last_raws = []

    for X, y, last_raw, home_idx in loader:

        X = X.to(device, non_blocking=True)
        home_idx = home_idx.to(device, non_blocking=True)

        pred = model(X, home_idx)

        pred_deltas.append(pred.cpu().numpy())
        actual_deltas.append(y.numpy())
        last_raws.append(last_raw.numpy())

    pred_deltas = np.concatenate(pred_deltas)
    actual_deltas = np.concatenate(actual_deltas)
    last_raws = np.concatenate(last_raws)

    pred_deltas_raw = delta_scaler.inverse_transform(
        pred_deltas.reshape(-1, 1)
    ).reshape(-1)
    actual_deltas_raw = delta_scaler.inverse_transform(
        actual_deltas.reshape(-1, 1)
    ).reshape(-1)

    preds_abs = last_raws + pred_deltas_raw
    actuals_abs = last_raws + actual_deltas_raw

    return actuals_abs, preds_abs


def calculate_metrics(actuals, preds):
    return {
        "MAE": float(mean_absolute_error(actuals, preds)),
        "RMSE": float(math.sqrt(mean_squared_error(actuals, preds))),
        "R2": float(r2_score(actuals, preds)),
    }


# ============================================================
# PLOTS  (unchanged from baseline, aside from labels)
# ============================================================

def save_training_plots(history):

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Huber loss (delta scale)")
    plt.title("SNN v2 Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (W)")
    plt.title("SNN v2 Validation RMSE (absolute grid)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "validation_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (W)")
    plt.title("SNN v2 Validation MAE (absolute grid)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "validation_mae.png", dpi=160)
    plt.close()


def save_test_plots(actuals, preds):

    n = min(1000, len(actuals))

    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:n], label="Actual")
    plt.plot(preds[:n], label="SNN v2 prediction")
    plt.xlabel("Test sample")
    plt.ylabel("Grid load")
    plt.title(f"SNN v2: Actual vs Predicted (first {n} test samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_actual_vs_pred.png", dpi=160)
    plt.close()

    residuals = preds - actuals

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=100)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error (W)")
    plt.ylabel("Count")
    plt.title("SNN v2 Test Residuals")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_residuals.png", dpi=160)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    seed_everything(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\nDevice: {device}")

    print("\nLoading CSV...")
    raw = pd.read_csv(args.csv, low_memory=False)
    print(f"Loaded {len(raw):,} rows")

    homes = prepare_homes(raw)
    if not homes:
        raise RuntimeError("No homes satisfy the coverage requirements.")

    home_to_idx = build_home_index(homes)
    num_homes = len(home_to_idx)

    # --------------------------------------------------------
    # Chronological split per home
    # --------------------------------------------------------

    train_homes, val_homes, test_homes = [], [], []

    for dataid, home in homes:
        n = len(home)
        train_end = int(n * TRAIN_FRAC)
        val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

        train_homes.append((dataid, home.iloc[:train_end].copy()))
        val_homes.append((dataid, home.iloc[train_end:val_end].copy()))
        test_homes.append((dataid, home.iloc[val_end:].copy()))

    # --------------------------------------------------------
    # Train-only bounds + imputation
    # --------------------------------------------------------

    bounds = get_train_bounds([df for _, df in train_homes])

    print("\nTrain-only outlier bounds:")
    for col, (lo, hi) in bounds.items():
        print(f"  {col}: {lo:.4f} -> {hi:.4f}")

    train_homes = [(d, apply_cleaning_and_imputation(df, bounds)) for d, df in train_homes]
    val_homes = [(d, apply_cleaning_and_imputation(df, bounds)) for d, df in val_homes]
    test_homes = [(d, apply_cleaning_and_imputation(df, bounds)) for d, df in test_homes]

    # --------------------------------------------------------
    # Train-only scaling: features as before, target now as DELTA
    # --------------------------------------------------------

    feature_scaler = StandardScaler()
    delta_scaler = StandardScaler()

    train_features = pd.concat([df[FEATURES] for _, df in train_homes], ignore_index=True)
    feature_scaler.fit(train_features)

    train_deltas = collect_train_deltas(train_homes)
    delta_scaler.fit(train_deltas)

    print(
        f"\nTrain delta stats: mean={train_deltas.mean():.4f}, "
        f"std={train_deltas.std():.4f}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print("\nBuilding datasets...")

    train_dataset = WindowDataset(train_homes, feature_scaler, delta_scaler, home_to_idx)
    val_dataset = WindowDataset(val_homes, feature_scaler, delta_scaler, home_to_idx)
    test_dataset = WindowDataset(test_homes, feature_scaler, delta_scaler, home_to_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SNNRegressor(input_size=len(FEATURES), num_homes=num_homes).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # --------------------------------------------------------
    # Training with early stopping
    # --------------------------------------------------------

    history = {"epoch": [], "train_loss": [], "val_mae": [], "val_rmse": [], "val_r2": [], "lr": []}

    best_val = float("inf")
    best_state = None
    epochs_since_improvement = 0

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_actuals, val_preds = predict(model, val_loader, delta_scaler, device)
        val_metrics = calculate_metrics(val_actuals, val_preds)

        scheduler.step(val_metrics["RMSE"])
        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_metrics["MAE"])
        history["val_rmse"].append(val_metrics["RMSE"])
        history["val_r2"].append(val_metrics["R2"])
        history["lr"].append(lr)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | Loss {train_loss:.5f} | "
            f"Val MAE {val_metrics['MAE']:.4f} | Val RMSE {val_metrics['RMSE']:.4f} | "
            f"Val R² {val_metrics['R2']:.4f} | LR {lr:.2e}"
        )

        if val_metrics["RMSE"] < best_val - 1e-5:
            best_val = val_metrics["RMSE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_actuals, test_preds = predict(model, test_loader, delta_scaler, device)
    test_metrics = calculate_metrics(test_actuals, test_preds)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"MAE  : {test_metrics['MAE']:.4f}")
    print(f"RMSE : {test_metrics['RMSE']:.4f}")
    print(f"R²   : {test_metrics['R2']:.4f}")

    # --------------------------------------------------------
    # Save model, metrics, history, predictions, plots
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_scaler": feature_scaler,
            "delta_scaler": delta_scaler,
            "home_to_idx": home_to_idx,
            "features": FEATURES,
            "target": TARGET,
            "window": WINDOW,
            "horizon": HORIZON,
        },
        RESULTS_DIR / "model.pt",
    )

    metrics = {
        "model": "SNN_v2_delta",
        "seed": SEED,
        "window": WINDOW,
        "horizon": HORIZON,
        "features": FEATURES,
        "target": f"{TARGET}_delta",
        "test": test_metrics,
        "best_validation_rmse": best_val,
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)

    pd.DataFrame(
        {"actual": test_actuals, "prediction": test_preds, "residual": test_preds - test_actuals}
    ).to_csv(RESULTS_DIR / "predictions.csv", index=False)

    save_training_plots(history)
    save_test_plots(test_actuals, test_preds)

    print("\nSaved results to:")
    print(f"  {RESULTS_DIR}/")
    print("\nPlots:")
    print(f"  {PLOTS_DIR}/training_loss.png")
    print(f"  {PLOTS_DIR}/validation_rmse.png")
    print(f"  {PLOTS_DIR}/validation_mae.png")
    print(f"  {PLOTS_DIR}/test_actual_vs_pred.png")
    print(f"  {PLOTS_DIR}/test_residuals.png")


if __name__ == "__main__":
    main()