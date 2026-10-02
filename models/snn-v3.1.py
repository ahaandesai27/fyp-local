#!/usr/bin/env python3

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

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN1 = 64
HIDDEN2 = 32
BETA = 0.90

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

RESULTS_DIR = Path("results/snn")
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
# DATA CLEANING
# ============================================================

def clean_home(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["local_15min"] = pd.to_datetime(
        df["local_15min"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(subset=["local_15min"])
    df = df.sort_values("local_15min")
    df = df.drop_duplicates("local_15min")

    for col in ["grid", "air1", "furnace1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Appliance measurements should not be negative.
    # Grid is intentionally allowed to be negative.
    for col in ["air1", "furnace1"]:
        df.loc[df[col] < 0, col] = np.nan

    # Convert UTC timestamps back to Austin local time for
    # calendar features.
    local_time = df["local_15min"].dt.tz_convert("America/Chicago")

    hour = (
        local_time.dt.hour
        + local_time.dt.minute / 60.0
    )

    dow = local_time.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return df


def coverage(df: pd.DataFrame, col: str) -> float:
    return df[col].notna().mean()


# ============================================================
# SELECT HOMES
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
            f"  {dataid}: "
            f"{len(home):,} rows | "
            f"air1={coverage(home, 'air1'):.1%} | "
            f"furnace1={coverage(home, 'furnace1'):.1%} | "
            f"grid={coverage(home, 'grid'):.1%}"
        )

    return homes


# ============================================================
# TRAIN-ONLY CLEANING PARAMETERS
# ============================================================

def get_train_bounds(train_parts):

    combined = pd.concat(
        train_parts,
        ignore_index=True,
    )

    bounds = {}

    for col in ["air1", "furnace1"]:

        s = combined[col].dropna()

        bounds[col] = (
            float(s.quantile(0.001)),
            float(s.quantile(0.999)),
        )

    return bounds


def apply_cleaning_and_imputation(df, bounds):

    df = df.copy()

    # Train-derived outlier bounds.
    for col, (lo, hi) in bounds.items():

        df.loc[
            (df[col] < lo) | (df[col] > hi),
            col,
        ] = np.nan

    # Interpolate short gaps.
    for col in ["grid", "air1", "furnace1"]:

        df[col] = (
            df[col]
            .interpolate(
                method="linear",
                limit=4,
                limit_direction="both",
            )
        )

    # Remaining appliance gaps -> zero.
    # Homes were already required to have >=90% coverage.
    for col in ["air1", "furnace1"]:
        df[col] = df[col].fillna(0.0)

    # Target cannot be fabricated.
    df = df.dropna(subset=["grid"])

    return df


# ============================================================
# DATASET
# ============================================================

class WindowDataset(Dataset):

    def __init__(
        self,
        homes,
        feature_scaler,
        target_scaler,
    ):

        self.X = []
        self.y = []

        for _, df in homes:

            x = df[FEATURES].to_numpy(
                dtype=np.float32
            )

            y = df[[TARGET]].to_numpy(
                dtype=np.float32
            )

            x = feature_scaler.transform(x)

            y = target_scaler.transform(y).reshape(-1)

            n = len(df)

            for end in range(
                WINDOW,
                n - HORIZON + 1,
            ):

                start = end - WINDOW
                target_idx = end + HORIZON - 1

                if target_idx >= n:
                    break

                self.X.append(
                    x[start:end]
                )

                self.y.append(
                    y[target_idx]
                )

        self.X = np.asarray(
            self.X,
            dtype=np.float32,
        )

        self.y = np.asarray(
            self.y,
            dtype=np.float32,
        )

        print(
            f"Dataset: {len(self.X):,} sequences"
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(
                self.y[idx],
                dtype=torch.float32,
            ),
        )


# ============================================================
# SNN
# ============================================================

class SNNRegressor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        beta=BETA,
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden1,
        )

        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=snn.surrogate.fast_sigmoid(),
        )

        self.fc2 = nn.Linear(
            hidden1,
            hidden2,
        )

        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=snn.surrogate.fast_sigmoid(),
        )

        self.fc3 = nn.Linear(
            hidden2,
            1,
        )

        # Non-spiking leaky readout for regression.
        self.readout = snn.Leaky(
            beta=beta,
            reset_mechanism="none",
            output=True,
        )

    def forward(self, x):

        batch_size = x.size(0)

        mem1 = torch.zeros(
            batch_size,
            self.fc1.out_features,
            device=x.device,
        )

        mem2 = torch.zeros(
            batch_size,
            self.fc2.out_features,
            device=x.device,
        )

        mem3 = torch.zeros(
            batch_size,
            1,
            device=x.device,
        )

        outputs = []

        for t in range(x.size(1)):

            cur1 = self.fc1(
                x[:, t, :]
            )

            spk1, mem1 = self.lif1(
                cur1,
                mem1,
            )

            cur2 = self.fc2(spk1)

            spk2, mem2 = self.lif2(
                cur2,
                mem2,
            )

            cur3 = self.fc3(spk2)

            _, mem3 = self.readout(
                cur3,
                mem3,
            )

            outputs.append(mem3)

        return outputs[-1].squeeze(-1)


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    for X, y in loader:

        X = X.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        pred = model(X)

        loss = criterion(
            pred,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item() * len(X)
        )

    return total_loss / len(loader.dataset)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    target_scaler,
    device,
):

    model.eval()

    preds = []
    actuals = []

    for X, y in loader:

        X = X.to(
            device,
            non_blocking=True,
        )

        pred = model(X)

        preds.append(
            pred.cpu().numpy()
        )

        actuals.append(
            y.numpy()
        )

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    preds = target_scaler.inverse_transform(
        preds.reshape(-1, 1)
    ).reshape(-1)

    actuals = target_scaler.inverse_transform(
        actuals.reshape(-1, 1)
    ).reshape(-1)

    return actuals, preds


def calculate_metrics(actuals, preds):

    return {
        "MAE": float(
            mean_absolute_error(
                actuals,
                preds,
            )
        ),
        "RMSE": float(
            math.sqrt(
                mean_squared_error(
                    actuals,
                    preds,
                )
            )
        ),
        "R2": float(
            r2_score(
                actuals,
                preds,
            )
        ),
    }


# ============================================================
# PLOTS
# ============================================================

def save_training_plots(history):

    # Training loss.
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title("SNN Training Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "training_loss.png",
        dpi=160,
    )

    plt.close()

    # Validation RMSE.
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["epoch"],
        history["val_rmse"],
        label="Validation RMSE",
    )

    plt.xlabel("Epoch")
    plt.ylabel("RMSE (W)")
    plt.title("SNN Validation RMSE")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "validation_rmse.png",
        dpi=160,
    )

    plt.close()

    # Validation MAE.
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["epoch"],
        history["val_mae"],
        label="Validation MAE",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MAE (W)")
    plt.title("SNN Validation MAE")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "validation_mae.png",
        dpi=160,
    )

    plt.close()


def save_test_plots(actuals, preds):

    # Actual vs predicted.
    plt.figure(figsize=(12, 6))

    # Plot only the first 1000 test samples so the plot remains readable.
    n = min(1000, len(actuals))

    plt.plot(
        actuals[:n],
        label="Actual",
    )

    plt.plot(
        preds[:n],
        label="SNN prediction",
    )

    plt.xlabel("Test sample")
    plt.ylabel("Grid load")
    plt.title(
        f"SNN: Actual vs Predicted "
        f"(first {n} test samples)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "test_actual_vs_pred.png",
        dpi=160,
    )

    plt.close()

    # Residuals.
    residuals = preds - actuals

    plt.figure(figsize=(10, 5))

    plt.hist(
        residuals,
        bins=100,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel("Prediction error (W)")
    plt.ylabel("Count")
    plt.title("SNN Test Residuals")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "test_residuals.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        type=str,
    )

    args = parser.parse_args()

    seed_everything(SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading CSV...")

    raw = pd.read_csv(
        args.csv,
        low_memory=False,
    )

    print(
        f"Loaded {len(raw):,} rows"
    )

    # --------------------------------------------------------
    # Homes
    # --------------------------------------------------------

    homes = prepare_homes(raw)

    if not homes:
        raise RuntimeError(
            "No homes satisfy the coverage requirements."
        )

    # --------------------------------------------------------
    # Chronological split per home
    # --------------------------------------------------------

    train_homes = []
    val_homes = []
    test_homes = []

    for dataid, home in homes:

        n = len(home)

        train_end = int(
            n * TRAIN_FRAC
        )

        val_end = int(
            n * (TRAIN_FRAC + VAL_FRAC)
        )

        train_homes.append(
            (
                dataid,
                home.iloc[:train_end].copy(),
            )
        )

        val_homes.append(
            (
                dataid,
                home.iloc[train_end:val_end].copy(),
            )
        )

        test_homes.append(
            (
                dataid,
                home.iloc[val_end:].copy(),
            )
        )

    # --------------------------------------------------------
    # Train-only bounds
    # --------------------------------------------------------

    bounds = get_train_bounds(
        [df for _, df in train_homes]
    )

    print("\nTrain-only outlier bounds:")

    for col, (lo, hi) in bounds.items():

        print(
            f"  {col}: "
            f"{lo:.4f} -> {hi:.4f}"
        )

    train_homes = [
        (
            dataid,
            apply_cleaning_and_imputation(
                df,
                bounds,
            ),
        )
        for dataid, df in train_homes
    ]

    val_homes = [
        (
            dataid,
            apply_cleaning_and_imputation(
                df,
                bounds,
            ),
        )
        for dataid, df in val_homes
    ]

    test_homes = [
        (
            dataid,
            apply_cleaning_and_imputation(
                df,
                bounds,
            ),
        )
        for dataid, df in test_homes
    ]

    # --------------------------------------------------------
    # Train-only scaling
    # --------------------------------------------------------

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_features = pd.concat(
        [
            df[FEATURES]
            for _, df in train_homes
        ],
        ignore_index=True,
    )

    train_target = pd.concat(
        [
            df[[TARGET]]
            for _, df in train_homes
        ],
        ignore_index=True,
    )

    feature_scaler.fit(
        train_features
    )

    target_scaler.fit(
        train_target
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print("\nBuilding datasets...")

    train_dataset = WindowDataset(
        train_homes,
        feature_scaler,
        target_scaler,
    )

    val_dataset = WindowDataset(
        val_homes,
        feature_scaler,
        target_scaler,
    )

    test_dataset = WindowDataset(
        test_homes,
        feature_scaler,
        target_scaler,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SNNRegressor(
        input_size=len(FEATURES),
    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: {params:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.HuberLoss(
        delta=1.0
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = {
        "epoch": [],
        "train_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
        "lr": [],
    }

    best_val = float("inf")
    best_state = None

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_actuals, val_preds = predict(
            model,
            val_loader,
            target_scaler,
            device,
        )

        val_metrics = calculate_metrics(
            val_actuals,
            val_preds,
        )

        scheduler.step(
            val_metrics["RMSE"]
        )

        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_mae"].append(
            val_metrics["MAE"]
        )
        history["val_rmse"].append(
            val_metrics["RMSE"]
        )
        history["val_r2"].append(
            val_metrics["R2"]
        )
        history["lr"].append(lr)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {train_loss:.5f} | "
            f"Val MAE {val_metrics['MAE']:.4f} | "
            f"Val RMSE {val_metrics['RMSE']:.4f} | "
            f"Val R² {val_metrics['R2']:.4f} | "
            f"LR {lr:.2e}"
        )

        if val_metrics["RMSE"] < best_val:

            best_val = val_metrics["RMSE"]

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_actuals, test_preds = predict(
        model,
        test_loader,
        target_scaler,
        device,
    )

    test_metrics = calculate_metrics(
        test_actuals,
        test_preds,
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"MAE  : {test_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {test_metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {test_metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "features": FEATURES,
            "target": TARGET,
            "window": WINDOW,
            "horizon": HORIZON,
        },
        RESULTS_DIR / "model.pt",
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "model": "SNN",
        "seed": SEED,
        "window": WINDOW,
        "horizon": HORIZON,
        "features": FEATURES,
        "target": TARGET,
        "test": test_metrics,
        "best_validation_rmse": best_val,
    }

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "actual": test_actuals,
            "prediction": test_preds,
            "residual": test_preds - test_actuals,
        }
    ).to_csv(
        RESULTS_DIR / "predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    save_training_plots(history)

    save_test_plots(
        test_actuals,
        test_preds,
    )

    print("\nSaved results to:")
    print(
        f"  {RESULTS_DIR}/"
    )

    print("\nPlots:")
    print(
        f"  {PLOTS_DIR}/training_loss.png"
    )
    print(
        f"  {PLOTS_DIR}/validation_rmse.png"
    )
    print(
        f"  {PLOTS_DIR}/validation_mae.png"
    )
    print(
        f"  {PLOTS_DIR}/test_actual_vs_pred.png"
    )
    print(
        f"  {PLOTS_DIR}/test_residuals.png"
    )


if __name__ == "__main__":
    main()