"""
Training & Evaluation Pipeline for Smart Home Energy Forecasting:
Trains and evaluates LSTM, GRU, SNN, and RSNN models, computes MAE, RMSE, R^2,
peak accuracy, spike sparsity, and generates visual comparison charts.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

from dataset.data_loader import EnergyDataPipeline
from models.lstm_model import LSTMForecaster
from models.gru_model import GRUForecaster
from models.snn_model import SNNForecaster
from models.rsnn_model import RSNNForecaster


# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_single_model(model_name, model, train_loader, val_loader, pipeline, epochs=20, lr=1e-3):
    """
    Trains a model (LSTM, GRU, SNN, or RSNN) and evaluates on the validation set.
    """
    print(f"\n=======================================================")
    print(f"  Training {model_name} on {DEVICE}")
    print(f"=======================================================")

    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    def compute_loss(pred_energy, y, pred_trend, y_trend):
        loss_raw = nn.functional.smooth_l1_loss(pred_energy, y, beta=0.05, reduction="none")
        # Give higher penalty to high-power peak states
        weights = 1.0 + 2.0 * (y > 0.8).float()
        loss_reg = (loss_raw * weights).mean()
        loss_cls = nn.functional.cross_entropy(pred_trend, y_trend)
        return loss_reg + 0.15 * loss_cls

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_loss = float("inf")
    best_weights = None

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            y_trend = batch["y_trend"].to(DEVICE)

            optimizer.zero_grad()

            if "SNN" in model_name:
                pred_energy, pred_trend, _ = model(x)
            else:
                pred_energy, pred_trend = model(x)

            loss = compute_loss(pred_energy, y, pred_trend, y_trend)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(x)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_scaled = []
        val_actuals_watts = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(DEVICE)
                y = batch["y"].to(DEVICE)
                y_raw = batch["y_raw"].cpu().numpy().flatten()
                y_trend = batch["y_trend"].to(DEVICE)

                if "SNN" in model_name:
                    pred_energy, pred_trend, _ = model(x)
                else:
                    pred_energy, pred_trend = model(x)

                loss = compute_loss(pred_energy, y, pred_trend, y_trend)

                val_loss += loss.item() * len(x)
                val_preds_scaled.append(pred_energy.cpu().numpy().flatten())
                val_actuals_watts.extend(y_raw)

        val_loss /= len(val_loader.dataset)
        val_preds_scaled = np.concatenate(val_preds_scaled)
        val_preds_watts = pipeline.inverse_transform_target(val_preds_scaled)
        val_mae = mean_absolute_error(val_actuals_watts, val_preds_watts)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.1f} W")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s | Best Val Loss: {best_val_loss:.4f}")

    if best_weights:
        model.load_state_dict(best_weights)

    return model, history, training_time


def evaluate_model(model_name, model, test_loader, pipeline):
    """
    Evaluates model performance on the test dataset.
    Computes MAE, RMSE, R^2, Peak Prediction Error, Trend Accuracy, and SynOps/Spikes.
    """
    model.eval()
    all_preds_scaled = []
    all_actuals_watts = []
    all_trend_preds = []
    all_trend_actuals = []

    total_spikes = 0
    total_synops = 0
    total_samples = 0
    inference_start = time.time()

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(DEVICE)
            y_raw = batch["y_raw"].cpu().numpy().flatten()
            y_trend = batch["y_trend"].cpu().numpy().flatten()

            if "SNN" in model_name:
                pred_energy, pred_trend, meta = model(x)
                total_spikes += meta["total_spikes"]
                total_synops += meta["synops"]
            else:
                pred_energy, pred_trend = model(x)

            all_preds_scaled.append(pred_energy.cpu().numpy().flatten())
            all_actuals_watts.extend(y_raw)
            all_trend_preds.extend(torch.argmax(pred_trend, dim=1).cpu().numpy().flatten())
            all_trend_actuals.extend(y_trend)
            total_samples += len(x)

    inference_time = (time.time() - inference_start) / total_samples

    all_preds_scaled = np.concatenate(all_preds_scaled)
    all_actuals_watts = np.array(all_actuals_watts)
    all_preds_watts = pipeline.inverse_transform_target(all_preds_scaled)

    # Core Regression Metrics
    mae = float(mean_absolute_error(all_actuals_watts, all_preds_watts))
    rmse = float(np.sqrt(mean_squared_error(all_actuals_watts, all_preds_watts)))
    r2 = float(r2_score(all_actuals_watts, all_preds_watts))

    # Peak Prediction Metrics (Samples >= 4500W, per Proposal Section 7)
    peak_mask = all_actuals_watts >= 4500.0
    if np.sum(peak_mask) > 0:
        peak_actual = all_actuals_watts[peak_mask]
        peak_pred = all_preds_watts[peak_mask]
        peak_mae = float(mean_absolute_error(peak_actual, peak_pred))
        peak_cap_mean = float(np.mean(peak_pred))
    else:
        peak_mae = 0.0
        peak_cap_mean = 0.0

    # Trend Classification Accuracy (Objective 4 of Proposal)
    trend_correct = np.sum(np.array(all_trend_preds) == np.array(all_trend_actuals))
    trend_acc = float(trend_correct / len(all_trend_actuals) * 100.0)

    # SynOps & Sparsity
    mean_spikes_per_sample = float(total_spikes / total_samples) if total_samples > 0 else 0.0
    mean_synops_per_sample = float(total_synops / total_samples) if total_samples > 0 else 0.0

    results = {
        "model_name": model_name,
        "MAE_Watts": round(mae, 2),
        "RMSE_Watts": round(rmse, 2),
        "R2_Score": round(r2, 4),
        "Peak_MAE_Watts": round(peak_mae, 2),
        "Peak_Predicted_Mean_Watts": round(peak_cap_mean, 2),
        "Trend_Accuracy_Pct": round(trend_acc, 2),
        "Inference_Latency_ms": round(inference_time * 1000.0, 3),
        "Mean_Spikes_Per_Sample": round(mean_spikes_per_sample, 1),
        "Mean_SynOps_Per_Sample": round(mean_synops_per_sample, 0),
        "predictions": all_preds_watts,
        "actuals": all_actuals_watts
    }

    print(f"\n[{model_name} Test Results]")
    print(f"  MAE  : {mae:.2f} W")
    print(f"  RMSE : {rmse:.2f} W")
    print(f"  R^2  : {r2:.4f}")
    print(f"  Trend Accuracy: {trend_acc:.1f}%")
    if "SNN" in model_name:
        print(f"  Mean Spikes: {mean_spikes_per_sample:.1f} | SynOps: {mean_synops_per_sample:,.0f}")
    if np.sum(peak_mask) > 0:
        print(f"  Peak MAE: {peak_mae:.2f} W | Peak Pred Mean: {peak_cap_mean:.1f} W (Actual Peaks: {np.mean(peak_actual):.1f} W)")

    return results


def run_all_experiments():
    """Main training & evaluation coordinator."""
    os.makedirs("results", exist_ok=True)

    print("Step 1: Preparing Data Pipeline using real Kaggle dataset...")
    pipeline_standard = EnergyDataPipeline(
        csv_path="dataset/kaggle_household_power_15min.csv",
        lookback=16, horizon=1, use_log=False, batch_size=32
    )

    models_config = [
        # (Name, Model Instance, Pipeline, Epochs, LR)
        ("LSTM Baseline (Control)", LSTMForecaster(input_dim=11, hidden_dim=64, num_layers=2, dropout=0.2), pipeline_standard, 20, 1e-3),
        ("GRU Baseline (Control)", GRUForecaster(input_dim=11, hidden_dim=64, num_layers=2, dropout=0.2), pipeline_standard, 20, 1e-3),
        ("SNN with LIF Neurons (Hypothesis)", SNNForecaster(input_dim=11, hidden_dim=64, beta=0.85, threshold=1.0), pipeline_standard, 25, 2e-3),
        ("RSNN with Recurrent LIF (Hypothesis)", RSNNForecaster(input_dim=11, hidden_dim=64, beta=0.85, threshold=1.0), pipeline_standard, 25, 2e-3),
    ]

    all_results = {}
    all_histories = {}
    predictions_dict = {}

    for name, model, pl, epochs, lr in models_config:
        trained_model, history, train_time = train_single_model(
            name, model, pl.train_loader, pl.val_loader, pl, epochs=epochs, lr=lr
        )
        res = evaluate_model(name, trained_model, pl.test_loader, pl)
        res["training_time_s"] = round(train_time, 2)

        predictions_dict[name] = res.pop("predictions")
        actuals = res.pop("actuals")
        all_results[name] = res
        all_histories[name] = history

        # Save model checkpoint
        torch.save(trained_model.state_dict(), f"results/{name.replace(' ', '_').replace('(', '').replace(')', '')}.pth")

    # Save summary JSON
    with open("results/metrics_comparison.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nMetrics summary saved to results/metrics_comparison.json")

    # Plot 1: Actual vs Predicted Power Trajectories (Matching Proposal Section 7)
    plot_actual_vs_predicted(actuals, predictions_dict)

    # Plot 2: Training & Validation Loss Curves
    plot_training_curves(all_histories)

    print("\nAll models trained, evaluated, and plotted successfully!")


def plot_actual_vs_predicted(actuals, predictions_dict, sample_range=200):
    """
    Renders actual vs predicted trajectory comparison, matching Proposal Section 7 figures.
    """
    plt.figure(figsize=(15, 10))

    # Slice a clear 200-step test window (~50 hours) with peaks and steady periods
    start_idx = 100
    end_idx = start_idx + sample_range
    sub_actuals = actuals[start_idx:end_idx]

    subplots = [
        (2, 2, 1, "LSTM Baseline (Control)"),
        (2, 2, 2, "GRU Baseline (Control)"),
        (2, 2, 3, "SNN with LIF Neurons (Hypothesis)"),
        (2, 2, 4, "RSNN with Recurrent LIF (Hypothesis)")
    ]

    for row, col, idx, model_name in subplots:
        plt.subplot(2, 2, idx)
        plt.plot(sub_actuals, label="Actual Power (W)", color="#2563eb", linewidth=1.5, alpha=0.9)
        if model_name in predictions_dict:
            pred_sub = predictions_dict[model_name][start_idx:end_idx]
            plt.plot(pred_sub, label="Predicted Power (W)", color="#ea580c", linewidth=1.5, linestyle="--")

        plt.title(f"{model_name}\nActual vs Predicted Power (15-min Interval)", fontsize=11, fontweight="bold")
        plt.xlabel("15-minute Time Intervals", fontsize=9)
        plt.ylabel("Power (W)", fontsize=9)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("results/actual_vs_predicted_comparison.png", dpi=200)
    plt.close()
    print("Saved trajectory comparison plot to results/actual_vs_predicted_comparison.png")


def plot_training_curves(histories):
    """Plots training and validation loss progression."""
    plt.figure(figsize=(12, 6))

    colors = ["#2563eb", "#059669", "#7c3aed", "#d97706"]

    for i, (name, hist) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        plt.plot(hist["val_loss"], label=f"{name} (Val Loss)", color=color, linewidth=2)
        plt.plot(hist["train_loss"], color=color, linewidth=1, linestyle=":", alpha=0.6)

    plt.title("Training & Validation Loss Convergence Across Models", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss (MSE + Auxiliary Classification)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=200)
    plt.close()
    print("Saved training curves plot to results/training_curves.png")


if __name__ == "__main__":
    run_all_experiments()
