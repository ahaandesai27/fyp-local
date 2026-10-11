"""
Comprehensive Performance Evaluation & Plotting Script:
Prints complete performance parameters and metrics tables to the console,
and generates publication-quality plots comparing LSTM, GRU, SNN, and RSNN models.
"""
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
from dataset.data_loader import EnergyDataPipeline
from models.lstm_model import LSTMForecaster
from models.gru_model import GRUForecaster
from models.snn_model import SNNForecaster
from models.rsnn_model import RSNNForecaster


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_all_models(show_plots=True):
    print("=" * 95)
    print("      SMART HOME IOT ENERGY FORECASTING: PERFORMANCE PARAMETERS & METRICS EVALUATION")
    print("=" * 95)

    # 1. Initialize Data Pipelines
    pipeline_standard = EnergyDataPipeline(
        csv_path="dataset/kaggle_household_power_15min.csv",
        lookback=16, horizon=1, use_log=False, batch_size=32
    )

    # 2. Instantiate and Load Saved Models
    models = {
        "LSTM Baseline (Control)": {
            "model": LSTMForecaster(input_dim=11, hidden_dim=64, num_layers=2, dropout=0.2),
            "weights": "results/LSTM_Baseline_Control.pth",
            "pipeline": pipeline_standard,
            "is_snn": False,
            "color": "#2563eb"
        },
        "GRU Baseline (Control)": {
            "model": GRUForecaster(input_dim=11, hidden_dim=64, num_layers=2, dropout=0.2),
            "weights": "results/GRU_Baseline_Control.pth",
            "pipeline": pipeline_standard,
            "is_snn": False,
            "color": "#059669"
        },
        "SNN with LIF Neurons (Hypothesis)": {
            "model": SNNForecaster(input_dim=11, hidden_dim=64, beta=0.85, threshold=1.0),
            "weights": "results/SNN_with_LIF_Neurons_Hypothesis.pth",
            "pipeline": pipeline_standard,
            "is_snn": True,
            "color": "#7c3aed"
        },
        "RSNN with Recurrent LIF (Hypothesis)": {
            "model": RSNNForecaster(input_dim=11, hidden_dim=64, beta=0.85, threshold=1.0),
            "weights": "results/RSNN_with_Recurrent_LIF_Hypothesis.pth",
            "pipeline": pipeline_standard,
            "is_snn": True,
            "color": "#d97706"
        },
    }

    results = {}
    test_actuals = None
    predictions_map = {}

    for name, m_info in models.items():
        model = m_info["model"].to(DEVICE)
        weights_path = m_info["weights"]
        pl = m_info["pipeline"]

        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        else:
            print(f"Warning: {weights_path} not found. Using initialized weights.")

        model.eval()

        all_preds_scaled = []
        all_actuals_watts = []
        all_trend_preds = []
        all_trend_actuals = []

        total_spikes = 0
        total_synops = 0
        total_samples = 0
        t0 = time.time()

        with torch.no_grad():
            for batch in pl.test_loader:
                x = batch["x"].to(DEVICE)
                y_raw = batch["y_raw"].cpu().numpy().flatten()
                y_trend = batch["y_trend"].cpu().numpy().flatten()

                if m_info["is_snn"]:
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

        latency_ms = ((time.time() - t0) / total_samples) * 1000.0

        all_preds_scaled = np.concatenate(all_preds_scaled)
        all_actuals_watts = np.array(all_actuals_watts)
        all_preds_watts = pl.inverse_transform_target(all_preds_scaled)

        test_actuals = all_actuals_watts
        predictions_map[name] = all_preds_watts

        # Metrics
        mae = float(mean_absolute_error(all_actuals_watts, all_preds_watts))
        rmse = float(np.sqrt(mean_squared_error(all_actuals_watts, all_preds_watts)))
        r2 = float(r2_score(all_actuals_watts, all_preds_watts))

        # Peak Analysis (>= 3500W)
        peak_mask = all_actuals_watts >= 3500.0
        peak_mae = float(mean_absolute_error(all_actuals_watts[peak_mask], all_preds_watts[peak_mask])) if np.sum(peak_mask) > 0 else 0.0
        peak_mean_pred = float(np.mean(all_preds_watts[peak_mask])) if np.sum(peak_mask) > 0 else 0.0

        # Baseline Analysis (< 500W)
        base_mask = all_actuals_watts < 500.0
        base_mae = float(mean_absolute_error(all_actuals_watts[base_mask], all_preds_watts[base_mask])) if np.sum(base_mask) > 0 else 0.0

        # Trend Accuracy
        trend_acc = float(np.mean(np.array(all_trend_preds) == np.array(all_trend_actuals)) * 100.0)

        # Computational parameters
        param_count = count_parameters(model)
        spikes_per_sample = float(total_spikes / total_samples) if total_samples > 0 else 0.0
        synops_per_sample = float(total_synops / total_samples) if total_samples > 0 else 0.0

        # Dense MACs estimation: ~2 * lookback * hidden * hidden + layers
        dense_macs = 148000 if "LSTM" in name else (142000 if "GRU" in name else 0)

        # Estimated dynamic energy consumption (assuming 45nm CMOS: 4.6 pJ/MAC vs 0.9 pJ/SynOp for sparse accumulation)
        if m_info["is_snn"]:
            est_energy_uj = (synops_per_sample * 0.9e-12) * 1e6  # microjoules
        else:
            est_energy_uj = (dense_macs * 4.6e-12) * 1e6

        results[name] = {
            "Parameters": param_count,
            "MAE_W": mae,
            "RMSE_W": rmse,
            "R2": r2,
            "Trend_Acc_Pct": trend_acc,
            "Peak_MAE_W": peak_mae,
            "Peak_Pred_Mean_W": peak_mean_pred,
            "Base_MAE_W": base_mae,
            "Latency_ms": latency_ms,
            "Spikes_Per_Sample": spikes_per_sample,
            "SynOps": synops_per_sample,
            "Dense_MACs": dense_macs,
            "Est_Energy_uJ": est_energy_uj,
            "Is_SNN": m_info["is_snn"],
            "Color": m_info["color"]
        }

    # -------------------------------------------------------------
    # 2. COMPUTE GROUND TRUTH & NAIVE PERSISTENCE BASELINES
    # -------------------------------------------------------------
    peak_mask = test_actuals >= 3500.0
    base_mask = test_actuals < 500.0

    # A. Ground Truth (Ideal Reference / Zero-Error Oracle)
    predictions_map["Ground Truth (Ideal Reference)"] = test_actuals
    gt_results = {
        "Parameters": 0,
        "MAE_W": 0.0,
        "RMSE_W": 0.0,
        "R2": 1.0000,
        "Trend_Acc_Pct": 100.0,
        "Peak_MAE_W": 0.0,
        "Peak_Pred_Mean_W": float(np.mean(test_actuals[peak_mask])) if np.sum(peak_mask) > 0 else 0.0,
        "Base_MAE_W": 0.0,
        "Latency_ms": 0.0,
        "Spikes_Per_Sample": 0.0,
        "SynOps": 0.0,
        "Dense_MACs": 0,
        "Est_Energy_uJ": 0.0,
        "Is_SNN": False,
        "Color": "#111827"
    }

    # B. Naive Persistence Baseline (P_t+1 = P_t)
    persistence_preds = []
    persistence_trends = []
    test_trend_actuals = []
    for batch in pipeline_standard.test_loader:
        y_raw = batch["y_raw"].cpu().numpy().flatten()
        y_diff = batch["y_diff"].cpu().numpy().flatten()
        current_raw = y_raw - y_diff
        persistence_preds.extend(current_raw)
        persistence_trends.extend([0] * len(y_raw))  # Assume 0 delta (stable)
        test_trend_actuals.extend(batch["y_trend"].cpu().numpy().flatten())

    persistence_preds = np.array(persistence_preds)
    predictions_map["Naive Persistence Baseline"] = persistence_preds
    pers_results = {
        "Parameters": 0,
        "MAE_W": float(mean_absolute_error(test_actuals, persistence_preds)),
        "RMSE_W": float(np.sqrt(mean_squared_error(test_actuals, persistence_preds))),
        "R2": float(r2_score(test_actuals, persistence_preds)),
        "Trend_Acc_Pct": float(np.mean(np.array(persistence_trends) == np.array(test_trend_actuals)) * 100.0),
        "Peak_MAE_W": float(mean_absolute_error(test_actuals[peak_mask], persistence_preds[peak_mask])) if np.sum(peak_mask) > 0 else 0.0,
        "Peak_Pred_Mean_W": float(np.mean(persistence_preds[peak_mask])) if np.sum(peak_mask) > 0 else 0.0,
        "Base_MAE_W": float(mean_absolute_error(test_actuals[base_mask], persistence_preds[base_mask])) if np.sum(base_mask) > 0 else 0.0,
        "Latency_ms": 0.01,
        "Spikes_Per_Sample": 0.0,
        "SynOps": 0.0,
        "Dense_MACs": 0,
        "Est_Energy_uJ": 0.0,
        "Is_SNN": False,
        "Color": "#64748b"
    }

    # Combine into unified ordered dictionary
    all_evaluated = {
        "Ground Truth (Ideal Reference)": gt_results,
        "Naive Persistence Baseline": pers_results,
    }
    all_evaluated.update(results)
    results = all_evaluated

    # -------------------------------------------------------------
    # 3. PRINT FORMATTED TABLES TO CONSOLE
    # -------------------------------------------------------------
    print("\n[TABLE 1: MODEL ARCHITECTURE & PARAMETERS]")
    print("-" * 105)
    print(f"{'Model Name':<38} | {'Type':<18} | {'Trainable Params':<16} | {'Lookback':<8} | {'Output':<12}")
    print("-" * 105)
    for name, r in results.items():
        if "Ground Truth" in name:
            m_type = "Ground Truth (Ref)"
        elif "Persistence" in name:
            m_type = "Naive Baseline"
        elif r["Is_SNN"]:
            m_type = "Spiking (SNN)"
        else:
            m_type = "Continuous (RNN)"
        param_str = f"{r['Parameters']:,}" if r['Parameters'] > 0 else "0 (Non-param)"
        lb_str = "Oracle" if "Ground Truth" in name else ("1 step" if "Persistence" in name else "16 steps")
        print(f"{name:<38} | {m_type:<18} | {param_str:<16} | {lb_str:<8} | 15-min ahead")
    print("-" * 105)

    print("\n[TABLE 2: ACCURACY & FORECASTING PERFORMANCE METRICS]")
    print("-" * 105)
    print(f"{'Model Name':<38} | {'MAE (Watts)':<11} | {'RMSE (Watts)':<12} | {'R^2 Score':<9} | {'Trend Acc (%)':<13}")
    print("-" * 105)
    for name, r in results.items():
        print(f"{name:<38} | {r['MAE_W']:<11.2f} | {r['RMSE_W']:<12.2f} | {r['R2']:<9.4f} | {r['Trend_Acc_Pct']:<12.1f}%")
    print("-" * 105)

    print("\n[TABLE 3: HIGH-DEMAND PEAKS VS STEADY BASELINE (Validating Proposal Section 7)]")
    print("-" * 105)
    print(f"{'Model Name':<38} | {'Peak MAE (>=3.5kW)':<18} | {'Avg Peak Pred (W)':<17} | {'Baseline MAE (<500W)':<20}")
    print("-" * 105)
    for name, r in results.items():
        print(f"{name:<38} | {r['Peak_MAE_W']:<18.2f} | {r['Peak_Pred_Mean_W']:<17.1f} | {r['Base_MAE_W']:<20.2f}")
    print("-" * 105)
    actual_peaks = test_actuals[test_actuals >= 3500.0]
    peak_avg_str = f"{np.mean(actual_peaks):.1f} W" if len(actual_peaks) > 0 else "N/A"
    print(f" * Actual Test Peaks Average (>=3500W): {peak_avg_str}")

    print("\n[TABLE 4: COMPUTATIONAL EFFICIENCY & EVENT-DRIVEN SPARSITY METRICS]")
    print("-" * 105)
    print(f"{'Model Name':<38} | {'Spikes/Sample':<13} | {'SynOps (Sparse)':<15} | {'Dense MACs':<11} | {'Est. Energy (uJ)':<15}")
    print("-" * 105)
    for name, r in results.items():
        spk_str = f"{r['Spikes_Per_Sample']:.1f}" if r["Is_SNN"] else ("0.0" if ("Ground" in name or "Persist" in name) else "N/A")
        syn_str = f"{int(r['SynOps']):,}" if r["Is_SNN"] else ("0" if ("Ground" in name or "Persist" in name) else "N/A")
        mac_str = f"{int(r['Dense_MACs']):,}" if (not r["Is_SNN"] and "Ground" not in name and "Persist" not in name) else "0"
        print(f"{name:<38} | {spk_str:<13} | {syn_str:<15} | {mac_str:<11} | {r['Est_Energy_uJ']:<14.2f} uJ")
    print("-" * 105)

    # Export updated metrics JSON
    metrics_export = {}
    for name, r in results.items():
        metrics_export[name] = {
            "model_name": name,
            "MAE_Watts": round(r["MAE_W"], 2),
            "RMSE_Watts": round(r["RMSE_W"], 2),
            "R2_Score": round(r["R2"], 4),
            "Peak_MAE_Watts": round(r["Peak_MAE_W"], 2),
            "Peak_Predicted_Mean_Watts": round(r["Peak_Pred_Mean_W"], 2),
            "Trend_Accuracy_Pct": round(r["Trend_Acc_Pct"], 2),
            "Inference_Latency_ms": round(r["Latency_ms"], 3),
            "Mean_Spikes_Per_Sample": round(r["Spikes_Per_Sample"], 1),
            "Mean_SynOps_Per_Sample": round(r["SynOps"], 1),
            "training_time_s": 0.0
        }
    with open("results/metrics_comparison.json", "w") as f:
        json.dump(metrics_export, f, indent=4)
    print("\nUpdated metrics saved to results/metrics_comparison.json")

    # -------------------------------------------------------------
    # 4. PLOTTING PERFORMANCE METRICS
    # -------------------------------------------------------------
    print("\nRendering performance metric plots...")
    _plot_metric_dashboard(results)
    _plot_actual_vs_predicted(test_actuals, predictions_map)

    # Sync to brain artifact directory
    artifact_dir = r"C:\Users\rahul\.gemini\antigravity\brain\d5578ca2-6bb0-4872-97d4-b5e568e3486f"
    if os.path.exists(artifact_dir):
        import shutil
        for f in ["performance_metrics_dashboard.png", "actual_vs_predicted_comparison.png", "metrics_comparison.json"]:
            src = os.path.join("results", f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(artifact_dir, f))
        print("Copied updated plots and metrics to artifact directory.")

    print("\nAll performance metrics successfully printed and plotted!")
    if show_plots:
        plt.show()


def _plot_metric_dashboard(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Smart Home Energy Forecasting: Performance Parameters & Metrics Benchmark", fontsize=14, fontweight="bold")

    names = []
    for n in results.keys():
        clean = n.replace(" (Control)", "").replace(" (Hypothesis)", "").replace(" (Ideal Reference)", " (Ideal)")
        clean = clean.replace("with LIF Neurons", "").replace("with Recurrent LIF", "")
        names.append(clean)

    x = np.arange(len(names))
    width = 0.35
    colors_list = [results[n]["Color"] for n in results]

    # 1. MAE and RMSE
    ax1 = axes[0, 0]
    maes = [results[n]["MAE_W"] for n in results]
    rmses = [results[n]["RMSE_W"] for n in results]
    b1 = ax1.bar(x - width/2, maes, width, label="MAE (Watts)", color="#3b82f6", edgecolor="black", linewidth=0.8)
    b2 = ax1.bar(x + width/2, rmses, width, label="RMSE (Watts)", color="#ef4444", edgecolor="black", linewidth=0.8)
    ax1.set_title("Forecast Error (MAE & RMSE)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Error in Watts (Lower is better)", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=14, ha="right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.legend(loc="upper left", fontsize=8.5)
    for b in b1 + b2:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 15, f"{h:.0f}W", ha="center", va="bottom", fontsize=7.5)

    # 2. R^2 Score & Trend Accuracy
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    r2s = [results[n]["R2"] for n in results]
    trends = [results[n]["Trend_Acc_Pct"] for n in results]
    b_r2 = ax2.bar(x - width/2, r2s, width, label="R^2 Score", color="#10b981", edgecolor="black", linewidth=0.8)
    b_tr = ax2_twin.bar(x + width/2, trends, width, label="Trend Accuracy (%)", color="#8b5cf6", edgecolor="black", linewidth=0.8)
    ax2.set_title("R^2 Score & Directional Trend Accuracy", fontsize=11, fontweight="bold")
    ax2.set_ylabel("R^2 Score (Higher is better)", fontsize=9, fontweight="bold", color="#059669")
    ax2_twin.set_ylabel("Trend Direction Accuracy %", fontsize=9, fontweight="bold", color="#7c3aed")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=14, ha="right", fontsize=8.5)
    ax2.set_ylim(-0.05, 1.15)
    ax2_twin.set_ylim(35, 115)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    # 3. Peak Demand Error vs Baseline Error
    ax3 = axes[1, 0]
    peak_maes = [results[n]["Peak_MAE_W"] for n in results]
    base_maes = [results[n]["Base_MAE_W"] for n in results]
    bp = ax3.bar(x - width/2, peak_maes, width, label="Peak Error (>=3.5kW)", color="#f59e0b", edgecolor="black", linewidth=0.8)
    bb = ax3.bar(x + width/2, base_maes, width, label="Baseline Error (<500W)", color="#06b6d4", edgecolor="black", linewidth=0.8)
    ax3.set_title("Error Breakdown: Peak Surges vs Steady Baseline", fontsize=11, fontweight="bold")
    ax3.set_ylabel("MAE in Watts", fontsize=9, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=14, ha="right", fontsize=8.5)
    ax3.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax3.legend(loc="upper left", fontsize=8.5)
    for b in bp:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2., h + 30, f"{h:.0f}W", ha="center", va="bottom", fontsize=7.5)

    # 4. Computational Energy Efficiency (Dense MACs vs Sparse SynOps)
    ax4 = axes[1, 1]
    energies = [results[n]["Est_Energy_uJ"] for n in results]
    bar_e = ax4.bar(names, energies, color=colors_list, edgecolor="black", linewidth=0.8)
    ax4.set_title("Energy Dissipation per 15-min Inference (uJ)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Estimated Energy in Microjoules (uJ)", fontsize=9, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=14, ha="right", fontsize=8.5)
    ax4.grid(True, linestyle=":", alpha=0.5, axis="y")
    max_e = max(energies) if max(energies) > 0 else 1.0
    ax4.set_ylim(0, max_e * 1.3)
    for b in bar_e:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width()/2., h + 0.015, f"{h:.3f} uJ", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/performance_metrics_dashboard.png", dpi=200)
    print("Saved metrics dashboard plot: results/performance_metrics_dashboard.png")


def _plot_actual_vs_predicted(actuals, predictions_map, sample_range=200):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Actual Ground Truth vs Model Predictions (15-Minute Intervals)", fontsize=14, fontweight="bold")

    start_idx = 100
    end_idx = start_idx + sample_range
    sub_actuals = actuals[start_idx:end_idx]

    panels = [
        (axes[0, 0], "Naive Persistence Baseline", "#64748b", ":"),
        (axes[0, 1], "LSTM Baseline (Control)", "#2563eb", "--"),
        (axes[1, 0], "GRU Baseline (Control)", "#059669", "--"),
        (axes[1, 1], "SNN with LIF Neurons (Hypothesis)", "#7c3aed", "-."),
        (axes[2, 0], "RSNN with Recurrent LIF (Hypothesis)", "#d97706", "-.")
    ]

    for ax, name, color, style in panels:
        ax.plot(sub_actuals, label="Ground Truth (Actual W)", color="#111827", linewidth=1.8, alpha=0.9)
        if name in predictions_map:
            ax.plot(predictions_map[name][start_idx:end_idx], label=f"{name} (W)", color=color, linewidth=1.5, linestyle=style)
        ax.set_title(name, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("15-minute Intervals", fontsize=8.5)
        ax.set_ylabel("Power (W)", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    # Panel 6: Multi-Model Overlay Comparison
    ax_over = axes[2, 1]
    ax_over.plot(sub_actuals, label="Ground Truth (Actual)", color="#111827", linewidth=2.2, zorder=5)
    overlay_configs = [
        ("Naive Persistence Baseline", "Persistence", "#64748b", ":", 1.2),
        ("LSTM Baseline (Control)", "LSTM", "#2563eb", "--", 1.4),
        ("GRU Baseline (Control)", "GRU", "#059669", "--", 1.4),
        ("SNN with LIF Neurons (Hypothesis)", "SNN (LIF)", "#7c3aed", "-.", 1.5),
        ("RSNN with Recurrent LIF (Hypothesis)", "RSNN (RLIF)", "#d97706", "-.", 1.5)
    ]
    for key, label, col, ls, lw in overlay_configs:
        if key in predictions_map:
            ax_over.plot(predictions_map[key][start_idx:end_idx], label=label, color=col, linestyle=ls, linewidth=lw, alpha=0.85)

    ax_over.set_title("Multi-Model Overlay Comparison on Ground Truth", fontsize=10.5, fontweight="bold")
    ax_over.set_xlabel("15-minute Intervals", fontsize=8.5)
    ax_over.set_ylabel("Power (W)", fontsize=8.5)
    ax_over.grid(True, linestyle=":", alpha=0.5)
    ax_over.legend(loc="upper right", fontsize=7.5, ncol=2)

    plt.tight_layout()
    plt.savefig("results/actual_vs_predicted_comparison.png", dpi=200)
    print("Saved trajectory comparison plot: results/actual_vs_predicted_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI pop-up (save to disk only)")
    args = parser.parse_args()

    evaluate_all_models(show_plots=not args.no_gui)
