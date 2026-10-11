"""
Generates and plots a spike raster and membrane potential trace for SNN / RSNN models.
"""
import torch
import matplotlib.pyplot as plt
from dataset.data_loader import EnergyDataPipeline
from models.snn_model import SNNForecaster
from models.rsnn_model import RSNNForecaster


def generate_spike_raster():
    pipeline = EnergyDataPipeline(csv_path="dataset/kaggle_household_power_15min.csv", lookback=16, horizon=1, batch_size=1)
    sample_batch = next(iter(pipeline.test_loader))
    x = sample_batch["x"]  # (1, 16, 11)

    snn = SNNForecaster(input_dim=11, hidden_dim=64, beta=0.85, threshold=1.0)
    # Load trained checkpoint if exists
    try:
        snn.load_state_dict(torch.load("results/SNN_with_LIF_Neurons_Hypothesis.pth", map_location="cpu"))
        print("Loaded trained SNN weights.")
    except Exception as e:
        print(f"Using initialized weights: {e}")

    snn.eval()

    # Capture spikes step-by-step
    spikes_layer1 = []
    mem_layer1 = []
    mem1 = None

    for step in range(x.shape[1]):
        x_t = x[:, step, :]
        cur1 = snn.fc1(x_t)
        if mem1 is None:
            mem1 = torch.zeros_like(cur1)
        spk1, mem1 = snn.lif1(cur1, mem1)
        spikes_layer1.append(spk1.squeeze(0).detach().cpu().numpy())
        mem_layer1.append(mem1.squeeze(0).detach().cpu().numpy())

    import numpy as np
    spikes_arr = np.array(spikes_layer1).T  # (neurons, timesteps)
    mem_arr = np.array(mem_layer1).T        # (neurons, timesteps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 1. Spike Raster Plot
    neuron_idx, step_idx = np.where(spikes_arr > 0)
    ax1.scatter(step_idx, neuron_idx, color="#7c3aed", marker="|", s=100, linewidth=2)
    ax1.set_ylabel("Neuron Index (0-63)", fontsize=10, fontweight="bold")
    ax1.set_title("SNN Layer 1: Event-Driven Spike Raster Across Timesteps", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Membrane Potential Trace of 3 sample neurons
    for n_i, col in zip([5, 18, 42], ["#2563eb", "#ea580c", "#059669"]):
        ax2.plot(mem_arr[n_i], label=f"Neuron {n_i} U_mem", color=col, linewidth=1.8)
    ax2.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Firing Threshold (U_th = 1.0)")
    ax2.set_xlabel("Time Step in Lookback Window (t = 0 to 15)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Membrane Potential (U)", fontsize=10, fontweight="bold")
    ax2.set_title("Leaky Integrate-and-Fire (LIF) Subthreshold Dynamics", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("results/snn_spike_raster.png", dpi=200)
    plt.close()
    print("Saved spike raster plot to results/snn_spike_raster.png")


if __name__ == "__main__":
    generate_spike_raster()
