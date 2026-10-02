import time
import psutil
import numpy as np


# ============================================================
# Configuration
# ============================================================

TIME_STEPS = 20
INPUT_SIZE = 10
HIDDEN_SIZE = 32
OUTPUT_SIZE = 4

BETA = 0.9
THRESHOLD = 1.0


# ============================================================
# Load weights
# ============================================================

weights = np.load("snn_weights.npz")

W1 = weights["fc1_weight"].astype(np.float32)
b1 = weights["fc1_bias"].astype(np.float32)

W2 = weights["fc2_weight"].astype(np.float32)
b2 = weights["fc2_bias"].astype(np.float32)


print("Weights loaded:")
print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)


# ============================================================
# LIF neuron
# ============================================================

def lif_step(cur, mem):
    """
    One timestep of a Leaky Integrate-and-Fire neuron.
    """

    # Membrane potential update
    mem = BETA * mem + cur

    # Generate spike
    spike = (mem >= THRESHOLD).astype(np.float32)

    # Reset membrane after spike
    mem = mem * (1.0 - spike)

    return spike, mem


# ============================================================
# SNN inference
# ============================================================

def inference(x):
    """
    x:
        [time_steps, input_size]

    Returns:
        prediction
        spike_counts
    """

    # Membrane potentials
    mem1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    mem2 = np.zeros(OUTPUT_SIZE, dtype=np.float32)

    output_spikes = np.zeros(
        (TIME_STEPS, OUTPUT_SIZE),
        dtype=np.float32
    )

    for t in range(TIME_STEPS):

        # -------------------------
        # Layer 1
        # -------------------------

        cur1 = W1 @ x[t] + b1

        spk1, mem1 = lif_step(
            cur1,
            mem1
        )

        # -------------------------
        # Layer 2
        # -------------------------

        cur2 = W2 @ spk1 + b2

        spk2, mem2 = lif_step(
            cur2,
            mem2
        )

        output_spikes[t] = spk2

    # -------------------------
    # Classification
    # -------------------------

    spike_counts = output_spikes.sum(axis=0)

    prediction = int(
        np.argmax(spike_counts)
    )

    return prediction, spike_counts


# ============================================================
# RAM / CPU monitoring
# ============================================================

process = psutil.Process()

print("\nStarting inference...")
print("Press Ctrl+C to stop.\n")


# ============================================================
# Continuous inference
# ============================================================

try:

    while True:

        start = time.perf_counter()

        # Example input
        #
        # [time_steps, input_size]
        #
        x = np.random.rand(
            TIME_STEPS,
            INPUT_SIZE
        ).astype(np.float32)

        prediction, spike_counts = inference(x)

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        # Process RAM
        ram_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        # Process CPU
        cpu = process.cpu_percent(
            interval=None
        )

        print(
            f"RAM: {ram_mb:7.2f} MB | "
            f"CPU: {cpu:5.1f}% | "
            f"Latency: {elapsed:7.3f} ms | "
            f"Prediction: {prediction} | "
            f"Spikes: {spike_counts.astype(int)}"
        )

        time.sleep(1)


except KeyboardInterrupt:

    print("\nInference stopped.")