import time
import torch
import psutil

from model import SNN


# Recreate the architecture
model = SNN(
    input_size=10,
    hidden_size=32,
    output_size=4
)

# Load saved weights
model.load_state_dict(
    torch.load("snn_model.pth", weights_only=True)
)

model.eval()

process = psutil.Process()

print("Starting continuous inference...")
print("Press Ctrl+C to stop.\n")


try:
    while True:

        # Example input
        # [time_steps, batch_size, input_size]
        x = torch.rand(20, 2, 10)

        # Inference
        with torch.no_grad():
            spikes = model(x)

        # Count output spikes over time
        spike_counts = spikes.sum(dim=0)

        # Pick neuron with highest spike count
        predictions = spike_counts.argmax(dim=1)

        # Resource usage
        cpu = process.cpu_percent(interval=None)
        ram_mb = process.memory_info().rss / (1024 ** 2)

        print(
            f"CPU: {cpu:5.1f}% | "
            f"RAM: {ram_mb:7.1f} MB | "
            f"Prediction: {predictions.tolist()}"
        )

        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped.")