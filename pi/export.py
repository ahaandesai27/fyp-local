import torch
from model import SNN

model = SNN(
    input_size=10,
    hidden_size=32,
    output_size=4
)

model.load_state_dict(
    torch.load("snn_model.pth", weights_only=True)
)

model.eval()

# Extract learned weights and biases
data = {
    "fc1_weight": model.fc1.weight.detach().numpy(),
    "fc1_bias": model.fc1.bias.detach().numpy(),

    "fc2_weight": model.fc2.weight.detach().numpy(),
    "fc2_bias": model.fc2.bias.detach().numpy(),
}

# Save them in NumPy format
import numpy as np

np.savez(
    "snn_weights.npz",
    **data
)

print("Exported to snn_weights.npz")