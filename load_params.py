import torch

state = torch.load("results/snn_v3/model.pt", map_location="cpu", weights_only=False)

for name, value in state.items():
    print(name, value)