import torch.nn as nn
import snntorch as snn
import torch 

class SNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=4):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        # x: [time_steps, batch_size, input_size]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        outputs = []

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            outputs.append(spk2)

        return __import__("torch").stack(outputs)



model = SNN(
    input_size=10,
    hidden_size=32,
    output_size=4
)

# Random initialization is already done by nn.Linear
torch.save(model.state_dict(), "snn_model.pth")

print("Randomly initialized SNN saved to snn_model.pth")