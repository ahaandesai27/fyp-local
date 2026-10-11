"""
Enhanced Hypothesis Model 2: Recurrent Spiking Neural Network (RSNN)
with Recurrent LIF Neurons, Learnable Dynamics, and Residual Delta Learning.
"""
import torch
import torch.nn as nn
from models.snn_layers import get_rlif_neuron, get_lif_neuron, SNNTORCH_AVAILABLE

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    pass


class RSNNForecaster(nn.Module):
    """
    Recurrent SNN with Recurrent LIF neurons and residual delta learning.
    """
    def __init__(self, input_dim=11, hidden_dim=64, beta=0.90, threshold=1.0,
                 reset_mechanism="subtract", slope=25.0, prefer_snntorch=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prefer_snntorch = prefer_snntorch

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        if prefer_snntorch and SNNTORCH_AVAILABLE:
            grad_fn = surrogate.fast_sigmoid(slope=slope)
            self.rlif = snn.RLeaky(beta=beta, threshold=threshold, learn_beta=True,
                                  linear_features=hidden_dim, all_to_all=True,
                                  reset_mechanism=reset_mechanism, spike_grad=grad_fn)
            self.lif2 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=True,
                                  reset_mechanism=reset_mechanism, spike_grad=grad_fn)
        else:
            self.rlif = get_rlif_neuron(size=hidden_dim, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism, slope=slope, prefer_snntorch=False)
            self.lif2 = get_lif_neuron(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism, slope=slope, prefer_snntorch=False)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        current_p = x[:, -1, 0:1]

        spk_rec = None
        mem_rec = None
        mem2 = None
        spike_accumulator = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        total_spikes_layer1 = 0
        total_spikes_layer2 = 0
        decay = 0.92

        for step in range(seq_len):
            x_t = x[:, step, :]
            cur_in = self.norm1(self.fc_in(x_t))

            # Recurrent LIF dynamics
            if self.prefer_snntorch and SNNTORCH_AVAILABLE:
                if spk_rec is None:
                    spk_rec = torch.zeros(batch_size, self.hidden_dim, device=x.device)
                if mem_rec is None:
                    mem_rec = torch.zeros(batch_size, self.hidden_dim, device=x.device)
                spk_rec, mem_rec = self.rlif(cur_in, spk_rec, mem_rec)
            else:
                spk_rec, mem_rec = self.rlif(cur_in, mem_rec, spk_rec)
            total_spikes_layer1 += spk_rec.sum().item()

            # Second layer
            cur2 = self.norm2(self.fc2(spk_rec))
            if self.prefer_snntorch and SNNTORCH_AVAILABLE:
                if mem2 is None:
                    mem2 = torch.zeros_like(cur2)
                spk2, mem2 = self.lif2(cur2, mem2)
            else:
                spk2, mem2 = self.lif2(cur2, mem2)
            total_spikes_layer2 += spk2.sum().item()

            spike_accumulator = decay * spike_accumulator + spk2

        delta = self.delta_head(spike_accumulator)
        energy_pred = current_p + delta
        trend_logits = self.trend_head(spike_accumulator)

        total_neurons = (self.hidden_dim + self.hidden_dim) * batch_size * seq_len
        total_spikes = total_spikes_layer1 + total_spikes_layer2
        spike_rate = total_spikes / max(1, total_neurons)
        synops = (total_spikes_layer1 * (self.hidden_dim + self.hidden_dim)) + (total_spikes_layer2 * 1)

        return energy_pred, trend_logits, {
            "total_spikes": total_spikes,
            "spike_rate": spike_rate,
            "synops": synops
        }
