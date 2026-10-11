"""
Enhanced Hypothesis Model 1: SNN with Leaky Integrate-and-Fire (LIF) Neurons,
Learnable Membrane Dynamics, and Residual Delta Decoding.
"""
import torch
import torch.nn as nn
from models.snn_layers import get_lif_neuron, SNNTORCH_AVAILABLE

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    pass


class SNNForecaster(nn.Module):
    """
    SNN energy forecasting model with learnable LIF dynamics and residual delta learning.
    """
    def __init__(self, input_dim=11, hidden_dim=64, beta=0.90, threshold=1.0,
                 reset_mechanism="subtract", slope=25.0, prefer_snntorch=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prefer_snntorch = prefer_snntorch

        # Input current projection with normalization to ensure proper spike excitation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        if prefer_snntorch and SNNTORCH_AVAILABLE:
            grad_fn = surrogate.fast_sigmoid(slope=slope)
            self.lif1 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=True,
                                  reset_mechanism=reset_mechanism, spike_grad=grad_fn)
            self.lif2 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=True,
                                  reset_mechanism=reset_mechanism, spike_grad=grad_fn)
        else:
            self.lif1 = get_lif_neuron(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism, slope=slope, prefer_snntorch=False)
            self.lif2 = get_lif_neuron(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism, slope=slope, prefer_snntorch=False)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Delta readout layers: converts integrated spike dynamics into change in power
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
        current_p = x[:, -1, 0:1]  # Residual baseline power at step t

        mem1 = None
        mem2 = None
        spike_accumulator = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        total_spikes_layer1 = 0
        total_spikes_layer2 = 0

        decay = 0.92

        for step in range(seq_len):
            x_t = x[:, step, :]

            # Layer 1
            cur1 = self.norm1(self.fc1(x_t))
            if self.prefer_snntorch and SNNTORCH_AVAILABLE:
                if mem1 is None:
                    mem1 = torch.zeros_like(cur1)
                spk1, mem1 = self.lif1(cur1, mem1)
            else:
                spk1, mem1 = self.lif1(cur1, mem1)
            total_spikes_layer1 += spk1.sum().item()

            # Layer 2
            cur2 = self.norm2(self.fc2(spk1))
            if self.prefer_snntorch and SNNTORCH_AVAILABLE:
                if mem2 is None:
                    mem2 = torch.zeros_like(cur2)
                spk2, mem2 = self.lif2(cur2, mem2)
            else:
                spk2, mem2 = self.lif2(cur2, mem2)
            total_spikes_layer2 += spk2.sum().item()

            # Leaky accumulation of output spikes across the temporal window
            spike_accumulator = decay * spike_accumulator + spk2

        # Delta prediction
        delta = self.delta_head(spike_accumulator)
        energy_pred = current_p + delta
        trend_logits = self.trend_head(spike_accumulator)

        total_neurons = (self.hidden_dim + self.hidden_dim) * batch_size * seq_len
        total_spikes = total_spikes_layer1 + total_spikes_layer2
        spike_rate = total_spikes / max(1, total_neurons)
        synops = (total_spikes_layer1 * self.hidden_dim) + (total_spikes_layer2 * 1)

        return energy_pred, trend_logits, {
            "total_spikes": total_spikes,
            "spike_rate": spike_rate,
            "synops": synops
        }
