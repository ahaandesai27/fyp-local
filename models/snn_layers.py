"""
SNN Layer Modules & Surrogate Gradients:
Provides configurable Leaky Integrate-and-Fire (LIF) and Recurrent LIF (RLIF)
layers using snnTorch and PyTorch autograd.
"""
import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate
    SNNTORCH_AVAILABLE = True
except ImportError:
    SNNTORCH_AVAILABLE = False


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    FastSigmoid surrogate gradient for non-differentiable step function:
    Forward: Heaviside step S = 1 if U >= U_th else 0
    Backward: dS/dU = 1 / (1 + slope * |U - U_th|)^2
    """
    @staticmethod
    def forward(ctx, mem, threshold=1.0, slope=25.0):
        ctx.save_for_backward(mem)
        ctx.threshold = threshold
        ctx.slope = slope
        return (mem >= threshold).to(mem.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        mem, = ctx.saved_tensors
        threshold = ctx.threshold
        slope = ctx.slope
        grad_input = grad_output / (1.0 + slope * torch.abs(mem - threshold)) ** 2
        return grad_input, None, None


class CustomLIFCell(nn.Module):
    """
    Native PyTorch Leaky Integrate-and-Fire (LIF) Neuron implementation:
    U[t] = beta * U[t-1] + (1 - beta) * R * I[t]
    S[t] = 1 if U[t] >= U_th else 0
    """
    def __init__(self, beta=0.85, threshold=1.0, reset_mechanism="subtract", slope=25.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.slope = slope

    def forward(self, input_current, mem):
        if mem is None:
            mem = torch.zeros_like(input_current)

        # Decay membrane potential and integrate input
        mem = self.beta * mem + (1.0 - self.beta) * input_current

        # Fire spike using surrogate gradient
        spike = FastSigmoidSurrogate.apply(mem, self.threshold, self.slope)

        # Reset mechanism
        if self.reset_mechanism == "subtract":
            mem = mem - spike * self.threshold
        else:  # "zero"
            mem = mem * (1.0 - spike)

        return spike, mem


class CustomRLIFCell(nn.Module):
    """
    Recurrent Leaky Integrate-and-Fire (RLIF) Neuron:
    U[t] = beta * U[t-1] + W_in * x[t] + W_rec * S[t-1]
    S[t] = Heaviside(U[t] - U_th)
    """
    def __init__(self, size, beta=0.85, threshold=1.0, reset_mechanism="subtract", slope=25.0):
        super().__init__()
        self.size = size
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.slope = slope
        self.recurrent = nn.Linear(size, size, bias=False)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.2)

    def forward(self, input_current, mem, prev_spike=None):
        if mem is None:
            mem = torch.zeros_like(input_current)
        if prev_spike is None:
            prev_spike = torch.zeros_like(input_current)

        # Recurrent input from previous spikes
        rec_input = self.recurrent(prev_spike)

        # Membrane dynamics with recurrent synaptic feedback
        mem = self.beta * mem + input_current + rec_input

        # Fire spike
        spike = FastSigmoidSurrogate.apply(mem, self.threshold, self.slope)

        # Reset
        if self.reset_mechanism == "subtract":
            mem = mem - spike * self.threshold
        else:
            mem = mem * (1.0 - spike)

        return spike, mem


def get_lif_neuron(beta=0.85, threshold=1.0, reset_mechanism="subtract", slope=25.0, prefer_snntorch=True):
    """
    Returns an instance of LIF neuron (snntorch snn.Leaky if available and requested,
    otherwise CustomLIFCell).
    """
    if prefer_snntorch and SNNTORCH_AVAILABLE:
        grad_fn = surrogate.fast_sigmoid(slope=slope)
        return snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism, spike_grad=grad_fn, learn_beta=False)
    else:
        return CustomLIFCell(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism, slope=slope)


def get_rlif_neuron(size, beta=0.85, threshold=1.0, reset_mechanism="subtract", slope=25.0, prefer_snntorch=True):
    """
    Returns an instance of Recurrent LIF neuron (snntorch snn.RLeaky if available,
    otherwise CustomRLIFCell).
    """
    if prefer_snntorch and SNNTORCH_AVAILABLE:
        grad_fn = surrogate.fast_sigmoid(slope=slope)
        return snn.RLeaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism, spike_grad=grad_fn,
                          linear_features=size, all_to_all=True)
    else:
        return CustomRLIFCell(size=size, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism, slope=slope)
