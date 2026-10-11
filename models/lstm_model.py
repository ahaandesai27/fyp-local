"""
Advanced Baseline Control Model 1: LSTM with Temporal Attention and Residual Delta Learning.
Significantly improves 15-minute ahead smart-home energy forecasting and peak capture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Computes soft attention weights across all timesteps in the lookback window."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights


class LSTMForecaster(nn.Module):
    """
    Enhanced Multi-layer LSTM with Temporal Attention and Residual Delta Connection.
    """
    def __init__(self, input_dim=11, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(hidden_dim)

        # Predicts the delta from current power P(t)
        self.delta_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.trend_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # The current power at step t is x[:, -1, 0:1]
        current_p = x[:, -1, 0:1]

        proj = self.input_proj(x)
        lstm_out, _ = self.lstm(proj)

        last_step = lstm_out[:, -1, :]
        context, _ = self.attention(lstm_out)
        combined = torch.cat([last_step, context], dim=-1)

        delta = self.delta_regressor(combined)
        # Residual forecasting: P(t+1) = P(t) + delta
        energy_pred = current_p + delta
        trend_logits = self.trend_classifier(combined)

        return energy_pred, trend_logits
