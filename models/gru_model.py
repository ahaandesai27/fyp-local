"""
Advanced Baseline Control Model 2: GRU with Temporal Attention and Residual Delta Learning.
Significantly improves 15-minute ahead smart-home energy forecasting and peak capture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lstm_model import TemporalAttention


class GRUForecaster(nn.Module):
    """
    Enhanced Multi-layer GRU with Temporal Attention and Residual Delta Connection.
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

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(hidden_dim)

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
        current_p = x[:, -1, 0:1]

        proj = self.input_proj(x)
        gru_out, _ = self.gru(proj)

        last_step = gru_out[:, -1, :]
        context, _ = self.attention(gru_out)
        combined = torch.cat([last_step, context], dim=-1)

        delta = self.delta_regressor(combined)
        energy_pred = current_p + delta
        trend_logits = self.trend_classifier(combined)

        return energy_pred, trend_logits
