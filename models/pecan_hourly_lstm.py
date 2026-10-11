#!/usr/bin/env python3
"""Train an LSTM baseline on Pecan Street data aggregated to hourly intervals."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from pecan_hourly_common import (
    add_common_arguments, build_loaders, prepare_hourly_homes,
    seed_everything, train_and_save, verify_metadata,
)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, num_homes, hidden_size=64):
        super().__init__()
        self.home_embedding = nn.Embedding(num_homes, 4)
        self.lstm = nn.LSTM(input_size + 4, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))

    def forward(self, x, home_idx):
        home = self.home_embedding(home_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        sequence, _ = self.lstm(torch.cat([x, home], dim=-1))
        return self.head(sequence[:, -1]).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    homes = prepare_hourly_homes(args.csv, args.min_coverage)
    verify_metadata(args.metadata, homes)
    loaders, datasets, feature_scaler, delta_scaler, home_to_idx = build_loaders(homes, args.batch_size, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(len(feature_scaler.mean_), len(home_to_idx), args.hidden_size)
    output_dir = args.output_dir or Path("results/pecan_hourly/lstm")
    train_and_save("LSTM", model, loaders, datasets, feature_scaler,
                   delta_scaler, home_to_idx, args, device, output_dir)


if __name__ == "__main__":
    main()
