#!/usr/bin/env python3
"""Train a surrogate-gradient SNN on hourly Pecan Street load data."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from pecan_hourly_common import (
    add_common_arguments, build_loaders, prepare_hourly_homes,
    seed_everything, train_and_save, verify_metadata,
)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        return grad_output / (1.0 + 5.0 * value.abs()).pow(2)


def spike(value):
    return SurrogateSpike.apply(value)


class SNNForecaster(nn.Module):
    def __init__(self, input_size, num_homes, hidden_size=64, beta=0.9, threshold=1.0):
        super().__init__()
        hidden2 = max(hidden_size // 2, 8)
        self.home_embedding = nn.Embedding(num_homes, 4)
        self.fc1 = nn.Linear(input_size + 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden2)
        self.readout = nn.Linear(hidden2, 1)
        self.head = nn.Linear(1, 1)
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, home_idx):
        batch = x.size(0)
        home = self.home_embedding(home_idx)
        mem1 = x.new_zeros((batch, self.fc1.out_features))
        mem2 = x.new_zeros((batch, self.fc2.out_features))
        readout_mem = x.new_zeros((batch, 1))
        for t in range(x.size(1)):
            current_input = torch.cat([x[:, t], home], dim=-1)
            current1 = self.fc1(current_input)
            mem1 = self.beta * mem1 + current1
            spk1 = spike(mem1 - self.threshold)
            mem1 = mem1 - spk1 * self.threshold

            current2 = self.fc2(spk1)
            mem2 = self.beta * mem2 + current2
            spk2 = spike(mem2 - self.threshold)
            mem2 = mem2 - spk2 * self.threshold

            readout_mem = self.beta * readout_mem + self.readout(spk2)
        return self.head(readout_mem).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    homes = prepare_hourly_homes(args.csv, args.min_coverage)
    verify_metadata(args.metadata, homes)
    loaders, datasets, feature_scaler, delta_scaler, home_to_idx = build_loaders(homes, args.batch_size, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNForecaster(len(feature_scaler.mean_), len(home_to_idx), args.hidden_size)
    output_dir = args.output_dir or Path("results/pecan_hourly/snn")
    train_and_save("SNN", model, loaders, datasets, feature_scaler,
                   delta_scaler, home_to_idx, args, device, output_dir)


if __name__ == "__main__":
    main()
