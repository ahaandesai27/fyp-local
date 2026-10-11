"""
Unit Test: Verifies LSTM, GRU, SNN, and RSNN model forward and backward passes.
"""
import torch
from dataset.data_loader import EnergyDataPipeline
from models.lstm_model import LSTMForecaster
from models.gru_model import GRUForecaster
from models.snn_model import SNNForecaster
from models.rsnn_model import RSNNForecaster


def test_models():
    print("Initializing Data Pipeline...")
    pipeline = EnergyDataPipeline(csv_path="dataset/household_power_15min.csv", lookback=16, horizon=1, batch_size=8)
    sample_batch = next(iter(pipeline.train_loader))
    x = sample_batch["x"]  # (8, 16, 11)
    y = sample_batch["y"]  # (8, 1)

    print(f"Sample batch shape: x={x.shape}, y={y.shape}")

    # 1. Test LSTM
    print("\n--- Testing LSTM Baseline ---")
    lstm = LSTMForecaster(input_dim=11, hidden_dim=32, num_layers=2)
    pred_lstm, trend_lstm = lstm(x)
    loss_lstm = torch.nn.functional.mse_loss(pred_lstm, y)
    loss_lstm.backward()
    print(f"LSTM Output: {pred_lstm.shape}, Loss: {loss_lstm.item():.4f} [OK]")

    # 2. Test GRU
    print("\n--- Testing GRU Baseline ---")
    gru = GRUForecaster(input_dim=11, hidden_dim=32, num_layers=2)
    pred_gru, trend_gru = gru(x)
    loss_gru = torch.nn.functional.mse_loss(pred_gru, y)
    loss_gru.backward()
    print(f"GRU Output: {pred_gru.shape}, Loss: {loss_gru.item():.4f} [OK]")

    # 3. Test SNN (LIF)
    print("\n--- Testing SNN (LIF) Hypothesis ---")
    snn = SNNForecaster(input_dim=11, hidden_dim=32, beta=0.85, threshold=1.0)
    pred_snn, trend_snn, snn_meta = snn(x)
    loss_snn = torch.nn.functional.mse_loss(pred_snn, y)
    loss_snn.backward()
    print(f"SNN Output: {pred_snn.shape}, Spikes: {snn_meta['total_spikes']:.0f}, Spike Rate: {snn_meta['spike_rate']:.3f}, Loss: {loss_snn.item():.4f} [OK]")

    # 4. Test RSNN (Recurrent LIF)
    print("\n--- Testing RSNN (Recurrent LIF) Hypothesis ---")
    rsnn = RSNNForecaster(input_dim=11, hidden_dim=32, beta=0.85, threshold=1.0)
    pred_rsnn, trend_rsnn, rsnn_meta = rsnn(x)
    loss_rsnn = torch.nn.functional.mse_loss(pred_rsnn, y)
    loss_rsnn.backward()
    print(f"RSNN Output: {pred_rsnn.shape}, Spikes: {rsnn_meta['total_spikes']:.0f}, Spike Rate: {rsnn_meta['spike_rate']:.3f}, Loss: {loss_rsnn.item():.4f} [OK]")

    print("\nAll 4 models successfully verified with valid forward and backward passes!")


if __name__ == "__main__":
    test_models()
