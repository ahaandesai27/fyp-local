"""
IoT Edge Node & Model Predictor:
Simulates deploying deep learning (LSTM, GRU) and neuromorphic spiking (SNN, RSNN)
energy forecasting models onto edge IoT network microcontrollers and neuromorphic co-processors.
Tracks inference latency, arithmetic operations (MACs vs SynOps), and on-device energy dissipation.
"""
import os
import pickle
import numpy as np
import torch
from collections import deque

from models.lstm_model import LSTMForecaster
from models.gru_model import GRUForecaster
from models.snn_model import SNNForecaster
from models.rsnn_model import RSNNForecaster


class IoTDeviceProfile:
    """
    Simulates hardware physical attributes of the IoT device hosting the model.
    Based on standard edge embedded hardware (ARM Cortex-A53 / Raspberry Pi)
    vs dedicated Neuromorphic Edge Co-processors (Intel Loihi / SynSense Speck / BrainChip Akida).
    """
    PROFILES = {
        "NO_MODEL": {
            "name": "Standard Edge Node (No AI Model)",
            "hardware_type": "Passive Microcontroller",
            "compute_mode": "Dumb Sensor Gateway",
            "energy_per_op_pj": 0.0,
            "idle_power_mw": 50.0,
            "ops_type": "None",
            "inference_time_ms": 0.0,
            "nominal_energy_per_inference_uj": 0.0
        },
        "LSTM": {
            "name": "Edge Gateway (ARM Cortex-A53)",
            "hardware_type": "Standard Edge Microprocessor",
            "compute_mode": "Synchronous Dense FP32 Matrix Mult",
            "energy_per_op_pj": 4.6,       # ~4.6 pJ per 32-bit MAC (Dense ANN)
            "idle_power_mw": 450.0,
            "ops_type": "Dense MACs",
            "nominal_ops": 148000,
            "inference_time_ms": 0.22,
            "nominal_energy_per_inference_uj": 0.681
        },
        "GRU": {
            "name": "Edge Gateway (ARM Cortex-A53)",
            "hardware_type": "Standard Edge Microprocessor",
            "compute_mode": "Synchronous Dense FP32 Matrix Mult",
            "energy_per_op_pj": 4.6,
            "idle_power_mw": 450.0,
            "ops_type": "Dense MACs",
            "nominal_ops": 142000,
            "inference_time_ms": 0.20,
            "nominal_energy_per_inference_uj": 0.653
        },
        "SNN": {
            "name": "Neuromorphic Edge Co-Processor (SynSense / Loihi)",
            "hardware_type": "Sub-milliwatt Neuromorphic Processor",
            "compute_mode": "Asynchronous Event-Driven Spikes (snnTorch LIF)",
            "energy_per_op_pj": 0.9,       # ~0.9 pJ per sparse synaptic addition (SynOp)
            "idle_power_mw": 0.05,        # Ultra-low static leakage in neuromorphic sleep
            "ops_type": "Sparse SynOps",
            "nominal_ops": 9317,
            "inference_time_ms": 0.36,
            "nominal_energy_per_inference_uj": 0.010
        },
        "RSNN": {
            "name": "Recurrent Neuromorphic Processor (Loihi / Akida)",
            "hardware_type": "Recurrent Neuromorphic Processor",
            "compute_mode": "Recurrent Event-Driven Spiking (snnTorch RLIF)",
            "energy_per_op_pj": 0.9,
            "idle_power_mw": 0.08,
            "ops_type": "Sparse SynOps",
            "nominal_ops": 24432,
            "inference_time_ms": 0.43,
            "nominal_energy_per_inference_uj": 0.024
        }
    }

    @classmethod
    def get(cls, model_key):
        return cls.PROFILES.get(model_key, cls.PROFILES["NO_MODEL"])


class IoTModelPredictor:
    """
    Loads trained models (LSTM, GRU, SNN, RSNN) and provides real-time 15-minute ahead
    energy forecasting deployed on simulated smart home IoT network devices.
    """

    def __init__(self, checkpoints_dir="results", lookback_steps=16):
        self.checkpoints_dir = checkpoints_dir
        self.lookback_steps = lookback_steps
        self.device = torch.device("cpu")  # Edge IoT inference runs on CPU/Edge hardware

        # Sliding window buffer: stores past 16 steps of 11-dim feature vectors
        self.history_buffer = deque(maxlen=lookback_steps)

        # Scalers
        self.feature_mean = None
        self.feature_scale = None
        self.target_mean = None
        self.target_scale = None
        self.feature_cols = None
        self._load_scalers()

        # Model instances
        self.models = {}
        self._load_models()

        # Active deployed model
        self.active_model_key = "SNN"  # Default: SNN Hypothesis

        # Telemetry & cumulative IoT device compute stats
        self.total_inferences = 0
        self.cumulative_edge_energy_joules = {
            "LSTM": 0.0,
            "GRU": 0.0,
            "SNN": 0.0,
            "RSNN": 0.0,
            "NO_MODEL": 0.0
        }
        self.last_prediction_watts = 0.0
        self.last_trend = 0  # 0=Stable, 1=Rising, 2=Falling
        self.last_spikes = 0
        self.last_synops_or_macs = 0
        self.last_edge_energy_uj = 0.0

    def _load_scalers(self):
        scalers_path = os.path.join(self.checkpoints_dir, "scalers.pkl")
        if os.path.exists(scalers_path):
            try:
                with open(scalers_path, "rb") as f:
                    data = pickle.load(f)
                self.feature_mean = data["feature_mean"]
                self.feature_scale = data["feature_scale"]
                self.target_mean = data["target_mean"][0]
                self.target_scale = data["target_scale"][0]
                self.feature_cols = data["feature_cols"]
                return
            except Exception as e:
                print(f"[IoT Predictor] Warning: Error loading {scalers_path}: {e}")

        # Fallback values from Kaggle dataset training
        self.target_mean = 1386.76
        self.target_scale = 1220.67
        self.feature_mean = np.array([1386.76, 175.4, 210.8, 480.2, 240.8, 6.05, 130.5, 0.0, 0.0, 3.0, 0.28])
        self.feature_scale = np.array([1220.67, 450.2, 580.4, 620.1, 4.2, 5.3, 110.2, 0.707, 0.707, 2.0, 0.45])

    def _load_models(self):
        """Loads available trained checkpoints into memory."""
        configs = {
            "LSTM": (LSTMForecaster(input_dim=11, hidden_dim=64, num_layers=2), "LSTM_Baseline_Control.pth"),
            "GRU": (GRUForecaster(input_dim=11, hidden_dim=64, num_layers=2), "GRU_Baseline_Control.pth"),
            "SNN": (SNNForecaster(input_dim=11, hidden_dim=64, beta=0.85, threshold=1.0), "SNN_with_LIF_Neurons_Hypothesis.pth"),
            "RSNN": (RSNNForecaster(input_dim=11, hidden_dim=64, beta=0.85, threshold=1.0), "RSNN_with_Recurrent_LIF_Hypothesis.pth")
        }

        for key, (model_inst, ckpt_name) in configs.items():
            ckpt_path = os.path.join(self.checkpoints_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                try:
                    state_dict = torch.load(ckpt_path, map_location=self.device)
                    model_inst.load_state_dict(state_dict)
                    model_inst.eval()
                    self.models[key] = model_inst
                except Exception as e:
                    print(f"[IoT Predictor] Could not load {ckpt_name}: {e}")
            else:
                print(f"[IoT Predictor] Checkpoint {ckpt_path} not found.")

    def set_active_model(self, model_key):
        """Switches the active model deployed on the IoT device."""
        if model_key in ["NO_MODEL", "LSTM", "GRU", "SNN", "RSNN"]:
            self.active_model_key = model_key

    def append_telemetry_step(self, total_power, kitchen_power, laundry_bath_power, hvac_power,
                              sim_hour, current_day, is_weekend, voltage=230.0):
        """
        Records a 15-minute telemetry snapshot into the lookback buffer.
        Maps the 11 feature dimensions matching the Kaggle dataset schema:
        [global_active_power, sub_meter_kitchen, sub_meter_laundry_bath, sub_meter_hvac,
         voltage, global_intensity, global_reactive_power, hour_sin, hour_cos, day_of_week, is_weekend]
        """
        # Derived physical electrical quantities
        intensity = total_power / max(100.0, voltage)  # Amperes
        reactive_power = total_power * 0.32  # Realistic inductive reactive power (VAR)
        
        # Cyclical time features
        hour_rad = 2.0 * np.pi * (sim_hour / 24.0)
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)
        day_of_week = (current_day - 1) % 7
        weekend_flag = 1.0 if is_weekend else 0.0

        feat = np.array([
            total_power,
            kitchen_power,
            laundry_bath_power,
            hvac_power,
            voltage,
            intensity,
            reactive_power,
            hour_sin,
            hour_cos,
            float(day_of_week),
            weekend_flag
        ], dtype=np.float32)

        self.history_buffer.append(feat)

        # Pre-fill buffer with initial estimates if not yet full
        while len(self.history_buffer) < self.lookback_steps:
            self.history_buffer.appendleft(feat.copy())

    def predict(self, model_key=None):
        """
        Executes an inference pass using the specified or currently active model.
        Returns:
            dict containing:
                - predicted_watts: float
                - trend: 0 (stable), 1 (rising), 2 (falling)
                - spikes: int (for SNN/RSNN)
                - ops: int (SynOps or MACs)
                - edge_energy_uj: float (microjoules dissipated by IoT device)
                - hardware_info: dict
        """
        key = model_key if model_key is not None else self.active_model_key
        prof = IoTDeviceProfile.get(key)

        if key == "NO_MODEL" or key not in self.models:
            # Baseline without predictive model: returns current power as naive persistence
            current_w = self.history_buffer[-1][0] if len(self.history_buffer) > 0 else 1000.0
            return {
                "model_key": "NO_MODEL",
                "predicted_watts": current_w,
                "trend": 0,
                "spikes": 0,
                "ops": 0,
                "edge_energy_uj": 0.0,
                "hardware_profile": prof
            }

        model = self.models[key]

        # Prepare normalized tensor [1, Lookback=16, Features=11]
        raw_x = np.array(list(self.history_buffer), dtype=np.float32)  # [16, 11]
        norm_x = (raw_x - self.feature_mean) / np.maximum(self.feature_scale, 1e-5)
        x_tensor = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if key in ["SNN", "RSNN"]:
                pred_scaled, pred_trend, meta = model(x_tensor)
                spikes = int(meta.get("total_spikes", prof.get("nominal_ops", 0) / 45.0))
                ops = int(meta.get("synops", prof.get("nominal_ops", 9317)))
            else:
                pred_scaled, pred_trend = model(x_tensor)
                spikes = 0
                ops = int(prof.get("nominal_ops", 145000))

            trend_idx = int(torch.argmax(pred_trend, dim=1).item())
            scaled_val = pred_scaled.item()

            # Inverse scale to Watts
            pred_watts = float(scaled_val * self.target_scale + self.target_mean)
            pred_watts = max(50.0, pred_watts)

        # Compute edge energy dissipation
        energy_pj = ops * prof["energy_per_op_pj"]
        energy_uj = energy_pj / 1e6

        # Update telemetry
        self.last_prediction_watts = pred_watts
        self.last_trend = trend_idx
        self.last_spikes = spikes
        self.last_synops_or_macs = ops
        self.last_edge_energy_uj = energy_uj
        self.cumulative_edge_energy_joules[key] += (energy_uj / 1e6)
        self.total_inferences += 1

        return {
            "model_key": key,
            "predicted_watts": round(pred_watts, 1),
            "trend": trend_idx,
            "spikes": spikes,
            "ops": ops,
            "edge_energy_uj": round(energy_uj, 4),
            "hardware_profile": prof
        }
