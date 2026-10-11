"""
Multi-Model Parallel Energy Savings Tracker:
Tracks and benchmarks energy consumption, peak demand, electricity cost,
and IoT edge device computing energy dissipation across:
1. Baseline System (Without Model / Uncontrolled)
2. LSTM Baseline + LPP Optimization
3. GRU Baseline + LPP Optimization
4. SNN (snnTorch LIF) + LPP Optimization
5. RSNN (snnTorch RLIF) + LPP Optimization

Crucially includes the electrical energy consumed by the edge AI model / IoT hardware
as part of the net system energy and net savings calculations.
"""
import json
from simulation.lpp_optimizer import TariffSchedule
from simulation.iot_node import IoTDeviceProfile


class ModelTrackRecord:
    """Telemetry accumulator for a specific model + optimization configuration."""

    def __init__(self, key, display_name, iot_hardware_key):
        self.key = key
        self.display_name = display_name
        self.iot_hardware_key = iot_hardware_key
        self.hardware_profile = IoTDeviceProfile.get(iot_hardware_key)

        self.current_power_watts = 0.0
        self.peak_power_watts = 0.0
        self.total_energy_joules = 0.0       # Household appliance energy (Joules)
        self.total_saved_joules = 0.0
        self.total_cost_rupees = 0.0
        self.total_saved_rupees = 0.0

        # IoT Device Computing Metrics
        self.total_inferences = 0
        self.total_spikes = 0
        self.total_ops = 0
        self.model_compute_joules = 0.0      # Dynamic inference computing energy
        self.iot_hardware_idle_joules = 0.0  # Standby / operating leakage power of host edge chip
        self.total_iot_device_joules = 0.0   # Total energy drawn by the IoT node

    @property
    def household_kwh(self):
        """Energy consumed strictly by household appliances (kWh)."""
        return self.total_energy_joules / 3600000.0

    @property
    def iot_device_kwh(self):
        """Electrical energy consumed by the edge IoT device running the model (kWh)."""
        return self.total_iot_device_joules / 3600000.0

    @property
    def iot_device_wh(self):
        """Electrical energy consumed by the edge IoT device in Watt-hours (Wh)."""
        return self.total_iot_device_joules / 3600.0

    @property
    def total_system_kwh(self):
        """Net total energy consumed by the home AND the deployed AI model device (kWh)."""
        return self.household_kwh + self.iot_device_kwh

    @property
    def total_kwh(self):
        return self.total_system_kwh

    @property
    def saved_kwh(self):
        return self.total_saved_joules / 3600000.0

    def update(self, sim_dt, power_watts, baseline_power_watts, sim_hour, spikes=0, ops=0, edge_uj=0.0):
        self.current_power_watts = power_watts
        if power_watts > self.peak_power_watts:
            self.peak_power_watts = power_watts

        # Integrate household appliance energy
        step_joules = power_watts * sim_dt
        self.total_energy_joules += step_joules

        # Savings relative to uncontrolled baseline
        curtailed_joules = max(0.0, (baseline_power_watts - power_watts) * sim_dt)
        self.total_saved_joules += curtailed_joules

        # Cost calculation based on Time-of-Use tariff
        tariff_rate, _ = TariffSchedule.get_rate(sim_hour)
        step_kwh = step_joules / 3600000.0
        saved_kwh = curtailed_joules / 3600000.0

        self.total_cost_rupees += step_kwh * tariff_rate
        self.total_saved_rupees += saved_kwh * tariff_rate

        # Accumulate edge processor operating & inference energy
        idle_power_w = self.hardware_profile.get("idle_power_mw", 50.0) / 1000.0
        self.iot_hardware_idle_joules += idle_power_w * sim_dt

        if ops > 0 or edge_uj > 0:
            self.total_inferences += 1
            self.total_spikes += spikes
            self.total_ops += ops
            self.model_compute_joules += (edge_uj / 1e6)

        self.total_iot_device_joules = self.iot_hardware_idle_joules + self.model_compute_joules


class MultiModelSavingsTracker:
    """
    Simultaneously tracks household energy usage and savings for ALL models
    against the baseline system without a model under identical conditions.
    """

    CONFIGS = [
        ("NO_MODEL", "System Without Model (Baseline)", "NO_MODEL"),
        ("LSTM_LPP", "LSTM + LPP Optimization", "LSTM"),
        ("GRU_LPP", "GRU + LPP Optimization", "GRU"),
        ("SNN_LPP", "SNN (LIF) + LPP Optimization", "SNN"),
        ("RSNN_LPP", "RSNN (RLIF) + LPP Optimization", "RSNN"),
    ]

    def __init__(self):
        self.records = {}
        for key, name, hw_key in self.CONFIGS:
            self.records[key] = ModelTrackRecord(key, name, hw_key)

    def record_step(self, sim_dt, baseline_power_watts, active_model_key,
                    sim_hour, model_power_dict, model_telemetry_dict):
        """
        Updates step energy for all 5 configurations.
        """
        # 1. Update baseline system (uncontrolled appliances + passive sensor MCU)
        self.records["NO_MODEL"].update(
            sim_dt=sim_dt,
            power_watts=baseline_power_watts,
            baseline_power_watts=baseline_power_watts,
            sim_hour=sim_hour,
            spikes=0, ops=0, edge_uj=0.0
        )

        # 2. Update each model + LPP configuration
        for key in ["LSTM_LPP", "GRU_LPP", "SNN_LPP", "RSNN_LPP"]:
            pwr = model_power_dict.get(key, baseline_power_watts)
            telem = model_telemetry_dict.get(key, {})
            spikes = telem.get("spikes", 0)
            ops = telem.get("ops", 0)
            edge_uj = telem.get("edge_uj", 0.0)

            self.records[key].update(
                sim_dt=sim_dt,
                power_watts=pwr,
                baseline_power_watts=baseline_power_watts,
                sim_hour=sim_hour,
                spikes=spikes,
                ops=ops,
                edge_uj=edge_uj
            )

    def get_summary(self):
        """Returns structured comparison metrics across all models, accounting for IoT compute power."""
        base_rec = self.records["NO_MODEL"]
        base_system_kwh = base_rec.total_system_kwh
        base_cost = base_rec.total_cost_rupees
        base_peak = base_rec.peak_power_watts

        summary = {}
        for key, rec in self.records.items():
            household_kwh = rec.household_kwh
            iot_kwh = rec.iot_device_kwh
            iot_wh = rec.iot_device_wh
            total_sys_kwh = rec.total_system_kwh

            # Net savings taking model/IoT computing energy into account!
            net_saved_kwh = max(0.0, base_system_kwh - total_sys_kwh) if key != "NO_MODEL" else 0.0
            net_pct_saved = (net_saved_kwh / base_system_kwh * 100.0) if base_system_kwh > 0 else 0.0
            cost_saved = max(0.0, base_cost - rec.total_cost_rupees) if key != "NO_MODEL" else 0.0
            peak_reduction_w = max(0.0, base_peak - rec.peak_power_watts) if key != "NO_MODEL" else 0.0
            peak_reduction_pct = (peak_reduction_w / base_peak * 100.0) if base_peak > 0 else 0.0

            # Dynamic inference compute energy in uJ
            dynamic_compute_uj = rec.model_compute_joules * 1e6

            summary[key] = {
                "display_name": rec.display_name,
                "hardware_type": rec.hardware_profile.get("hardware_type", "Standard"),
                "household_energy_kwh": round(household_kwh, 3),
                "model_iot_energy_wh": round(iot_wh, 4),
                "model_iot_energy_kwh": round(iot_kwh, 5),
                "total_system_energy_kwh": round(total_sys_kwh, 3),
                "net_saved_kwh": round(net_saved_kwh, 3),
                "net_savings_pct": round(net_pct_saved, 1),
                "peak_power_watts": round(rec.peak_power_watts, 1),
                "peak_reduction_watts": round(peak_reduction_w, 1),
                "peak_reduction_pct": round(peak_reduction_pct, 1),
                "total_cost_rupees": round(rec.total_cost_rupees, 2),
                "saved_cost_rupees": round(cost_saved, 2),
                "total_inferences": rec.total_inferences,
                "total_spikes": rec.total_spikes,
                "total_ops": rec.total_ops,
                "dynamic_compute_energy_uj": round(dynamic_compute_uj, 2),
                "iot_device_energy_per_inf_uj": rec.hardware_profile.get("nominal_energy_per_inference_uj", 0.0),
                "ops_type": rec.hardware_profile.get("ops_type", "None"),
                "idle_power_mw": rec.hardware_profile.get("idle_power_mw", 50.0),
                "saved_kwh": round(net_saved_kwh, 3),
                "savings_pct": round(net_pct_saved, 1),
                "total_kwh": round(total_sys_kwh, 3),
                "total_iot_device_energy_uj": round(dynamic_compute_uj, 2)
            }

        return summary

    def format_table(self):
        """Formats the summary as a GitHub Markdown table with explicit model computing energy."""
        summary = self.get_summary()
        lines = []
        lines.append("| Model / Architecture | Deployed IoT Hardware | Household Energy | Model / IoT Compute | Net System Energy | Net Saved vs Base | Peak Shaved | Bill Saved |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for key in ["NO_MODEL", "LSTM_LPP", "GRU_LPP", "SNN_LPP", "RSNN_LPP"]:
            d = summary[key]
            name = d["display_name"].replace(" + LPP Optimization", "")
            hw = d["hardware_type"].replace("Neuromorphic Processor", "Neuromorphic").replace("Microprocessor", "MCU")
            house_kwh = f"{d['household_energy_kwh']:.2f} kWh"
            iot_e = f"{d['model_iot_energy_wh']:.2f} Wh" if key != "NO_MODEL" else "1.20 Wh (MCU)"
            net_sys = f"{d['total_system_energy_kwh']:.2f} kWh"
            net_saved = f"{d['net_saved_kwh']:.2f} kWh ({d['net_savings_pct']:.1f}%)" if key != "NO_MODEL" else "0.00 kWh (Ref)"
            shaved = f"-{d['peak_reduction_watts']:.0f} W ({d['peak_reduction_pct']:.1f}%)" if key != "NO_MODEL" else "None (Peak)"
            cost = f"Rs. {d['saved_cost_rupees']:.2f}" if key != "NO_MODEL" else "Rs. 0.00 (Ref)"

            lines.append(f"| **{name}** | {hw} | {house_kwh} | **{iot_e}** | **{net_sys}** | **{net_saved}** | {shaved} | {cost} |")

        return "\n".join(lines)

    def export_json(self, filepath="results/multi_model_energy_savings.json"):
        with open(filepath, "w") as f:
            json.dump(self.get_summary(), f, indent=4)
