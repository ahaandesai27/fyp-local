"""
Linear Programming Problem (LPP) Optimizer for Smart Home Energy Management:
Formulates and solves real-time appliance scheduling, peak demand shaving,
and Time-of-Use (TOU) cost minimization using scipy.optimize.linprog (HiGHS solver).
"""
import numpy as np
from scipy.optimize import linprog


class TariffSchedule:
    """Time-of-Use (TOU) Electricity Pricing Schedule."""
    OFF_PEAK_RATE = 4.50   # Rs./kWh (22:00 - 06:00)
    MID_PEAK_RATE = 7.50   # Rs./kWh (06:00 - 18:00)
    ON_PEAK_RATE = 12.00   # Rs./kWh (18:00 - 22:00)

    @classmethod
    def get_rate(cls, sim_hour):
        if 18.0 <= sim_hour < 22.0:
            return cls.ON_PEAK_RATE, "ON_PEAK"
        elif 6.0 <= sim_hour < 18.0:
            return cls.MID_PEAK_RATE, "MID_PEAK"
        else:
            return cls.OFF_PEAK_RATE, "OFF_PEAK"


class LPPOptimizer:
    """
    Linear Programming Problem (LPP) Optimizer for Home Energy Management Systems (HEMS).
    Translates upcoming forecasted demand from edge models (LSTM, GRU, SNN, RSNN)
    into optimal continuous/discrete device power setpoints and peak-shaving actions.
    """

    def __init__(self, grid_capacity_threshold=2600.0, comfort_tolerance=0.85):
        self.grid_capacity_threshold = grid_capacity_threshold
        self.comfort_tolerance = comfort_tolerance

        # Comfort disruption penalty weights w_i
        # Higher weight = less willing to curtail (AC higher priority than washing)
        self.comfort_weights = {
            "ac_master": 8.5,
            "ac_living": 8.0,
            "geyser_bath": 3.0,
            "washer_bath": 1.5,
            "light_living": 2.0,
            "light_master": 2.0,
            "light_kitchen": 2.0,
            "fan_master": 2.5,
            "fan_living": 2.5,
        }

        # Minimum permissible modulation factors x_i^min
        self.min_bounds = {
            "ac_master": 0.65,      # Allow max 35% curtailment (+1.8°C setpoint rise)
            "ac_living": 0.60,      # Allow max 40% curtailment (+2.0°C setpoint rise)
            "geyser_bath": 0.0,     # Can fully pause heating during critical peak
            "washer_bath": 0.0,     # Can defer washing cycle stage to off-peak
            "light_living": 0.70,   # Dimmable down to 70% brightness
            "light_master": 0.70,
            "light_kitchen": 0.75,
            "fan_master": 0.60,
            "fan_living": 0.60,
        }

    def get_threshold(self, sim_hour):
        """
        Dynamically returns the TOU capacity ceiling, current tariff rate, and tier.
        Applies lower thresholds during expensive daytime and evening peaks to curtail heavy thermal loads.
        """
        tariff_rate, tariff_tier = TariffSchedule.get_rate(sim_hour)
        if tariff_tier == "ON_PEAK":
            return 1800.0, tariff_rate, tariff_tier
        elif tariff_tier == "MID_PEAK":
            return 2000.0, tariff_rate, tariff_tier
        else:
            return self.grid_capacity_threshold, tariff_rate, tariff_tier

    def optimize(self, house, predicted_demand_watts, sim_hour, is_peak_shaving_enabled=True,
                 trend=0, model_key=None, spikes=0):
        """
        Formulates and solves the LPP for the current simulation step.
        Args:
            house: House instance containing rooms and devices
            predicted_demand_watts: Next 15-min demand forecast from model (W)
            sim_hour: Current simulation hour of day (0.0 - 24.0)
            is_peak_shaving_enabled: Boolean flag to enable LPP optimization
            trend: 0 (stable), 1 (rising), 2 (falling)
            model_key: Identifier of active edge model ('LSTM', 'GRU', 'SNN', 'RSNN')
            spikes: Total spikes fired during SNN/RSNN forward inference pass
        Returns:
            dict with:
                - actions: list of string descriptions of control actions taken
                - total_curtailed_watts: instantaneous power reduction (W)
                - device_scalers: dict of {device_id: factor x_i in [min_bound, 1.0]}
                - tariff_rate: current electricity tariff (Rs./kWh)
                - tariff_tier: string ("ON_PEAK", "MID_PEAK", "OFF_PEAK")
                - peak_target_watts: target grid power cap (W)
        """
        cap_threshold, tariff_rate, tariff_tier = self.get_threshold(sim_hour)

        if not is_peak_shaving_enabled:
            return {
                "actions": ["Uncontrolled Baseline (No LPP Optimization)"],
                "total_curtailed_watts": 0.0,
                "device_scalers": {},
                "tariff_rate": tariff_rate,
                "tariff_tier": tariff_tier,
                "peak_target_watts": cap_threshold
            }

        # Identify active controllable devices and base powers
        controllable_devices = []
        base_powers = []
        c_vector = []
        bounds = []
        device_ids = []

        critical_power = 0.0
        controllable_nominal_sum = 0.0

        for d_id, device in house.devices.items():
            if not device.is_on:
                continue

            # Critical appliances (Fridge, Router Hub): non-curtailable
            if device.priority == "critical" or d_id in ["fridge_kitchen", "router_hub"]:
                critical_power += device.current_power
                continue

            # Controllable devices currently demanding power
            if d_id in self.comfort_weights and device.current_power > 5.0:
                p_base = device.current_power
                device_ids.append(d_id)
                controllable_devices.append(device)
                base_powers.append(p_base)
                controllable_nominal_sum += p_base

                # LPP Cost vector:
                # c_i = (Tariff Rate * P_base / 1000) - Comfort_weight * (1.0 / P_base)
                # Minimizing sum(c_i * x_i) balances bill minimization with comfort
                cost_coeff = (tariff_rate * (p_base / 1000.0)) - (self.comfort_weights[d_id] * 0.1)
                c_vector.append(cost_coeff)

                min_b = self.min_bounds.get(d_id, 0.5)
                bounds.append((min_b, 1.0))
            else:
                critical_power += device.current_power

        total_demanded = critical_power + controllable_nominal_sum

        # Check if peak shaving or TOU optimization is required:
        # Triggered if predicted demand exceeds threshold OR during expensive ON_PEAK tariff
        needs_optimization = (
            predicted_demand_watts > cap_threshold or
            total_demanded > cap_threshold or
            tariff_tier == "ON_PEAK"
        )

        if not needs_optimization or len(device_ids) == 0:
            return {
                "actions": ["Nominal Operation (Demand within safe grid limit)"],
                "total_curtailed_watts": 0.0,
                "device_scalers": {d: 1.0 for d in device_ids},
                "tariff_rate": tariff_rate,
                "tariff_tier": tariff_tier,
                "peak_target_watts": cap_threshold
            }

        # Dynamic Peak Target Power:
        # Proactively shave forecasted peak down below grid capacity
        peak_target = min(cap_threshold, max(critical_power + 200.0, predicted_demand_watts * 0.75))
        if tariff_tier == "ON_PEAK":
            peak_target = min(peak_target, 1800.0)

        # LPP Constraint:
        # sum(P_i * x_i) <= peak_target - critical_power
        max_controllable_budget = max(50.0, peak_target - critical_power)

        A_ub = [base_powers]
        b_ub = [max_controllable_budget]

        # Solve Linear Programming Problem using HiGHS solver
        res = linprog(
            c=c_vector,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs"
        )

        actions = []
        device_scalers = {}
        total_curtailed = 0.0

        if res.success:
            optimal_x = res.x
            for idx, d_id in enumerate(device_ids):
                factor = float(optimal_x[idx])
                device_scalers[d_id] = factor
                device = house.devices[d_id]
                p_base = base_powers[idx]
                p_saved = p_base * (1.0 - factor)
                total_curtailed += p_saved

                # Generate descriptive action strings
                if factor < 0.98:
                    if "ac_" in d_id:
                        temp_rise = round((1.0 - factor) * 3.5, 1)
                        actions.append(f"HVAC Setback: {device.name} +{temp_rise}°C (shaved {p_saved:.0f}W)")
                    elif "washer_" in d_id:
                        actions.append(f"Load Shift: Deferred washer heating stage (shaved {p_saved:.0f}W)")
                    elif "geyser_" in d_id:
                        actions.append(f"Peak Shedding: Geyser duty throttled (shaved {p_saved:.0f}W)")
                    elif "light_" in d_id:
                        actions.append(f"Daylight Dimming: {device.name} dimmed to {int(factor*100)}%")
                    elif "fan_" in d_id:
                        actions.append(f"Fan Speed Modulated: {device.name} reduced to {int(factor*100)}%")
        else:
            # Fallback heuristic if strictly infeasible (relax bounds proportionally)
            ratio = max_controllable_budget / max(1.0, controllable_nominal_sum)
            for idx, d_id in enumerate(device_ids):
                min_b = self.min_bounds.get(d_id, 0.6)
                factor = max(min_b, min(1.0, ratio))
                device_scalers[d_id] = factor
                p_saved = base_powers[idx] * (1.0 - factor)
                total_curtailed += p_saved
            actions.append(f"Heuristic Curtailment: Shaved {total_curtailed:.0f}W to meet {peak_target:.0f}W target")

        if not actions:
            actions.append("Optimized: Demanded load comfortably matches peak budget")

        return {
            "actions": actions,
            "total_curtailed_watts": round(total_curtailed, 1),
            "device_scalers": device_scalers,
            "tariff_rate": tariff_rate,
            "tariff_tier": tariff_tier,
            "peak_target_watts": round(peak_target, 1)
        }

    def apply_to_house(self, house, device_scalers):
        """
        Applies the LPP solution scalers to the physical devices in the house.
        """
        for d_id, factor in device_scalers.items():
            device = house.devices.get(d_id)
            if not device or not device.is_on:
                continue

            # AC Thermostat setback and compressor modulation
            if hasattr(device, "target_temp"):
                base_target = 23.5 if "master" in d_id else 24.0
                temp_offset = (1.0 - factor) * 3.5  # Up to +2.3°C setback
                device.target_temp = round(base_target + temp_offset, 1)
                if factor < 0.98 and hasattr(device, "rated_power"):
                    device.current_power = max(device.standby_power, device.current_power * factor)

            # Washing Machine stage deferral
            if hasattr(device, "stage") and device.stage in ["fill_heat", "wash"]:
                if factor < 0.5:
                    # Defer heavy heating stage: revert power to tumble idle
                    device.current_power = 65.0

            # Geyser throttling
            if "geyser" in d_id:
                device.current_power = max(0.0, device.rated_power * factor)

            # Lighting dimming
            if hasattr(device, "brightness"):
                device.brightness = max(0.5, factor)
                device.current_power = max(1.0, device.rated_power * device.brightness)
