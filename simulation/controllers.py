"""
Automation & Optimization Controllers:
Includes Baseline Uncontrolled mode (No Model),
Predictive AI + Linear Programming Optimization (LSTM + LPP, GRU + LPP, SNN + LPP, RSNN + LPP),
Legacy Rule-Based automation, and Manual mode.
"""
from simulation.devices import (
    LightDevice, FanDevice, ACDevice, TVDevice,
    MicrowaveDevice, GeyserDevice, ComputerDevice
)
from simulation.iot_node import IoTModelPredictor, IoTDeviceProfile
from simulation.lpp_optimizer import LPPOptimizer, TariffSchedule
from simulation.multi_model_tracker import MultiModelSavingsTracker


class ControlMode:
    NO_MODEL = "No Model (Baseline)"
    LSTM_LPP = "LSTM + LPP"
    GRU_LPP = "GRU + LPP"
    SNN_LPP = "SNN (LIF) + LPP"
    RSNN_LPP = "RSNN (RLIF) + LPP"
    SMART_RULE = "Smart Rule-Based"
    MANUAL = "Manual Only"

    ALL_MODES = [NO_MODEL, SNN_LPP, RSNN_LPP, LSTM_LPP, GRU_LPP, SMART_RULE, MANUAL]


class BaseController:
    def __init__(self, mode=ControlMode.SNN_LPP):
        self.mode = mode
        self.room_vacancy_timers = {}  # room_id -> seconds vacant
        self.last_sensor_state = {}

        # SNN Event compute metrics
        self.total_spike_events = 0
        self.continuous_compute_steps = 0
        self.event_compute_steps = 0

        # Predictive IoT Node & LPP Optimization
        self.predictor = IoTModelPredictor()
        self.optimizer = LPPOptimizer(grid_capacity_threshold=2600.0)
        self.savings_tracker = MultiModelSavingsTracker()

        # Telemetry timer: sample 15-min intervals (900 sim seconds)
        self.telemetry_timer = 900.0
        self.telemetry_interval = 900.0

        # Active telemetry state
        self.last_predictions = {
            "LSTM": 1200.0,
            "GRU": 1200.0,
            "SNN": 1200.0,
            "RSNN": 1200.0,
            "NO_MODEL": 1200.0
        }
        self.last_trends = {"LSTM": 0, "GRU": 0, "SNN": 0, "RSNN": 0}
        self.last_optimization_result = {
            "actions": ["Initializing Smart Home IoT Energy Optimization..."],
            "total_curtailed_watts": 0.0,
            "device_scalers": {},
            "tariff_rate": 7.50,
            "tariff_tier": "MID_PEAK",
            "peak_target_watts": 2600.0
        }
        self.last_iot_telemetry = {
            "SNN": {"spikes": 208, "ops": 9317, "edge_uj": 0.0084},
            "RSNN": {"spikes": 250, "ops": 24432, "edge_uj": 0.0220},
            "LSTM": {"spikes": 0, "ops": 148000, "edge_uj": 0.6808},
            "GRU": {"spikes": 0, "ops": 142000, "edge_uj": 0.6532},
            "NO_MODEL": {"spikes": 0, "ops": 0, "edge_uj": 0.0}
        }

    def _get_active_model_key(self):
        mapping = {
            ControlMode.SNN_LPP: "SNN",
            ControlMode.RSNN_LPP: "RSNN",
            ControlMode.LSTM_LPP: "LSTM",
            ControlMode.GRU_LPP: "GRU",
            ControlMode.NO_MODEL: "NO_MODEL"
        }
        return mapping.get(self.mode, "SNN")

    def update(self, sim_dt, house, occupants, sim_hour, outdoor_lux, current_day=1):
        """
        Updates controller logic, model predictions, LPP scheduling, and savings tracking.
        Returns the number of spike/compute events generated in this timestep.
        """
        self.continuous_compute_steps += 1
        spikes = 0

        # 1. Track room vacancy
        for room_id, room in house.rooms.items():
            if room.is_occupied:
                self.room_vacancy_timers[room_id] = 0.0
            else:
                self.room_vacancy_timers[room_id] = self.room_vacancy_timers.get(room_id, 0.0) + sim_dt

        # 2. Check for event triggers (SNN Spike Encoding)
        sensor_events = self._detect_sensor_spikes(house, sim_hour, outdoor_lux)
        spikes = len(sensor_events)
        self.total_spike_events += spikes

        # 3. Base Occupant Device Control
        if self.mode == ControlMode.MANUAL:
            return 0
        elif self.mode == ControlMode.NO_MODEL:
            # Baseline system: standard unoptimized usage
            self._apply_baseline_uncontrolled_logic(house, occupants, sim_hour, outdoor_lux)
        else:
            # Smart rule base
            self._apply_control_logic(house, occupants, sim_hour, outdoor_lux, event_driven=(self.mode == ControlMode.SNN_LPP))

        # 4. Periodic 15-Minute Model Forecasting & LPP Optimization
        self.telemetry_timer += sim_dt
        is_telemetry_step = (self.telemetry_timer >= self.telemetry_interval)

        if is_telemetry_step:
            self.telemetry_timer = 0.0
            self._execute_predictive_cycle(house, sim_hour, current_day)

        # 5. Measure instantaneous power for all configurations & update shadow savings tracker
        current_house_pwr = sum(d.current_power for d in house.devices.values())
        baseline_pwr = self._estimate_baseline_power(house, sim_hour, outdoor_lux)

        # Model powers under LPP
        model_power_dict = {}
        for m_key in ["LSTM_LPP", "GRU_LPP", "SNN_LPP", "RSNN_LPP"]:
            opt_curtail = self.last_optimization_result.get("total_curtailed_watts", 0.0)
            if self.mode in [ControlMode.LSTM_LPP, ControlMode.GRU_LPP, ControlMode.SNN_LPP, ControlMode.RSNN_LPP]:
                # If currently in an optimization mode, house power is already optimized
                model_power_dict[m_key] = current_house_pwr
            else:
                # If in baseline mode, simulate what power would be under optimization
                model_power_dict[m_key] = max(150.0, baseline_pwr - opt_curtail)

        # Map telemetry dict
        telem_dict = {
            "LSTM_LPP": self.last_iot_telemetry["LSTM"],
            "GRU_LPP": self.last_iot_telemetry["GRU"],
            "SNN_LPP": self.last_iot_telemetry["SNN"],
            "RSNN_LPP": self.last_iot_telemetry["RSNN"]
        }

        self.savings_tracker.record_step(
            sim_dt=sim_dt,
            baseline_power_watts=baseline_pwr,
            active_model_key=self.mode,
            sim_hour=sim_hour,
            model_power_dict=model_power_dict,
            model_telemetry_dict=telem_dict
        )

        return spikes

    def _execute_predictive_cycle(self, house, sim_hour, current_day):
        """Runs the 15-min forward inference pass and executes LPP optimization."""
        # Calculate categorical power
        total_p = sum(d.current_power for d in house.devices.values())
        kitchen_p = sum(d.current_power for d in house.devices.values() if d.room_id == "kitchen")
        laundry_p = sum(d.current_power for d in house.devices.values() if d.room_id == "bathroom")
        hvac_p = sum(d.current_power for d in house.devices.values() if "ac_" in d.device_id or "fan_" in d.device_id)

        # Record step in IoT node lookback buffer
        self.predictor.append_telemetry_step(
            total_power=total_p,
            kitchen_power=kitchen_p,
            laundry_bath_power=laundry_p,
            hvac_power=hvac_p,
            sim_hour=sim_hour,
            current_day=current_day,
            is_weekend=((current_day - 1) % 7 in [5, 6])
        )

        # Predict with all models
        for m_key in ["LSTM", "GRU", "SNN", "RSNN"]:
            pred_res = self.predictor.predict(m_key)
            self.last_predictions[m_key] = pred_res["predicted_watts"]
            self.last_trends[m_key] = pred_res["trend"]
            self.last_iot_telemetry[m_key] = {
                "spikes": pred_res["spikes"],
                "ops": pred_res["ops"],
                "edge_uj": pred_res["edge_energy_uj"]
            }

        # Active model prediction drives the optimization
        active_key = self._get_active_model_key()
        active_pred = self.last_predictions.get(active_key, total_p)
        active_trend = self.last_trends.get(active_key, 0)
        active_spikes = self.last_iot_telemetry.get(active_key, {}).get("spikes", 0)

        is_lpp_enabled = self.mode in [
            ControlMode.SNN_LPP, ControlMode.RSNN_LPP,
            ControlMode.LSTM_LPP, ControlMode.GRU_LPP
        ]

        # Solve LPP
        opt_res = self.optimizer.optimize(
            house=house,
            predicted_demand_watts=active_pred,
            sim_hour=sim_hour,
            is_peak_shaving_enabled=is_lpp_enabled,
            trend=active_trend,
            model_key=active_key,
            spikes=active_spikes
        )
        self.last_optimization_result = opt_res
        self.active_device_scalers = opt_res.get("device_scalers", {})

        # Apply LPP device setpoints if enabled
        if is_lpp_enabled:
            self.optimizer.apply_to_house(house, self.active_device_scalers)

    def post_physics_update(self, house):
        """
        Enforces LPP load curtailment and appliance setpoints after house physics update.
        Prevents devices from resetting power back to uncontrolled levels during active optimization.
        """
        is_lpp_enabled = self.mode in [
            ControlMode.SNN_LPP, ControlMode.RSNN_LPP,
            ControlMode.LSTM_LPP, ControlMode.GRU_LPP
        ]
        if is_lpp_enabled and hasattr(self, "active_device_scalers") and self.active_device_scalers:
            self.optimizer.apply_to_house(house, self.active_device_scalers)

    def _estimate_baseline_power(self, house, sim_hour, outdoor_lux):
        """Estimates what the house would consume without any LPP optimization."""
        base_power = 0.0
        for d in house.devices.values():
            if not d.is_on:
                base_power += d.standby_power
                continue
            
            # If AC is currently set to eco/setback by LPP, restore full nominal compressor demand
            if hasattr(d, "compressor_on") and hasattr(d, "target_temp"):
                base_power += d.rated_power * 0.85
            elif hasattr(d, "brightness"):
                base_power += d.rated_power  # 100% brightness
            else:
                base_power += d.current_power
        return max(150.0, base_power)

    def _detect_sensor_spikes(self, house, sim_hour, outdoor_lux):
        """Spike Encoder: Generates event spikes on discrete state transitions."""
        spikes = []
        for room_id, room in house.rooms.items():
            last_occ = self.last_sensor_state.get(f"{room_id}_occ", None)
            curr_occ = room.is_occupied
            if last_occ is not None and last_occ != curr_occ:
                spikes.append(("OCCUPANCY_SPIKE", room_id, curr_occ))
            self.last_sensor_state[f"{room_id}_occ"] = curr_occ

            last_temp = self.last_sensor_state.get(f"{room_id}_temp", room.temperature)
            if abs(room.temperature - last_temp) >= 0.5:
                spikes.append(("TEMP_SPIKE", room_id, room.temperature))
                self.last_sensor_state[f"{room_id}_temp"] = room.temperature

        return spikes

    def _apply_baseline_uncontrolled_logic(self, house, occupants, sim_hour, outdoor_lux):
        """
        Uncontrolled Baseline mode:
        Appliances operate at standard, uncurtailed settings without peak shaving or load shifting.
        """
        for room_id, room in house.rooms.items():
            for d in room.devices:
                if d.manual_override:
                    continue

                if isinstance(d, LightDevice):
                    # Lights ON if room occupied or if dark
                    if room.is_occupied or (sim_hour > 18.0 or sim_hour < 6.0):
                        d.set_state(True)
                        d.brightness = 1.0
                    else:
                        d.set_state(False)

                elif isinstance(d, ACDevice):
                    # Baseline AC runs chilled at 21.5°C without energy setback
                    if room.is_occupied or (sim_hour >= 22.0 or sim_hour < 7.0 and room_id == "master_bed"):
                        d.set_state(True)
                        d.target_temp = 21.5
                    else:
                        d.set_state(False)

                elif isinstance(d, GeyserDevice):
                    # Runs full blast in morning
                    if (6.0 <= sim_hour <= 8.5):
                        d.set_state(True)
                        d.current_power = d.rated_power
                    else:
                        d.set_state(False)

                elif isinstance(d, FanDevice):
                    if room.is_occupied:
                        d.set_state(True)
                        d.speed = 4
                    else:
                        d.set_state(False)

                elif isinstance(d, TVDevice):
                    d.set_state(room.is_occupied and (18.0 <= sim_hour <= 23.0))

                elif isinstance(d, ComputerDevice):
                    d.set_state(room.is_occupied and (8.0 <= sim_hour <= 23.0))

    def _apply_control_logic(self, house, occupants, sim_hour, outdoor_lux, event_driven=False):
        """Standard smart rule-based device logic."""
        for room_id, room in house.rooms.items():
            vacant_time = self.room_vacancy_timers.get(room_id, 0.0)

            for d in room.devices:
                # 1. Lighting Control
                if isinstance(d, LightDevice):
                    if not d.manual_override:
                        if room.is_occupied:
                            if outdoor_lux < 15000 or (sim_hour < 6.5 or sim_hour > 18.5):
                                if (room_id in ["master_bed", "study"]) and (sim_hour < 6.0 or sim_hour > 23.5):
                                    d.set_state(False)
                                else:
                                    d.set_state(True)
                            else:
                                d.set_state(False)
                        else:
                            if vacant_time >= 120.0:
                                d.set_state(False)

                # 2. Ceiling Fans
                elif isinstance(d, FanDevice):
                    if not d.manual_override:
                        if room.is_occupied and room.temperature >= 26.5:
                            d.set_state(True)
                            if room.temperature >= 31.0:
                                d.speed = 5
                            elif room.temperature >= 28.5:
                                d.speed = 4
                            else:
                                d.speed = 3
                        else:
                            if vacant_time >= 120.0:
                                d.set_state(False)

                # 3. Air Conditioners
                elif isinstance(d, ACDevice):
                    if not d.manual_override:
                        is_sleeping_time = (sim_hour >= 22.5 or sim_hour < 6.5)
                        if room.is_occupied:
                            if room_id == "master_bed" and is_sleeping_time:
                                d.set_state(True)
                                d.target_temp = 23.5
                            elif room.temperature >= 26.5:
                                d.set_state(True)
                                d.target_temp = 23.5 if room_id == "master_bed" else 24.0
                            elif room.temperature < 22.0 and not is_sleeping_time:
                                d.set_state(False)
                        else:
                            if vacant_time >= 120.0:
                                d.set_state(False)

                # 4. Living Room TV
                elif isinstance(d, TVDevice):
                    if not d.manual_override:
                        if room.is_occupied and (18.0 <= sim_hour <= 22.5):
                            d.set_state(True)
                        else:
                            if vacant_time >= 180.0 or sim_hour < 17.5 or sim_hour > 23.0:
                                d.set_state(False)

                # 5. Study PC
                elif isinstance(d, ComputerDevice):
                    if not d.manual_override:
                        if room.is_occupied and (8.5 <= sim_hour <= 22.5):
                            d.set_state(True)
                        else:
                            if vacant_time >= 300.0:
                                d.set_state(False)

                # 6. Bathroom Geyser
                elif isinstance(d, GeyserDevice):
                    if not d.manual_override:
                        if room.is_occupied and (6.5 <= sim_hour <= 8.2):
                            d.set_state(True)
                        else:
                            if vacant_time >= 60.0 or sim_hour > 9.0:
                                d.set_state(False)

                # 7. Kitchen Microwave
                elif isinstance(d, MicrowaveDevice):
                    if not d.manual_override:
                        if room.is_occupied and not d.is_on:
                            if (7.4 <= sim_hour <= 7.7) or (12.6 <= sim_hour <= 12.9) or (20.4 <= sim_hour <= 20.7):
                                d.start_heating(sim_seconds=180.0)
