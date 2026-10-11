"""
Energy Tracker Module: Calculates live instantaneous power, cumulative energy (kWh),
cost estimation, and maintains historical telemetry buffers.
"""
from collections import deque
from simulation.devices import DeviceCategory


class EnergyTracker:
    """
    Real-time energy tracking and telemetry accumulator.
    """

    def __init__(self, cost_per_kwh=8.50, currency_symbol="Rs. "):
        self.cost_per_kwh = cost_per_kwh
        self.currency_symbol = currency_symbol

        self.current_total_power = 0.0     # Watts
        self.total_joules = 0.0            # Total energy in Joules since start
        self.today_joules = 0.0            # Energy in Joules for the current day
        self.current_day_index = 1         # Current simulation day
        self.yesterday_kwh = 0.0           # Energy consumed on previous completed day
        self.daily_history = {}            # {day_number: total_kwh}
        self.peak_power = 0.0              # Max power observed (Watts)

        # Categorical energy accumulated (in Joules)
        self.category_energy = {
            DeviceCategory.HVAC: 0.0,
            DeviceCategory.LIGHTING: 0.0,
            DeviceCategory.KITCHEN: 0.0,
            DeviceCategory.HEAVY: 0.0,
            DeviceCategory.ELECTRONICS: 0.0,
        }

        # Room-wise energy accumulated (in Joules)
        self.room_energy = {}

        # Rolling history buffer for live graphs: (sim_time_sec, total_watts, spike_events)
        self.history = deque(maxlen=600)  # Stores up to 600 sample points
        self.sample_timer = 0.0
        self.sample_interval = 1.0        # Sample every 1 sim second for fine granularity

    def update(self, sim_dt, sim_time_seconds, devices, spike_count=0, current_day=1):
        """
        Calculates live power draw and integrates cumulative and per-day energy.
        """
        # Day transition detection
        if current_day != self.current_day_index:
            if self.current_day_index > 0 and self.today_joules > 0:
                day_kwh = self.today_joules / 3600000.0
                self.yesterday_kwh = day_kwh
                self.daily_history[self.current_day_index] = day_kwh
            self.today_joules = 0.0
            self.current_day_index = current_day
        instantaneous_power = 0.0
        category_power = {cat: 0.0 for cat in self.category_energy}
        room_power = {}

        for device in devices.values():
            p = device.current_power
            instantaneous_power += p
            
            # Category accumulation
            if device.category in category_power:
                category_power[device.category] += p
            
            # Room accumulation
            room_power[device.room_id] = room_power.get(device.room_id, 0.0) + p

        self.current_total_power = round(instantaneous_power, 1)
        if self.current_total_power > self.peak_power:
            self.peak_power = self.current_total_power

        # Integrate energy: Power (W) * DeltaTime (s) = Joules
        step_joules = instantaneous_power * sim_dt
        self.total_joules += step_joules
        self.today_joules += step_joules

        for cat, p_val in category_power.items():
            self.category_energy[cat] += p_val * sim_dt

        for r_id, p_val in room_power.items():
            self.room_energy[r_id] = self.room_energy.get(r_id, 0.0) + p_val * sim_dt

        # Log history sample
        self.sample_timer += sim_dt
        if self.sample_timer >= self.sample_interval or len(self.history) == 0:
            self.sample_timer = 0.0
            self.history.append((sim_time_seconds, self.current_total_power, spike_count))

    @property
    def total_kwh(self):
        """Total energy consumed since start in kWh."""
        return self.total_joules / 3600000.0

    @property
    def total_cost(self):
        """Total electricity bill cost."""
        return self.total_kwh * self.cost_per_kwh

    @property
    def today_kwh(self):
        """Energy consumed so far in the current day in kWh."""
        return self.today_joules / 3600000.0

    @property
    def today_cost(self):
        """Electricity cost for the current day."""
        return self.today_kwh * self.cost_per_kwh

    @property
    def daily_average_kwh(self):
        """Average kWh per completed day, or current day usage."""
        if self.daily_history:
            return sum(self.daily_history.values()) / len(self.daily_history)
        return self.today_kwh

    def get_category_kwh(self, category):
        """Returns kWh consumed by a specific category."""
        return self.category_energy.get(category, 0.0) / 3600000.0

    def get_room_kwh(self, room_id):
        """Returns kWh consumed by a specific room."""
        return self.room_energy.get(room_id, 0.0) / 3600000.0

    def get_category_percentages(self):
        """Returns percentage breakdown of energy by category."""
        total = self.total_joules
        if total <= 0:
            return {cat: 0.0 for cat in self.category_energy}
        return {
            cat: round((val / total) * 100.0, 1)
            for cat, val in self.category_energy.items()
        }
