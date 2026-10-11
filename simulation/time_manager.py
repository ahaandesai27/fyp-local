"""
Time Manager Module for Smart Home Simulation.
Handles 24-hour clock, simulation acceleration, and environmental day/night cycles.
"""
import math


class TimeManager:
    """
    Manages simulation time, speed scaling, and time-of-day environmental calculations.
    """

    SPEED_PRESETS = [
        ("Pause", 0),
        ("1x (Real-time)", 1),
        ("10x", 10),
        ("60x (1s=1m)", 60),
        ("300x (1s=5m)", 300),
        ("1200x (1s=20m)", 1200),
    ]

    def __init__(self, initial_hour=6, initial_minute=0):
        # Time in simulated seconds from start of Day 1
        self.sim_time_seconds = (initial_hour * 3600) + (initial_minute * 60)
        self.total_elapsed_sim_seconds = 0.0  # From start of simulation
        self.time_scale = 60.0  # Default to 60x (1 real second = 1 sim minute)
        self.is_paused = False
        self.previous_time_scale = 60.0

    def update(self, real_dt):
        """
        Updates the simulation time based on real frame delta time (in seconds).
        Returns the simulated delta time in seconds.
        """
        if self.is_paused or self.time_scale <= 0:
            return 0.0

        sim_dt = real_dt * self.time_scale
        self.sim_time_seconds += sim_dt
        self.total_elapsed_sim_seconds += sim_dt
        return sim_dt

    def toggle_pause(self):
        """Toggles between paused and running state."""
        self.is_paused = not self.is_paused

    def set_speed(self, scale):
        """Sets the simulation speed multiplier."""
        self.time_scale = max(0.0, float(scale))
        if self.time_scale > 0:
            self.is_paused = False

    def jump_to_hour(self, hour):
        """Jumps the simulation clock to a specific hour of the current day."""
        current_day = int(self.sim_time_seconds // 86400)
        self.sim_time_seconds = (current_day * 86400) + (hour % 24) * 3600

    @property
    def current_day(self):
        """Current simulation day (1-indexed)."""
        return int(self.sim_time_seconds // 86400) + 1

    @property
    def day_seconds(self):
        """Seconds into the current 24-hour day (0 to 86399)."""
        return self.sim_time_seconds % 86400

    @property
    def hours(self):
        """Current hour of day (0-23)."""
        return int(self.day_seconds // 3600)

    @property
    def minutes(self):
        """Current minute of hour (0-59)."""
        return int((self.day_seconds % 3600) // 60)

    @property
    def seconds(self):
        """Current second of minute (0-59)."""
        return int(self.day_seconds % 60)

    @property
    def time_fraction(self):
        """Time of day as a float from 0.0 to 24.0."""
        return (self.day_seconds / 3600.0)

    def get_formatted_time(self):
        """Returns HH:MM:SS string."""
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}"

    def get_time_period_name(self):
        """Returns human-readable name of current time period."""
        h = self.time_fraction
        if 5.0 <= h < 7.0:
            return "Dawn / Sunrise"
        elif 7.0 <= h < 12.0:
            return "Morning"
        elif 12.0 <= h < 16.5:
            return "Afternoon"
        elif 16.5 <= h < 19.5:
            return "Evening / Sunset"
        elif 19.5 <= h < 23.0:
            return "Night"
        else:
            return "Late Night"

    def get_outdoor_daylight(self):
        """
        Calculates outdoor natural daylight factor (0.0 = total dark, 1.0 = peak noon).
        Sunrise around 06:00, Sunset around 19:00, peak at 12:30.
        """
        h = self.time_fraction
        if 5.5 <= h <= 19.0:
            # Daytime sine curve
            progress = (h - 5.5) / (19.0 - 5.5)
            # Smooth bell curve
            return math.sin(progress * math.pi)
        else:
            # Night time ambient
            return 0.0

    def get_outdoor_lux(self):
        """Returns outdoor illuminance in lux (0 lux at night to 80,000 lux at noon)."""
        return self.get_outdoor_daylight() * 80000.0

    def get_outdoor_temperature(self):
        """
        Calculates outdoor ambient temperature (°C) following a diurnal cycle.
        Coolest around 05:00 (~22°C), Peak warmest around 14:30 (~34°C).
        """
        h = self.time_fraction
        # Shift so peak is around 14.5 hr
        angle = ((h - 14.5) / 24.0) * 2.0 * math.pi
        # Temp varies between 22°C and 34°C (mean 28°C, amp 6°C)
        base_temp = 28.0 + 6.0 * math.cos(angle)
        return round(base_temp, 1)
