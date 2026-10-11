"""
IoT Device Models with realistic power ratings, duty cycles, and visual states.
"""
import math
import random


class DeviceCategory:
    LIGHTING = "Lighting"
    HVAC = "HVAC / Cooling"
    KITCHEN = "Kitchen Appliances"
    HEAVY = "Heavy Appliances"
    ELECTRONICS = "Electronics / IoT"


class IoTDevice:
    """Base class for all IoT appliances in the smart home."""

    def __init__(self, device_id, name, room_id, category, rated_power, standby_power=1.0, priority="flexible"):
        self.device_id = device_id
        self.name = name
        self.room_id = room_id
        self.category = category
        self.rated_power = rated_power      # Max/nominal active power in Watts
        self.standby_power = standby_power  # Idle/off standby power in Watts
        self.priority = priority            # "critical", "semi-critical", "flexible"
        
        self.is_on = False
        self.manual_override = False        # User manual lock
        self.current_power = standby_power
        self.total_energy_joules = 0.0      # Energy accumulated in Joules
        self.active_time_sim_seconds = 0.0  # Time spent active
        
        # Position offset within room (normalized 0.0 - 1.0 or pixel offset)
        self.rel_pos = (0.5, 0.5)

    def update(self, sim_dt, env_data):
        """
        Updates device physics and calculates instantaneous power for this time step.
        To be overridden by subclasses.
        """
        if self.is_on:
            self.current_power = self.rated_power
            self.active_time_sim_seconds += sim_dt
        else:
            self.current_power = self.standby_power

        # Accumulate energy: Watts * Seconds = Joules
        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power

    def toggle(self):
        """Toggle device ON/OFF with manual override flag."""
        self.is_on = not self.is_on
        self.manual_override = True

    def set_state(self, state, is_manual=False):
        """Set device ON/OFF."""
        self.is_on = bool(state)
        if is_manual:
            self.manual_override = True

    @property
    def total_kwh(self):
        """Total energy consumed in kWh."""
        return self.total_energy_joules / 3600000.0


class LightDevice(IoTDevice):
    """Smart LED Light fixture."""

    def __init__(self, device_id, name, room_id, rated_power=15.0, rel_pos=(0.5, 0.5), color=(255, 245, 200)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.LIGHTING,
            rated_power=rated_power,
            standby_power=0.2,
            priority="flexible"
        )
        self.brightness = 1.0  # 0.0 to 1.0
        self.color = color
        self.rel_pos = rel_pos

    def update(self, sim_dt, env_data):
        if self.is_on and self.brightness > 0:
            self.current_power = max(1.0, self.rated_power * self.brightness)
            self.active_time_sim_seconds += sim_dt
        else:
            self.current_power = self.standby_power

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class FanDevice(IoTDevice):
    """Ceiling Fan with 5-speed regulator and rotation animation."""

    def __init__(self, device_id, name, room_id, rated_power=75.0, rel_pos=(0.5, 0.5)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.HVAC,
            rated_power=rated_power,
            standby_power=0.0,
            priority="semi-critical"
        )
        self.speed = 3         # Speed 1 to 5
        self.angle = 0.0       # Rotation angle for rendering
        self.rel_pos = rel_pos

    def update(self, sim_dt, env_data):
        if self.is_on and self.speed > 0:
            # Power proportional to speed level (Speed 1: ~25W, Speed 3: ~50W, Speed 5: ~75W)
            power_ratio = 0.3 + 0.7 * (self.speed / 5.0)
            self.current_power = self.rated_power * power_ratio
            self.active_time_sim_seconds += sim_dt
            
            # Animate fan spinning (angle in degrees)
            spin_rate = (self.speed * 360.0) # degrees per second in real animation
            self.angle = (self.angle + spin_rate * 0.016) % 360.0
        else:
            self.current_power = self.standby_power

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class ACDevice(IoTDevice):
    """Inverter Split Air Conditioner with thermostat and thermal physics."""

    def __init__(self, device_id, name, room_id, rated_power=1800.0, target_temp=24.0, rel_pos=(0.5, 0.1)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.HVAC,
            rated_power=rated_power,
            standby_power=3.5,
            priority="semi-critical"
        )
        self.target_temp = target_temp
        self.mode = "cool"          # "cool", "fan", "eco"
        self.compressor_on = False
        self.fan_only_power = 38.0  # Fan running without compressor
        self.rel_pos = rel_pos
        self.airflow_particles = [] # For cool air visual effect

    def update(self, sim_dt, env_data):
        room_temp = env_data.get(f"{self.room_id}_temp", 27.0)
        
        if self.is_on:
            self.active_time_sim_seconds += sim_dt
            # Thermostatic control: turn compressor on if room is warmer than target + 0.5°C
            if room_temp > self.target_temp + 0.3:
                self.compressor_on = True
            elif room_temp < self.target_temp - 0.3:
                self.compressor_on = False

            if self.compressor_on:
                # Inverter AC: power scales with temperature difference (delta T)
                delta_t = max(0.5, room_temp - self.target_temp)
                load_factor = min(1.0, 0.45 + 0.12 * delta_t)
                self.current_power = self.fan_only_power + (self.rated_power - self.fan_only_power) * load_factor
            else:
                self.current_power = self.fan_only_power
        else:
            self.compressor_on = False
            self.current_power = self.standby_power

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class FridgeDevice(IoTDevice):
    """Smart Refrigerator with compressor thermal duty cycles (Critical device)."""

    def __init__(self, device_id, name, room_id, rated_power=170.0, rel_pos=(0.85, 0.2)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.KITCHEN,
            rated_power=rated_power,
            standby_power=12.0,  # Base electronics / LED / fan
            priority="critical"
        )
        self.is_on = True       # Fridge is always ON 24/7
        self.internal_temp = 3.8
        self.target_temp = 3.5
        self.compressor_on = False
        self.rel_pos = rel_pos
        self.cycle_timer = random.uniform(0, 600)  # Offset initial cycle

    def update(self, sim_dt, env_data):
        # Fridge internal temperature model
        # Internal temp slowly rises due to insulation heat transfer (~1°C per 40 mins)
        heat_leak_rate = 0.00045 * sim_dt
        self.internal_temp += heat_leak_rate

        # If someone accesses kitchen frequently or ambient temp is high, extra heat
        ambient = env_data.get("outdoor_temp", 28.0)
        self.internal_temp += (ambient - 25.0) * 0.00005 * sim_dt

        # Thermostat hysteresis: 3.0°C to 4.5°C
        if self.internal_temp >= 4.3:
            self.compressor_on = True
        elif self.internal_temp <= 3.2:
            self.compressor_on = False

        if self.is_on:
            if self.compressor_on:
                # Cooling reduces internal temp
                cooling_rate = 0.0018 * sim_dt
                self.internal_temp = max(2.8, self.internal_temp - cooling_rate)
                self.current_power = self.rated_power + random.uniform(-5.0, 5.0)
                self.active_time_sim_seconds += sim_dt
            else:
                self.current_power = self.standby_power
        else:
            self.compressor_on = False
            self.current_power = 0.0

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class WashingMachineDevice(IoTDevice):
    """Smart Washing Machine with daily scheduled cycle stages."""

    CYCLE_STAGES = {
        "idle": (0.0, 2.0),
        "fill_heat": (600.0, 1750.0),    # 10 min water heating
        "wash": (1200.0, 520.0),        # 20 min agitation
        "rinse": (500.0, 420.0),        # ~8 min drain/rinse
        "spin": (400.0, 1450.0),        # ~7 min high-speed spin
    }
    TOTAL_CYCLE_SECONDS = 2700.0  # 45 minutes total

    def __init__(self, device_id, name, room_id, rated_power=1800.0, rel_pos=(0.2, 0.8)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.HEAVY,
            rated_power=rated_power,
            standby_power=2.0,
            priority="flexible"
        )
        self.stage = "idle"
        self.stage_elapsed = 0.0
        self.cycle_elapsed = 0.0
        self.drum_angle = 0.0
        self.has_run_today = False
        self.last_run_day = -1
        self.rel_pos = rel_pos

    def start_cycle(self):
        """Starts a full 45-min washing cycle."""
        self.is_on = True
        self.stage = "fill_heat"
        self.stage_elapsed = 0.0
        self.cycle_elapsed = 0.0

    def update(self, sim_dt, env_data):
        current_day = env_data.get("current_day", 1)
        sim_hour = env_data.get("sim_hour", 10.0)

        # Reset daily run flag on new day
        if current_day != self.last_run_day:
            self.has_run_today = False
            self.last_run_day = current_day

        # Autonomous trigger: schedule laundry once a day around 10:15 AM if not manually run
        if not self.has_run_today and not self.is_on and not self.manual_override:
            if 10.2 <= sim_hour <= 10.4:
                self.start_cycle()
                self.has_run_today = True

        if self.is_on and self.stage != "idle":
            self.cycle_elapsed += sim_dt
            self.stage_elapsed += sim_dt
            self.active_time_sim_seconds += sim_dt

            # Progress through wash stages
            if self.stage == "fill_heat":
                self.current_power = 1650.0 + random.uniform(-40, 40)
                if self.stage_elapsed >= 600.0:
                    self.stage = "wash"
                    self.stage_elapsed = 0.0
                # Slow tumble
                self.drum_angle = (self.drum_angle + 120.0 * 0.016) % 360.0

            elif self.stage == "wash":
                self.current_power = 480.0 + random.uniform(-30, 30)
                if self.stage_elapsed >= 1200.0:
                    self.stage = "rinse"
                    self.stage_elapsed = 0.0
                # Reversing agitation
                tumble_dir = 1 if int(self.stage_elapsed // 4) % 2 == 0 else -1
                self.drum_angle = (self.drum_angle + tumble_dir * 280.0 * 0.016) % 360.0

            elif self.stage == "rinse":
                self.current_power = 410.0 + random.uniform(-20, 20)
                if self.stage_elapsed >= 500.0:
                    self.stage = "spin"
                    self.stage_elapsed = 0.0
                self.drum_angle = (self.drum_angle + 360.0 * 0.016) % 360.0

            elif self.stage == "spin":
                self.current_power = 1350.0 + random.uniform(-50, 50)
                if self.stage_elapsed >= 400.0 or self.cycle_elapsed >= self.TOTAL_CYCLE_SECONDS:
                    self.stage = "idle"
                    self.is_on = False
                    self.has_run_today = True
                # Fast spin
                self.drum_angle = (self.drum_angle + 1200.0 * 0.016) % 360.0
        else:
            self.stage = "idle"
            self.is_on = False
            self.current_power = self.standby_power

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class TVDevice(IoTDevice):
    """Living Room Smart TV."""

    def __init__(self, device_id, name, room_id, rated_power=95.0, rel_pos=(0.5, 0.95)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.ELECTRONICS,
            rated_power=rated_power,
            standby_power=1.2,
            priority="flexible"
        )
        self.rel_pos = rel_pos
        self.flicker_color = (180, 210, 255)

    def update(self, sim_dt, env_data):
        if self.is_on:
            self.current_power = self.rated_power + random.uniform(-4.0, 4.0)
            self.active_time_sim_seconds += sim_dt
            # Subtle visual color shifts simulating TV show scenes
            if random.random() < 0.05:
                r = random.randint(140, 240)
                g = random.randint(170, 250)
                b = random.randint(200, 255)
                self.flicker_color = (r, g, b)
        else:
            self.current_power = self.standby_power

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class MicrowaveDevice(IoTDevice):
    """Kitchen Microwave Oven."""

    def __init__(self, device_id, name, room_id, rated_power=1150.0, rel_pos=(0.3, 0.15)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.KITCHEN,
            rated_power=rated_power,
            standby_power=1.8,
            priority="flexible"
        )
        self.timer_remaining = 0.0
        self.rel_pos = rel_pos

    def start_heating(self, sim_seconds=180.0):
        self.is_on = True
        self.timer_remaining = sim_seconds

    def update(self, sim_dt, env_data):
        if self.is_on and self.timer_remaining > 0:
            self.timer_remaining = max(0.0, self.timer_remaining - sim_dt)
            self.current_power = self.rated_power + random.uniform(-15.0, 15.0)
            self.active_time_sim_seconds += sim_dt
            if self.timer_remaining <= 0:
                self.is_on = False
        else:
            self.is_on = False
            self.current_power = self.standby_power

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class GeyserDevice(IoTDevice):
    """Bathroom Water Heater (Geyser)."""

    def __init__(self, device_id, name, room_id, rated_power=2000.0, rel_pos=(0.85, 0.85)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.HEAVY,
            rated_power=rated_power,
            standby_power=0.0,
            priority="semi-critical"
        )
        self.rel_pos = rel_pos

    def update(self, sim_dt, env_data):
        if self.is_on:
            self.current_power = self.rated_power
            self.active_time_sim_seconds += sim_dt
        else:
            self.current_power = self.standby_power

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class ComputerDevice(IoTDevice):
    """Study PC / Workstation."""

    def __init__(self, device_id, name, room_id, rated_power=110.0, rel_pos=(0.7, 0.3)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.ELECTRONICS,
            rated_power=rated_power,
            standby_power=4.5,
            priority="flexible"
        )
        self.rel_pos = rel_pos

    def update(self, sim_dt, env_data):
        if self.is_on:
            # Active load fluctuates with task
            self.current_power = self.rated_power + random.uniform(-10.0, 15.0)
            self.active_time_sim_seconds += sim_dt
        else:
            self.current_power = self.standby_power

        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power


class RouterDevice(IoTDevice):
    """Smart Home IoT Hub and Wi-Fi Router (Always ON 24/7)."""

    def __init__(self, device_id, name, room_id="foyer", rated_power=12.0, rel_pos=(0.5, 0.5)):
        super().__init__(
            device_id=device_id,
            name=name,
            room_id=room_id,
            category=DeviceCategory.ELECTRONICS,
            rated_power=rated_power,
            standby_power=rated_power,
            priority="critical"
        )
        self.is_on = True
        self.rel_pos = rel_pos

    def update(self, sim_dt, env_data):
        # Always consumes steady ~12W base load
        self.current_power = self.rated_power + random.uniform(-0.5, 0.5)
        self.active_time_sim_seconds += sim_dt
        self.total_energy_joules += self.current_power * sim_dt
        return self.current_power
