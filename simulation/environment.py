"""
Environment Module: Defines the house layout, rooms, thermal physics, and sensor readings.
"""
import math
from simulation.devices import (
    LightDevice, FanDevice, ACDevice, FridgeDevice,
    WashingMachineDevice, TVDevice, MicrowaveDevice,
    GeyserDevice, ComputerDevice, RouterDevice
)


class Room:
    """Represents a room in the smart home."""

    def __init__(self, room_id, name, bounds, floor_type="wood"):
        self.room_id = room_id
        self.name = name
        # Normalized bounding box (x, y, width, height) relative to floorplan rect
        self.bounds = bounds
        self.floor_type = floor_type
        
        # Environmental state
        self.temperature = 26.0   # Current temperature in °C
        self.target_temp = 24.0    # Thermostat setpoint
        self.lux = 50.0            # Interior illuminance in lux
        self.occupants = []        # Occupant objects currently in this room
        self.devices = []          # IoTDevice instances located here
        self.doors = []            # Door positions [(x, y), ...] for movement

    @property
    def is_occupied(self):
        return len(self.occupants) > 0

    def add_device(self, device):
        self.devices.append(device)

    def update_physics(self, sim_dt, outdoor_temp, outdoor_lux):
        """
        Simulates thermal dynamics and lighting levels for this room.
        """
        # --- 1. Thermal Dynamics ---
        # Heat transfer towards outdoor ambient (insulation effect)
        insulation_k = 0.00015  # Room thermal transfer rate
        self.temperature += (outdoor_temp - self.temperature) * insulation_k * sim_dt

        # Heat gain from occupants (~100W body heat per person)
        if self.occupants:
            self.temperature += len(self.occupants) * 0.00008 * sim_dt

        # Heat gain from active electronics
        for d in self.devices:
            if d.is_on:
                if isinstance(d, (MicrowaveDevice, GeyserDevice, TVDevice, ComputerDevice)):
                    self.temperature += (d.current_power / 1000.0) * 0.00005 * sim_dt

        # Cooling from AC
        for d in self.devices:
            if isinstance(d, ACDevice) and d.is_on and d.compressor_on:
                cooling_power = 0.0012 * sim_dt
                if self.temperature > d.target_temp:
                    self.temperature = max(d.target_temp, self.temperature - cooling_power)

        self.temperature = round(self.temperature, 2)

        # --- 2. Lighting / Illuminance ---
        # Natural window light infiltration
        window_transmittance = 0.025  # ~2.5% of outdoor lux enters via windows
        natural_lux = outdoor_lux * window_transmittance

        # Artificial light addition
        artificial_lux = 0.0
        for d in self.devices:
            if isinstance(d, LightDevice) and d.is_on:
                artificial_lux += 250.0 * d.brightness

        self.lux = round(natural_lux + artificial_lux, 1)


class House:
    """Main house container holding rooms, devices, and global layout."""

    def __init__(self):
        self.rooms = {}
        self.devices = {}
        self._build_layout()

    def _build_layout(self):
        # Define 6 realistic rooms in normalized coordinates (0.0 to 1.0)
        # Top row: Foyer (left), Bathroom/Laundry (mid), Kitchen (right)
        # Bottom row: Master Bed (left), Living Room (mid), Study/Bed2 (right)

        # 1. Foyer / Entrance (Top-Left)
        foyer = Room("foyer", "Entrance Foyer", (0.04, 0.04, 0.28, 0.34), floor_type="tile")
        foyer.doors = [(0.32, 0.22)]

        # 2. Bathroom & Laundry (Top-Mid)
        bath = Room("bathroom", "Bath & Laundry", (0.32, 0.04, 0.32, 0.34), floor_type="tile_aqua")
        bath.doors = [(0.48, 0.38)]

        # 3. Kitchen & Dining (Top-Right)
        kitchen = Room("kitchen", "Kitchen & Dining", (0.64, 0.04, 0.32, 0.34), floor_type="tile_warm")
        kitchen.doors = [(0.68, 0.38)]

        # 4. Master Bedroom (Bottom-Left)
        master_bed = Room("master_bed", "Master Bedroom", (0.04, 0.38, 0.30, 0.58), floor_type="wood_dark")
        master_bed.doors = [(0.34, 0.58)]

        # 5. Living Room (Bottom-Mid / Central)
        living = Room("living_room", "Living Room", (0.34, 0.38, 0.34, 0.58), floor_type="wood_light")
        living.doors = [(0.32, 0.22), (0.48, 0.38), (0.68, 0.38), (0.34, 0.58), (0.68, 0.58)]

        # 6. Bedroom 2 / Study (Bottom-Right)
        study = Room("study", "Study / Bedroom 2", (0.68, 0.38, 0.28, 0.58), floor_type="wood_medium")
        study.doors = [(0.68, 0.58)]

        self.rooms = {
            "foyer": foyer,
            "bathroom": bath,
            "kitchen": kitchen,
            "master_bed": master_bed,
            "living_room": living,
            "study": study
        }

        # --- Create IoT Devices ---
        dev_list = [
            # Foyer
            LightDevice("light_foyer", "Foyer Ceiling Light", "foyer", rated_power=10.0, rel_pos=(0.5, 0.5)),
            RouterDevice("router_hub", "Smart IoT Gateway", "foyer", rated_power=12.0, rel_pos=(0.8, 0.25)),

            # Bathroom & Laundry
            LightDevice("light_bath", "Bathroom Vanity Light", "bathroom", rated_power=12.0, rel_pos=(0.4, 0.3)),
            GeyserDevice("geyser_bath", "Storage Geyser (25L)", "bathroom", rated_power=2000.0, rel_pos=(0.82, 0.2)),
            WashingMachineDevice("washer_bath", "Inverter Washing Machine", "bathroom", rated_power=1800.0, rel_pos=(0.22, 0.75)),

            # Kitchen & Dining
            LightDevice("light_kitchen", "Kitchen Recessed Lights", "kitchen", rated_power=18.0, rel_pos=(0.5, 0.4)),
            FridgeDevice("fridge_kitchen", "Double-Door Inverter Refrigerator", "kitchen", rated_power=165.0, rel_pos=(0.82, 0.22)),
            MicrowaveDevice("microwave_kitchen", "Convection Microwave", "kitchen", rated_power=1150.0, rel_pos=(0.25, 0.22)),

            # Master Bedroom
            LightDevice("light_master", "Master Bedroom Chandelier", "master_bed", rated_power=16.0, rel_pos=(0.5, 0.5)),
            FanDevice("fan_master", "Master BLDC Fan", "master_bed", rated_power=65.0, rel_pos=(0.5, 0.52)),
            ACDevice("ac_master", "Master Split AC (1.5T)", "master_bed", rated_power=1750.0, target_temp=23.5, rel_pos=(0.5, 0.08)),

            # Living Room
            LightDevice("light_living", "Living Room Main Light", "living_room", rated_power=22.0, rel_pos=(0.5, 0.45)),
            FanDevice("fan_living", "Living Room Ceiling Fan", "living_room", rated_power=75.0, rel_pos=(0.5, 0.47)),
            ACDevice("ac_living", "Living Dual Inverter AC (2T)", "living_room", rated_power=1950.0, target_temp=24.0, rel_pos=(0.5, 0.08)),
            TVDevice("tv_living", "4K OLED Smart TV (55\")", "living_room", rated_power=95.0, rel_pos=(0.5, 0.94)),

            # Study / Bed 2
            LightDevice("light_study", "Study Desk & Ceiling Light", "study", rated_power=14.0, rel_pos=(0.5, 0.45)),
            FanDevice("fan_study", "Study Room Fan", "study", rated_power=65.0, rel_pos=(0.5, 0.48)),
            ComputerDevice("pc_study", "Workstation PC & Dual Monitors", "study", rated_power=115.0, rel_pos=(0.75, 0.25)),
        ]

        for d in dev_list:
            self.devices[d.device_id] = d
            if d.room_id in self.rooms:
                self.rooms[d.room_id].add_device(d)

    def get_room_for_pos(self, norm_x, norm_y):
        """Returns the Room object containing the given (norm_x, norm_y) or None."""
        for room in self.rooms.values():
            rx, ry, rw, rh = room.bounds
            if rx <= norm_x <= rx + rw and ry <= norm_y <= ry + rh:
                return room
        return None

    def update(self, sim_dt, outdoor_temp, outdoor_lux, current_day, sim_hour):
        """Updates all room physics and attached devices."""
        env_data = {
            "outdoor_temp": outdoor_temp,
            "outdoor_lux": outdoor_lux,
            "current_day": current_day,
            "sim_hour": sim_hour,
        }
        for r_id, room in self.rooms.items():
            env_data[f"{r_id}_temp"] = room.temperature
            env_data[f"{r_id}_lux"] = room.lux

        for room in self.rooms.values():
            room.update_physics(sim_dt, outdoor_temp, outdoor_lux)

        for device in self.devices.values():
            device.update(sim_dt, env_data)
