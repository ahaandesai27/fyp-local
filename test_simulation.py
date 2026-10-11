"""
Automated Verification Test for Smart Home Simulation.
Tests headless execution across a full 24-hour day, verifying device cycles,
energy calculations, and Pygame rendering without crashing.
"""
import os
import unittest

# Use dummy video driver for headless environment testing
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from simulation.time_manager import TimeManager
from simulation.environment import House
from simulation.occupant import Occupant
from simulation.energy_tracker import EnergyTracker
from simulation.controllers import BaseController, ControlMode
from simulation.devices import WashingMachineDevice, FridgeDevice, ACDevice


class TestSmartHomeSimulation(unittest.TestCase):

    def setUp(self):
        self.time_mgr = TimeManager(initial_hour=6, initial_minute=0)
        self.house = House()
        self.occupants = [
            Occupant("occ_alex", "Alex", color=(59, 130, 246), home_room_id="master_bed", schedule_type="alex"),
            Occupant("occ_sam", "Sam", color=(249, 115, 22), home_room_id="study", schedule_type="sam"),
        ]
        self.energy_tracker = EnergyTracker(cost_per_kwh=8.50, currency_symbol="₹")
        self.controller = BaseController(mode=ControlMode.SMART_RULE)

    def test_24_hour_simulation_cycle(self):
        """Simulates 24 simulated hours in 1-second discrete steps."""
        dt_sim = 10.0  # 10 simulated seconds per step
        steps = int(86400 / dt_sim)  # 8640 steps = 24 hours

        washer_ran = False
        fridge_cooled = False
        initial_kwh = self.energy_tracker.total_kwh

        for _ in range(steps):
            self.time_mgr.sim_time_seconds += dt_sim
            sim_hour = self.time_mgr.time_fraction
            outdoor_temp = self.time_mgr.get_outdoor_temperature()
            outdoor_lux = self.time_mgr.get_outdoor_lux()
            current_day = self.time_mgr.current_day

            # Update occupants
            for occ in self.occupants:
                occ.update(0.016, dt_sim, sim_hour, self.house)

            # Update controller
            spikes = self.controller.update(dt_sim, self.house, self.occupants, sim_hour, outdoor_lux)

            # Update house & devices
            self.house.update(dt_sim, outdoor_temp, outdoor_lux, current_day, sim_hour)

            # Track energy
            self.energy_tracker.update(dt_sim, self.time_mgr.sim_time_seconds, self.house.devices, spikes, current_day=current_day)

            # Check washer activity
            washer = self.house.devices.get("washer_bath")
            if washer and washer.is_on and washer.stage != "idle":
                washer_ran = True

            # Check fridge compressor activity
            fridge = self.house.devices.get("fridge_kitchen")
            if fridge and fridge.compressor_on:
                fridge_cooled = True

        # Assertions
        self.assertTrue(washer_ran, "Washing machine should have run during the 24-hour cycle.")
        self.assertTrue(fridge_cooled, "Fridge compressor should have cycled during the 24-hour cycle.")
        self.assertGreater(self.energy_tracker.total_kwh, initial_kwh, "Energy should accumulate over 24h.")
        self.assertGreater(self.energy_tracker.today_kwh, 0.0, "Today's energy should be tracked.")
        self.assertGreater(self.energy_tracker.peak_power, 1000.0, "Peak power should register heavy appliances.")
        self.assertGreater(self.energy_tracker.total_cost, 0.0, "Total cost should be calculated.")
        
        print(f"\n[Test Result] 24-Hour Energy Consumed: {self.energy_tracker.total_kwh:.3f} kWh")
        print(f"[Test Result] Today's Energy Consumed: {self.energy_tracker.today_kwh:.3f} kWh")
        print(f"[Test Result] 24-Hour Electricity Bill: INR {self.energy_tracker.total_cost:.2f}")
        print(f"[Test Result] Peak Power Demand: {self.energy_tracker.peak_power:.1f} W")

    def test_pygame_rendering_pipeline(self):
        """Tests that Pygame rendering components draw without exceptions."""
        pygame.init()
        screen = pygame.display.set_mode((1440, 860))
        from main import init_fonts
        from ui.floorplan_renderer import FloorplanRenderer
        from ui.graph_renderer import GraphRenderer
        from ui.dashboard_renderer import DashboardRenderer

        fonts = init_fonts()
        floorplan_ui = FloorplanRenderer((16, 16, 920, 560))
        graph_ui = GraphRenderer((16, 588, 920, 256))
        dashboard_ui = DashboardRenderer((950, 16, 474, 828))

        # Run 10 render frames
        for _ in range(10):
            screen.fill((15, 20, 28))
            floorplan_ui.draw(screen, self.house, self.occupants, fonts, 50000.0)
            graph_ui.draw(screen, self.energy_tracker, fonts, mode_name="Test Mode")
            dashboard_ui.draw(screen, self.time_mgr, self.energy_tracker, self.controller, fonts, house=self.house)
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    unittest.main()
