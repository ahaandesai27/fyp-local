"""
Frame Renderer: Generates screenshot snapshots of the simulation at various times of day.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from main import init_fonts
from simulation.time_manager import TimeManager
from simulation.environment import House
from simulation.occupant import Occupant
from simulation.energy_tracker import EnergyTracker
from simulation.controllers import BaseController, ControlMode
from ui.floorplan_renderer import FloorplanRenderer
from ui.graph_renderer import GraphRenderer
from ui.dashboard_renderer import DashboardRenderer


def generate_snapshot(target_hour, output_path):
    pygame.init()
    from main import compute_layout
    screen_w, screen_h = 1264, 712
    screen = pygame.Surface((screen_w, screen_h))
    fonts = init_fonts()

    time_mgr = TimeManager(initial_hour=target_hour, initial_minute=15)
    house = House()
    occupants = [
        Occupant("occ_alex", "Alex", color=(59, 130, 246), home_room_id="master_bed", schedule_type="alex"),
        Occupant("occ_sam", "Sam", color=(249, 115, 22), home_room_id="study", schedule_type="sam"),
    ]
    energy_tracker = EnergyTracker(cost_per_kwh=8.50, currency_symbol="Rs. ")
    controller = BaseController(mode=ControlMode.SMART_RULE)

    floorplan_rect, graph_rect, dashboard_rect = compute_layout(screen_w, screen_h)
    floorplan_ui = FloorplanRenderer(floorplan_rect)
    graph_ui = GraphRenderer(graph_rect)
    dashboard_ui = DashboardRenderer(dashboard_rect)

    # Warm up simulation history for 120 steps
    for step in range(120):
        dt = 5.0
        time_mgr.sim_time_seconds += dt
        sim_hour = time_mgr.time_fraction
        outdoor_temp = time_mgr.get_outdoor_temperature()
        outdoor_lux = time_mgr.get_outdoor_lux()
        current_day = time_mgr.current_day

        for occ in occupants:
            occ.update(0.016, dt, sim_hour, house)

        spikes = controller.update(dt, house, occupants, sim_hour, outdoor_lux)
        house.update(dt, outdoor_temp, outdoor_lux, current_day, sim_hour)
        energy_tracker.update(dt, time_mgr.sim_time_seconds, house.devices, spikes, current_day=current_day)

    # Render frame
    screen.fill((15, 20, 28))
    floorplan_ui.draw(screen, house, occupants, fonts, outdoor_lux)
    graph_ui.draw(screen, energy_tracker, fonts, mode_name=controller.mode)
    dashboard_ui.draw(screen, time_mgr, energy_tracker, controller, fonts, house=house)

    pygame.image.save(screen, output_path)
    pygame.quit()
    print(f"Saved snapshot to {output_path}")


if __name__ == "__main__":
    out_dir = r"C:\Users\rahul\.gemini\antigravity\brain\d5578ca2-6bb0-4872-97d4-b5e568e3486f"
    generate_snapshot(10, os.path.join(out_dir, "simulation_morning.png"))
    generate_snapshot(20, os.path.join(out_dir, "simulation_evening.png"))
