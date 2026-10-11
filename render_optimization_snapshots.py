"""
Generates visual snapshots of the enhanced simulation:
1. SNN+LPP active dashboard during peak demand.
2. Multi-Model Comparison Table overlay.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from main import init_fonts, compute_layout
from simulation.time_manager import TimeManager
from simulation.environment import House
from simulation.occupant import Occupant
from simulation.energy_tracker import EnergyTracker
from simulation.controllers import BaseController, ControlMode
from ui.floorplan_renderer import FloorplanRenderer
from ui.graph_renderer import GraphRenderer
from ui.dashboard_renderer import DashboardRenderer


def render_snapshots():
    pygame.init()
    screen_w, screen_h = 1360, 768
    screen = pygame.Surface((screen_w, screen_h))
    fonts = init_fonts()

    time_mgr = TimeManager(initial_hour=19, initial_minute=30)  # Evening On-Peak period
    house = House()
    occupants = [
        Occupant("occ_alex", "Alex", color=(59, 130, 246), home_room_id="living_room", schedule_type="alex"),
        Occupant("occ_sam", "Sam", color=(249, 115, 22), home_room_id="kitchen", schedule_type="sam"),
    ]
    energy_tracker = EnergyTracker(cost_per_kwh=8.50, currency_symbol="Rs. ")
    controller = BaseController(mode=ControlMode.SNN_LPP)

    floorplan_rect, graph_rect, dashboard_rect = compute_layout(screen_w, screen_h)
    floorplan_ui = FloorplanRenderer(floorplan_rect)
    graph_ui = GraphRenderer(graph_rect)
    dashboard_ui = DashboardRenderer(dashboard_rect)

    # Warm up simulation
    for step in range(180):
        dt = 5.0
        time_mgr.sim_time_seconds += dt
        sim_hour = time_mgr.time_fraction
        outdoor_temp = time_mgr.get_outdoor_temperature()
        outdoor_lux = time_mgr.get_outdoor_lux()
        current_day = time_mgr.current_day

        for occ in occupants:
            occ.update(0.016, dt, sim_hour, house)

        spikes = controller.update(dt, house, occupants, sim_hour, outdoor_lux, current_day=current_day)
        house.update(dt, outdoor_temp, outdoor_lux, current_day, sim_hour)
        controller.post_physics_update(house)
        energy_tracker.update(dt, time_mgr.sim_time_seconds, house.devices, spikes, current_day=current_day)

    # Snapshot 1: Active SNN + LPP Dashboard
    screen.fill((15, 20, 28))
    floorplan_ui.draw(screen, house, occupants, fonts, outdoor_lux)
    graph_ui.draw(screen, energy_tracker, fonts, mode_name=controller.mode)
    dashboard_ui.draw(screen, time_mgr, energy_tracker, controller, fonts, house=house)
    
    os.makedirs("results", exist_ok=True)
    snap1_path = "results/simulation_snn_lpp_active.png"
    pygame.image.save(screen, snap1_path)
    print(f"Saved snapshot 1 to {snap1_path}")

    # Snapshot 2: Comparison Modal Overlay
    dashboard_ui.show_comparison_modal = True
    screen.fill((15, 20, 28))
    floorplan_ui.draw(screen, house, occupants, fonts, outdoor_lux)
    graph_ui.draw(screen, energy_tracker, fonts, mode_name=controller.mode)
    dashboard_ui.draw(screen, time_mgr, energy_tracker, controller, fonts, house=house)
    
    snap2_path = "results/simulation_comparison_modal.png"
    pygame.image.save(screen, snap2_path)
    print(f"Saved snapshot 2 to {snap2_path}")

    # Copy to artifact folder
    artifact_dir = r"C:\Users\rahul\.gemini\antigravity\brain\d5578ca2-6bb0-4872-97d4-b5e568e3486f"
    import shutil
    shutil.copy(snap1_path, os.path.join(artifact_dir, "simulation_snn_lpp_active.png"))
    shutil.copy(snap2_path, os.path.join(artifact_dir, "simulation_comparison_modal.png"))
    print("Copied snapshots to artifact directory.")
    pygame.quit()


if __name__ == "__main__":
    render_snapshots()
