"""
Smart Home IoT Energy Simulation with 24-Hour Day/Night Physics,
Floor Plan Visualization, SNN Event-Driven Optimization, and Live Telemetry.
Runs in windowed mode maximized to screen work area with dynamic resizing.
"""
import sys
import time
import pygame

from simulation.time_manager import TimeManager
from simulation.environment import House
from simulation.occupant import Occupant
from simulation.energy_tracker import EnergyTracker
from simulation.controllers import BaseController, ControlMode
from simulation.mentor_report import print_final_simulation_report
from ui.floorplan_renderer import FloorplanRenderer
from ui.graph_renderer import GraphRenderer
from ui.dashboard_renderer import DashboardRenderer
from ui.widgets import BG_DARK


def get_max_windowed_size():
    """
    Returns (width, height) for the maximum usable windowed size on the user's desktop,
    respecting the Windows taskbar and window decorations.
    """
    try:
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        # SPI_GETWORKAREA = 0x0030
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            work_w = rect.right - rect.left
            work_h = rect.bottom - rect.top
            # Leave small padding for titlebar and window border
            return max(1024, work_w - 16), max(640, work_h - 40)
    except Exception:
        pass

    # Fallback to pygame display info
    info = pygame.display.Info()
    w = max(1024, info.current_w - 30) if info.current_w > 0 else 1280
    h = max(640, info.current_h - 80) if info.current_h > 0 else 720
    return w, h


def compute_layout(screen_w, screen_h):
    """Calculates responsive viewport rectangles for floorplan, graph, and dashboard."""
    margin = 12
    gap = 12

    # Dashboard on the right
    dash_w = min(460, max(360, int(screen_w * 0.32)))
    dash_x = screen_w - dash_w - margin
    dash_y = margin
    dash_h = screen_h - 2 * margin
    dashboard_rect = pygame.Rect(dash_x, dash_y, dash_w, dash_h)

    # Left Column (Floorplan + Graph)
    left_w = max(400, dash_x - margin - gap)
    left_x = margin

    floor_h = int((screen_h - 2 * margin - gap) * 0.68)
    floor_y = margin
    floorplan_rect = pygame.Rect(left_x, floor_y, left_w, floor_h)

    graph_y = floor_y + floor_h + gap
    graph_h = max(120, screen_h - margin - graph_y)
    graph_rect = pygame.Rect(left_x, graph_y, left_w, graph_h)

    return floorplan_rect, graph_rect, dashboard_rect


def init_fonts():
    """Initializes scalable fonts with clean antialiasing."""
    pygame.font.init()
    
    font_names = ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica", "sans-serif"]
    preferred_font = None
    for name in font_names:
        match = pygame.font.match_font(name)
        if match:
            preferred_font = match
            break

    def load_font(size, bold=False):
        if preferred_font:
            try:
                f = pygame.font.Font(preferred_font, size)
                f.set_bold(bold)
                return f
            except Exception:
                pass
        return pygame.font.SysFont("arial", size, bold=bold)

    return {
        "clock": load_font(26, bold=True),
        "bold_large": load_font(18, bold=True),
        "bold_medium": load_font(14, bold=True),
        "bold_small": load_font(11, bold=True),
        "large": load_font(16),
        "medium": load_font(13),
        "small": load_font(11),
        "tiny": load_font(10),
    }


def main():
    pygame.init()
    pygame.display.set_caption("Smart Home IoT Energy Simulation — SNN Event-Driven Optimization")

    # Determine maximum windowed desktop resolution
    screen_width, screen_height = get_max_windowed_size()
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    fonts = init_fonts()

    # --- Initialize Core Simulation Components ---
    time_mgr = TimeManager(initial_hour=6, initial_minute=0)
    house = House()
    
    occupants = [
        Occupant("occ_alex", "Alex", color=(59, 130, 246), home_room_id="master_bed", schedule_type="alex"),
        Occupant("occ_sam", "Sam", color=(249, 115, 22), home_room_id="study", schedule_type="sam"),
    ]
    
    energy_tracker = EnergyTracker(cost_per_kwh=8.50, currency_symbol="Rs. ")
    controller = BaseController(mode=ControlMode.SNN_LPP)

    # --- UI Layout Geometries ---
    floorplan_rect, graph_rect, dashboard_rect = compute_layout(screen_width, screen_height)

    # UI Renderers
    floorplan_ui = FloorplanRenderer(floorplan_rect)
    graph_ui = GraphRenderer(graph_rect)

    def on_mode_change(new_mode):
        controller.mode = new_mode

    def on_speed_change(new_speed):
        if new_speed == 0:
            time_mgr.is_paused = True
        else:
            time_mgr.set_speed(new_speed)

    def on_jump_time(target_hour):
        time_mgr.jump_to_hour(target_hour)

    dashboard_ui = DashboardRenderer(
        dashboard_rect,
        on_mode_change=on_mode_change,
        on_speed_change=on_speed_change,
        on_jump_time=on_jump_time
    )

    # Record real wall-clock start time for duration & speedup tracking
    start_real_time = time.time()

    # Main Application Loop
    running = True
    try:
        while running:
            real_dt = clock.tick(60) / 1000.0
            real_dt = min(0.1, real_dt)

            # 1. Process Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                elif event.type == pygame.VIDEORESIZE:
                    screen_width, screen_height = max(1000, event.w), max(600, event.h)
                    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                    floorplan_rect, graph_rect, dashboard_rect = compute_layout(screen_width, screen_height)
                    floorplan_ui.resize(floorplan_rect)
                    graph_ui.resize(graph_rect)
                    dashboard_ui.resize(dashboard_rect)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        time_mgr.toggle_pause()
                        dashboard_ui.speed_tabs.selected_index = 0 if time_mgr.is_paused else 3
                        dashboard_ui.cur_speed_idx = dashboard_ui.speed_tabs.selected_index
                    elif event.key == pygame.K_1:
                        time_mgr.set_speed(1)
                        dashboard_ui.speed_tabs.selected_index = 1
                        dashboard_ui.cur_speed_idx = 1
                    elif event.key == pygame.K_2:
                        time_mgr.set_speed(10)
                        dashboard_ui.speed_tabs.selected_index = 2
                        dashboard_ui.cur_speed_idx = 2
                    elif event.key == pygame.K_3:
                        time_mgr.set_speed(60)
                        dashboard_ui.speed_tabs.selected_index = 3
                        dashboard_ui.cur_speed_idx = 3
                    elif event.key == pygame.K_4:
                        time_mgr.set_speed(300)
                        dashboard_ui.speed_tabs.selected_index = 4
                        dashboard_ui.cur_speed_idx = 4
                    elif event.key == pygame.K_5:
                        time_mgr.set_speed(1200)
                        dashboard_ui.speed_tabs.selected_index = 5
                        dashboard_ui.cur_speed_idx = 5
                    elif event.key == pygame.K_m:
                        modes = dashboard_ui.mode_map
                        cur_idx = modes.index(controller.mode) if controller.mode in modes else 0
                        next_idx = (cur_idx + 1) % len(modes)
                        controller.mode = modes[next_idx]
                        dashboard_ui.mode_tabs.selected_index = next_idx
                        dashboard_ui.cur_mode_idx = next_idx
                    elif event.key == pygame.K_c:
                        dashboard_ui.show_comparison_modal = not dashboard_ui.show_comparison_modal
                    elif event.key == pygame.K_h:
                        dashboard_ui.show_mentor_guide = not dashboard_ui.show_mentor_guide
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        break

                dashboard_ui.handle_event(event)

                if event.type == pygame.MOUSEMOTION:
                    floorplan_ui.handle_mouse_motion(event.pos, house)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    floorplan_ui.handle_click(event.pos, house)

            # 2. Update Simulation Physics & Logic
            sim_dt = time_mgr.update(real_dt)
            sim_hour = time_mgr.time_fraction
            outdoor_temp = time_mgr.get_outdoor_temperature()
            outdoor_lux = time_mgr.get_outdoor_lux()
            current_day = time_mgr.current_day

            for occ in occupants:
                occ.update(real_dt, sim_dt, sim_hour, house)

            spike_count = controller.update(sim_dt, house, occupants, sim_hour, outdoor_lux, current_day=current_day)
            house.update(sim_dt, outdoor_temp, outdoor_lux, current_day, sim_hour)
            controller.post_physics_update(house)
            energy_tracker.update(sim_dt, time_mgr.sim_time_seconds, house.devices, spike_count, current_day=current_day)

            # 3. Render Frame
            screen.fill(BG_DARK)

            floorplan_ui.draw(screen, house, occupants, fonts, outdoor_lux, controller=controller)
            graph_ui.draw(screen, energy_tracker, fonts, mode_name=controller.mode)
            dashboard_ui.draw(screen, time_mgr, energy_tracker, controller, fonts, house=house)

            pygame.display.flip()

    finally:
        real_elapsed = time.time() - start_real_time
        print_final_simulation_report(controller, time_mgr, energy_tracker, real_elapsed)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
