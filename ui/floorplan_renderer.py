"""
Floor Plan Renderer: Draws the 2D architectural top-down house floor plan,
furniture, active device visual animations, and occupant agents.
"""
import math
import random
import pygame
from simulation.devices import (
    LightDevice, FanDevice, ACDevice, FridgeDevice,
    WashingMachineDevice, TVDevice, MicrowaveDevice,
    GeyserDevice, ComputerDevice, RouterDevice
)
from ui.widgets import TEXT_PRIMARY, TEXT_SECONDARY, COLOR_ACCENT, COLOR_WARNING, COLOR_SUCCESS


class FloorplanRenderer:
    """Renders the top-view architectural floor plan and dynamic smart devices."""

    # Architectural Color Palette
    WALL_COLOR = (30, 41, 59)          # Outer/inner structural walls
    WALL_BORDER = (15, 23, 42)
    DOORWAY_COLOR = (203, 213, 225)

    FLOOR_COLORS = {
        "wood_dark": (160, 115, 80),    # Master Bedroom
        "wood_light": (210, 175, 135),  # Living Room
        "wood_medium": (185, 140, 95),  # Study
        "tile": (220, 225, 230),        # Foyer slate
        "tile_aqua": (190, 220, 225),   # Bathroom
        "tile_warm": (235, 215, 195),   # Kitchen
    }

    def __init__(self, viewport_rect):
        self.rect = pygame.Rect(viewport_rect)
        self.hovered_device = None
        self.selected_device = None
        self.ac_particles = []          # [(x, y, vx, vy, alpha, life)]
        self.light_surface = None       # Alpha surface for dynamic lighting
        self._init_light_surface()

    def resize(self, new_rect):
        """Updates viewport dimensions and reallocates lighting surfaces."""
        self.rect = pygame.Rect(new_rect)
        self._init_light_surface()

    def _init_light_surface(self):
        self.light_surface = pygame.Surface((max(10, self.rect.width), max(10, self.rect.height)), pygame.SRCALPHA)

    def norm_to_screen(self, norm_x, norm_y):
        """Converts normalized (0.0 - 1.0) house coordinate to screen pixel coordinates."""
        px = self.rect.x + int(norm_x * self.rect.width)
        py = self.rect.y + int(norm_y * self.rect.height)
        return px, py

    def get_room_rect(self, room):
        """Returns pixel pygame.Rect for a room."""
        rx, ry, rw, rh = room.bounds
        x = self.rect.x + int(rx * self.rect.width)
        y = self.rect.y + int(ry * self.rect.height)
        w = int(rw * self.rect.width)
        h = int(rh * self.rect.height)
        return pygame.Rect(x, y, w, h)

    def handle_mouse_motion(self, pos, house):
        """Detects if mouse is hovering over any device or room."""
        self.hovered_device = None
        if not self.rect.collidepoint(pos):
            return None

        # Check devices
        for dev in house.devices.values():
            room = house.rooms.get(dev.room_id)
            if not room:
                continue
            r_rect = self.get_room_rect(room)
            dx = r_rect.x + int(dev.rel_pos[0] * r_rect.width)
            dy = r_rect.y + int(dev.rel_pos[1] * r_rect.height)
            if math.hypot(pos[0] - dx, pos[1] - dy) < 22:
                self.hovered_device = dev
                return dev
        return None

    def handle_click(self, pos, house):
        """Handles user click to toggle or select a device."""
        if not self.rect.collidepoint(pos):
            return None

        for dev in house.devices.values():
            room = house.rooms.get(dev.room_id)
            if not room:
                continue
            r_rect = self.get_room_rect(room)
            dx = r_rect.x + int(dev.rel_pos[0] * r_rect.width)
            dy = r_rect.y + int(dev.rel_pos[1] * r_rect.height)
            if math.hypot(pos[0] - dx, pos[1] - dy) < 22:
                self.selected_device = dev
                dev.toggle()
                return dev
        return None

    def draw(self, surface, house, occupants, fonts, outdoor_lux, controller=None):
        """Main draw call for the floor plan."""
        # 1. House boundary container shadow & background
        pygame.draw.rect(surface, (10, 14, 20), self.rect.inflate(12, 12), border_radius=12)
        pygame.draw.rect(surface, (18, 24, 34), self.rect, border_radius=8)

        # 2. Draw room floors and textures
        for room in house.rooms.values():
            self._draw_room_floor(surface, room)

        # 3. Draw architectural furniture in each room
        for room in house.rooms.values():
            self._draw_room_furniture(surface, room)

        # 4. Draw walls and doorways
        self._draw_walls_and_doors(surface, house)

        # 5. Draw dynamic device lighting overlays
        self._draw_lighting_effects(surface, house, outdoor_lux)

        # Active LPP modulation scalers
        active_scalers = getattr(controller, "active_device_scalers", {}) if controller else {}

        # 6. Draw dynamic devices (fans, AC airflow, washer drum, TV glow)
        for room in house.rooms.values():
            r_rect = self.get_room_rect(room)
            for dev in room.devices:
                dev_factor = active_scalers.get(dev.device_id, 1.0)
                self._draw_device(surface, dev, r_rect, fonts, factor=dev_factor)

        # 7. Draw occupant agents
        for occ in occupants:
            self._draw_occupant(surface, occ, fonts)

        # 8. Draw room labels & live mini sensor badges
        for room in house.rooms.values():
            self._draw_room_label(surface, room, fonts)

        # 9. Draw hover tooltip if any device is hovered
        if self.hovered_device:
            dev_factor = active_scalers.get(self.hovered_device.device_id, 1.0)
            self._draw_device_tooltip(surface, self.hovered_device, house, fonts, factor=dev_factor)

        # 10. Draw prominent AI Pipeline & Model Flow status banner across top
        if controller:
            self._draw_ai_pipeline_banner(surface, controller, fonts)

    def _draw_room_floor(self, surface, room):
        r_rect = self.get_room_rect(room)
        base_color = self.FLOOR_COLORS.get(room.floor_type, (200, 200, 200))
        pygame.draw.rect(surface, base_color, r_rect)

        # Subtle floor plank / tile grid lines
        if "wood" in room.floor_type:
            plank_h = 16
            line_col = (max(0, base_color[0] - 25), max(0, base_color[1] - 25), max(0, base_color[2] - 25))
            for y in range(r_rect.y + plank_h, r_rect.y + r_rect.height, plank_h):
                pygame.draw.line(surface, line_col, (r_rect.x, y), (r_rect.x + r_rect.width, y), 1)
        elif "tile" in room.floor_type:
            tile_size = 28
            line_col = (max(0, base_color[0] - 20), max(0, base_color[1] - 20), max(0, base_color[2] - 20))
            for x in range(r_rect.x + tile_size, r_rect.x + r_rect.width, tile_size):
                pygame.draw.line(surface, line_col, (x, r_rect.y), (x, r_rect.y + r_rect.height), 1)
            for y in range(r_rect.y + tile_size, r_rect.y + r_rect.height, tile_size):
                pygame.draw.line(surface, line_col, (r_rect.x, y), (r_rect.x + r_rect.width, y), 1)

    def _draw_room_furniture(self, surface, room):
        r = self.get_room_rect(room)

        if room.room_id == "master_bed":
            # King Bed (top-left of room)
            bed_w, bed_h = int(r.width * 0.52), int(r.height * 0.46)
            bx, by = r.x + 14, r.y + 24
            # Bed frame & headboard
            pygame.draw.rect(surface, (90, 60, 40), (bx - 4, by - 6, bed_w + 8, 10), border_radius=2)
            pygame.draw.rect(surface, (235, 230, 220), (bx, by, bed_w, bed_h), border_radius=4)
            # Pillows
            pw, ph = int(bed_w * 0.4), 16
            pygame.draw.rect(surface, (255, 255, 255), (bx + 6, by + 4, pw, ph), border_radius=3)
            pygame.draw.rect(surface, (255, 255, 255), (bx + bed_w - pw - 6, by + 4, pw, ph), border_radius=3)
            # Blanket / Duvet
            pygame.draw.rect(surface, (70, 110, 140), (bx, by + 26, bed_w, bed_h - 26), border_radius=3)
            # Wardrobe along bottom wall
            pygame.draw.rect(surface, (110, 75, 45), (r.x + 12, r.y + r.height - 24, r.width - 24, 16), border_radius=2)

        elif room.room_id == "living_room":
            # Area Rug
            rug_rect = pygame.Rect(r.x + int(r.width * 0.15), r.y + int(r.height * 0.25), int(r.width * 0.7), int(r.height * 0.48))
            pygame.draw.rect(surface, (215, 205, 190), rug_rect, border_radius=6)
            pygame.draw.rect(surface, (180, 165, 145), rug_rect, width=2, border_radius=6)

            # L-Shaped Sectional Sofa
            sofa_col = (75, 85, 99)
            # Main sofa back
            pygame.draw.rect(surface, sofa_col, (rug_rect.x + 6, rug_rect.y + 6, rug_rect.width - 12, 28), border_radius=5)
            # L-section return
            pygame.draw.rect(surface, sofa_col, (rug_rect.x + 6, rug_rect.y + 6, 32, rug_rect.height - 12), border_radius=5)
            # Coffee table
            table_rect = pygame.Rect(rug_rect.centerx - 22, rug_rect.centery - 10, 44, 24)
            pygame.draw.rect(surface, (140, 100, 70), table_rect, border_radius=3)
            pygame.draw.rect(surface, (80, 50, 30), table_rect, width=1, border_radius=3)

            # TV Unit along bottom wall
            tv_stand = pygame.Rect(r.centerx - 55, r.y + r.height - 18, 110, 12)
            pygame.draw.rect(surface, (40, 45, 55), tv_stand, border_radius=2)

        elif room.room_id == "kitchen":
            # L-Countertop along top and left
            counter_col = (210, 215, 220)
            c_edge = (160, 165, 170)
            # Top counter
            top_c = pygame.Rect(r.x + 8, r.y + 8, r.width - 16, 26)
            pygame.draw.rect(surface, counter_col, top_c)
            pygame.draw.rect(surface, c_edge, top_c, width=1)
            # Stainless steel sink
            sink_rect = pygame.Rect(r.x + int(r.width * 0.45), r.y + 11, 38, 20)
            pygame.draw.rect(surface, (180, 190, 195), sink_rect, border_radius=3)
            pygame.draw.rect(surface, (130, 140, 145), sink_rect, width=1, border_radius=3)
            # Gas stove (4 burners)
            stove_rect = pygame.Rect(r.x + 14, r.y + 11, 40, 20)
            pygame.draw.rect(surface, (30, 30, 35), stove_rect, border_radius=2)
            for bx_i in [stove_rect.x + 10, stove_rect.x + 30]:
                for by_i in [stove_rect.y + 5, stove_rect.y + 15]:
                    pygame.draw.circle(surface, (180, 180, 180), (bx_i, by_i), 3)

            # Dining Counter / Island with Stools
            island_rect = pygame.Rect(r.x + 14, r.y + int(r.height * 0.62), int(r.width * 0.7), 24)
            pygame.draw.rect(surface, (160, 120, 85), island_rect, border_radius=3)
            # 2 Stools
            for sx in [island_rect.x + 20, island_rect.x + island_rect.width - 20]:
                pygame.draw.circle(surface, (110, 80, 50), (sx, island_rect.bottom + 8), 7)

        elif room.room_id == "bathroom":
            # Glass Shower stall (top-left)
            shower_rect = pygame.Rect(r.x + 8, r.y + 8, int(r.width * 0.45), int(r.height * 0.45))
            pygame.draw.rect(surface, (170, 215, 225), shower_rect, border_radius=3)
            pygame.draw.rect(surface, (100, 160, 180), shower_rect, width=2, border_radius=3)
            # Shower drain
            pygame.draw.circle(surface, (120, 130, 140), shower_rect.center, 4)

            # Washbasin Vanity (top-right)
            vanity_rect = pygame.Rect(r.x + int(r.width * 0.6), r.y + 8, int(r.width * 0.35), 24)
            pygame.draw.rect(surface, (245, 245, 250), vanity_rect, border_radius=3)
            pygame.draw.ellipse(surface, (200, 220, 230), vanity_rect.inflate(-8, -6))

        elif room.room_id == "study":
            # Work Desk & Dual Monitors
            desk_rect = pygame.Rect(r.x + int(r.width * 0.4), r.y + 12, int(r.width * 0.55), 28)
            pygame.draw.rect(surface, (120, 90, 65), desk_rect, border_radius=3)
            # Dual monitors
            m1 = pygame.Rect(desk_rect.x + 8, desk_rect.y + 4, 22, 5)
            m2 = pygame.Rect(desk_rect.x + 34, desk_rect.y + 4, 22, 5)
            pygame.draw.rect(surface, (20, 20, 20), m1)
            pygame.draw.rect(surface, (20, 20, 20), m2)
            # Office Chair
            pygame.draw.circle(surface, (50, 60, 70), (desk_rect.centerx, desk_rect.bottom + 12), 10)

            # Single bed along bottom wall
            s_bed = pygame.Rect(r.x + 10, r.y + r.height - 52, int(r.width * 0.8), 44)
            pygame.draw.rect(surface, (225, 225, 230), s_bed, border_radius=3)
            pygame.draw.rect(surface, (60, 120, 90), (s_bed.x, s_bed.y + 16, s_bed.width, s_bed.height - 16), border_radius=2)

        elif room.room_id == "foyer":
            # Entrance Welcome Mat
            mat_rect = pygame.Rect(r.x + 10, r.y + int(r.height * 0.45), 28, 44)
            pygame.draw.rect(surface, (140, 60, 50), mat_rect, border_radius=3)
            # Shoe rack / console
            console_rect = pygame.Rect(r.x + 10, r.y + 8, r.width - 20, 16)
            pygame.draw.rect(surface, (100, 70, 50), console_rect, border_radius=2)

    def _draw_walls_and_doors(self, surface, house):
        # Draw structural dividing walls
        wall_thick = 6

        # Outer bounding walls
        pygame.draw.rect(surface, self.WALL_COLOR, self.rect, width=wall_thick, border_radius=6)

        # Draw room dividing wall outlines
        for room in house.rooms.values():
            r = self.get_room_rect(room)
            pygame.draw.rect(surface, self.WALL_COLOR, r, width=3)

            # Architectural Door Openings & Swings
            for dx_n, dy_n in room.doors:
                door_px = self.rect.x + int(dx_n * self.rect.width)
                door_py = self.rect.y + int(dy_n * self.rect.height)
                # Door hinge pivot
                pygame.draw.circle(surface, (71, 85, 105), (door_px, door_py), 3)
                # Door panel leaf (subtle wooden angled line)
                pygame.draw.line(surface, (148, 163, 184), (door_px, door_py), (door_px + 8, door_py - 8), 2)

    def _draw_lighting_effects(self, surface, house, outdoor_lux):
        # Clear light surface
        self.light_surface.fill((0, 0, 0, 0))

        # Ambient darkness overlay based on outdoor lux (cool deep blue-black night tint)
        ambient_darkness = max(0, min(165, int(165 * (1.0 - outdoor_lux / 80000.0))))
        if ambient_darkness > 0:
            dark_rect = pygame.Rect(0, 0, self.rect.width, self.rect.height)
            self.light_surface.fill((8, 12, 22, ambient_darkness), dark_rect)

        # Draw soft radial illumination for active lights clipped to each room
        for room in house.rooms.values():
            r = self.get_room_rect(room)
            room_rel_rect = pygame.Rect(r.x - self.rect.x, r.y - self.rect.y, r.width, r.height)

            for d in room.devices:
                if isinstance(d, LightDevice) and d.is_on:
                    lx = r.x - self.rect.x + int(d.rel_pos[0] * r.width)
                    ly = r.y - self.rect.y + int(d.rel_pos[1] * r.height)
                    glow_rad = int(max(r.width, r.height) * 0.95 * d.brightness)
                    
                    # Create soft gradient radial alpha surface
                    glow_surf = pygame.Surface((glow_rad * 2, glow_rad * 2), pygame.SRCALPHA)
                    num_rings = 16
                    for ring_i in range(num_rings, 0, -1):
                        ring_r = int(glow_rad * (ring_i / num_rings))
                        alpha = int(48 * (1.0 - (ring_i / num_rings)))
                        pygame.draw.circle(glow_surf, (255, 245, 180, alpha), (glow_rad, glow_rad), ring_r)

                    self.light_surface.set_clip(room_rel_rect)
                    self.light_surface.blit(glow_surf, (lx - glow_rad, ly - glow_rad), special_flags=pygame.BLEND_RGBA_ADD)
                    self.light_surface.set_clip(None)

        # Blit composite light overlay onto main surface
        surface.blit(self.light_surface, self.rect.topleft)

    def _draw_device(self, surface, dev, r_rect, fonts, factor=1.0):
        dx = r_rect.x + int(dev.rel_pos[0] * r_rect.width)
        dy = r_rect.y + int(dev.rel_pos[1] * r_rect.height)

        # Highlight if selected or hovered
        is_hovered = (self.hovered_device == dev)
        is_selected = (self.selected_device == dev)

        if is_hovered or is_selected:
            pygame.draw.circle(surface, (59, 130, 246, 120), (dx, dy), 18, width=2)

        # 1. Light Device
        if isinstance(dev, LightDevice):
            col = (255, 235, 120) if dev.is_on else (100, 116, 139)
            pygame.draw.circle(surface, (40, 50, 60), (dx, dy), 9)
            pygame.draw.circle(surface, col, (dx, dy), 6)
            if dev.is_on:
                pygame.draw.circle(surface, (255, 255, 255), (dx, dy), 3)

        # 2. Ceiling Fan
        elif isinstance(dev, FanDevice):
            # Center housing
            pygame.draw.circle(surface, (60, 70, 85), (dx, dy), 8)
            # 3 Blades
            rad = 18
            for i in range(3):
                angle_rad = math.radians(dev.angle + i * 120)
                bx = dx + int(rad * math.cos(angle_rad))
                by = dy + int(rad * math.sin(angle_rad))
                b_col = (90, 105, 125) if dev.is_on else (70, 80, 95)
                pygame.draw.line(surface, b_col, (dx, dy), (bx, by), 4)
            pygame.draw.circle(surface, (30, 40, 50), (dx, dy), 4)

        # 3. Split AC
        elif isinstance(dev, ACDevice):
            ac_w, ac_h = 44, 14
            ac_r = pygame.Rect(dx - ac_w // 2, dy - ac_h // 2, ac_w, ac_h)
            ac_col = (240, 245, 250) if dev.is_on else (180, 190, 200)
            pygame.draw.rect(surface, ac_col, ac_r, border_radius=2)
            pygame.draw.rect(surface, (100, 116, 139), ac_r, width=1, border_radius=2)
            
            # AC status LED & air louvers
            led_col = (34, 197, 94) if (dev.is_on and dev.compressor_on) else ((59, 130, 246) if dev.is_on else (150, 150, 150))
            pygame.draw.circle(surface, led_col, (ac_r.right - 6, ac_r.centery), 2)
            pygame.draw.line(surface, (140, 150, 160), (ac_r.x + 4, ac_r.bottom - 3), (ac_r.right - 12, ac_r.bottom - 3), 1)

            # Airflow cool breeze particles
            if dev.is_on and dev.compressor_on:
                for _ in range(2):
                    if len(self.ac_particles) < 25:
                        px = random.randint(ac_r.x + 6, ac_r.right - 6)
                        self.ac_particles.append([px, ac_r.bottom + 2, random.uniform(-0.2, 0.2), random.uniform(0.8, 1.8), 180])

        # 4. Refrigerator
        elif isinstance(dev, FridgeDevice):
            fr_w, fr_h = 32, 28
            fr_r = pygame.Rect(dx - fr_w // 2, dy - fr_h // 2, fr_w, fr_h)
            pygame.draw.rect(surface, (220, 225, 230), fr_r, border_radius=3)
            pygame.draw.rect(surface, (100, 116, 139), fr_r, width=1, border_radius=3)
            # Freezer divider line
            pygame.draw.line(surface, (140, 150, 160), (fr_r.x, fr_r.y + 10), (fr_r.right, fr_r.y + 10), 1)
            # Compressor LED
            c_led = (34, 197, 94) if dev.compressor_on else (100, 116, 139)
            pygame.draw.circle(surface, c_led, (fr_r.right - 5, fr_r.y + 5), 2)

        # 5. Washing Machine
        elif isinstance(dev, WashingMachineDevice):
            wm_w, wm_h = 28, 28
            wm_r = pygame.Rect(dx - wm_w // 2, dy - wm_h // 2, wm_w, wm_h)
            pygame.draw.rect(surface, (235, 240, 245), wm_r, border_radius=3)
            pygame.draw.rect(surface, (100, 116, 139), wm_r, width=1, border_radius=3)
            # Glass porthole drum
            drum_center = (wm_r.centerx, wm_r.centery + 2)
            pygame.draw.circle(surface, (70, 90, 110), drum_center, 8)
            pygame.draw.circle(surface, (140, 190, 220), drum_center, 6)
            # Rotating agitation marker
            if dev.is_on:
                a_rad = math.radians(dev.drum_angle)
                ax = drum_center[0] + int(4 * math.cos(a_rad))
                ay = drum_center[1] + int(4 * math.sin(a_rad))
                pygame.draw.circle(surface, (255, 255, 255), (ax, ay), 2)

        # 6. TV Screen
        elif isinstance(dev, TVDevice):
            tv_w, tv_h = 50, 8
            tv_r = pygame.Rect(dx - tv_w // 2, dy - tv_h // 2, tv_w, tv_h)
            if dev.is_on:
                pygame.draw.rect(surface, dev.flicker_color, tv_r, border_radius=2)
                # TV glow onto floor
                glow_s = pygame.Surface((tv_w + 20, 26), pygame.SRCALPHA)
                glow_s.fill((*dev.flicker_color, 45))
                surface.blit(glow_s, (tv_r.x - 10, tv_r.y - 28))
            else:
                pygame.draw.rect(surface, (20, 25, 30), tv_r, border_radius=2)
            pygame.draw.rect(surface, (70, 80, 95), tv_r, width=1, border_radius=2)

        # 7. Microwave
        elif isinstance(dev, MicrowaveDevice):
            mw_w, mw_h = 24, 16
            mw_r = pygame.Rect(dx - mw_w // 2, dy - mw_h // 2, mw_w, mw_h)
            pygame.draw.rect(surface, (50, 55, 65), mw_r, border_radius=2)
            # Window
            win_col = (255, 220, 100) if dev.is_on else (20, 20, 25)
            pygame.draw.rect(surface, win_col, (mw_r.x + 3, mw_r.y + 3, 14, 10))

        # 8. Geyser
        elif isinstance(dev, GeyserDevice):
            gy_w, gy_h = 16, 20
            gy_r = pygame.Rect(dx - gy_w // 2, dy - gy_h // 2, gy_w, gy_h)
            pygame.draw.rect(surface, (240, 240, 245), gy_r, border_radius=3)
            pygame.draw.rect(surface, (120, 130, 140), gy_r, width=1, border_radius=3)
            # Heating LED
            h_led = (239, 68, 68) if dev.is_on else (100, 116, 139)
            pygame.draw.circle(surface, h_led, (gy_r.centerx, gy_r.bottom - 4), 2)

        # 9. Computer
        elif isinstance(dev, ComputerDevice):
            pc_w, pc_h = 22, 14
            pc_r = pygame.Rect(dx - pc_w // 2, dy - pc_h // 2, pc_w, pc_h)
            screen_col = (59, 130, 246) if dev.is_on else (20, 25, 30)
            pygame.draw.rect(surface, screen_col, pc_r, border_radius=2)
            pygame.draw.rect(surface, (80, 90, 105), pc_r, width=1, border_radius=2)

        # 10. IoT Router
        elif isinstance(dev, RouterDevice):
            pygame.draw.circle(surface, (40, 45, 55), (dx, dy), 7)
            pygame.draw.circle(surface, (34, 197, 94), (dx, dy), 3)

        # 11. Draw LPP curtailment badge if actively modulated by AI optimizer
        if factor < 0.98 and dev.is_on and dev.current_power > 5.0:
            pct_saved = int((1.0 - factor) * 100)
            badge_text = f"⚡ -{pct_saved}%"
            badge_surf = fonts["tiny"].render(badge_text, True, (255, 255, 255))
            bw = badge_surf.get_width() + 8
            bh = badge_surf.get_height() + 4
            bx = dx - bw // 2
            by = dy - 18 if dy - 18 > r_rect.y + 4 else dy + 14
            b_rect = pygame.Rect(bx, by, bw, bh)
            pygame.draw.rect(surface, (16, 185, 129), b_rect, border_radius=3)
            pygame.draw.rect(surface, (255, 255, 255), b_rect, width=1, border_radius=3)
            surface.blit(badge_surf, (bx + 4, by + 2))

    def _draw_occupant(self, surface, occ, fonts):
        if not occ.is_home:
            return

        px, py = self.norm_to_screen(occ.pos[0], occ.pos[1])
        # Slight walking bob
        bob_offset = int(math.sin(occ.walk_phase) * 2) if occ.walk_phase > 0 else 0
        py += bob_offset

        # Drop shadow
        shadow_surf = pygame.Surface((24, 12), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 80), (0, 0, 24, 12))
        surface.blit(shadow_surf, (px - 12, py + 4))

        # Avatar circle
        pygame.draw.circle(surface, occ.color, (px, py), 10)
        pygame.draw.circle(surface, (255, 255, 255), (px, py), 10, width=2)
        # Center core
        pygame.draw.circle(surface, (255, 255, 255), (px, py), 4)

        # Name & activity bubble
        font_sm = fonts["small"]
        tag_text = f"{occ.name}: {occ.current_activity}"
        tag_surf = font_sm.render(tag_text, True, TEXT_PRIMARY)
        tag_rect = tag_surf.get_rect(center=(px, py - 18))

        # Badge pill background
        pill_rect = tag_rect.inflate(10, 4)
        pygame.draw.rect(surface, (15, 23, 42, 220), pill_rect, border_radius=4)
        pygame.draw.rect(surface, occ.color, pill_rect, width=1, border_radius=4)
        surface.blit(tag_surf, tag_rect)

    def _draw_room_label(self, surface, room, fonts):
        r = self.get_room_rect(room)
        font_main = fonts["bold_small"]
        font_sub = fonts["tiny"]

        name_surf = font_main.render(room.name, True, TEXT_PRIMARY)
        sens_surf = font_sub.render(f"{room.temperature:.1f}°C • {int(room.lux)} lx", True, (148, 163, 184))

        bw = max(name_surf.get_width(), sens_surf.get_width()) + 14
        bh = name_surf.get_height() + sens_surf.get_height() + 8
        badge_rect = pygame.Rect(r.x + 8, r.y + 8, bw, bh)

        # Translucent dark pill background
        pill_s = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pill_s.fill((15, 23, 42, 210))
        surface.blit(pill_s, badge_rect.topleft)
        pygame.draw.rect(surface, (51, 65, 85), badge_rect, width=1, border_radius=4)

        surface.blit(name_surf, (r.x + 15, r.y + 11))
        surface.blit(sens_surf, (r.x + 15, r.y + 11 + name_surf.get_height()))

    def _draw_device_tooltip(self, surface, dev, house, fonts, factor=1.0):
        room = house.rooms.get(dev.room_id)
        if not room:
            return

        r = self.get_room_rect(room)
        dx = r.x + int(dev.rel_pos[0] * r.width)
        dy = r.y + int(dev.rel_pos[1] * r.height)

        font_m = fonts["small"]
        font_b = fonts["bold_small"]

        # Info strings
        status_str = "ON" if dev.is_on else "OFF/STANDBY"
        status_col = COLOR_SUCCESS if dev.is_on else (148, 163, 184)
        override_str = " [Manual]" if dev.manual_override else ""

        l1 = f"{dev.name}{override_str}"
        l2 = f"Power: {dev.current_power:.1f} W ({status_str})"
        l3 = f"Total: {dev.total_kwh:.3f} kWh  | Click to toggle"

        s1 = font_b.render(l1, True, TEXT_PRIMARY)
        s2 = font_m.render(l2, True, status_col)
        s3 = font_m.render(l3, True, TEXT_SECONDARY)

        text_surfs = [s1, s2, s3]
        if factor < 0.98 and dev.is_on:
            s4 = font_b.render(f"AI Optimization: Active (-{int((1.0 - factor)*100)}% shaved)", True, (16, 185, 129))
            text_surfs.append(s4)

        w = max(s.get_width() for s in text_surfs) + 16
        h = sum(s.get_height() for s in text_surfs) + 6 + len(text_surfs) * 4

        tx = min(self.rect.right - w - 10, max(self.rect.x + 10, dx - w // 2))
        ty = dy - h - 14 if dy - h - 14 > self.rect.y + 10 else dy + 24

        tip_rect = pygame.Rect(tx, ty, w, h)
        pygame.draw.rect(surface, (15, 23, 42), tip_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_ACCENT, tip_rect, width=1, border_radius=6)

        curr_y = ty + 5
        for s in text_surfs:
            surface.blit(s, (tx + 8, curr_y))
            curr_y += s.get_height() + 3

    def _draw_ai_pipeline_banner(self, surface, controller, fonts):
        """
        Draws an intuitive end-to-end AI Pipeline status banner across the top of the floorplan,
        making it immediately obvious to mentors/examiners where the models enter the simulation.
        """
        active_key = controller._get_active_model_key()
        pred_w = controller.last_predictions.get(active_key, 0.0)
        trend_code = controller.last_trends.get(active_key, 0)
        trend_sym = "▲ Surge" if trend_code == 1 else ("▼ Drop" if trend_code == 2 else "► Stable")
        opt_res = controller.last_optimization_result
        curt_w = opt_res.get("total_curtailed_watts", 0.0)
        cap_w = opt_res.get("peak_target_watts", 2000.0)
        telem = controller.last_iot_telemetry.get(active_key, {})
        spikes = telem.get("spikes", 0)

        bw = self.rect.width - 24
        bh = 26
        bx = self.rect.x + 12
        by = self.rect.y + 8

        b_rect = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(surface, (12, 18, 30), b_rect, border_radius=6)
        border_col = (139, 92, 246) if active_key in ["SNN", "RSNN"] else (59, 130, 246)
        pygame.draw.rect(surface, border_col, b_rect, width=1, border_radius=6)

        font_b = fonts["bold_small"]
        font_t = fonts["tiny"]

        if active_key == "NO_MODEL":
            s1 = font_b.render("AI PIPELINE: [1. Sensors Active]", True, (148, 163, 184))
            s2 = font_t.render("➔ [2. No Edge Model: Uncontrolled Baseline]", True, (239, 68, 68))
            s3 = font_t.render("➔ [3. Grid: Uncapped Load]", True, (245, 158, 11))
            surfs = [s1, s2, s3]
        else:
            model_tag = f"SNN LIF ({spikes} Spikes)" if active_key == "SNN" else (f"RSNN RLIF ({spikes} Spk)" if active_key == "RSNN" else f"{active_key} Gateway")
            model_col = (168, 85, 247) if active_key in ["SNN", "RSNN"] else (59, 130, 246)

            s1 = font_b.render("AI PIPELINE: [1. Sensors]", True, (56, 189, 248))
            s2 = font_t.render(f"➔ [2. Edge AI: {model_tag}]", True, model_col)
            s3 = font_t.render(f"➔ [3. Forecast: {pred_w:.0f}W {trend_sym}]", True, (250, 204, 21))
            s4 = font_t.render(f"➔ [4. LPP Cap: {cap_w:.0f}W]", True, (52, 211, 153))
            s5 = font_b.render(f"➔ [Shaved: {curt_w:.0f}W]" if curt_w > 0 else "➔ [Optimal]", True, (34, 197, 94) if curt_w > 0 else (148, 163, 184))
            surfs = [s1, s2, s3, s4, s5]

        cur_x = bx + 8
        for s in surfs:
            if cur_x + s.get_width() < bx + bw - 4:
                surface.blit(s, (cur_x, by + 5))
                cur_x += s.get_width() + 6
