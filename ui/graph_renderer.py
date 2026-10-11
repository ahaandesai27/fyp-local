"""
Graph Renderer: Plots real-time live scrolling power consumption (Watts vs Time)
and SNN event spike markers.
"""
import pygame
from ui.widgets import (
    PANEL_BG, PANEL_HEADER, BORDER_COLOR, TEXT_PRIMARY,
    TEXT_SECONDARY, TEXT_MUTED, COLOR_ACCENT, COLOR_PURPLE, COLOR_WARNING, draw_card
)


class GraphRenderer:
    """Renders real-time power telemetry chart."""

    def __init__(self, rect):
        self.resize(rect)

    def resize(self, new_rect):
        self.rect = pygame.Rect(new_rect)
        self.plot_rect = pygame.Rect(
            self.rect.x + 50,
            self.rect.y + 32,
            max(10, self.rect.width - 65),
            max(10, self.rect.height - 46)
        )

    def draw(self, surface, energy_tracker, fonts, mode_name=""):
        # Draw card container
        title = f"Live Power Demand (Watts vs Time)  —  [{mode_name}]"
        draw_card(surface, self.rect, title=title, header_height=26)

        # Draw chart title text
        font_sm = fonts["small"]
        font_xs = fonts["tiny"]
        font_b = fonts["bold_small"]

        title_surf = font_b.render(title, True, TEXT_PRIMARY)
        surface.blit(title_surf, (self.rect.x + 12, self.rect.y + 5))

        # Plot background
        pygame.draw.rect(surface, (14, 19, 28), self.plot_rect, border_radius=4)
        pygame.draw.rect(surface, BORDER_COLOR, self.plot_rect, width=1, border_radius=4)

        history = list(energy_tracker.history)
        if not history or len(history) < 2:
            no_data = font_sm.render("Collecting live power telemetry...", True, TEXT_MUTED)
            surface.blit(no_data, no_data.get_rect(center=self.plot_rect.center))
            return

        # Determine dynamic Y-axis max scale
        max_val = max(1500.0, max(pt[1] for pt in history))
        # Round up to nearest 500W
        y_max = ((int(max_val) // 500) + 1) * 500

        # Draw horizontal grid lines & Y labels (4 steps)
        steps = 4
        for i in range(steps + 1):
            val = int((y_max / steps) * i)
            y_pos = self.plot_rect.bottom - int((val / y_max) * self.plot_rect.height)
            
            # Grid line
            if 0 < i <= steps:
                pygame.draw.line(surface, (28, 36, 50), (self.plot_rect.x, y_pos), (self.plot_rect.right, y_pos), 1)

            # Y-axis label
            lbl = f"{val}W" if val < 1000 else f"{val/1000:.1f}kW"
            lbl_surf = font_xs.render(lbl, True, TEXT_MUTED)
            surface.blit(lbl_surf, (self.plot_rect.x - lbl_surf.get_width() - 6, y_pos - 6))

        # Draw power curve
        points = []
        spike_dots = []
        n = len(history)
        
        for idx, (t_sec, watts, spikes) in enumerate(history):
            x = self.plot_rect.x + int((idx / (n - 1)) * self.plot_rect.width)
            clamped_watts = max(0.0, min(float(y_max), float(watts)))
            y = self.plot_rect.bottom - int((clamped_watts / y_max) * self.plot_rect.height)
            points.append((x, y))

            if spikes > 0:
                spike_dots.append((x, self.plot_rect.bottom - 4))

        # Draw filled gradient polygon under curve
        if len(points) >= 2:
            poly_points = [(self.plot_rect.x, self.plot_rect.bottom)] + points + [(points[-1][0], self.plot_rect.bottom)]
            fill_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            pygame.draw.polygon(fill_surf, (59, 130, 246, 45), poly_points)
            surface.blit(fill_surf, (0, 0))

            # Draw glowing line
            pygame.draw.lines(surface, COLOR_ACCENT, False, points, 2)

            # Draw current live dot at head
            last_pt = points[-1]
            pygame.draw.circle(surface, (255, 255, 255), last_pt, 4)
            pygame.draw.circle(surface, COLOR_ACCENT, last_pt, 6, width=2)

        # Draw SNN event spikes along bottom
        for sx, sy in spike_dots:
            pygame.draw.circle(surface, COLOR_PURPLE, (sx, sy), 3)

        # Current live value tag on top right of chart
        cur_p = energy_tracker.current_total_power
        val_txt = f"{cur_p:,.0f} W"
        cur_surf = font_b.render(val_txt, True, COLOR_WARNING if cur_p > 2000 else COLOR_ACCENT)
        surface.blit(cur_surf, (self.plot_rect.right - cur_surf.get_width() - 8, self.plot_rect.y + 6))
