"""
Reusable UI Widgets: Buttons, Cards, Progress Bars, Tabs, and Tooltips.
"""
import pygame


# UI Color Palette (Modern Dark Slate Theme)
BG_DARK = (15, 20, 28)
PANEL_BG = (23, 31, 44)
PANEL_HEADER = (30, 41, 59)
BORDER_COLOR = (45, 59, 82)
BORDER_HIGHLIGHT = (79, 70, 229)

TEXT_PRIMARY = (241, 245, 249)
TEXT_SECONDARY = (148, 163, 184)
TEXT_MUTED = (100, 116, 139)

COLOR_ACCENT = (59, 130, 246)      # Electric Blue
COLOR_SUCCESS = (34, 197, 94)      # Emerald Green
COLOR_WARNING = (245, 158, 11)     # Amber Yellow
COLOR_DANGER = (239, 68, 68)       # Coral Red
COLOR_PURPLE = (168, 85, 247)      # Violet SNN


class Button:
    """Interactive button with hover, active, and toggle states."""

    def __init__(self, rect, text, callback=None, icon=None, bg_color=PANEL_HEADER,
                 hover_color=(51, 65, 85), active_color=COLOR_ACCENT, text_color=TEXT_PRIMARY,
                 border_color=BORDER_COLOR, border_radius=6, is_toggle=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.icon = icon
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.active_color = active_color
        self.text_color = text_color
        self.border_color = border_color
        self.border_radius = border_radius
        self.is_toggle = is_toggle
        self.is_active = False
        self.is_hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.is_toggle:
                    self.is_active = not self.is_active
                if self.callback:
                    self.callback()
                return True
        return False

    def draw(self, surface, font):
        # Determine background color
        if self.is_active:
            bg = self.active_color
            border = COLOR_ACCENT
            txt_col = (255, 255, 255)
        elif self.is_hovered:
            bg = self.hover_color
            border = (100, 116, 139)
            txt_col = TEXT_PRIMARY
        else:
            bg = self.bg_color
            border = self.border_color
            txt_col = self.text_color

        # Draw rounded background
        pygame.draw.rect(surface, bg, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=self.border_radius)

        # Draw text centered
        txt_surf = font.render(self.text, True, txt_col)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class SegmentedControl:
    """Horizontal tab control for selecting mutually exclusive options."""

    def __init__(self, rect, options, on_change=None, selected_index=0):
        self.rect = pygame.Rect(rect)
        self.options = options  # List of option labels
        self.on_change = on_change
        self.selected_index = selected_index
        self.hovered_index = -1

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                item_w = self.rect.width / len(self.options)
                self.hovered_index = int((event.pos[0] - self.rect.x) // item_w)
            else:
                self.hovered_index = -1

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                item_w = self.rect.width / len(self.options)
                idx = int((event.pos[0] - self.rect.x) // item_w)
                if 0 <= idx < len(self.options):
                    self.selected_index = idx
                    if self.on_change:
                        self.on_change(idx, self.options[idx])
                    return True
        return False

    def draw(self, surface, font):
        # Draw base container
        pygame.draw.rect(surface, (18, 24, 38), self.rect, border_radius=6)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, width=1, border_radius=6)

        num_items = len(self.options)
        item_w = self.rect.width / num_items

        for i, opt in enumerate(self.options):
            item_rect = pygame.Rect(self.rect.x + i * item_w + 2, self.rect.y + 2, item_w - 4, self.rect.height - 4)

            if i == self.selected_index:
                pygame.draw.rect(surface, COLOR_ACCENT, item_rect, border_radius=4)
                txt_col = (255, 255, 255)
            elif i == self.hovered_index:
                pygame.draw.rect(surface, (38, 48, 68), item_rect, border_radius=4)
                txt_col = TEXT_PRIMARY
            else:
                txt_col = TEXT_SECONDARY

            txt_surf = font.render(opt, True, txt_col)
            txt_rect = txt_surf.get_rect(center=item_rect.center)
            surface.blit(txt_surf, txt_rect)


class ProgressBar:
    """Horizontal progress bar with value label."""

    def __init__(self, rect, fill_color=COLOR_ACCENT, bg_color=(20, 26, 38)):
        self.rect = pygame.Rect(rect)
        self.fill_color = fill_color
        self.bg_color = bg_color

    def draw(self, surface, fraction, label=""):
        fraction = max(0.0, min(1.0, fraction))
        pygame.draw.rect(surface, self.bg_color, self.rect, border_radius=4)
        
        if fraction > 0:
            fill_w = max(4, int(self.rect.width * fraction))
            fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.height)
            pygame.draw.rect(surface, self.fill_color, fill_rect, border_radius=4)

        pygame.draw.rect(surface, BORDER_COLOR, self.rect, width=1, border_radius=4)


def draw_card(surface, rect, title="", header_height=26, bg_color=PANEL_BG, border_color=BORDER_COLOR):
    """Draws a styled container card with an optional header strip."""
    r = pygame.Rect(rect)
    pygame.draw.rect(surface, bg_color, r, border_radius=8)
    pygame.draw.rect(surface, border_color, r, width=1, border_radius=8)

    if title:
        header_rect = pygame.Rect(r.x, r.y, r.width, header_height)
        pygame.draw.rect(surface, PANEL_HEADER, header_rect, border_top_left_radius=8, border_top_right_radius=8)
        pygame.draw.rect(surface, border_color, header_rect, width=1, border_top_left_radius=8, border_top_right_radius=8)
