import math
import pygame
from collections import deque

from config import (
    BG_COLOR, NODE_COLOR, NODE_HIGHLIGHT_COLOR, PLAYER_COLOR, TRAIL_COLOR,
    HUD_COLOR,
    NODE_RADIUS, NODE_HIGHLIGHT_RADIUS,
    PLAYER_HEIGHT, PLAYER_BASE, TRAIL_LENGTH, FOV_LINE_LENGTH,
)
from utils import haversine_distance


def _precompute_connections(nodes) -> list[tuple]:
    """One connection line per nearest-neighbor pair (deduplicated)."""
    pairs: set[tuple[int, int]] = set()
    for i, a in enumerate(nodes):
        best_j, best_d = -1, float("inf")
        for j, b in enumerate(nodes):
            if i == j:
                continue
            d = haversine_distance(a.lat, a.lon, b.lat, b.lon)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            pairs.add((min(i, best_j), max(i, best_j)))
    return [(nodes[i].screen_pos, nodes[j].screen_pos) for i, j in pairs]


class MapView:
    def __init__(self, width: int, height: int, nodes, tile_map):
        self.width    = width
        self.height   = height
        self.tile_map = tile_map
        self.trail: deque = deque(maxlen=TRAIL_LENGTH)
        self._connections = _precompute_connections(nodes)

        pygame.font.init()
        self._font_small = pygame.font.SysFont("monospace", 10)
        self._font_hud   = pygame.font.SysFont("monospace", 12)
        self._alpha_surf = pygame.Surface((width, height), pygame.SRCALPHA)

        # Semi-transparent panel for HUD readability
        self._hud_bg = pygame.Surface((220, 90), pygame.SRCALPHA)
        self._hud_bg.fill((0, 0, 0, 140))

    # ------------------------------------------------------------------

    def _draw_connections(self, surface) -> None:
        for a, b in self._connections:
            pygame.draw.line(surface, (80, 80, 130), a, b, 2)

    def _draw_fov_cone(self, alpha_surf, pos, heading_deg, color=None, elevation_deg=0.0) -> None:
        r, g, b = color if color else PLAYER_COLOR
        half = 45.0 * math.cos(math.radians(elevation_deg))
        for sign in (-1, 1):
            rad = math.radians(heading_deg + sign * half - 90.0)
            end = (
                pos[0] + int(FOV_LINE_LENGTH * math.cos(rad)),
                pos[1] + int(FOV_LINE_LENGTH * math.sin(rad)),
            )
            if sign == -1:
                end_left = end
            else:
                end_right = end

        pygame.draw.polygon(alpha_surf, (r, g, b, 22), [pos, end_left, end_right])
        pygame.draw.line(alpha_surf, (r, g, b, 80), pos, end_left, 1)
        pygame.draw.line(alpha_surf, (r, g, b, 80), pos, end_right, 1)

    def _draw_player(self, surface, pos, heading_deg) -> None:
        angle = math.radians(heading_deg - 90.0)
        perp  = angle + math.pi / 2
        half  = PLAYER_BASE // 2
        tip   = (pos[0] + int(PLAYER_HEIGHT * math.cos(angle)),
                 pos[1] + int(PLAYER_HEIGHT * math.sin(angle)))
        left  = (pos[0] + int(half * math.cos(perp)),
                 pos[1] + int(half * math.sin(perp)))
        right = (pos[0] - int(half * math.cos(perp)),
                 pos[1] - int(half * math.sin(perp)))
        pygame.draw.polygon(surface, PLAYER_COLOR, [tip, left, right])
        # White outline for readability on both light and dark backgrounds
        pygame.draw.polygon(surface, (255, 255, 255), [tip, left, right], 1)

    def _draw_node_label(self, surface, text: str, pos: tuple) -> None:
        label = self._font_small.render(text, True, (255, 255, 255))
        lx = pos[0] - label.get_width() // 2
        ly = pos[1] - NODE_RADIUS - 14
        bg = pygame.Surface((label.get_width() + 4, label.get_height()), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        surface.blit(bg, (lx - 2, ly))
        surface.blit(label, (lx, ly))

    # ------------------------------------------------------------------

    def render(self, surface, player, nodes, cue_data, fps: float) -> None:
        # ── Background: tile map (or solid dark fallback) ──────────────
        if self.tile_map.available:
            surface.blit(self.tile_map.surface, (0, 0))
        else:
            surface.fill(BG_COLOR)

        self._draw_connections(surface)

        player_pos = self.tile_map.latlon_to_pane_px(player.lat, player.lon)
        self.trail.append(player_pos)

        # ── Alpha layer: trail + FOV cone ───────────────────────────────
        self._alpha_surf.fill((0, 0, 0, 0))
        r, g, b = TRAIL_COLOR
        for pos in self.trail:
            pygame.draw.circle(self._alpha_surf, (r, g, b, 110), pos, 2)
        elev = player.elevation_deg
        self._draw_fov_cone(self._alpha_surf, player_pos, player.heading_deg, elevation_deg=elev)

        node_map = {n.id: n for n in nodes}
        for node_id, color in (
            (cue_data.nearest_node_id,       (60, 200, 140)),
            (cue_data.second_nearest_node_id, (60, 140, 220)),
        ):
            node = node_map.get(node_id)
            if node:
                self._draw_fov_cone(self._alpha_surf, node.screen_pos, player.heading_deg,
                                    color=color, elevation_deg=elev)

        surface.blit(self._alpha_surf, (0, 0))

        # ── Nodes ───────────────────────────────────────────────────────
        for node in nodes:
            if node.id == cue_data.nearest_node_id:
                pygame.draw.circle(surface, NODE_HIGHLIGHT_COLOR, node.screen_pos, NODE_HIGHLIGHT_RADIUS, 2)
            pygame.draw.circle(surface, NODE_COLOR, node.screen_pos, NODE_RADIUS)
            self._draw_node_label(surface, node.id, node.screen_pos)

        # ── Player arrow ────────────────────────────────────────────────
        self._draw_player(surface, player_pos, player.heading_deg)

        # ── HUD ─────────────────────────────────────────────────────────
        hud = [
            f"FPS: {fps:.0f}",
            f"Pos: ({player.lat:.4f}, {player.lon:.4f})",
            f"Heading: {player.heading_deg:06.2f}°",
            f"Speed: {player.speed_mps:.1f} m/s",
            f"Nearest: {cue_data.nearest_node_id} ({cue_data.nearest_node_dist_m:.1f}m)",
        ]
        hud_y = self.height - len(hud) * 16 - 14
        surface.blit(self._hud_bg, (6, hud_y - 4))
        for line in hud:
            surf = self._font_hud.render(line, True, HUD_COLOR)
            surface.blit(surf, (10, hud_y))
            hud_y += 16
