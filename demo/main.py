import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, PANEL_SECTION_W,
    YAW_SENS, PITCH_SENS,
    TARGET_FPS, DATA_FOLDER,
)
from loader import load_scene, build_graph
from tile_map import TileMap
from viewer import StreetViewer
from minimap import Minimap
from cue_engine import CueEngine
from cue_panel import CuePanel


class _ViewerAsPlayer:
    """Thin adapter so CueEngine can read viewer state without knowing about it."""
    def __init__(self):
        self.lat           = 0.0
        self.lon           = 0.0
        self.heading_deg   = 0.0
        self.elevation_deg = 0.0
        self.speed_mps     = 0.0
        self.dx_m          = 0.0
        self.dy_m          = 0.0
        self.delta_heading = 0.0

    def sync(self, viewer):
        self.lat           = viewer.current_node.lat
        self.lon           = viewer.current_node.lon
        self.heading_deg   = viewer.heading_deg
        self.elevation_deg = viewer.pitch_deg


def main() -> None:
    folder_path = DATA_FOLDER
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        sys.exit(1)

    print("Loading scene…")
    nodes = load_scene(folder_path)
    build_graph(nodes)
    print(f"  {len(nodes)} nodes loaded")

    center_lat = sum(n.lat for n in nodes) / len(nodes)
    center_lon = sum(n.lon for n in nodes) / len(nodes)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".tile_cache")
    tile_map  = TileMap(center_lat, center_lon, WINDOW_WIDTH, WINDOW_HEIGHT,
                        cache_dir=cache_dir, nodes=nodes)

    total_w = WINDOW_WIDTH + PANEL_SECTION_W

    pygame.init()
    screen   = pygame.display.set_mode((total_w, WINDOW_HEIGHT))
    pygame.display.set_caption("Street View")
    clock    = pygame.time.Clock()
    fps_font = pygame.font.SysFont("monospace", 11)

    viewer     = StreetViewer(nodes, WINDOW_WIDTH, WINDOW_HEIGHT)
    minimap    = Minimap(tile_map, nodes)
    cue_engine = CueEngine(nodes)
    cue_panel  = CuePanel(WINDOW_WIDTH, 0, PANEL_SECTION_W, WINDOW_HEIGHT)
    player     = _ViewerAsPlayer()
    gsv_surf   = screen.subsurface(pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))

    from utils import bearing_between
    if len(nodes) >= 2:
        viewer.heading_deg = bearing_between(
            nodes[0].lat, nodes[0].lon, nodes[1].lat, nodes[1].lon,
        )

    dragging   = False
    drag_start = (0, 0)
    drag_moved = False

    while True:
        clock.tick(TARGET_FPS)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                elif event.key == pygame.K_r:
                    viewer.heading_deg = 0.0
                    viewer.pitch_deg   = 0.0
                    viewer.fov_deg     = 90.0
                    viewer._dirty      = True

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if mouse_pos[0] < WINDOW_WIDTH:
                    dragging   = True
                    drag_start = event.pos
                    drag_moved = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    if abs(dx) > 2 or abs(dy) > 2:
                        drag_moved = True
                    viewer.rotate(dx * YAW_SENS, -dy * PITCH_SENS)
                    drag_start = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
                if not drag_moved and mouse_pos[0] < WINDOW_WIDTH:
                    node = viewer.node_at_screen(*event.pos)
                    if node:
                        viewer.navigate_to(node)

            elif event.type == pygame.MOUSEWHEEL:
                if mouse_pos[0] < WINDOW_WIDTH:
                    viewer.zoom(-event.y * 5.0)

        # Cue data (computed first so minimap can show anchor cones)
        player.sync(viewer)
        cue_data = cue_engine.update(player)
        node_map  = {n.id: n for n in nodes}
        nearest_node = node_map.get(cue_data.nearest_node_id)
        second_node  = node_map.get(cue_data.second_nearest_node_id)

        # GSV side
        viewer.render(gsv_surf, mouse_pos)
        minimap.render(gsv_surf, viewer.current_node, viewer.heading_deg, viewer.fov_deg,
                       viewer.pitch_deg, nearest_node, second_node)

        # Cue panel side
        cue_panel.render(screen, cue_data)

        # Divider
        pygame.draw.line(screen, (60, 60, 80),
                         (WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), 2)

        fps = clock.get_fps()
        fps_surf = fps_font.render(f"{fps:.0f} fps", True, (160, 160, 160))
        screen.blit(fps_surf, (WINDOW_WIDTH - fps_surf.get_width() - 10,
                               WINDOW_HEIGHT - fps_surf.get_height() - 8))

        pygame.display.flip()


if __name__ == "__main__":
    main()
