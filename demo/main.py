import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, MAP_WIDTH, CUE_WIDTH,
    DIVIDER_COLOR, TARGET_FPS,
)
from loader import load_scene, build_graph
from tile_map import TileMap
from player import Player
from cue_engine import CueEngine
from map_view import MapView
from cue_panel import CuePanel
from utils import bearing_between


def main() -> None:
    folder_path = "/Users/a/GitHub/InfiniteRace-model-input-demo/mapillary_data" #input("Enter the folder path containing coordinates.json and panorama images:\n> ").strip()
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        sys.exit(1)

    print("Loading scene…")
    nodes = load_scene(folder_path)
    build_graph(nodes)  # builds nearest-neighbor connections (used by MapView)
    print(f"  {len(nodes)} nodes loaded")

    center_lat = sum(n.lat for n in nodes) / len(nodes)
    center_lon = sum(n.lon for n in nodes) / len(nodes)

    # Tile cache lives at project root so it survives across runs
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".tile_cache")
    tile_map = TileMap(center_lat, center_lon, MAP_WIDTH, WINDOW_HEIGHT,
                       cache_dir=cache_dir, nodes=nodes)

    # Update every node's screen_pos using Mercator-correct projection
    for node in nodes:
        node.screen_pos = tile_map.latlon_to_pane_px(node.lat, node.lon)

    # Start at node 0, facing toward node 1
    start_heading = 0.0
    if len(nodes) >= 2:
        start_heading = bearing_between(nodes[0].lat, nodes[0].lon, nodes[1].lat, nodes[1].lon)
    player = Player(nodes[0].lat, nodes[0].lon, heading=start_heading)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags=0)
    pygame.display.set_caption("World Model Input Preview")
    clock = pygame.time.Clock()

    map_surface = pygame.Surface((MAP_WIDTH, WINDOW_HEIGHT))
    cue_surface = pygame.Surface((CUE_WIDTH, WINDOW_HEIGHT))

    map_view   = MapView(MAP_WIDTH, WINDOW_HEIGHT, nodes, tile_map)
    cue_panel  = CuePanel(0, 0, CUE_WIDTH, WINDOW_HEIGHT)
    cue_engine = CueEngine(nodes)

    running = True

    while running:
        dt = min(clock.tick(TARGET_FPS) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    player.reset()

        keys = pygame.key.get_pressed()
        player.update(dt, keys)
        cue_data = cue_engine.update(player)

        map_view.render(map_surface, player, nodes, cue_data, clock.get_fps())
        cue_panel.render(cue_surface, cue_data)

        screen.blit(map_surface, (0, 0))
        screen.blit(cue_surface, (MAP_WIDTH, 0))
        pygame.draw.line(screen, DIVIDER_COLOR, (MAP_WIDTH, 0), (MAP_WIDTH, WINDOW_HEIGHT), 1)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
