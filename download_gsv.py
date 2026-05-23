#!/usr/bin/env python3
"""
Download Google Street View panoramas near the target location and build coordinates.json.
Queries a grid of coverage tiles to collect enough unique panoramas.
Usage: python download_gsv.py
"""
import os, sys, json, time, math
from pathlib import Path

import ssl
import aiohttp

# aiohttp ignores the stdlib ssl default — patch at the connector level
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

_orig_init = aiohttp.TCPConnector.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.setdefault('ssl', _ssl_ctx)
    _orig_init(self, *args, **kwargs)
aiohttp.TCPConnector.__init__ = _patched_init

try:
    from streetlevel import streetview
except ImportError:
    print("Error: streetlevel library is not installed. Run 'pip install streetlevel'")
    sys.exit(1)

# ── target location ──────────────────────────────────────────────────────────
CENTER_LAT  = 37.67603099708181
CENTER_LON  = 126.8133136104401
MAX_IMAGES  = 100
GRID_RADIUS = 3          # searches (2*GRID_RADIUS+1)² tiles; increase if still not enough
OUT_DIR     = "gsv_data"
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def offset_latlon(lat, lon, delta_lat_m, delta_lon_m):
    R = 6_371_000
    dlat = math.degrees(delta_lat_m / R)
    dlon = math.degrees(delta_lon_m / (R * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon

def collect_panos(center_lat, center_lon, grid_radius, max_images):
    """Query a grid of coverage tiles and return unique panos sorted by distance."""
    TILE_STEP_M = 120  # GSV coverage tile is ~150m at zoom 17

    seen_ids = set()
    all_panos = []
    total_tiles = (2 * grid_radius + 1) ** 2
    tile_num = 0

    for di in range(-grid_radius, grid_radius + 1):
        for dj in range(-grid_radius, grid_radius + 1):
            tile_num += 1
            lat, lon = offset_latlon(center_lat, center_lon,
                                     di * TILE_STEP_M, dj * TILE_STEP_M)
            try:
                panos = streetview.get_coverage_tile_by_latlon(lat, lon)
            except Exception as e:
                print(f"  tile ({di:+d},{dj:+d}): error — {e}")
                continue

            new = 0
            for p in (panos or []):
                if p.id not in seen_ids:
                    seen_ids.add(p.id)
                    all_panos.append(p)
                    new += 1

            print(f"  tile ({di:+d},{dj:+d}) [{tile_num}/{total_tiles}]: "
                  f"{len(panos or [])} panos, {new} new  (total unique: {len(all_panos)})")

            if len(all_panos) >= max_images * 3:
                break
            time.sleep(0.1)
        if len(all_panos) >= max_images * 3:
            break

    all_panos.sort(key=lambda p: haversine(center_lat, center_lon, p.lat, p.lon))
    return all_panos


def main():
    print(f"Collecting panoramas near {CENTER_LAT:.5f}, {CENTER_LON:.5f}…")
    print(f"Grid: {(2*GRID_RADIUS+1)**2} tiles, target: {MAX_IMAGES} images\n")

    panos = collect_panos(CENTER_LAT, CENTER_LON, GRID_RADIUS, MAX_IMAGES)
    print(f"\n{len(panos)} unique panoramas found — downloading up to {MAX_IMAGES}…\n")

    if not panos:
        print("No panoramas found. Try a different location or increase GRID_RADIUS.")
        sys.exit(1)

    # Load existing coordinates.json to skip already-downloaded nodes
    Path(OUT_DIR).mkdir(exist_ok=True)
    coords_path = os.path.join(OUT_DIR, "coordinates.json")
    existing_nodes = []
    if os.path.exists(coords_path):
        with open(coords_path) as f:
            existing_nodes = json.load(f)
        print(f"  {len(existing_nodes)} existing nodes found — resuming from node_{len(existing_nodes):03d}.\n")

    nodes = list(existing_nodes)
    count = len(nodes)
    existing_pano_filenames = {n["image"] for n in existing_nodes}

    for pano in panos:
        if count >= MAX_IMAGES:
            break

        filename = f"node_{count:03d}.jpg"
        if filename in existing_pano_filenames:
            count += 1
            continue

        out_path = os.path.join(OUT_DIR, filename)
        print(f"  [{count+1}/{MAX_IMAGES}] {pano.id} → {filename}", end="", flush=True)

        if os.path.exists(out_path):
            print("  (file exists, reading metadata only)")
            try:
                full_pano = streetview.find_panorama_by_id(pano.id, download_depth=False)
                heading_rad = getattr(full_pano, "heading", 0.0) if full_pano else 0.0
            except Exception:
                heading_rad = 0.0
        else:
            try:
                full_pano = streetview.find_panorama_by_id(pano.id, download_depth=False)
                if not full_pano or not full_pano.image_sizes:
                    print("  FAILED: no image sizes")
                    continue
                streetview.download_panorama(full_pano, out_path)
                size_kb = os.path.getsize(out_path) // 1024
                print(f"  ({size_kb} KB)")
                heading_rad = getattr(full_pano, "heading", 0.0)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

        nodes.append({
            "id": f"node_{count:03d}",
            "lat": pano.lat,
            "lon": pano.lon,
            "image": filename,
            "compass_angle": math.degrees(heading_rad),
        })
        count += 1

        # Save after every download so a crash doesn't lose progress
        with open(coords_path, "w") as f:
            json.dump(nodes, f, indent=2)

        time.sleep(0.3)

    print(f"\n✓ {len(nodes)} panoramas saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
