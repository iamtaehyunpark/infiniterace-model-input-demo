#!/usr/bin/env python3
"""
Download Google Street View panoramas near the target location and build coordinates.json.
Usage: python download_gsv.py
"""
import os, sys, json, time, math
from pathlib import Path

import ssl
ssl.create_default_context = ssl._create_unverified_context

try:
    from streetlevel import streetview
except ImportError:
    print("Error: streetlevel library is not installed. Run 'pip install streetlevel'")
    sys.exit(1)

# ── target location ──────────────────────────────────────────────────────────
CENTER_LAT  = 37.67603099708181
CENTER_LON  = 126.8133136104401
MAX_IMAGES  = 30         # cap on downloaded panoramas
OUT_DIR     = "gsv_data"
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def main():
    print(f"Querying Google Street View API near {CENTER_LAT}, {CENTER_LON}…")
    
    panos = streetview.get_coverage_tile_by_latlon(CENTER_LAT, CENTER_LON)
    if not panos:
        print("No panoramas found.")
        sys.exit(1)
        
    print(f"  {len(panos)} panoramas found in coverage tile")
    
    # Sort panos by distance to center
    panos.sort(key=lambda p: haversine(CENTER_LAT, CENTER_LON, p.lat, p.lon))
    
    Path(OUT_DIR).mkdir(exist_ok=True)
    nodes = []

    count = 0
    for i, pano in enumerate(panos):
        if count >= MAX_IMAGES:
            break
            
        img_id = pano.id
        lat, lon = pano.lat, pano.lon
        
        filename = f"node_{count:03d}.jpg"
        out_path = os.path.join(OUT_DIR, filename)
        
        print(f"  [{count+1}/{min(len(panos), MAX_IMAGES)}] {img_id} → {filename}", end="", flush=True)

        try:
            # Need to get full metadata first
            full_pano = streetview.find_panorama_by_id(pano.id, download_depth=False)
            if not full_pano.image_sizes:
                print(f"  FAILED: no image sizes available")
                continue
            
            # Download equirectangular panorama
            streetview.download_panorama(full_pano, out_path)
            size_kb = os.path.getsize(out_path) // 1024
            print(f"  ({size_kb} KB)")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # streetlevel GSV panoramas provide `heading` in RADIANS!
        # The left edge of the panorama (x=0) corresponds exactly to this vehicle heading.
        heading_rad = getattr(full_pano, "heading", 0.0)
        compass_angle_deg = math.degrees(heading_rad)
        
        nodes.append({
            "id": f"node_{count:03d}", 
            "lat": lat, 
            "lon": lon, 
            "image": filename,
            "compass_angle": compass_angle_deg
        })
        count += 1
        time.sleep(0.5)

    coords_path = os.path.join(OUT_DIR, "coordinates.json")
    with open(coords_path, "w") as f:
        json.dump(nodes, f, indent=2)

    print(f"\n✓ {len(nodes)} panoramas saved to '{OUT_DIR}/'")
    print(f"\nRun the demo:")
    print(f"  cd demo && python main.py")
    print(f"  > {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
