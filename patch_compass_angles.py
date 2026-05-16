#!/usr/bin/env python3
"""
Patch existing coordinates.json with compass_angle from the Mapillary API.
Each node is matched by searching a tiny bbox around its GPS position.

Usage:  MLY_TOKEN='MLY|...' python patch_compass_angles.py mapillary_data
"""
import os, sys, json, math, urllib.request, urllib.parse, ssl

ssl._create_default_https_context = ssl._create_unverified_context

TOKEN = os.environ.get("MLY_TOKEN", "").strip()
if not TOKEN:
    print("Error: export MLY_TOKEN='MLY|...' first")
    sys.exit(1)

folder = sys.argv[1] if len(sys.argv) > 1 else "mapillary_data"
coords_path = os.path.join(folder, "coordinates.json")

with open(coords_path) as f:
    nodes = json.load(f)

MATCH_M = 3.0   # metres — nodes within this distance are considered the same image

def search_nearby(lat, lon, radius_m=10.0):
    d = radius_m / 111_000.0
    bbox = f"{lon-d:.7f},{lat-d:.7f},{lon+d:.7f},{lat+d:.7f}"
    url = (
        "https://graph.mapillary.com/images"
        f"?fields=id,geometry,compass_angle,is_pano"
        f"&bbox={bbox}&limit=10"
        f"&access_token={TOKEN}"
    )
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read()).get("data", [])

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

patched = 0
for node in nodes:
    if "compass_angle" in node:
        continue  # already has it
    lat, lon = node["lat"], node["lon"]
    try:
        candidates = search_nearby(lat, lon)
    except Exception as e:
        print(f"  {node['id']}: API error — {e}")
        continue

    # Pick the closest match (prefer panoramas)
    best = None
    best_d = float("inf")
    for c in candidates:
        clat, clon = c["geometry"]["coordinates"][1], c["geometry"]["coordinates"][0]
        d = haversine(lat, lon, clat, clon)
        if d < best_d:
            best_d = d
            best = c

    if best and best_d < MATCH_M and best.get("compass_angle") is not None:
        node["compass_angle"] = best["compass_angle"]
        print(f"  {node['id']}: compass_angle={best['compass_angle']:.1f}° (dist={best_d:.1f}m)")
        patched += 1
    else:
        print(f"  {node['id']}: no close match found (closest={best_d:.1f}m), defaulting to 0°")
        node["compass_angle"] = 0.0

with open(coords_path, "w") as f:
    json.dump(nodes, f, indent=2)

print(f"\n✓ Patched {patched}/{len(nodes)} nodes → {coords_path}")
