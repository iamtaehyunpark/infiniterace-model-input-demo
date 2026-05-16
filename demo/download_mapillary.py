#!/usr/bin/env python3
"""
Download Mapillary panoramas near the target location and build coordinates.json.
Usage: MLY_TOKEN=<token> python download_mapillary.py
"""
import os, sys, json, math, time, urllib.request, urllib.error
from pathlib import Path

# ── target location (from the Mapillary URL) ──────────────────────────────────
CENTER_LAT  = 43.076030012719
CENTER_LON  = -89.375652539068
RADIUS_M    = 300        # search radius in meters; increase if too few results
MAX_IMAGES  = 30         # cap on downloaded panoramas
OUT_DIR     = "mapillary_data"
# ─────────────────────────────────────────────────────────────────────────────

TOKEN = os.environ.get("MLY_TOKEN", "").strip()
if not TOKEN:
    print("Error: set the MLY_TOKEN environment variable to your Mapillary client token.")
    print("  export MLY_TOKEN=MLY|27513766691562190|b6839d35e6804db157c6d01bf8d98ad4")
    sys.exit(1)

# Bounding box
dlat = RADIUS_M / 111_000.0
dlon = RADIUS_M / (111_000.0 * math.cos(math.radians(CENTER_LAT)))
bbox = f"{CENTER_LON-dlon:.6f},{CENTER_LAT-dlat:.6f},{CENTER_LON+dlon:.6f},{CENTER_LAT+dlat:.6f}"

url = (
    "https://graph.mapillary.com/images"
    f"?fields=id,geometry,thumb_original_url,is_pano"
    f"&bbox={bbox}"
    f"&limit=100"
    f"&access_token={TOKEN}"
)

print(f"Querying Mapillary API (bbox {bbox})…")
try:
    with urllib.request.urlopen(url, timeout=30) as r:
        payload = json.loads(r.read())
except urllib.error.HTTPError as e:
    print(f"API error {e.code}: {e.read().decode()}")
    sys.exit(1)

all_images = payload.get("data", [])
panos = [img for img in all_images if img.get("is_pano")]
print(f"  {len(all_images)} images found, {len(panos)} are panoramas")

if not panos:
    print("No panoramas in range. Try increasing RADIUS_M.")
    sys.exit(1)

Path(OUT_DIR).mkdir(exist_ok=True)
nodes = []

for i, img in enumerate(panos[:MAX_IMAGES]):
    img_id   = img["id"]
    lon, lat = img["geometry"]["coordinates"]
    dl_url   = img.get("thumb_original_url", "")

    if not dl_url:
        print(f"  [{i+1}] {img_id}: no URL, skipping")
        continue

    filename = f"node_{i:03d}.jpg"
    out_path = os.path.join(OUT_DIR, filename)
    print(f"  [{i+1}/{min(len(panos), MAX_IMAGES)}] {img_id} → {filename}", end="", flush=True)

    try:
        urllib.request.urlretrieve(dl_url, out_path)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"  ({size_kb} KB)")
    except Exception as e:
        print(f"  FAILED: {e}")
        continue

    nodes.append({"id": f"node_{i:03d}", "lat": lat, "lon": lon, "image": filename})
    time.sleep(0.05)

coords_path = os.path.join(OUT_DIR, "coordinates.json")
with open(coords_path, "w") as f:
    json.dump(nodes, f, indent=2)

print(f"\n✓ {len(nodes)} panoramas saved to '{OUT_DIR}/'")
print(f"\nRun the demo:")
print(f"  cd demo && python main.py")
print(f"  > {os.path.abspath(OUT_DIR)}")
