#!/usr/bin/env python3
"""
Download Mapillary panoramas near the target location and build coordinates.json.
Usage: MLY_TOKEN=<token> python download_mapillary.py
"""
import os, sys, json, math, time, urllib.request, urllib.error, urllib.parse, ssl
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

CENTER_LAT  = 43.076030012719
CENTER_LON  = -89.375652539068
RADIUS_M    = 500        # increase if you get fewer than 5 results
MAX_IMAGES  = 200
OUT_DIR     = "mapillary_data"

TOKEN = os.environ.get("MLY_TOKEN", "").strip()
if not TOKEN:
    print("Error: export MLY_TOKEN='MLY|...' first")
    sys.exit(1)

# ── 1. Verify token by fetching the specific anchor image ─────────────────────
print("Verifying token with anchor image 925175394928093…")
anchor_url = (
    "https://graph.mapillary.com/925175394928093"
    "?fields=id,geometry,is_pano,computed_compass_angle,thumb_2048_url,thumb_original_url"
    f"&access_token={TOKEN}"
)
try:
    with urllib.request.urlopen(anchor_url, timeout=15) as r:
        anchor = json.loads(r.read())
    print(f"  OK — is_pano={anchor.get('is_pano')}, id={anchor.get('id')}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"  Token check failed ({e.code}): {body}")
    sys.exit(1)

# ── 2. Search bbox ────────────────────────────────────────────────────────────
dlat = RADIUS_M / 111_000.0
dlon = RADIUS_M / (111_000.0 * math.cos(math.radians(CENTER_LAT)))
bbox = f"{CENTER_LON-dlon:.6f},{CENTER_LAT-dlat:.6f},{CENTER_LON+dlon:.6f},{CENTER_LAT+dlat:.6f}"

search_url = (
    "https://graph.mapillary.com/images"
    f"?fields=id,geometry,is_pano,computed_compass_angle,thumb_2048_url,thumb_original_url"
    f"&bbox={bbox}"
    f"&limit=1000"
    f"&access_token={TOKEN}"
)
print(f"\nSearching bbox {bbox} (radius ~{RADIUS_M}m)…")
with urllib.request.urlopen(search_url, timeout=30) as r:
    payload = json.loads(r.read())

all_imgs = payload.get("data", [])
print(f"  Total images returned: {len(all_imgs)}")
print(f"  Panoramas (is_pano=True): {sum(1 for i in all_imgs if i.get('is_pano'))}")

# Filter: prefer panoramas, fall back to all images if fewer than 3 panos
panos = [i for i in all_imgs if i.get("is_pano")]
candidates = panos if len(panos) >= 3 else all_imgs
if not candidates:
    print("No images found. Try increasing RADIUS_M and re-running.")
    sys.exit(1)
if len(panos) < 3:
    print(f"  Warning: only {len(panos)} panoramas found — including non-pano images too")

# ── 3. Download ───────────────────────────────────────────────────────────────
Path(OUT_DIR).mkdir(exist_ok=True)
nodes = []

for i, img in enumerate(candidates[:MAX_IMAGES]):
    img_id   = img["id"]
    lon, lat = img["geometry"]["coordinates"]
    # prefer original (full equirectangular), fall back to 2048px thumb
    dl_url   = img.get("thumb_original_url") or img.get("thumb_2048_url", "")

    if not dl_url:
        print(f"  [{i+1}] {img_id}: no URL, skipping")
        continue

    filename = f"node_{i:03d}.jpg"
    out_path = os.path.join(OUT_DIR, filename)
    print(f"  [{i+1}/{min(len(candidates), MAX_IMAGES)}] {img_id} → {filename}", end="", flush=True)

    try:
        urllib.request.urlretrieve(dl_url, out_path)
        size_kb = os.path.getsize(out_path) // 1024
        is_pano = img.get("is_pano", False)
        print(f"  ({size_kb} KB, pano={is_pano})")
    except Exception as e:
        print(f"  FAILED: {e}")
        continue

    compass_angle = img.get("computed_compass_angle", img.get("compass_angle", 0.0))
    nodes.append({"id": f"node_{i:03d}", "mapillary_id": img_id,
                  "lat": lat, "lon": lon,
                  "image": filename, "compass_angle": compass_angle})
    time.sleep(0.05)

if len(nodes) < 2:
    print(f"\nOnly {len(nodes)} images downloaded. Increase RADIUS_M or check the area on mapillary.com.")
    sys.exit(1)

coords_path = os.path.join(OUT_DIR, "coordinates.json")
with open(coords_path, "w") as f:
    json.dump(nodes, f, indent=2)

print(f"\n✓ {len(nodes)} images saved to '{OUT_DIR}/'")
print(f"\nNow run:")
print(f"  cd demo && python main.py")
print(f"  > {os.path.abspath(OUT_DIR)}")
