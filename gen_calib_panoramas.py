"""
Calibration panorama generator.

Places several coloured pillars at known world positions.  Each node renders
the scene from ITS OWN (east_m, north_m) position, so nearby pillars appear
at noticeably different azimuths across nodes while distant ones barely shift.

Calibration check
─────────────────
Run the demo with DATA_FOLDER = calib_data/.
The MERGED VIEW (Panel 0) should show every pillar as a SINGLE sharp stripe.
Smearing or doubling → the parallax correction in _merged_crop has a bug.

The individual node panels (1/2/3) will show pillars at DIFFERENT azimuths
from one another, matching the prismatoid intersection on the minimap.
"""

import json, math, os
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(__file__), "calib_data")
os.makedirs(OUT_DIR, exist_ok=True)

PW, PH   = 4096, 2048
FONT     = cv2.FONT_HERSHEY_SIMPLEX
EARTH_R  = 6_371_000.0

# ── World-space pillars: (east_m, north_m, BGR_colour, label) ─────────────
# Distances range from 10 m to 120 m so you can see both strong and weak parallax.
PILLARS = [
    (   0,  10, (0,  0, 255), "R10N"),    # 10 m north     — very strong parallax
    (  10,  10, (0,200,  0),  "G14NE"),   # 14 m NE
    ( -10,  10, (255, 80, 0), "O14NW"),   # 14 m NW  (orange)
    (   0,  30, (255,255,0),  "Y30N"),    # 30 m north     — moderate parallax
    (  20,  30, (255,  0,255),"M36NE"),   # 36 m NE  (magenta)
    (   0,  80, (0,200,255),  "C80N"),    # 80 m north     — weak parallax
    (  40,  60, (200,200,200),"W72"),     # 72 m NE  (white)
    ( -40,  60, (100,100,255),"B72NW"),   # 72 m NW  (light blue)
    (   0, 120, (80, 255,80), "LG120N"),  # 120 m north    — almost no parallax
]

PILLAR_HALF_AZ  = 1.2    # degrees half-width of each pillar stripe
PILLAR_HALF_EL  = 12.0   # degrees half-height (vertical extent)

# ── Synthetic node layout (east_m, north_m offsets from anchor) ───────────
LAYOUT = [
    # id           east_m  north_m  compass_angle
    ("node_000",      0,      0,    90.0),
    ("node_001",    -12,     18,    90.0),
    ("node_002",     14,     15,    90.0),
    ("node_003",    -10,    -16,    90.0),
    ("node_004",     13,    -14,    90.0),
    ("node_005",      0,     22,    90.0),
    ("node_006",     22,      0,    90.0),
]

# ── Helpers ───────────────────────────────────────────────────────────────

def world_to_pano_col(world_az_deg: float, compass_angle: float) -> int:
    """Map absolute compass bearing → panorama pixel column."""
    return int(((world_az_deg - compass_angle - 180) % 360) / 360 * PW) % PW


def bearing(from_e, from_n, to_e, to_n) -> float:
    """Compass bearing (degrees) from one world point to another."""
    return math.degrees(math.atan2(to_e - from_e, to_n - from_n)) % 360


def make_panorama(node_id: str, east_m: float, north_m: float,
                  compass_angle: float) -> np.ndarray:

    img = np.full((PH, PW, 3), 30, dtype=np.uint8)   # dark grey background

    ppd = PW / 360.0   # pixels per degree

    # ── Sky gradient: top half blue→white, bottom half brown (ground) ─────
    for row in range(PH):
        el = 90 - row / PH * 180   # +90 at top, -90 at bottom
        if el >= 0:
            t = el / 90
            c = int(40 + t * 180)
            img[row] = (c, c // 2, 10)          # blue sky
        else:
            t = -el / 90
            img[row] = (int(30 + t * 40),        # brown ground
                        int(50 + t * 60),
                        int(60 + t * 80))

    # ── Azimuth grid every 15° ─────────────────────────────────────────────
    for az in range(0, 360, 15):
        col = world_to_pano_col(az, compass_angle)
        thick = 2 if az % 45 == 0 else 1
        color = (200, 200, 200) if az % 45 == 0 else (80, 80, 80)
        cv2.line(img, (col, 0), (col, PH), color, thick)
        if az % 45 == 0:
            row_lbl = int(PH * 0.48)
            cv2.putText(img, f"{az}", (col + 3, row_lbl), FONT, 0.6,
                        (200, 200, 200), 2)

    # ── Elevation grid every 15° ───────────────────────────────────────────
    for el in range(-75, 91, 15):
        row = int(PH * (0.5 - el / 180.0))
        if 0 <= row < PH:
            thick = 2 if el == 0 else 1
            color = (180, 180, 180) if el == 0 else (70, 70, 70)
            cv2.line(img, (0, row), (PW, row), color, thick)
            cv2.putText(img, f"{el:+d}", (8, row - 3), FONT, 0.4,
                        (160, 160, 160), 1)

    # ── Cardinal labels ────────────────────────────────────────────────────
    for az, name in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        col = world_to_pano_col(az, compass_angle)
        cv2.putText(img, name, (col - 20, PH // 2 - 100), FONT, 2.5,
                    (255, 255, 255), 5)

    # ── Pillars ────────────────────────────────────────────────────────────
    for (p_east, p_north, bgr, label) in PILLARS:
        dist = math.hypot(p_east - east_m, p_north - north_m)
        if dist < 0.5:
            continue   # pillar is at same position as this node

        az_deg = bearing(east_m, north_m, p_east, p_north)

        # Draw stripe columns
        for daz in np.arange(-PILLAR_HALF_AZ, PILLAR_HALF_AZ, 360 / PW):
            col = world_to_pano_col(az_deg + daz, compass_angle)
            for del_el in np.arange(-PILLAR_HALF_EL, PILLAR_HALF_EL, 180 / PH):
                row = int(PH * (0.5 - del_el / 180.0))
                if 0 <= row < PH:
                    img[row, col % PW] = bgr

        # Label: show bearing + distance from THIS node
        col_c = world_to_pano_col(az_deg, compass_angle)
        row_top = int(PH * (0.5 - PILLAR_HALF_EL / 180.0)) - 30
        cv2.putText(img, label, (col_c - 20, max(12, row_top)), FONT, 0.55,
                    bgr, 2)
        cv2.putText(img, f"{az_deg:.1f}", (col_c - 22, max(30, row_top + 20)),
                    FONT, 0.45, bgr, 1)
        cv2.putText(img, f"{dist:.0f}m", (col_c - 18, max(46, row_top + 38)),
                    FONT, 0.45, bgr, 1)

    # ── Node identity label ────────────────────────────────────────────────
    info = (f"{node_id}  compass={compass_angle:.0f}  "
            f"pos=({east_m:+.0f}m E, {north_m:+.0f}m N)")
    cv2.putText(img, info, (30, PH // 2 + 80), FONT, 1.1, (0, 255, 255), 3)

    return img


# ── Derive anchor lat/lon from the first real node ─────────────────────────
real       = json.load(open(os.path.join(os.path.dirname(__file__),
                                         "gsv_data", "coordinates.json")))[0]
anchor_lat = real["lat"]
anchor_lon = real["lon"]
mpdlat     = EARTH_R * math.pi / 180.0
mpdlon     = EARTH_R * math.cos(math.radians(anchor_lat)) * math.pi / 180.0

records = []
for node_id, east_m, north_m, compass_angle in LAYOUT:
    lat = anchor_lat + north_m / mpdlat
    lon = anchor_lon + east_m  / mpdlon
    pano = make_panorama(node_id, east_m, north_m, compass_angle)
    path = os.path.join(OUT_DIR, f"{node_id}.jpg")
    cv2.imwrite(path, pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  {node_id}  ({east_m:+.0f}m E, {north_m:+.0f}m N)  → {path}")
    records.append({"id": node_id, "lat": lat, "lon": lon,
                    "image": f"{node_id}.jpg", "compass_angle": compass_angle})

json.dump(records, open(os.path.join(OUT_DIR, "coordinates.json"), "w"), indent=2)

print(f"\nDone — {len(records)} nodes in {OUT_DIR}/")
print("\nPillar world positions (east_m, north_m from node_000):")
for p_e, p_n, bgr, label in PILLARS:
    print(f"  {label:8s}  ({p_e:+4.0f} E, {p_n:+4.0f} N)  dist from origin: {math.hypot(p_e,p_n):.0f} m")
print("\nCalibration: in the merged view every pillar must be a single sharp stripe.")
print("In individual node panels, same pillar appears at DIFFERENT azimuths.")
