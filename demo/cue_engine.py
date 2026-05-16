import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

from config import NEAREST_CACHE_DIST_M, ANCHOR_CROP_SIZE, FOV_DEG, EARTH_RADIUS_M
from utils import haversine_distance, bearing_between

_LOOK_AHEAD_M = 40.0   # reference scene depth for parallax


@dataclass
class CueData:
    # Panel 4 — movement vector
    heading_deg: float
    speed_mps: float
    dx_m: float
    dy_m: float
    delta_heading_deg: float
    # Panel 2 — POV direction
    azimuth_deg: float
    elevation_deg: float
    # Panel 3 — second nearest anchor toward heading
    second_nearest_crop: Optional[np.ndarray]   # 256×256 BGR
    second_nearest_node_id: str
    # Panel 0 — merged crop (main world-model input)
    anchor_crop: np.ndarray                     # 256×256 BGR
    nearest_node_id: str
    nearest_node_dist_m: float
    # Panel 5 — warped node-entry reference
    warped_frame: Optional[np.ndarray]          # 256×256 BGR
    # Panel 6 — residual
    residual: Optional[np.ndarray]              # 256×256 grayscale→BGR
    max_delta: float
    mean_delta: float
    # Panel 1 & 2 — individual node crops at player heading
    nearest_crop_isect: Optional[np.ndarray]    # 256×256 BGR
    second_crop_isect: Optional[np.ndarray]     # 256×256 BGR
    # Extra info for map drawing
    bear_to_nearest: float
    bear_to_second: float


class CueEngine:
    def __init__(self, nodes):
        self.nodes = nodes
        self._cache_lat: Optional[float] = None
        self._cache_lon: Optional[float] = None
        self._cache_sorted: list = []

        self._ref_node_id: Optional[str] = None
        self._ref_crop:    Optional[np.ndarray] = None
        self._ref_heading: float = 0.0

    # ------------------------------------------------------------------

    def _refresh_cache(self, lat, lon) -> None:
        if (self._cache_lat is not None
                and haversine_distance(lat, lon, self._cache_lat, self._cache_lon) < NEAREST_CACHE_DIST_M):
            return
        self._cache_sorted = sorted(
            ((haversine_distance(lat, lon, n.lat, n.lon), n) for n in self.nodes),
            key=lambda x: x[0],
        )
        self._cache_lat, self._cache_lon = lat, lon

    def _nearest(self, lat, lon):
        self._refresh_cache(lat, lon)
        return self._cache_sorted[0]

    def _second_nearest(self, lat, lon):
        self._refresh_cache(lat, lon)
        return self._cache_sorted[1] if len(self._cache_sorted) >= 2 else self._cache_sorted[0]

    # ------------------------------------------------------------------
    # Panorama cropping
    # ------------------------------------------------------------------

    def _crop_at_heading(self, img: np.ndarray, heading_deg: float,
                         compass_angle: float = 0.0, elevation_deg: float = 0.0,
                         fov: float = FOV_DEG) -> np.ndarray:
        """
        Extract a crop from an equirectangular panorama.

        Horizontal: column 0 = camera forward (compass_angle); columns increase clockwise.
        Vertical:   row 0 = zenith (+90°), row H = nadir (-90°), row H/2 = horizon.
                    elevation_deg shifts the vertical centre of the crop.
        """
        H, W = img.shape[:2]
        ppd = W / 360.0

        # ── Horizontal ────────────────────────────────────────────────
        adjusted  = (heading_deg - compass_angle) % 360.0
        center_px = int(adjusted * ppd) % W
        half_px   = int((fov / 2.0) * ppd)
        cs = (center_px - half_px) % W
        ce = (center_px + half_px) % W

        # ── Vertical: shift by elevation ──────────────────────────────
        # Row for elevation E: H * (0.5 - E / 180)   (±90° spans full H)
        r_center = H * (0.5 - elevation_deg / 180.0)
        half_h   = H * 0.25                           # ±45° of vertical range
        r0 = max(0, int(r_center - half_h))
        r1 = min(H, int(r_center + half_h))
        if r1 <= r0:
            r0, r1 = int(H * 0.25), int(H * 0.75)

        if cs < ce:
            strip = img[r0:r1, cs:ce]
        else:
            strip = np.concatenate([img[r0:r1, cs:], img[r0:r1, :ce]], axis=1)

        if strip.size == 0:
            strip = img[r0:r1, :]

        return cv2.resize(strip, (ANCHOR_CROP_SIZE, ANCHOR_CROP_SIZE),
                          interpolation=cv2.INTER_LINEAR)

    def _overlap_crop(
        self,
        img: np.ndarray,
        heading_deg: float,
        compass_angle: float,
        bear_player_to_anchor: float,
        elevation_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Crop anchor panorama at player heading H, then black out columns outside
        the overlap of player's FOV [H±45°] with anchor's cone [bear_P→A ±45°].
        No stretching — natural scale with black padding.
        """
        half  = FOV_DEG / 2.0
        delta = (bear_player_to_anchor - heading_deg + 180.0) % 360.0 - 180.0
        lo    = max(-half, delta - half)
        hi    = min( half, delta + half)

        # Full player-FOV crop at natural scale
        full = self._crop_at_heading(img, heading_deg, compass_angle, elevation_deg)

        if hi <= lo:
            return self._BLACK.copy()

        # Column range within the ANCHOR_CROP_SIZE-wide output that falls in the overlap
        S = ANCHOR_CROP_SIZE
        col_s = max(0,   int((lo + half) / FOV_DEG * S))
        col_e = min(S,   int((hi + half) / FOV_DEG * S))

        result = self._BLACK.copy()
        result[:, col_s:col_e] = full[:, col_s:col_e]
        return result

    def _blend(self, a: np.ndarray, w_a: float, b: np.ndarray, w_b: float) -> np.ndarray:
        return np.clip(a.astype(np.float32) * w_a + b.astype(np.float32) * w_b, 0, 255).astype(np.uint8)

    _BLACK = np.zeros((ANCHOR_CROP_SIZE, ANCHOR_CROP_SIZE, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, player) -> CueData:
        lat, lon  = player.lat, player.lon
        heading   = player.heading_deg
        elevation = player.elevation_deg

        dist_near, nearest = self._nearest(lat, lon)
        dist_2nd,  second  = self._second_nearest(lat, lon)

        # Bearings player→anchor (for map display and blend weight)
        bear_to_n1 = bearing_between(lat, lon, nearest.lat, nearest.lon)
        bear_to_n2 = bearing_between(lat, lon, second.lat,  second.lon)

        # Reference point: _LOOK_AHEAD_M metres ahead of the player along heading.
        # Each anchor crops toward this point so the picture responds to both
        # turning and forward/backward movement (parallax).
        h_rad   = math.radians(heading)
        lat_rad = math.radians(lat)
        ref_lat = lat + (_LOOK_AHEAD_M * math.cos(h_rad)) / EARTH_RADIUS_M * (180.0 / math.pi)
        ref_lon = lon + (_LOOK_AHEAD_M * math.sin(h_rad)) / (EARTH_RADIUS_M * math.cos(lat_rad)) * (180.0 / math.pi)

        n1_crop_hdg = bearing_between(nearest.lat, nearest.lon, ref_lat, ref_lon)
        n2_crop_hdg = bearing_between(second.lat,  second.lon,  ref_lat, ref_lon)

        crop_n1 = self._crop_at_heading(nearest.image_bgr, n1_crop_hdg, nearest.compass_angle, elevation)
        crop_n2 = self._crop_at_heading(second.image_bgr,  n2_crop_hdg, second.compass_angle,  elevation)

        # Merged: blend whichever crops exist, weighted by inverse distance
        total  = dist_near + dist_2nd
        w_near = dist_2nd  / total if total > 1e-3 else 0.5
        w_2nd  = dist_near / total if total > 1e-3 else 0.5

        anchor_crop = self._blend(crop_n1, w_near, crop_n2, w_2nd)

        # Context: second nearest crop at same reference heading
        second_crop = self._crop_at_heading(second.image_bgr, n2_crop_hdg, second.compass_angle, elevation)

        # Warp reference: reset when nearest node changes
        if nearest.id != self._ref_node_id:
            self._ref_node_id = nearest.id
            self._ref_crop    = anchor_crop.copy()
            self._ref_heading = heading

        # ── Pose warp ─────────────────────────────────────────────────
        warped_frame: Optional[np.ndarray] = None
        if self._ref_crop is not None:
            ppd_256  = 256.0 / FOV_DEG
            delta_az = (heading - self._ref_heading + 360.0) % 360.0
            if delta_az > 180.0:
                delta_az -= 360.0
            warped_frame = np.roll(self._ref_crop, -int(delta_az * ppd_256), axis=1)

        # ── Residual ──────────────────────────────────────────────────
        residual:  Optional[np.ndarray] = None
        max_delta = mean_delta = 0.0

        if warped_frame is not None:
            diff      = cv2.absdiff(anchor_crop, warped_frame)
            amplified = np.clip(diff.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
            gray      = cv2.cvtColor(amplified, cv2.COLOR_BGR2GRAY)
            residual  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            max_delta  = float(diff.max())  / 255.0
            mean_delta = float(diff.mean()) / 255.0

        return CueData(
            heading_deg=heading,
            speed_mps=player.speed_mps,
            dx_m=player.dx_m,
            dy_m=player.dy_m,
            delta_heading_deg=player.delta_heading,
            azimuth_deg=heading,
            elevation_deg=elevation,
            second_nearest_crop=second_crop,
            second_nearest_node_id=second.id,
            anchor_crop=anchor_crop,
            nearest_node_id=nearest.id,
            nearest_node_dist_m=dist_near,
            warped_frame=warped_frame,
            residual=residual,
            max_delta=max_delta,
            mean_delta=mean_delta,
            nearest_crop_isect=crop_n1,
            second_crop_isect=crop_n2,
            bear_to_nearest=bear_to_n1,
            bear_to_second=bear_to_n2,
        )
