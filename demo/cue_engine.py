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
    # Panel 3 — third nearest anchor intersection
    third_crop_isect: Optional[np.ndarray]      # 256×256 BGR
    third_crop_fov: float
    third_nearest_node_id: str
    # (kept for legacy — second nearest full heading)
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
    # Panel 1 / 2 / 3 — individual node crops at player heading
    nearest_crop_isect: Optional[np.ndarray]    # 256×256 BGR
    second_crop_isect: Optional[np.ndarray]     # 256×256 BGR
    nearest_crop_fov: float                     # angular width of nearest intersection (deg)
    second_crop_fov: float                      # angular width of second intersection (deg)
    # Headings from each anchor toward the look-ahead reference point
    nearest_crop_hdg: float
    second_crop_hdg: float
    third_crop_hdg: float
    # Extra info for map drawing
    bear_to_nearest: float
    bear_to_second: float
    # Node positions relative to player (East=X, North=Y, metres)
    nearest_east_m: float
    nearest_north_m: float
    second_east_m: float
    second_north_m: float
    third_east_m: float
    third_north_m: float


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

    def _third_nearest(self, lat, lon):
        self._refresh_cache(lat, lon)
        return self._cache_sorted[2] if len(self._cache_sorted) >= 3 else self._cache_sorted[-1]

    # ------------------------------------------------------------------
    # Panorama cropping
    # ------------------------------------------------------------------

    def _crop_at_heading(self, img: np.ndarray, heading_deg: float,
                         compass_angle: float = 0.0, elevation_deg: float = 0.0,
                         fov: float = FOV_DEG) -> np.ndarray:
        """Rectilinear perspective crop from an equirectangular panorama (square output)."""
        S = ANCHOR_CROP_SIZE
        PH, PW = img.shape[:2]

        hr = math.radians(heading_deg)
        pr = math.radians(elevation_deg)
        ch, sh = math.cos(hr), math.sin(hr)
        cp, sp = math.cos(pr), math.sin(pr)

        right   = np.array([ ch,      -sh,       0.0], dtype=np.float32)
        fwd     = np.array([ sh * cp,  ch * cp,  sp ], dtype=np.float32)
        up      = np.array([-sh * sp, -ch * sp,  cp ], dtype=np.float32)

        tan_h = math.tan(math.radians(fov / 2.0))
        xs = np.linspace(-tan_h,  tan_h, S, dtype=np.float32)
        ys = np.linspace( tan_h, -tan_h, S, dtype=np.float32)
        cx, cy = np.meshgrid(xs, ys)

        wx = cx * right[0] + cy * up[0] + fwd[0]
        wy = cx * right[1] + cy * up[1] + fwd[1]
        wz = cx * right[2] + cy * up[2] + fwd[2]

        azimuth   = np.arctan2(wx.astype(np.float64), wy.astype(np.float64))
        elevation = np.arctan2(wz.astype(np.float64),
                               np.sqrt(wx.astype(np.float64)**2 + wy.astype(np.float64)**2))

        c_off = math.radians(compass_angle + 180.0)
        map_x = (((azimuth - c_off) / (2.0 * math.pi)) % 1.0 * PW).astype(np.float32)
        map_y = ((0.5 - elevation / math.pi) * PH).astype(np.float32)

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def _merged_crop(self,
                     nodes_dists: list,   # [(dist_m, node, east_m, north_m), ...]
                     heading_deg: float, elevation_deg: float,
                     scene_depth_m: float = _LOOK_AHEAD_M) -> np.ndarray:
        """
        Parallax-corrected merge: for each output pixel, trace the player's ray to
        scene_depth_m, compute the world point, then sample each node's panorama at
        the exact bearing from THAT NODE'S POSITION to the world point.
        Blend samples weighted by inverse distance.
        """
        S = ANCHOR_CROP_SIZE
        hr = math.radians(heading_deg)
        pr = math.radians(elevation_deg)
        ch, sh = math.cos(hr), math.sin(hr)
        cp, sp = math.cos(pr), math.sin(pr)

        right = np.array([ ch,      -sh,       0.0], dtype=np.float64)
        fwd   = np.array([ sh * cp,  ch * cp,  sp ], dtype=np.float64)
        up    = np.array([-sh * sp, -ch * sp,  cp ], dtype=np.float64)

        tan_h = math.tan(math.radians(FOV_DEG / 2.0))
        xs = np.linspace(-tan_h,  tan_h, S, dtype=np.float64)
        ys = np.linspace( tan_h, -tan_h, S, dtype=np.float64)
        cx, cy_g = np.meshgrid(xs, ys)

        # World directions (S×S)
        dx = cx * right[0] + cy_g * up[0] + fwd[0]
        dy = cx * right[1] + cy_g * up[1] + fwd[1]
        dz = cx * right[2] + cy_g * up[2] + fwd[2]

        # Forward component for depth scaling
        fwd_comp = dx * fwd[0] + dy * fwd[1] + dz * fwd[2]
        fwd_comp = np.maximum(fwd_comp, 1e-6)

        # World points at scene_depth (East=X, North=Y, Up=Z offsets from player)
        t = scene_depth_m / fwd_comp
        wx = dx * t
        wy = dy * t
        wz = dz * t

        accum  = np.zeros((S, S, 3), dtype=np.float64)
        w_sum  = np.zeros((S, S),    dtype=np.float64)

        for dist_m, node, east_m, north_m in nodes_dists:
            PH, PW = node.image_bgr.shape[:2]
            w = 1.0 / max(dist_m, 0.1)

            # Vector from this node to each world point
            rx = wx - east_m
            ry = wy - north_m
            rz = wz  # same height

            az  = np.arctan2(rx, ry)
            el  = np.arctan2(rz, np.sqrt(rx * rx + ry * ry))

            c_off = math.radians(node.compass_angle + 180.0)
            map_x = (((az - c_off) / (2.0 * math.pi)) % 1.0 * PW).astype(np.float32)
            map_y = ((0.5 - el / math.pi) * PH).astype(np.float32)

            sample = cv2.remap(node.image_bgr, map_x, map_y,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            accum  += sample.astype(np.float64) * w
            w_sum  += w

        w_sum = np.maximum(w_sum[:, :, np.newaxis], 1e-9)
        return np.clip(accum / w_sum, 0, 255).astype(np.uint8)

    def _blend(self, a: np.ndarray, w_a: float, b: np.ndarray, w_b: float) -> np.ndarray:
        return np.clip(a.astype(np.float32) * w_a + b.astype(np.float32) * w_b, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, player) -> CueData:
        lat, lon  = player.lat, player.lon
        heading   = player.heading_deg
        elevation = player.elevation_deg

        dist_near, nearest = self._nearest(lat, lon)
        dist_2nd,  second  = self._second_nearest(lat, lon)
        dist_3rd,  third   = self._third_nearest(lat, lon)

        # Bearings player→anchor (for map display and blend weight)
        bear_to_n1 = bearing_between(lat, lon, nearest.lat, nearest.lon)
        bear_to_n2 = bearing_between(lat, lon, second.lat,  second.lon)

        # Node offsets from player in metres (East=X, North=Y)
        _lat_rad   = math.radians(lat)
        _mpdlat    = EARTH_RADIUS_M * math.pi / 180.0
        _mpdlon    = EARTH_RADIUS_M * math.cos(_lat_rad) * math.pi / 180.0
        n1_east_m  = (nearest.lon - lon) * _mpdlon
        n1_north_m = (nearest.lat - lat) * _mpdlat
        n2_east_m  = (second.lon  - lon) * _mpdlon
        n2_north_m = (second.lat  - lat) * _mpdlat
        n3_east_m  = (third.lon   - lon) * _mpdlon
        n3_north_m = (third.lat   - lat) * _mpdlat

        # Reference point: _LOOK_AHEAD_M metres ahead of the player along heading.
        # Each anchor crops toward this point so the picture responds to both
        # turning and forward/backward movement (parallax).
        h_rad   = math.radians(heading)
        lat_rad = math.radians(lat)
        ref_lat = lat + (_LOOK_AHEAD_M * math.cos(h_rad)) / EARTH_RADIUS_M * (180.0 / math.pi)
        ref_lon = lon + (_LOOK_AHEAD_M * math.sin(h_rad)) / (EARTH_RADIUS_M * math.cos(lat_rad)) * (180.0 / math.pi)

        # Calculate R_left
        h_left_rad = math.radians(heading - FOV_DEG / 2.0)
        ref_left_lat = lat + (_LOOK_AHEAD_M * math.cos(h_left_rad)) / EARTH_RADIUS_M * (180.0 / math.pi)
        ref_left_lon = lon + (_LOOK_AHEAD_M * math.sin(h_left_rad)) / (EARTH_RADIUS_M * math.cos(lat_rad)) * (180.0 / math.pi)

        # Calculate R_right
        h_right_rad = math.radians(heading + FOV_DEG / 2.0)
        ref_right_lat = lat + (_LOOK_AHEAD_M * math.cos(h_right_rad)) / EARTH_RADIUS_M * (180.0 / math.pi)
        ref_right_lon = lon + (_LOOK_AHEAD_M * math.sin(h_right_rad)) / (EARTH_RADIUS_M * math.cos(lat_rad)) * (180.0 / math.pi)

        n1_crop_hdg = bearing_between(nearest.lat, nearest.lon, ref_lat, ref_lon)
        n1_left_hdg = bearing_between(nearest.lat, nearest.lon, ref_left_lat, ref_left_lon)
        n1_right_hdg = bearing_between(nearest.lat, nearest.lon, ref_right_lat, ref_right_lon)

        n1_diff_left = abs((n1_left_hdg - n1_crop_hdg + 180.0) % 360.0 - 180.0)
        n1_diff_right = abs((n1_right_hdg - n1_crop_hdg + 180.0) % 360.0 - 180.0)
        n1_fov = max(10.0, min(170.0, n1_diff_left + n1_diff_right))

        n2_crop_hdg = bearing_between(second.lat, second.lon, ref_lat, ref_lon)
        n2_left_hdg = bearing_between(second.lat, second.lon, ref_left_lat, ref_left_lon)
        n2_right_hdg = bearing_between(second.lat, second.lon, ref_right_lat, ref_right_lon)

        n2_diff_left = abs((n2_left_hdg - n2_crop_hdg + 180.0) % 360.0 - 180.0)
        n2_diff_right = abs((n2_right_hdg - n2_crop_hdg + 180.0) % 360.0 - 180.0)
        n2_fov = max(10.0, min(170.0, n2_diff_left + n2_diff_right))

        n3_crop_hdg  = bearing_between(third.lat, third.lon, ref_lat, ref_lon)
        n3_left_hdg  = bearing_between(third.lat, third.lon, ref_left_lat, ref_left_lon)
        n3_right_hdg = bearing_between(third.lat, third.lon, ref_right_lat, ref_right_lon)
        n3_diff_left  = abs((n3_left_hdg  - n3_crop_hdg + 180.0) % 360.0 - 180.0)
        n3_diff_right = abs((n3_right_hdg - n3_crop_hdg + 180.0) % 360.0 - 180.0)
        n3_fov = max(10.0, min(170.0, n3_diff_left + n3_diff_right))

        crop_n1 = self._crop_at_heading(nearest.image_bgr, heading, nearest.compass_angle, elevation, fov=FOV_DEG)
        crop_n2 = self._crop_at_heading(second.image_bgr,  heading, second.compass_angle,  elevation, fov=FOV_DEG)
        crop_n3 = self._crop_at_heading(third.image_bgr,   heading, third.compass_angle,   elevation, fov=FOV_DEG)

        # Merged: parallax-corrected per-pixel blend from all 3 nodes
        anchor_crop = self._merged_crop(
            [
                (dist_near, nearest, n1_east_m, n1_north_m),
                (dist_2nd,  second,  n2_east_m, n2_north_m),
                (dist_3rd,  third,   n3_east_m, n3_north_m),
            ],
            heading, elevation,
        )

        second_crop = self._crop_at_heading(second.image_bgr, heading, second.compass_angle, elevation, fov=n2_fov)

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
            nearest_crop_fov=n1_fov,
            second_crop_fov=n2_fov,
            nearest_crop_hdg=heading,
            second_crop_hdg=heading,
            third_crop_hdg=heading,
            third_crop_isect=crop_n3,
            third_crop_fov=n3_fov,
            third_nearest_node_id=third.id,
            bear_to_nearest=bear_to_n1,
            bear_to_second=bear_to_n2,
            nearest_east_m=n1_east_m,
            nearest_north_m=n1_north_m,
            second_east_m=n2_east_m,
            second_north_m=n2_north_m,
            third_east_m=n3_east_m,
            third_north_m=n3_north_m,
        )
