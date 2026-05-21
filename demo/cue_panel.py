import math
import numpy as np
import cv2
import pygame

from config import (
    PLAYER_COLOR,
    PANEL_PADDING, PANEL_GAP, PANEL_W, PANEL_H,
    FOV_DEG, WINDOW_WIDTH, WINDOW_HEIGHT,
)

_HEADER_H   = 22   # px reserved for panel title
_BORDER_COL = (55, 55, 75)
_BG_COL     = (18, 18, 30)
_TEXT_COL   = (230, 230, 230)
_DIM_COL    = (110, 110, 130)
_NONE_COL   = (80, 80, 100)


def _bgr_to_surface(bgr: np.ndarray) -> pygame.Surface:
    arr = np.ascontiguousarray(bgr[:, :, ::-1].transpose(1, 0, 2))
    return pygame.surfarray.make_surface(arr)


class CuePanel:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self._panel_origins: list[tuple[int, int]] = []
        for row in range(3):
            for col in range(2):
                px = PANEL_PADDING + col * (PANEL_W + PANEL_GAP)
                py = PANEL_PADDING + row * (PANEL_H + PANEL_GAP)
                self._panel_origins.append((px, py))

        pygame.font.init()
        self._f_label = pygame.font.SysFont("monospace", 11, bold=True)
        self._f_data  = pygame.font.SysFont("monospace", 11)
        self._f_tag   = pygame.font.SysFont("monospace", 10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _panel_rect(self, idx: int) -> tuple[int, int, int, int]:
        rx, ry = self._panel_origins[idx]
        return self.x + rx, self.y + ry, PANEL_W, PANEL_H

    def _draw_bg(self, surf, rx, ry) -> None:
        pygame.draw.rect(surf, _BG_COL, (rx, ry, PANEL_W, PANEL_H))
        pygame.draw.rect(surf, _BORDER_COL, (rx, ry, PANEL_W, PANEL_H), 1)

    def _header(self, surf, rx, ry, title: str) -> int:
        """Draw slim title bar; return y of content area."""
        pygame.draw.rect(surf, (28, 28, 44), (rx, ry, PANEL_W, _HEADER_H))
        pygame.draw.line(surf, _BORDER_COL, (rx, ry + _HEADER_H), (rx + PANEL_W, ry + _HEADER_H), 1)
        t = self._f_label.render(title, True, _TEXT_COL)
        surf.blit(t, (rx + 6, ry + (_HEADER_H - t.get_height()) // 2))
        return ry + _HEADER_H + 2   # content_y

    def _blit_image(self, surf, img_bgr: np.ndarray, rx, ry, w, h) -> None:
        if w <= 0 or h <= 0 or img_bgr is None or img_bgr.size == 0:
            return
        scaled = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        surf.blit(_bgr_to_surface(scaled), (rx, ry))

    def _tag(self, surf, text: str, rx, ry, w, h) -> None:
        """Small label overlaid at bottom-right of an image area."""
        t = self._f_tag.render(text, True, (255, 255, 255))
        bg = pygame.Surface((t.get_width() + 6, t.get_height() + 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        bx = rx + w - bg.get_width() - 3
        by = ry + h - bg.get_height() - 3
        surf.blit(bg, (bx, by))
        surf.blit(t, (bx + 3, by + 1))

    def _no_data(self, surf, rx, ry, msg="No data") -> None:
        t = self._f_tag.render(msg, True, _NONE_COL)
        cx = rx + PANEL_W // 2 - t.get_width() // 2
        cy = ry + PANEL_H // 2 - t.get_height() // 2
        surf.blit(t, (cx, cy))

    # ------------------------------------------------------------------
    # Panel 0 — Merged intersection crop  (main world-model input)
    # ------------------------------------------------------------------

    def _panel0(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, "Merged view  ·  nearest ∩ 2nd-nearest ∩ player FOV")
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        self._blit_image(surf, cd.anchor_crop, rx + 2, cy, iw, ih)
        self._tag(surf, f"{cd.nearest_node_id}  {cd.nearest_node_dist_m:.1f} m", rx + 2, cy, iw, ih)

    # ------------------------------------------------------------------
    # Panel 1 — Nearest node at intersection
    # ------------------------------------------------------------------

    @staticmethod
    def _cam_axes(hdg_deg: float, pitch_deg: float):
        import math as _m
        H = _m.radians(hdg_deg);   P = _m.radians(pitch_deg)
        ch, sh = _m.cos(H), _m.sin(H)
        cp, sp = _m.cos(P), _m.sin(P)
        right = ( ch,       -sh,       0.0)
        fwd   = ( sh * cp,   ch * cp,  sp)
        up    = (-sh * sp,  -ch * sp,  cp)
        return right, up, fwd

    _SCENE_DEPTH_M = 40.0   # assumed scene depth for parallax projection

    def _frustum_quad_in_panel(self,
                               heading_deg: float, pitch_deg: float,
                               crop_hdg_deg: float, crop_elev_deg: float,
                               node_east_m: float, node_north_m: float,
                               rx: int, cy: int, pw: int, ph: int,
                               color: tuple, surf) -> None:
        """
        For each of the 4 player screen-corner rays:
          1. Trace from the player's position to a world point at SCENE_DEPTH_M.
          2. Re-project that world point from the NODE's position into its panel.

        This correctly accounts for the spatial offset between player and node,
        producing an outline that matches the minimap prismatoid intersection.
        """
        import math as _m

        p_right, p_up, p_fwd = self._cam_axes(heading_deg, pitch_deg)
        n_right, n_up, n_fwd = self._cam_axes(crop_hdg_deg, crop_elev_deg)

        tan_h = _m.tan(_m.radians(FOV_DEG / 2.0))
        tan_v = _m.tan(_m.radians(FOV_DEG / 2.0) * WINDOW_HEIGHT / WINDOW_WIDTH)
        tan_n = _m.tan(_m.radians(FOV_DEG / 2.0))

        pts = []
        for ccx, ccy in [(-tan_h, -tan_v), (tan_h, -tan_v),
                          (tan_h,  tan_v), (-tan_h,  tan_v)]:
            # Player frustum corner ray direction
            wx = ccx * p_right[0] + ccy * p_up[0] + p_fwd[0]
            wy = ccx * p_right[1] + ccy * p_up[1] + p_fwd[1]
            wz = ccx * p_right[2] + ccy * p_up[2] + p_fwd[2]

            # Forward component along player's look direction (for depth scaling)
            fwd_comp = wx * p_fwd[0] + wy * p_fwd[1] + wz * p_fwd[2]
            if fwd_comp <= 1e-6:
                return

            # Trace ray to scene depth — world offset from player (East, North, Up)
            t = self._SCENE_DEPTH_M / fwd_comp
            world_x = wx * t   # East
            world_y = wy * t   # North
            world_z = wz * t   # Up

            # Vector from NODE to that world point
            rel_x = world_x - node_east_m
            rel_y = world_y - node_north_m
            rel_z = world_z  # node at same height as player

            # Project into node's camera
            cam_x = rel_x * n_right[0] + rel_y * n_right[1] + rel_z * n_right[2]
            cam_y = rel_x * n_up[0]    + rel_y * n_up[1]    + rel_z * n_up[2]
            cam_z = rel_x * n_fwd[0]   + rel_y * n_fwd[1]   + rel_z * n_fwd[2]

            if cam_z <= 1e-6:
                return

            px = int(( cam_x / cam_z / tan_n * 0.5 + 0.5) * pw)
            py = int((-cam_y / cam_z / tan_n * 0.5 + 0.5) * ph)
            pts.append((rx + px, cy + py))

        pygame.draw.polygon(surf, color, pts, 2)

    def _panel1(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, f"Nearest  ·  {cd.nearest_node_id}  ·  {cd.nearest_crop_fov:.0f}°")
        crop = cd.nearest_crop_isect
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        if crop is None or crop.size == 0:
            self._no_data(surf, rx, ry + _HEADER_H)
            return
        self._blit_image(surf, crop, rx + 2, cy, iw, ih)
        self._frustum_quad_in_panel(cd.heading_deg, cd.elevation_deg,
                                    cd.nearest_crop_hdg, cd.elevation_deg,
                                    cd.nearest_east_m, cd.nearest_north_m,
                                    rx + 2, cy, iw, ih, (255, 200, 50), surf)
        self._tag(surf, f"{cd.nearest_node_dist_m:.1f} m", rx + 2, cy, iw, ih)

    # ------------------------------------------------------------------
    # Panel 2 — Second nearest node at intersection
    # ------------------------------------------------------------------

    def _panel2(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, f"2nd nearest  ·  {cd.second_nearest_node_id}  ·  {cd.second_crop_fov:.0f}°")
        crop = cd.second_crop_isect
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        if crop is None or crop.size == 0:
            self._no_data(surf, rx, ry + _HEADER_H)
            return
        self._blit_image(surf, crop, rx + 2, cy, iw, ih)
        self._frustum_quad_in_panel(cd.heading_deg, cd.elevation_deg,
                                    cd.second_crop_hdg, cd.elevation_deg,
                                    cd.second_east_m, cd.second_north_m,
                                    rx + 2, cy, iw, ih, (60, 200, 140), surf)

    # ------------------------------------------------------------------
    # Panel 3 — Third nearest node at intersection
    # ------------------------------------------------------------------

    def _panel3(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, f"3rd nearest  ·  {cd.third_nearest_node_id}  ·  {cd.third_crop_fov:.0f}°")
        crop = cd.third_crop_isect
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        if crop is None or crop.size == 0:
            self._no_data(surf, rx, ry + _HEADER_H)
            return
        self._blit_image(surf, crop, rx + 2, cy, iw, ih)
        self._frustum_quad_in_panel(cd.heading_deg, cd.elevation_deg,
                                    cd.third_crop_hdg, cd.elevation_deg,
                                    cd.third_east_m, cd.third_north_m,
                                    rx + 2, cy, iw, ih, (60, 140, 220), surf)

    # ------------------------------------------------------------------
    # Panel 4 — Movement / compass
    # ------------------------------------------------------------------

    def _panel4(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, "Movement vector")

        # Compass
        r = min(PANEL_H // 2 - _HEADER_H - 10, PANEL_W // 4 - 4)
        cx_c = rx + PANEL_W // 4
        cy_c = cy + r + 4

        pygame.draw.circle(surf, (28, 28, 44), (cx_c, cy_c), r)
        pygame.draw.circle(surf, _BORDER_COL, (cx_c, cy_c), r, 1)

        for label, az in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            rad = math.radians(az - 90)
            tx = cx_c + int((r - 11) * math.cos(rad))
            ty = cy_c + int((r - 11) * math.sin(rad))
            t = self._f_tag.render(label, True, _DIM_COL)
            surf.blit(t, (tx - t.get_width() // 2, ty - t.get_height() // 2))

        arrow_rad = math.radians(cd.heading_deg - 90)
        ax = cx_c + int((r - 8) * math.cos(arrow_rad))
        ay = cy_c + int((r - 8) * math.sin(arrow_rad))
        pygame.draw.line(surf, PLAYER_COLOR, (cx_c, cy_c), (ax, ay), 2)
        pygame.draw.circle(surf, PLAYER_COLOR, (ax, ay), 3)

        # Stats
        tx = rx + PANEL_W // 2 + 4
        ty = cy + 4
        for line in (
            f"Hdg  {cd.heading_deg:06.2f}°",
            f"Spd  {cd.speed_mps:.1f} m/s",
            f"dX   {cd.dx_m:+.2f} m",
            f"dY   {cd.dy_m:+.2f} m",
            f"ΔHdg {cd.delta_heading_deg:+.1f}°",
        ):
            if ty + 13 > ry + PANEL_H - 4:
                break
            surf.blit(self._f_data.render(line, True, _TEXT_COL), (tx, ty))
            ty += 14

    # ------------------------------------------------------------------
    # Panel 5 — Warped reference
    # ------------------------------------------------------------------

    def _panel5(self, surf, rx, ry, cd) -> None:
        cy = self._header(surf, rx, ry, "Warped reference  ·  pose-shifted entry frame")
        iw = PANEL_W - 4
        ih = ry + PANEL_H - cy - 2
        if cd.warped_frame is None:
            self._no_data(surf, rx, ry + _HEADER_H, "Waiting…")
            return
        self._blit_image(surf, cd.warped_frame, rx + 2, cy, iw, ih)

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(self, surface, cue_data) -> None:
        pygame.draw.rect(surface, (12, 12, 22), (self.x, self.y, self.width, self.height))

        renderers = [
            self._panel0,   # merged main
            self._panel1,   # nearest ∩
            self._panel2,   # 2nd nearest ∩
            self._panel3,   # 2nd nearest full heading
            self._panel4,   # compass / movement
            self._panel5,   # warped reference
        ]

        for idx, renderer in enumerate(renderers):
            rx, ry, _, _ = self._panel_rect(idx)
            self._draw_bg(surface, rx, ry)
            renderer(surface, rx, ry, cue_data)
