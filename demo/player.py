import math
import pygame
from config import TURN_RATE, ELEVATION_RATE, MAX_SPEED, EARTH_RADIUS_M

_ELEV_MAX = 75.0   # degrees up
_ELEV_MIN = -75.0  # degrees down


class Player:
    def __init__(self, lat: float, lon: float, heading: float = 0.0):
        self.lat = lat
        self.lon = lon
        self.heading_deg   = heading
        self.elevation_deg = 0.0
        self.speed_mps     = 0.0
        self.dx_m          = 0.0
        self.dy_m          = 0.0
        self.delta_heading = 0.0
        self._init_lat     = lat
        self._init_lon     = lon
        self._init_heading = heading

    def update(self, dt: float, keys) -> None:
        prev_heading = self.heading_deg

        # Horizontal turn
        if keys[pygame.K_a]:
            self.heading_deg = (self.heading_deg - TURN_RATE * dt) % 360.0
        if keys[pygame.K_d]:
            self.heading_deg = (self.heading_deg + TURN_RATE * dt) % 360.0

        # Vertical look
        if keys[pygame.K_UP]:
            self.elevation_deg = min(self.elevation_deg + ELEVATION_RATE * dt, _ELEV_MAX)
        if keys[pygame.K_DOWN]:
            self.elevation_deg = max(self.elevation_deg - ELEVATION_RATE * dt, _ELEV_MIN)

        # Forward / backward
        if keys[pygame.K_w]:
            self.speed_mps = MAX_SPEED
        elif keys[pygame.K_s]:
            self.speed_mps = -MAX_SPEED
        else:
            self.speed_mps = 0.0

        dist        = self.speed_mps * dt
        heading_rad = math.radians(self.heading_deg)
        lat_rad     = math.radians(self.lat)

        self.lat += (dist * math.cos(heading_rad)) / EARTH_RADIUS_M * (180.0 / math.pi)
        cos_lat = math.cos(lat_rad)
        if cos_lat > 1e-9:
            self.lon += (dist * math.sin(heading_rad)) / (EARTH_RADIUS_M * cos_lat) * (180.0 / math.pi)

        self.dx_m          = dist * math.sin(heading_rad)
        self.dy_m          = dist * math.cos(heading_rad)
        self.delta_heading = (self.heading_deg - prev_heading + 180.0) % 360.0 - 180.0

    def reset(self) -> None:
        self.lat           = self._init_lat
        self.lon           = self._init_lon
        self.heading_deg   = self._init_heading
        self.elevation_deg = 0.0
        self.speed_mps     = 0.0
        self.dx_m = self.dy_m = self.delta_heading = 0.0
