"""
runtime/camera.py
=================
Drives the CARLA spectator: a top-down view that follows the ego while there
is no route, and pulls back to frame the whole town once one is planned.
A large town would put that view kilometres up, above everything CARLA has
streamed in; there it frames the ego and what was just planned (the route,
or the candidates to pick from) instead, no higher than ``max_height_m``.

Both views look straight down with the same yaw, so switching between them
is a pure zoom -- the map never rotates under the viewer.
"""

from __future__ import annotations

import math
import threading
import time

import carla

HFOV_DEG = 90.0          # the spectator's default horizontal field of view
ASPECT = 16.0 / 9.0      # CARLA's default window


class SpectatorCamera:
    def __init__(self, world, vehicle, ego_height: float = 50.0,
                 transition_s: float = 2.0, margin: float = 1.1,
                 large_above_m: float = 1500.0, max_height_m: float = 800.0) -> None:
        self.spectator = world.get_spectator()
        self.vehicle = vehicle
        self.ego_height = ego_height
        self.transition_s = transition_s
        self.margin = margin
        self.max_height_m = max_height_m
        self._town, self.yaw, span = _town_view(world.get_map(), margin)
        self.large = span > large_above_m
        self._framed: tuple[float, float, float] | None = None   # large town: what to show

        self._lock = threading.Lock()
        self.mode = "ego"
        self._from: tuple[float, float, float] | None = None   # pose a transition starts at
        self._t0 = 0.0
        self._pose: tuple[float, float, float] | None = None   # last pose set

    # ── mode switches (command thread) ──────────────────────────────────
    def follow_ego(self) -> None:
        self._switch("ego")

    def show_town(self, points=None) -> None:
        """Pull back: the whole town, or in a large one the ego and
        ``points`` ((x, y, z) of the route or the candidates)."""
        if self.large:
            self._framed = self._frame(points or ())
        self._switch("town")

    def toggle(self) -> None:
        self._switch("town" if self.mode == "ego" else "ego")

    def _switch(self, mode: str) -> None:
        with self._lock:
            if mode == self.mode:
                return
            self._from = self._pose
            self._t0 = time.time()
            self.mode = mode

    # ── per tick (control thread) ───────────────────────────────────────
    def update(self) -> None:
        with self._lock:
            mode, start, t0 = self.mode, self._from, self._t0

        if mode == "town" and not self.large:
            target = self._town
        elif mode == "town" and self._framed is not None:
            target = self._framed
        else:
            loc = self.vehicle.get_location()
            up = self.max_height_m if mode == "town" else self.ego_height
            target = (loc.x, loc.y, loc.z + up)

        pose = target
        if start is not None:
            a = (time.time() - t0) / max(1e-3, self.transition_s)
            if a >= 1.0:
                with self._lock:
                    if self._t0 == t0:
                        self._from = None
            else:
                s = a * a * (3.0 - 2.0 * a)                      # smoothstep
                # Height eases geometrically, so the zoom feels uniform.
                z = math.exp(math.log(max(1.0, start[2])) * (1 - s)
                             + math.log(max(1.0, target[2])) * s)
                pose = (start[0] + (target[0] - start[0]) * s,
                        start[1] + (target[1] - start[1]) * s, z)

        self._pose = pose
        self.spectator.set_transform(carla.Transform(
            carla.Location(x=pose[0], y=pose[1], z=pose[2]),
            carla.Rotation(pitch=-90.0, yaw=self.yaw)))


    def _frame(self, points) -> tuple[float, float, float] | None:
        """A straight-down pose over the ego and ``points`` at this camera's
        yaw, or None if that would be higher than ``max_height_m`` -- the
        view then stays over the ego at that height."""
        loc = self.vehicle.get_location()
        pts = [(loc.x, loc.y, loc.z), *((p[0], p[1], p[2] if len(p) > 2 else loc.z)
                                         for p in points)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        half_x, half_y = (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2
        tan_h = math.tan(math.radians(HFOV_DEG / 2))
        tan_v = tan_h / ASPECT
        across, along = (half_x, half_y) if self.yaw == -90.0 else (half_y, half_x)
        dist = max(across / tan_h, along / tan_v, self.ego_height) * self.margin
        if dist > self.max_height_m:
            return None
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                max(p[2] for p in pts) + dist)


def _town_view(carla_map, margin: float) -> tuple[tuple[float, float, float], float, float]:
    """Pose (x, y, z) and yaw of a straight-down view that frames every road,
    and the town's larger extent in metres."""
    locs = [w.transform.location for w in carla_map.generate_waypoints(5.0)]
    xs = [l.x for l in locs]
    ys = [l.y for l in locs]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_x, half_y = (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2

    tan_h = math.tan(math.radians(HFOV_DEG / 2))
    tan_v = tan_h / ASPECT
    # Looking straight down, yaw -90 puts +x to screen-right and yaw 0 puts
    # +y there. Lay the map's longer side along the wider screen axis.
    x_across = max(half_x / tan_h, half_y / tan_v)
    y_across = max(half_y / tan_h, half_x / tan_v)
    yaw, dist = (-90.0, x_across) if x_across <= y_across else (0.0, y_across)
    return (cx, cy, max(l.z for l in locs) + dist * margin), yaw, 2 * max(half_x, half_y)
