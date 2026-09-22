"""
core/tracking.py
================
Sparse route tracking for target-point consumers (VLA / E2E backends).

A VLA such as SimLingo was trained on target points produced by the CARLA
leaderboard's ``RoutePlanner``: a *sparse* route (downsampled at ~50 m and at
every road-option change) from which passed points are popped as the ego
advances, with the model reading entries [1] and [2].

Feeding it dense 1 m waypoints instead would be off-distribution, so this
class reproduces that behaviour on top of our own :class:`GlobalPlan`.
"""

from __future__ import annotations

import math
from collections import deque

from .types import GlobalPlan, RoutePoint, world_to_ego


class SparseRouteTracker:
    """Pops route points the ego has passed and exposes the next few."""

    def __init__(self, min_distance: float = 7.5, max_distance: float = 50.0,
                 sample_factor: float = 50.0, behind_tolerance: float = 2.0) -> None:
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.sample_factor = sample_factor
        self.behind_tolerance = behind_tolerance
        self._route: deque[RoutePoint] = deque()
        self._revision: int | None = None

    # ── plan management ─────────────────────────────────────────────────
    def set_plan(self, plan: GlobalPlan | None) -> None:
        self._route.clear()
        if plan:
            self._route.extend(plan.downsample(self.sample_factor))
            self._revision = plan.revision

    def is_stale(self, plan: GlobalPlan | None) -> bool:
        rev = plan.revision if plan else None
        return rev != self._revision

    def __len__(self) -> int:
        return len(self._route)

    # ── per-tick update ─────────────────────────────────────────────────
    def run_step(self, x: float, y: float, yaw_deg: float | None = None) -> None:
        """
        Drop points the ego has driven past.

        The primary rule is the leaderboard's (pop points come within
        ``min_distance``), so the sparsity a VLA sees matches its training
        distribution. That rule alone is not enough here: it only fires when
        the ego passes *close* to a point, so a replan, a respawn or a lane
        offset can leave a point sitting behind the car — and a target point
        behind the ego is the classic way to make a VLA steer backwards. When
        ``yaw_deg`` is given, any leading point that has fallen behind is
        dropped as well.
        """
        if len(self._route) <= 1:
            return

        to_pop = 0
        farthest_in_range = -math.inf
        cumulative = 0.0
        route = list(self._route)

        for i in range(1, len(route)):
            if cumulative > self.max_distance:
                break
            cumulative += route[i].distance_to(route[i - 1])
            distance = math.hypot(route[i].x - x, route[i].y - y)
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self._route) > 2:
                self._route.popleft()

        if yaw_deg is not None:
            self._drop_points_behind(x, y, yaw_deg)

    def _drop_points_behind(self, x: float, y: float, yaw_deg: float) -> None:
        """Pop upcoming points that are behind the ego's forward direction."""
        fx = math.cos(math.radians(yaw_deg))
        fy = math.sin(math.radians(yaw_deg))
        while len(self._route) > 2:
            nxt = self._route[1]
            ahead = (nxt.x - x) * fx + (nxt.y - y) * fy
            if ahead >= -self.behind_tolerance:
                break
            self._route.popleft()

    # ── read-out ────────────────────────────────────────────────────────
    def next_points(self, n: int = 2) -> list[RoutePoint]:
        """The next ``n`` sparse route points, indices [1..n] like the leaderboard."""
        if not self._route:
            return []
        route = list(self._route)
        return [route[min(i, len(route) - 1)] for i in range(1, n + 1)]

    def target_points_ego(self, ego_transform, n: int = 2,
                          convention: str = "forward_right",
                          fallback: tuple[float, float] = (5.0, 10.0)):
        """``numpy [n, 2]`` — ready for SimLingo's ``target_points_ego``."""
        import numpy as np

        pts = self.next_points(n)
        if not pts:
            return np.array([[d, 0.0] for d in fallback[:n]], dtype=np.float32)
        return np.asarray(
            [world_to_ego(ego_transform, p.x, p.y, convention) for p in pts],
            dtype=np.float32,
        )
