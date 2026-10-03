"""
core/route_provider.py
======================
Turns (start, goal) into a dense :class:`GlobalPlan`, and snaps landmark
coordinates onto the road network.

``CarlaRouteProvider`` is the only implementation today, but keeping this
behind an interface is what lets the mission layer be reused on a different
map backend (lanelet2, nuPlan, a recorded route file) later — the backends
downstream never see the router at all.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .types import RoutePoint


class RouteProvider(Protocol):
    def snap(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Project a raw coordinate onto the drivable road network."""
        ...

    def trace(self, start: Sequence[float], goal: Sequence[float]) -> list[RoutePoint]:
        """Dense route from ``start`` to ``goal``; empty list if unreachable."""
        ...

    def trace_chain(self, points: Sequence[Sequence[float]]) -> list[RoutePoint]:
        """Dense route through a keypoint chain, as one logical plan."""
        ...

    def reset(self) -> None:
        """Forget any per-plan router state. See CarlaRouteProvider.reset."""
        ...


class CarlaRouteProvider:
    """Wraps CARLA's ``GlobalRoutePlanner``."""

    def __init__(self, carla_map, sampling_resolution: float = 1.0) -> None:
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        self.map = carla_map
        self._grp = GlobalRoutePlanner(carla_map, sampling_resolution=sampling_resolution)

    def reset(self) -> None:
        """Clear the router's carry-over turn state.

        ``GlobalRoutePlanner`` keeps ``_previous_decision`` and
        ``_intersection_end_node`` on the *instance* and reads them in
        ``_turn_decision``, so the RoadOption labels it returns depend on which
        routes it happened to trace before. The leaderboard never notices:
        ``interpolate_trajectory`` builds a throwaway planner per call. We keep
        one planner for the whole session, so without this a re-plan can label
        the same geometry LANEFOLLOW where the benchmark labels it STRAIGHT --
        and RoadOption is exactly what a backend eats as its high-level
        command.

        Call this once per logical plan, never between the legs of one plan:
        the reference route carries state across its own legs and we have to
        reproduce that too.
        """
        from agents.navigation.local_planner import RoadOption

        self._grp._intersection_end_node = -1
        self._grp._previous_decision = RoadOption.VOID

    def snap(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        import carla

        wp = self.map.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=True)
        loc = wp.transform.location
        return (loc.x, loc.y, loc.z)

    def snap_heading(self, x: float, y: float, z: float, yaw_deg: float,
                     ahead_m: float = 0.0) -> tuple[float, float, float]:
        """Snap onto the nearest driving lane that runs the way the ego faces,
        then move ``ahead_m`` along that lane.

        Plain :meth:`snap` takes the nearest lane whatever its direction, so a
        car that has drifted over the centre line gets put on the oncoming
        lane and is routed back the way it came. Moving ahead along the lane
        rather than along the car's heading keeps a crooked car from
        projecting its start across the line too.
        """
        import carla

        loc = carla.Location(x=x, y=y, z=z)
        wp = self.map.get_waypoint(loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        if wp is None:
            return self.snap(x, y, z)
        if not _aligned(wp, yaw_deg):
            wp = (_aligned_neighbour(wp, yaw_deg, loc)
                  or _aligned_across(self.map, loc, yaw_deg) or wp)
        if ahead_m > 0:
            nxt = wp.next(ahead_m)
            if nxt:
                wp = min(nxt, key=lambda w: _yaw_diff(w.transform.rotation.yaw, yaw_deg))
        out = wp.transform.location
        return (out.x, out.y, out.z)

    def trace(self, start: Sequence[float], goal: Sequence[float]) -> list[RoutePoint]:
        import carla

        s = carla.Location(x=float(start[0]), y=float(start[1]), z=float(start[2]))
        g = carla.Location(x=float(goal[0]), y=float(goal[1]), z=float(goal[2]))
        try:
            raw = self._grp.trace_route(s, g)
        except Exception as exc:                      # unreachable / off-road
            print(f"[route] trace_route failed: {exc}")
            return []
        out = []
        for wp, option in raw:
            tf = wp.transform
            out.append(RoutePoint(x=tf.location.x, y=tf.location.y, z=tf.location.z,
                                  yaw=tf.rotation.yaw, option=int(option.value)))
        return out

    def trace_chain(self, points: Sequence[Sequence[float]]) -> list[RoutePoint]:
        """Trace ``points[0] -> points[1] -> ... -> points[-1]`` as ONE plan.

        Mirrors ``leaderboard/utils/route_manipulation.py::
        interpolate_trajectory``: state is reset once at the front, then each
        leg is traced in order and concatenated. Returns [] if any leg fails.
        """
        pts = [tuple(p) for p in points]
        if len(pts) < 2:
            return []
        self.reset()
        out: list[RoutePoint] = []
        for a, b in zip(pts, pts[1:]):
            leg = self.trace(a, b)
            if not leg:
                return []
            out.extend(leg)
        return out


def _yaw_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _aligned(wp, yaw_deg: float) -> bool:
    return _yaw_diff(wp.transform.rotation.yaw, yaw_deg) < 90.0


def _aligned_neighbour(wp, yaw_deg: float, loc, max_hops: int = 4):
    """Nearest driving lane beside ``wp`` (either side, across the centre
    line too) that runs within 90 degrees of ``yaw_deg``; None if none does."""
    import carla

    seen = {(wp.road_id, wp.section_id, wp.lane_id)}
    frontier, found = [wp], []
    for _ in range(max_hops):
        nxt = []
        for w in frontier:
            for side in (w.get_left_lane(), w.get_right_lane()):
                if side is None:
                    continue
                key = (side.road_id, side.section_id, side.lane_id)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(side)
                if side.lane_type == carla.LaneType.Driving and _aligned(side, yaw_deg):
                    found.append(side)
        frontier = nxt
    if not found:
        return None
    return min(found, key=lambda w: w.transform.location.distance(loc))


def _aligned_across(carla_map, loc, yaw_deg: float, half_width_m: float = 6.0,
                    step_m: float = 0.5):
    """Fallback for where lanes have no neighbours to walk (junction
    connectors, road ends): probe sideways across the car's heading and keep
    the nearest driving lane running its way."""
    import math

    import carla

    rx = -math.sin(math.radians(yaw_deg))          # the car's right, in CARLA's frame
    ry = math.cos(math.radians(yaw_deg))
    found = []
    n = int(half_width_m / step_m)
    for i in range(-n, n + 1):
        probe = carla.Location(x=loc.x + rx * i * step_m, y=loc.y + ry * i * step_m, z=loc.z)
        w = carla_map.get_waypoint(probe, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        if w is not None and _aligned(w, yaw_deg):
            found.append(w)
    if not found:
        return None
    return min(found, key=lambda w: w.transform.location.distance(loc))
