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
