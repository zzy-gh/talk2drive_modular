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

import math
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

    def route_length(self, start: Sequence[float], goal: Sequence[float]) -> float:
        """Metres driven from ``start`` to ``goal`` along the road network;
        ``math.inf`` if unreachable. Optional: without it the mission layer
        ranks places by straight-line distance."""
        ...

    def route_lengths(self, start: Sequence[float],
                      goals: Sequence[Sequence[float]]) -> list[float]:
        """Metres from ``start`` to each of ``goals`` in one graph search;
        ``math.inf`` for an unreachable one. Optional, like route_length."""
        ...

    def route_lengths_to(self, starts: Sequence[Sequence[float]],
                         goal: Sequence[float]) -> list[float]:
        """Metres from each of ``starts`` to ``goal`` in one graph search."""
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
        self._grp._find_closest_in_list = _closest_in_list_cached()
        self._road_graph: _RoadGraph | None = None       # built on first use

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
        if self._stalled(raw, g):
            # GlobalRoutePlanner cannot route to a goal behind the start on
            # the same lane segment: it returns the start alone, and the next
            # leg then begins out of nowhere -- the car turns round to chase
            # it. Drive on to the end of this lane first; from the next
            # junction the planner finds the way round.
            ahead = self.map.get_waypoint(s).next_until_lane_end(2.0)
            if ahead:
                nxt = ahead[-1].next(5.0)
                mid = (nxt[0] if nxt else ahead[-1]).transform.location
                try:
                    raw = self._grp.trace_route(s, mid) + self._grp.trace_route(mid, g)[1:]
                except Exception as exc:
                    print(f"[route] trace_route failed: {exc}")
                    return []
            if self._stalled(raw, g):
                print(f"[route] no route from ({s.x:.0f}, {s.y:.0f}) to ({g.x:.0f}, {g.y:.0f})")
                return []
        out = []
        for wp, option in raw:
            tf = wp.transform
            out.append(RoutePoint(x=tf.location.x, y=tf.location.y, z=tf.location.z,
                                  yaw=tf.rotation.yaw, option=int(option.value)))
        return out

    def route_length(self, start: Sequence[float], goal: Sequence[float]) -> float:
        """Metres along the route :meth:`trace` would plan from ``start`` to
        ``goal``, plus the straight-line gaps between ``start`` and the route's
        first point and between its last point and ``goal``; ``math.inf`` if
        it finds no route.

        The gaps matter: GlobalRoutePlanner starts a route up to a segment
        from ``start`` and now and then ends one far short of ``goal``.
        Counting them keeps the result >= the straight line (triangle
        inequality), which the mission layer's top-k cut relies on, and
        charges a route that stops short instead of ranking it nearest.

        Tracing advances the router's turn state (see :meth:`reset`), so that
        state is saved and restored: measuring candidates must not change the
        RoadOption labels of the plan traced after them.
        """
        saved = (self._grp._intersection_end_node, self._grp._previous_decision)
        try:
            pts = self.trace(start, goal)
        finally:
            self._grp._intersection_end_node, self._grp._previous_decision = saved
        if not pts:
            return math.inf
        first, last = pts[0], pts[-1]
        return (math.hypot(first.x - start[0], first.y - start[1])
                + sum(a.distance_to(b) for a, b in zip(pts, pts[1:]))
                + math.hypot(goal[0] - last.x, goal[1] - last.y))

    def route_lengths(self, start: Sequence[float],
                      goals: Sequence[Sequence[float]]) -> list[float]:
        """Metres from ``start`` to each of ``goals``, all from one Dijkstra
        over the planner's lane graph -- the cost of the search does not grow
        with the number of goals, which a call to :meth:`route_length` per goal
        does. The lengths are of the shortest path in metres, which is not
        always the path :meth:`trace` plans: GlobalRoutePlanner's A* weighs an
        edge by its waypoint count. ``math.inf`` for an unreachable goal."""
        return self._lane_graph().lengths_from(start, goals)

    def route_lengths_to(self, starts: Sequence[Sequence[float]],
                         goal: Sequence[float]) -> list[float]:
        """Metres from each of ``starts`` to ``goal``: one Dijkstra on the
        reversed lane graph. See :meth:`route_lengths`."""
        return self._lane_graph().lengths_to(starts, goal)

    def _lane_graph(self) -> "_RoadGraph":
        if self._road_graph is None:
            self._road_graph = _RoadGraph(self._grp)
        return self._road_graph

    def _stalled(self, raw, goal, tol_m: float = 10.0) -> bool:
        """trace_route went nowhere -- the start alone, or points that never
        leave it -- while the goal (projected onto the road, as trace_route
        targets it) is elsewhere. Only this case is touched: a trace that runs
        on but ends short of its goal is left exactly as GlobalRoutePlanner
        made it."""
        if not raw:
            return True
        start = raw[0][0].transform.location
        target = self.map.get_waypoint(goal).transform.location
        moved = max(wp.transform.location.distance(start) for wp, _ in raw)
        return moved < 2.0 and start.distance(target) > tol_m

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


def _closest_in_list_cached():
    """A drop-in for ``GlobalRoutePlanner._find_closest_in_list``.

    ``trace_route`` calls it on every edge of a route to find where to join
    that edge's waypoints, and the original walks the list in Python, reading
    each waypoint's location through the C++ binding. That is about 90 % of a
    trace, and it grows with route length: 17 ms median and up to seconds per
    trace on Town12/13, whose routes run for kilometres. Here each list's
    coordinates are read once into an array and searched vectorised.

    Same answer: the first index of the minimum distance, in float32 as the
    binding computes it. The lists it is given are, or are built from, the
    planner's own edge waypoints, which live as long as the planner, so the
    ids of their ends and their length key the cache.
    """
    import numpy as np

    cache: dict[tuple, "np.ndarray"] = {}

    def find(current_waypoint, waypoint_list) -> int:
        if not waypoint_list:
            return -1
        key = (id(waypoint_list[0]), id(waypoint_list[-1]), len(waypoint_list))
        xyz = cache.get(key)
        if xyz is None:
            xyz = np.array([(l.x, l.y, l.z) for l in
                            (w.transform.location for w in waypoint_list)], dtype=np.float32)
            cache[key] = xyz
        c = current_waypoint.transform.location
        d = xyz - np.array((c.x, c.y, c.z), dtype=np.float32)
        return int(np.sqrt(np.einsum("ij,ij->i", d, d)).argmin())

    return find


class _RoadGraph:
    """GlobalRoutePlanner's lane graph, weighted in metres, for one-to-many
    driving distances.

    The planner's own edge ``length`` is a waypoint count, and a place sits
    part-way along an edge, so neither its nodes nor its weights give metres
    directly. Here every lane-follow edge keeps its polyline (entry, path,
    exit) with cumulative arc length, a position is (edge, metres from the
    edge's entry), and a search is seeded from the start's position rather
    than from a node:

    * from ``start`` on edge (u, v) at offset s, node v is reached after
      ``L(u, v) - s``; a goal at offset g on edge (a, b) costs
      ``dist(a) + g``, or ``g - s`` when it is ahead on the start's own edge;
    * the planner allows a lane change only at an edge's entry, at no cost,
      to the entry of the neighbour lane's edge -- which may start well
      ahead; such a change is charged the distance it jumps ahead. A car
      part-way along a multi-lane road can change lanes too, so the start is
      also placed on each neighbouring lane at the same fraction of its
      length, and the goal likewise for the reverse search.

    The straight-line gaps between a coordinate and the polyline point it
    is matched to are added at both ends, as in ``route_length``.

    ``SAME_SPOT_M`` is how close counts as the same spot: a goal that close
    behind the start on its lane is where the car already is, not a loop
    round the block (as ``trace`` sees it too); a start that close to its
    edge's entry can still take any branch of that fork, and a goal that
    close to its edge's exit is reached from any lane that joins there.
    """

    SAME_SPOT_M = 5.0

    def __init__(self, grp) -> None:
        import numpy as np
        from agents.navigation.local_planner import RoadOption

        self._grp = grp
        self._geom: dict[tuple, tuple] = {}     # edge -> (xy[n, 2], cum[n])
        self.succ: dict[int, list] = {}         # node -> [(node, metres)]
        self.pred: dict[int, list] = {}
        self._changes: dict[tuple, list] = {}   # edge -> neighbour lanes it can change into
        self._changes_in: dict[tuple, list] = {}

        follow = (RoadOption.LANEFOLLOW, RoadOption.VOID)
        changes = []
        for u, v, d in grp._graph.edges(data=True):
            if d["type"] in follow:
                wps = [d["entry_waypoint"], *d["path"], d["exit_waypoint"]]
                xy = np.array([(w.transform.location.x, w.transform.location.y) for w in wps])
                cum = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))))
                self._geom[(u, v)] = (xy, cum)
                w = float(cum[-1])
            else:
                # The planner links a lane change entry-to-entry at no cost,
                # but the neighbour's segment may start well ahead: charge
                # the way it is ahead along the lane, or it is a free jump.
                changes.append(d)
                f = d["entry_waypoint"].transform.get_forward_vector()
                (x0, y0, _), (x1, y1, _) = (grp._graph.nodes[n]["vertex"] for n in (u, v))
                w = max(0.0, (x1 - x0) * f.x + (y1 - y0) * f.y)
            self.succ.setdefault(u, []).append((v, w))
            self.pred.setdefault(v, []).append((u, w))
        for d in changes:
            src = grp._localize(d["entry_waypoint"].transform.location)
            dst = grp._localize(d["change_waypoint"].transform.location)
            if src in self._geom and dst in self._geom:
                self._changes.setdefault(src, []).append(dst)
                self._changes_in.setdefault(dst, []).append(src)

    def locate(self, xyz: Sequence[float]):
        """(edge, metres from its entry, straight-line gap) or None."""
        import carla

        edge = self._grp._localize(carla.Location(x=float(xyz[0]), y=float(xyz[1]),
                                                  z=float(xyz[2])))
        if edge not in self._geom:
            return None
        xy, cum = self._geom[edge]
        d2 = (xy[:, 0] - xyz[0]) ** 2 + (xy[:, 1] - xyz[1]) ** 2
        i = int(d2.argmin())
        return edge, float(cum[i]), math.sqrt(float(d2[i]))

    def _length(self, edge) -> float:
        return float(self._geom[edge][1][-1])

    def _lanes(self, edge, off: float, links: dict) -> list[tuple]:
        """(edge, offset) of ``edge`` and every lane reachable from it by lane
        changes along ``links``, at the same fraction of the way along."""
        frac = off / self._length(edge) if self._length(edge) > 0 else 0.0
        out, seen, todo = [(edge, off)], {edge}, [edge]
        while todo:
            for nb in links.get(todo.pop(), ()):
                if nb not in seen:
                    seen.add(nb)
                    todo.append(nb)
                    out.append((nb, frac * self._length(nb)))
        return out

    @staticmethod
    def _dijkstra(seeds: dict, adj: dict) -> dict:
        import heapq

        dist = dict(seeds)
        heap = [(c, n) for n, c in dist.items()]
        heapq.heapify(heap)
        done = set()
        while heap:
            c, n = heapq.heappop(heap)
            if n in done:
                continue
            done.add(n)
            for m, w in adj.get(n, ()):
                if c + w < dist.get(m, math.inf):
                    dist[m] = c + w
                    heapq.heappush(heap, (c + w, m))
        return dist

    def lengths_from(self, start, goals) -> list[float]:
        s = self.locate(start)
        if s is None:
            return [math.inf] * len(goals)
        lanes = self._lanes(s[0], s[1], self._changes)
        seeds: dict = {}
        for e, off in lanes:
            seeds[e[1]] = min(seeds.get(e[1], math.inf), self._length(e) - off)
            if off <= self.SAME_SPOT_M:                     # still at the fork
                seeds[e[0]] = min(seeds.get(e[0], math.inf), off)
        dist = self._dijkstra(seeds, self.succ)
        out = []
        for goal in goals:
            g = self.locate(goal)
            if g is None:
                out.append(math.inf)
                continue
            best = math.inf
            for edge, off_g in self._lanes(g[0], g[1], self._changes_in):
                best = min(best, dist.get(edge[0], math.inf) + off_g)
                tail = self._length(edge) - off_g
                if tail <= self.SAME_SPOT_M:                # where its lanes join
                    best = min(best, dist.get(edge[1], math.inf) + tail)
                for e, off in lanes:
                    if e == edge and off_g >= off - self.SAME_SPOT_M:
                        best = min(best, abs(off_g - off))
            out.append(best + s[2] + g[2])
        return out

    def lengths_to(self, starts, goal) -> list[float]:
        g = self.locate(goal)
        if g is None:
            return [math.inf] * len(starts)
        lanes = self._lanes(g[0], g[1], self._changes_in)
        seeds: dict = {}
        for e, off in lanes:
            seeds[e[0]] = min(seeds.get(e[0], math.inf), off)
            tail = self._length(e) - off
            if tail <= self.SAME_SPOT_M:
                seeds[e[1]] = min(seeds.get(e[1], math.inf), tail)
        dist = self._dijkstra(seeds, self.pred)      # metres from a node to the goal
        out = []
        for start in starts:
            s = self.locate(start)
            if s is None:
                out.append(math.inf)
                continue
            best = math.inf
            for edge, off_s in self._lanes(s[0], s[1], self._changes):
                best = min(best, self._length(edge) - off_s + dist.get(edge[1], math.inf))
                if off_s <= self.SAME_SPOT_M:
                    best = min(best, off_s + dist.get(edge[0], math.inf))
                for e, off in lanes:
                    if e == edge and off >= off_s - self.SAME_SPOT_M:
                        best = min(best, abs(off - off_s))
            out.append(best + s[2] + g[2])
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
