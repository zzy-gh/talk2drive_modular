"""
tools/selftest.py
=================
Offline regression checks — no CARLA server, no GPU, no LLM.

Everything below exercises the mission layer and the plan conversions with a
fake route provider, which is exactly the point of the split: the part that
decides *where to go* is testable without a simulator running.

    python tools/selftest.py
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mission import MissionPlanner
from core.resolver import AutoInteraction, CliInteraction, DeferredInteraction
from core.tracking import SparseRouteTracker
from core.types import (Directive, GlobalPlan, Place, RoutePoint,
                        location_to_gps, world_to_ego)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


# ─────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────

class FakeKB:
    """Two hospitals, one museum, three cafes spread along +x."""

    DATA = {
        "hospital":    [(100.0, 0.0), (300.0, 0.0)],
        "museum":      [(200.0, 40.0)],
        "cafe":        [(50.0, 0.0), (150.0, 0.0), (900.0, 900.0)],
        "gas_station": [(80.0, 10.0)],
    }

    def find_all(self, town, label):
        return [{"building_type": label, "carla_name": f"{label}_{i}",
                 "x": x, "y": y, "z": 0.0}
                for i, (x, y) in enumerate(self.DATA.get(label, []))]

    def get_landmark_types(self, town):
        return sorted(self.DATA)

    # ── the rest of the KB surface, so assert_kb_untouched can probe it ──
    def get_landmarks(self, town):
        return [row for label in sorted(self.DATA) for row in self.find_all(town, label)]

    def find_coordinate(self, town, label, index=0):
        cands = self.find_all(town, label)
        if not cands:
            return None
        return cands[index if 0 <= index < len(cands) else 0]

    def find_nearest_coordinate(self, town, label, ref_x, ref_y):
        cands = self.find_all(town, label)
        if not cands:
            return None
        return min(cands, key=lambda r: math.hypot(r["x"] - ref_x, r["y"] - ref_y))

    @property
    def by_town(self):
        return {"Town01": self.get_landmarks("Town01")}


class FakeRoutes:
    """Straight line at 1 m resolution; snapping is a no-op."""

    def snap(self, x, y, z):
        return (x, y, z)

    def trace(self, start, goal):
        d = math.hypot(goal[0] - start[0], goal[1] - start[1])
        n = max(2, int(d))
        return [RoutePoint(x=start[0] + (goal[0] - start[0]) * i / n,
                           y=start[1] + (goal[1] - start[1]) * i / n,
                           z=0.0, yaw=0.0, option=4)
                for i in range(n + 1)]


def planner(interaction=None) -> MissionPlanner:
    return MissionPlanner(FakeKB(), FakeRoutes(),
                          interaction=interaction or AutoInteraction(),
                          town="TownX")


# ─────────────────────────────────────────────
# Mission state machine
# ─────────────────────────────────────────────

def test_plan_route():
    print("\n[mission] plan_route")
    m = planner()
    r = m.apply({"command_type": "plan_route", "destination": "hospital",
                 "waypoints": [{"type": "cafe", "order": 1, "confidence": 0.9}],
                 "urgency": "high"})
    check("apply needs_replan", r.needs_replan)
    check("urgency captured", m.directive.urgency == "high")
    check("target speed follows urgency", m.directive.target_speed_kph() == 45.0)
    check("urgency reaches a VLA as language",
          "faster" in (m.directive.as_speed_instruction() or ""))

    plan = m.build_plan((0.0, 0.0, 0.0))
    check("plan built", plan is not None)
    check("nearest hospital chosen", abs(plan.goal.x - 100.0) < 1e-6,
          f"got {plan.goal.x}")
    check("one stop resolved", len(plan.stops) == 1)
    check("on-the-way cafe chosen, not the far one",
          abs(plan.stops[0].x - 50.0) < 1e-6, f"got {plan.stops[0].x}")
    check("route is dense", len(plan) > 100)
    check("revision starts at 1", plan.revision == 1)


def test_edits():
    print("\n[mission] edits while driving")
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1, "confidence": 0.9}]})
    m.build_plan((0.0, 0.0, 0.0))

    m.apply({"command_type": "insert_stop",
             "insert": {"type": "gas_station", "position": "next", "confidence": 1.0}})
    check("insert next goes first", m.stops[0].label == "gas_station")

    m.apply({"command_type": "insert_stop",
             "insert": {"type": "museum", "position": "enroute", "confidence": 1.0,
                        "after": "cafe"}})
    check("insert after 'cafe' lands right after it",
          [s.label for s in m.stops] == ["gas_station", "cafe", "museum"],
          str([s.label for s in m.stops]))

    m.apply({"command_type": "remove_stop",
             "remove": {"type": "cafe", "fallback": "next_waypoint"}})
    check("remove drops it", "cafe" not in [s.label for s in m.stops])
    check("orders renumbered", [s.order for s in m.stops] == [1, 2])

    m.apply({"command_type": "new_destination", "destination": "museum",
             "keep_waypoints": True})
    check("keep_waypoints demotes old destination to a stop",
          m.stops[-1].label == "hospital", str([s.label for s in m.stops]))

    m.apply({"command_type": "new_destination", "destination": "hospital",
             "keep_waypoints": False})
    check("'instead' clears the stops", m.stops == [])

    r = m.apply({"command_type": "cancel_route"})
    check("cancel reports cancelled", r.cancelled)
    check("cancel clears state", not m.has_mission and m.plan is None)

    r = m.apply({"command_type": "remove_stop", "remove": {"type": "cafe"}})
    check("edit with no route is rejected, not crashed", not r.ok)
    r = m.apply({"command_type": "teleport"})
    check("unknown command is rejected", not r.ok)


def test_enroute_position():
    print("\n[mission] enroute insert position policy")
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1, "confidence": 1.0}]})
    m.build_plan((0.0, 0.0, 0.0))
    m.apply({"command_type": "insert_stop",
             "insert": {"type": "museum", "position": "enroute", "confidence": 1.0}})
    check("enroute stop is flagged", any(s.enroute for s in m.stops))
    plan = m.build_plan((0.0, 0.0, 0.0))
    check("AutoInteraction appends it last", plan.stops[-1].label == "museum",
          str([s.label for s in plan.stops]))
    check("flag cleared once committed", not any(s.enroute for s in m.stops))

    class Ask(AutoInteraction):
        asked = []

        def choose_insert_position(self, req):
            self.asked.append(req)
            return 0                              # passenger: "put it first"

    m = planner(Ask())
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1, "confidence": 1.0}]})
    m.build_plan((0.0, 0.0, 0.0))
    m.apply({"command_type": "insert_stop",
             "insert": {"type": "museum", "position": "enroute", "after": None}})
    plan = m.build_plan((0.0, 0.0, 0.0))
    check("unordered stop asks the passenger",
          [(r.label, r.stop_labels) for r in Ask.asked] == [("museum", ["cafe"])])
    check("passenger's answer decides the order",
          [s.label for s in plan.stops] == ["museum", "cafe"])


class FakeRoadRoutes(FakeRoutes):
    """Driving distance is the straight line, except to places across a
    river: the hospital at (100, 0) and the cafe at (50, 0) are 1000 m away by
    road from anywhere."""

    ACROSS = {(100.0, 0.0), (50.0, 0.0)}

    def __init__(self):
        self.measured = []

    def route_length(self, start, goal):
        self.measured.append((tuple(start[:2]), tuple(goal[:2])))
        if tuple(goal[:2]) in self.ACROSS:
            return 1000.0
        return math.hypot(goal[0] - start[0], goal[1] - start[1])


class FakeGraphRoutes(FakeRoadRoutes):
    """FakeRoadRoutes plus the one-search batch calls; counts the searches."""

    def __init__(self):
        super().__init__()
        self.searches = 0

    def route_lengths(self, start, goals):
        self.searches += 1
        return [FakeRoadRoutes.route_length(self, start, g) for g in goals]

    def route_lengths_to(self, starts, goal):
        self.searches += 1
        return [FakeRoadRoutes.route_length(self, s, goal) for s in starts]


def test_place_distance():
    print("\n[mission] nearest by road, not by straight line")

    def road_planner(**kw):
        routes = FakeRoadRoutes()
        return MissionPlanner(FakeKB(), routes, town="TownX", **kw), routes

    for kind, want in (("euclidean", 100.0), ("route", 300.0)):
        m, _ = road_planner(place_distance=kind)
        m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
        plan = m.build_plan((0.0, 0.0, 0.0))
        check(f"{kind}: nearest hospital at x={want:.0f}", plan.goal.x == want,
              f"got {plan.goal.x}")

    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    check("a router that cannot measure falls back to the straight line",
          m.build_plan((0.0, 0.0, 0.0)).goal.x == 100.0)

    m, routes = road_planner(route_top_k=1)
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1, "confidence": 1.0}]})
    plan = m.build_plan((0.0, 0.0, 0.0))
    check("top-k=1 still finds the nearest by road (straight line is a lower bound)",
          plan.goal.x == 300.0, f"got {plan.goal.x}")
    check("cafe across the river is not 'on the way'; the reachable one is",
          plan.stops[0].x == 150.0, f"got {plan.stops[0].x}")
    from_ego = [g for s, g in routes.measured if s == (0.0, 0.0)]
    check("the far cafe is never measured by road", (900.0, 900.0) not in from_ego,
          str(from_ego))

    class BehindKB(FakeKB):
        """Cafe #1 is 200 m behind the car, #2 is ahead towards the hospital."""
        DATA = {"hospital": [(1000.0, 0.0)], "cafe": [(-200.0, 0.0), (450.0, 200.0)]}

    for k in (1, 10):
        routes = FakeRoadRoutes()
        m = MissionPlanner(BehindKB(), routes, town="TownX", route_top_k=k)
        m.apply({"command_type": "plan_route", "destination": "hospital",
                 "waypoints": [{"type": "cafe", "order": 1, "confidence": 1.0}]})
        plan = m.build_plan((0.0, 0.0, 0.0))
        check(f"top-k={k}: the stop that lengthens the trip least, not the nearest (#2)",
              plan.stops[0].index == 2, f"got #{plan.stops[0].index}")
    routes = FakeRoadRoutes()
    m = MissionPlanner(BehindKB(), routes, town="TownX", route_top_k=1)
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1, "confidence": 1.0}]})
    m.build_plan((0.0, 0.0, 0.0))
    check("top-k=1 never measures the cafe behind (its bound is already worse)",
          (-200.0, 0.0) not in [g for _, g in routes.measured], str(routes.measured))

    class Ask(AutoInteraction):
        req = None

        def choose_place(self, req):
            Ask.req = req
            return super().choose_place(req)

    m = MissionPlanner(FakeKB(), FakeRoadRoutes(), interaction=Ask(), town="TownX")
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    m.build_plan((0.0, 0.0, 0.0))
    check("the question lists candidates nearest-by-road first, with distances",
          [p.x for p in Ask.req.candidates] == [300.0, 100.0]
          and Ask.req.distances == [300.0, 1000.0], str(Ask.req))

    for kind in ("route", "graph", "euclidean"):
        Ask.req = None
        m = MissionPlanner(FakeKB(), FakeGraphRoutes(), interaction=Ask(), town="TownX",
                           place_distance=kind, route_top_k=2)
        m.apply({"command_type": "plan_route", "destination": "cafe", "waypoints": []})
        plan = m.build_plan((0.0, 0.0, 0.0))
        check(f"{kind}: a question offers at most top-k, and says how many there are",
              len(Ask.req.candidates) == 2 and Ask.req.total == 3, str(Ask.req))
        check(f"{kind}: the cap keeps the best", plan.goal.x == 150.0 if kind != "euclidean"
              else plan.goal.x == 50.0, f"got {plan.goal.x}")

    try:
        MissionPlanner(FakeKB(), FakeRoutes(), place_distance="manhattan")
        ok = False
    except ValueError:
        ok = True
    check("unknown place_distance is rejected", ok)

    print("\n[mission] graph mode: every candidate scored by one search")
    for k in (1, 10):
        routes = FakeGraphRoutes()
        m = MissionPlanner(FakeKB(), routes, town="TownX", place_distance="graph",
                           route_top_k=k)
        m.apply({"command_type": "plan_route", "destination": "hospital",
                 "waypoints": [{"type": "cafe", "order": 1, "confidence": 1.0}]})
        plan = m.build_plan((0.0, 0.0, 0.0))
        check(f"top-k={k}: same picks as route mode (hospital x=300, cafe x=150)",
              (plan.goal.x, plan.stops[0].x) == (300.0, 150.0),
              f"got {(plan.goal.x, plan.stops[0].x)}")
        check(f"top-k={k}: one search for the destination, two for the stop",
              routes.searches == 3, str(routes.searches))
    m = MissionPlanner(BehindKB(), FakeGraphRoutes(), town="TownX", place_distance="graph")
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1, "confidence": 1.0}]})
    check("graph mode also picks the cafe ahead, not the one behind",
          m.build_plan((0.0, 0.0, 0.0)).stops[0].index == 2)
    m = MissionPlanner(FakeKB(), FakeRoadRoutes(), town="TownX", place_distance="graph")
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    check("graph mode without batch calls falls back to tracing each",
          m.build_plan((0.0, 0.0, 0.0)).goal.x == 300.0)


def test_landmark_ids():
    print("\n[mission] landmarks carry the ID the map shows and the parser means")
    import builtins
    import contextlib
    import io

    m = planner()
    cafes = m._candidates("cafe")
    check("IDs are 1..n in knowledge-base order", [p.index for p in cafes] == [1, 2, 3])
    check("a place shows as 'label #ID'", cafes[1].tag == "cafe #2", cafes[1].tag)

    m.apply({"command_type": "plan_route", "destination": "cafe",
             "destination_index": 3, "waypoints": []})
    check("index 3 is cafe #3", m.build_plan((0.0, 0.0, 0.0)).goal.index == 3)

    # hospital #1 is 1000 m by road, #2 is 300 m: listed #2 first, and an
    # answer of "1" means hospital #1, not the first line of the list
    shown = []
    cli = CliInteraction(preview=lambda ps: shown.extend(p.index for p in ps))
    m = MissionPlanner(FakeKB(), FakeRoadRoutes(), interaction=cli, town="TownX")
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    out, real_input = io.StringIO(), builtins.input
    builtins.input = lambda prompt="": "1"
    try:
        with contextlib.redirect_stdout(out):
            plan = m.build_plan((0.0, 0.0, 0.0))
    finally:
        builtins.input = real_input
    rows = [l.split()[0] for l in out.getvalue().splitlines() if l.strip().startswith("[")]
    check("CLI lists candidates nearest first, by ID", rows == ["[2]", "[1]"], str(rows))
    check("CLI previews them in the same order", shown == [2, 1], str(shown))
    check("the answer is an ID, not a list position", plan.goal.index == 1,
          f"got #{plan.goal.index}")

    from runtime.viz import LandmarkBoard
    near = LandmarkBoard(None, cafes, near_m=100.0)      # cafes at x=50, 150, (900, 900)
    check("before the ego is placed every landmark shows", len(near.snapshot()) == 3)
    near.set_ego(0.0, 0.0)
    check("only idle landmarks near the ego show",
          [p.index for p, *_ in near.snapshot()] == [1], str(near.snapshot()))
    near.set_candidates([cafes[2]])
    check("a candidate shows however far",
          sorted(p.index for p, *_ in near.snapshot()) == [1, 3])
    board = LandmarkBoard(None, cafes)
    board.set_candidates([cafes[2], cafes[0]])
    marks = {p.index: cand for p, _, _, cand in board.snapshot()}
    check("candidate markers are labelled with the ID", marks == {1: 1, 2: None, 3: 3},
          str(marks))


def test_route_length():
    print("\n[route] route_length on Town01")
    try:
        import carla
        from agents.navigation.local_planner import RoadOption
        from core.paths import section_path
        from core.route_provider import CarlaRouteProvider
    except ImportError as exc:
        print(f"  skip ({exc})")
        return
    xodr_dir = section_path("covlm", "opendrive_dir")
    xodr = xodr_dir / "Town01.xodr" if xodr_dir else None
    if xodr is None or not xodr.is_file():
        print("  skip (no Town01.xodr under [covlm] opendrive_dir)")
        return
    m = carla.Map("Town01", xodr.read_text())
    rp = CarlaRouteProvider(m, 1.0)
    wps = m.generate_waypoints(50.0)
    pairs = [(wps[i], wps[-1 - i]) for i in range(0, 40, 4)]
    ok_bound = ok_sum = True
    for a, b in pairs:
        s, g = a.transform.location, b.transform.location
        s, g = (s.x, s.y, s.z), (g.x, g.y, g.z)
        d = rp.route_length(s, g)
        pts = rp.trace(s, g)
        ok_bound &= d + 1e-6 >= math.hypot(g[0] - s[0], g[1] - s[1])
        gaps = (math.hypot(pts[0].x - s[0], pts[0].y - s[1])
                + math.hypot(g[0] - pts[-1].x, g[1] - pts[-1].y))
        ok_sum &= abs(d - gaps - sum(p.distance_to(q) for p, q in zip(pts, pts[1:]))) < 1e-6
    check("driving distance is never below the straight line", ok_bound)
    check("driving distance is the traced route plus its end gaps", ok_sum)
    rp._grp._previous_decision, rp._grp._intersection_end_node = RoadOption.LEFT, 7
    rp.route_length((pairs[0][0].transform.location.x, pairs[0][0].transform.location.y, 0.0),
                    (pairs[0][1].transform.location.x, pairs[0][1].transform.location.y, 0.0))
    check("measuring leaves the router's turn state alone",
          (rp._grp._previous_decision, rp._grp._intersection_end_node) == (RoadOption.LEFT, 7))

    # Off junctions: a start part-way into a junction connector is where the
    # traced route may leave by a branch already behind the car.
    road = [w for w in wps if not w.is_junction]
    pairs = [(road[i], road[-1 - i]) for i in range(0, 40, 4)]
    ends = [(a.transform.location, b.transform.location) for a, b in pairs]
    ends = [((a.x, a.y, a.z), (b.x, b.y, b.z)) for a, b in ends]
    fwd = [rp.route_lengths(a, [b])[0] for a, b in ends]
    bwd = [rp.route_lengths_to([a], b)[0] for a, b in ends]
    check("one-to-many forward and reverse searches agree",
          all(abs(f - b) < 1e-6 for f, b in zip(fwd, bwd)), str(list(zip(fwd, bwd))))
    traced = [rp.route_length(a, b) for a, b in ends]
    close = sum(abs(f - t) <= max(20.0, 0.05 * t) for f, t in zip(fwd, traced))
    check("graph lengths match the traced routes (within 20 m or 5%)",
          close >= len(ends) - 1, f"{close}/{len(ends)}: {list(zip(fwd, traced))}")
    stock = CarlaRouteProvider(m, 1.0)
    del stock._grp._find_closest_in_list            # GlobalRoutePlanner's own search
    check("the cached closest-waypoint search plans exactly the stock routes",
          all(rp.trace_chain([a, b]) == stock.trace_chain([a, b]) for a, b in ends))
    batch = rp.route_lengths(ends[0][0], [b for _, b in ends])
    check("a batch equals one call per goal",
          all(abs(x - rp.route_lengths(ends[0][0], [b])[0]) < 1e-6
              for x, (_, b) in zip(batch, ends)))


def test_interaction_policies():
    print("\n[interaction] policies")
    check("AutoInteraction is non-blocking", not AutoInteraction().blocking)
    check("CliInteraction declares itself blocking", CliInteraction().blocking)
    check("DeferredInteraction is non-blocking", not DeferredInteraction().blocking)

    deferred = DeferredInteraction()
    m = planner(deferred)
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    plan = m.build_plan((0.0, 0.0, 0.0))
    check("deferred returns no plan instead of blocking", plan is None)
    check("deferred queued the question", len(deferred.take_pending()) == 1)

    from runtime.passenger import Talk2DrivePassenger
    try:
        Talk2DrivePassenger(planner(CliInteraction()))
        ok = False
    except ValueError:
        ok = True
    check("passenger mode refuses a blocking policy", ok)


# ─────────────────────────────────────────────
# Plan conversions
# ─────────────────────────────────────────────

def test_downsample():
    print("\n[plan] downsample")
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    plan = m.build_plan((0.0, 0.0, 0.0))
    sparse = plan.downsample(50.0)
    check("sparse is much smaller", len(sparse) < len(plan) / 10,
          f"{len(sparse)} vs {len(plan)}")
    check("keeps the first point", abs(sparse[0].x - plan.points[0].x) < 1e-6)
    check("keeps the last point", abs(sparse[-1].x - plan.points[-1].x) < 1e-6)
    gaps = [math.hypot(sparse[i].x - sparse[i - 1].x, sparse[i].y - sparse[i - 1].y)
            for i in range(1, len(sparse))]
    check("no gap far beyond the sample factor", max(gaps) <= 55.0, f"max {max(gaps):.1f}")


def test_geometry():
    print("\n[geometry] ego frame (CARLA's own forward/right vectors)")
    try:
        import carla
    except ImportError:
        print("  skip  carla not importable")
        return

    # Facing +x: a point 10 m ahead and 3 m to the ego's right (+y in CARLA).
    tf = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(yaw=0.0))
    f, r = world_to_ego(tf, 10.0, 3.0)
    check("yaw=0 forward", abs(f - 10.0) < 1e-4, f"got {f}")
    check("yaw=0 right", abs(r - 3.0) < 1e-4, f"got {r}")

    # Facing +y (yaw=90): world +y is now straight ahead, world -x is right.
    tf = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(yaw=90.0))
    f, r = world_to_ego(tf, 0.0, 10.0)
    check("yaw=90 forward", abs(f - 10.0) < 1e-3, f"got {f}")
    check("yaw=90 lateral is zero", abs(r) < 1e-3, f"got {r}")
    f, r = world_to_ego(tf, -5.0, 0.0)
    check("yaw=90 world -x is to the right", abs(r - 5.0) < 1e-3, f"got {r}")

    # A point behind must come out negative, never mirrored.
    tf = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(yaw=0.0))
    f, _ = world_to_ego(tf, -7.0, 0.0)
    check("behind is negative forward", f < 0, f"got {f}")

    f, r = world_to_ego(tf, 10.0, 3.0, convention="right_forward")
    check("right_forward swaps the axes", abs(f - 3.0) < 1e-4 and abs(r - 10.0) < 1e-4)


def test_tracker():
    print("\n[plan] sparse tracker (SimLingo target points)")
    try:
        import carla
        import numpy as np
    except ImportError:
        print("  skip  carla/numpy not importable")
        return

    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    plan = m.build_plan((0.0, 0.0, 0.0))

    t = SparseRouteTracker(min_distance=7.5, max_distance=50.0)
    t.set_plan(plan)
    check("tracker rejects a stale plan", not t.is_stale(plan))
    n0 = len(t)

    tf = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(yaw=0.0))
    tp = t.target_points_ego(tf, n=2)
    check("target points have shape [2,2]", tp.shape == (2, 2), str(tp.shape))
    check("target points are ahead of the ego", tp[0][0] > 0 and tp[1][0] > tp[0][0],
          str(tp))
    check("target points are laterally ~0 on a straight road", abs(tp[0][1]) < 1e-3)

    # Driving the route tick by tick: the leaderboard rule pops as we pass.
    for x in range(0, 95):
        t.run_step(float(x), 0.0)
    check("passed points are popped while driving", len(t) < n0, f"{len(t)} vs {n0}")
    tf = carla.Transform(carla.Location(94, 0, 0), carla.Rotation(yaw=0.0))
    tp = t.target_points_ego(tf, n=2)
    check("target stays ahead while driving", tp[0][0] > 0, str(tp))

    # Jumping past a sparse point (replan / respawn / lane offset) is exactly
    # the case the leaderboard rule misses: without a heading the stale point
    # stays, with one it is dropped.
    t2 = SparseRouteTracker()
    t2.set_plan(plan)
    t2.run_step(90.0, 0.0)
    tp_stale = t2.target_points_ego(
        carla.Transform(carla.Location(90, 0, 0), carla.Rotation(yaw=0.0)), n=2)
    check("without heading a skipped point can sit behind the ego",
          tp_stale[0][0] < 0, str(tp_stale))

    t3 = SparseRouteTracker()
    t3.set_plan(plan)
    t3.run_step(90.0, 0.0, yaw_deg=0.0)
    tp_fixed = t3.target_points_ego(
        carla.Transform(carla.Location(90, 0, 0), carla.Rotation(yaw=0.0)), n=2)
    check("heading drops the point behind the ego", tp_fixed[0][0] > 0, str(tp_fixed))

    # A point just barely behind must not be dropped (tolerance), or the
    # tracker would chew through the route on lateral offsets.
    t4 = SparseRouteTracker(behind_tolerance=2.0)
    t4.set_plan(plan)
    before = len(t4)
    t4.run_step(-1.0, 0.0, yaw_deg=0.0)
    check("tolerance keeps a barely-behind point", len(t4) == before)


def test_gps():
    print("\n[plan] gps conversion (leaderboard agents)")
    try:
        import carla
    except ImportError:
        print("  skip  carla not importable")
        return
    lat_ref, lon_ref = 42.0, 2.0
    at_origin = location_to_gps(lat_ref, lon_ref, carla.Location(0, 0, 0))
    check("origin maps to the geo reference",
          abs(at_origin["lat"] - lat_ref) < 1e-6 and abs(at_origin["lon"] - lon_ref) < 1e-6,
          str(at_origin))
    east = location_to_gps(lat_ref, lon_ref, carla.Location(1000, 0, 0))
    north = location_to_gps(lat_ref, lon_ref, carla.Location(0, -1000, 0))
    check("+x increases longitude", east["lon"] > lon_ref)
    check("-y increases latitude", north["lat"] > lat_ref)


def test_arrival():
    print("\n[mission] arrival")
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    plan = m.build_plan((0.0, 0.0, 0.0))
    check("not arrived at the start", not m.is_arrived(0.0, 0.0))
    check("arrived at the goal", m.is_arrived(plan.goal.x, plan.goal.y))
    check("remaining distance shrinks",
          plan.remaining_distance(90.0, 0.0) < plan.remaining_distance(10.0, 0.0))

    # Round trip: goal at the start, one stop between.
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "gas_station", "order": 1}]})
    plan = m.build_plan((0.0, 0.0, 0.0))
    stop = plan.stops[0]
    check("not arrived at the goal before the stop",
          not m.is_arrived(plan.goal.x, plan.goal.y))
    m.reached_stop(stop.x, stop.y)
    check("arrived at the goal once the stop is reached", m.is_arrived(plan.goal.x, plan.goal.y))


def test_stop_progress():
    print("\n[mission] reaching stops")
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "gas_station", "order": 1},
                           {"type": "cafe", "order": 2, "index": 1}]})
    plan = m.build_plan((0.0, 0.0, 0.0))
    gas, cafe = plan.stops
    check("stops planned in order", [p.label for p in plan.stops] == ["gas_station", "cafe"])
    check("nothing reached at the start", m.reached_stop(0.0, 0.0) is None)
    check("a later stop is not reached out of order", m.reached_stop(cafe.x, cafe.y) is None)
    check("first stop reached", m.reached_stop(gas.x, gas.y) is gas)
    check("reached stop leaves the mission", [s.label for s in m.stops] == ["cafe"])
    check("reached only once", m.reached_stop(gas.x, gas.y) is None)
    check("second stop reached", m.reached_stop(cafe.x, cafe.y) is cafe)
    check("re-plan does not route back", m.build_plan((cafe.x, cafe.y, 0.0)).stops == [])
    check("all_places covers the town", len(m.all_places()) == 7)


def test_extend_destination():
    print("\n[mission] going somewhere after the destination")
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "destination_index": 2, "waypoints": []})
    m.apply({"command_type": "new_destination", "destination": "museum",
             "keep_waypoints": True})
    check("old destination kept as the last stop",
          [(s.label, s.index) for s in m.stops] == [("hospital", 2)])
    plan = m.build_plan((0.0, 0.0, 0.0))
    check("kept stop is still the second hospital", abs(plan.stops[0].x - 300.0) < 1e-6,
          f"got {plan.stops[0].x}")

    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1}]})
    r = m.apply({"command_type": "insert_stop",
                 "insert": {"type": "museum", "position": "enroute", "after": "hospital"}})
    check("insert after the destination extends the route", r.needs_replan)
    check("new place becomes the destination", m.destination == "museum")
    check("old destination becomes the last stop",
          [s.label for s in m.stops] == ["cafe", "hospital"])

    m = planner()
    r = m.apply({"command_type": "new_destination", "destination": "museum",
                 "keep_waypoints": True})
    check("'then go to X' while idle just starts a route",
          r.needs_replan and m.destination == "museum" and m.stops == []
          and m.build_plan((0.0, 0.0, 0.0)) is not None)


def test_route_prompt():
    print("\n[llm] current route in the prompt")
    m = planner()
    check("idle summary has no destination", m.route_summary()["destination"] is None)
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "destination_index": 2, "waypoints": [{"type": "cafe", "order": 1}]})
    summary = m.route_summary()
    check("summary lists stops and destination",
          summary == {"destination": "hospital", "destination_index": 2,
                      "stops": [{"type": "cafe", "index": None}]}, str(summary))

    try:
        from core.llm_inference import _build_system_prompt
    except ImportError as exc:                 # google-genai not in this env
        print(f"  skip prompt checks ({exc})")
        return
    types_ = ["cafe", "hospital", "park"]
    legacy = _build_system_prompt("Town01", types_)
    check("no route: no route section", "Current route" not in legacy)
    idle = _build_system_prompt("Town01", types_, planner().route_summary())
    check("idle route says so", "Current route: none" in idle)
    active = _build_system_prompt("Town01", types_, summary)
    check("active route lists the stop", "1. stop: cafe" in active)
    check("active route names the indexed destination",
          "-> destination: hospital (#2)" in active)
    check("'after the destination' maps to new_destination",
          'after the hospital") -> new_destination' in active.replace("\n  ", " "))


def test_multi_and_move():
    print("\n[mission] several commands, moving and 'before'")
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1}, {"type": "museum", "order": 2},
                           {"type": "gas_station", "order": 3}]})
    order = lambda: [s.label for s in m.stops]
    r = m.apply({"command_type": "move_stop", "move": {"type": "museum", "position": "next"}})
    check("move to next", r.needs_replan and order() == ["museum", "cafe", "gas_station"])
    m.apply({"command_type": "move_stop", "move": {"type": "museum", "position": "last"}})
    check("move to last", order() == ["cafe", "gas_station", "museum"])
    m.apply({"command_type": "move_stop", "move": {"type": "museum", "before": "gas_station"}})
    check("move before a stop", order() == ["cafe", "museum", "gas_station"])
    m.apply({"command_type": "move_stop", "move": {"type": "cafe", "after": "gas_station"}})
    check("move after a stop", order() == ["museum", "gas_station", "cafe"])
    r = m.apply({"command_type": "move_stop", "move": {"type": "hotel", "position": "next"}})
    check("moving a place that is not a stop is refused", not r.ok)
    m.apply({"command_type": "insert_stop", "insert": {"type": "hotel", "before": "gas_station"}})
    check("insert before a stop", order() == ["museum", "hotel", "gas_station", "cafe"])
    m.apply({"command_type": "move_stop", "move": {"type": "museum", "after": "hospital"}})
    check("moving after the destination extends the route",
          m.destination == "museum" and order()[-1] == "hospital")

    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1}]})
    m.apply({"command_type": "insert_stop", "insert": {"type": "museum", "before": "hospital"}})
    check("'before the destination' is the last stop", order() == ["cafe", "museum"])
    r = m.apply({"commands": [
        {"command_type": "remove_stop", "remove": {"type": "cafe"}},
        {"command_type": "insert_stop", "insert": {"type": "gas_station", "position": "next"}},
        {"command_type": "new_destination", "destination": "hotel", "keep_waypoints": True}]})
    check("several commands apply in order, one re-plan",
          r.ok and r.needs_replan and order() == ["gas_station", "museum", "hospital"]
          and m.destination == "hotel", f"{order()} -> {m.destination}")
    r = m.apply({"commands": [
        {"command_type": "remove_stop", "remove": {"type": "nowhere"}},
        {"command_type": "move_stop", "move": {"type": "nowhere", "position": "next"}}]})
    check("a batch that changes nothing does not re-plan",
          not r.needs_replan and order() == ["gas_station", "museum", "hospital"])
    r = m.apply({"commands": [{"command_type": "cancel_route"},
                              {"command_type": "plan_route", "destination": "museum"}]})
    check("cancel then a new route is a new route, not a cancel",
          not r.cancelled and r.needs_replan and m.destination == "museum")
    r = m.apply({"commands": [{"command_type": "cancel_route"}]})
    check("a cancel inside a batch still cancels", r.cancelled)


def test_landmark_board():
    print("\n[viz] landmark board")
    from runtime.viz import LandmarkBoard, _key
    m = planner()
    board = LandmarkBoard(world=None, places=m.all_places())
    state = lambda p: board._state[_key(p)][0]

    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "gas_station", "order": 1}]})
    plan = m.build_plan((0.0, 0.0, 0.0))
    board.set_plan(plan)
    gas = plan.stops[0]
    check("goal is the destination", state(plan.goal) == "destination")
    check("stop is a stop", state(gas) == "stop")
    check("others stay idle",
          sum(s == "idle" for s, _ in board._state.values()) == len(board._state) - 2)

    board.mark_visited(gas)
    m.apply({"command_type": "new_destination", "destination": "museum"})
    new = m.build_plan((gas.x, gas.y, 0.0))
    board.set_plan(new)
    check("reached stop stays reached", state(gas) == "visited")
    check("replaced goal is cancelled", state(plan.goal) == "cancelled")
    check("new goal is the destination", state(new.goal) == "destination")

    board.set_candidates([plan.goal])
    check("candidates overlay the state", board._candidates == {_key(plan.goal): 1})
    board.clear_candidates()
    board.cancel_active()
    check("cancel marks the goal cancelled", state(new.goal) == "cancelled")
    board.reset()
    check("reset greys everything", {s for s, _ in board._state.values()} == {"idle"})
    depot = Place("depot", 5.0, 5.0)
    board.set_plan(GlobalPlan(points=[], goal=depot))
    check("a place outside the knowledge base is adopted", state(depot) == "destination")


def test_bev_render():
    print("\n[bev] renderer")
    try:
        from runtime.bev import BevRenderer
    except ImportError as exc:
        print(f"  skip ({exc})")
        return
    from types import SimpleNamespace as NS

    class Wp:
        def __init__(self, x, y):
            self.transform = NS(location=NS(x=x, y=y, z=0.0))
            self.lane_width = 3.5

        def next(self, d):
            return [Wp(self.transform.location.x + d, self.transform.location.y)] \
                if self.transform.location.x < 100 else []

    class FakeMap:
        def generate_waypoints(self, d):
            return [Wp(float(x), y) for y in (0.0, 50.0) for x in range(0, 101, 2)]

    r = BevRenderer(FakeMap(), size_px=300, margin_px=10)
    check("origin maps to the margin", r.px(0.0, 0.0) == (10, 10))
    check("x grows right, y grows down", r.px(100.0, 0.0)[0] > 10 and r.px(0.0, 50.0)[1] > 10)
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    plan = m.build_plan((0.0, 0.0, 0.0))
    img = r.render(ego=(10.0, 0.0, 0.0), plan=plan,
                   landmarks=[(plan.goal, "destination", None, None)])
    check("map plus legend column", img.shape[0] == r.h and img.shape[1] == r.w + 150,
          str(img.shape))
    check("route is drawn", (img[:, :r.w] == (0, 200, 0)).all(axis=2).any())

    check("a small town is shown whole", not r.follow)

    # The same 100 m road, as a "large" town: the view follows the ego
    f = BevRenderer(FakeMap(), size_px=300, window_m=40.0, follow_above_m=50.0, max_tiles=3)
    check("a large town follows the ego", f.follow)
    img = f.render(ego=(30.0, 0.0, 0.0), plan=plan, landmarks=[])
    check("the window is centred on the ego", f.px(30.0, 0.0) == (150, 150))
    check("follow view: map plus legend column", img.shape[:2] == (300, 450), str(img.shape))
    from runtime.bev import ROAD
    check("roads near the ego are drawn",       # beside the route, inside the lane
          (img[158, :300] == ROAD).all(axis=1).any())
    for x in range(0, 101, 10):
        f.render(ego=(float(x), 0.0, 0.0), plan=None, landmarks=[])
    check("tile cache stays bounded", len(f._tiles) <= 3, str(len(f._tiles)))


def test_fpv():
    print("\n[fpv] model-input window")
    try:
        from runtime.fpv import annotate, project
    except ImportError as exc:
        print(f"  skip ({exc})")
        return
    import numpy as np
    from backends.base import FpvFrame, camera_rgb

    bgra = np.zeros((4, 6, 4), dtype=np.uint8)
    bgra[..., 0] = 200                                  # blue in BGRA
    rgb = camera_rgb((7, bgra))
    check("leaderboard (frame, BGRA) -> RGB", rgb.shape == (4, 6, 3) and rgb[0, 0, 2] == 200)

    # TCP's front camera.
    cam = {"x": -1.5, "y": 0.0, "z": 2.0, "pitch": 0.0, "fov": 100}
    size = (256, 900)
    ahead = project(cam, size, 10.0, 0.0)
    check("point straight ahead is centred, below the horizon",
          ahead is not None and ahead[0] == 450 and ahead[1] > 128, str(ahead))
    right = project(cam, size, 10.0, 2.0)
    check("point to the right is right of centre", right is not None and right[0] > 450)
    check("point behind the camera is dropped", project(cam, size, -3.0, 0.0) is None)

    img = np.full((256, 900, 3), 50, dtype=np.uint8)
    tcp = annotate(FpvFrame(rgb=img, control={"steer": 0.1, "throttle": 0.5, "brake": 0.0},
                            speed_mps=5.0, model="TCPAgent", notes=("branch traj",),
                            camera=cam, waypoints=((2.0, 0.0), (4.0, 0.2), (6.0, 0.4))))
    check("TCP frame: image plus strip", tcp.shape == (256 + 110, 900, 3), str(tcp.shape))
    check("TCP waypoints drawn", (tcp[:256] == (0, 230, 0)).all(axis=2).any())
    check("no crop band without crop_from_row", (tcp[200:256, :5] == 50).all())
    sim = annotate(FpvFrame(rgb=img, control={}, speed_mps=0.0, model="SimLingo",
                            notes=("go straight",), crop_from_row=180))
    check("SimLingo crop band is dimmed", sim[250, 450, 0] < 50 and sim[100, 450, 0] == 50)


def test_text_only_viz():
    print("\n[viz] text-only drawing stays out of camera images")
    from types import SimpleNamespace as NS
    from runtime.viz import PlanVisualizer

    class Recorder:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return lambda *a, **k: self.calls.append(name)

    debug = Recorder()
    viz = PlanVisualizer(NS(debug=debug), planner().all_places(), text_only=True)
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital",
             "waypoints": [{"type": "cafe", "order": 1}]})
    plan = m.build_plan((0.0, 0.0, 0.0))
    viz.show(plan)
    viz.redraw(life_time=0.75)
    import carla
    viz.beacon(carla.Location(x=1.0, y=2.0, z=0.0))
    viz.clear()
    check("only draw_string is used", set(debug.calls) == {"draw_string"},
          str(sorted(set(debug.calls))))
    from runtime.session import SessionConfig
    check("text-only is the default, whatever the backend",
          SessionConfig().viz_text_only and PlanVisualizer(NS(debug=debug)).text_only)
    check("route drawn about every 3 m",
          debug.calls.count("draw_string") >= int(plan.length() / 3.0))


def test_snap_heading():
    print("\n[route] re-plan start stays on the ego's own lane")
    try:
        import carla
        from core.paths import section_path
        from core.route_provider import CarlaRouteProvider, _yaw_diff
    except ImportError as exc:
        print(f"  skip ({exc})")
        return
    xodr_dir = section_path("covlm", "opendrive_dir")
    xodr = xodr_dir / "Town01.xodr" if xodr_dir else None
    if xodr is None or not xodr.is_file():
        print("  skip (no Town01.xodr under [covlm] opendrive_dir)")
        return
    m = carla.Map("Town01", xodr.read_text())
    rp = CarlaRouteProvider(m, 1.0)
    old_bad = new_bad = n = 0
    for wp in m.generate_waypoints(25.0):
        opp = wp.get_left_lane()
        if wp.is_junction or opp is None or opp.lane_type != carla.LaneType.Driving:
            continue
        yaw = wp.transform.rotation.yaw
        if _yaw_diff(opp.transform.rotation.yaw, yaw) < 90:
            continue
        n += 1
        l = opp.transform.location            # drifted onto the oncoming lane
        old = m.get_waypoint(carla.Location(*rp.snap(l.x, l.y, l.z)))
        new = m.get_waypoint(carla.Location(*rp.snap_heading(l.x, l.y, l.z, yaw + 15.0)))
        old_bad += _yaw_diff(old.transform.rotation.yaw, yaw) > 90
        new_bad += _yaw_diff(new.transform.rotation.yaw, yaw) > 90
    check("plain snap puts a drifted ego on the oncoming lane", old_bad == n, f"{old_bad}/{n}")
    check("snap_heading keeps it on its own lane", n > 0 and new_bad == 0, f"{new_bad}/{n} wrong")
    own = m.generate_waypoints(25.0)[5]
    l = own.transform.location
    back = m.get_waypoint(carla.Location(*rp.snap_heading(l.x, l.y, l.z, own.transform.rotation.yaw)))
    check("an ego on its own lane stays there", (back.road_id, back.lane_id) == (own.road_id, own.lane_id))


def test_sensor_attributes():
    print("\n[sensors] leaderboard specs -> blueprint attributes")
    from runtime.sensors import blueprint_attributes
    cam = blueprint_attributes({"type": "sensor.camera.rgb", "id": "rgb",
                                "width": 900, "height": 256, "fov": 100})
    check("camera size uses image_size_x/y",
          cam.get("image_size_x") == 900 and cam.get("image_size_y") == 256
          and cam.get("fov") == 100, str(cam))
    check("no raw width/height (not blueprint attributes)", "width" not in cam and "height" not in cam)
    check("rgb gets the leaderboard lens and chromatic aberration",
          cam["lens_circle_multiplier"] == 3.0 and cam["chromatic_aberration_intensity"] == 0.5)
    depth = blueprint_attributes({"type": "sensor.camera.depth", "width": 10, "height": 10, "fov": 90})
    check("depth camera has no chromatic aberration", "chromatic_aberration_intensity" not in depth)
    check("gnss gets zero bias", blueprint_attributes({"type": "sensor.other.gnss"})
          == {"noise_alt_bias": 0.0, "noise_lat_bias": 0.0, "noise_lon_bias": 0.0})


def test_language_benchmark():
    print("\n[benchmark] language dataset and scoring")
    from benchmark.language import generate as gen
    from benchmark.language.evaluate import score
    from benchmark.language.paraphrase import faithful, required
    from benchmark.language.state import apply_intent, empty_state, route_key
    from benchmark.language.taxonomy import FUNCTIONS
    from core.landmark_kb import LandmarkKnowledgeBase
    from core.paths import KB_CSV

    kb = LandmarkKnowledgeBase(str(KB_CSV))
    episodes = gen.build(kb, sorted(kb.by_town), reps=1, seed=0)
    covered = {f for e in episodes for t in e["turns"] for f in t["functions"]}
    check("every functional intent is covered", covered == set(FUNCTIONS),
          str(sorted(set(FUNCTIONS) - covered)))
    turns = [t for e in episodes for t in e["turns"]]
    check("gold states are MissionPlanner.apply of the gold intent",
          all(apply_intent(t["state_before"], t["gold_intent"]) == t["state_after"] for t in turns))
    check("templates pass the paraphrase faithfulness check",
          not [t for t in turns if faithful(t["utterance"], required(t))])
    oracle = {(e["id"], i, t["utterance"]): {"intent": t["gold_intent"], "error": None}
              for e in episodes for i, t in enumerate(e["turns"])}
    check("the gold answers score 100%", score(episodes, oracle)["state_acc"] == 100.0)
    silent = {k: {"intent": None, "error": None} for k in oracle}
    scored = [t for t in turns if not t["setup"]]
    rejects = sum(t["gold_intent"] is None for t in scored)
    check("answering nothing only gets the rejections right",
          abs(score(episodes, silent)["state_acc"] - 100.0 * rejects / len(scored)) < 0.1)
    s = empty_state()
    s = apply_intent(s, {"command_type": "plan_route", "destination": "park", "waypoints": []})
    a = apply_intent(s, {"command_type": "insert_stop",
                         "insert": {"type": "cafe", "position": "enroute", "after": "park"}})
    b = apply_intent(s, {"command_type": "new_destination", "destination": "cafe",
                         "keep_waypoints": True})
    check("equivalent intents leave the same route", route_key(a) == route_key(b))
    vague = next(t for e in episodes for t in e["turns"]
                 if "X7" in t["functions"] and t["accept"].get("destination"))
    first = vague["gold_intent"]["commands"][0]
    alt = next(x for x in vague["accept"]["destination"] if x != first["destination"])
    from benchmark.language.evaluate import gold_keys
    pred = apply_intent(vague["state_before"], {"commands": [{**first, "destination": alt}]})
    check("a vague request accepts any matching type", route_key(pred) in gold_keys(vague))
    check("setup turns are kept out of the difficulty tiers",
          all((t["tier"] == 0) == t["setup"] for t in turns))


def test_town_view():
    print("\n[camera] town view framing")
    from types import SimpleNamespace as NS
    from runtime.camera import _town_view

    class FakeMap:
        def __init__(self, w, h):
            self.pts = [(0, 0), (w, 0), (0, h), (w, h)]

        def generate_waypoints(self, _):
            return [NS(transform=NS(location=NS(x=x, y=y, z=0.0))) for x, y in self.pts]

    (cx, cy, z), yaw, span = _town_view(FakeMap(400.0, 200.0), margin=1.0)
    check("centred on the map", (cx, cy) == (200.0, 100.0))
    check("wide map runs x across the screen", yaw == -90.0)
    check("height fits the width", abs(z - 200.0) < 1e-6, f"got {z}")
    check("reports the town's extent", span == 400.0)
    (_, _, z), yaw, _ = _town_view(FakeMap(100.0, 400.0), margin=1.0)
    check("tall map runs y across the screen", yaw == 0.0)
    check("height fits the long side", abs(z - 200.0) < 1e-6, f"got {z}")

    from runtime.camera import SpectatorCamera
    poses = []

    class Spectator:
        def set_transform(self, tf):
            poses.append((tf.location.x, tf.location.y, tf.location.z))

    def camera(w, h):
        world = NS(get_spectator=Spectator, get_map=lambda: FakeMap(w, h))
        car = NS(get_location=lambda: NS(x=1000.0, y=500.0, z=10.0))
        return SpectatorCamera(world, car, transition_s=0.0, margin=1.0)

    small = camera(400.0, 200.0)
    small.show_town([(5000.0, 500.0, 0.0)])
    small.update()
    check("a small town still pulls back to the whole town",      # Location is float32
          all(abs(a - b) < 1e-3 for a, b in zip(poses[-1], small._town)), str(poses[-1]))

    big = camera(9000.0, 8000.0)
    check("a 9 km town counts as large", big.large)
    big.show_town([(1200.0, 500.0, 10.0)])         # a short route, 200 m east
    big.update()
    x, y, z = poses[-1]
    check("large town: frames the ego and the route, not the town",
          (x, y) == (1100.0, 500.0) and z < 200.0, str(poses[-1]))
    big.show_town([(4000.0, 500.0, 10.0)])         # 3 km away: too high to frame
    big.update()
    check("large town: too wide to frame stays over the ego, capped",
          poses[-1] == (1000.0, 500.0, 10.0 + big.max_height_m), str(poses[-1]))



# ─────────────────────────────────────────────
# Backend contracts (verified against the official repos in "AD software/")
# ─────────────────────────────────────────────

LB1_AGENT = """
# leaderboard 1.0 (what TCP uses): __init__ takes the config and calls setup
def get_entry_point(): return "Lb1Agent"
class Lb1Agent:
    def __init__(self, path_to_conf_file):
        self.ctor_arg = path_to_conf_file
        self.setup_calls = 0
        self.setup(path_to_conf_file)
    def setup(self, path_to_conf_file): self.setup_calls += 1
    def sensors(self): return []
"""

LB2_AGENT = """
# leaderboard 2.0 / Bench2Drive / SimLingo: __init__ takes host/port, host calls setup
def get_entry_point(): return "Lb2Agent"
class Lb2Agent:
    def __init__(self, carla_host, carla_port, debug=False):
        self.ctor_arg = (carla_host, carla_port, debug)
        self.setup_calls = 0
    def setup(self, path_to_conf_file): self.setup_calls += 1
    def sensors(self): return []
"""


def test_leaderboard_replan():
    print("\n[backends] leaderboard agent picks up a re-plan")
    from backends.leaderboard import LeaderboardBackend
    from core.types import Observation

    class TcpLike:
        """TCP's shape: the route planner is built once, in _init()."""

        def __init__(self):
            self.initialized = False
            self._global_plan = None
            self.planned = None

        def set_global_plan(self, gps, world):
            self._global_plan = gps

        def run_step(self, input_data, timestamp):
            if not self.initialized:
                self.planned = self._global_plan          # _init()
                self.initialized = True
            return None

    agent = TcpLike()
    be = LeaderboardBackend(agent_instance=agent)
    m = planner()
    m.apply({"command_type": "plan_route", "destination": "hospital", "waypoints": []})
    first = m.build_plan((0.0, 0.0, 0.0))
    be.set_plan(first, m.directive)
    be.run_step(Observation(ego_transform=None, speed_mps=0.0, sensors={}, timestamp=0.0))
    planned_first = agent.planned
    m.apply({"command_type": "new_destination", "destination": "museum"})
    be.set_plan(m.build_plan((0.0, 0.0, 0.0)), m.directive)
    be.run_step(Observation(ego_transform=None, speed_mps=0.0, sensors={}, timestamp=0.0))
    check("agent's route planner is rebuilt from the new plan",
          agent.planned is not planned_first and agent.planned == agent._global_plan)


def test_leaderboard_ctor():
    print("\n[backends] leaderboard agent construction")
    import tempfile
    from backends.leaderboard import LeaderboardBackend

    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "lb1_agent.py")
        p2 = os.path.join(d, "lb2_agent.py")
        open(p1, "w").write(LB1_AGENT)
        open(p2, "w").write(LB2_AGENT)

        a1 = LeaderboardBackend._load(p1, "CONF", "localhost", 2000, False)
        check("leaderboard 1.0 gets the config in its constructor",
              a1.ctor_arg == "CONF", str(a1.ctor_arg))
        check("leaderboard 1.0 setup runs exactly once",
              a1.setup_calls == 1, f"{a1.setup_calls} calls")

        a2 = LeaderboardBackend._load(p2, "CONF", "myhost", 2001, True)
        check("leaderboard 2.0 gets host/port in its constructor",
              a2.ctor_arg == ("myhost", 2001, True), str(a2.ctor_arg))
        check("leaderboard 2.0 setup runs exactly once",
              a2.setup_calls == 1, f"{a2.setup_calls} calls")


def test_downsample_defaults():
    print("\n[backends] downsampling is not applied twice")
    import inspect
    from backends.leaderboard import LeaderboardBackend
    from backends.simlingo import SimLingoBackend

    sig = inspect.signature(LeaderboardBackend.__init__)
    check("leaderboard backend does not pre-downsample",
          sig.parameters["sample_factor"].default is None,
          str(sig.parameters["sample_factor"].default))

    sig = inspect.signature(SimLingoBackend.__init__)
    check("simlingo uses its fork's 200 m sparsity, not 50",
          sig.parameters["route_sample_factor"].default == 200.0,
          str(sig.parameters["route_sample_factor"].default))
    check("simlingo route planner bounds match agent_simlingo.py",
          sig.parameters["min_distance"].default == 7.5
          and sig.parameters["max_distance"].default == 50.0)


def test_llm_bridge_protocol():
    """LLMBridge's subprocess/JSON wiring, against a stub worker -- not the
    real core/llm_worker.py, which needs google-genai and real credentials.
    Proves the plumbing (spawn, ready handshake, parse round-trip, timeout
    surfaced as None, shutdown) without touching Gemini."""
    print("\n[llm] bridge protocol round-trip")
    import sys
    import tempfile
    from core.llm_bridge import LLMBridge

    stub = '''
import json, sys

def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

reply({"status": "ok", "message": "ready"})
for line in sys.stdin:
    req = json.loads(line)
    if req.get("type") == "shutdown":
        reply({"status": "ok"})
        break
    if req.get("type") == "parse":
        if req["text"] == "__fail__":
            reply({"status": "error", "message": "stub-induced failure"})
        else:
            reply({"status": "ok",
                   "intent": {"command_type": "plan_route",
                              "destination": req["text"], "town": req["town"],
                              "route": req.get("route")}})
'''
    with tempfile.TemporaryDirectory() as d:
        stub_path = Path(d) / "stub_worker.py"
        stub_path.write_text(stub, encoding="utf-8")

        bridge = LLMBridge(python_bin=sys.executable,
                           cmd=[sys.executable, str(stub_path)])
        try:
            intent = bridge.parse("hospital", town="Town01")
            check("parse round-trips a real intent", intent is not None
                  and intent["destination"] == "hospital", str(intent))

            failed = bridge.parse("__fail__", town="Town01")
            check("a worker-reported error surfaces as None, not a crash",
                  failed is None)

            routed = bridge.parse("cafe", town="Town01",
                                  route={"destination": "park", "stops": []})
            check("route reaches the worker",
                  routed is not None and routed["route"]["destination"] == "park",
                  str(routed))
            check("no route, no route field on the wire", intent.get("route") is None)

            still_alive = bridge.parse("cafe", town="Town01")
            check("bridge keeps working after a reported error",
                  still_alive is not None and still_alive["destination"] == "cafe",
                  str(still_alive))
        finally:
            proc = bridge.proc
            bridge.close()
        proc.wait(timeout=3.0)
        check("close() actually stops the subprocess", proc.poll() is not None)

        # raise_errors: failures raise; the bridge recovers afterwards.
        strict = LLMBridge(python_bin=sys.executable,
                           cmd=[sys.executable, str(stub_path)], raise_errors=True)
        try:
            try:
                strict.parse("__fail__", town="Town01")
                raised = False
            except RuntimeError:
                raised = True
            check("raise_errors: a worker-reported error raises", raised)
            again = strict.parse("cafe", town="Town01")
            check("raise_errors: the bridge still answers afterwards",
                  again is not None and again["destination"] == "cafe", str(again))
        finally:
            strict.close()


def test_simlingo_frame_matches_official():
    print("\n[geometry] ego frame vs SimLingo's inverse_conversion_2d")
    try:
        import carla
        import numpy as np
    except ImportError:
        print("  skip  carla/numpy not importable")
        return

    def official(point, translation, yaw):
        """team_code/transfuser_utils.py::inverse_conversion_2d, verbatim.
        `translation` is the ego in CARLA world xy and `yaw` its CARLA yaw in
        radians (what preprocess_compass returns)."""
        r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return r.T @ (np.asarray(point) - np.asarray(translation))

    for yaw_deg in (0.0, 37.0, 90.0, 180.0, -125.0):
        tf = carla.Transform(carla.Location(12.0, -4.0, 0.0),
                             carla.Rotation(yaw=yaw_deg))
        for px, py in ((50.0, 20.0), (-30.0, 5.0), (12.0, -4.0)):
            ours = np.asarray(world_to_ego(tf, px, py))
            theirs = official((px, py), (12.0, -4.0), math.radians(yaw_deg))
            if not np.allclose(ours, theirs, atol=1e-4):
                check(f"matches official at yaw={yaw_deg}", False,
                      f"ours={ours} theirs={theirs}")
                return
    check("matches official at every yaw and offset tested", True)


# ─────────────────────────────────────────────
# Package boundary
# ─────────────────────────────────────────────

def test_covlm_stays_out_of_core():
    """covlm/ may import core/; core/ may never import covlm/.

    The CoVLM / InterDrive work lives in covlm/ so the mission layer never
    learns that vocabulary -- scenario ids, r-numbers, vehicle suffixes. This
    check is what keeps that true after someone adds "just one import".
    """
    print("\n[boundary] core/ does not depend on covlm/")
    import ast

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for pkg in ("core", "backends", "runtime", "tools"):
        for path in sorted((root / pkg).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            # ast, not a regex: prose like "core must never import covlm" is a
            # docstring, not a dependency, and a regex cannot tell them apart.
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n == "covlm" or n.startswith("covlm.") for n in names):
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    check("no general package imports covlm", not offenders, ", ".join(offenders))

    check("no InterDrive artefact sits in the general data/",
          not (root / "data" / "route_anchors.json").exists())

    # Everything below only applies when covlm/ is actually checked out. This
    # file must never require it -- that would be the same coupling it is here
    # to forbid, and deleting covlm/ would break the general selftest.
    if (root / "covlm").is_dir():
        check("covlm/ is a package with its own selftest",
              (root / "covlm" / "__init__.py").is_file()
              and (root / "covlm" / "selftest.py").is_file())
        check("covlm keeps its generated data under covlm/data/",
              (root / "covlm" / "data" / "route_anchors.json").is_file())
    else:
        print("  --   covlm/ not checked out; its layout checks skipped")

    cfg = (root / "config.yaml")
    if cfg.is_file():
        from core import paths
        sections = paths._parse_minimal(cfg.read_text(encoding="utf-8"))
        general = set(sections.get("paths", {}))
        check("no InterDrive path leaked into the general [paths] section",
              not (general & {"interdrive_dir", "driver_intents", "opendrive_dir"}),
              str(sorted(general)))


# ─────────────────────────────────────────────
# Portability
# ─────────────────────────────────────────────

def test_config_paths():
    """config.yaml supplies the per-machine paths; the csv ships in the repo."""
    print("\n[paths] config")
    from core import paths

    check("config.yaml is present", paths.CONFIG_FILE.is_file(), str(paths.CONFIG_FILE))

    ok, detail = paths._selftest_config_parsers()
    check("pyyaml and the built-in fallback parser agree on config.yaml", ok, detail)
    check("config.yaml is YAML, not TOML",
          not any(ln.strip().startswith("[") for ln
                  in paths.CONFIG_FILE.read_text(encoding="utf-8").splitlines()),
          "a [section] line means someone followed a stale TOML example")

    example = paths.PKG_ROOT / "config.example.yaml"
    check("config.example.yaml exists, as .gitignore promises", example.is_file())
    if example.is_file():
        live = set(paths._parse_minimal(paths.CONFIG_FILE.read_text(encoding="utf-8")))
        tmpl = set(paths._parse_minimal(example.read_text(encoding="utf-8")))
        check("example and live config have the same sections", live == tmpl,
              f"live={sorted(live)} example={sorted(tmpl)}")
    check("kb csv is set and exists",
          paths.KB_CSV is not None and paths.KB_CSV.is_file(), str(paths.KB_CSV))
    check("kb csv lives inside the repo, so the folder can be moved",
          paths.PKG_ROOT in paths.KB_CSV.parents, str(paths.KB_CSV))
    check("a relative config value resolves from the package root",
          paths._path("paths", "kb_csv") == paths.PKG_ROOT / "data/special_buildings_en.csv",
          str(paths._path("paths", "kb_csv")))


def test_agent_import_roots():
    """An E2E agent needs its whole repo on sys.path, not just its own dir."""
    print("\n[backends] agent repo import roots")
    import tempfile
    from pathlib import Path
    from backends.leaderboard import LeaderboardBackend
    from core.paths import PKG_ROOT

    # Synthetic TCP-shaped tree: leaderboard/team_code nested inside
    # leaderboard/, which is what defeats a naive "looks like a root" check.
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "TCP"
        for sub in ("leaderboard/team_code", "leaderboard/leaderboard",
                    "scenario_runner/srunner", "TCP"):
            (repo / sub).mkdir(parents=True)
        (repo / ".git").mkdir()
        agent = repo / "leaderboard/team_code/tcp_agent.py"
        agent.write_text("")

        roots = LeaderboardBackend._repo_roots(agent)
        for needed, why in ((repo, "repo root, for `TCP.model`"),
                            (repo / "leaderboard", "for `leaderboard.autoagents`"),
                            (repo / "scenario_runner", "for `srunner`"),
                            (repo / "leaderboard/team_code", "for `team_code.planner`")):
            check(f"includes {needed.name or needed} ({why})", needed in roots,
                  str([str(r) for r in roots]))
        check("stops at the repo root, does not leak above it",
              all(repo in r.parents or r == repo for r in roots),
              str([str(r) for r in roots]))

    # The real checkouts, when they are present.
    real = PKG_ROOT / "AD software" / "TCP" / "leaderboard/team_code/tcp_agent.py"
    if real.is_file():
        roots = LeaderboardBackend._repo_roots(real)
        repo = PKG_ROOT / "AD software" / "TCP"
        check("real TCP checkout resolves all four roots",
              all(r in roots for r in (repo, repo / "leaderboard",
                                       repo / "scenario_runner",
                                       repo / "leaderboard/team_code")),
              str([str(r) for r in roots]))
    else:
        print("  skip  no TCP checkout under AD software/")



def main() -> int:
    print("talk2drive_modular self-test (no CARLA server required)")
    for fn in (test_plan_route, test_edits, test_enroute_position,
               test_place_distance, test_landmark_ids, test_route_length,
               test_interaction_policies, test_downsample, test_geometry,
               test_tracker, test_gps, test_arrival, test_stop_progress, test_extend_destination, test_multi_and_move, test_route_prompt,
               test_landmark_board, test_text_only_viz, test_snap_heading, test_sensor_attributes, test_language_benchmark, test_town_view, test_bev_render, test_fpv,
               test_leaderboard_replan, test_leaderboard_ctor, test_downsample_defaults,
               test_llm_bridge_protocol,
               test_simlingo_frame_matches_official,
               test_covlm_stays_out_of_core, test_config_paths,
               test_agent_import_roots):
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
