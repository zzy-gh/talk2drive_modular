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

    r = m.apply({"command_type": "cancel_route", "pull_over": True})
    check("cancel reports cancelled", r.cancelled and r.pull_over)
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


def test_town_view():
    print("\n[camera] town view framing")
    from types import SimpleNamespace as NS
    from runtime.camera import _town_view

    class FakeMap:
        def __init__(self, w, h):
            self.pts = [(0, 0), (w, 0), (0, h), (w, h)]

        def generate_waypoints(self, _):
            return [NS(transform=NS(location=NS(x=x, y=y, z=0.0))) for x, y in self.pts]

    (cx, cy, z), yaw = _town_view(FakeMap(400.0, 200.0), margin=1.0)
    check("centred on the map", (cx, cy) == (200.0, 100.0))
    check("wide map runs x across the screen", yaw == -90.0)
    check("height fits the width", abs(z - 200.0) < 1e-6, f"got {z}")
    (_, _, z), yaw = _town_view(FakeMap(100.0, 400.0), margin=1.0)
    check("tall map runs y across the screen", yaw == 0.0)
    check("height fits the long side", abs(z - 200.0) < 1e-6, f"got {z}")



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
               test_interaction_policies, test_downsample, test_geometry,
               test_tracker, test_gps, test_arrival, test_stop_progress, test_extend_destination, test_route_prompt,
               test_landmark_board, test_text_only_viz, test_snap_heading, test_sensor_attributes, test_town_view, test_bev_render,
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
