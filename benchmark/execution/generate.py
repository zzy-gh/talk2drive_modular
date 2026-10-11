"""
benchmark/execution/generate.py
===============================
Closed-loop scenarios for the execution benchmark, generated offline from the
towns' OpenDRIVE files -- no CARLA server.

A scenario is a start pose, a script of *gold* intents with their triggers,
and the expected outcome. The outcome is not written by hand: an offline
drive follows each planned route point by point, fires the events where the
runner will fire them, re-plans exactly as the session does
(``CarlaRouteProvider.snap_heading`` + ``MissionPlanner``), and records which
places it passes. A scenario is kept only if it is well posed: route length
in range, every trigger fires, a removed or replaced place stays clear of the
route actually driven, a "behind" destination really is behind.

A stop counts as visited when the car passes within ``VISIT_M`` of it.
Every place carries its index, so no disambiguation is ever asked.

    python3 benchmark/execution/generate.py        # -> data/scenarios.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.landmark_kb import LandmarkKnowledgeBase      # noqa: E402
from core.mission import MissionPlanner                 # noqa: E402
from core.paths import KB_CSV, section_path             # noqa: E402
from core.resolver import Interaction                   # noqa: E402
from core.route_provider import CarlaRouteProvider      # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "scenarios.jsonl"
VISIT_M = 5.0              # pass within this of a stop = visited (MissionPlanner.arrival_radius)
CLEAR_M = 15.0             # a place that must not be visited stays at least this far away
LOOK_AHEAD_M = 8.0         # the session plans from this far ahead of the car
TRIGGER_MARGIN_M = 30.0    # a distance trigger fires at least this far before the next stop
LENGTH_M = (400.0, 900.0)  # first planned route
MAX_TOTAL_M = 2000.0       # everything driven in one scenario
ATTEMPTS = 600             # random draws per template and town before deferring it
MIN_SPEED_MPS = 3.0        # time limit = distance driven / this + 60 s

# What each scenario template exercises (execution-level functions).
TEMPLATES = {
    "plan_two_stops":       ["P1", "P3"],
    "plan_two_stops_hurry": ["P1", "P3", "P5"],
    "insert_next_last":     ["P3", "I1", "I2"],
    "insert_unordered_before": ["P3", "I3", "I6"],
    "insert_after_remove":  ["P3", "I4", "R1"],
    "move_next_last":       ["P3", "O1", "O2"],
    "move_relative":        ["P3", "O3"],
    "replace_behind":       ["P1", "D1"],
    "extend":               ["P3", "D2"],
    "rush_elsewhere":       ["P1", "D4"],
    "cancel":               ["P3", "C1"],
    "two_commands":         ["P3", "M1", "R1", "I1"],
    "three_commands":       ["P3", "M2", "R1", "I1", "D2"],
}
# plan_two_stops and its hurry twin always come as a pair in one town.
CYCLE = ["pair", "insert_next_last", "insert_unordered_before", "insert_after_remove",
         "move_next_last", "move_relative", "replace_behind", "extend", "rush_elsewhere",
         "cancel", "two_commands", "three_commands"]


class Infeasible(Exception):
    pass


class Scripted(Interaction):
    """The passenger, as the script answers: never asked which place (every
    place is indexed); asked where an unordered stop goes, answers ``at``."""

    def __init__(self) -> None:
        self.at = None

    def choose_place(self, req):
        raise Infeasible("a place without an index reached disambiguation")

    def choose_insert_position(self, req):
        if self.at is None:
            raise Infeasible("an unordered stop with no scripted answer")
        return min(self.at, len(req.stop_labels))


def key(label: str, index: int | None) -> str:
    return f"{label}#{index}" if index else label


# ─────────────────────────────────────────────
# Offline drive
# ─────────────────────────────────────────────

def drive(town: "Town", start: dict, events: list[dict]) -> dict:
    """Follow the plan point by point, firing ``events`` where the runner
    would, and return what happened: visits in order, arrival, the path."""
    routes = town.routes
    inter = Scripted()
    m = MissionPlanner(town.kb, routes, inter, town=town.name)
    pending = list(events)
    visits, path, fired = [], [], []
    driven = 0.0
    x, y, z, yaw = start["x"], start["y"], start["z"], start["yaw"]
    plan, i, cancelled = None, 0, False

    def fire(ev) -> None:
        nonlocal plan, i, cancelled
        inter.at = (ev.get("answer") or {}).get("insert_at")
        result = m.apply(ev["intent"])
        if not result.ok:
            raise Infeasible(f"event did not apply: {result.message}")
        fired.append({"driven_m": round(driven, 1), "x": round(x, 1), "y": round(y, 1)})
        if result.cancelled:
            cancelled = True
            return
        s = routes.snap_heading(x, y, z, yaw, ahead_m=LOOK_AHEAD_M)
        new = m.build_plan(s)
        if new is None:
            raise Infeasible("re-plan failed")
        check_plan(new)
        plan, i = new, 0

    fire(pending.pop(0))                                     # the start event
    for _ in range(100000):
        if cancelled:
            break
        if i >= len(plan.points):
            raise Infeasible("ran off the end of a plan")
        p = plan.points[i]
        if path:
            driven += math.hypot(p.x - path[-1][0], p.y - path[-1][1])
        x, y, z, yaw = p.x, p.y, p.z, p.yaw
        path.append((x, y))
        reached = m.reached_stop(x, y)
        if reached is not None:
            visits.append(town.keyof(reached))
        if pending:
            w = pending[0]["when"]
            if ("driven_m" in w and driven >= w["driven_m"]) or (
                    "reached" in w and reached is not None
                    and town.keyof(reached) == w["reached"]):
                fire(pending.pop(0))
                continue
        if m.is_arrived(x, y):
            goal = town.keyof(m.plan.goal)
            return {"visits": visits, "arrive": goal, "cancelled": False,
                    "driven_m": driven, "path": path, "fired": fired, "left": pending}
        i += 1
    return {"visits": visits, "arrive": None, "cancelled": cancelled, "driven_m": driven,
            "path": path, "fired": fired, "left": pending}


def check_plan(plan) -> None:
    """The plan really drives past every stop and the goal, and comes near
    them in the planned order. build_plan skips a stop whose leg fails to
    trace, and a later stop passed on the way to an earlier one is never
    counted (visits are matched in order) -- both make a scenario ill posed."""
    for a, b in zip(plan.points, plan.points[1:]):
        if math.hypot(b.x - a.x, b.y - a.y) > 10.0:
            raise Infeasible("the plan jumps")
    targets = list(plan.stops) + [plan.goal]
    first = []
    for t in targets:
        n = next((i for i, p in enumerate(plan.points)
                  if math.hypot(p.x - t.x, p.y - t.y) <= VISIT_M), None)
        if n is None:
            raise Infeasible(f"the plan never passes {t.label}")
        first.append(n)
    if first != sorted(first):
        raise Infeasible("the plan passes its stops out of order")


def min_dist(path, place) -> float:
    return min(math.hypot(px - place.x, py - place.y) for px, py in path)


# ─────────────────────────────────────────────
# Templates: start pose, places, events
# ─────────────────────────────────────────────

def place_table(kb, routes: CarlaRouteProvider, town: str) -> dict:
    """"label#i" -> the road-snapped Place, exactly as MissionPlanner resolves
    an indexed place. Shared with the runner, so both judge visits against
    the same points."""
    planner = MissionPlanner(kb, routes, town=town)
    return {key(label, i): p
            for label in kb.get_landmark_types(town)
            for i, p in enumerate(planner._candidates(label), start=1)}


class Town:
    def __init__(self, name: str, xodr: Path, kb) -> None:
        import carla
        self.name = name
        self.map = carla.Map(name, xodr.read_text())
        self.routes = CarlaRouteProvider(self.map, sampling_resolution=1.0)
        self.kb = kb
        self.places = place_table(kb, self.routes, name)
        self.starts = [w.transform for w in self.map.generate_waypoints(10.0)
                       if not w.is_junction and w.lane_type == carla.LaneType.Driving]
        self._by_pos = {(p.label, round(p.x, 1), round(p.y, 1)): k for k, p in self.places.items()}

    def keyof(self, place) -> str:
        """'label#index' of a planned Place (Place itself carries no index)."""
        return self._by_pos[(place.label, round(place.x, 1), round(place.y, 1))]


def plan_cmd(dest, stops=(), hurry=False) -> dict:
    lab, idx = dest.split("#") if "#" in dest else (dest, None)
    return {"command_type": "plan_route", "destination": lab,
            "destination_index": int(idx) if idx else None,
            "waypoints": [{"type": s.split("#")[0], "order": n + 1,
                           "index": int(s.split("#")[1]) if "#" in s else None}
                          for n, s in enumerate(stops)],
            "urgency": "high" if hurry else "normal", "avoid": [], "route_preference": None}


def _lab(k):
    return k.split("#")[0]


def _idx(k):
    return int(k.split("#")[1]) if "#" in k else None


def insert_cmd(k, position="enroute", after=None, before=None) -> dict:
    return {"command_type": "insert_stop",
            "insert": {"type": _lab(k), "index": _idx(k), "position": position,
                       "after": after and _lab(after), "before": before and _lab(before)}}


def move_cmd(k, position=None, after=None, before=None) -> dict:
    return {"command_type": "move_stop",
            "move": {"type": _lab(k), "position": position,
                     "after": after and _lab(after), "before": before and _lab(before)}}


def remove_cmd(k) -> dict:
    return {"command_type": "remove_stop", "remove": {"type": _lab(k)}}


def dest_cmd(k, keep=False, hurry=False) -> dict:
    return {"command_type": "new_destination", "destination": _lab(k),
            "destination_index": _idx(k), "keep_waypoints": keep,
            "urgency": "high" if hurry else "normal"}


def ev(when, *commands, answer=None) -> dict:
    e = {"when": when, "intent": {"commands": list(commands)}}
    if answer:
        e["answer"] = answer
    return e


def leg_trigger(town: Town, start, dest, stops, frac=0.5) -> float:
    """A distance along the first planned leg, ``frac`` of the way to the
    first stop but at least TRIGGER_MARGIN_M short of it."""
    out = drive(town, start, [ev({"start": True}, plan_cmd(dest, stops))])
    first = town.places[stops[0]] if stops else town.places[dest]
    d_first = next((d for d, (px, py) in _cumulative(out["path"])
                    if math.hypot(px - first.x, py - first.y) <= VISIT_M), None)
    if d_first is None or d_first < 2 * TRIGGER_MARGIN_M:
        raise Infeasible("first leg too short to trigger on")
    return round(min(d_first * frac, d_first - TRIGGER_MARGIN_M), 1)


def _cumulative(path):
    d = 0.0
    for n, pt in enumerate(path):
        if n:
            d += math.hypot(pt[0] - path[n - 1][0], pt[1] - path[n - 1][1])
        yield d, pt


def build(town: Town, template: str, rng: random.Random) -> list[dict]:
    """One scenario (two for the hurry pair) of ``template``; Infeasible if
    this draw is not well posed."""
    t = town.starts[rng.randrange(len(town.starts))]
    start = {"x": round(t.location.x, 2), "y": round(t.location.y, 2),
             "z": round(t.location.z, 2), "yaw": round(t.rotation.yaw, 1)}
    labels = sorted({_lab(k) for k in town.places})
    pick = rng.sample(labels, min(len(labels), 6))           # distinct types: missions key by type
    if len(pick) < 5:
        raise Infeasible("too few place types")
    k = [rng.choice([p for p in town.places if _lab(p) == lab]) for lab in pick]
    dest, a, b, c, x = k[0], k[1], k[2], k[3], k[4]

    if template in ("pair", "plan_two_stops", "plan_two_stops_hurry"):
        out = []
        for hurry in (False, True):
            name = "plan_two_stops_hurry" if hurry else "plan_two_stops"
            out.append((name, [ev({"start": True}, plan_cmd(dest, [a, b], hurry))]))
        return [finish(town, n, start, e) for n, e in out]
    if template == "insert_next_last":
        d = leg_trigger(town, start, dest, [a])
        events = [ev({"start": True}, plan_cmd(dest, [a])),
                  ev({"driven_m": d}, insert_cmd(x, "next")),
                  ev({"reached": x}, insert_cmd(b, "last"))]
    elif template == "insert_unordered_before":
        d = leg_trigger(town, start, dest, [a, b])
        events = [ev({"start": True}, plan_cmd(dest, [a, b])),
                  ev({"driven_m": d}, insert_cmd(x), answer={"insert_at": 1}),
                  ev({"reached": a}, insert_cmd(c, before=b))]
    elif template == "insert_after_remove":
        d = leg_trigger(town, start, dest, [a, b])
        events = [ev({"start": True}, plan_cmd(dest, [a, b])),
                  ev({"driven_m": d}, insert_cmd(x, after=a)),
                  ev({"reached": a}, remove_cmd(b))]
    elif template == "move_next_last":
        d = leg_trigger(town, start, dest, [a, b, c])
        events = [ev({"start": True}, plan_cmd(dest, [a, b, c])),
                  ev({"driven_m": d}, move_cmd(c, "next")),
                  ev({"reached": c}, move_cmd(a, "last"))]
    elif template == "move_relative":
        # Two stops are enough to reorder relative to another stop, and far
        # easier to place well than three: a -> b becomes b -> a.
        d = leg_trigger(town, start, dest, [a, b])
        move = move_cmd(a, after=b) if rng.random() < 0.5 else move_cmd(b, before=a)
        events = [ev({"start": True}, plan_cmd(dest, [a, b])),
                  ev({"driven_m": d}, move)]
    elif template == "replace_behind":
        d = leg_trigger(town, start, dest, [], frac=0.4)
        events = [ev({"start": True}, plan_cmd(dest)),
                  ev({"driven_m": d}, dest_cmd(x))]
    elif template == "extend":
        events = [ev({"start": True}, plan_cmd(dest, [a])),
                  ev({"reached": a}, dest_cmd(x, keep=True))]
    elif template == "rush_elsewhere":
        d = leg_trigger(town, start, dest, [], frac=0.4)
        events = [ev({"start": True}, plan_cmd(dest)),
                  ev({"driven_m": d}, dest_cmd(x, hurry=True))]
    elif template == "cancel":
        events = [ev({"start": True}, plan_cmd(dest, [a])),
                  ev({"reached": a}, {"command_type": "cancel_route"})]
    elif template == "two_commands":
        d = leg_trigger(town, start, dest, [a, b])
        events = [ev({"start": True}, plan_cmd(dest, [a, b])),
                  ev({"driven_m": d}, remove_cmd(b), insert_cmd(x, "next"))]
    elif template == "three_commands":
        d = leg_trigger(town, start, dest, [a, b])
        events = [ev({"start": True}, plan_cmd(dest, [a, b])),
                  ev({"driven_m": d}, remove_cmd(b), insert_cmd(x, "next"),
                     dest_cmd(c, keep=True))]
    else:
        raise ValueError(template)
    return [finish(town, template, start, events)]


def finish(town: Town, template: str, start: dict, events: list[dict]) -> dict:
    first = drive(town, start, events[:1])
    if not LENGTH_M[0] <= first["driven_m"] <= LENGTH_M[1]:
        raise Infeasible(f"first route {first['driven_m']:.0f} m")
    out = drive(town, start, events)
    if out["driven_m"] > MAX_TOTAL_M:
        raise Infeasible(f"{out['driven_m']:.0f} m in total")
    if out["left"]:
        raise Infeasible("a trigger never fired")
    if not out["cancelled"] and out["arrive"] is None:
        raise Infeasible("never arrived")

    # Places the passenger dropped must stay clear of the route driven.
    mentioned = set()
    for e in events:
        for c in e["intent"]["commands"]:
            for f in ("destination",):
                if c.get(f):
                    mentioned.add(key(c[f], c.get("destination_index")))
            for w in c.get("waypoints") or []:
                mentioned.add(key(w["type"], w["index"]))
            for sub in ("insert", "move", "remove"):
                if c.get(sub):
                    k_ = c[sub]["type"]
                    match = [p for p in mentioned if _lab(p) == k_]
                    mentioned.add(match[0] if match else key(k_, c[sub].get("index")))
    reached = set(out["visits"]) | ({out["arrive"]} if out["arrive"] else set())
    never = sorted(mentioned - reached)
    for p in never:
        if min_dist(out["path"], town.places[p]) < CLEAR_M:
            raise Infeasible(f"{p} lies on the route driven")
    if template == "replace_behind":
        f = out["fired"][1]
        new = town.places[events[1]["intent"]["commands"][0]["destination"] + "#"
                          + str(events[1]["intent"]["commands"][0]["destination_index"])]
        heading = _heading_at(out["path"], f["x"], f["y"])
        if (new.x - f["x"]) * heading[0] + (new.y - f["y"]) * heading[1] >= 0:
            raise Infeasible("new destination is not behind")
    if template == "cancel" and not out["cancelled"]:
        raise Infeasible("cancel never fired")

    return {"template": template, "functions": TEMPLATES[template], "town": town.name,
            "start": start, "events": events,
            "expect": {"visits": out["visits"], "arrive": out["arrive"],
                       "cancelled": out["cancelled"], "never_visit": never,
                       "hurry": any(c.get("urgency") == "high" for e in events
                                    for c in e["intent"]["commands"]),
                       "planned_m": round(out["driven_m"], 1),
                       "time_limit_s": round(out["driven_m"] / MIN_SPEED_MPS + 60.0)}}


def _heading_at(path, x, y):
    n = min(range(len(path) - 1), key=lambda i: math.hypot(path[i][0] - x, path[i][1] - y))
    dx, dy = path[n + 1][0] - path[n][0], path[n + 1][1] - path[n][1]
    norm = math.hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-town", type=int, default=5, help="most scenarios one town takes")
    ap.add_argument("--reps", type=int, default=3, help="scenarios wanted per template")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xodr", default=None, help="default: [covlm] opendrive_dir")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    xodr_dir = Path(args.xodr) if args.xodr else section_path("covlm", "opendrive_dir")
    kb = LandmarkKnowledgeBase(str(KB_CSV))
    # Every template is wanted REPS times. Each town takes the templates
    # still most wanted first; one it cannot pose well is simply left for the
    # towns after it. Stops once every template has its REPS.
    want = {t: args.reps for t in CYCLE}
    scenarios, n = [], 0
    for name in sorted(kb.by_town):
        town = Town(name, xodr_dir / f"{name}.xodr", kb)
        slots = args.per_town
        for template in sorted(CYCLE, key=lambda t: (-want[t], CYCLE.index(t))):
            size = 2 if template == "pair" else 1
            if want[template] <= 0 or slots < size:
                continue
            t_start = time.time()
            for attempt in range(ATTEMPTS):
                rng = random.Random(f"{args.seed}/{name}/{template}/{n}/{attempt}")
                try:
                    made = build(town, template, rng)
                except Infeasible:
                    continue
                n += 1
                print(f"[generate] {name}: {template} after {attempt + 1} draws "
                      f"({time.time() - t_start:.0f} s)", flush=True)
                for sc in made:
                    sc["id"] = f"{name}/{sc['template']}/{n}"
                scenarios += made
                slots -= size
                want[template] -= 1
                break
            else:
                print(f"[generate] {name}: no well-posed {template} in {ATTEMPTS} draws",
                      flush=True)
    unmet = {t: k for t, k in want.items() if k > 0}
    if unmet:
        print(f"[generate] still wanted after every town: {unmet}")

    funcs = Counter(f for s in scenarios for f in s["functions"])
    towns_per = {f: len({s["town"] for s in scenarios if f in s["functions"]}) for f in funcs}
    print(f"[generate] {len(scenarios)} scenarios")
    for f, n in sorted(funcs.items()):
        print(f"  {f}: {n} scenarios in {towns_per[f]} towns")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for s in scenarios:
            fh.write(json.dumps(s) + "\n")
    print(f"[generate] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
