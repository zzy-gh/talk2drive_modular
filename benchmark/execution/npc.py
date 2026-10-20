"""
benchmark/execution/npc.py
==========================
Traffic for the execution scenarios, placed where the ego will meet it.

A town-wide random fleet mostly drives where the ego never goes. Here every
scenario's whole drive -- the route after every update, from the same
offline replay that gives the expected outcome -- is replayed once, and
vehicles are put along it:

* **lead**     on the ego's lane ahead of it, slower than the limit and
               told to follow the ego's own route, so the ego catches up and
               has to follow or brake behind it;
* **oncoming** in the opposite lane next to the route, driving towards it;
* **crossing** on another approach of each junction the route passes,
               a little before the junction, so it reaches the junction
               while the ego may.

The layout is fixed per scenario (seeded), so every backend meets the same
traffic; Traffic Manager runs synchronously with a fixed seed in run.py.

    python3 benchmark/execution/npc.py        # -> data/npc_layout.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.execution import generate as G                        # noqa: E402
from core.landmark_kb import LandmarkKnowledgeBase                   # noqa: E402
from core.paths import KB_CSV, section_path                          # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "npc_layout.jsonl"
LEAD_EVERY_M = 120.0       # a slower vehicle ahead on the route this often
ONCOMING_EVERY_M = 150.0   # one in the opposite lane this often
CROSS_BACK_M = 30.0        # a crossing vehicle starts this far before its junction
START_CLEAR_M = 30.0       # nothing spawns this close to the ego's start
SPACING_M = 10.0           # nor this close to another vehicle
MAX_NPC = 25
LEAD_SLOWER_PCT = 30.0     # lead vehicles drive 30% below the speed limit
LEAD_PATH_M = 400.0        # how far along the ego's route a lead vehicle is told to go
# Ordinary passenger cars (four wheels, similar size), so every scenario gets
# a comparable fleet whatever the town.
BLUEPRINTS = ["vehicle.audi.a2", "vehicle.audi.tt", "vehicle.citroen.c3",
              "vehicle.mini.cooper_s", "vehicle.nissan.micra", "vehicle.seat.leon",
              "vehicle.toyota.prius", "vehicle.volkswagen.t2", "vehicle.ford.mustang",
              "vehicle.chevrolet.impala"]


def _arc(path):
    """Cumulative arc length along the path."""
    s, out = 0.0, [0.0]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        s += math.hypot(x1 - x0, y1 - y0)
        out.append(s)
    return out


def _tf(wp) -> dict:
    t = wp.transform
    return {"x": round(t.location.x, 2), "y": round(t.location.y, 2),
            "z": round(t.location.z, 2), "yaw": round(t.rotation.yaw, 1)}


def layout(town, sc: dict, rng: random.Random) -> list[dict]:
    import carla
    out = G.drive(town, sc["start"], sc["events"])
    path = out["path"]
    if len(path) < 2:
        return []
    arc = _arc(path)
    total = arc[-1]
    m = town.map
    start = (sc["start"]["x"], sc["start"]["y"])
    placed: list[tuple[float, float]] = []

    def free(x, y) -> bool:
        if math.hypot(x - start[0], y - start[1]) < START_CLEAR_M:
            return False
        return all(math.hypot(x - px, y - py) >= SPACING_M for px, py in placed)

    def at(s):
        i = min(range(len(arc)), key=lambda k: abs(arc[k] - s))
        return i, path[i]

    npcs = []

    # Lead vehicles: on the ego's lane, ahead, following the ego's route.
    s = 60.0
    while s < total - 40.0 and len(npcs) < MAX_NPC:
        i, (x, y) = at(s + rng.uniform(-15, 15))
        wp = m.get_waypoint(carla.Location(x=x, y=y), project_to_road=True,
                            lane_type=carla.LaneType.Driving)
        if wp and not wp.is_junction and free(wp.transform.location.x, wp.transform.location.y):
            route = [path[k] for k in range(i, len(path))
                     if arc[k] - arc[i] <= LEAD_PATH_M][::10]
            npcs.append({"role": "lead", **_tf(wp),
                         "path": [[round(px, 1), round(py, 1)] for px, py in route],
                         "slower_pct": LEAD_SLOWER_PCT})
            placed.append((wp.transform.location.x, wp.transform.location.y))
        s += LEAD_EVERY_M

    # Oncoming: the nearest lane of the opposite direction.
    s = 80.0
    while s < total - 20.0 and len(npcs) < MAX_NPC:
        _, (x, y) = at(s + rng.uniform(-20, 20))
        wp = m.get_waypoint(carla.Location(x=x, y=y), project_to_road=True,
                            lane_type=carla.LaneType.Driving)
        opp = None
        if wp and not wp.is_junction:
            side = wp.get_left_lane()
            while side and side.lane_type == carla.LaneType.Driving \
                    and side.lane_id * wp.lane_id > 0:
                side = side.get_left_lane()
            if side and side.lane_type == carla.LaneType.Driving and side.lane_id * wp.lane_id < 0:
                opp = side
        if opp and free(opp.transform.location.x, opp.transform.location.y):
            npcs.append({"role": "oncoming", **_tf(opp)})
            placed.append((opp.transform.location.x, opp.transform.location.y))
        s += ONCOMING_EVERY_M

    # Crossing: one per junction on the route, on an approach the route does not use.
    seen = set()
    for k in range(0, len(path), 3):
        if len(npcs) >= MAX_NPC:
            break
        x, y = path[k]
        wp = m.get_waypoint(carla.Location(x=x, y=y), project_to_road=True,
                            lane_type=carla.LaneType.Driving)
        if not wp or not wp.is_junction:
            continue
        j = wp.get_junction()
        if j.id in seen:
            continue
        seen.add(j.id)
        entries = [a for a, _ in j.get_waypoints(carla.LaneType.Driving)]
        rng.shuffle(entries)
        for e in entries:
            back = e.previous(CROSS_BACK_M)
            if not back:
                continue
            b = back[0]
            bx, by = b.transform.location.x, b.transform.location.y
            # Not on the ego's own approach: none of the route within 6 m.
            if any(abs(bx - px) < 6 and abs(by - py) < 6 for px, py in path):
                continue
            if b.is_junction or not free(bx, by):
                continue
            npcs.append({"role": "crossing", **_tf(b)})
            placed.append((bx, by))
            break

    for n in npcs:
        n["blueprint"] = rng.choice(BLUEPRINTS)
    return npcs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=str(G.OUT))
    ap.add_argument("--xodr", default=None, help="default: [covlm] opendrive_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    xodr = Path(args.xodr) if args.xodr else section_path("covlm", "opendrive_dir")
    kb = LandmarkKnowledgeBase(str(KB_CSV))
    with open(args.scenarios, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]
    towns, rows = {}, []
    for sc in scenarios:
        if sc["town"] not in towns:
            towns[sc["town"]] = G.Town(sc["town"], xodr / f"{sc['town']}.xodr", kb)
        rng = random.Random(f"{args.seed}/{sc['id']}/npc")
        npcs = layout(towns[sc["town"]], sc, rng)
        roles = {r: sum(n["role"] == r for n in npcs) for r in ("lead", "oncoming", "crossing")}
        print(f"[npc] {sc['id']}: {len(npcs)} vehicles {roles}", flush=True)
        rows.append({"id": sc["id"], "seed": args.seed, "npcs": npcs})
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[npc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
