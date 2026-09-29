"""
tools/smoke_carla.py
====================
In-the-loop smoke test. Needs a running CARLA server; needs no LLM and no
model checkpoint — the intent is hand-written, so this isolates the plumbing
(route provider -> plan -> backend -> vehicle) from the language layer.

    python tools/smoke_carla.py --town 1 --backend basic_agent --seconds 25

What it proves: a plan is produced, the backend accepts it, the car actually
moves along it, and a mid-drive re-plan swaps routes without stopping the car.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla

import backends
from core.landmark_kb import LandmarkKnowledgeBase
from core.llm_inference import normalize_town
from core.mission import MissionPlanner
from core.resolver import AutoInteraction
from core.route_provider import CarlaRouteProvider
from runtime.session import DriveSession, SessionConfig

from core.paths import KB_CSV


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--town", default="1")
    p.add_argument("--backend", default="basic_agent")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--seconds", type=float, default=25.0)
    p.add_argument("--csv", default=None)
    p.add_argument("--spawn", type=int, default=0)
    args = p.parse_args()

    town = normalize_town(args.town)
    kb = LandmarkKnowledgeBase(str(args.csv or KB_CSV))
    types = kb.get_landmark_types(town)
    if len(types) < 2:
        print(f"[smoke] {town} has too few landmark types: {types}")
        return 1
    print(f"[smoke] {town} landmark types: {types}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(town)

    mission = MissionPlanner(kb, CarlaRouteProvider(world.get_map()),
                             interaction=AutoInteraction(), town=town)
    backend = backends.build(args.backend)
    session = DriveSession(world, backend, mission,
                           SessionConfig(spawn_index=args.spawn))

    failures = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
        if not cond:
            failures.append(name)

    try:
        print(f"\n[smoke] planning to '{types[0]}'")
        session.submit({"command_type": "plan_route", "destination": types[0],
                        "waypoints": [], "urgency": "normal", "_utterance": "smoke test"})
        check("plan produced", mission.plan is not None)
        if mission.plan is None:
            return 1
        check("plan is non-trivial", len(mission.plan) > 10, f"{len(mission.plan)} pts")

        start = session.ego_xyz()
        half = args.seconds / 2
        time.sleep(half)
        mid = session.ego_xyz()
        moved = math.dist(start[:2], mid[:2])
        check("vehicle moved", moved > 3.0, f"{moved:.1f} m")
        check("speed is sane", session.speed_mps() < 40.0,
              f"{session.speed_mps():.1f} m/s")

        print(f"\n[smoke] mid-drive re-plan to '{types[1]}'")
        rev_before = mission.plan.revision
        session.submit({"command_type": "new_destination", "destination": types[1],
                        "keep_waypoints": False, "urgency": "high",
                        "_utterance": "smoke replan"})
        check("re-plan bumped the revision",
              mission.plan is not None and mission.plan.revision > rev_before)
        check("urgency propagated", mission.directive.urgency == "high")

        before = session.ego_xyz()
        time.sleep(half)
        after = session.ego_xyz()
        check("still driving after the re-plan",
              math.dist(before[:2], after[:2]) > 1.0,
              f"{math.dist(before[:2], after[:2]):.1f} m")

        print("\n[smoke] cancelling")
        session.clear()
        time.sleep(1.5)
        check("cancel stops the car", session.speed_mps() < 3.0,
              f"{session.speed_mps():.1f} m/s")
    finally:
        session.close()

    print(f"\n[smoke] {'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
