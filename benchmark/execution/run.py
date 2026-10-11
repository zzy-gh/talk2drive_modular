"""
benchmark/execution/run.py
==========================
Drive every scenario in CARLA with one backend and log what happened.

Each scenario: load the town, step synchronously, spawn the ego at the start
pose, hand the start intent to a ``DriveSession``, and fire the remaining
intents from a per-step observer -- on the control thread, so the simulation
is held still while an event is applied and re-planned and every run of a
scenario is the same. The run ends on arrival, on the car standing after a
cancel, on the time limit, or when the car has not moved for a minute.

The log (pose, speed, control per step; collisions; lane invasions; whether
the car was against its lane's direction) is scored by evaluate.py.

    python3 benchmark/execution/run.py --backend basic_agent
    python3 benchmark/execution/run.py --backend simlingo --only Town03/
    python3 benchmark/execution/run.py --backend basic_agent --only Town01/plan --watch --rerun
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import carla                                                         # noqa: E402

from benchmark.execution.generate import (OUT as SCENARIOS, VISIT_M, Scripted,  # noqa: E402
                                          place_table)
from core.landmark_kb import LandmarkKnowledgeBase                   # noqa: E402
from core.mission import MissionPlanner                              # noqa: E402
from core.paths import KB_CSV                                        # noqa: E402
from core.route_provider import CarlaRouteProvider                   # noqa: E402
from runtime.cli import add_backend_args, backend_defaults, build_backend  # noqa: E402
from runtime.session import DriveSession, SessionConfig              # noqa: E402

RUNS = ROOT / "benchmark" / "runs" / "execution"
STAND_MPS = 0.5          # "stopped"
CANCEL_SETTLE_S = 2.0    # after a cancel, stopped this long = done
CANCEL_LIMIT_S = 20.0    # ... or give up this long after the cancel
STUCK_S = 60.0           # driving but not moving this long = stuck


def wrong_way(carla_map, x, y, z, yaw) -> bool:
    """Facing against the lane it is on (outside junctions)."""
    wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=False,
                                lane_type=carla.LaneType.Driving)
    if wp is None or wp.is_junction:
        return False
    return abs((wp.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0) > 90.0


def run_scenario(client, args, vehicle_bp: str, sc: dict, kb) -> dict:
    world = client.load_world(sc["town"])
    carla_map = world.get_map()
    routes = CarlaRouteProvider(carla_map, sampling_resolution=1.0)
    places = place_table(kb, routes, sc["town"])
    inter = Scripted()
    mission = MissionPlanner(kb, routes, inter, town=sc["town"])
    backend = build_backend(args, world, client)

    bp = world.get_blueprint_library().find(vehicle_bp)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    s = sc["start"]
    vehicle = world.try_spawn_actor(bp, carla.Transform(
        carla.Location(x=s["x"], y=s["y"], z=s["z"] + 0.5), carla.Rotation(yaw=s["yaw"])))
    if vehicle is None:
        backend.destroy()
        return {"id": sc["id"], "result": "spawn_failed"}

    collisions, invasions = [], []
    bl = world.get_blueprint_library()
    col = world.spawn_actor(bl.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
    lane = world.spawn_actor(bl.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=vehicle)
    col.listen(lambda e: collisions.append({"t": e.timestamp, "other": e.other_actor.type_id}))
    lane.listen(lambda e: invasions.append({"t": e.timestamp, "types": sorted(
        {str(m.type) for m in e.crossed_lane_markings})}))

    # --watch: spectator on the ego, the route and EGO drawn as text (never
    # in a camera image), real-time pace. Off for benchmark runs.
    w = args.watch
    session = DriveSession(world, backend, mission,
                           SessionConfig(control_hz=args.control_hz, sync=True, realtime=w,
                                         visualize=w, viz_text_only=True, camera=w, beacon=w),
                           vehicle=vehicle)

    events = list(sc["events"])
    st = {"t0": None, "driven": 0.0, "last": None, "fired": [], "log": [],
          "still_since": None, "cancel_t": None, "result": None}
    done = threading.Event()
    limit = sc["expect"]["time_limit_s"]

    def fire(sess, ev, t) -> None:
        inter.at = (ev.get("answer") or {}).get("insert_at")
        result = sess.submit(ev["intent"])
        st["fired"].append({"t": round(t - st["t0"], 2), "driven_m": round(st["driven"], 1),
                            "ok": result.ok, "message": result.message})
        if result.cancelled:
            st["cancel_t"] = t

    def observe(sess, info) -> None:
        if done.is_set():
            return
        t = info["sim_time"]
        if st["t0"] is None:
            st["t0"] = t
        if st["last"] is not None:
            st["driven"] += math.hypot(info["x"] - st["last"][0], info["y"] - st["last"][1])
        st["last"] = (info["x"], info["y"])
        c = info["control"] or (0.0, 0.0, 0.0)
        st["log"].append([round(t - st["t0"], 3), round(info["x"], 2), round(info["y"], 2),
                          round(info["yaw"], 1), round(info["speed"], 2),
                          round(c[0], 3), round(c[1], 3), round(c[2], 3), int(info["driving"]),
                          int(wrong_way(carla_map, info["x"], info["y"], info["z"], info["yaw"]))])

        if events:                                   # scripted passenger
            w = events[0]["when"]
            target = places.get(w.get("reached", ""))
            if ("driven_m" in w and st["driven"] >= w["driven_m"]) or (
                    target and math.hypot(info["x"] - target.x, info["y"] - target.y) <= VISIT_M):
                fire(sess, events.pop(0), t)

        elapsed = t - st["t0"]
        if st["cancel_t"] is not None:
            if info["speed"] < STAND_MPS:
                st["still_since"] = st["still_since"] or t
                if t - st["still_since"] >= CANCEL_SETTLE_S:
                    st["result"] = "stopped"
            else:
                st["still_since"] = None
            if t - st["cancel_t"] > CANCEL_LIMIT_S:
                st["result"] = "not_stopped"
        elif not events and sess.mission.destination is None:
            st["result"] = "arrived"                  # the session cleared the mission
        elif info["driving"] and info["speed"] < 0.2:
            st["still_since"] = st["still_since"] or t
            if t - st["still_since"] > STUCK_S:
                st["result"] = "stuck"
        else:
            st["still_since"] = None
        if st["result"] is None and elapsed > limit:
            st["result"] = "timeout"
        if st["result"] is not None:
            done.set()

    wall0 = time.time()
    # A freshly spawned actor reads as (0, 0, 0) until the world has ticked;
    # planning before that starts the route from the map origin and sends the
    # car turning back. Wait until it is really at its start pose.
    t_wait = time.time()
    while math.hypot(vehicle.get_location().x - s["x"],
                     vehicle.get_location().y - s["y"]) > 2.0:
        if time.time() - t_wait > 30.0:
            raise RuntimeError("ego never reached its spawn pose")
        time.sleep(0.05)
    session.submit(events.pop(0)["intent"])           # the start event
    st["fired"].append({"t": 0.0, "driven_m": 0.0, "ok": True, "message": "start"})
    session.observers.append(observe)
    # Wall-clock safety net: a hung backend must not hang the whole batch.
    done.wait(timeout=max(600.0, limit * 30))
    result = st["result"] or "hung"
    session.observers.clear()
    for sensor in (col, lane):
        sensor.stop()
        sensor.destroy()
    session.close()
    return {"id": sc["id"], "town": sc["town"], "backend": args.backend, "result": result,
            "sim_s": st["log"][-1][0] if st["log"] else 0.0,
            "wall_s": round(time.time() - wall0, 1), "driven_m": round(st["driven"], 1),
            "fired": st["fired"], "collisions": collisions, "lane_invasions": invasions,
            "log_columns": ["t", "x", "y", "yaw", "speed", "steer", "throttle", "brake",
                            "driving", "wrong_way"],
            "log": st["log"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_backend_args(ap)
    ap.add_argument("--scenarios", default=str(SCENARIOS))
    ap.add_argument("--only", default=None, help="run scenarios whose id starts with this")
    ap.add_argument("--out", default=None, help="default: benchmark/runs/execution/<backend>/")
    ap.add_argument("--rerun", action="store_true", help="redo scenarios already logged")
    ap.add_argument("--watch", action="store_true",
                    help="follow the car with the CARLA spectator and draw the route, "
                         "at real-time pace -- for looking at a few scenarios, not for "
                         "a benchmark run")
    args = ap.parse_args()
    vehicle_bp = backend_defaults(args)
    args.sync = True                                  # always: one result per scenario

    with open(args.scenarios, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]
    if args.only:
        scenarios = [s for s in scenarios if s["id"].startswith(args.only)]
    out_dir = Path(args.out) if args.out else RUNS / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)

    kb = LandmarkKnowledgeBase(str(KB_CSV))
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    for n, sc in enumerate(scenarios, start=1):
        path = out_dir / (sc["id"].replace("/", "__") + ".json")
        if path.exists() and not args.rerun:
            print(f"[run] {n}/{len(scenarios)} {sc['id']}: already logged")
            continue
        print(f"[run] {n}/{len(scenarios)} {sc['id']} ...", flush=True)
        try:
            rec = run_scenario(client, args, vehicle_bp, sc, kb)
        except Exception as exc:                      # log it, keep the batch going
            rec = {"id": sc["id"], "result": "error", "error": f"{type(exc).__name__}: {exc}"}
        path.write_text(json.dumps(rec))
        print(f"[run]    -> {rec['result']}  ({rec.get('sim_s', 0):.0f} s sim, "
              f"{rec.get('wall_s', 0):.0f} s wall)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
