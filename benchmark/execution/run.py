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

``--language`` runs end to end: each event is an utterance
(data/scenarios_language.jsonl) parsed by the LLM, grounded (the scripted
passenger answers clarifications with the landmark ID), planned and driven.
The world is paused while this runs. Each event logs the parse and the
resulting mission state for evaluate.py.

    python3 benchmark/execution/run.py --backend basic_agent
    python3 benchmark/execution/run.py --backend basic_agent --language --llm-model gemini-3.8-flash
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
from benchmark.execution.utterances import OUT as LANGUAGE_SCENARIOS, ById  # noqa: E402
from core.landmark_kb import LandmarkKnowledgeBase                   # noqa: E402
from core.mission import ApplyResult, MissionPlanner                 # noqa: E402
from core.paths import KB_CSV                                        # noqa: E402
from core.route_provider import CarlaRouteProvider                   # noqa: E402
from runtime.cli import add_backend_args, backend_defaults, build_backend  # noqa: E402
from runtime.session import DriveSession, SessionConfig              # noqa: E402

RUNS = ROOT / "benchmark" / "runs" / "execution"
STAND_MPS = 0.5          # "stopped"
CANCEL_SETTLE_S = 2.0    # after a cancel, stopped this long = done
CANCEL_LIMIT_S = 20.0    # ... or give up this long after the cancel
STUCK_S = 60.0           # driving but not moving this long = stuck
SETTLE_STEPS = 40        # steps at rest before the start


def wrong_way(carla_map, x, y, z, yaw) -> bool:
    """Facing against the lane it is on (outside junctions)."""
    wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=False,
                                lane_type=carla.LaneType.Driving)
    if wp is None or wp.is_junction:
        return False
    return abs((wp.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0) > 90.0


def mission_state(m, result) -> dict:
    """Mission state after an event, in the shape of ``expect_state``."""
    if result.cancelled:
        return {"cancelled": True}
    plan = m.plan
    if plan is None:
        return {"cancelled": False, "goal": None, "stops": [], "urgency": m.directive.urgency}
    return {"cancelled": False, "goal": f"{plan.goal.label}#{plan.goal.index}",
            "stops": [f"{p.label}#{p.index}" for p in plan.stops],
            "urgency": m.directive.urgency}


class ApiFailure(Exception):
    pass


def make_llm(args, kb):
    """The LLM parser, in-process or via --llm-python; failures raise."""
    import os
    if args.llm_python:
        from core.llm_bridge import LLMBridge
        os.environ["GEMINI_MODEL"] = args.llm_model       # the worker reads it
        return LLMBridge(args.llm_python, csv_path=str(KB_CSV),
                         request_timeout_s=120.0, raise_errors=True)
    from benchmark.language.models import make_parser
    return make_parser(args.llm_provider, args.llm_model, kb)


def parse(llm, text: str, town: str, route: dict, retries: int = 8):
    """The intent, retrying a provider error; ApiFailure if it never answers."""
    from benchmark.language.run import retry_delay
    for attempt in range(retries):
        try:
            return llm.parse(text, town=town, route=route)
        except Exception as exc:                      # 429 / 5xx / network
            print(f"[run] LLM request failed ({exc}); retrying", flush=True)
            time.sleep(retry_delay(exc, attempt))
    raise ApiFailure(f"no answer after {retries} attempts")


def spawn_traffic(client, world, npcs: list, tm_port: int, seed: int):
    """Spawn the scenario's NPCs (npc.py) under a synchronous, seeded Traffic Manager."""
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    tm.set_hybrid_physics_mode(False)
    lib, cmap, actors = world.get_blueprint_library(), world.get_map(), []
    for n in npcs:
        bp = lib.find(n["blueprint"])
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "npc")
        if bp.has_attribute("color"):
            bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
        v = world.try_spawn_actor(bp, carla.Transform(
            carla.Location(x=n["x"], y=n["y"], z=n["z"] + 0.3), carla.Rotation(yaw=n["yaw"])))
        if v is None:
            continue
        v.set_autopilot(True, tm_port)
        if n["role"] == "lead":
            tm.vehicle_percentage_speed_difference(v, n["slower_pct"])
            tm.auto_lane_change(v, False)
            path = [cmap.get_waypoint(carla.Location(x=x, y=y)).transform.location
                    for x, y in n["path"]]
            if path:
                tm.set_path(v, path)
        actors.append(v)
    return tm, actors


def run_scenario(client, args, vehicle_bp: str, sc: dict, kb, llm=None, npcs=None) -> dict:
    world = client.load_world(sc["town"])
    carla_map = world.get_map()
    routes = CarlaRouteProvider(carla_map, sampling_resolution=1.0)
    places = place_table(kb, routes, sc["town"])
    inter = ById(sc["answer_ids"]) if llm else Scripted()
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
    # Handbrake until the backend's first control, so the car does not roll on slopes.
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

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

    def fire(sess, ev, t) -> ApplyResult:
        inter.at = (ev.get("answer") or {}).get("insert_at")
        rec = {"t": round(t - st["t0"], 2) if st["t0"] is not None else 0.0,
               "driven_m": round(st["driven"], 1)}
        if llm is None:
            result = sess.submit(ev["intent"])
        else:
            # Parse against the current route, then ground and plan.
            asked = len(inter.asked)
            t_say = time.time()
            intent = parse(llm, ev["utterance"], sc["town"], sess.mission.route_summary())
            parse_s = time.time() - t_say
            t_apply = time.time()
            if intent:
                result = sess.submit({**intent, "_utterance": ev["utterance"]})
            else:
                result = ApplyResult(ok=False, message="not parsed")
            state = mission_state(sess.mission, result)
            rec.update({"utterance": ev["utterance"], "intent": intent,
                        "raw": getattr(llm, "last_raw", None),
                        "parse_s": round(parse_s, 3),
                        "apply_s": round(time.time() - t_apply, 3),
                        "asked": inter.asked[asked:], "state": state,
                        "expect_state": ev["expect_state"],
                        "event_ok": state == ev["expect_state"]})
        rec.update({"ok": result.ok, "message": result.message})
        st["fired"].append(rec)
        if result.cancelled:
            st["cancel_t"] = t
        return result

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
                try:
                    fire(sess, events.pop(0), t)
                except ApiFailure as exc:
                    print(f"[run] {exc}", flush=True)
                    st["result"] = "api_error"
                    done.set()
                    return

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
    # Start on one control step after the ego settles: reset lights, spawn NPCs, first request.
    start_ev = events.pop(0)
    started, box = threading.Event(), {"tm": None, "traffic": []}

    def starter(sess, info) -> None:
        if started.is_set() or math.hypot(info["x"] - s["x"], info["y"] - s["y"]) > 2.0:
            return
        box["settle"] = box.get("settle", 0) + 1
        if box["settle"] < SETTLE_STEPS or info["speed"] > 0.1:
            return
        sess.world.reset_all_traffic_lights()
        try:
            if npcs is not None:
                box["tm"], box["traffic"] = spawn_traffic(
                    client, world, npcs, args.tm_port or args.port + 6000, args.npc_seed)
            fire(sess, start_ev, 0.0)
            if llm is not None and sess.mission.plan is None:
                st["result"] = "no_plan"              # parse or grounding failed
        except ApiFailure as exc:
            st["result"] = "api_error"
            print(f"[run] {exc}", flush=True)
        except Exception as exc:                      # surfaced on the main thread
            box["error"] = exc
        sess.observers.remove(starter)
        if st["result"] is None and "error" not in box:
            sess.observers.append(observe)
        started.set()

    session.observers.append(starter)
    if not started.wait(timeout=60.0):
        session.observers.clear()
        session.close()
        raise RuntimeError("ego never reached its spawn pose")
    if "error" in box:
        session.observers.clear()
        session.close()
        raise box["error"]
    if st["result"] is None:
        # Wall-clock limit for a hung backend.
        done.wait(timeout=max(600.0, limit * 30))
    tm, traffic = box["tm"], box["traffic"]
    result = st["result"] or "hung"
    session.observers.clear()
    for sensor in (col, lane):
        sensor.stop()
        sensor.destroy()
    if traffic:
        client.apply_batch([carla.command.DestroyActor(a) for a in traffic])
    if tm is not None:
        tm.set_synchronous_mode(False)
    session.close()
    return {"id": sc["id"], "town": sc["town"], "backend": args.backend, "result": result,
            "llm": args.llm_model if llm else None, "vehicle": vehicle_bp,
            "npc": len(traffic) if npcs is not None else None,
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
    ap.add_argument("--scenarios", default=None,
                    help="default: data/scenarios.jsonl, or with --language "
                         "data/scenarios_language.jsonl")
    ap.add_argument("--language", action="store_true",
                    help="end to end: each event is a sentence the LLM parses")
    ap.add_argument("--npc", action="store_true",
                    help="add the scenario's traffic (data/npc_layout.jsonl, from npc.py)")
    ap.add_argument("--npc-seed", type=int, default=0, help="Traffic Manager seed")
    ap.add_argument("--tm-port", type=int, default=None, help="default: CARLA port + 6000")
    ap.add_argument("--llm-provider", default="gemini")
    ap.add_argument("--llm-model", default="gemini-3.8-flash")
    ap.add_argument("--llm-python", default=None,
                    help="parse in this python (another env), as the CLI does -- "
                         "for a backend whose env has no LLM SDK")
    ap.add_argument("--only", default=None, help="run scenarios whose id starts with this")
    ap.add_argument("--out", default=None,
                    help="default: benchmark/runs/execution/<backend>/, or with "
                         "--language <backend>__e2e__<llm model>/")
    ap.add_argument("--rerun", action="store_true", help="redo scenarios already logged")
    ap.add_argument("--watch", action="store_true",
                    help="follow the car with the CARLA spectator and draw the route, "
                         "at real-time pace -- for looking at a few scenarios, not for "
                         "a benchmark run")
    args = ap.parse_args()
    vehicle_bp = backend_defaults(args)
    args.sync = True                                  # always: one result per scenario

    path = args.scenarios or (LANGUAGE_SCENARIOS if args.language else SCENARIOS)
    with open(path, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]
    if args.only:
        scenarios = [s for s in scenarios if s["id"].startswith(args.only)]
    name = f"{args.backend}__e2e__{args.llm_model}" if args.language else args.backend
    name += "__npc" if args.npc else ""
    out_dir = Path(args.out) if args.out else RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    kb = LandmarkKnowledgeBase(str(KB_CSV))
    llm = make_llm(args, kb) if args.language else None
    layouts = {}
    if args.npc:
        from benchmark.execution.npc import OUT as NPC_LAYOUT
        with open(NPC_LAYOUT, encoding="utf-8") as fh:
            layouts = {r["id"]: r["npcs"] for r in map(json.loads, fh) if r}
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    for n, sc in enumerate(scenarios, start=1):
        path = out_dir / (sc["id"].replace("/", "__") + ".json")
        logged = json.loads(path.read_text()).get("result") if path.exists() else None
        # API failures and simulator crashes are retried.
        if logged not in (None, "api_error", "error") and not args.rerun:
            print(f"[run] {n}/{len(scenarios)} {sc['id']}: already logged")
            continue
        print(f"[run] {n}/{len(scenarios)} {sc['id']} ...", flush=True)
        try:
            rec = run_scenario(client, args, vehicle_bp, sc, kb, llm,
                               layouts.get(sc["id"], []) if args.npc else None)
        except Exception as exc:                      # log it, keep the batch going
            rec = {"id": sc["id"], "result": "error", "error": f"{type(exc).__name__}: {exc}"}
        path.write_text(json.dumps(rec))
        if "waiting for the simulator" in rec.get("error", ""):
            # Simulator is gone; stop the batch.
            print("[run] the simulator stopped answering; stopping this batch", flush=True)
            return 3
        print(f"[run]    -> {rec['result']}  ({rec.get('sim_s', 0):.0f} s sim, "
              f"{rec.get('wall_s', 0):.0f} s wall)", flush=True)
    if llm is not None and hasattr(llm, "close"):
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
