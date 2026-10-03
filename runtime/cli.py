"""
runtime/cli.py
==============
The interactive demo — same experience as the old ``drive_demo.py``, but the
controller is now a ``--backend`` flag.

    python -m runtime.cli --backend basic_agent --town 1
    python -m runtime.cli --backend simlingo --simlingo-ckpt /path/to/ckpt
    python -m runtime.cli --backend leaderboard --agent-module .../tcp_agent.py \
                          --agent-config .../tcp_config.py

One backend per run -- an evaluation measures the stack you name, nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla

import backends
from core.landmark_kb import LandmarkKnowledgeBase
from core import paths
from core.llm_bridge import LLMBridge
from core.llm_inference import GeminiCommandParser, normalize_town
from core.mission import MissionPlanner
from core.resolver import AutoInteraction, CliInteraction
from core.route_provider import CarlaRouteProvider
from runtime.session import DriveSession, SessionConfig



HELP = """
Commands:
  <anything else>   a spoken instruction, e.g. "take me to the hospital, hurry"
  clear             cancel the route and reset every landmark to grey
  view              toggle the camera between the ego and the whole town
  plan              print the current mission state
  quit / exit       leave

Landmarks: grey = unused, red = destination, orange = stop,
           green = reached, purple = cancelled, yellow [n] = pick one
"""


def _camera_mount(spec: str | None) -> dict | None:
    """'x,y,z[,pitch]' -> the backend's camera overrides."""
    if not spec:
        return None
    try:
        vals = [float(v) for v in spec.split(",")]
    except ValueError:
        vals = []
    if len(vals) not in (3, 4):
        raise SystemExit(f"--simlingo-camera wants x,y,z[,pitch], got {spec!r}")
    return dict(zip(("x", "y", "z", "pitch"), vals))


def build_backend(args, world, client=None):
    def make(name):
        if name == "basic_agent":
            return backends.build("basic_agent", target_speed=args.target_speed)
        if name == "simlingo":
            ckpt = args.simlingo_ckpt or paths.SIMLINGO_CKPT
            if not ckpt:
                raise SystemExit("--backend simlingo needs --simlingo-ckpt "
                                 "or [simlingo] checkpoint in config.yaml")
            return backends.build("simlingo",
                                  checkpoint_path=str(ckpt),
                                  repo_path=str(args.simlingo_repo or paths.SIMLINGO_REPO),
                                  python_bin=str(args.simlingo_python or paths.SIMLINGO_PYTHON),
                                  hydra_config_path=args.simlingo_config,
                                  inference_interval=args.simlingo_interval
                                  or (1 if args.sync else 15),
                                  camera=_camera_mount(args.simlingo_camera),
                                  debug=args.debug)
        if name == "leaderboard":
            module = args.agent_module or paths.AGENT_MODULE
            if not module:
                raise SystemExit("--backend leaderboard needs --agent-module "
                                 "or [leaderboard] agent_module in config.yaml")
            cfg = args.agent_config or paths.AGENT_CONFIG
            return backends.build("leaderboard", module_path=str(module),
                                  config=str(cfg) if cfg else None, host=args.host,
                                  port=args.port, debug=args.debug,
                                  world=world, client=client,
                                  pythonpath=tuple(args.agent_pythonpath))
        raise SystemExit(f"unknown backend {name!r}; available: {backends.available()}")

    return make(args.backend)


def main() -> None:
    p = argparse.ArgumentParser(description="talk2drive — modular")
    p.add_argument("--backend", default="basic_agent",
                   help="basic_agent | simlingo | leaderboard")
    p.add_argument("--town", default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--target-speed", type=float, default=30.0)
    p.add_argument("--spawn", type=int, default=0)
    p.add_argument("--control-hz", type=float, default=20.0)
    p.add_argument("--csv", default=None, help="override [paths] kb_csv")
    p.add_argument("--llm-python", default=None,
                   help="run Gemini parsing in its own subprocess, using this "
                        "interpreter, instead of in-process (see [paths] "
                        "llm_python in config.yaml)")
    p.add_argument("--paths", action="store_true",
                   help="print the paths from config.yaml, then exit")
    p.add_argument("--auto", action="store_true",
                   help="never prompt: nearest landmark, append enroute stops")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--sync", action=argparse.BooleanOptionalAction, default=None,
                   help="synchronous 1/control-hz simulation steps (default: on "
                        "for simlingo, which runs that way officially)")
    p.add_argument("--world-viz", action="store_true",
                   help="draw route and landmarks with points and lines even for a "
                        "camera-driven backend (its camera sees them too); by "
                        "default such a backend gets text-only drawing")
    p.add_argument("--bev", action="store_true",
                   help="open a bird's-eye-view window: roads, route, landmarks, "
                        "ego -- drawn outside CARLA, so no camera ever sees it")
    p.add_argument("--fpv", action="store_true",
                   help="open a window with the camera frame the driving model "
                        "last saw (simlingo backend)")
    p.add_argument("--vehicle", default=None,
                   help="ego blueprint (default: the Lincoln MKZ SimLingo was "
                        "trained with for --backend simlingo, else a Tesla Model 3)")
    p.add_argument("--no-reload", action="store_true",
                   help="use the running map instead of reloading it")
    p.add_argument("--debug", action="store_true")
    # SimLingo
    p.add_argument("--simlingo-ckpt", default=None)
    p.add_argument("--simlingo-repo", default=None)
    p.add_argument("--simlingo-python", default=None)
    p.add_argument("--simlingo-config", default=None)
    p.add_argument("--simlingo-interval", type=int, default=None,
                   help="control ticks per inference (default: 1 in sync mode "
                        "as SimLingo runs officially, else 15)")
    p.add_argument("--simlingo-camera", default=None, metavar="X,Y,Z[,PITCH]",
                   help="front camera mount on the ego in metres (x forward, "
                        "z up) and degrees; default -0.5,0,2.0,0, 1 m ahead of "
                        "SimLingo's training mount -1.5,0,2.0, which its model "
                        "still assumes")
    # Leaderboard / E2E
    p.add_argument("--agent-module", default=None)
    p.add_argument("--agent-config", default=None)
    p.add_argument("--agent-pythonpath", action="append", default=[],
                   help="extra import root for the agent's repo (repeatable); "
                        "usually derived automatically from --agent-module")
    args = p.parse_args()
    if args.sync is None:
        args.sync = args.backend == "simlingo"

    if args.paths:
        print(paths.describe())
        return

    csv_path = args.csv or paths.KB_CSV
    print(f"[cli] knowledge base: {csv_path}")
    kb = LandmarkKnowledgeBase(str(csv_path))    # mission planning needs this either way

    llm_python = args.llm_python or paths.LLM_PYTHON
    if llm_python:
        parser = LLMBridge(str(llm_python), csv_path=str(csv_path))
    else:
        parser = GeminiCommandParser(kb)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    town = normalize_town(args.town) if args.town else \
        normalize_town(client.get_world().get_map().name.split("/")[-1])
    if args.no_reload:
        world = client.get_world()
        print(f"[cli] using running map: {town}")
    else:
        print(f"[cli] loading {town} (clears previous drawings) ...")
        world = client.load_world(town)

    routes = CarlaRouteProvider(world.get_map(), sampling_resolution=1.0)
    mission = MissionPlanner(kb, routes, interaction=AutoInteraction(), town=town)

    backend = build_backend(args, world, client)
    # SimLingo's camera sits behind the roof, so the ego's own roof is in every
    # frame; a car it never saw in training (the Model 3's black glass roof)
    # can read as an obstacle.
    vehicle = args.vehicle or ("vehicle.lincoln.mkz_2020" if args.backend == "simlingo"
                               else SessionConfig.vehicle_filter)
    # Debug points/lines are rendered into the world, so a camera sensor sees
    # them as objects on the road; debug strings are server-side only. For a
    # camera-driven backend draw text only, unless shapes are asked for.
    text_only = backend.caps.needs_sensors and not args.world_viz
    if text_only and not args.no_viz:
        print("[cli] camera-driven backend: only the route and EGO are drawn in CARLA, "
              "as text (kept out of the camera image); landmarks are in --bev")
    session = DriveSession(world, backend, mission,
                           SessionConfig(control_hz=args.control_hz,
                                         vehicle_filter=vehicle,
                                         spawn_index=args.spawn,
                                         visualize=not args.no_viz,
                                         viz_text_only=text_only,
                                         camera=not args.no_viz,
                                         sync=args.sync))

    # Interactive disambiguation, with the numbered markers drawn in CARLA.
    if not args.auto:
        mission.interaction = CliInteraction(
            preview=session.preview_places,
            clear_preview=session.clear_previews)

    panels = []
    if args.bev:
        from runtime.bev import BevPanel
        panels.append(BevPanel(session))
    if args.fpv:
        from runtime.fpv import FpvPanel
        try:
            panels.append(FpvPanel(backend))
        except ValueError as exc:
            print(f"[cli] --fpv: {exc}")
    gui = None
    if panels:
        from runtime.gui import GuiLoop
        gui = GuiLoop(panels)

    print(HELP)
    try:
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            low = text.lower()
            if low in {"quit", "exit"}:
                break
            if low == "help":
                print(HELP)
                continue
            if low == "clear":
                session.clear()
                print("-> cleared\n")
                continue
            if low == "view":
                if session.camera:
                    session.camera.toggle()
                    print(f"-> camera: {session.camera.mode}\n")
                continue
            if low == "plan":
                print(json.dumps({"town": mission.town,
                                  "destination": mission.destination,
                                  "stops": [s.label for s in mission.stops],
                                  "urgency": mission.directive.urgency,
                                  "revision": mission.plan.revision if mission.plan else None},
                                 indent=2))
                continue

            intent = parser.parse(text, town=town, route=mission.route_summary())
            if not intent:
                print("-> could not parse command\n")
                continue
            intent["_utterance"] = text
            print(f"Intent: {json.dumps(intent, ensure_ascii=False, indent=2)}")
            session.submit(intent)
            print()
    finally:
        if gui:
            gui.close()
        session.close()
        if isinstance(parser, LLMBridge):
            parser.close()


if __name__ == "__main__":
    main()
