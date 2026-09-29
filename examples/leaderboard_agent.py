"""
examples/leaderboard_agent.py
=============================
Drop-in team_code agent: run *your existing E2E policy* under Bench2Drive /
the CARLA leaderboard, but let a passenger choose the destination by speech.

The policy is untouched. talk2drive rides inside the agent (loop shape B):
commands arrive on stdin, are parsed and planned on a worker thread, and the
finished route is handed to the policy through the ``set_global_plan`` it
already implements.

Usage — point the leaderboard at this file::

    --agent examples/leaderboard_agent.py --agent-config <your policy config>

and set the two environment variables below.

    T2D_BASE_AGENT=/path/to/TCP/tcp_agent.py     # the policy to wrap
    (paths come from config.yaml)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.landmark_kb import LandmarkKnowledgeBase
from core.paths import KB_CSV
from core.llm_inference import GeminiCommandParser, normalize_town
from core.mission import MissionPlanner
from core.resolver import AutoInteraction
from core.route_provider import CarlaRouteProvider
from runtime.passenger import Talk2DrivePassenger, make_talk2drive_agent

BASE_AGENT = os.environ.get("T2D_BASE_AGENT")



def _load_base_agent_class():
    if not BASE_AGENT:
        raise RuntimeError("set T2D_BASE_AGENT to the policy agent .py to wrap")
    path = Path(BASE_AGENT).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return getattr(module, getattr(module, "get_entry_point")())


def _make_passenger(agent) -> Talk2DrivePassenger:
    """Called from setup(), once CARLA is up and the map is known."""
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    world = CarlaDataProvider.get_world()
    town = normalize_town(world.get_map().name.split("/")[-1])
    kb = LandmarkKnowledgeBase(str(KB_CSV))

    mission = MissionPlanner(
        kb,
        CarlaRouteProvider(world.get_map(), sampling_resolution=1.0),
        # AutoInteraction, not CliInteraction: a prompt here would stall the
        # simulation tick. Ambiguity resolves to the nearest landmark.
        interaction=AutoInteraction(),
        town=town,
    )
    return Talk2DrivePassenger(mission, parser=GeminiCommandParser(kb), town=town)


Talk2DriveAgent = make_talk2drive_agent(_load_base_agent_class(), _make_passenger)


def get_entry_point():
    return "Talk2DriveAgent"
