"""
backends/leaderboard.py
=======================
Any CARLA-leaderboard ``AutonomousAgent`` — TCP, Bench2Drive baselines,
InterFuser, your own team_code — driven by talk2drive.

These agents already have the exact contract we want, just spelled
differently: ``set_global_plan(gps_plan, world_plan)`` instead of
``set_plan``, and ``run_step(input_data, timestamp)`` instead of
``run_step(obs)``. So this backend is pure translation.

Note the direction of control. Here *we* own the loop and tick the E2E agent.
The mirror image — an evaluation harness owning the loop with talk2drive
riding inside it — is ``runtime/passenger.py``, not a backend.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from core.types import Directive, GlobalPlan, Observation, get_latlon_ref

from .base import BackendCaps, DrivingBackend, brake_control, print_throttled, register


def _seed_data_provider(world=None, client=None) -> None:
    """Populate CarlaDataProvider so leaderboard agents can be run standalone."""
    try:
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    except ImportError:
        return
    if client is not None:
        CarlaDataProvider.set_client(client)
    if world is not None and CarlaDataProvider.get_world() is not world:
        CarlaDataProvider.set_world(world)


def _lb1_compat() -> None:
    """Let a leaderboard-1.0 agent (TCP) import on CARLA >= 0.9.12.

    leaderboard 1.0's route_manipulation.py imports GlobalRoutePlannerDAO at
    module level, and CARLA removed that class in 0.9.12. The agent only needs
    downsample_route from that module; the DAO is used by
    interpolate_trajectory, which talk2drive never calls (it plans routes
    itself). So stand in a module whose DAO fails loudly if anything does use it.
    """
    name = "agents.navigation.global_route_planner_dao"
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    import types

    class GlobalRoutePlannerDAO:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "GlobalRoutePlannerDAO was removed in CARLA 0.9.12; leaderboard 1.0's "
                "interpolate_trajectory is not supported here (talk2drive plans the route)")

    module = types.ModuleType(name)
    module.GlobalRoutePlannerDAO = GlobalRoutePlannerDAO
    sys.modules[name] = module


@register("leaderboard")
class LeaderboardBackend(DrivingBackend):

    caps = BackendCaps(
        plan_format="gps",
        consumes_language=False,
        control_rate_hz=20.0,
        needs_sensors=True,
        reports_done=False,          # E2E agents drive; the mission layer decides arrival
    )

    def __init__(self, agent_instance=None, module_path: str | None = None,
                 config: str | None = None, host: str = "localhost",
                 port: int = 2000, debug: bool = False,
                 sample_factor: float | None = None, world=None, client=None,
                 pythonpath: tuple[str, ...] = ()) -> None:
        # None on purpose: every AutonomousAgent base class already downsamples
        # inside set_global_plan (factor 50 for leaderboard 1.0 / Bench2Drive,
        # 200 for SimLingo's fork). Pre-downsampling here would compound with
        # that and stretch the spacing the policy was trained on.
        self.sample_factor = sample_factor
        self.lat_ref, self.lon_ref = 42.0, 2.0
        self._active = False
        # Per-step debug lines go to a file, not the terminal: at 20 Hz they
        # would bury the passenger's input prompt. `tail -F` it instead.
        self._debug_log = None
        if debug:
            log_path = Path(__file__).resolve().parents[1] / "logs" / "leaderboard_debug.log"
            log_path.parent.mkdir(exist_ok=True)
            self._debug_log = open(log_path, "a", buffering=1, encoding="utf-8")
            print(f"[leaderboard] per-step debug -> {log_path}  (tail -F it)")
        # Bench2Drive's AutonomousAgent.__init__ calls get_hero(), which reads
        # CarlaDataProvider.get_world(). Outside a leaderboard run nobody has
        # populated it, so the agent crashes before setup() is ever reached.
        _seed_data_provider(world, client)
        self.agent = agent_instance or self._load(module_path, config, host, port,
                                                  debug, pythonpath)

    # ── construction ────────────────────────────────────────────────────
    @staticmethod
    def _repo_roots(agent_file: Path, extra=()) -> list[Path]:
        """
        Import roots an E2E agent needs on sys.path.

        A leaderboard policy is not a standalone file — TCP's agent alone
        reaches for four different roots::

            from leaderboard.autoagents import autonomous_agent  # <repo>/leaderboard
            from TCP.model import TCP                            # <repo>
            from team_code.planner import RoutePlanner           # <repo>/leaderboard
            from srunner...  (via leaderboard)                   # <repo>/scenario_runner

        Adding only the file's own directory leaves every one of those
        unimportable, so derive the repo root and add its known subtrees
        instead of making the caller export PYTHONPATH by hand.
        """
        roots = [Path(e).expanduser().resolve() for e in extra]
        roots.append(agent_file.parent)

        # Walk up collecting ancestors, and find the repo root. `.git` is the
        # only reliable marker: TCP nests leaderboard/team_code inside
        # leaderboard/, so "this directory holds a leaderboard/ and a
        # team_code/" matches the wrong level and loses <repo> itself (needed
        # for `TCP.model`) and <repo>/scenario_runner (needed for `srunner`).
        ancestors = []
        cur = agent_file.parent
        for _ in range(5):
            cur = cur.parent
            if cur == cur.parent:
                break
            ancestors.append(cur)

        repo = next((a for a in ancestors if (a / ".git").exists()), None)
        if repo is None:
            # No checkout metadata (a copied tree): take the OUTERMOST ancestor
            # that still looks like a stack root, not the first one.
            scored = [a for a in ancestors
                      if (a / "scenario_runner").is_dir()
                      or sum((a / m).is_dir() for m in
                             ("leaderboard", "team_code", "srunner")) >= 2]
            repo = scored[-1] if scored else None

        for a in ancestors:
            roots.append(a)
            if repo is not None and a == repo:
                break

        if repo is not None:
            for name in ("leaderboard", "scenario_runner", "scenario_runner_autopilot"):
                if (repo / name).is_dir():
                    roots.append(repo / name)

        seen, out = set(), []
        for r in roots:
            if r.is_dir() and str(r) not in seen:
                seen.add(str(r))
                out.append(r)
        return out

    @staticmethod
    def _load(module_path, config, host, port, debug, pythonpath=()):
        if not module_path:
            raise ValueError("pass either agent_instance= or module_path=")
        path = Path(module_path).expanduser().resolve()
        if path.suffix == ".py":
            roots = LeaderboardBackend._repo_roots(path, pythonpath)
            for r in reversed(roots):
                sys.path.insert(0, str(r))
            print(f"[leaderboard] import roots: {[str(r) for r in roots]}")
            _lb1_compat()
            spec = importlib.util.spec_from_file_location(path.stem, str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)

        entry = getattr(module, "get_entry_point")()
        cls = getattr(module, entry)

        # Two incompatible AutonomousAgent base classes are in the wild:
        #
        #   leaderboard 1.0 (TCP)          __init__(path_to_conf_file) and it
        #                                  calls setup() itself
        #   leaderboard 2.0 / Bench2Drive  __init__(carla_host, carla_port,
        #   / SimLingo                     debug); setup() is called by the host
        #
        # Guessing wrong either crashes the constructor or runs setup() twice —
        # which for TCP means loading the checkpoint twice and tripping its
        # mkdir(exist_ok=False). So pick by signature.
        params = [p for p in inspect.signature(cls.__init__).parameters
                  if p not in ("self", "args", "kwargs")]
        if "carla_host" in params or len(params) >= 3:
            agent = cls(host, port, debug)
            setup_needed = True
        elif params:                       # leaderboard 1.0: setup runs in __init__
            agent = cls(config)
            setup_needed = False
        else:
            agent = cls()
            setup_needed = True

        if setup_needed and config is not None and hasattr(agent, "setup"):
            try:
                agent.setup(config)
            except Exception as exc:
                print(f"[leaderboard] agent.setup({config!r}) failed: {exc}")
        print(f"[leaderboard] loaded {entry} from {module_path} "
              f"(ctor params={params}, setup_called={setup_needed})")
        return agent

    # ── wiring ──────────────────────────────────────────────────────────
    def sensors(self) -> list[dict]:
        return list(self.agent.sensors()) if hasattr(self.agent, "sensors") else []

    def attach(self, vehicle, world) -> None:
        self.lat_ref, self.lon_ref = get_latlon_ref(world)
        _seed_data_provider(world)
        # The ego did not exist yet when the agent was constructed, so whatever
        # get_hero() resolved then was None. Re-resolve now.
        self.agent.hero_actor = vehicle
        if hasattr(self.agent, "get_hero"):
            try:
                self.agent.get_hero()
            except Exception:
                self.agent.hero_actor = vehicle
        if getattr(self.agent, "hero_actor", None) is None:
            self.agent.hero_actor = vehicle

    # ── mission ─────────────────────────────────────────────────────────
    def set_plan(self, plan: GlobalPlan, directive: Directive) -> None:
        gps_plan, world_plan = plan.to_gps_plan(self.lat_ref, self.lon_ref,
                                                self.sample_factor)
        self.agent.set_global_plan(gps_plan, world_plan)
        # Leaderboard agents (TCP, the TransFuser line, InterFuser, ...) build
        # their route planner from _global_plan once, in _init(), on the first
        # run_step after `initialized` is False. A leaderboard run sets the
        # plan once, so nothing ever resets it: without this a re-plan updates
        # _global_plan and the agent keeps following the old route.
        if getattr(self.agent, "initialized", False):
            self.agent.initialized = False
        self._active = True
        kept = len(getattr(self.agent, "_global_plan", []) or [])
        print(f"[leaderboard] global plan set: {len(gps_plan)} points in, "
              f"{kept} kept after the agent's own downsampling")

    # ── control ─────────────────────────────────────────────────────────
    def run_step(self, obs: Observation):
        if not self._active:
            return brake_control()
        try:
            control = self.agent.run_step(obs.sensors, obs.timestamp)
        except Exception as exc:
            print_throttled("lb-run-step", f"[leaderboard] agent.run_step failed: {exc}")
            return brake_control()
        if hasattr(control, "manual_gear_shift"):
            control.manual_gear_shift = False
        if self._debug_log and control is not None:
            self._write_debug(obs, control)
        return control

    def _write_debug(self, obs: Observation, control) -> None:
        """One line per step: speed, the control sent, and the agent's own
        per-step metadata (TCP: pid_metadata -- traj/ctrl branch, both
        branches' outputs, predicted waypoints, desired speed)."""
        import json
        import time

        import numpy as np

        def tidy(v):
            if isinstance(v, dict):
                return {k: tidy(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [tidy(x) for x in v]
            if isinstance(v, (bool, str)) or v is None:
                return v
            try:
                return np.round(np.asarray(v, dtype=float), 3).tolist()
            except (TypeError, ValueError):
                return str(v)

        meta = getattr(self.agent, "pid_metadata", None)
        line = (f"{time.strftime('%H:%M:%S')} step={getattr(self.agent, 'step', '-')} "
                f"speed={obs.speed_mps * 3.6:.1f}km/h steer={control.steer:+.3f} "
                f"throttle={control.throttle:.2f} brake={control.brake:.2f}")
        if meta:
            line += " " + json.dumps(tidy(meta))
        self._debug_log.write(line + "\n")

    def cancel(self) -> None:
        self._active = False

    def destroy(self) -> None:
        self._active = False
        if self._debug_log:
            self._debug_log.close()
        if hasattr(self.agent, "destroy"):
            try:
                self.agent.destroy()
            except Exception as exc:
                print(f"[leaderboard] agent.destroy failed: {exc}")
