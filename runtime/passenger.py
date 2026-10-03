"""
runtime/passenger.py
====================
Loop shape B: somebody else owns the loop, talk2drive rides along.

Under Bench2Drive / the CARLA leaderboard the harness calls
``agent.run_step()`` — talk2drive cannot own a while-loop, and anything slow or
blocking inside ``run_step`` stalls the simulation. So here talk2drive is a
passive service:

* ``submit`` only enqueues text; it returns immediately.
* a worker thread does the slow work — LLM parse, landmark resolution,
  ``trace_route`` — off the simulation thread.
* ``poll`` is non-blocking: it hands back a finished plan if one is ready,
  otherwise None, and the agent keeps driving whatever it already had.

:func:`make_talk2drive_agent` wraps this around an existing leaderboard agent
class (TCP, InterFuser, a Bench2Drive baseline), so an E2E policy gets its
destination from speech with no change to the policy itself.
"""

from __future__ import annotations

import queue
import sys
import threading

from core.mission import MissionPlanner
from core.resolver import AutoInteraction
from core.types import Directive, GlobalPlan


class Talk2DrivePassenger:
    """Non-blocking mission service for hosts that own the control loop."""

    def __init__(self, mission: MissionPlanner, parser=None, town: str | None = None,
                 verbose: bool = True) -> None:
        if mission.interaction.blocking:
            raise ValueError(
                "a blocking Interaction would stall the host's control loop; "
                "use AutoInteraction or DeferredInteraction in passenger mode")
        self.mission = mission
        self.parser = parser
        self.town = town or mission.town
        self.verbose = verbose

        self._inbox: queue.Queue = queue.Queue()
        self._ready: GlobalPlan | None = None
        self._directive = Directive()
        self._cancelled = False
        self._lock = threading.Lock()
        self._ego = (0.0, 0.0, 0.0)
        self._running = True
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    # ── producer side (any thread) ──────────────────────────────────────
    def submit(self, text: str) -> None:
        self._inbox.put(("text", text))

    def submit_intent(self, intent: dict) -> None:
        self._inbox.put(("intent", intent))

    # ── consumer side (the host's control thread) ───────────────────────
    def poll(self, ego_xyz) -> GlobalPlan | None:
        """Returns a newly finished plan, or None. Never blocks."""
        with self._lock:
            self._ego = tuple(ego_xyz)
            plan, self._ready = self._ready, None
        return plan

    def take_cancel(self) -> bool:
        with self._lock:
            flag, self._cancelled = self._cancelled, False
        return flag

    @property
    def directive(self) -> Directive:
        with self._lock:
            return self._directive

    # ── worker ──────────────────────────────────────────────────────────
    def _work(self) -> None:
        while self._running:
            try:
                kind, payload = self._inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle(kind, payload)
            except Exception as exc:
                print(f"[passenger] command failed: {exc}", flush=True)

    def _handle(self, kind: str, payload) -> None:
        if kind == "text":
            if self.parser is None:
                print("[passenger] no parser configured", flush=True)
                return
            intent = self.parser.parse(payload, town=self.town,
                                       route=self.mission.route_summary())
            if not intent:
                print("[passenger] could not parse command", flush=True)
                return
            intent["_utterance"] = payload
        else:
            intent = payload

        result = self.mission.apply(intent)
        if self.verbose:
            print(f"[passenger] {result.message}", flush=True)
        if not result.ok:
            return
        if result.cancelled:
            with self._lock:
                self._cancelled = True
                self._ready = None
            return
        if not result.needs_replan:
            return

        with self._lock:
            ego = self._ego
        plan = self.mission.build_plan(ego)
        if plan is None:
            print("[passenger] no plan produced", flush=True)
            return
        with self._lock:
            self._ready = plan
            self._directive = self.mission.directive
        if self.verbose:
            print(f"[passenger] plan rev={plan.revision}: {len(plan)} pts "
                  f"-> {plan.goal.label}", flush=True)

    def close(self) -> None:
        self._running = False


def stdin_pump(passenger: Talk2DrivePassenger) -> threading.Thread:
    """Feed typed commands into a passenger without touching the sim thread."""

    def run():
        for line in sys.stdin:
            line = line.strip()
            if line:
                passenger.submit(line)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────
# Wrapping an existing leaderboard agent
# ─────────────────────────────────────────────

def make_talk2drive_agent(base_cls, passenger_factory, command_pump=stdin_pump):
    """
    Build a leaderboard agent class that takes its destination from speech.

    ``base_cls`` is any ``AutonomousAgent`` subclass (TCP, InterFuser, ...).
    ``passenger_factory(agent) -> Talk2DrivePassenger`` is called from
    ``setup``, once CARLA is up and the map is known.

    Usage in your team_code::

        from TCP.tcp_agent import TCPAgent
        Talk2DriveTCP = make_talk2drive_agent(TCPAgent, my_factory)
        def get_entry_point(): return "Talk2DriveTCP"
    """

    class Talk2DriveAgent(base_cls):

        def setup(self, path_to_conf_file):
            super().setup(path_to_conf_file)
            self.passenger = passenger_factory(self)
            self._pump = command_pump(self.passenger) if command_pump else None
            self._t2d_plan = None

        def run_step(self, input_data, timestamp):
            ego = self._t2d_ego_xyz()
            plan = self.passenger.poll(ego)
            if plan is not None:
                lat_ref, lon_ref = self._t2d_geo_ref()
                gps_plan, world_plan = plan.to_gps_plan(lat_ref, lon_ref)
                self.set_global_plan(gps_plan, world_plan)
                self._t2d_plan = plan
                self._t2d_on_new_plan(plan)
            if self.passenger.take_cancel():
                self._t2d_plan = None
            return super().run_step(input_data, timestamp)

        # ── hooks a concrete agent may override ─────────────────────────
        def _t2d_ego_xyz(self):
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            hero = getattr(self, "_hero_actor", None) or CarlaDataProvider.get_hero_actor()
            loc = hero.get_location()
            return (loc.x, loc.y, loc.z)

        def _t2d_geo_ref(self):
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            from core.types import get_latlon_ref
            if not hasattr(self, "_t2d_geo"):
                self._t2d_geo = get_latlon_ref(CarlaDataProvider.get_world())
            return self._t2d_geo

        def _t2d_on_new_plan(self, plan):
            print(f"[talk2drive] new plan rev={plan.revision} -> {plan.goal.label}",
                  flush=True)

        def destroy(self):
            if getattr(self, "passenger", None):
                self.passenger.close()
            super().destroy()

    Talk2DriveAgent.__name__ = f"Talk2Drive{base_cls.__name__}"
    return Talk2DriveAgent
