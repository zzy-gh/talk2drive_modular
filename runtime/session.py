"""
runtime/session.py
==================
Loop shape A: talk2drive owns the driving loop.

The session is the only place that knows about all four parties — vehicle,
sensors, backend, mission — and it keeps two threads strictly separated:

* the **control thread** ticks the backend at a fixed rate and never blocks
  on anything human;
* the **command thread** (usually the CLI) parses utterances and re-plans.
  Disambiguation prompts happen here, so the car keeps driving its previous
  route while the passenger is being asked which hospital they meant.

The plan swap between the two is a single locked assignment.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import carla

from backends.base import DrivingBackend
from core.mission import ApplyResult, MissionPlanner
from core.types import Observation

from .sensors import SensorRig
from .viz import PlanVisualizer


@dataclass
class SessionConfig:
    control_hz: float = 20.0
    spawn_index: int = 0
    vehicle_filter: str = "vehicle.tesla.model3"
    vehicle_color: str = "255,50,0"
    role_name: str = "hero"          # leaderboard agents look the ego up by this
    look_ahead_m: float = 8.0        # plan from ahead of the car, avoids a U-turn
    visualize: bool = True
    beacon: bool = True


class DriveSession:
    def __init__(self, world, backend: DrivingBackend, mission: MissionPlanner,
                 config: SessionConfig | None = None, vehicle=None) -> None:
        self.world = world
        self.backend = backend
        self.mission = mission
        self.config = config or SessionConfig()

        self.vehicle = vehicle or self._spawn_vehicle()
        self.viz = PlanVisualizer(world) if self.config.visualize else None

        self.backend.attach(self.vehicle, world)
        self.rig = SensorRig(world, self.vehicle, self.backend.sensors()) \
            if self.backend.caps.needs_sensors else None

        self._lock = threading.Lock()
        self._driving = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[session] backend={self.backend.name} caps={self.backend.caps}")

    # ─────────────────────────────────────────
    # Vehicle
    # ─────────────────────────────────────────

    def _spawn_vehicle(self):
        bl = self.world.get_blueprint_library()
        try:
            bp = bl.find(self.config.vehicle_filter)
        except Exception:
            bp = bl.filter("vehicle.*")[0]
        if bp.has_attribute("color"):
            bp.set_attribute("color", self.config.vehicle_color)
        if bp.has_attribute("role_name"):
            # Leaderboard agents locate the ego by role_name == 'hero'.
            bp.set_attribute("role_name", self.config.role_name)
        spawns = self.world.get_map().get_spawn_points()
        tf = spawns[self.config.spawn_index % len(spawns)]
        vehicle = self.world.try_spawn_actor(bp, tf) or self.world.spawn_actor(
            bl.filter("vehicle.*")[0], tf)
        print(f"[session] spawned {vehicle.type_id} @ "
              f"({tf.location.x:.1f}, {tf.location.y:.1f})")
        return vehicle

    def ego_xyz(self) -> tuple[float, float, float]:
        loc = self.vehicle.get_location()
        return (loc.x, loc.y, loc.z)

    def ego_ahead(self, meters: float | None = None) -> tuple[float, float, float]:
        m = self.config.look_ahead_m if meters is None else meters
        tf = self.vehicle.get_transform()
        fwd = tf.get_forward_vector()
        return (tf.location.x + fwd.x * m, tf.location.y + fwd.y * m, tf.location.z)

    def speed_mps(self) -> float:
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    # ─────────────────────────────────────────
    # Command thread
    # ─────────────────────────────────────────

    def submit(self, intent: dict) -> ApplyResult:
        """Apply an intent and re-plan. May prompt — never call from the
        control thread."""
        result = self.mission.apply(intent)
        if not result.ok:
            print(f"[session] {result.message}")
            return result
        print(f"[session] {result.message}")

        if result.cancelled:
            self.cancel()
            return result
        if result.needs_replan:
            self.replan()
        return result

    def replan(self) -> bool:
        plan = self.mission.build_plan(self.ego_ahead(), start_fn=self.ego_ahead)
        if plan is None:
            print("[session] no plan produced")
            return False
        with self._lock:
            self.backend.set_plan(plan, self.mission.directive)
            self._driving = True
        if self.viz:
            self.viz.show(plan, urgency=self.mission.directive.urgency)
        print(f"[session] plan rev={plan.revision}: {len(plan)} pts, "
              f"{plan.length():.0f} m -> {plan.goal.label}")
        return True

    def cancel(self) -> None:
        with self._lock:
            self._driving = False
            self.backend.cancel()
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
        if self.viz:
            self.viz.clear()

    def clear(self) -> None:
        self.mission.reset()
        self.cancel()

    # ─────────────────────────────────────────
    # Control thread
    # ─────────────────────────────────────────

    def _loop(self) -> None:
        period = 1.0 / max(1.0, self.config.control_hz)
        while self._running:
            t0 = time.time()
            try:
                self._tick()
            except Exception as exc:
                print(f"[session] tick failed: {exc}")
            if self.config.beacon and self.viz:
                self.viz.beacon(self.vehicle.get_location())
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _tick(self) -> None:
        with self._lock:
            driving = self._driving
        if not driving:
            return

        tf = self.vehicle.get_transform()
        obs = Observation(
            ego_transform=tf,
            speed_mps=self.speed_mps(),
            sensors=self.rig.read() if self.rig else {},
            timestamp=self.world.get_snapshot().timestamp.elapsed_seconds,
        )

        if self._arrived(tf):
            print("[session] destination reached")
            self.clear()
            return

        with self._lock:
            control = self.backend.run_step(obs)
        if control is not None:
            self.vehicle.apply_control(control)

    def _arrived(self, tf) -> bool:
        # A VLA has no notion of mission completion, so the mission layer is
        # authoritative unless the backend explicitly says otherwise.
        if self.backend.caps.reports_done and self.backend.done():
            return True
        return self.mission.is_arrived(tf.location.x, tf.location.y)

    # ─────────────────────────────────────────
    # Teardown
    # ─────────────────────────────────────────

    def close(self) -> None:
        self._running = False
        # No join timeout: tearing the vehicle down while the loop still has an
        # in-flight RPC on it can hard-crash the CARLA client. The loop sleeps
        # at most one period, so this returns almost immediately.
        self._thread.join()
        if self.viz:
            self.viz.clear()
        if self.rig:
            self.rig.destroy()
        self.backend.destroy()
        try:
            self.vehicle.destroy()
        except Exception:
            pass
        print("[session] closed")
