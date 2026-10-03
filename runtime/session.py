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

from backends.base import DrivingBackend, print_throttled
from core.mission import ApplyResult, MissionPlanner
from core.types import Observation

from .camera import SpectatorCamera
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
    viz_text_only: bool = False      # draw_string only: seen in CARLA, not by cameras
    camera: bool = True              # drive the spectator (ego BEV <-> town BEV)
    # Synchronous mode: the simulator advances one fixed 1/control_hz step per
    # loop and waits for the backend -- how the leaderboard runs, and the only
    # timing a slow VLA like SimLingo sees as it was trained (20 Hz, an
    # inference every frame). Restored to asynchronous on close().
    sync: bool = False
    ego_view_height_m: float = 50.0
    landmark_redraw_s: float = 0.5


class DriveSession:
    def __init__(self, world, backend: DrivingBackend, mission: MissionPlanner,
                 config: SessionConfig | None = None, vehicle=None) -> None:
        self.world = world
        self.backend = backend
        self.mission = mission
        self.config = config or SessionConfig()

        self._orig_settings = None
        if self.config.sync:
            self._orig_settings = world.get_settings()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / max(1.0, self.config.control_hz)
            world.apply_settings(settings)
            print(f"[session] synchronous mode, {settings.fixed_delta_seconds:.3f} s/step")

        self.vehicle = vehicle or self._spawn_vehicle()
        # Always track plan/landmark state (the BEV window reads it); draw it
        # into the CARLA world only when ``visualize`` is on.
        self.viz = PlanVisualizer(world, mission.all_places(),
                                  draw_world=self.config.visualize,
                                  text_only=self.config.viz_text_only)
        self.camera = SpectatorCamera(world, self.vehicle,
                                      ego_height=self.config.ego_view_height_m) \
            if self.config.camera else None

        self.backend.attach(self.vehicle, world)
        self.rig = SensorRig(world, self.vehicle, self.backend.sensors()) \
            if self.backend.caps.needs_sensors else None

        self._lock = threading.Lock()
        self._driving = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Landmark labels are redrawn on their own wall-clock thread: debug
        # strings expire in wall-clock time, so tying them to the control
        # loop (slow in sync mode) made them blink.
        self._viz_thread = None
        if self.viz.draw_world:
            self._viz_thread = threading.Thread(target=self._viz_loop, daemon=True)
            self._viz_thread.start()
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

    def plan_start(self) -> tuple[float, float, float]:
        """Where a re-plan starts: on the lane the ego is driving along, even
        if it has swerved over the centre line, ``look_ahead_m`` down that lane."""
        snap = getattr(self.mission.routes, "snap_heading", None)
        if snap is None:
            return self.ego_ahead()
        tf = self.vehicle.get_transform()
        return snap(tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw,
                    ahead_m=self.config.look_ahead_m)

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
        plan = self.mission.build_plan(self.plan_start(), start_fn=self.plan_start)
        if plan is None:
            print("[session] no plan produced")
            return False
        with self._lock:
            self.backend.set_plan(plan, self.mission.directive)
            self._driving = True
        if self.viz:
            self.viz.show(plan, urgency=self.mission.directive.urgency)
        if self.camera:
            self.camera.show_town()
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
            self.viz.board.cancel_active()
        if self.camera:
            self.camera.follow_ego()

    def clear(self) -> None:
        """Cancel, and also forget which landmarks were visited/cancelled."""
        self.mission.reset()
        self.cancel()
        if self.viz:
            self.viz.board.reset()

    # Disambiguation hooks for CliInteraction: the candidates can be anywhere
    # in town, so pull the camera back while the passenger picks one.
    def preview_places(self, places) -> None:
        if self.viz:
            self.viz.preview_places(places)
        if self.camera:
            self.camera.show_town()

    def clear_previews(self) -> None:
        if self.viz:
            self.viz.clear_previews()

    # ─────────────────────────────────────────
    # Control thread
    # ─────────────────────────────────────────

    def _loop(self) -> None:
        period = 1.0 / max(1.0, self.config.control_hz)
        while self._running:
            t0 = time.time()
            if self.config.sync:
                try:
                    frame = self.world.tick()
                    if self.rig and not self.rig.wait_for(frame):
                        print_throttled("late", f"[session] sensors late for frame {frame}")
                except Exception as exc:
                    print_throttled("tick", f"[session] tick failed: {exc}")
            try:
                self._tick()
            except Exception as exc:
                print_throttled("tick", f"[session] tick failed: {exc}")
            try:
                if self.camera:
                    self.camera.update()
            except Exception as exc:
                print_throttled("camera", f"[session] camera failed: {exc}")
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _viz_loop(self) -> None:
        """Beacon every 0.1 s, landmarks every ``landmark_redraw_s``, on the
        wall clock the debug strings expire by."""
        redraw = self.config.landmark_redraw_s
        beacon_s = 0.1
        next_redraw = 0.0
        while self._running:
            t0 = time.time()
            try:
                if self.config.beacon:
                    self.viz.beacon(self.vehicle.get_location(), life_time=beacon_s * 1.5)
                if t0 >= next_redraw:
                    # Live three periods, so a late redraw never leaves a gap.
                    self.viz.redraw(life_time=redraw * 3.0)
                    next_redraw = t0 + redraw
            except Exception as exc:
                print_throttled("viz", f"[session] viz failed: {exc}")
            time.sleep(max(0.0, beacon_s - (time.time() - t0)))

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

        stop = self.mission.reached_stop(tf.location.x, tf.location.y)
        if stop is not None:
            print(f"[session] stop reached: {stop.label}")
            if self.viz:
                self.viz.board.mark_visited(stop)

        if self._arrived(tf):
            print("[session] destination reached")
            if self.viz and self.mission.plan:
                self.viz.board.mark_visited(self.mission.plan.goal)
            self.mission.reset()
            self.cancel()
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
        if self._viz_thread:
            self._viz_thread.join()
        if self._orig_settings is not None:
            # Left in sync mode, the server would freeze waiting for a tick.
            try:
                self.world.apply_settings(self._orig_settings)
            except Exception as exc:
                print(f"[session] could not restore async mode: {exc}")
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
