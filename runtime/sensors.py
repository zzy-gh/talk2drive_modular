"""
runtime/sensors.py
==================
Spawns the sensor rig a backend asks for and keeps the latest frame of each.

Backends declare sensors in the CARLA-leaderboard format, and this rig hands
data back in the leaderboard's shape — ``{id: (frame, data)}`` — so a TCP or
Bench2Drive agent runs unchanged whether it is ticked by the leaderboard or
by our own session.
"""

from __future__ import annotations

import math
import threading
import weakref

import carla
import numpy as np


class SensorRig:
    def __init__(self, world, vehicle, specs: list[dict]) -> None:
        self.world = world
        self.vehicle = vehicle
        self._lock = threading.Lock()
        self._data: dict[str, tuple[int, object]] = {}
        self._actors: list[carla.Actor] = []
        self._pseudo: list[dict] = []
        for spec in specs or []:
            self._spawn(spec)
        if self._actors:
            print(f"[sensors] {len(self._actors)} sensor(s): "
                  f"{[s['id'] for s in specs]}")

    # ── spawning ────────────────────────────────────────────────────────
    def _spawn(self, spec: dict) -> None:
        stype, sid = spec["type"], spec["id"]
        if stype == "sensor.speedometer":
            self._pseudo.append(spec)
            return

        bl = self.world.get_blueprint_library()
        try:
            bp = bl.find(stype)
        except Exception:
            print(f"[sensors] unknown sensor type {stype!r}, skipped")
            return

        for attr in ("width", "height", "fov", "range", "rotation_frequency",
                     "points_per_second", "upper_fov", "lower_fov", "channels",
                     "sensor_tick"):
            if attr in spec and bp.has_attribute(attr):
                bp.set_attribute(attr, str(spec[attr]))

        tf = carla.Transform(
            carla.Location(x=float(spec.get("x", 0.0)), y=float(spec.get("y", 0.0)),
                           z=float(spec.get("z", 0.0))),
            carla.Rotation(roll=float(spec.get("roll", 0.0)),
                           pitch=float(spec.get("pitch", 0.0)),
                           yaw=float(spec.get("yaw", 0.0))),
        )
        actor = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
        ref = weakref.ref(self)
        actor.listen(lambda data, _id=sid, _t=stype: SensorRig._on_data(ref, _id, _t, data))
        self._actors.append(actor)

    @staticmethod
    def _on_data(ref, sid, stype, data) -> None:
        self = ref()
        if self is None:
            return
        if stype == "sensor.camera.rgb":
            arr = np.frombuffer(data.raw_data, dtype=np.uint8)
            payload = arr.reshape((data.height, data.width, 4))     # BGRA
        elif stype == "sensor.other.imu":
            payload = np.array([data.accelerometer.x, data.accelerometer.y,
                                data.accelerometer.z, data.gyroscope.x,
                                data.gyroscope.y, data.gyroscope.z,
                                data.compass], dtype=np.float64)
        elif stype == "sensor.other.gnss":
            payload = np.array([data.latitude, data.longitude, data.altitude],
                               dtype=np.float64)
        elif stype.startswith("sensor.lidar"):
            payload = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        else:
            payload = data
        with self._lock:
            self._data[sid] = (data.frame, payload)

    # ── read-out ────────────────────────────────────────────────────────
    def read(self) -> dict:
        with self._lock:
            out = dict(self._data)
        for spec in self._pseudo:
            out[spec["id"]] = (0, {"speed": self.speed_mps()})
        return out

    def speed_mps(self) -> float:
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def destroy(self) -> None:
        for actor in self._actors:
            try:
                actor.stop()
                actor.destroy()
            except Exception:
                pass
        self._actors = []
