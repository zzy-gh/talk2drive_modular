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


def blueprint_attributes(spec: dict) -> dict:
    """Blueprint attributes for a leaderboard sensor spec.

    A spec says ``width``/``height``; the CARLA blueprint calls them
    ``image_size_x``/``image_size_y``. Mirrors
    leaderboard/autoagents/agent_wrapper.py::setup_sensors exactly, fixed lens
    and noise settings included -- they are part of what policies were
    trained on. (Setting ``width`` directly is silently a no-op, which left
    every camera at CARLA's default 800x600, FOV 90.)
    """
    stype = spec["type"]
    attrs: dict = {}
    if stype.startswith("sensor.camera"):
        attrs = {"image_size_x": spec["width"], "image_size_y": spec["height"],
                 "fov": spec["fov"],
                 "lens_circle_multiplier": 3.0, "lens_circle_falloff": 3.0}
        if not stype.startswith(("sensor.camera.semantic_segmentation",
                                 "sensor.camera.depth")):
            attrs.update(chromatic_aberration_intensity=0.5, chromatic_aberration_offset=0)
    elif stype.startswith("sensor.lidar"):
        attrs = {"range": 85, "rotation_frequency": 10, "channels": 64,
                 "upper_fov": 10, "lower_fov": -30, "points_per_second": 600000}
        if not stype.startswith("sensor.lidar.ray_cast_semantic"):
            attrs.update(atmosphere_attenuation_rate=0.004, dropoff_general_rate=0.45,
                         dropoff_intensity_limit=0.8, dropoff_zero_intensity=0.4)
    elif stype.startswith("sensor.other.radar"):
        attrs = {"horizontal_fov": spec["fov"], "vertical_fov": spec["fov"],
                 "points_per_second": 1500, "range": 100}
    elif stype.startswith("sensor.other.gnss"):
        attrs = {"noise_alt_bias": 0.0, "noise_lat_bias": 0.0, "noise_lon_bias": 0.0}
    elif stype.startswith("sensor.other.imu"):
        attrs = {"noise_accel_stddev_x": 0.001, "noise_accel_stddev_y": 0.001,
                 "noise_accel_stddev_z": 0.015, "noise_gyro_stddev_x": 0.001,
                 "noise_gyro_stddev_y": 0.001, "noise_gyro_stddev_z": 0.001}
    # No sensor_tick, though TCP's specs carry one: the leaderboard ignores it,
    # and IMU at 0.05 s on a 0.05 s step drops frames to float rounding --
    # which in sync mode stalls every other tick waiting for it.
    return attrs


class SensorRig:
    def __init__(self, world, vehicle, specs: list[dict]) -> None:
        self.world = world
        self.vehicle = vehicle
        self._lock = threading.Condition()
        self._data: dict[str, tuple[int, object]] = {}
        self._ids: list[str] = []                 # real (non-pseudo) sensors
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

        for attr, value in blueprint_attributes(spec).items():
            if bp.has_attribute(attr):
                bp.set_attribute(attr, str(value))
            else:
                print(f"[sensors] {stype} has no attribute {attr!r}, left at default")

        rotation = carla.Rotation() if stype.startswith("sensor.other.gnss") else \
            carla.Rotation(roll=float(spec.get("roll", 0.0)),
                           pitch=float(spec.get("pitch", 0.0)),
                           yaw=float(spec.get("yaw", 0.0)))
        tf = carla.Transform(
            carla.Location(x=float(spec.get("x", 0.0)), y=float(spec.get("y", 0.0)),
                           z=float(spec.get("z", 0.0))),
            rotation,
        )
        actor = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
        ref = weakref.ref(self)
        actor.listen(lambda data, _id=sid, _t=stype: SensorRig._on_data(ref, _id, _t, data))
        self._actors.append(actor)
        self._ids.append(sid)

    @staticmethod
    def _on_data(ref, sid, stype, data) -> None:
        self = ref()
        if self is None:
            return
        # raw_data is a view of memory CARLA frees once this callback returns,
        # so anything kept past it must be copied (a bare view reads back as
        # zeros -- an all-black camera frame).
        if stype == "sensor.camera.rgb":
            arr = np.frombuffer(data.raw_data, dtype=np.uint8).copy()
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
            payload = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        else:
            payload = data
        with self._lock:
            self._data[sid] = (data.frame, payload)
            self._lock.notify_all()

    # ── read-out ────────────────────────────────────────────────────────
    def wait_for(self, frame: int, timeout: float = 2.0) -> bool:
        """Synchronous mode: block until every sensor has delivered ``frame``
        (or a later one). False on timeout -- the caller then uses what it has."""
        def ready():
            return all(self._data.get(sid, (-1,))[0] >= frame for sid in self._ids)
        with self._lock:
            return self._lock.wait_for(ready, timeout=timeout)

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
