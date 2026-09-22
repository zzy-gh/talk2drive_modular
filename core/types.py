"""
core/types.py
=============
Stack-agnostic data types exchanged between the mission layer (talk2drive)
and the control layer (BasicAgent / SimLingo / TCP / ...).

Nothing in this module imports carla at module scope, so the mission layer
can be imported and unit-tested without a CARLA installation. The few
conversion helpers that need carla import it lazily.

Frame conventions
-----------------
World coordinates are CARLA world coordinates (left-handed: +x east,
+y south, +z up; yaw in degrees, growing from +x towards +y).

Ego coordinates are produced by :func:`world_to_ego` and default to
``forward_right`` — ``[along the ego forward vector, along the ego right
vector]`` in metres. This matches the frame SimLingo / CoVLM feed into
``target_points_ego``. Backends that need another layout pass
``convention="right_forward"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# RoadOption.LANEFOLLOW.value — duplicated here so this module stays carla-free.
LANEFOLLOW = 4


# ─────────────────────────────────────────────
# Places
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Place:
    """A resolved, road-snapped landmark."""

    label: str                  # building_type, e.g. "hospital"
    x: float
    y: float
    z: float = 0.0
    name: str = ""              # carla_name from the knowledge base
    index: int | None = None    # 1-based index among same-labelled places,
                                # ONLY meaningful when source == "kb"
    source: str = "kb"          # where this Place came from. "kb" is a row of
                                # the landmark csv and is the only source an
                                # `index` applies to. Anything else was bound
                                # from outside the knowledge base, carries no
                                # ordinal, and must not be re-snapped -- see
                                # whoever set it.

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class RoutePoint:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    option: int = LANEFOLLOW     # RoadOption value

    def distance_to(self, other: "RoutePoint") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class GlobalPlan:
    """
    A route from the ego's current pose, through ``stops``, to ``goal``.

    This is the single object talk2drive hands to any backend. The
    ``to_*`` methods translate it into whatever shape that backend wants;
    no backend should ever reach into ``points`` and re-invent one.
    """

    points: list[RoutePoint]
    goal: Place
    stops: list[Place] = field(default_factory=list)
    revision: int = 0                       # bumped on every replan
    start: tuple[float, float, float] | None = None

    # ── basics ──────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.points)

    def __bool__(self) -> bool:
        return bool(self.points)

    def length(self) -> float:
        return sum(self.points[i].distance_to(self.points[i - 1])
                   for i in range(1, len(self.points)))

    def nearest_index(self, x: float, y: float, start: int = 0) -> int:
        if not self.points:
            return 0
        best_i, best_d = start, float("inf")
        for i in range(start, len(self.points)):
            d = (self.points[i].x - x) ** 2 + (self.points[i].y - y) ** 2
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def remaining_distance(self, x: float, y: float) -> float:
        """Route distance from the point nearest (x, y) to the goal."""
        if not self.points:
            return 0.0
        i = self.nearest_index(x, y)
        tail = sum(self.points[j].distance_to(self.points[j - 1])
                   for j in range(i + 1, len(self.points)))
        return math.hypot(self.points[i].x - x, self.points[i].y - y) + tail

    def distance_to_goal(self, x: float, y: float) -> float:
        """Straight-line distance to the goal (cheap arrival test)."""
        return self.goal.distance_to(x, y)

    # ── downsampling ────────────────────────────────────────────────────
    def downsample(self, sample_factor: float = 50.0) -> list[RoutePoint]:
        """
        Same rule the CARLA leaderboard uses (``downsample_route``): keep the
        first point, every road-option change, every lane change, the last
        point, and otherwise one point per ``sample_factor`` metres.

        Both the GPS plan for leaderboard agents and the sparse tracker that
        feeds VLA target points are built on top of this, so the sparsity a
        VLA sees here matches the sparsity it saw during training.
        """
        CHANGELANELEFT, CHANGELANERIGHT = 5, 6
        kept: list[RoutePoint] = []
        prev_option: int | None = None
        dist = 0.0

        for i, p in enumerate(self.points):
            if prev_option is None:
                kept.append(p); dist = 0.0
            elif p.option in (CHANGELANELEFT, CHANGELANERIGHT):
                kept.append(p); dist = 0.0
            elif p.option != prev_option and prev_option not in (CHANGELANELEFT, CHANGELANERIGHT):
                kept.append(p); dist = 0.0
            elif dist > sample_factor:
                kept.append(p); dist = 0.0
            elif i == len(self.points) - 1:
                kept.append(p); dist = 0.0
            else:
                dist += p.distance_to(self.points[i - 1])
            prev_option = p.option

        if kept and kept[-1] is not self.points[-1]:
            kept.append(self.points[-1])
        return kept

    # ── backend-specific views ──────────────────────────────────────────
    def to_carla_plan(self, carla_map) -> list[tuple[Any, Any]]:
        """``[(carla.Waypoint, RoadOption), ...]`` — BasicAgent / LocalPlanner."""
        import carla
        from agents.navigation.local_planner import RoadOption

        plan = []
        for p in self.points:
            wp = carla_map.get_waypoint(carla.Location(x=p.x, y=p.y, z=p.z),
                                        project_to_road=True)
            plan.append((wp, RoadOption(p.option)))
        return plan

    def to_transform_plan(self, sample_factor: float | None = None) -> list[tuple[Any, Any]]:
        """``[(carla.Transform, RoadOption), ...]`` — leaderboard world-coord plan."""
        import carla
        from agents.navigation.local_planner import RoadOption

        pts = self.downsample(sample_factor) if sample_factor else self.points
        return [(carla.Transform(carla.Location(x=p.x, y=p.y, z=p.z),
                                 carla.Rotation(yaw=p.yaw)),
                 RoadOption(p.option)) for p in pts]

    def to_gps_plan(self, lat_ref: float, lon_ref: float,
                    sample_factor: float | None = 50.0):
        """
        ``(global_plan_gps, global_plan_world_coord)`` — exactly the pair
        ``AutonomousAgent.set_global_plan`` expects (TCP, Bench2Drive, ...).
        """
        world_plan = self.to_transform_plan(sample_factor)
        gps_plan = [(location_to_gps(lat_ref, lon_ref, tf.location), opt)
                    for tf, opt in world_plan]
        return gps_plan, world_plan

    def to_ego_target_points(self, ego_transform, lookaheads: Sequence[float] = (10.0, 20.0),
                             convention: str = "forward_right"):
        """
        ``numpy [n, 2]`` ego-frame target points sampled at fixed arc-length
        lookaheads — the stateless variant, handy for smoke tests.

        For SimLingo prefer :class:`~talk2drive_modular.core.tracking.SparseRouteTracker`,
        which reproduces the leaderboard route planner the model was trained with.
        """
        import numpy as np

        loc = ego_transform.location
        if not self.points:
            return np.array([[la, 0.0] for la in lookaheads], dtype=np.float32)

        i0 = self.nearest_index(loc.x, loc.y)
        out = []
        for la in lookaheads:
            travelled, idx = 0.0, i0
            while idx + 1 < len(self.points) and travelled < la:
                travelled += self.points[idx + 1].distance_to(self.points[idx])
                idx += 1
            p = self.points[idx]
            out.append(world_to_ego(ego_transform, p.x, p.y, convention))
        return np.asarray(out, dtype=np.float32)


# ─────────────────────────────────────────────
# Frame helpers
# ─────────────────────────────────────────────

def world_to_ego(ego_transform, x: float, y: float,
                 convention: str = "forward_right") -> tuple[float, float]:
    """
    Project a world point into the ego frame using CARLA's own forward/right
    vectors, so no sign convention has to be guessed. Verified against
    ``Transform.get_forward_vector()`` / ``get_right_vector()``.
    """
    loc = ego_transform.location
    fwd = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    dx, dy = x - loc.x, y - loc.y
    f = dx * fwd.x + dy * fwd.y
    r = dx * right.x + dy * right.y
    if convention == "forward_right":
        return (f, r)
    if convention == "right_forward":
        return (r, f)
    raise ValueError(f"unknown ego convention: {convention!r}")


def location_to_gps(lat_ref: float, lon_ref: float, location) -> dict:
    """CARLA world location → ``{'lat','lon','z'}`` (leaderboard formula)."""
    EARTH_RADIUS_EQUA = 6378137.0
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0 + location.x
    my = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + lat_ref) * math.pi / 360.0)) - location.y
    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    return {"lat": lat, "lon": lon, "z": location.z}


def get_latlon_ref(world) -> tuple[float, float]:
    """Read the geo-reference out of the map's OpenDRIVE header."""
    import xml.etree.ElementTree as ET

    lat_ref, lon_ref = 42.0, 2.0
    tree = ET.ElementTree(ET.fromstring(world.get_map().to_opendrive()))
    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    for item in georef.text.split(" "):
                        if "+lat_0" in item:
                            lat_ref = float(item.split("=")[1])
                        if "+lon_0" in item:
                            lon_ref = float(item.split("=")[1])
    return lat_ref, lon_ref


# ─────────────────────────────────────────────
# Directive — the language / style channel
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Directive:
    """
    Everything the passenger said that is *not* "where to go".

    A PID backend consumes only ``target_speed_kph()``; a VLA backend also
    consumes ``as_speed_instruction()``. Keeping the raw utterance means a
    language-native backend is not restricted to what the intent schema
    happened to capture.
    """

    utterance: str = ""
    urgency: str = "normal"                 # "high" | "normal"
    preference: str | None = None           # "scenic" | "fastest" | "shortest"
    avoid: tuple[str, ...] = ()

    def target_speed_kph(self) -> float:
        return 45.0 if self.urgency == "high" else 30.0

    def as_speed_instruction(self) -> str | None:
        """The extra sentence appended to a VLA prompt, or None."""
        if self.urgency == "high":
            return "Please hurry, drive faster."
        if self.preference == "scenic":
            return "Take it easy and drive calmly."
        return None


# ─────────────────────────────────────────────
# Observation — what a backend gets each tick
# ─────────────────────────────────────────────

@dataclass
class Observation:
    ego_transform: Any                      # carla.Transform
    speed_mps: float = 0.0
    sensors: dict = field(default_factory=dict)   # sensor id → data
    timestamp: float = 0.0
    frame: int = 0

    @property
    def x(self) -> float:
        return self.ego_transform.location.x

    @property
    def y(self) -> float:
        return self.ego_transform.location.y

    @property
    def z(self) -> float:
        return self.ego_transform.location.z
