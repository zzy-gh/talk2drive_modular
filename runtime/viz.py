"""
runtime/viz.py
==============
All CARLA debug drawing. Lifted out of the planner so the mission layer can
run headless (evaluation, unit tests, a ROS node) without dragging debug
rendering along.
"""

from __future__ import annotations

import carla

from core.types import GlobalPlan, Place

GREEN = carla.Color(0, 255, 0)
ORANGE = carla.Color(255, 80, 0)
STOP_ORANGE = carla.Color(255, 165, 0)
BLUE = carla.Color(0, 0, 255)
RED = carla.Color(255, 0, 0)
YELLOW = carla.Color(255, 220, 0)
ERASE = carla.Color(40, 40, 40)


def _dim(color: carla.Color, factor: float) -> carla.Color:
    return carla.Color(int(color.r * factor), int(color.g * factor), int(color.b * factor))


def draw_points(world, points, color=GREEN, life_time=0.0) -> None:
    arrow_color = _dim(color, 0.04)
    for i, p in enumerate(points):
        loc = carla.Location(x=p.x, y=p.y, z=p.z + 0.5)
        world.debug.draw_point(loc, size=0.05, color=color, life_time=life_time)
        if i > 0:
            prev = points[i - 1]
            world.debug.draw_line(carla.Location(x=prev.x, y=prev.y, z=prev.z + 0.5),
                                  loc, thickness=0.02, color=color, life_time=life_time)
        if i % 20 == 0 and i + 1 < len(points):
            nxt = points[i + 1]
            world.debug.draw_arrow(loc, carla.Location(x=nxt.x, y=nxt.y, z=nxt.z + 0.5),
                                   thickness=0.08, arrow_size=0.45,
                                   color=arrow_color, life_time=life_time)


def draw_marker(world, x, y, z, label, color, life_time=0.0) -> None:
    loc = carla.Location(x=x, y=y, z=z + 1.0)
    world.debug.draw_point(loc, size=0.15, color=color, life_time=life_time)
    world.debug.draw_string(loc + carla.Location(z=0.6), label, color=color,
                            life_time=life_time)


class PlanVisualizer:
    """Draws one plan at a time and erases the previous one on replace."""

    def __init__(self, world) -> None:
        self.world = world
        self._points = None
        self._markers: list[tuple[float, float, float, str]] = []
        self._previews: list[tuple[float, float, float, str]] = []

    # ── plan ────────────────────────────────────────────────────────────
    def show(self, plan: GlobalPlan, urgency: str = "normal") -> None:
        self.clear()
        color = ORANGE if urgency == "high" else GREEN
        draw_points(self.world, plan.points, color=color)
        self._points = plan.points

        if plan.start:
            self._marker(*plan.start, "START", BLUE)
        for i, stop in enumerate(plan.stops):
            self._marker(stop.x, stop.y, stop.z, f"STOP {i + 1}: {stop.label}", STOP_ORANGE)
        self._marker(plan.goal.x, plan.goal.y, plan.goal.z,
                     f"GOAL: {plan.goal.label}", RED)

    def clear(self) -> None:
        if self._points:
            draw_points(self.world, self._points, color=ERASE)
            self._points = None
        for x, y, z, label in self._markers:
            draw_marker(self.world, x, y, z, label, ERASE)
        self._markers = []

    def _marker(self, x, y, z, label, color) -> None:
        draw_marker(self.world, x, y, z, label, color)
        self._markers.append((x, y, z, label))

    # ── disambiguation previews (hooks for CliInteraction) ──────────────
    def preview_places(self, places: list[Place]) -> None:
        self.clear_previews()
        for i, p in enumerate(places):
            label = f"[{i + 1}] {p.label}"
            draw_marker(self.world, p.x, p.y, p.z, label, YELLOW, life_time=120.0)
            self._previews.append((p.x, p.y, p.z, label))

    def clear_previews(self) -> None:
        for x, y, z, label in self._previews:
            draw_marker(self.world, x, y, z, label, carla.Color(0, 0, 0))
        self._previews = []

    # ── misc ────────────────────────────────────────────────────────────
    def beacon(self, location, color=carla.Color(0, 220, 255)) -> None:
        self.world.debug.draw_point(location + carla.Location(z=3.5), size=0.25,
                                    color=color, life_time=0.1)
