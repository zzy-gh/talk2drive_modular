"""
runtime/viz.py
==============
All CARLA debug drawing. Lifted out of the planner so the mission layer can
run headless (evaluation, unit tests, a ROS node) without dragging debug
rendering along.
"""

from __future__ import annotations

import threading

import carla

from core.types import GlobalPlan, Place

GREEN = carla.Color(0, 255, 0)
ORANGE = carla.Color(255, 80, 0)
STOP_ORANGE = carla.Color(255, 165, 0)
BLUE = carla.Color(0, 0, 255)
RED = carla.Color(255, 0, 0)
YELLOW = carla.Color(255, 220, 0)
GREY = carla.Color(170, 170, 170)
DONE_GREEN = carla.Color(0, 200, 90)
PURPLE = carla.Color(170, 60, 255)
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


def draw_marker(world, x, y, z, label, color, life_time=0.0, size=0.15) -> None:
    loc = carla.Location(x=x, y=y, z=z + 1.0)
    world.debug.draw_point(loc, size=size, color=color, life_time=life_time)
    world.debug.draw_string(loc + carla.Location(z=0.6), label, color=color,
                            life_time=life_time)


# draw_string is "only seen server-side" (python_api.md): it shows in the
# CARLA window but never in a camera sensor's image, unlike points and lines.
# Strings live on the server HUD (Game/CarlaHUD.cpp), which drops each one at
# wall-clock time now + life_time, so 0 is NOT permanent for strings. The HUD
# draws them in the order added, so the same text at the same spot drawn later
# in another colour covers the earlier one -- that is how a route is "erased".
ROUTE_CHAR = "o"
ROUTE_SPACING_M = 3.0
TEXT_PERSIST_S = 24 * 3600.0     # "permanent" for a string
# Text is anti-aliased, so one grey glyph over a green one leaves a green
# fringe; each extra layer blends that fringe further toward grey.
TEXT_ERASE = carla.Color(80, 80, 80)
TEXT_ERASE_LAYERS = 3


def draw_route_text(world, points, color=GREEN, life_time=TEXT_PERSIST_S,
                    spacing_m: float = ROUTE_SPACING_M) -> None:
    """The route as a dotted line of characters -- invisible to sensors."""
    step = max(1, int(round(spacing_m / max(1e-3, _spacing(points)))))
    for p in points[::step]:
        world.debug.draw_string(carla.Location(x=p.x, y=p.y, z=p.z + 0.5), ROUTE_CHAR,
                                color=color, life_time=life_time)


def _spacing(points) -> float:
    """Mean distance between consecutive route points."""
    if len(points) < 2:
        return 1.0
    total = sum(points[i].distance_to(points[i - 1]) for i in range(1, len(points)))
    return total / (len(points) - 1)


def _key(place: Place) -> tuple[str, float, float]:
    return (place.label.lower(), round(place.x, 1), round(place.y, 1))


class LandmarkBoard:
    """
    Every landmark in the town, always on screen, coloured by its role:

        grey    not part of the mission     orange  planned stop
        red     destination                 green   reached (stop or goal)
        purple  cancelled                   yellow  numbered candidate

    Markers are drawn with a short ``life_time`` and redrawn by the session
    loop, so a state change replaces the old marker instead of painting a
    dark one over it.
    """

    # state -> (colour, point size, label prefix)
    STYLE = {
        "idle":        (GREY, 0.08, ""),
        "stop":        (STOP_ORANGE, 0.2, "STOP {n}: "),
        "destination": (RED, 0.25, "GOAL: "),
        "visited":     (DONE_GREEN, 0.2, "DONE: "),
        "cancelled":   (PURPLE, 0.15, "CANCELLED: "),
    }

    def __init__(self, world, places: list[Place]) -> None:
        self.world = world
        self._lock = threading.Lock()
        self._places = {_key(p): p for p in places}
        self._state: dict[tuple, tuple[str, int | None]] = \
            {k: ("idle", None) for k in self._places}
        self._candidates: dict[tuple, int] = {}

    # ── state changes (command thread) ──────────────────────────────────
    def set_plan(self, plan: GlobalPlan) -> None:
        """Stops and goal of ``plan`` become active; whatever was active
        before and is no longer in the plan counts as cancelled."""
        new: dict[tuple, tuple[str, int | None]] = {}
        for i, p in enumerate(plan.stops):
            new[_key(p)] = ("stop", i + 1)
        new[_key(plan.goal)] = ("destination", None)
        with self._lock:
            for p in (*plan.stops, plan.goal):
                self._places.setdefault(_key(p), p)
            for k, (state, _) in list(self._state.items()):
                if state in ("stop", "destination") and k not in new:
                    self._state[k] = ("cancelled", None)
            self._state.update(new)

    def mark_visited(self, place: Place) -> None:
        with self._lock:
            self._places.setdefault(_key(place), place)
            self._state[_key(place)] = ("visited", None)

    def cancel_active(self) -> None:
        with self._lock:
            for k, (state, _) in list(self._state.items()):
                if state in ("stop", "destination"):
                    self._state[k] = ("cancelled", None)

    def reset(self) -> None:
        with self._lock:
            self._state = {k: ("idle", None) for k in self._places}
            self._candidates = {}

    def set_candidates(self, places: list[Place]) -> None:
        with self._lock:
            for p in places:
                self._places.setdefault(_key(p), p)
            self._candidates = {_key(p): i + 1 for i, p in enumerate(places)}

    def clear_candidates(self) -> None:
        with self._lock:
            self._candidates = {}

    # ── read-out ────────────────────────────────────────────────────────
    def snapshot(self) -> list[tuple[Place, str, int | None, int | None]]:
        """(place, state, stop number, candidate number) for every landmark."""
        with self._lock:
            return [(self._places[k], *self._state.get(k, ("idle", None)),
                     self._candidates.get(k)) for k in self._places]

    # ── drawing (control thread) ────────────────────────────────────────
    def draw(self, life_time: float) -> None:
        for p, state, n, cand in self.snapshot():
            if cand is not None:
                color, size, label = YELLOW, 0.25, f"[{cand}] {p.label}"
            else:
                color, size, prefix = self.STYLE[state]
                label = prefix.format(n=n) + p.label
            draw_marker(self.world, p.x, p.y, p.z, label, color,
                        life_time=life_time, size=size)


class PlanVisualizer:
    """Draws one plan at a time and erases the previous one on replace.
    Landmark markers (goal, stops, candidates) live on :attr:`board`.

    With ``draw_world=False`` nothing is drawn into the CARLA world -- a
    camera sensor would see it -- but the plan and landmark state are still
    tracked, for runtime/bev.py to show in its own window.

    With ``text_only=True`` everything is drawn with ``draw_string`` only,
    which the CARLA window shows and camera sensors do not -- and only the
    route and an "EGO" beacon: landmarks and the start are left to the BEV.
    The route is drawn once to last a day; replacing it paints the old one
    over in grey, as with points and lines."""

    def __init__(self, world, places: list[Place] = (), draw_world: bool = True,
                 text_only: bool = False) -> None:
        self.world = world
        self.draw_world = draw_world
        self.text_only = text_only
        self.board = LandmarkBoard(world, list(places))
        self.plan: GlobalPlan | None = None      # what is on screen, for the BEV
        self.urgency = "normal"
        self._points = None
        self._markers: list[tuple[float, float, float, str]] = []

    # ── plan ────────────────────────────────────────────────────────────
    def show(self, plan: GlobalPlan, urgency: str = "normal") -> None:
        self.clear()
        self.plan, self.urgency = plan, urgency
        self.board.set_plan(plan)
        if not self.draw_world:
            return
        self._route(plan.points, ORANGE if urgency == "high" else GREEN)
        self._points = plan.points
        if plan.start and not self.text_only:
            self._marker(*plan.start, "START", BLUE)

    def clear(self) -> None:
        self.plan = None
        if self._points:
            self._route(self._points, ERASE)
            self._points = None
        for x, y, z, label in self._markers:
            draw_marker(self.world, x, y, z, label, ERASE)
        self._markers = []

    def redraw(self, life_time: float) -> None:
        """Periodic redraw of the landmarks, from the session loop (not in
        text-only mode, where they are shown in the BEV window only)."""
        if self.text_only:
            return
        self.board.draw(life_time)

    def _route(self, points, color) -> None:
        if not self.text_only:
            draw_points(self.world, points, color=color)
        elif color is ERASE:
            for _ in range(TEXT_ERASE_LAYERS):
                draw_route_text(self.world, points, color=TEXT_ERASE)
        else:
            draw_route_text(self.world, points, color=color)

    def _marker(self, x, y, z, label, color) -> None:
        draw_marker(self.world, x, y, z, label, color)
        self._markers.append((x, y, z, label))

    # ── disambiguation previews (hooks for CliInteraction) ──────────────
    def preview_places(self, places: list[Place]) -> None:
        self.board.set_candidates(places)

    def clear_previews(self) -> None:
        self.board.clear_candidates()

    # ── misc ────────────────────────────────────────────────────────────
    def beacon(self, location, color=carla.Color(0, 220, 255), life_time: float = 0.1) -> None:
        if not self.draw_world:
            return
        if self.text_only:
            self.world.debug.draw_string(location + carla.Location(z=3.5), "EGO",
                                         color=color, life_time=life_time)
            return
        self.world.debug.draw_point(location + carla.Location(z=3.5), size=0.25,
                                    color=color, life_time=life_time)
