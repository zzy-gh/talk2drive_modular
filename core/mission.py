"""
core/mission.py
===============
The mission layer: natural-language intent in, :class:`GlobalPlan` out.

This is the part of talk2drive that is worth reusing across AD stacks, so it
knows nothing about controllers, vehicles, sensors or debug drawing. Two rules
keep it portable:

* :meth:`MissionPlanner.apply` only mutates mission state. It never blocks,
  never prompts, never touches the map — safe to call from a real-time
  ``run_step``.
* :meth:`MissionPlanner.build_plan` does the map work and may ask the injected
  :class:`~.resolver.Interaction` a question. Hosts that own a real-time loop
  inject a non-blocking policy or call it off the control thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .landmark_kb import LandmarkKnowledgeBase
from .resolver import (AutoInteraction, InsertChoice, Interaction, PlaceChoice)
from .route_provider import RouteProvider
from .types import Directive, GlobalPlan, Place, RoutePoint


@dataclass
class Stop:
    label: str
    order: int = 0
    confidence: float = 1.0
    index: int | None = None
    enroute: bool = False


@dataclass
class ApplyResult:
    """What ``apply`` did, so the host knows whether to re-plan."""

    ok: bool = True
    needs_replan: bool = False
    cancelled: bool = False
    pull_over: bool = False
    message: str = ""


class MissionPlanner:
    def __init__(self,
                 kb: LandmarkKnowledgeBase,
                 route_provider: RouteProvider,
                 interaction: Interaction | None = None,
                 town: str = "Town01",
                 detour_ratio: float = 1.4,
                 arrival_radius: float = 5.0) -> None:
        self.kb = kb
        self.routes = route_provider
        self.interaction = interaction or AutoInteraction()
        self.town = town
        self.detour_ratio = detour_ratio
        self.arrival_radius = arrival_radius

        # Mission state
        self.destination: str | None = None
        self.destination_index: int | None = None
        self.stops: list[Stop] = []
        self.directive = Directive()
        self.plan: GlobalPlan | None = None

        self._place_cache: dict[str, Place] = {}
        self._revision = 0

    # ─────────────────────────────────────────
    # State machine — non-blocking
    # ─────────────────────────────────────────

    def apply(self, intent: dict) -> ApplyResult:
        cmd = (intent or {}).get("command_type")

        if cmd == "plan_route":
            self._place_cache.clear()
            self.destination = intent.get("destination")
            self.destination_index = intent.get("destination_index")
            self.stops = [
                Stop(label=s["type"], order=s.get("order", i + 1),
                     confidence=s.get("confidence", 1.0), index=s.get("index"))
                for i, s in enumerate(sorted(intent.get("waypoints", []) or [],
                                             key=lambda s: s.get("order", 0)))
            ]
            self.directive = _directive_from(intent)
            return ApplyResult(needs_replan=True,
                               message=f"route to {self.destination}")

        if cmd == "new_destination":
            old = self.destination
            self._place_cache.pop(intent.get("destination"), None)
            self.destination = intent.get("destination")
            self.destination_index = intent.get("destination_index")
            if intent.get("keep_waypoints", False):
                if old:
                    self.stops.append(Stop(label=old, order=len(self.stops) + 1))
            else:
                self.stops = []
            self.directive = _directive_from(intent, self.directive)
            self._renumber()
            return ApplyResult(needs_replan=True,
                               message=f"destination -> {self.destination}")

        if cmd == "remove_stop":
            if self.destination is None:
                return ApplyResult(ok=False, message="no active route")
            label = (intent.get("remove") or {}).get("type")
            before = len(self.stops)
            self.stops = [s for s in self.stops if s.label != label]
            self._place_cache.pop(label, None)
            self._renumber()
            return ApplyResult(needs_replan=before != len(self.stops),
                               message=f"removed '{label}' ({before} -> {len(self.stops)})")

        if cmd == "insert_stop":
            if self.destination is None:
                return ApplyResult(ok=False, message="no active route")
            ins = intent.get("insert") or {}
            stop = Stop(label=ins["type"], confidence=ins.get("confidence", 1.0),
                        index=ins.get("index"))
            position = ins.get("position")
            after = ins.get("after")
            if position == "next":
                self.stops.insert(0, stop)
            elif position == "last":
                self.stops.append(stop)
            elif after:
                at = next((i for i, s in enumerate(self.stops) if s.label == after), None)
                self.stops.insert(at + 1 if at is not None else len(self.stops), stop)
            else:
                stop.enroute = True          # position decided at plan time
                self.stops.append(stop)
            self._renumber()
            return ApplyResult(needs_replan=True,
                               message=f"inserted '{stop.label}' ({position})")

        if cmd == "cancel_route":
            self.reset()
            return ApplyResult(cancelled=True,
                               pull_over=bool(intent.get("pull_over", True)),
                               message="route cancelled")

        return ApplyResult(ok=False, message=f"unknown command_type: {cmd!r}")

    def reset(self) -> None:
        self.destination = None
        self.destination_index = None
        self.stops = []
        self.plan = None
        self.directive = Directive()
        self._place_cache.clear()

    @property
    def has_mission(self) -> bool:
        return self.destination is not None

    def _renumber(self) -> None:
        for i, s in enumerate(self.stops):
            s.order = i + 1

    # ─────────────────────────────────────────
    # Planning — touches the map, may interact
    # ─────────────────────────────────────────

    def build_plan(self, ego: Sequence[float],
                   start_fn: Callable[[], Sequence[float]] | None = None) -> GlobalPlan | None:
        """
        Resolve every landmark and trace the full route.

        ``ego`` is (x, y, z) used as the reference for "nearest" and
        "on the way". ``start_fn``, if given, is re-read *after* all
        interaction so the route starts from where the car actually is
        rather than where it was when the question was asked.
        """
        if self.destination is None:
            return None

        # Phase 1 — resolve places (the only phase that may prompt)
        dest = self._resolve(self.destination, "destination", ego[0], ego[1],
                             index=self.destination_index)
        if dest is None:
            return None

        resolved: list[tuple[Stop, Place]] = []
        probe_x, probe_y = ego[0], ego[1]
        for stop in self.stops:
            place = self._resolve(stop.label, "stop", probe_x, probe_y,
                                  index=stop.index, dest=dest)
            if place is None:
                continue
            resolved.append((stop, place))
            probe_x, probe_y = place.x, place.y

        fixed = [(s, p) for s, p in resolved if not s.enroute]
        for s, p in (x for x in resolved if x[0].enroute):
            at = self.interaction.choose_insert_position(
                InsertChoice(label=s.label,
                             stop_labels=[st.label for st, _ in fixed],
                             destination_label=self.destination))
            fixed.insert(max(0, min(at, len(fixed))), (s, p))
            s.enroute = False                      # position is now committed

        # Phase 2 — trace, from the freshest start pose available
        start = tuple(start_fn()) if start_fn else tuple(ego)
        start = self.routes.snap(*start)

        points: list[RoutePoint] = []
        cursor = start
        for _, place in fixed:
            leg = self.routes.trace(cursor, place.as_tuple())
            if not leg:
                print(f"[mission] no route leg to '{place.label}', skipping it")
                continue
            points.extend(leg)
            cursor = place.as_tuple()

        leg = self.routes.trace(cursor, dest.as_tuple())
        if not leg:
            print(f"[mission] no route to destination '{dest.label}'")
            return None
        points.extend(leg)

        self._revision += 1
        self.plan = GlobalPlan(points=points, goal=dest,
                               stops=[p for _, p in fixed],
                               revision=self._revision, start=start)
        # Keep mission state consistent with what was actually planned
        self.stops = [s for s, _ in fixed]
        self._renumber()
        return self.plan

    def is_arrived(self, x: float, y: float) -> bool:
        return bool(self.plan) and self.plan.distance_to_goal(x, y) <= self.arrival_radius

    # ─────────────────────────────────────────
    # Landmark resolution
    # ─────────────────────────────────────────

    def _candidates(self, label: str) -> list[Place]:
        out = []
        for row in self.kb.find_all(self.town, label):
            x, y, z = self.routes.snap(row["x"], row["y"], row["z"])
            out.append(Place(label=label, x=x, y=y, z=z, name=row.get("carla_name", "")))
        return out

    def _resolve(self, label: str, role: str, ref_x: float, ref_y: float,
                 index: int | None = None, dest: Place | None = None) -> Place | None:
        if label in self._place_cache:
            return self._place_cache[label]

        cands = self._candidates(label)
        if not cands:
            print(f"[mission] no '{label}' in {self.town}")
            return None

        # Passenger named the instance explicitly ("the second museum")
        if index is not None and 1 <= index <= len(cands):
            chosen = cands[index - 1]
            self._place_cache[label] = chosen
            return chosen

        # Stops: prefer the ones that do not blow up the detour
        if role == "stop" and dest is not None and len(cands) > 1:
            direct = math.hypot(dest.x - ref_x, dest.y - ref_y)
            if direct > 1.0:
                on_way = [p for p in cands
                          if (p.distance_to(ref_x, ref_y)
                              + math.hypot(dest.x - p.x, dest.y - p.y)) / direct
                          < self.detour_ratio]
                if on_way:
                    if len(on_way) < len(cands):
                        print(f"[mission] {len(on_way)}/{len(cands)} '{label}' are on the way")
                    cands = on_way

        if len(cands) == 1:
            self._place_cache[label] = cands[0]
            return cands[0]

        chosen = self.interaction.choose_place(
            PlaceChoice(label=label, role=role, candidates=cands,
                        ref_x=ref_x, ref_y=ref_y))
        if chosen is not None:
            self._place_cache[label] = chosen
        return chosen


def _directive_from(intent: dict, previous: Directive | None = None) -> Directive:
    prev = previous or Directive()
    return Directive(
        utterance=intent.get("_utterance", prev.utterance),
        urgency=intent.get("urgency", "normal"),
        preference=intent.get("route_preference", prev.preference),
        avoid=tuple(intent.get("avoid", ()) or ()),
    )
