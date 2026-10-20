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

from .intent_schema import commands_of
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
    message: str = ""


class MissionPlanner:
    def __init__(self,
                 kb: LandmarkKnowledgeBase,
                 route_provider: RouteProvider,
                 interaction: Interaction | None = None,
                 town: str = "Town01",
                 detour_ratio: float = 1.4,
                 arrival_radius: float = 5.0,
                 place_distance: str = "route",
                 route_top_k: int = 10) -> None:
        """``place_distance`` is how "nearest" and "on the way" measure:

        * ``"route"``: the route the planner traces to each candidate, best
          first with a straight-line lower bound so that few are traced. The
          lengths are those of the routes actually driven. For towns with a
          handful of places per type.
        * ``"graph"``: one Dijkstra from the car (and, for a stop, one back
          from the destination) over the lane graph in metres scores every
          candidate at once. For large towns with hundreds of places.
        * ``"euclidean"``: the straight line throughout.

        Either road mode falls back to the next one the route provider
        supports, down to the straight line. ``route_top_k`` is how many
        candidates a question offers -- the best ones; in ``"route"`` mode
        also the fewest traced. <= 0 means all. The best candidate does not
        depend on it."""
        if place_distance not in ("route", "graph", "euclidean"):
            raise ValueError(f"place_distance must be 'route', 'graph' or 'euclidean', "
                             f"not {place_distance!r}")
        self.kb = kb
        self.routes = route_provider
        self.interaction = interaction or AutoInteraction()
        self.town = town
        self.detour_ratio = detour_ratio
        self.arrival_radius = arrival_radius
        self.place_distance = place_distance
        self.route_top_k = route_top_k

        # Mission state
        self.destination: str | None = None
        self.destination_index: int | None = None
        self.stops: list[Stop] = []
        self.directive = Directive()
        self.plan: GlobalPlan | None = None

        self._place_cache: dict[str, Place] = {}
        self._revision = 0
        self._next_stop = 0          # index into plan.stops of the next one to reach

    # ─────────────────────────────────────────
    # State machine — non-blocking
    # ─────────────────────────────────────────

    def apply(self, intent: dict) -> ApplyResult:
        """Apply one command, or ``{"commands": [...]}`` in order. Several
        commands come back as one result, so the host re-plans once."""
        if isinstance(intent, dict) and "commands" in intent:
            return self._apply_all(intent)
        return self._apply_one(intent)

    def _apply_all(self, intent: dict) -> ApplyResult:
        utterance = intent.get("_utterance")
        results = []
        for c in commands_of(intent):
            if utterance is not None:
                c = {**c, "_utterance": utterance}
            results.append(self._apply_one(c))
        message = "; ".join(r.message for r in results) or "no command"
        done = [r for r in results if r.ok]
        if not done:
            return ApplyResult(ok=False, message=message)
        # Cancelled only if the route is gone at the end: "cancel, then take
        # me to the park" leaves a route.
        cancelled = self.destination is None and any(r.cancelled for r in done)
        return ApplyResult(needs_replan=not cancelled and any(r.needs_replan or r.cancelled
                                                              for r in done),
                           cancelled=cancelled, message=message)

    def _stop_at(self, label: str | None) -> int | None:
        return next((i for i, s in enumerate(self.stops) if s.label == label), None)

    def _place(self, stop: Stop, position, after, before) -> bool:
        """Put ``stop`` into the stop list where it was asked to go; False if
        no position was given. A named stop that is not there means the end."""
        if position == "next":
            self.stops.insert(0, stop)
        elif position == "last":
            self.stops.append(stop)
        elif after:
            at = self._stop_at(after)
            self.stops.insert(at + 1 if at is not None else len(self.stops), stop)
        elif before:
            at = self._stop_at(before)
            self.stops.insert(at if at is not None else len(self.stops), stop)
        else:
            return False
        return True

    def _apply_one(self, intent: dict) -> ApplyResult:
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
            old, old_index = self.destination, self.destination_index
            self._place_cache.pop(intent.get("destination"), None)
            self.destination = intent.get("destination")
            self.destination_index = intent.get("destination_index")
            if intent.get("keep_waypoints", False):
                if old:
                    self.stops.append(Stop(label=old, order=len(self.stops) + 1,
                                           index=old_index))
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
            if ins.get("after") and ins["after"] == self.destination:
                # "after the <destination>" is past the end of the route: the
                # new place becomes the destination, the old one its last stop.
                return self._extend(ins["type"], ins.get("index"), intent)
            stop = Stop(label=ins["type"], confidence=ins.get("confidence", 1.0),
                        index=ins.get("index"))
            position, before = ins.get("position"), ins.get("before")
            if before and before == self.destination:      # "before the destination"
                position, before = "last", None
            if not self._place(stop, position, ins.get("after"), before):
                stop.enroute = True          # position decided at plan time
                self.stops.append(stop)
            self._renumber()
            return ApplyResult(needs_replan=True,
                               message=f"inserted '{stop.label}' ({position})")

        if cmd == "move_stop":
            mv = intent.get("move") or {}
            label = mv.get("type")
            at = self._stop_at(label)
            if self.destination is None or at is None:
                return ApplyResult(ok=False, message=f"no stop '{label}' to move")
            position, after, before = mv.get("position"), mv.get("after"), mv.get("before")
            if label in (after, before):
                return ApplyResult(ok=False, message=f"cannot move '{label}' relative to itself")
            if after and after == self.destination:
                stop = self.stops.pop(at)
                return self._extend(stop.label, stop.index, intent)
            if before and before == self.destination:
                position, before = "last", None
            stop = self.stops.pop(at)
            stop.enroute = False
            if not self._place(stop, position, after, before):
                self.stops.insert(at, stop)                 # put it back
                return ApplyResult(ok=False, message=f"no position to move '{label}' to")
            self._renumber()
            return ApplyResult(needs_replan=True, message=f"moved '{label}'")

        if cmd == "cancel_route":
            self.reset()
            return ApplyResult(cancelled=True, message="route cancelled")

        return ApplyResult(ok=False, message=f"unknown command_type: {cmd!r}")

    def _extend(self, label: str, index: int | None, intent: dict) -> ApplyResult:
        """Going somewhere after the destination: it becomes the destination
        and the old one its last stop."""
        return self._apply_one({"command_type": "new_destination", "destination": label,
                                "destination_index": index, "keep_waypoints": True,
                                "urgency": self.directive.urgency,
                                "_utterance": intent.get("_utterance", self.directive.utterance)})

    def reset(self) -> None:
        self.destination = None
        self.destination_index = None
        self.stops = []
        self.plan = None
        self.directive = Directive()
        self._place_cache.clear()
        self._next_stop = 0

    def route_summary(self) -> dict:
        """The active route as the parser needs it -- what "after that" or
        "skip the cafe" refers to. ``destination`` is None when idle."""
        return {"destination": self.destination,
                "destination_index": self.destination_index,
                "stops": [{"type": s.label, "index": s.index} for s in list(self.stops)]}

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
        ego_z = ego[2] if len(ego) > 2 else 0.0
        dest = self._resolve(self.destination, "destination", ego[0], ego[1],
                             index=self.destination_index, ref_z=ego_z)
        if dest is None:
            return None

        resolved: list[tuple[Stop, Place]] = []
        probe = (ego[0], ego[1], ego_z)
        for stop in self.stops:
            place = self._resolve(stop.label, "stop", probe[0], probe[1],
                                  index=stop.index, dest=dest, ref_z=probe[2])
            if place is None:
                continue
            resolved.append((stop, place))
            probe = place.as_tuple()

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
        self._next_stop = 0
        # Keep mission state consistent with what was actually planned
        self.stops = [s for s, _ in fixed]
        self._renumber()
        return self.plan

    def is_arrived(self, x: float, y: float) -> bool:
        """At the goal with every stop reached."""
        return (bool(self.plan) and self._next_stop >= len(self.plan.stops)
                and self.plan.distance_to_goal(x, y) <= self.arrival_radius)

    def reached_stop(self, x: float, y: float) -> Place | None:
        """The next planned stop, once the car is within ``arrival_radius`` of
        it. Stops are reached in plan order, and a reached one is dropped from
        the mission so a later re-plan does not route back to it."""
        if not self.plan or self._next_stop >= len(self.plan.stops):
            return None
        place = self.plan.stops[self._next_stop]
        if place.distance_to(x, y) > self.arrival_radius:
            return None
        self._next_stop += 1
        at = next((i for i, s in enumerate(self.stops) if s.label == place.label), None)
        if at is not None:
            del self.stops[at]
            self._renumber()
        return place

    def all_places(self) -> list[Place]:
        """Every landmark in the town, road-snapped exactly as
        :meth:`build_plan` resolves them."""
        return [p for label in self.kb.get_landmark_types(self.town)
                for p in self._candidates(label)]

    # ─────────────────────────────────────────
    # Landmark resolution
    # ─────────────────────────────────────────

    def _candidates(self, label: str) -> list[Place]:
        out = []
        for i, row in enumerate(self.kb.find_all(self.town, label), start=1):
            x, y, z = self.routes.snap(row["x"], row["y"], row["z"])
            out.append(Place(label=label, x=x, y=y, z=z, name=row.get("carla_name", ""),
                             index=i))
        return out

    def _resolve(self, label: str, role: str, ref_x: float, ref_y: float,
                 index: int | None = None, dest: Place | None = None,
                 ref_z: float = 0.0) -> Place | None:
        if label in self._place_cache:
            return self._place_cache[label]

        cands = self._candidates(label)
        total = len(cands)
        if not cands:
            print(f"[mission] no '{label}' in {self.town}")
            return None

        # Passenger named the instance explicitly ("the second museum")
        if index is not None and 1 <= index <= len(cands):
            chosen = cands[index - 1]
            self._place_cache[label] = chosen
            return chosen

        ref = (ref_x, ref_y, ref_z)
        meter = self._road_meter()
        dist = direct = None
        if len(cands) > 1 and self.place_distance == "graph" \
                and hasattr(self.routes, "route_lengths"):
            cands, dist, direct = self._rank_by_graph(cands, ref, role, dest)
        elif len(cands) > 1 and meter is not None:
            cands, dist, direct = self._rank_by_road(cands, ref, role, dest, meter)
        if dist is None and role == "stop" and dest is not None and len(cands) > 1:
            cands = self._on_the_way_by_line(label, cands, ref_x, ref_y, dest)

        # A question offers the best route_top_k, however many the town has
        k = self.route_top_k if self.route_top_k > 0 else len(cands)
        if len(cands) > k:
            if dist is None:
                cands = sorted(cands, key=lambda p: p.distance_to(ref_x, ref_y))[:k]
            else:
                cands, dist = cands[:k], dist[:k]

        if len(cands) == 1:
            self._place_cache[label] = cands[0]
            return cands[0]

        chosen = self.interaction.choose_place(
            PlaceChoice(label=label, role=role, candidates=cands,
                        ref_x=ref_x, ref_y=ref_y, distances=dist, direct=direct,
                        total=total))
        if chosen is not None:
            self._place_cache[label] = chosen
        return chosen

    def _road_meter(self) -> Callable[[Sequence[float], Sequence[float]], float] | None:
        """The route provider's driving-distance function, or None when
        places are ranked by the straight line."""
        if self.place_distance == "euclidean":
            return None
        return getattr(self.routes, "route_length", None)

    def _rank_by_road(self, cands: list[Place], ref: Sequence[float], role: str,
                      dest: Place | None,
                      meter: Callable[[Sequence[float], Sequence[float]], float]
                      ) -> tuple[list[Place], list[float] | None, float | None]:
        """Candidates best first, their costs, and -- for a stop -- the trip
        without it.

        A destination costs its driving distance from ``ref``. A stop costs
        the whole trip ``ref -> stop -> dest``, so the best stop is the one
        that lengthens the trip least, not the one nearest the car (that may
        lie behind it). Stops that stretch the trip past ``detour_ratio`` are
        dropped while any other is on the way.

        Costs are None if nothing measured was reachable; the candidates then
        keep their straight-line order.
        """
        line = lambda p: p.distance_to(ref[0], ref[1])
        if role != "stop" or dest is None:
            ranked, cost = self._best_first(cands, line, lambda p: meter(ref, p.as_tuple()))
            return ranked, cost, None

        goal = dest.as_tuple()
        ranked, cost = self._best_first(
            cands,
            lambda p: line(p) + math.hypot(dest.x - p.x, dest.y - p.y),
            lambda p: meter(ref, p.as_tuple()) + meter(p.as_tuple(), goal))
        direct = meter(ref, goal)
        return self._on_the_way(ranked, cost, direct)

    def _rank_by_graph(self, cands: list[Place], ref: Sequence[float], role: str,
                       dest: Place | None
                       ) -> tuple[list[Place], list[float] | None, float | None]:
        """As :meth:`_rank_by_road`, but every candidate is scored by one
        search from ``ref`` (and, for a stop, one back from ``dest``), and only
        the ``route_top_k`` best are returned."""
        goals = [p.as_tuple() for p in cands]
        stop = role == "stop" and dest is not None
        fwd = self.routes.route_lengths(ref, goals + ([dest.as_tuple()] if stop else []))
        direct = fwd.pop() if stop else None
        cost = fwd
        if stop:
            cost = [a + b for a, b in
                    zip(fwd, self.routes.route_lengths_to(goals, dest.as_tuple()))]
        if all(math.isinf(c) for c in cost):
            return cands, None, None
        line = lambda i: cands[i].distance_to(ref[0], ref[1])
        order = sorted(range(len(cands)), key=lambda i: (cost[i], line(i)))
        ranked, cost = [cands[i] for i in order], [cost[i] for i in order]
        if stop:
            ranked, cost, direct = self._on_the_way(ranked, cost, direct)
        k = self.route_top_k if self.route_top_k > 0 else len(ranked)
        return ranked[:k], cost[:k], direct

    def _on_the_way(self, ranked: list[Place], cost: list[float] | None,
                    direct: float) -> tuple[list[Place], list[float] | None, float | None]:
        """Drop the stops that stretch the trip past ``detour_ratio``, unless
        none is left. ``direct`` comes back None if it is no use as a base."""
        if cost is None or not 1.0 < direct < math.inf:
            return ranked, cost, None
        keep = [i for i, c in enumerate(cost) if c / direct < self.detour_ratio]
        if keep and len(keep) < len(ranked):
            print(f"[mission] {len(keep)}/{len(ranked)} '{ranked[0].label}' "
                  f"measured are on the way")
            ranked, cost = [ranked[i] for i in keep], [cost[i] for i in keep]
        return ranked, cost, direct

    def _best_first(self, cands: list[Place], bound: Callable[[Place], float],
                    cost: Callable[[Place], float]
                    ) -> tuple[list[Place], list[float] | None]:
        """``cands`` by ``cost``, cheapest first, measuring as few as it can.

        ``bound`` is a straight-line lower bound on ``cost``. Candidates are
        measured in order of bound: the ``route_top_k`` lowest, then any whose
        bound is still below the cheapest cost so far -- nothing left
        unmeasured can be cheaper, so the cheapest is exact for any k. k only
        sets how many come back to choose from. Unreachable ones go last; if
        none of the first k is reachable, the search stops there and returns
        them in bound order with no costs.
        """
        by_bound = sorted(cands, key=bound)
        k = self.route_top_k if self.route_top_k > 0 else len(by_bound)
        measured: list[tuple[float, Place]] = []
        for i, p in enumerate(by_bound):
            if i >= k:
                best = min(c for c, _ in measured)
                if math.isinf(best) or bound(p) >= best:
                    break
            measured.append((cost(p), p))
        if all(math.isinf(c) for c, _ in measured):
            return by_bound[:k], None
        measured.sort(key=lambda t: t[0])           # stable: ties keep bound order
        return [p for _, p in measured], [c for c, _ in measured]

    def _on_the_way_by_line(self, label: str, cands: list[Place], ref_x: float,
                            ref_y: float, dest: Place) -> list[Place]:
        """Straight-line ranking: the stops that do not blow up the detour."""
        direct = math.hypot(dest.x - ref_x, dest.y - ref_y)
        if direct <= 1.0:
            return cands
        on_way = [p for p in cands
                  if (p.distance_to(ref_x, ref_y)
                      + math.hypot(dest.x - p.x, dest.y - p.y)) / direct
                  < self.detour_ratio]
        if not on_way:
            return cands
        if len(on_way) < len(cands):
            print(f"[mission] {len(on_way)}/{len(cands)} '{label}' are on the way")
        return on_way


def _directive_from(intent: dict, previous: Directive | None = None) -> Directive:
    prev = previous or Directive()
    return Directive(
        utterance=intent.get("_utterance", prev.utterance),
        urgency=intent.get("urgency", "normal"),
        preference=intent.get("route_preference", prev.preference),
        avoid=tuple(intent.get("avoid", ()) or ()),
    )
