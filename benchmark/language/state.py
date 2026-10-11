"""
benchmark/language/state.py
===========================
The route state a turn starts from and ends in, and the one function that
moves between them: ``apply_intent``.

Both the gold intent and a model's prediction are pushed through the real
``MissionPlanner.apply``, so "is the route right afterwards" is judged by the
same state machine that drives the car. Two intents that differ on paper but
leave the same route (e.g. inserting a stop "after the destination" vs.
extending the route with a new destination) score the same.

A state is plain JSON::

    {"destination": "cafe", "destination_index": 2,
     "stops": [{"type": "bakery", "index": null, "enroute": false}],
     "urgency": "normal", "preference": null, "avoid": []}
"""

from __future__ import annotations

import copy

from core.mission import MissionPlanner, Stop
from core.types import Directive

EMPTY: dict = {"destination": None, "destination_index": None, "stops": [],
               "urgency": "normal", "preference": None, "avoid": []}


def empty_state() -> dict:
    return copy.deepcopy(EMPTY)


def _planner(state: dict) -> MissionPlanner:
    # apply() never touches the knowledge base or the map.
    m = MissionPlanner(kb=None, route_provider=None)
    m.destination = state["destination"]
    m.destination_index = state["destination_index"]
    m.stops = [Stop(label=s["type"], order=i + 1, index=s.get("index"),
                    enroute=bool(s.get("enroute")))
               for i, s in enumerate(state["stops"])]
    m.directive = Directive(urgency=state.get("urgency", "normal"),
                            preference=state.get("preference"),
                            avoid=tuple(state.get("avoid") or ()))
    return m


def _snapshot(m: MissionPlanner) -> dict:
    return {"destination": m.destination,
            "destination_index": m.destination_index,
            "stops": [{"type": s.label, "index": s.index, "enroute": bool(s.enroute)}
                      for s in m.stops],
            "urgency": m.directive.urgency,
            "preference": m.directive.preference,
            "avoid": list(m.directive.avoid)}


def apply_intent(state: dict, intent: dict | None) -> dict:
    """The state after ``intent``. None (rejected / unparsed) changes nothing."""
    if not intent:
        return copy.deepcopy(state)
    m = _planner(state)
    try:
        result = m.apply(intent)
    except (KeyError, TypeError, AttributeError):   # malformed prediction
        return copy.deepcopy(state)
    if not result.ok:
        return copy.deepcopy(state)
    return _snapshot(m)


def route_summary(state: dict) -> dict:
    """What the parser is shown: MissionPlanner.route_summary() of ``state``."""
    return _planner(state).route_summary()


def route_key(state: dict) -> tuple:
    """The part of a state that decides where the car goes, normalised.

    An unordered ('enroute') stop only matters while other stops exist --
    with none, there is nothing to ask and every position is the same.
    """
    stops = state["stops"]
    fixed = [s for s in stops if not s.get("enroute")]
    lone = len(stops) == 1 and not fixed
    return (state["destination"], state["destination_index"],
            tuple((s["type"], s.get("index"), bool(s.get("enroute")) and not lone)
                  for s in stops))
