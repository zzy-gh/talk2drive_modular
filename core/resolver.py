"""
core/resolver.py
================
Disambiguation policy.

The original code called ``input()`` from inside route planning. That is fine
for a CLI demo and fatal anywhere else: inside a leaderboard agent's
``run_step`` it blocks the simulation tick, and inside an evaluation run there
is nobody to answer. So the question "which hospital?" becomes a *policy* the
caller injects.

Three policies ship here:

``AutoInteraction``     nearest candidate (by road when the mission layer
                        measured it), append enroute stops — headless,
                        never blocks. Use for evaluation.
``CliInteraction``      the original interactive prompts. Use for the demo.
``DeferredInteraction`` answers nothing, records the open question and lets
                        the car keep driving its current route; the host can
                        surface the question and re-plan once answered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .types import Place


@dataclass
class PlaceChoice:
    """'Which of these N places do you mean?'"""

    label: str                       # building_type
    role: str                        # "destination" | "stop"
    candidates: list[Place]
    ref_x: float
    ref_y: float
    # Cost of each candidate by road, aligned with ``candidates`` (which then
    # come cheapest first); None = rank by the straight line. The driving
    # distance from the reference for a destination; for a stop, the whole
    # trip reference -> stop -> destination when ``direct`` is set.
    distances: list[float] | None = None
    direct: float | None = None      # the trip to the destination without the stop
    total: int | None = None         # how many of this label the town has, all told

    def distance(self, i: int) -> float:
        if self.distances is not None:
            return self.distances[i]
        return self.candidates[i].distance_to(self.ref_x, self.ref_y)

    def nearest(self) -> Place:
        """The cheapest: nearest, or for a stop ranked by road, least detour."""
        return self.candidates[min(range(len(self.candidates)), key=self.distance)]


@dataclass
class InsertChoice:
    """'Where in the current route should this new stop go?'"""

    label: str                       # the stop being inserted
    stop_labels: list[str]           # existing stops, in order
    destination_label: str


class Interaction(ABC):
    """Everything the mission layer might need to ask a human."""

    @abstractmethod
    def choose_place(self, req: PlaceChoice) -> Place | None:
        ...

    @abstractmethod
    def choose_insert_position(self, req: InsertChoice) -> int:
        ...

    @property
    def blocking(self) -> bool:
        """True if a call may wait on a human. Hosts that own a real-time
        loop must refuse to run a blocking policy on the control thread."""
        return False


# ─────────────────────────────────────────────
# Headless
# ─────────────────────────────────────────────

class AutoInteraction(Interaction):
    def choose_place(self, req: PlaceChoice) -> Place | None:
        if not req.candidates:
            return None
        return req.nearest()

    def choose_insert_position(self, req: InsertChoice) -> int:
        return len(req.stop_labels)          # append just before the destination


# ─────────────────────────────────────────────
# Deferred
# ─────────────────────────────────────────────

class DeferredInteraction(Interaction):
    """Never answers; queues the question for the host to resolve later."""

    def __init__(self) -> None:
        self.pending: list[PlaceChoice | InsertChoice] = []

    def choose_place(self, req: PlaceChoice) -> Place | None:
        if len(req.candidates) == 1:
            return req.candidates[0]
        self.pending.append(req)
        return None

    def choose_insert_position(self, req: InsertChoice) -> int:
        self.pending.append(req)
        return len(req.stop_labels)

    def take_pending(self):
        out, self.pending = self.pending, []
        return out


# ─────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────

class CliInteraction(Interaction):
    """
    The demo's numbered prompts. ``preview`` / ``clear_preview`` are optional
    hooks so the runtime can draw the numbered markers in CARLA without this
    module importing carla.
    """

    def __init__(self,
                 preview: Callable[[list[Place]], None] | None = None,
                 clear_preview: Callable[[], None] | None = None) -> None:
        self.preview = preview
        self.clear_preview = clear_preview

    @property
    def blocking(self) -> bool:
        return True

    def choose_place(self, req: PlaceChoice) -> Place | None:
        cands = req.candidates
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        # Listed nearest first, each under the ID the map shows (and the
        # parser's index means), so the passenger answers with that one
        # number whether typing here or speaking.
        order = sorted(range(len(cands)), key=req.distance)
        ids = ([cands[i].index for i in order] if all(p.index for p in cands)
               else list(range(1, len(cands) + 1)))
        if req.distances is None:
            first, cost = "nearest", lambda i: f"{req.distance(i):5.0f} m away"
        elif req.direct is None:
            first, cost = "nearest", lambda i: f"{req.distance(i):5.0f} m by road"
        else:
            first, cost = "least detour", lambda i: f"+{req.distance(i) - req.direct:4.0f} m detour"
        print(f"\n  Which {req.label} for the {req.role}?  ({first} first)")
        for n, i in zip(ids, order):
            print(f"    [{n}] {req.label}  {cost(i)}")
        if req.total and req.total > len(cands):
            print(f"    ({len(cands)} best of {req.total} shown)")
        if self.preview:
            self.preview([cands[i] for i in order])

        try:
            pick = _prompt_id(f"  Enter {req.label} # (Enter = {first}): ", ids)
        finally:
            if self.clear_preview:
                self.clear_preview()

        chosen = req.nearest() if pick is None else cands[order[ids.index(pick)]]
        print(f"  -> Selected: {chosen.tag}")
        return chosen

    def choose_insert_position(self, req: InsertChoice) -> int:
        n = len(req.stop_labels)
        if n == 0:
            return 0
        print(f"\n  Where to insert '{req.label}'?")
        for i in range(n + 1):
            chain = (["START"] + req.stop_labels[:i] + [f"[{req.label}]"]
                     + req.stop_labels[i:] + [req.destination_label])
            print(f"  [{i + 1}] {' -> '.join(chain)}")
        idx = _prompt_index(f"  Enter 1-{n + 1}: ", n + 1, allow_empty=False)
        return idx or 0


def _prompt_id(prompt: str, ids: list[int]) -> int | None:
    """One of ``ids``, or None on an empty answer."""
    while True:
        try:
            raw = input(prompt).strip().lstrip("#")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return None
        if raw.isdigit() and int(raw) in ids:
            return int(raw)
        print(f"  Please enter one of: {', '.join(f'#{i}' for i in ids)}")


def _prompt_index(prompt: str, n: int, allow_empty: bool) -> int | None:
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            if allow_empty:
                return None
            print(f"  Please enter a number between 1 and {n}")
            continue
        try:
            v = int(raw)
        except ValueError:
            print(f"  Please enter a number between 1 and {n}")
            continue
        if 1 <= v <= n:
            return v - 1
        print(f"  Please enter a number between 1 and {n}")
