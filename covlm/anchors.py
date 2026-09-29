"""
covlm/anchors.py
================
Route anchors for the CoVLM / InterDrive benchmark.

Why this module exists
----------------------
InterDrive's r1-r46 are *short* junction manoeuvres: 15-172 m, median 40 m.
Their end waypoints sit a median of 36 m from the nearest real landmark, and
only 58 of the 175 vehicle routes end within 20 m of one. So a passenger
utterance alone ("take me to the supermarket") can never reproduce them --
the planner would sail straight through the junction and keep going.

An *anchor* fixes that by taking the destination straight off the route
geometry: the last keypoint of the vehicle's route file is the mission goal,
the intermediate keypoints are via points. Since the leaderboard itself builds
its reference route with ``interpolate_trajectory`` (GlobalRoutePlanner over
exactly those keypoints), tracing start -> vias -> goal reproduces it.

The label ("bus_stop", "hospital", ...) is COSMETIC. It exists so the mission
log and the passenger phrasing read naturally. It never determines a
coordinate. Read ``RouteAnchor.label_source`` before trusting it for anything.

The four index namespaces -- keep them apart
--------------------------------------------
This is the one thing that must not get muddled, so it is spelled out:

1. ``kb.by_town[town]`` row order          -- 0-based, positional, the CSV's
   own order. ``find_coordinate(town, type, index)`` indexes into it.
2. ``MissionPlanner._candidates(label)``   -- 1-based passenger ordinal, the
   "second museum" of ``destination_index`` / ``Stop.index``. It is namespace
   1 filtered by building_type, so appending ANY row to the KB silently
   renumbers it.
3. ``CliInteraction`` menu numbers ``[1..N]`` -- the display order of
   namespace 2. Shifting 2 shifts what the human sees next to each number.
4. ``RouteAnchor.anchor_id`` ("r11_town05_ins_sl#veh1") -- this module. Keyed
   by (scenario, vehicle). NOT a number, NOT an ordinal, NOT comparable to
   1-3. It lives in ``covlm/`` precisely so InterDrive's vocabulary stays out
   of namespaces 1-3, which are talk2drive's.

So anchors are **never inserted into the knowledge base**. ``ScenarioKB``
below delegates every KB call through untouched and exposes anchors only via
``anchor_place()``, which returns a ``Place`` with ``source="anchor"`` and
``index=None``. A mission binds that Place directly as its destination and
skips candidate resolution entirely, so namespaces 1-3 cannot move.
``assert_kb_untouched()`` turns that promise into a test.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from core.types import Place

XYZ = tuple[float, float, float]

SCENARIO_RE = re.compile(r"^r(\d+)_(town\w+?)_(.+)$")


# ─────────────────────────────────────────────
# One anchor
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class RouteAnchor:
    """The mission a single vehicle in a single InterDrive scenario is on."""

    scenario: str                  # "r11_town05_ins_sl"
    veh: int                       # route-file suffix: ..._1.xml -> 1
    town: str                      # "Town05"
    start: XYZ                     # first keypoint of the route file
    vias: tuple[XYZ, ...]          # intermediate keypoints, in order
    goal: XYZ                      # last keypoint == the mission destination

    label: str                     # cosmetic building_type, see module docstring
    label_source: str              # "utterance" | "nearest_kb" | "unlabelled"
    utterance: str = ""            # the passenger line from driver_intents.yaml
    nearest_kb_label: str = ""     # diagnostics: closest real landmark ...
    nearest_kb_m: float = -1.0     # ... and how far it is from `goal`

    # ── identity ────────────────────────────
    @property
    def anchor_id(self) -> str:
        """Namespace 4. Never an integer; never compared to a KB index."""
        return f"{self.scenario}#veh{self.veh}"

    @property
    def route_id(self) -> int:
        """The r-number, e.g. 11. Identifies the SCENARIO, not a landmark."""
        m = SCENARIO_RE.match(self.scenario)
        return int(m.group(1)) if m else -1

    # ── geometry ────────────────────────────
    @property
    def keypoints(self) -> list[XYZ]:
        """Exactly what the route xml holds -- what the leaderboard feeds
        ``interpolate_trajectory``."""
        return [self.start, *self.vias, self.goal]

    # ── mission handoff ─────────────────────
    def as_place(self) -> Place:
        """The destination, as the mission layer wants it.

        ``index=None`` and ``source="anchor"`` are load-bearing: they say this
        Place did not come from a KB candidate list, so no 1-based ordinal
        applies to it.
        """
        return Place(label=self.label, x=self.goal[0], y=self.goal[1], z=self.goal[2],
                     name=f"anchor::{self.anchor_id}", index=None, source="anchor")

    def via_places(self) -> list[Place]:
        return [Place(label=f"{self.label}_via{i}", x=v[0], y=v[1], z=v[2],
                      name=f"anchor::{self.anchor_id}::via{i}", index=None, source="anchor")
                for i, v in enumerate(self.vias)]

    # ── serialisation ───────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["vias"] = [list(v) for v in self.vias]
        d["start"] = list(self.start)
        d["goal"] = list(self.goal)
        d["anchor_id"] = self.anchor_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RouteAnchor":
        return cls(scenario=d["scenario"], veh=int(d["veh"]), town=d["town"],
                   start=tuple(d["start"]), vias=tuple(tuple(v) for v in d["vias"]),
                   goal=tuple(d["goal"]), label=d["label"],
                   label_source=d["label_source"], utterance=d.get("utterance", ""),
                   nearest_kb_label=d.get("nearest_kb_label", ""),
                   nearest_kb_m=float(d.get("nearest_kb_m", -1.0)))


# ─────────────────────────────────────────────
# The book of anchors
# ─────────────────────────────────────────────

class AnchorBook:
    """All anchors, keyed by ``(scenario, veh)``."""

    def __init__(self, anchors: Sequence[RouteAnchor] = ()) -> None:
        self._by_key: dict[tuple[str, int], RouteAnchor] = {}
        for a in anchors:
            self.add(a)

    def add(self, a: RouteAnchor) -> None:
        key = (a.scenario, a.veh)
        if key in self._by_key:
            raise ValueError(f"duplicate anchor {a.anchor_id}")
        self._by_key[key] = a

    def get(self, scenario: str, veh: int) -> RouteAnchor | None:
        return self._by_key.get((scenario, int(veh)))

    def for_scenario(self, scenario: str) -> list[RouteAnchor]:
        return sorted((a for a in self._by_key.values() if a.scenario == scenario),
                      key=lambda a: a.veh)

    def scenarios(self) -> list[str]:
        return sorted({a.scenario for a in self._by_key.values()},
                      key=lambda s: (int(SCENARIO_RE.match(s).group(1))
                                     if SCENARIO_RE.match(s) else 1 << 30, s))

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[RouteAnchor]:
        return iter(sorted(self._by_key.values(), key=lambda a: (a.route_id, a.scenario, a.veh)))

    # ── io ──────────────────────────────────
    def save(self, path: str | Path, meta: dict | None = None) -> None:
        payload = {"meta": meta or {}, "anchors": [a.to_dict() for a in self]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AnchorBook":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([RouteAnchor.from_dict(d) for d in payload["anchors"]])


# ─────────────────────────────────────────────
# KB + anchors, without mixing them
# ─────────────────────────────────────────────

class ScenarioKB:
    """A ``LandmarkKnowledgeBase`` plus an anchor book, side by side.

    Every knowledge-base method delegates straight through to the wrapped KB
    and returns exactly what it would have returned on its own. Anchors reach
    the mission layer only through :meth:`anchor_place`. Nothing is appended,
    reordered or shadowed, so KB index namespaces 1-3 are bit-for-bit
    unchanged whether or not anchors are loaded.
    """

    def __init__(self, kb, book: AnchorBook | None = None,
                 scenario: str | None = None, veh: int | None = None) -> None:
        self._kb = kb
        self.book = book or AnchorBook()
        self.scenario = scenario
        self.veh = veh

    # ── pure delegation: do not "improve" these ──
    def get_landmarks(self, town):                 return self._kb.get_landmarks(town)
    def get_landmark_types(self, town):            return self._kb.get_landmark_types(town)
    def find_coordinate(self, town, bt, index=0):  return self._kb.find_coordinate(town, bt, index)
    def find_all(self, town, bt):                  return self._kb.find_all(town, bt)
    def find_nearest_coordinate(self, town, bt, x, y):
        return self._kb.find_nearest_coordinate(town, bt, x, y)

    @property
    def by_town(self):
        return self._kb.by_town

    # ── the anchor side ──────────────────────
    def bind(self, scenario: str, veh: int) -> "ScenarioKB":
        """Return a view bound to one vehicle of one scenario."""
        return ScenarioKB(self._kb, self.book, scenario, int(veh))

    def anchor(self) -> RouteAnchor | None:
        if self.scenario is None or self.veh is None:
            return None
        return self.book.get(self.scenario, self.veh)

    def anchor_place(self) -> Place | None:
        a = self.anchor()
        return a.as_place() if a else None


def assert_kb_untouched(kb, book: AnchorBook, towns: Sequence[str] = ()) -> None:
    """Prove anchors did not disturb any KB index namespace.

    Snapshots every KB lookup a mission can make, wraps the KB in a
    ``ScenarioKB`` bound to each anchor in turn, and re-runs them. Raises on
    the first difference. Cheap enough to run in ``tools/selftest.py``.
    """
    towns = list(towns) or sorted(kb.by_town.keys())
    before = {}
    for town in towns:
        types = list(kb.get_landmark_types(town))
        before[(town, "__types__")] = types
        before[(town, "__rows__")] = [dict(r) for r in kb.get_landmarks(town)]
        for bt in types:
            before[(town, bt)] = [dict(r) for r in kb.find_all(town, bt)]
            for i in range(len(before[(town, bt)])):
                before[(town, bt, i)] = dict(kb.find_coordinate(town, bt, i))

    wrapped = ScenarioKB(kb, book)
    views = [wrapped] + [wrapped.bind(a.scenario, a.veh) for a in book]
    for view in views:
        for town in towns:
            types = list(view.get_landmark_types(town))
            if types != before[(town, "__types__")]:
                raise AssertionError(f"{town}: landmark types changed -> LLM prompt would change")
            rows = [dict(r) for r in view.get_landmarks(town)]
            if rows != before[(town, "__rows__")]:
                raise AssertionError(f"{town}: namespace 1 (by_town row order) changed")
            for bt in types:
                cands = [dict(r) for r in view.find_all(town, bt)]
                if cands != before[(town, bt)]:
                    raise AssertionError(
                        f"{town}/{bt}: namespace 2 changed ({len(before[(town, bt)])} "
                        f"-> {len(cands)} candidates) -- 'the second {bt}' now means "
                        f"a different building, and the CLI menu renumbered with it")
                for i in range(len(cands)):
                    if dict(view.find_coordinate(town, bt, i)) != before[(town, bt, i)]:
                        raise AssertionError(f"{town}/{bt}[{i}]: namespace 1 lookup changed")

    for a in book:
        p = a.as_place()
        if p.source != "anchor" or p.index is not None:
            raise AssertionError(f"{a.anchor_id}: anchor Place must carry source='anchor', index=None")
