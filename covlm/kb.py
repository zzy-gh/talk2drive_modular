"""
covlm/kb.py
===========
The knowledge base a CoVLM run sees: the real landmark table, plus the one
landmark per vehicle that sits on that vehicle's route end.

Why this exists
---------------
For the destination to come from the passenger's words, the words have to name
something the knowledge base can find, and finding it has to land on the route's
end point. Town05's real table holds 12 types -- bakery, cafe, cinema, hotel and
so on -- while the r1-r46 utterances are about hospitals, homes and schools. 73
of 175 named a type the town has no entry for and were rejected outright;
another 67 named nothing at all.

``covlm/design_passenger_inputs.py`` closes that by putting a landmark ON each
route end and writing an utterance that names it. This module is what serves
those landmarks to the mission layer and to the LLM prompt.

What is guaranteed, exactly
---------------------------
1. ``special_buildings_en.csv`` and ``LandmarkKnowledgeBase`` are never touched
   or mutated. Not appended to, not reordered.
2. An **unbound** view returns exactly what the wrapped knowledge base returns,
   lookup for lookup. Plain talk2drive cannot tell this class is in the way, so
   "the second museum" and the CLI's numbered menu are unchanged.
3. A **bound** view -- ``bind(scenario, veh)`` -- additionally exposes the
   landmarks of that one scenario. This is deliberate and is the whole point:
   without it the passenger's word cannot resolve. It is a *separate namespace*,
   entered only by asking for a scenario, and every row it adds carries
   ``source="anchor"`` so the two can always be told apart.

So the earlier promise "a bound view is identical too" no longer holds, and
should not: it was written when anchors bypassed the knowledge base entirely.
What still holds is the part that matters -- the original table's own numbering
never moves, and nothing outside a scenario binding can see an anchor.
``assert_covlm_kb_sane`` checks all three points above.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from core.types import Place


class CovlmKnowledgeBase:
    """``LandmarkKnowledgeBase`` + this scenario's route-end landmarks."""

    def __init__(self, kb, landmarks_csv: str | Path | None = None,
                 scenario: str | None = None, veh: int | None = None) -> None:
        self._kb = kb
        self.scenario = scenario
        self.veh = veh
        self._by_scenario: dict[str, list[dict]] = defaultdict(list)
        if landmarks_csv is not None:
            self._load(Path(landmarks_csv))

    def _load(self, path: Path) -> None:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._by_scenario[row["scenario"]].append({
                    "building_type": row["building_type"],
                    "carla_name": row["carla_name"],
                    "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
                    "town": row["town"], "scenario": row["scenario"],
                    "veh": int(row["veh"]), "source": "anchor",
                })

    # ── binding ─────────────────────────────
    def bind(self, scenario: str, veh: int | None = None) -> "CovlmKnowledgeBase":
        view = CovlmKnowledgeBase(self._kb, None, scenario, veh)
        view._by_scenario = self._by_scenario
        return view

    def _extra(self, town: str) -> list[dict]:
        """This scenario's landmarks, or nothing at all when unbound."""
        if self.scenario is None:
            return []
        return [r for r in self._by_scenario.get(self.scenario, []) if r["town"] == town]

    # ── the LandmarkKnowledgeBase surface ───
    def get_landmarks(self, town):
        return list(self._kb.get_landmarks(town)) + self._extra(town)

    def get_landmark_types(self, town):
        """Base types plus this scenario's. Feeds the LLM prompt and the
        parser's validation -- the reason 'hospital' stops being rejected."""
        return sorted(set(self._kb.get_landmark_types(town))
                      | {r["building_type"] for r in self._extra(town)})

    def find_all(self, town, building_type):
        base = list(self._kb.find_all(town, building_type))
        extra = [r for r in self._extra(town)
                 if r["building_type"].lower() == building_type.lower()]
        return base + extra

    def find_coordinate(self, town, building_type, index=0):
        cands = self.find_all(town, building_type)
        if not cands:
            return None
        return cands[index if 0 <= index < len(cands) else 0]

    def find_nearest_coordinate(self, town, building_type, ref_x, ref_y):
        cands = self.find_all(town, building_type)
        if not cands:
            return None
        return min(cands, key=lambda r: (r["x"] - ref_x) ** 2 + (r["y"] - ref_y) ** 2)

    @property
    def by_town(self):
        return self._kb.by_town

    # ── convenience ─────────────────────────
    def mission_landmark(self) -> dict | None:
        """The row this bound vehicle's utterance is supposed to resolve to."""
        if self.scenario is None or self.veh is None:
            return None
        for r in self._by_scenario.get(self.scenario, []):
            if r["veh"] == self.veh:
                return r
        return None

    def as_place(self, row: dict) -> Place:
        return Place(label=row["building_type"], x=row["x"], y=row["y"], z=row["z"],
                     name=row["carla_name"], index=None,
                     source=row.get("source", "kb"))


# ─────────────────────────────────────────────
# The guarantee, as a test
# ─────────────────────────────────────────────

def assert_covlm_kb_sane(kb, covlm: CovlmKnowledgeBase,
                         scenarios: Sequence[str] = (),
                         towns: Iterable[str] = ()) -> None:
    """Check the three points in this module's docstring. Raises on the first
    violation, so ``covlm/selftest.py`` can call it on the shipped data."""
    towns = list(towns) or sorted(kb.by_town.keys())

    snap = {}
    for town in towns:
        types = list(kb.get_landmark_types(town))
        snap[(town, "types")] = types
        snap[(town, "rows")] = [dict(r) for r in kb.get_landmarks(town)]
        for bt in types:
            snap[(town, bt)] = [dict(r) for r in kb.find_all(town, bt)]

    # (2) unbound is transparent
    for town in towns:
        if list(covlm.get_landmark_types(town)) != snap[(town, "types")]:
            raise AssertionError("%s: unbound view changed the type list" % town)
        if [dict(r) for r in covlm.get_landmarks(town)] != snap[(town, "rows")]:
            raise AssertionError("%s: unbound view changed the row order" % town)
        for bt in snap[(town, "types")]:
            if [dict(r) for r in covlm.find_all(town, bt)] != snap[(town, bt)]:
                raise AssertionError(
                    "%s/%s: unbound view changed the candidate list -- "
                    "'the second %s' would mean a different building" % (town, bt, bt))

    # (3) bound adds only this scenario, and every addition is tagged
    for scenario in (list(scenarios) or list(covlm._by_scenario)):
        view = covlm.bind(scenario)
        for town in towns:
            added = [r for r in view.get_landmarks(town) if r not in snap[(town, "rows")]]
            for r in added:
                if r.get("scenario") != scenario:
                    raise AssertionError("%s leaked a landmark from %s"
                                         % (scenario, r.get("scenario")))
                if r.get("source") != "anchor":
                    raise AssertionError("%s: added row is not tagged source='anchor': %r"
                                         % (scenario, r))
            for bt in snap[(town, "types")]:
                base = snap[(town, bt)]
                got = [dict(r) for r in view.find_all(town, bt)]
                if got[:len(base)] != base:
                    raise AssertionError(
                        "%s/%s/%s: the original candidates must stay first and "
                        "in order, so their 1-based ordinals do not move"
                        % (scenario, town, bt))

    # (1) the wrapped knowledge base was not mutated along the way
    for town in towns:
        if [dict(r) for r in kb.get_landmarks(town)] != snap[(town, "rows")]:
            raise AssertionError("%s: the underlying knowledge base was mutated" % town)
