#!/usr/bin/env python3
"""
covlm/selftest.py
=================
Regression checks for the CoVLM / InterDrive subpackage.

Self-contained on purpose: it shares no fixtures with ``tools/selftest.py``, so
the general package's checks never depend on this benchmark's, and deleting
``covlm/`` cannot break them. Needs no CARLA server, no GPU, no LLM.

    python covlm/selftest.py

The route-reproduction sweep is a separate tool -- it needs a carla module for
``carla.Map`` -- see ``covlm/verify_nav_routes.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.types import Place                                                  # noqa: E402

from covlm.anchors import (AnchorBook, RouteAnchor, ScenarioKB,               # noqa: E402
                           assert_kb_untouched)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


class FakeKB:
    """Two hospitals, one museum, three cafes -- enough to have an ordinal."""

    DATA = {
        "hospital":    [(100.0, 0.0), (300.0, 0.0)],
        "museum":      [(200.0, 40.0)],
        "cafe":        [(50.0, 0.0), (150.0, 0.0), (900.0, 900.0)],
        "gas_station": [(80.0, 10.0)],
    }

    def find_all(self, town, label):
        return [{"building_type": label, "carla_name": f"{label}_{i}",
                 "x": x, "y": y, "z": 0.0}
                for i, (x, y) in enumerate(self.DATA.get(label, []))]

    def get_landmark_types(self, town):
        return sorted(self.DATA)

    def get_landmarks(self, town):
        return [row for label in sorted(self.DATA) for row in self.find_all(town, label)]

    def find_coordinate(self, town, label, index=0):
        cands = self.find_all(town, label)
        if not cands:
            return None
        return cands[index if 0 <= index < len(cands) else 0]

    def find_nearest_coordinate(self, town, label, ref_x, ref_y):
        cands = self.find_all(town, label)
        if not cands:
            return None
        return min(cands, key=lambda r: math.hypot(r["x"] - ref_x, r["y"] - ref_y))

    @property
    def by_town(self):
        return {"Town01": self.get_landmarks("Town01")}


def _book() -> AnchorBook:
    return AnchorBook([
        RouteAnchor(scenario="r11_town05_ins_sl", veh=1, town="Town01",
                    start=(0.0, 0.0, 0.0), vias=((10.0, 0.0, 0.0),),
                    goal=(20.0, 0.0, 0.0), label="cafe",
                    label_source="nearest_kb", utterance="just grabbing coffee"),
        RouteAnchor(scenario="r11_town05_ins_sl", veh=2, town="Town01",
                    start=(0.0, 5.0, 0.0), vias=(), goal=(30.0, 5.0, 0.0),
                    label="cafe", label_source="utterance", utterance="coffee run"),
    ])


# ─────────────────────────────────────────────
# The index namespaces stay apart
# ─────────────────────────────────────────────

def test_anchor_index_isolation():
    print("\n[anchors] anchors never disturb a KB index")
    kb = FakeKB()
    before = {bt: [dict(r) for r in kb.find_all("Town01", bt)]
              for bt in kb.get_landmark_types("Town01")}
    n_cafes = len(before.get("cafe", []))
    book = _book()

    view = ScenarioKB(kb, book).bind("r11_town05_ins_sl", 1)
    after = {bt: [dict(r) for r in view.find_all("Town01", bt)]
             for bt in view.get_landmark_types("Town01")}
    check("candidate lists are byte-identical through ScenarioKB", after == before)
    check(f"'cafe' still has {n_cafes} candidates, so 'the second cafe' "
          f"still means the same building",
          len(view.find_all("Town01", "cafe")) == n_cafes)
    check("landmark types unchanged, so the LLM prompt is unchanged",
          view.get_landmark_types("Town01") == kb.get_landmark_types("Town01"))

    try:
        assert_kb_untouched(kb, book, towns=["Town01"])
        ok, why = True, ""
    except AssertionError as exc:
        ok, why = False, str(exc)
    check("assert_kb_untouched passes on all namespaces", ok, why)

    a = book.get("r11_town05_ins_sl", 1)
    place = a.as_place()
    check("anchor Place is tagged source='anchor'", place.source == "anchor")
    check("anchor Place carries no KB ordinal", place.index is None)
    check("anchor id is a key, not a number", a.anchor_id == "r11_town05_ins_sl#veh1")
    check("route_id identifies the scenario, not a landmark", a.route_id == 11)
    check("a KB Place still defaults to source='kb'",
          Place(label="cafe", x=0.0, y=0.0).source == "kb")
    check("keypoints are start + vias + goal, in order",
          a.keypoints == [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)])


# ─────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────

def test_anchor_roundtrip():
    print("\n[anchors] json round-trip")
    import tempfile

    book = _book()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "anchors.json"
        book.save(path, {"note": "test"})
        back = AnchorBook.load(path)
    check("every anchor survives a save/load", len(back) == len(book))
    check("fields survive verbatim",
          [a.to_dict() for a in back] == [a.to_dict() for a in book])
    check("tuples come back as tuples, not lists",
          isinstance(back.get("r11_town05_ins_sl", 1).goal, tuple))
    check("a duplicate (scenario, veh) is rejected",
          _rejects_duplicate())


def _rejects_duplicate() -> bool:
    b = _book()
    try:
        b.add(b.get("r11_town05_ins_sl", 1))
        return False
    except ValueError:
        return True


# ─────────────────────────────────────────────
# The shipped artefact still describes 46 scenarios
# ─────────────────────────────────────────────

def test_shipped_anchors():
    print("\n[anchors] data/route_anchors.json")
    from covlm.paths import ROUTE_ANCHORS

    if not ROUTE_ANCHORS.is_file():
        check("route_anchors.json is present", False, str(ROUTE_ANCHORS))
        return
    book = AnchorBook.load(ROUTE_ANCHORS)
    check("46 InterDrive scenarios", len(book.scenarios()) == 46,
          f"got {len(book.scenarios())}")
    check("175 vehicle routes", len(book) == 175, f"got {len(book)}")
    check("every anchor has at least a start and a goal",
          all(len(a.keypoints) >= 2 for a in book))
    check("no anchor Place carries a KB ordinal",
          all(a.as_place().index is None for a in book))
    check("every anchor Place is tagged source='anchor'",
          all(a.as_place().source == "anchor" for a in book))
    towns = sorted({a.town for a in book})
    check("towns are Town05/06/07", towns == ["Town05", "Town06", "Town07"], str(towns))


def main() -> int:
    print("covlm self-test (no CARLA server required)")
    for fn in (test_anchor_index_isolation, test_anchor_roundtrip, test_shipped_anchors):
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
