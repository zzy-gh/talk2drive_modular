#!/usr/bin/env python3
"""
covlm/build_route_anchors.py
============================
Compile InterDrive's route files into ``covlm/data/route_anchors.json``.

Pure parsing -- no CARLA, no server, no LLM, no network. Deterministic, so the
output is a reviewable artefact rather than something regenerated per run.

For every ``r<N>_<town>_<tag>/..._<k>.xml`` under the InterDrive data
directory it emits one :class:`RouteAnchor`:

    start = first keypoint,  vias = middle keypoints,  goal = last keypoint

and a cosmetic ``label``, taken from the passenger utterance when that
utterance names a destination, otherwise from the nearest real landmark. The
label never sets a coordinate -- see ``covlm/anchors.py``.

About a third of the r1-r46 utterances deliberately name no destination at all
("just cruising, no particular plan"): they were written to carry negotiation
*priority*, not navigation. Those come out ``label_source="nearest_kb"``, and
that is fine -- the goal coordinate comes from the route either way.

    python covlm/build_route_anchors.py [--limit 46] [-o covlm/data/route_anchors.json]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.landmark_kb import LandmarkKnowledgeBase                             # noqa: E402
from core.paths import KB_CSV, load_simple_yaml                               # noqa: E402

from covlm.anchors import (SCENARIO_RE, AnchorBook, RouteAnchor,              # noqa: E402
                           assert_kb_untouched)
from covlm.paths import DRIVER_INTENTS, INTERDRIVE_DIR, ROUTE_ANCHORS         # noqa: E402

DEF_INTERDRIVE = INTERDRIVE_DIR
DEF_INTENTS = DRIVER_INTENTS
DEF_CSV = None                  # falls back to core.paths.KB_CSV
DEF_OUT = ROUTE_ANCHORS

VEH_RE = re.compile(r"_(\d+)\.xml$")

# Utterance phrase -> BuildingType. Most specific first; first hit wins.
# Only phrases that actually name a place are listed -- "running late for a
# meeting" names no place and is meant to fall through.
LABEL_RULES: list[tuple[str, str]] = [
    (r"\b(er|emergency room|urgent care|hospital|ambulance)\b", "hospital"),
    (r"\b(surgery|in labor|labor|pediatrician|dialysis|clinic|doctor'?s appointment)\b", "hospital"),
    (r"\b(emergency vet|the vet|vet just called)\b", "hospital"),
    (r"\b(pharmacy|prescription|insulin|inhaler|medication|glucose)\b", "pharmacy"),
    (r"\b(school|parent-teacher|daycare|soccer practice|camp)\b", "school"),
    (r"\b(airport|flight|boarding|check-?in before|connecting flight)\b", "airport"),
    (r"\b(hotel|check-?in)\b", "hotel"),
    (r"\b(train|ferry)\b", "train_station"),
    (r"\b(bus stop|the bus)\b", "bus_stop"),
    (r"\b(groceries|grocery)\b", "supermarket"),
    (r"\b(bookstore|the shop|open up the shop|store|sale|mall|dry cleaning|package)\b", "store"),
    (r"\b(library|library book)\b", "library"),
    (r"\b(movie|showing|cinema|recital)\b", "cinema"),
    (r"\b(gym)\b", "gym"),
    (r"\b(coffee|cafe|café)\b", "cafe"),
    (r"\b(lunch|dinner|restaurant)\b", "restaurant"),
    (r"\b(park|walk the dog)\b", "park"),
    (r"\b(parking|parking meter|tow truck|car wash|repair shop|rental car|the shop before it closes)\b", "parking"),
    (r"\b(gas|propane|furnace|fuel)\b", "gas_station"),
    (r"\b(apartment)\b", "apartment"),
    (r"\b(home|my house|neighbou?r'?s house|friend'?s place|get to her|get to him)\b", "house"),
    (r"\b(job site|work shift|my shift|clocking in|job orientation|interview)\b", "factory"),
    (r"\b(dmv|post office)\b", "post_office"),
]
COMPILED = [(re.compile(p, re.I), bt) for p, bt in LABEL_RULES]


def label_from_utterance(utterance: str) -> str | None:
    for rx, bt in COMPILED:
        if rx.search(utterance or ""):
            return bt
    return None


def load_intents(path: Path | None, required: bool) -> dict[str, dict[str, str]]:
    """Read driver_intents.yaml, or fail loudly.

    An earlier version swallowed a missing pyyaml and returned ``{}``, which
    silently rebuilt route_anchors.json with every label downgraded to
    nearest_kb -- 175 anchors quietly worse than the file it overwrote. The
    labels are cosmetic, but an artefact that changes depending on which
    interpreter regenerated it is not. So: parse it with
    ``core.paths.load_simple_yaml`` (pyyaml when present, the built-in parser
    when not -- it handles this file exactly), and refuse to write unless
    ``--no-intents`` says the downgrade is intended.
    """
    if path is None or not path.exists():
        if required:
            raise SystemExit(
                f"[build] driver_intents not found ({path}).\n"
                f"         Set [paths] driver_intents in config.yaml, or pass "
                f"--no-intents to build anchors with nearest-landmark labels only.")
        print("[build] no driver_intents; labelling from the nearest landmark only")
        return {}

    data = load_simple_yaml(path) or {}
    if not data and required:
        raise SystemExit(f"[build] {path} parsed to nothing -- refusing to "
                         f"overwrite the anchors with unlabelled ones.")
    return data


def parse_route_xml(path: Path):
    route = ET.parse(path).getroot().find("route")
    town = route.get("town")
    wps = [(float(w.get("x")), float(w.get("y")), float(w.get("z", 0.0)))
           for w in route.findall("waypoint")]
    return town, wps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interdrive", type=Path, default=DEF_INTERDRIVE)
    ap.add_argument("--intents", type=Path, default=DEF_INTENTS)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=DEF_OUT)
    ap.add_argument("--no-intents", action="store_true",
                    help="build without driver_intents.yaml; every label then "
                         "comes from the nearest landmark. Off by default so a "
                         "missing file fails instead of silently downgrading "
                         "the committed anchors.")
    ap.add_argument("--limit", type=int, default=46,
                    help="highest r-number to include (default 46 = the CoVLM set)")
    args = ap.parse_args()

    csv_path = args.csv or KB_CSV
    kb = LandmarkKnowledgeBase(str(csv_path))
    intents = load_intents(args.intents, required=not args.no_intents)

    book = AnchorBook()
    skipped: list[str] = []
    for scen_dir in sorted(d for d in args.interdrive.iterdir() if d.is_dir()):
        m = SCENARIO_RE.match(scen_dir.name)
        if not m or int(m.group(1)) > args.limit:
            continue
        scenario = scen_dir.name
        scen_intents = intents.get(scenario, {}) or {}

        for xml in sorted(scen_dir.glob("*.xml")):
            vm = VEH_RE.search(xml.name)
            if not vm:
                skipped.append(f"{xml.name}: no _<veh>.xml suffix")
                continue
            veh = int(vm.group(1))
            town, wps = parse_route_xml(xml)
            if len(wps) < 2:
                skipped.append(f"{xml.name}: {len(wps)} waypoint(s)")
                continue

            goal = wps[-1]
            near, near_d = None, -1.0
            for row in kb.get_landmarks(town):
                d = math.hypot(row["x"] - goal[0], row["y"] - goal[1])
                if near is None or d < near_d:
                    near, near_d = row, d

            utterance = scen_intents.get(f"veh_{veh}", "") or ""
            lab = label_from_utterance(utterance)
            if lab:
                src = "utterance"
            elif near is not None:
                lab, src = near["building_type"], "nearest_kb"
            else:
                lab, src = "store", "unlabelled"

            book.add(RouteAnchor(
                scenario=scenario, veh=veh, town=town,
                start=wps[0], vias=tuple(wps[1:-1]), goal=goal,
                label=lab, label_source=src, utterance=utterance,
                nearest_kb_label=(near or {}).get("building_type", ""),
                nearest_kb_m=round(near_d, 2) if near is not None else -1.0,
            ))

    # The promise this whole design rests on.
    assert_kb_untouched(kb, book)

    meta = {"interdrive": str(args.interdrive), "intents": str(args.intents),
            "csv": str(csv_path), "limit": args.limit,
            "intents_used": bool(intents),
            "scenarios": len(book.scenarios()), "vehicles": len(book)}
    book.save(args.out, meta)

    by_town = Counter(a.town for a in book)
    by_src = Counter(a.label_source for a in book)
    by_lab = Counter(a.label for a in book)
    vehs = defaultdict(int)
    for a in book:
        vehs[a.scenario] += 1
    d_all = sorted(a.nearest_kb_m for a in book if a.nearest_kb_m >= 0)

    print(f"\n[build] {len(book.scenarios())} scenarios, {len(book)} vehicle routes "
          f"-> {args.out}")
    print(f"[build] towns            : {dict(by_town)}")
    print(f"[build] vehicles/scenario: {dict(Counter(vehs.values()))}")
    print(f"[build] label source     : {dict(by_src)}")
    print(f"[build] labels           : {dict(by_lab.most_common())}")
    if d_all:
        print(f"[build] goal -> nearest real landmark: min {d_all[0]:.1f} m, "
              f"median {d_all[len(d_all)//2]:.1f} m, max {d_all[-1]:.1f} m "
              f"({sum(1 for d in d_all if d < 20)}/{len(d_all)} within 20 m)")
    print("[build] KB index namespaces verified untouched (assert_kb_untouched)")
    if skipped:
        print(f"[build] skipped {len(skipped)}: " + "; ".join(skipped[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
