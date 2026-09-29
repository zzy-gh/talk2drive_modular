#!/usr/bin/env python3
"""
covlm/compute_min_vias.py
=========================
For every vehicle whose route needs more than a start and a goal, find the
SMALLEST subset of the route file's intermediate keypoints that still
reproduces the benchmark route.

Why smallest: each surviving keypoint has to become something the passenger
says -- "stop at the pharmacy on the way" -- so every one that is not load
bearing is a clause nobody would utter. Most turn out not to be: of the 17
routes that need any, 10 need exactly one of theirs.

The result also splits the 17 into two kinds, which need different handling:

  a stop        1-2 keypoints. A passenger can say this, so it becomes a real
                waypoints:[Stop] in the intent, resolved through the knowledge
                base like the destination.
  a manoeuvre   3+ keypoints. Look at r28_town06_hw_c#veh0: y runs
                237.1 -> 244.6 -> 240.8 -> 244.6 -> 248.4 -> 252.2, weaving
                across a multi-lane highway. That is lane-change choreography,
                not a list of errands, and no wording makes it one.

Exhaustive over subsets, smallest first, so the answer is minimal rather than
merely small. 7 keypoints is 128 subsets -- cheap next to building the map.

    python covlm/compute_min_vias.py [--report <chain report>] [-o data/min_vias.json]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.route_provider import CarlaRouteProvider                           # noqa: E402

from covlm.paths import DATA_DIR, OPENDRIVE_DIR                              # noqa: E402

STOP_LIMIT = 2          # at most this many keypoints can be spoken as stops


def max_dev(ra, rb) -> float:
    if not ra or not rb:
        return float("inf")

    def one(p, q):
        return max(min(math.hypot(a.x - b.x, a.y - b.y) for b in q) for a in p)

    return max(one(ra, rb), one(rb, ra))


def options(route):
    out = []
    for p in route:
        if not out or out[-1] != p.option:
            out.append(p.option)
    return out


def minimal_subset(prov, start, vias, goal, tol):
    """Smallest index subset of ``vias`` whose route matches the full one."""
    ref = prov.trace_chain([start] + list(vias) + [goal])
    if not ref:
        return None, ref
    for k in range(len(vias) + 1):
        for sub in itertools.combinations(range(len(vias)), k):
            cand = prov.trace_chain([start] + [vias[i] for i in sub] + [goal])
            if cand and max_dev(ref, cand) <= tol and options(cand) == options(ref):
                return list(sub), ref
    return None, ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", type=Path, default=DATA_DIR / "passenger_missions.json")
    ap.add_argument("--report", type=Path, default=DATA_DIR / "passenger_chain_report.csv",
                    help="chain report; only vehicles whose route did not match "
                         "are examined. Omit to examine every vehicle.")
    ap.add_argument("--xodr", type=Path, default=OPENDRIVE_DIR)
    ap.add_argument("-o", "--out", type=Path, default=DATA_DIR / "min_vias.json")
    ap.add_argument("--tol", type=float, default=1.1)
    ap.add_argument("--stop-limit", type=int, default=STOP_LIMIT)
    args = ap.parse_args()

    import carla

    missions = json.loads(args.missions.read_text(encoding="utf-8"))["missions"]
    if args.report.is_file():
        with args.report.open(newline="", encoding="utf-8") as f:
            wanted = {r["anchor_id"] for r in csv.DictReader(f) if r.get("route_ok") != "1"}
        missions = {k: v for k, v in missions.items() if k in wanted}
    print("[vias] examining %d vehicle(s)" % len(missions))

    provs = {}
    out = {}
    for aid, m in sorted(missions.items()):
        town = m["town"]
        if town not in provs:
            provs[town] = CarlaRouteProvider(
                carla.Map(town, (args.xodr / (town + ".xodr")).read_text(encoding="utf-8")), 1.0)
        vias = [tuple(v) for v in m["vias"]]
        sub, ref = minimal_subset(provs[town], tuple(m["start"]), vias, tuple(m["goal"]), args.tol)
        if sub is None:
            out[aid] = {"kind": "unreachable", "indices": [], "points": []}
            continue
        kind = "stop" if len(sub) <= args.stop_limit else "manoeuvre"
        out[aid] = {"kind": kind, "indices": sub,
                    "points": [list(vias[i]) for i in sub],
                    "of": len(vias), "ref_points": len(ref)}
        print("[vias] %-34s %d of %d keypoints -> %s"
              % (aid, len(sub), len(vias), kind))

    args.out.write_text(json.dumps({"meta": {"tol": args.tol,
                                             "stop_limit": args.stop_limit},
                                    "vias": out}, indent=2), encoding="utf-8")
    kinds = Counter(v["kind"] for v in out.values())
    sizes = Counter(len(v["indices"]) for v in out.values())
    print("\n[vias] %s" % dict(kinds))
    print("[vias] minimal keypoints needed: %s" % dict(sorted(sizes.items())))
    print("[vias] -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
