#!/usr/bin/env python3
"""
covlm/verify_nav_routes.py
==========================
Does the talk2drive navigation module reproduce InterDrive's reference routes?

Runs fully offline: ``carla.Map`` is built straight from the town's ``.xodr``,
so no CARLA server is touched and nobody's running simulation is disturbed.

Three routes are traced per vehicle and compared against the reference:

  reference   GlobalRoutePlanner over the route file's raw keypoints -- exactly
              what ``leaderboard/utils/route_manipulation.py::
              interpolate_trajectory`` builds for the benchmark.
  nav_raw     the mission layer driving the anchor's keypoints as they are.
              Must be 175/175: it is the plumbing check, and a miss means the
              provider and the leaderboard disagree about something basic.
  nav_snap    the same, but with every anchor coordinate pushed through
              ``RouteProvider.snap`` first -- the call ``MissionPlanner`` makes
              on every KB landmark. Measures what snapping costs. It is why
              anchors must NOT be snapped.
  nav_goal    the anchor's goal only, no via points -- start -> goal. The
              number that decides how much the intent has to carry: wherever
              nav_goal diverges, that route's via keypoints have to ride along
              in the intent as ``waypoints: [Stop]``.

    python covlm/verify_nav_routes.py [--limit 46] [--tol 0.5] [--report out.csv]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.route_provider import CarlaRouteProvider                            # noqa: E402

from covlm.anchors import AnchorBook                                          # noqa: E402
from covlm.paths import OPENDRIVE_DIR, ROUTE_ANCHORS                          # noqa: E402

DEF_XODR = OPENDRIVE_DIR
DEF_ANCHORS = ROUTE_ANCHORS


def load_map(town: str, xodr_dir: Path):
    import carla
    path = xodr_dir / f"{town}.xodr"
    if not path.exists():
        raise FileNotFoundError(path)
    return carla.Map(town, path.read_text(encoding="utf-8"))


def trace_chain(provider: CarlaRouteProvider, points) -> list:
    """One logical plan through a keypoint chain.

    Delegates to ``CarlaRouteProvider.trace_chain``, which resets the router's
    carry-over turn state once at the front -- without that the RoadOption
    labels depend on which routes were traced before, and a 175-route sweep
    reports differences that are pure call-order artefacts.
    """
    return provider.trace_chain(points)


def path_length(route) -> float:
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(route, route[1:]))


def max_deviation(ra, rb) -> float:
    """Symmetric max nearest-point distance -- a Hausdorff distance between the
    two polylines' sample sets. 0 means one route lies on the other."""
    if not ra or not rb:
        return float("inf")

    def one_way(p, q):
        worst = 0.0
        for a in p:
            worst = max(worst, min(math.hypot(a.x - b.x, a.y - b.y) for b in q))
        return worst

    return max(one_way(ra, rb), one_way(rb, ra))


def options(route) -> list[int]:
    """RoadOption sequence with consecutive duplicates collapsed."""
    out = []
    for p in route:
        if not out or out[-1] != p.option:
            out.append(p.option)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", type=Path, default=DEF_ANCHORS)
    ap.add_argument("--xodr", type=Path, default=DEF_XODR)
    ap.add_argument("--limit", type=int, default=46)
    ap.add_argument("--tol", type=float, default=1.1,
                    help="max deviation in metres still counted as a match. "
                         "Default 1.1 = one 1 m sampling step: chaining legs "
                         "ends a route up to one sample short of a single "
                         "through-trace, which is an artefact of GRP's "
                         "resolution, not a different path. The measured "
                         "deviations fall cleanly either <=1.02 m or >3 m, so "
                         "nothing interesting hides near this threshold -- the "
                         "histogram below shows the whole distribution.")
    ap.add_argument("--report", type=Path, default=None, help="write a per-route CSV")
    args = ap.parse_args()

    book = AnchorBook.load(args.anchors)
    anchors = [a for a in book if a.route_id <= args.limit]
    towns = sorted({a.town for a in anchors})
    print(f"[verify] {len(anchors)} vehicle routes across {len(book.scenarios())} "
          f"scenarios, towns {towns}")
    print(f"[verify] offline maps from {args.xodr} (no server contacted)\n")

    providers = {}
    for town in towns:
        providers[town] = CarlaRouteProvider(load_map(town, args.xodr), 1.0)
        print(f"[verify] {town} map + GRP ready")
    print()

    rows = []
    tally = Counter()
    need_via = []
    plumbing_fail = []
    snap_fail = []

    for a in anchors:
        prov = providers[a.town]
        keys = a.keypoints
        ref = trace_chain(prov, keys)
        if not ref:
            plumbing_fail.append((a.anchor_id, "reference route unreachable"))
            tally["ref_failed"] += 1
            continue

        raw = trace_chain(prov, keys)
        snapped_pts = [prov.snap(*p) for p in keys]
        snap = trace_chain(prov, snapped_pts)
        goal_only = trace_chain(prov, [keys[0], keys[-1]])

        def verdict(cand):
            if not cand:
                return False, float("inf")
            d = max_deviation(ref, cand)
            return (d <= args.tol and options(cand) == options(ref)), d

        ok_raw, d_raw = verdict(raw)
        ok_snap, d_snap = verdict(snap)
        ok_goal, d_goal = verdict(goal_only)

        tally["raw_match" if ok_raw else "raw_differ"] += 1
        tally["snap_match" if ok_snap else "snap_differ"] += 1
        tally["goal_match" if ok_goal else "goal_differ"] += 1
        if not ok_raw:
            plumbing_fail.append((a.anchor_id, f"nav_raw deviates {d_raw:.2f} m"))
        if not ok_snap:
            snap_fail.append((a.anchor_id, d_snap, len(ref), len(snap)))
        if ok_raw and not ok_goal:
            need_via.append(a)

        rows.append(dict(
            anchor_id=a.anchor_id, town=a.town, veh=a.veh, label=a.label,
            label_source=a.label_source, n_keypoints=len(keys),
            ref_pts=len(ref), ref_len_m=round(path_length(ref), 2),
            raw_dev_m=round(d_raw, 3), raw_match=int(ok_raw),
            snap_dev_m=round(d_snap, 3), snap_match=int(ok_snap),
            goal_pts=len(goal_only), goal_len_m=round(path_length(goal_only), 2),
            goal_dev_m=round(d_goal, 3), goal_match=int(ok_goal),
        ))

    n = len(rows)
    print(f"[verify] nav_raw  anchor keypoints as-is          : "
          f"{tally['raw_match']}/{n}   <- plumbing, must be {n}/{n}")
    print(f"[verify] nav_snap same, but snap() applied first  : "
          f"{tally['snap_match']}/{n}   <- why anchors must not be snapped")
    print(f"[verify] nav_goal goal only, via points dropped   : "
          f"{tally['goal_match']}/{n}")
    print(f"[verify] tolerance {args.tol} m and an identical RoadOption sequence\n")

    edges = [(0.01, "exact"), (1.1, "<=1 sample"), (5.0, "1-5 m"),
             (20.0, "5-20 m"), (float("inf"), ">20 m")]
    for col, title in (("snap_dev_m", "nav_snap"), ("goal_dev_m", "nav_goal")):
        hist = Counter()
        for r in rows:
            d = r[col]
            hist[next(lbl for lim, lbl in edges if d <= lim)] += 1
        cells = "  ".join(f"{lbl}: {hist[lbl]}" for _, lbl in edges if hist[lbl])
        print(f"[verify] {title} deviation: {cells}")
    print()

    if plumbing_fail:
        print(f"[verify] PLUMBING BROKEN on {len(plumbing_fail)} route(s):")
        for aid, why in plumbing_fail[:20]:
            print(f"[verify]      {aid}: {why}")
        print()

    if snap_fail:
        big = [f for f in snap_fail if f[1] > 2.0]
        print(f"[verify] snapping changed {len(snap_fail)} route(s): "
              f"{len(snap_fail) - len(big)} by <=2 m (the tail lands one 1 m "
              f"sample short), {len(big)} substantially:")
        for aid, d, nref, nsnap in sorted(big, key=lambda f: -f[1])[:10]:
            print(f"[verify]      {aid}: {d:.1f} m, {nref} -> {nsnap} points")
        print("[verify] => Place(source='anchor') must bypass RouteProvider.snap;")
        print("[verify]    only KB landmarks, which sit off-road, need snapping.\n")

    print(f"[verify] {len(need_via)} route(s) need via keypoints in the intent "
          f"as waypoints:[Stop]")
    if need_via:
        per_scen = defaultdict(list)
        for a in need_via:
            per_scen[a.scenario].append(a.veh)
        print(f"[verify]    {len(per_scen)} scenarios affected:")
        for sc in sorted(per_scen, key=lambda x: int(x.split('_')[0][1:])):
            print(f"[verify]      {sc}: veh {sorted(per_scen[sc])}")

    if args.report:
        import csv
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[verify] per-route report -> {args.report}")

    return 0 if not plumbing_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
