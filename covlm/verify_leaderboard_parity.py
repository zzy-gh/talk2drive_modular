#!/usr/bin/env python3
"""
covlm/verify_leaderboard_parity.py
==================================
Does the route sweep still hold under the CARLA that CoVLM actually runs?

``verify_nav_routes.py`` uses whatever carla module is on the path -- 0.9.15 or
0.9.16 here. CoLMDriver runs **0.9.10**, and builds its planner differently:
``GlobalRoutePlannerDAO`` + ``grp.setup()`` + its own ``register_dead_end_lanes``
patch, none of which exist in the newer API. Different map files, different
planner, potentially different routes. A 175/175 measured on 0.9.15 says
nothing about the benchmark until this is checked.

So this script runs the same traces under either CARLA and writes a dump; run
it once per environment and compare the two dumps. It is deliberately
**Python 3.7 compatible and imports nothing from covlm/** beyond the JSON file,
because CoLMDriver's env is py3.7 and cannot parse this package's type syntax.

    # the reference environment: CoLMDriver's own carla 0.9.10
    CARLA=/path/to/colmdriver/carla
    PYTHONPATH=$CARLA/PythonAPI/carla:$CARLA/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg \\
    python covlm/verify_leaderboard_parity.py --dump /tmp/routes_0910.json \\
        --xodr $CARLA/CarlaUE4/Content/Carla/Maps/OpenDrive \\
        --leaderboard /path/to/CoLMDriver-main/simulation/leaderboard

    # the environment the earlier sweep ran in
    PYTHONPATH=/path/to/carla_0.9.15/PythonAPI/carla \\
    python covlm/verify_leaderboard_parity.py --dump /tmp/routes_0915.json \\
        --xodr /path/to/carla_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive

    # then, in either environment
    python covlm/verify_leaderboard_parity.py --compare /tmp/routes_0910.json /tmp/routes_0915.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


# ─────────────────────────────────────────────
# Planner, built the way this CARLA wants
# ─────────────────────────────────────────────

def build_planner(carla_map, hop=1.0, leaderboard_root=None):
    """Return (grp, reset_fn, how) for whichever agents API is importable.

    0.9.10 takes a DAO and needs setup(); 0.9.12+ takes the map directly. The
    leaderboard's dead-end patch is imported from its own source file rather
    than reimplemented, so this cannot drift from what the benchmark runs.
    """
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption

    notes = []
    try:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
        grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, hop))
        grp.setup()
        notes.append("DAO API (0.9.10)")
    except ImportError:
        grp = GlobalRoutePlanner(carla_map, hop)
        notes.append("map API (0.9.12+)")

    if leaderboard_root:
        sys.path.insert(0, leaderboard_root)
        try:
            from leaderboard.utils.route_manipulation import register_dead_end_lanes
            added = register_dead_end_lanes(grp)
            notes.append("register_dead_end_lanes (+%s edges)" % added)
        except Exception as exc:                       # noqa: BLE001
            notes.append("register_dead_end_lanes UNAVAILABLE (%s)" % exc)

    def reset():
        # Same instance-state leak in both versions: _turn_decision reads
        # _previous_decision / _intersection_end_node, so RoadOption labels
        # depend on call history. Reset once per plan, never between its legs.
        grp._intersection_end_node = -1
        grp._previous_decision = RoadOption.VOID

    return grp, reset, ", ".join(notes)


def trace_chain(grp, reset, points):
    """One logical plan through a keypoint chain, as interpolate_trajectory does."""
    import carla

    reset()
    out = []
    for a, b in zip(points, points[1:]):
        la = carla.Location(x=float(a[0]), y=float(a[1]), z=float(a[2]))
        lb = carla.Location(x=float(b[0]), y=float(b[1]), z=float(b[2]))
        try:
            leg = grp.trace_route(la, lb)
        except Exception:                              # noqa: BLE001
            return []
        for wp, option in leg:
            loc = wp.transform.location
            out.append([round(loc.x, 3), round(loc.y, 3), int(option.value)])
    return out


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def path_length(route):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(route, route[1:]))


def max_deviation(ra, rb):
    if not ra or not rb:
        return float("inf")

    def one_way(p, q):
        worst = 0.0
        for a in p:
            worst = max(worst, min(math.hypot(a[0] - b[0], a[1] - b[1]) for b in q))
        return worst

    return max(one_way(ra, rb), one_way(rb, ra))


def options(route):
    out = []
    for p in route:
        if not out or out[-1] != p[2]:
            out.append(p[2])
    return out


# ─────────────────────────────────────────────
# Dump
# ─────────────────────────────────────────────

def do_dump(args):
    import carla

    anchors = json.load(open(args.anchors))["anchors"]
    anchors = [a for a in anchors if _route_id(a["scenario"]) <= args.limit]
    towns = sorted(set(a["town"] for a in anchors))
    print("[parity] %d vehicle routes, towns %s" % (len(anchors), towns))
    print("[parity] carla client %s" % getattr(carla, "__file__", "?"))

    planners = {}
    for town in towns:
        path = os.path.join(args.xodr, town + ".xodr")
        if not os.path.exists(path):
            raise SystemExit("[parity] missing %s" % path)
        with open(path) as f:
            cmap = carla.Map(town, f.read())
        planners[town] = build_planner(cmap, 1.0, args.leaderboard)
        print("[parity] %s: %s" % (town, planners[town][2]))

    out = {"meta": {"xodr": args.xodr, "leaderboard": args.leaderboard,
                    "api": planners[towns[0]][2], "python": sys.version.split()[0]},
           "routes": {}}
    n_fail = 0
    for a in anchors:
        grp, reset, _ = planners[a["town"]]
        keys = [a["start"]] + [list(v) for v in a["vias"]] + [a["goal"]]
        ref = trace_chain(grp, reset, keys)
        goal = trace_chain(grp, reset, [keys[0], keys[-1]])
        if not ref:
            n_fail += 1
        out["routes"][a["anchor_id"]] = {"town": a["town"], "ref": ref, "goal": goal}

    with open(args.dump, "w") as f:
        json.dump(out, f)
    ok = sum(1 for r in out["routes"].values() if r["ref"])
    print("[parity] %d/%d reference routes traced -> %s"
          % (ok, len(anchors), args.dump))
    if n_fail:
        print("[parity] %d route(s) could not be traced at all" % n_fail)
    _report_goal(out, args.tol)
    return 0


def _report_goal(dump, tol):
    rows = dump["routes"]
    match = sum(1 for r in rows.values()
                if r["ref"] and r["goal"]
                and max_deviation(r["ref"], r["goal"]) <= tol
                and options(r["goal"]) == options(r["ref"]))
    print("[parity] nav_goal (vias dropped) reproduces %d/%d in THIS carla"
          % (match, len(rows)))


def _route_id(scenario):
    digits = ""
    for ch in scenario[1:]:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else 1 << 30


# ─────────────────────────────────────────────
# Compare two dumps
# ─────────────────────────────────────────────

def do_compare(args):
    a_path, b_path = args.compare
    A, B = json.load(open(a_path)), json.load(open(b_path))
    print("[parity] A = %s  (%s, py%s)" % (a_path, A["meta"]["api"], A["meta"]["python"]))
    print("[parity] B = %s  (%s, py%s)\n" % (b_path, B["meta"]["api"], B["meta"]["python"]))

    keys = sorted(set(A["routes"]) & set(B["routes"]), key=lambda k: (_route_id(k), k))
    only_a = sorted(set(A["routes"]) - set(B["routes"]))
    only_b = sorted(set(B["routes"]) - set(A["routes"]))
    if only_a or only_b:
        print("[parity] only in A: %d, only in B: %d" % (len(only_a), len(only_b)))

    same, differ, missing = 0, [], []
    for k in keys:
        ra, rb = A["routes"][k]["ref"], B["routes"][k]["ref"]
        if not ra or not rb:
            missing.append((k, bool(ra), bool(rb)))
            continue
        d = max_deviation(ra, rb)
        opts_same = options(ra) == options(rb)
        if d <= args.tol and opts_same:
            same += 1
        else:
            differ.append((k, d, len(ra), len(rb), opts_same))

    print("[parity] reference routes identical across the two CARLAs: %d/%d "
          "(tolerance %.1f m + identical RoadOption sequence)"
          % (same, len(keys), args.tol))
    if missing:
        print("[parity] %d route(s) traceable in only one of them:" % len(missing))
        for k, in_a, in_b in missing[:15]:
            print("[parity]      %-34s A=%s B=%s" % (k, in_a, in_b))
    if differ:
        print("[parity] %d route(s) differ:" % len(differ))
        for k, d, na, nb, o in sorted(differ, key=lambda t: -t[1])[:20]:
            why = "%.2f m" % d + ("" if o else ", RoadOption differs")
            print("[parity]      %-34s %s   %d vs %d points" % (k, why, na, nb))
    if not differ and not missing:
        print("[parity] => the earlier sweep's numbers carry over to CoVLM's CARLA.")
    return 0 if not differ and not missing else 1


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", default=os.path.join(here, "data/route_anchors.json"))
    ap.add_argument("--dump", help="write this environment's routes here")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--xodr", help="OpenDrive dir of THIS carla")
    ap.add_argument("--leaderboard", default=None,
                    help="CoLMDriver's simulation/leaderboard, for its "
                         "register_dead_end_lanes patch")
    ap.add_argument("--limit", type=int, default=46)
    ap.add_argument("--tol", type=float, default=1.1)
    args = ap.parse_args()

    if args.compare:
        return do_compare(args)
    if not args.dump or not args.xodr:
        ap.error("--dump and --xodr are required unless --compare is used")
    return do_dump(args)


if __name__ == "__main__":
    sys.exit(main())
