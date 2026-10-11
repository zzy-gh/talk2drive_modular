"""
benchmark/execution/evaluate.py
===============================
Score execution runs against their scenarios.

A place counts as visited when the car comes within VISIT_M of it. Expected
stops are matched in order -- each one at its first approach *after* the
previous one was visited -- as MissionPlanner itself does, so passing a place
before it was ever added to the route does not count as visiting it early.

**Success** (headline): every expected stop visited, in the expected order;
arrived at the expected destination (or, for a cancel, standing still soon
after it); no place the passenger dropped was visited; within the time limit.

    python3 benchmark/execution/evaluate.py benchmark/runs/execution/<backend> [<backend dir> ...]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.execution.generate import OUT as SCENARIOS, VISIT_M   # noqa: E402

STAND_MPS = 0.5
DEDUP_S = 1.0


def approaches(log: list, places: dict) -> dict:
    """key -> every time the car enters VISIT_M of it."""
    out = {k: [] for k in places}
    inside = set()
    for row in log:
        t, x, y = row[0], row[1], row[2]
        for k, (px, py) in places.items():
            near = math.hypot(x - px, y - py) <= VISIT_M
            if near and k not in inside:
                out[k].append(t)
            (inside.add if near else inside.discard)(k)
    return out


def _dedup(events: list, gap: float = DEDUP_S) -> int:
    n, last = 0, None
    for e in sorted(events, key=lambda e: e["t"]):
        if last is None or e["t"] - last > gap:
            n += 1
        last = e["t"]
    return n


def score(sc: dict, rec: dict, coords: dict) -> dict:
    e = sc["expect"]
    if rec.get("result") == "not_run":
        return {"id": sc["id"], "result": "not_run", "functions": sc["functions"],
                "template": sc["template"]}
    if rec.get("result") in (None, "error", "spawn_failed", "hung"):
        return {"id": sc["id"], "result": rec.get("result"), "success": False,
                "completion": 0.0, "functions": sc["functions"], "template": sc["template"]}
    log = rec["log"]
    targets = e["visits"] + ([e["arrive"]] if e["arrive"] else [])
    keys = set(targets) | set(e["never_visit"])
    seen = approaches(log, {k: coords[k] for k in keys})

    # In order: each target at its first approach after the previous target.
    reached, last_t = 0, -1.0
    for k in targets:
        t = next((t for t in seen[k] if t > last_t), None)
        if t is None:
            break
        reached += 1
        last_t = t
    in_order = reached == len(targets)
    never_ok = not any(seen[k] for k in e["never_visit"])
    if e["cancelled"]:
        ended_ok = rec["result"] == "stopped" and reached == len(e["visits"])
    else:
        ended_ok = rec["result"] == "arrived" and in_order

    moving = [row[4] for row in log if row[4] > STAND_MPS]
    dt = [b[0] - a[0] for a, b in zip(log, log[1:])]
    out = {"id": sc["id"], "template": sc["template"], "functions": sc["functions"],
           "result": rec["result"],
           "success": bool(in_order and never_ok and ended_ok),
           "completion": round(reached / len(targets), 3) if targets else 1.0,
           "order_ok": in_order, "dropped_place_visited": not never_ok,
           "sim_s": rec["sim_s"], "driven_m": rec["driven_m"],
           "detour": round(rec["driven_m"] / e["planned_m"], 3) if e["planned_m"] else None,
           "collisions": _dedup(rec["collisions"]),
           "lane_invasions": _dedup(rec["lane_invasions"], 0.5),
           "wrong_way_s": round(sum(d for d, row in zip(dt, log[1:]) if row[9]), 1),
           "mean_speed": round(statistics.mean(moving), 2) if moving else 0.0}
    if e["cancelled"] and len(rec["fired"]) > 1:
        t_cancel = rec["fired"][-1]["t"]
        after = [row for row in log if row[0] >= t_cancel]
        stop = next((row for row in after if row[4] < STAND_MPS), None)
        out["cancel_stop_s"] = round(stop[0] - t_cancel, 2) if stop else None
    if sc["template"] == "rush_elsewhere" and len(rec["fired"]) > 1:
        t_rush = rec["fired"][1]["t"]
        before = [row[4] for row in log if row[0] < t_rush and row[4] > STAND_MPS]
        after_ = [row[4] for row in log if row[0] >= t_rush and row[4] > STAND_MPS]
        if before and after_:
            out["rush_speed_ratio"] = round(statistics.mean(after_) / statistics.mean(before), 3)
    return out


def load_coords(sc_list: list) -> dict:
    """(town) -> key -> (x, y), from the same place table the runner uses."""
    from benchmark.execution.generate import place_table
    from core.landmark_kb import LandmarkKnowledgeBase
    from core.paths import KB_CSV, section_path
    from core.route_provider import CarlaRouteProvider
    import carla

    kb = LandmarkKnowledgeBase(str(KB_CSV))
    xodr = section_path("covlm", "opendrive_dir")
    out = {}
    for town in sorted({s["town"] for s in sc_list}):
        m = carla.Map(town, (xodr / f"{town}.xodr").read_text())
        table = place_table(kb, CarlaRouteProvider(m, 1.0), town)
        out[town] = {k: (p.x, p.y) for k, p in table.items()}
    return out


def summarise(rows: list) -> dict:
    def rate(xs):
        return round(100.0 * sum(xs) / len(xs), 1) if xs else None

    results = defaultdict(int)
    for r in rows:
        results[str(r["result"])] += 1
    rows = [r for r in rows if r["result"] != "not_run"]   # rates are over what ran
    by_func, by_tpl = defaultdict(list), defaultdict(list)
    for r in rows:
        for f in r["functions"]:
            by_func[f].append(r["success"])
        by_tpl[r["template"]].append(r["success"])
    ok = [r for r in rows if "sim_s" in r]
    pairs = defaultdict(dict)
    for r in ok:
        if r["template"].startswith("plan_two_stops"):
            town, _, n = r["id"].split("/")
            pairs[(town, n)][r["template"].endswith("hurry")] = r["mean_speed"]
    ratios = [p[True] / p[False] for p in pairs.values() if True in p and p.get(False)]
    return {
        "scenarios": len(rows), "not_run": results.get("not_run", 0),
        "success": rate([r["success"] for r in rows]),
        "completion": round(100 * statistics.mean(r["completion"] for r in rows), 1) if rows else None,
        "results": dict(sorted(results.items())),
        "dropped_place_visited": sum(r.get("dropped_place_visited", False) for r in rows),
        "collisions": sum(r.get("collisions", 0) for r in ok),
        "lane_invasions_per_km": round(sum(r.get("lane_invasions", 0) for r in ok)
                                       / max(1e-3, sum(r.get("driven_m", 0) for r in ok) / 1000), 2),
        "wrong_way_s": round(sum(r.get("wrong_way_s", 0) for r in ok), 1),
        "detour_median": round(statistics.median([r["detour"] for r in ok if r.get("detour")]), 3) if ok else None,
        "hurry_speed_ratio": round(statistics.mean(ratios), 3) if ratios else None,
        "rush_speed_ratio": round(statistics.mean([r["rush_speed_ratio"] for r in ok
                                                   if "rush_speed_ratio" in r]), 3)
        if any("rush_speed_ratio" in r for r in ok) else None,
        "cancel_stop_s": round(statistics.mean([r["cancel_stop_s"] for r in ok
                                                if r.get("cancel_stop_s") is not None]), 2)
        if any(r.get("cancel_stop_s") is not None for r in ok) else None,
        "by_function": {f: (rate(v), len(v)) for f, v in sorted(by_func.items())},
        "by_template": {t: (rate(v), len(v)) for t, v in sorted(by_tpl.items())},
    }


def report(names: list, sums: list) -> str:
    lines = [f"| metric | {' | '.join(names)} |", "|---|" + "---|" * len(names)]
    for k in ("scenarios", "not_run", "success", "completion", "results", "dropped_place_visited",
              "collisions", "lane_invasions_per_km", "wrong_way_s", "detour_median",
              "hurry_speed_ratio", "rush_speed_ratio", "cancel_stop_s"):
        lines.append(f"| {k} | " + " | ".join(str(s[k]) for s in sums) + " |")
    for group in ("by_function", "by_template"):
        lines.append(f"| **{group.replace('_', ' ')}** |" + " |" * len(names))
        for k in sorted({k for s in sums for k in s[group]}):
            lines.append(f"| {k} | " + " | ".join("%s (n=%s)" % s[group].get(k, (None, 0))
                                                for s in sums) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="one directory of logs per backend")
    ap.add_argument("--scenarios", default=str(SCENARIOS))
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    with open(args.scenarios, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]
    coords = load_coords(scenarios)
    names, sums = [], []
    for d in map(Path, args.runs):
        rows = []
        for sc in scenarios:
            f = d / (sc["id"].replace("/", "__") + ".json")
            rec = json.loads(f.read_text()) if f.exists() else {"result": "not_run"}
            rows.append(score(sc, rec, coords[sc["town"]]))
        (d / "scores.json").write_text(json.dumps(rows, indent=1))
        names.append(d.name)
        sums.append(summarise(rows))
    text = report(names, sums)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
