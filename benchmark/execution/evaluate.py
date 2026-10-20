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

An end-to-end run (run.py --language) is also scored per event: whether the
mission state matches the gold intent's. A failure is **language** if an
event left a wrong state or the first request gave no route, else
**execution**. ``--gold <dir>`` (same backend, gold intents) adds the
language loss: gold success minus end-to-end success.

    python3 benchmark/execution/evaluate.py benchmark/runs/execution/<backend> [<backend dir> ...]
    python3 benchmark/execution/evaluate.py benchmark/runs/execution/basic_agent__e2e__gemini-3.8-flash \
        --gold benchmark/runs/execution/basic_agent
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
# Slack for log positions rounded to the centimetre.
LOG_TOLERANCE_M = 0.1
DEDUP_S = 1.0


def approaches(log: list, places: dict) -> dict:
    """key -> every time the car enters VISIT_M of it."""
    out = {k: [] for k in places}
    inside = set()
    for row in log:
        t, x, y = row[0], row[1], row[2]
        for k, (px, py) in places.items():
            near = math.hypot(x - px, y - py) <= VISIT_M + LOG_TOLERANCE_M
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


def language_part(sc: dict, rec: dict) -> dict:
    """Per-event scoring of an end-to-end run; empty for a gold run."""
    said = [f for f in rec.get("fired", []) if "event_ok" in f]
    if not said and rec.get("result") != "no_plan":
        return {}
    ok = [f["event_ok"] for f in said]
    missed = [a for f in said for a in f.get("asked", []) if a["want"] not in a["offered"]]
    return {"events": len(sc["events"]), "events_fired": len(said),
            "events_ok": sum(ok), "all_events_ok": len(ok) == len(sc["events"]) and all(ok),
            "language_error": not all(ok) or rec.get("result") == "no_plan",
            "grounding_miss": bool(missed),
            "asks": sum(len(f.get("asked", [])) for f in said),
            "parse_s": [f["parse_s"] for f in said],
            "apply_s": [f["apply_s"] for f in said]}


def cause(row: dict) -> str | None:
    """Why an end-to-end scenario failed: language or execution."""
    if row.get("success") or "language_error" not in row:
        return None
    return "language" if row["language_error"] else "execution"


def score(sc: dict, rec: dict, coords: dict) -> dict:
    e = sc["expect"]
    if rec.get("result") in ("not_run", "api_error"):
        return {"id": sc["id"], "result": rec["result"], "functions": sc["functions"],
                "template": sc["template"]}
    if rec.get("result") in (None, "error", "spawn_failed", "hung", "no_plan"):
        return {"id": sc["id"], "result": rec.get("result"), "success": False,
                "completion": 0.0, "functions": sc["functions"], "template": sc["template"],
                **language_part(sc, rec)}
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
           "collisions_vehicle": _dedup([c for c in rec["collisions"]
                                         if c["other"].startswith(("vehicle.", "walker."))]),
           "collisions_static": _dedup([c for c in rec["collisions"]
                                        if not c["other"].startswith(("vehicle.", "walker."))]),
           "lane_invasions": _dedup(rec["lane_invasions"], 0.5),
           "wrong_way_s": round(sum(d for d, row in zip(dt, log[1:]) if row[9]), 1),
           "mean_speed": round(statistics.mean(moving), 2) if moving else 0.0,
           **language_part(sc, rec)}
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

    def pct(xs, q):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None

    results = defaultdict(int)
    for r in rows:
        results[str(r["result"])] += 1
    # Rates are over what ran; an API failure is re-run, not scored.
    rows = [r for r in rows if r["result"] not in ("not_run", "api_error")]
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
    lang = [r for r in rows if "language_error" in r]
    gold = [r for r in rows if "gold_success" in r]
    causes = defaultdict(int)
    for r in lang:
        c = cause(r)
        if c:
            causes[c] += 1
    e2e = {}
    if lang:
        parse_s = [x for r in lang for x in r["parse_s"]]
        apply_s = [x for r in lang for x in r["apply_s"]]
        fired = sum(r["events_fired"] for r in lang)
        e2e = {
            "event_state_acc": round(100.0 * sum(r["events_ok"] for r in lang) / fired, 1)
            if fired else None,
            "events_fired": f"{fired}/{sum(r['events'] for r in lang)}",
            "all_events_ok": rate([r["all_events_ok"] for r in lang]),
            "asks_per_scenario": round(sum(r["asks"] for r in lang) / len(lang), 2),
            "grounding_misses": sum(r["grounding_miss"] for r in lang),
            "failure_cause": dict(sorted(causes.items())),
            "language_failures_gold_failed_too": sum(
                1 for r in lang if cause(r) == "language" and r.get("gold_success") is False),
            "parse_s_p50": pct(parse_s, 0.5), "parse_s_p95": pct(parse_s, 0.95),
            "plan_s_p50": pct(apply_s, 0.5), "plan_s_p95": pct(apply_s, 0.95),
        }
        if gold:
            g = rate([r["gold_success"] for r in gold])
            mine = rate([r["success"] for r in gold])
            e2e.update({"gold_success": g, "language_loss": round(g - mine, 1)})
    return {
        "scenarios": len(rows), "not_run": results.get("not_run", 0),
        "success": rate([r["success"] for r in rows]),
        "completion": round(100 * statistics.mean(r["completion"] for r in rows), 1) if rows else None,
        "results": dict(sorted(results.items())),
        "dropped_place_visited": sum(r.get("dropped_place_visited", False) for r in rows),
        "collisions": sum(r.get("collisions", 0) for r in ok),
        "collisions_vehicle": sum(r.get("collisions_vehicle", 0) for r in ok),
        "collisions_static": sum(r.get("collisions_static", 0) for r in ok),
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
        **e2e,
    }


def report(names: list, sums: list) -> str:
    lines = [f"| metric | {' | '.join(names)} |", "|---|" + "---|" * len(names)]
    for k in ("scenarios", "not_run", "success", "gold_success", "language_loss",
              "completion", "results", "failure_cause", "language_failures_gold_failed_too",
              "event_state_acc", "events_fired",
              "all_events_ok", "asks_per_scenario", "grounding_misses",
              "parse_s_p50", "parse_s_p95", "plan_s_p50", "plan_s_p95",
              "dropped_place_visited", "collisions", "collisions_vehicle", "collisions_static",
              "lane_invasions_per_km", "wrong_way_s", "detour_median",
              "hurry_speed_ratio", "rush_speed_ratio", "cancel_stop_s"):
        if not any(k in s for s in sums):
            continue
        lines.append(f"| {k} | " + " | ".join(str(s.get(k, "")) for s in sums) + " |")
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
    ap.add_argument("--gold", default=None,
                    help="the same backend's gold-intent run dir: adds gold_success, "
                         "language_loss and the 'both' failure cause")
    args = ap.parse_args()

    with open(args.scenarios, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]
    coords = load_coords(scenarios)

    def load(d: Path, sc: dict) -> dict:
        f = d / (sc["id"].replace("/", "__") + ".json")
        return json.loads(f.read_text()) if f.exists() else {"result": "not_run"}

    gold = {}
    if args.gold:
        for sc in scenarios:
            r = score(sc, load(Path(args.gold), sc), coords[sc["town"]])
            if r["result"] not in ("not_run", "api_error"):
                gold[sc["id"]] = r["success"]
    names, sums = [], []
    for d in map(Path, args.runs):
        rows = []
        for sc in scenarios:
            row = score(sc, load(d, sc), coords[sc["town"]])
            if sc["id"] in gold and row["result"] not in ("not_run", "api_error"):
                row["gold_success"] = gold[sc["id"]]
            if "language_error" in row:
                row["cause"] = cause(row)
            rows.append(row)
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
