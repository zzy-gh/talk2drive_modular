"""
covlm/v2v_compare.py
====================
Compare CoVLM (PlugV2V) InterDrive runs with and without driver intents.

* **driving**     RC (mean over vehicles), IP (product), DS = RC x IP,
                  success (DS > 99.95), by IC / LM / LC; vehicle collisions.
* **ordering**    pairs negotiated in one group with different urgency tiers:
                  share of negotiations where the more urgent vehicle gets the
                  higher priority (tie = 0.5), averaged over pairs.
* **travel time** (with - without) per tier, over vehicles completing both.

Tiers are read from the comments of driver_intents_nav.yaml.

    python3 covlm/v2v_compare.py --with <results_driving_covlm dir> --without <dir> \\
        --intents <driver_intents_nav.yaml>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from itertools import combinations

TIER_RANK = {"low": 0, "med": 1, "high": 2}
GROUPS = {"IC": ("ins_ss", "ins_sl", "ins_oppo", "ins_chaos"),
          "LM": ("ins_sr", "ins_c", "ins_rl", "hw_merge"),
          "LC": ("crosschange", "hw_c")}


def route_of(dirname: str) -> str:
    return dirname.replace("Interdrive_no_npc_", "").replace("_nav", "")


def group_of(route: str) -> str:
    # crosschange and hw_c before the shorter patterns they contain
    if "crosschange" in route or "hw_c" in route:
        return "LC"
    for g in ("IC", "LM"):
        if any(p in route for p in GROUPS[g]):
            return g
    return "?"


def load_tiers(path: str) -> dict:
    """(route, vehicle index) -> tier."""
    tiers, route = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^(r\d+_[a-z0-9_]+?)(?:_nav)?:\s*$", line)
        if m:
            route = m.group(1)
            continue
        m = re.search(r"#\s*veh_(\d+)\s*->.*tier (\w+)", line)
        if m and route:
            tiers[(route, int(m.group(1)))] = m.group(2)
    return tiers


def load_driving(root: str) -> dict:
    """route -> {"egos": {k: record}}"""
    out = {}
    for d in sorted(os.listdir(f"{root}/v2x_final")):
        if not d.startswith("Interdrive_no_npc_") or not os.path.isdir(f"{root}/v2x_final/{d}"):
            continue
        egos = {}
        for f in glob.glob(f"{root}/v2x_final/{d}/ego_vehicle_*/results.json"):
            k = int(re.search(r"ego_vehicle_(\d+)", f).group(1))
            recs = json.load(open(f))["_checkpoint"]["records"]
            if recs:
                egos[k] = recs[0]
        out[route_of(d)] = {"egos": egos}
    return out


def scenario_scores(egos: dict) -> dict:
    rc = statistics.mean(r["scores"]["score_route"] for r in egos.values())
    ip = 1.0
    for r in egos.values():
        ip *= r["scores"]["score_penalty"]
    ds = rc * ip
    vcol = sum(len(r["infractions"].get("collisions_vehicle", [])) for r in egos.values())
    return {"RC": rc, "IP": ip, "DS": ds, "success": ds > 99.95, "veh_collisions": vcol}


def driving_table(runs: dict) -> list:
    lines = ["| | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    per = {name: {r: scenario_scores(v["egos"]) for r, v in run.items() if v["egos"]}
           for name, run in runs.items()}
    for g in ("all", "IC", "LM", "LC"):
        for key, fmt in (("DS", "{:.1f}"), ("RC", "{:.1f}"), ("success", "{:.1f}%")):
            cells = []
            for name in runs:
                rows = [s for r, s in per[name].items() if g == "all" or group_of(r) == g]
                v = (100 * statistics.mean(s[key] for s in rows) if key == "success"
                     else statistics.mean(s[key] for s in rows)) if rows else float("nan")
                cells.append(fmt.format(v) + (f" (n={len(rows)})" if key == "DS" else ""))
            lines.append(f"| {g} {key} | " + " | ".join(cells) + " |")
    lines.append("| vehicle collisions | " + " | ".join(
        str(sum(s["veh_collisions"] for s in per[n].values())) for n in runs) + " |")
    return lines


def negotiations(root: str):
    """route -> [priority dict per group], consecutive duplicates removed."""
    out = defaultdict(list)
    for run in sorted(glob.glob(f"{root}/image/v2x_final/Interdrive_no_npc_*/*/")):
        route = route_of(os.path.basename(os.path.dirname(os.path.dirname(run))))
        seen = None
        for f in sorted(glob.glob(run + "ego_vehicle_0/measurements/*.json"),
                        key=lambda p: int(os.path.basename(p).split(".")[0])):
            dp = json.load(open(f)).get("discussion_process")
            if not isinstance(dp, dict):
                continue
            key = hashlib.md5(json.dumps(dp.get("group_discussions"), sort_keys=True,
                                         default=str).encode()).hexdigest()
            if key == seen:
                continue
            seen = key
            for g in dp.get("group_discussions", []):
                pr = {int(str(p["vehicle_id"]).split("_")[1]): p.get("priority_score")
                      for p in g.get("priorities", []) if p.get("priority_score") is not None}
                if len(pr) >= 2:
                    out[route].append(pr)
    return out


def ordering(root: str, tiers: dict) -> dict:
    negs = negotiations(root)
    pairs = defaultdict(list)                    # (route, i, j) -> [1, 0, 0.5, ...]
    by_tier = defaultdict(list)
    ties = total = 0
    for route, groups in negs.items():
        for pr in groups:
            for k, s in pr.items():
                if (route, k) in tiers:
                    by_tier[tiers[(route, k)]].append(s)
            for i, j in combinations(sorted(pr), 2):
                ti, tj = tiers.get((route, i)), tiers.get((route, j))
                if not ti or not tj or ti == tj:
                    continue
                hi, lo = (i, j) if TIER_RANK[ti] > TIER_RANK[tj] else (j, i)
                v = 1.0 if pr[hi] > pr[lo] else 0.5 if pr[hi] == pr[lo] else 0.0
                pairs[(route, i, j)].append(v)
                ties += pr[hi] == pr[lo]
                total += 1
    per_pair = [statistics.mean(v) for v in pairs.values()]
    return {"negotiations": sum(len(g) for g in negs.values()), "pairs": len(per_pair),
            "comparisons": total,
            "ordering_rate": 100 * statistics.mean(per_pair) if per_pair else None,
            "tie_share": 100 * ties / total if total else None,
            "mean_priority": {t: (round(statistics.mean(v), 1), len(v))
                              for t, v in sorted(by_tier.items(), key=lambda x: TIER_RANK[x[0]])}}


def travel_time(a: dict, b: dict, tiers: dict) -> dict:
    """Per tier, mean (with - without) over vehicles completing in both."""
    diff = defaultdict(list)
    for route in a:
        if route not in b:
            continue
        for k, ra in a[route]["egos"].items():
            rb = b[route]["egos"].get(k)
            t = tiers.get((route, k))
            if not rb or not t:
                continue
            if ra["scores"]["score_route"] >= 100 and rb["scores"]["score_route"] >= 100:
                diff[t].append(ra["meta"]["duration_game"] - rb["meta"]["duration_game"])
    return {t: (round(statistics.mean(v), 2), round(statistics.median(v), 2), len(v))
            for t, v in sorted(diff.items(), key=lambda x: TIER_RANK[x[0]])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with", dest="with_", required=True)
    ap.add_argument("--without", required=True)
    ap.add_argument("--intents", required=True)
    args = ap.parse_args()

    tiers = load_tiers(args.intents)
    a, b = load_driving(args.with_), load_driving(args.without)
    print(f"[v2v] tiers for {len(tiers)} vehicles; scenarios with={len(a)} without={len(b)}")
    print("\n".join(driving_table({"with requests": a, "without requests": b})))
    for name, root in (("with requests", args.with_), ("without requests", args.without)):
        o = ordering(root, tiers)
        print(f"\n[ordering] {name}: {o}")
    print("\n[travel time, with - without, s] tier: (mean, median, n vehicles completing in both)")
    for t, v in travel_time(a, b, tiers).items():
        print(f"  {t}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
