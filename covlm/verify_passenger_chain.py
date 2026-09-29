#!/usr/bin/env python3
"""
covlm/verify_passenger_chain.py
===============================
The whole chain, for real, on all 175 vehicles:

    utterance -> Gemini -> intent.destination -> knowledge base -> GRP -> route

and then: is that route the benchmark's route?

This is the check the other tools cannot make. ``verify_nav_routes.py`` starts
from the anchor's coordinate, so it only ever tested the planner.  Here nothing
is injected -- the destination is whatever the LLM read out of the passenger's
words, resolved through the knowledge base like any other trip. If the route
still matches, the benchmark route genuinely follows from what the passenger
said.

Three things are measured separately, because they fail for different reasons:

    parsed      did Gemini return a usable intent at all?
    destination did it name the type the mission intends, and does that resolve
                to the coordinate the route ends at?
    route       does the planned route match the benchmark's, within --tol?

LLM answers are cached in data/llm_cache.json keyed by (town, types, utterance),
so a re-run costs nothing and a changed prompt invalidates itself.

    python covlm/verify_passenger_chain.py [--limit 46] [--no-llm] [--refresh]

Needs the Gemini credentials (``zhiyuan_gemini``'s activate hook) and a carla
module for ``carla.Map``; contacts no CARLA server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.landmark_kb import LandmarkKnowledgeBase                           # noqa: E402
from core.paths import KB_CSV                                                # noqa: E402
from core.route_provider import CarlaRouteProvider                           # noqa: E402

from covlm.kb import CovlmKnowledgeBase, assert_covlm_kb_sane                # noqa: E402
from covlm.paths import DATA_DIR, OPENDRIVE_DIR                              # noqa: E402


def route_id(scenario: str) -> int:
    digits = ""
    for ch in scenario[1:]:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else 1 << 30


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


class Cache:
    def __init__(self, path: Path, refresh: bool) -> None:
        self.path = path
        self.data = {} if refresh or not path.is_file() else json.loads(
            path.read_text(encoding="utf-8"))
        self.hits = self.misses = 0

    @staticmethod
    def key(town, types, utterance) -> str:
        blob = "%s|%s|%s" % (town, ",".join(sorted(types)), utterance)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, k):
        if k in self.data:
            self.hits += 1
            return self.data[k]
        self.misses += 1
        return None

    def put(self, k, v):
        self.data[k] = v

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")



def _parse_with_backoff(parser, utterance, town, retries, base_sleep):
    """Vertex answers 429 RESOURCE_EXHAUSTED under a burst of 175 calls.

    A failed call is indistinguishable from "the LLM could not parse this" once
    it reaches the tally, which would quietly turn a quota problem into a
    reported accuracy problem. So retry with exponential backoff and only give
    up -- loudly -- after `retries` attempts.
    """
    delay = max(base_sleep, 1.0)
    for attempt in range(retries + 1):
        try:
            return parser.parse(utterance, town=town)
        except Exception as exc:                         # noqa: BLE001
            transient = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) \
                or "503" in str(exc) or "UNAVAILABLE" in str(exc)
            if not transient or attempt == retries:
                print("[chain] giving up on a call after %d attempt(s): %s"
                      % (attempt + 1, str(exc)[:120]))
                raise
            print("[chain] rate limited, backing off %.0fs (attempt %d/%d)"
                  % (delay, attempt + 1, retries))
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", type=Path, default=DATA_DIR / "passenger_missions.json")
    ap.add_argument("--landmarks", type=Path, default=DATA_DIR / "covlm_landmarks.csv")
    ap.add_argument("--cache", type=Path, default=DATA_DIR / "llm_cache.json")
    ap.add_argument("--xodr", type=Path, default=OPENDRIVE_DIR)
    ap.add_argument("--limit", type=int, default=46)
    ap.add_argument("--tol", type=float, default=1.1)
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--no-llm", action="store_true",
                    help="cache only; fail instead of calling Gemini")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--sleep", type=float, default=0.6,
                    help="pause between LLM calls; Vertex 429s on a 175-call burst")
    ap.add_argument("--retries", type=int, default=6,
                    help="backoff attempts per call before failing loudly")
    args = ap.parse_args()

    import carla

    base = LandmarkKnowledgeBase(str(KB_CSV))
    covlm = CovlmKnowledgeBase(base, args.landmarks)
    assert_covlm_kb_sane(base, covlm)
    print("[chain] knowledge base guarantees hold (assert_covlm_kb_sane)")

    missions = json.loads(args.missions.read_text(encoding="utf-8"))["missions"]
    missions = {k: m for k, m in missions.items() if route_id(m["scenario"]) <= args.limit}
    towns = sorted({m["town"] for m in missions.values()})

    providers = {}
    for town in towns:
        path = args.xodr / (town + ".xodr")
        providers[town] = CarlaRouteProvider(
            carla.Map(town, path.read_text(encoding="utf-8")), 1.0)
        print("[chain] %s map ready" % town)

    cache = Cache(args.cache, args.refresh)
    import atexit
    atexit.register(cache.save)          # a 429 partway through must not lose work
    parser = None
    tally = Counter()
    rows = []
    bad_dest, bad_route, bad_parse = [], [], []

    for aid in sorted(missions, key=lambda k: (route_id(missions[k]["scenario"]),
                                               missions[k]["scenario"],
                                               missions[k]["veh"])):
        m = missions[aid]
        view = covlm.bind(m["scenario"], m["veh"])
        types = view.get_landmark_types(m["town"])
        key = Cache.key(m["town"], types, m["utterance"])

        intent = cache.get(key)
        if intent is None:
            if args.no_llm:
                print("[chain] cache miss for %s and --no-llm given" % aid)
                return 2
            if parser is None:
                from core.llm_inference import GeminiCommandParser
                parser = GeminiCommandParser(view)
            parser.kb = view
            intent = _parse_with_backoff(parser, m["utterance"], m["town"],
                                         args.retries, args.sleep)
            cache.put(key, intent)
            if cache.misses % 10 == 0:
                cache.save()
            if args.sleep:
                time.sleep(args.sleep)

        row = {"anchor_id": aid, "town": m["town"], "tier": m["tier"],
               "label": m["label"], "kept_original": m["kept_original"],
               "parsed": int(bool(intent)), "llm_destination": "",
               "llm_stops": "", "want_stops": "", "stops_ok": 0,
               "dest_ok": 0, "route_ok": 0, "dev_m": ""}

        if not intent:
            tally["parse_failed"] += 1
            bad_parse.append((aid, m["utterance"]))
            rows.append(row)
            continue
        tally["parsed"] += 1

        # Walk the mission exactly as MissionPlanner would: each stop is
        # resolved from where the car currently is, then the destination from
        # the last stop -- NOT all of them from the start. Resolving the
        # destination from the start would pass here and diverge in the real
        # planner.
        dest = intent.get("destination") or ""
        row["llm_destination"] = dest
        want_stops = [s["label"] for s in m.get("stops", [])]
        said_stops = [w.get("type") for w in (intent.get("waypoints") or [])]
        row["llm_stops"] = "|".join(str(x) for x in said_stops)
        row["want_stops"] = "|".join(want_stops)
        row["stops_ok"] = int(said_stops == want_stops)

        here = (m["start"][0], m["start"][1], m["start"][2])
        chain, resolved = [here], True
        for label in said_stops:
            hit = view.find_nearest_coordinate(m["town"], label, here[0], here[1]) if label else None
            if hit is None:
                resolved = False
                break
            here = (hit["x"], hit["y"], hit["z"])
            chain.append(here)

        found = view.find_nearest_coordinate(m["town"], dest, here[0], here[1]) \
            if (dest and resolved) else None
        goal = m["goal"]
        if found and math.hypot(found["x"] - goal[0], found["y"] - goal[1]) < 0.5 \
                and said_stops == want_stops:
            tally["dest_ok"] += 1
            row["dest_ok"] = 1
        else:
            got = ("(%.1f, %.1f)" % (found["x"], found["y"])) if found else "nothing"
            if said_stops != want_stops:
                got += "  stops %s != %s" % (said_stops, want_stops)
            bad_dest.append((aid, m["label"], dest, got))
            rows.append(row)
            continue

        prov = providers[m["town"]]
        ref = prov.trace_chain([m["start"]] + m["vias"] + [goal])
        # A manoeuvre chain is route shape, not language: it cannot be spoken,
        # so the mission carries it and the planner is handed it directly.
        shape = [tuple(p) for p in m.get("shape_vias", [])]
        got = prov.trace_chain(chain + shape + [(found["x"], found["y"], found["z"])])
        dev = max_dev(ref, got)
        row["dev_m"] = round(dev, 3)
        if dev <= args.tol and options(got) == options(ref):
            tally["route_ok"] += 1
            row["route_ok"] = 1
        else:
            bad_route.append((aid, dev, len(ref), len(got)))
        rows.append(row)

    cache.save()
    n = len(missions)
    print("\n[chain] Gemini calls: %d new, %d from cache" % (cache.misses, cache.hits))
    print("[chain] parsed                              : %d/%d" % (tally["parsed"], n))
    print("[chain] destination resolves to the route end: %d/%d" % (tally["dest_ok"], n))
    print("[chain] route matches the benchmark          : %d/%d" % (tally["route_ok"], n))
    spoken = [r for r in rows if r["want_stops"]]
    if spoken:
        print("[chain] of those, %d carry a spoken stop; %d had the stop read back "
              "correctly" % (len(spoken), sum(r["stops_ok"] for r in spoken)))

    if bad_parse:
        print("\n[chain] %d utterance(s) Gemini could not turn into an intent:" % len(bad_parse))
        for aid, u in bad_parse[:10]:
            print("[chain]      %-34s %s" % (aid, u[:70]))
    if bad_dest:
        print("\n[chain] %d destination(s) resolved to the wrong place:" % len(bad_dest))
        for aid, want, got, where in bad_dest[:15]:
            print("[chain]      %-34s wanted '%s', LLM said '%s' -> %s"
                  % (aid, want, got, where))
    if bad_route:
        print("\n[chain] %d route(s) differ from the benchmark:" % len(bad_route))
        for aid, dev, na, nb in sorted(bad_route, key=lambda t: -t[1])[:15]:
            print("[chain]      %-34s %.2f m   %d vs %d points" % (aid, dev, na, nb))

    if args.report:
        import csv as _csv
        with args.report.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n[chain] per-vehicle report -> %s" % args.report)

    return 0 if tally["route_ok"] == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
