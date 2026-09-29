#!/usr/bin/env python3
"""
covlm/export_nav_routes.py
==========================
Turn the passenger missions into something CoLMDriver can run, without
changing a line of CoLMDriver.

For every vehicle it writes a leaderboard route file whose keypoints are the
ones the passenger's words produced -- start, the stops the LLM read out of the
sentence, and the destination it resolved -- so the benchmark drives the route
the language asked for. ``covlm/verify_passenger_chain.py`` has shown that
Gemini reproduces exactly this chain for all 175 vehicles, so the export is
deterministic and needs no API call.

Two details make this drop in with no edits upstream:

  the directory name   ``eval_driving.sh`` hardcodes
                       ``ROUTES_DIR=<leaderboard>/data/Interdrive/$3``, so the
                       export goes in that same folder under a suffixed
                       scenario name (``r1_town05_ins_c_nav``) and is selected
                       just by passing that name as $3.
  the yaml keys        ``cov2v_bridge._current_scenario_name`` keys driver
                       intents by ``Path(routes_dir).name``, so the utterances
                       are written under the suffixed names too.

Route ``id`` and ``town`` are carried over from the original file, and the
per-vehicle ``_<k>.xml`` suffix is preserved because the evaluator indexes
vehicles by it (``route_name.split('.')[0].split('_')[-1]``).

    python covlm/export_nav_routes.py                      # stage locally
    python covlm/export_nav_routes.py --install <CoLMDriver-main>

``--install`` only adds new ``*_nav`` directories and a new yaml; it never
touches the original routes or driver_intents.yaml.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.landmark_kb import LandmarkKnowledgeBase                           # noqa: E402
from core.paths import KB_CSV                                                # noqa: E402

from covlm.kb import CovlmKnowledgeBase                                      # noqa: E402
from covlm.paths import DATA_DIR, INTERDRIVE_DIR                             # noqa: E402

SUFFIX = "_nav"


def route_id(scenario: str) -> int:
    digits = ""
    for ch in scenario[1:]:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else 1 << 30


def resolve_chain(view, mission):
    """start -> each stop -> destination, resolved the way the planner will.

    Each leg is looked up from where the car currently is, not from the start:
    that is what ``MissionPlanner`` does, and resolving everything from the
    start gives a different answer wherever two landmarks of one type exist.
    """
    town = mission["town"]
    here = tuple(mission["start"])
    chain = [here]
    for stop in mission.get("stops", []):
        hit = view.find_nearest_coordinate(town, stop["label"], here[0], here[1])
        if hit is None:
            return None
        here = (hit["x"], hit["y"], hit["z"])
        chain.append(here)
    chain.extend(tuple(p) for p in mission.get("shape_vias", []))
    hit = view.find_nearest_coordinate(town, mission["label"], here[0], here[1])
    if hit is None:
        return None
    chain.append((hit["x"], hit["y"], hit["z"]))
    return chain


def write_route(src_xml: Path, dst_xml: Path, chain) -> None:
    src = ET.parse(src_xml).getroot().find("route")
    # RouteParser reads only x/y/z, but carrying the source's pitch/roll/yaw
    # keeps the exported file a diff of coordinates alone, so anything that
    # does look at them later sees what the benchmark always had.
    first = src.find("waypoint")
    pose = {k: (first.get(k) if first is not None else d)
            for k, d in (("pitch", "360.0"), ("roll", "0.0"), ("yaw", "0"))}
    routes = ET.Element("routes")
    route = ET.SubElement(routes, "route",
                          {"id": src.get("id"), "town": src.get("town")})
    for x, y, z in chain:
        attrs = dict(pose)
        attrs.update({"x": "%.6g" % x, "y": "%.6g" % y, "z": "%.6g" % z})
        ET.SubElement(route, "waypoint", attrs)
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(routes).write(dst_xml, encoding="utf-8", xml_declaration=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", type=Path, default=DATA_DIR / "passenger_missions.json")
    ap.add_argument("--landmarks", type=Path, default=DATA_DIR / "covlm_landmarks.csv")
    ap.add_argument("--interdrive", type=Path, default=INTERDRIVE_DIR,
                    help="the ORIGINAL route directory, read for route id and town")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "export",
                    help="staging directory (default); --install copies from here")
    ap.add_argument("--install", type=Path, default=None,
                    help="a CoLMDriver-main checkout; adds the *_nav route "
                         "directories and driver_intents_nav.yaml into it")
    ap.add_argument("--suffix", default=SUFFIX)
    ap.add_argument("--limit", type=int, default=46)
    args = ap.parse_args()

    base = LandmarkKnowledgeBase(str(KB_CSV))
    covlm = CovlmKnowledgeBase(base, args.landmarks)
    missions = json.loads(args.missions.read_text(encoding="utf-8"))["missions"]
    missions = {k: m for k, m in missions.items() if route_id(m["scenario"]) <= args.limit}

    routes_dir = args.out / "Interdrive"
    if routes_dir.exists():
        shutil.rmtree(routes_dir)

    by_scenario: dict[str, dict[int, str]] = {}
    kinds = Counter()
    failures = []
    for aid, m in sorted(missions.items()):
        scenario, veh = m["scenario"], m["veh"]
        view = covlm.bind(scenario, veh)
        chain = resolve_chain(view, m)
        if chain is None:
            failures.append(aid)
            continue
        src = args.interdrive / scenario / ("%s_%d.xml" % (scenario, veh))
        if not src.is_file():
            failures.append("%s (no source %s)" % (aid, src))
            continue
        name = scenario + args.suffix
        write_route(src, routes_dir / name / ("%s_%d.xml" % (name, veh)), chain)
        by_scenario.setdefault(name, {})[veh] = m["utterance"]
        kinds[m.get("via_mode", "none")] += 1

    intents = args.out / "driver_intents_nav.yaml"
    with intents.open("w", encoding="utf-8") as f:
        f.write("# Passenger utterances for the *_nav route directories.\n"
                "#\n"
                "# Generated by covlm/export_nav_routes.py. Keys match the route\n"
                "# DIRECTORY name, because cov2v_bridge keys driver intents by\n"
                "# Path(routes_dir).name -- not by the original scenario name.\n"
                "#\n"
                "# Point [cov2v] driver_intents_path at this file to use them.\n")
        for name in sorted(by_scenario, key=lambda s: (route_id(s), s)):
            f.write("\n%s:\n" % name)
            for veh in sorted(by_scenario[name]):
                said = by_scenario[name][veh].replace('"', '\\"')
                f.write('  veh_%d: "%s"\n' % (veh, said))

    print("[export] %d scenario(s), %d vehicle route(s) -> %s"
          % (len(by_scenario), sum(len(v) for v in by_scenario.values()), routes_dir))
    print("[export] via mode: %s" % dict(kinds))
    print("[export] utterances -> %s" % intents)
    if failures:
        print("[export] %d FAILED: %s" % (len(failures), failures[:5]))
        return 1

    if args.install:
        lb = args.install / "simulation/leaderboard"
        dst_routes = lb / "data/Interdrive"
        dst_yaml = lb / "team_code/agent_config/driver_intents_nav.yaml"
        if not dst_routes.is_dir():
            raise SystemExit("[export] %s is not a CoLMDriver checkout" % args.install)
        added = 0
        for d in sorted(routes_dir.iterdir()):
            target = dst_routes / d.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(d, target)
            added += 1
        shutil.copy2(intents, dst_yaml)
        print("\n[export] installed %d route directories into %s" % (added, dst_routes))
        print("[export] installed utterances at %s" % dst_yaml)
        print("[export] the original Interdrive routes and driver_intents.yaml "
              "were not touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
