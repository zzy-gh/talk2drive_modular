#!/usr/bin/env python3
"""
covlm/design_passenger_inputs.py
================================
Make the destination come from the passenger's words.

Without this, a CoVLM run cannot take its destination from what the passenger
says. Town05's knowledge base holds 12 building types -- bakery, cafe, cinema,
hotel and the like -- and the r1-r46 utterances are about hospitals, homes and
schools, because they were written to carry *negotiation priority*, not
navigation. 73 of 175 name a type that town has no entry for, so
``GeminiCommandParser.parse`` rejects the LLM's perfectly correct answer, and
another 67 name no destination at all. 115 of 175 cannot work.

This script closes that gap by producing three artefacts:

  data/covlm_landmarks.csv       one landmark per vehicle route, sitting ON that
                                 route's end point. "Take me to the hospital"
                                 then resolves to a coordinate that IS where the
                                 benchmark route ends -- so the route matches
                                 *because* the passenger asked for it, not
                                 because a coordinate was injected.
  data/driver_intents_nav.yaml   one utterance per vehicle, naming that landmark
                                 and keeping the original's urgency tier.
  data/passenger_missions.json   the mapping, with provenance for every choice.

Two constraints decide the label assignment:

  uniqueness   from this vehicle's start, its own landmark must be the NEAREST
               one of that type -- otherwise "take me to the hospital" drives to
               a different hospital and the route is wrong. Competitors are the
               town's real landmarks and the other vehicles in the same scenario.
  a template   the label must be one CoLMDriver's own intent_templates.py covers
               in all three tiers, so the rewritten utterance reads like the
               originals rather than like generated filler.

28 types satisfy both, and all 175 vehicles can be assigned inside that set.
Original utterances are kept verbatim wherever the type they already name
survives both constraints; only the rest are rewritten.

    python covlm/design_passenger_inputs.py [--templates <CoLMDriver>/scripts/scenario_gen]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.intent_schema import JSON_SCHEMA_STR                               # noqa: E402
from core.landmark_kb import LandmarkKnowledgeBase                           # noqa: E402
from core.paths import KB_CSV                                                # noqa: E402

from covlm.anchors import AnchorBook                                         # noqa: E402
from covlm.paths import DATA_DIR, ROUTE_ANCHORS                              # noqa: E402
from covlm.utterance_overrides import override_for                           # noqa: E402

DEF_TEMPLATES = Path("/home/dellpro2/Zhiyuan/covlm-agent-main/CoLMDriver-main"
                     "/scripts/scenario_gen")

# ─────────────────────────────────────────────
# Urgency tier, read out of the original wording
# ─────────────────────────────────────────────
#
# The tier is the one thing a rewrite must not change: CoLMDriver's negotiation
# scores which vehicle's claim is stronger by reading it out of the words, and
# the sweep in scripts/eval/run_intent_sweep.sh compares runs with and without
# these utterances. Change a tier and that comparison stops meaning anything.

HIGH = (r"(emergency|\bER\b|urgent care|collapsed|in labor|labor|chest pain|"
        r"seizure|stroke|allergic reaction|can't get up|can't stand|high fever|"
        r"dangerously|not breathing|trouble breathing|fire|smoking|gas leak|"
        r"leaking|flooding|rising fast|immediately|as fast as|every minute counts|"
        r"racing|rushing|stranded|scared|crying|locked out|won't stop|diabetic|"
        r"oxygen|transplant|dialysis|premature|surgery|fell|broke his leg|"
        r"panic attack|asthma|medication needs)")
LOW = (r"(no rush|no hurry|nothing urgent|nothing pressing|nothing time|"
       r"no particular|no schedule|no plans|no real plans|no destination|"
       r"no time pressure|no deadline|leisurely|casual|relaxed|quiet|slow|"
       r"killing time|just cruising|just driving|just passing|just out|"
       r"just heading|just taking|just meeting|just running|just going|fresh air)")


def tier_of(utterance: str) -> str:
    if re.search(HIGH, utterance or "", re.I):
        return "high"
    if re.search(LOW, utterance or "", re.I):
        return "low"
    return "med"


def schema_types() -> list[str]:
    body = re.search(r"Valid BuildingType values:\n(.*)", JSON_SCHEMA_STR, re.S).group(1)
    return [t.strip() for t in body.replace("\n", " ").split(",") if t.strip()]


# ─────────────────────────────────────────────
# Label assignment
# ─────────────────────────────────────────────

def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def assign_labels(book: AnchorBook, kb, usable: list[str], tiers: dict) -> dict:
    """Give every vehicle a label its own anchor is the nearest instance of.

    Greedy, but ordered so the result is stable and as close to the original
    data as possible: the type the vehicle's own utterance already implies is
    tried first, then types the town genuinely has (so the wording suits the
    map), then the rest. Vehicles with the fewest legal options choose first.
    """
    out = {}
    for scenario in book.scenarios():
        vehicles = book.for_scenario(scenario)
        town = vehicles[0].town
        town_types = [t for t in usable if kb.find_all(town, t)]
        rest = [t for t in usable if t not in town_types]

        def legal_vs_kb(a, label):
            own = _d(a.start, a.goal)
            return all(_d(a.start, (r["x"], r["y"])) >= own
                       for r in kb.find_all(a.town, label))

        options = {v.veh: [t for t in usable if legal_vs_kb(v, t)] for v in vehicles}
        taken: list[tuple] = []          # (anchor, label) already assigned here

        for v in sorted(vehicles, key=lambda v: (len(options[v.veh]), v.veh)):
            own = _d(v.start, v.goal)
            preferred = ([v.label] if v.label_source == "utterance" else []) \
                + town_types + rest
            seen, order = set(), []
            for t in preferred:
                if t in options[v.veh] and t not in seen:
                    seen.add(t)
                    order.append(t)

            # A rewritten utterance is TEMPLATES[label][tier], so two vehicles in
            # one scenario landing on the same (label, tier) would say exactly
            # the same sentence -- and a scenario where two cars make an
            # identical claim tells the negotiation nothing. Keeping the
            # vehicle's own original wording still wins (those are all distinct);
            # below that, prefer a label whose (label, tier) is still free here.
            mine = tiers[v.anchor_id]
            used_pairs = {(lbl, tiers[o.anchor_id]) for o, lbl in taken}
            keepable = order[:1] if (order and order[0] == v.label
                                     and v.label_source == "utterance") else []
            rest = [t for t in order if t not in keepable]
            order = keepable + sorted(rest, key=lambda t: (t, mine) in used_pairs)

            pick, why = None, ""
            for t in order:
                if _clash(v, t, taken):
                    continue
                pick = t
                why = ("kept" if t == v.label and v.label_source == "utterance"
                       else "town" if t in town_types else "fallback")
                break
            if pick is None:
                pick, why = v.label, "UNRESOLVED"
            taken.append((v, pick))
            out[v.anchor_id] = {"label": pick, "how": why}

        # Greedy is order-dependent: a vehicle assigned early cannot know that a
        # later one will take the same label and sit nearer to it. _clash is
        # symmetric, so that case is only *detected* once both are placed --
        # repair it by moving whichever vehicle still has a legal alternative.
        for _ in range(len(vehicles) * len(usable)):
            bad = next(((v, lbl) for v, lbl in taken if _clash(v, lbl, taken)), None)
            if bad is None:
                break
            v, _lbl = bad
            others = [(o, l) for o, l in taken if o is not v]
            alt = next((t for t in options[v.veh] if not _clash(v, t, others)), None)
            if alt is None:
                out[v.anchor_id] = {"label": _lbl, "how": "UNRESOLVED"}
                taken = others + [(v, _lbl)]
                continue
            taken = others + [(v, alt)]
            out[v.anchor_id] = {"label": alt, "how": "repaired"}
    return out


def _clash(v, label, taken) -> bool:
    """Would label ``label`` on ``v`` break nearest-of-type for anyone?

    Symmetric on purpose. "Take me to the hospital" resolves to the nearest
    hospital from where you are, so the constraint is violated if some other
    vehicle's landmark of the same type is nearer to ME than my own, *and* if
    mine is nearer to THEM than theirs -- either way one of us drives to the
    wrong place.
    """
    own = _d(v.start, v.goal)
    for o, lbl in taken:
        if o is v or lbl != label:
            continue
        if _d(v.start, o.goal) < own:
            return True
        if _d(o.start, v.goal) < _d(o.start, o.goal):
            return True
    return False


# ─────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────

def verify(book: AnchorBook, kb, assignment: dict) -> list[str]:
    """Would 'take me to the <label>' actually resolve to this anchor?"""
    problems = []
    for scenario in book.scenarios():
        vehicles = book.for_scenario(scenario)
        for v in vehicles:
            label = assignment[v.anchor_id]["label"]
            own = _d(v.start, v.goal)
            rivals = [((r["x"], r["y"]), "kb:" + r.get("carla_name", "?"))
                      for r in kb.find_all(v.town, label)]
            rivals += [((o.goal[0], o.goal[1]), "anchor:veh%d" % o.veh)
                       for o in vehicles
                       if o.veh != v.veh and assignment[o.anchor_id]["label"] == label]
            nearer = [(round(_d(v.start, xy), 1), src) for xy, src in rivals
                      if _d(v.start, xy) < own]
            if nearer:
                problems.append("%s wants '%s' at %.1f m but %s is at %.1f m"
                                % (v.anchor_id, label, own,
                                   sorted(nearer)[0][1], sorted(nearer)[0][0]))
    return problems




STOP_CLAUSE = {
    "low":  "Swing by",
    "med":  "Stop at",
    "high": "Pull in at",
}


def _spoken(label: str) -> str:
    """'gas_station' -> 'gas station'. The LLM maps the words back itself."""
    return label.replace("_", " ")


def _decapitalise(sentence: str) -> str:
    """Lower the first letter, except where that would write the pronoun 'I'
    as 'i' -- which is what a first pass did to every "I'm rushing ..." line."""
    if re.match(r"I\b|I'", sentence):
        return sentence
    return sentence[0].lower() + sentence[1:]


def compose_with_stop(destination_sentence: str, stop_labels: list, tier: str) -> str:
    """Put the stops in front of the destination sentence, as a passenger would.

    The intent schema already models this -- ``plan_route.waypoints`` is a list
    of Stops -- and core/intent_schema.py's own examples are exactly this shape
    ("Stop at the pharmacy first, then take me to the hospital."). So a route
    keypoint becomes a real thing the passenger asked for, resolved through the
    knowledge base like the destination, rather than a coordinate smuggled in
    beside the language.

    Stops are named in travel order; composing them one at a time reverses it,
    which is how a first pass told a driver to reach the farm before the bus
    stop it actually passes first. The destination sentence is left intact so
    its urgency wording -- what CoLMDriver's negotiation reads -- is not diluted.
    """
    if not stop_labels:
        return destination_sentence
    verb = STOP_CLAUSE.get(tier, STOP_CLAUSE["med"])
    names = [("the %s" % _spoken(l)) for l in stop_labels]
    if len(names) == 1:
        clause = "%s %s on the way" % (verb, names[0])
    else:
        clause = "%s %s and then %s on the way" % (verb, ", ".join(names[:-1]), names[-1])
    return "%s, then %s" % (clause, _decapitalise(destination_sentence))


def assign_stop_labels(book, kb, usable, min_vias, dest, tiers):
    """Give every spoken stop a label whose nearest instance IS its own keypoint.

    Adding a stop changes where the destination is looked up from: the mission
    becomes start -> stop -> goal, and ``MissionPlanner`` resolves each leg from
    where the car currently is. So the whole chain has to hold, not just one hop.

    Assignment is greedy but validated with full information afterwards. Greedy
    alone is not enough, and the failure is quiet: placing r6's veh0 before veh1
    let veh1's later bus stop land 5 m from veh0's start, closer than veh0's own
    at 18 m, so veh0 drove to the wrong one and its route was 3.6 m off. Nothing
    detects that until every landmark exists, hence the repair loop.

    Returns {anchor_id: [(label, point), ...]} in travel order.
    """
    out = {}
    for scenario in book.scenarios():
        vehicles = book.for_scenario(scenario)
        town = vehicles[0].town
        need = [(v, spec["points"]) for v in vehicles
                for spec in [min_vias.get(v.anchor_id) or {}]
                if spec.get("kind") == "stop" and spec.get("points")]
        if not need:
            continue

        state = {v.anchor_id: [None] * len(pts) for v, pts in need}
        points = {v.anchor_id: [tuple(p) for p in pts] for v, pts in need}
        prefs = {v.anchor_id: _stop_preference(usable, town, kb,
                                               dest[v.anchor_id]["label"])
                 for v, _ in need}

        def landmarks():
            """Everything this scenario places, once all choices are known."""
            out_ = [(dest[v.anchor_id]["label"], (v.goal[0], v.goal[1])) for v in vehicles]
            for aid, labels in state.items():
                for lbl, pt in zip(labels, points[aid]):
                    if lbl:
                        out_.append((lbl, (pt[0], pt[1])))
            return out_

        for v, _ in need:
            for i, pt in enumerate(points[v.anchor_id]):
                state[v.anchor_id][i] = next(
                    (t for t in prefs[v.anchor_id]
                     if _chain_ok(v, state, points, dest, kb, landmarks, vehicles,
                                  trial=(v.anchor_id, i, t))), None)

        for _ in range(64):
            broken = _first_broken(vehicles, state, points, dest, kb, landmarks)
            if broken is None:
                break
            aid, i = broken
            v = next(x for x in vehicles if x.anchor_id == aid)
            current = state[aid][i]
            alt = next((t for t in prefs[aid] if t != current
                        and _chain_ok(v, state, points, dest, kb, landmarks, vehicles,
                                      trial=(aid, i, t))), None)
            state[aid][i] = alt          # None marks it unplaceable, loudly
            if alt is None:
                break

        for v, _ in need:
            out[v.anchor_id] = list(zip(state[v.anchor_id], points[v.anchor_id]))
    return out


def _stop_preference(usable, town, kb, dest_label):
    """Types the town actually has come first -- the wording then suits the map
    -- and never the vehicle's own destination type, which would make
    'stop at the bakery, then to the bakery' resolve to one place."""
    town_types = [t for t in usable if kb.find_all(town, t) and t != dest_label]
    rest = [t for t in usable if t not in town_types and t != dest_label]
    return town_types + rest


def _nearest(label, frm, kb, town, placed):
    best, best_d = None, float("inf")
    for r in kb.find_all(town, label):
        d = _d(frm, (r["x"], r["y"]))
        if d < best_d:
            best, best_d = (r["x"], r["y"]), d
    for lbl, xy in placed:
        if lbl != label:
            continue
        d = _d(frm, xy)
        if d < best_d:
            best, best_d = xy, d
    return best


def _walk_ok(v, labels, pts, dest, kb, placed) -> bool:
    """Does resolving this vehicle's mission leg by leg land on its own points?"""
    here = (v.start[0], v.start[1])
    for lbl, pt in zip(labels, pts):
        if lbl is None:
            return False
        want = (pt[0], pt[1])
        if _nearest(lbl, here, kb, v.town, placed) != want:
            return False
        here = want
    goal = (v.goal[0], v.goal[1])
    return _nearest(dest[v.anchor_id]["label"], here, kb, v.town, placed) == goal


def _chain_ok(v, state, points, dest, kb, landmarks, vehicles, trial) -> bool:
    aid, i, label = trial
    saved = state[aid][i]
    state[aid][i] = label
    try:
        placed = landmarks()
        return all(_walk_ok(o, state.get(o.anchor_id, []), points.get(o.anchor_id, []),
                            dest, kb, placed) for o in vehicles
                   if all(x is not None for x in state.get(o.anchor_id, [])))
    finally:
        state[aid][i] = saved


def _first_broken(vehicles, state, points, dest, kb, landmarks):
    placed = landmarks()
    for v in vehicles:
        labels = state.get(v.anchor_id, [])
        if not labels:
            continue
        if not _walk_ok(v, labels, points.get(v.anchor_id, []), dest, kb, placed):
            for i, lbl in enumerate(labels):
                if lbl is None:
                    return (v.anchor_id, i)
            return (v.anchor_id, 0)
    return None


def _read_feedback(report, ledger_path, missions_path) -> dict:
    """Utterances the LLM has ever failed to resolve, accumulated.

    A report is generated from whatever utterances were current when it ran, so
    a fixed line stops appearing in it -- and reading only the latest report
    makes the fix erase its own evidence and flip straight back. (It did: 14
    rewrites reverted to 13 originals on the next pass.) So failures are kept in
    a ledger on disk instead.

    The ledger stores the exact sentence that failed, not just the vehicle, so
    it expires by itself: edit that line, or a template it came from, and the
    entry no longer matches anything and stops blocking.

    Returns {anchor_id: failed_utterance}.
    """
    import csv as _csv
    import json as _json

    ledger = {}
    if ledger_path and Path(ledger_path).is_file():
        ledger = _json.loads(Path(ledger_path).read_text(encoding="utf-8"))

    if report and Path(report).is_file() and missions_path and Path(missions_path).is_file():
        prev = _json.loads(Path(missions_path).read_text(encoding="utf-8"))["missions"]
        with open(report, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if r.get("dest_ok") == "1":
                    continue
                aid = r["anchor_id"]
                said = (prev.get(aid) or {}).get("utterance")
                if said:
                    ledger.setdefault(aid, [])
                    if said not in ledger[aid]:
                        ledger[aid].append(said)

    if ledger_path:
        Path(ledger_path).write_text(_json.dumps(ledger, indent=1, ensure_ascii=False),
                                     encoding="utf-8")
    return ledger



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", type=Path, default=ROUTE_ANCHORS)
    ap.add_argument("--templates", type=Path, default=DEF_TEMPLATES,
                    help="directory holding CoLMDriver's intent_templates.py")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--min-vias", type=Path, default=DATA_DIR / "min_vias.json",
                    help="output of covlm/compute_min_vias.py. Keypoints marked "
                         "'stop' become spoken waypoints with their own landmark; "
                         "'manoeuvre' ones stay route-shape data.")
    ap.add_argument("--ledger", type=Path, default=DATA_DIR / "utterance_ledger.json",
                    help="accumulated record of sentences the LLM failed on")
    ap.add_argument("--feedback", type=Path,
                    default=DATA_DIR / "passenger_chain_report.csv",
                    help="a covlm/verify_passenger_chain.py report. Any vehicle "
                         "whose ORIGINAL utterance the LLM failed to resolve to "
                         "its assigned type is rewritten from a template instead "
                         "of being kept. Absent on a first run; re-run this "
                         "script after the chain check to apply it.")
    args = ap.parse_args()

    sys.path.insert(0, str(args.templates))
    try:
        from intent_templates import TEMPLATES
    except ImportError as exc:
        raise SystemExit("[design] cannot import intent_templates from %s (%s).\n"
                         "         Pass --templates <CoLMDriver>/scripts/scenario_gen."
                         % (args.templates, exc))

    kb = LandmarkKnowledgeBase(str(KB_CSV))
    book = AnchorBook.load(args.anchors)
    usable = sorted(t for t in set(TEMPLATES) & set(schema_types())
                    if {"low", "med", "high"} <= set(TEMPLATES[t]))
    print("[design] %d types have a template in all three tiers and are legal "
          "in the intent schema" % len(usable))

    min_vias = {}
    if args.min_vias and Path(args.min_vias).is_file():
        min_vias = json.loads(Path(args.min_vias).read_text(encoding="utf-8"))["vias"]
        kinds = Counter(v["kind"] for v in min_vias.values())
        print("[design] min_vias: %s" % dict(kinds))

    tiers = {a.anchor_id: tier_of(a.utterance) for a in book}
    assignment = assign_labels(book, kb, usable, tiers)
    stops = assign_stop_labels(book, kb, usable, min_vias, assignment, tiers)
    n_stop_pts = sum(len(v) for v in stops.values())
    n_unplaced = sum(1 for v in stops.values() for lbl, _ in v if lbl is None)
    if stops:
        print("[design] spoken stops: %d point(s) across %d vehicle(s)%s"
              % (n_stop_pts, len(stops),
                 ", %d UNPLACEABLE" % n_unplaced if n_unplaced else ""))
    problems = verify(book, kb, assignment)

    # An original utterance is only worth keeping if the LLM actually reads the
    # assigned type out of it. "Trying to make it to a job orientation on time"
    # names a reason, not a place: Gemini returns nothing, or picks 'store' over
    # 'factory' -- both defensible, both the wrong coordinate. Checking that the
    # type merely *exists* in the knowledge base, which is all the first pass
    # did, cannot catch either. So the chain check's verdict feeds back here.
    ledger = _read_feedback(args.feedback, args.ledger, args.out_dir / "passenger_missions.json")
    if ledger:
        print("[design] ledger: %d vehicle(s) have an utterance the LLM has failed "
              "to resolve; those exact sentences will not be used again"
              % len(ledger))

    missions, rows = {}, []
    kept = 0
    for a in book:
        label = assignment[a.anchor_id]["label"]
        how = assignment[a.anchor_id]["how"]
        tier = tier_of(a.utterance)
        # Escalate only as far as the measurements force: keep the original,
        # fall back to the template when the LLM could not resolve the original,
        # and reach for an override only when the template failed too.
        blocked = set(ledger.get(a.anchor_id, []))
        my_stops = [(lbl, pt) for lbl, pt in stops.get(a.anchor_id, []) if lbl]
        spec = min_vias.get(a.anchor_id) or {}
        shape_vias = spec["points"] if spec.get("kind") == "manoeuvre" else []
        stop_labels = [lbl for lbl, _ in my_stops]

        # Escalate on the FINAL sentence, not on the base clause. A stop clause
        # is prepended after the base is chosen, so judging the base alone lets
        # a composed sentence the LLM has already failed on be rebuilt verbatim
        # every run -- which is exactly what kept r20's veh2 stuck.
        candidates = []
        if how == "kept":
            candidates.append((a.utterance, "kept"))
        candidates.append((TEMPLATES[label][tier],
                           "template_after_llm_feedback" if how == "kept" else how))
        alt = override_for(label, tier)
        if alt:
            candidates.append((alt, "override_after_llm_feedback"))

        utterance, how = None, "STILL_BLOCKED"
        for base, tag in candidates:
            said = compose_with_stop(base, stop_labels, tier)
            if said not in blocked:
                utterance, how = said, tag
                break
        if utterance is None:
            base, tag = candidates[-1]
            utterance, how = compose_with_stop(base, stop_labels, tier), "STILL_BLOCKED"
        keep_original = (how == "kept")

        for lbl, pt in my_stops:
            rows.append({"town": a.town, "building_type": lbl,
                         "carla_name": "covlm_%s_veh%d_stop" % (a.scenario, a.veh),
                         "x": pt[0], "y": pt[1], "z": pt[2],
                         "scenario": a.scenario, "veh": a.veh})

        kept += int(keep_original)
        missions[a.anchor_id] = {
            "stops": [{"label": lbl, "point": list(pt)} for lbl, pt in my_stops],
            "shape_vias": [list(p) for p in shape_vias],
            "via_mode": spec.get("kind", "none"),
            "scenario": a.scenario, "veh": a.veh, "town": a.town,
            "label": label, "label_how": how, "tier": tier,
            "utterance": utterance, "kept_original": keep_original,
            "original_utterance": a.utterance,
            "goal": list(a.goal), "start": list(a.start),
            "vias": [list(v) for v in a.vias],
        }
        rows.append({"town": a.town, "building_type": label,
                     "carla_name": "covlm_%s_veh%d" % (a.scenario, a.veh),
                     "x": a.goal[0], "y": a.goal[1], "z": a.goal[2],
                     "scenario": a.scenario, "veh": a.veh})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "covlm_landmarks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["town", "building_type", "carla_name",
                                          "x", "y", "z", "scenario", "veh"])
        w.writeheader()
        w.writerows(rows)

    yaml_path = args.out_dir / "driver_intents_nav.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("# Passenger utterances for the CoVLM nav runs, by scenario and vehicle.\n"
                "#\n"
                "# Generated by covlm/design_passenger_inputs.py -- do not hand-edit;\n"
                "# edit the generator or the templates it reads.\n"
                "#\n"
                "# Every line names a destination that exists in covlm_landmarks.csv AND\n"
                "# sits on that vehicle's route end, so the route follows from the words.\n"
                "# The urgency tier is carried over from the original driver_intents.yaml\n"
                "# unchanged, because CoLMDriver's negotiation scores that, not the\n"
                "# destination.\n")
        for scenario in book.scenarios():
            f.write("\n%s:\n" % scenario)
            for v in book.for_scenario(scenario):
                m = missions[v.anchor_id]
                f.write("  # veh_%d -> %s at (%.1f, %.1f), tier %s%s\n"
                        % (v.veh, m["label"], m["goal"][0], m["goal"][1], m["tier"],
                           ", original kept" if m["kept_original"] else ""))
                f.write('  veh_%d: "%s"\n' % (v.veh, m["utterance"].replace('"', '\\"')))

    json_path = args.out_dir / "passenger_missions.json"
    json_path.write_text(json.dumps({"meta": {
        "anchors": str(args.anchors), "templates": str(args.templates),
        "usable_types": usable, "kept_original": kept, "vehicles": len(book),
    }, "missions": missions}, indent=2), encoding="utf-8")

    how = Counter(v["how"] for v in assignment.values())
    tiers = Counter(m["tier"] for m in missions.values())
    labels = Counter(m["label"] for m in missions.values())
    per_town = defaultdict(Counter)
    for m in missions.values():
        per_town[m["town"]][m["label"]] += 1

    print("[design] label source     : %s" % dict(how))
    print("[design] urgency tiers    : %s" % dict(tiers))
    print("[design] labels used (%d) : %s" % (len(labels), dict(labels.most_common(10))))
    print("[design] original utterance kept verbatim: %d/%d" % (kept, len(book)))
    print("[design] -> %s" % csv_path)
    print("[design] -> %s" % yaml_path)
    print("[design] -> %s" % json_path)
    if problems:
        print("\n[design] %d vehicle(s) would resolve to the WRONG landmark:" % len(problems))
        for p in problems[:15]:
            print("[design]      %s" % p)
        return 1
    print("[design] every vehicle's own landmark is the nearest of its type "
          "from its own start -- 'take me to the <type>' resolves to the route end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
