"""
benchmark/language/generate.py
==============================
Template stage of the language dataset: multi-turn passenger episodes for
every town that has a knowledge base, covering every function in
``taxonomy.FUNCTIONS``.

Structure first, words second: a recipe fixes what each turn does, the
generator fills it with places that are legal *in that town and that route
state*, the gold intent is built from that choice, and only then is a
sentence rendered. Each gold state is the real ``MissionPlanner.apply`` of the
gold intent, and the run aborts if any function is left uncovered.

    python3 benchmark/language/generate.py            # -> data/episodes_template.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.language import templates as tp                     # noqa: E402
from benchmark.language.state import apply_intent, empty_state, route_key  # noqa: E402
from benchmark.language.taxonomy import FUNCTIONS, tier_of           # noqa: E402
from core.landmark_kb import LandmarkKnowledgeBase                  # noqa: E402
from core.paths import KB_CSV                                       # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "episodes_template.jsonl"


class Infeasible(Exception):
    """This recipe cannot be filled in this town / state; skip it."""


class Town:
    def __init__(self, name: str, kb: LandmarkKnowledgeBase) -> None:
        self.name = name
        self.count = Counter(r["building_type"] for r in kb.get_landmarks(name))
        self.types = sorted(self.count)
        self.indexable = sorted(t for t, c in self.count.items() if c > 1)
        blocked = set(self.types)
        for group in tp.SIMILAR:
            if group & blocked:
                blocked |= group
        self.missing = sorted(set(tp.SURFACE) - blocked)


class Episode:
    def __init__(self, town: Town, rng: random.Random, n_turns: int = 1) -> None:
        self.town = town
        self.rng = rng
        self.n_turns = n_turns
        self.state = empty_state()
        self.turns: list[dict] = []

    # ── choosing places ────────────────────────────────────────────────
    def in_route(self) -> set[str]:
        s = self.state
        return ({s["destination"]} if s["destination"] else set()) | {x["type"] for x in s["stops"]}

    def vague(self, exclude=()) -> tuple[str, list[str]]:
        """A vague category with at least two of its types on this map (minus
        ``exclude``): (phrase, acceptable types)."""
        options = [(name, sorted(types & set(self.town.types) - set(exclude)), phrases)
                   for name, (types, phrases) in sorted(tp.VAGUE.items())]
        options = [o for o in options if len(o[1]) >= 2]
        if not options:
            raise Infeasible
        _, accept, phrases = self.rng.choice(options)
        return self.rng.choice(phrases), accept

    def pick(self, exclude=(), indexable=False, need=False) -> str:
        pool = self.town.indexable if indexable else self.town.types
        pool = [t for t in pool if t not in set(exclude) and (not need or t in tp.NEED)]
        if not pool:
            raise Infeasible
        return self.rng.choice(pool)

    def index(self, t: str) -> int:
        return self.rng.randint(1, self.town.count[t])

    def say(self, t: str) -> str:
        return tp.surface(t, self.rng)

    def fill(self, key: str, **slots) -> tuple[str, str]:
        template = self.rng.choice(tp.T[key])
        caps = {k[0].upper() + k[1:]: v[0].upper() + v[1:] for k, v in slots.items()
                if isinstance(v, str) and v}
        return template.format(**slots, **caps), template

    # ── recording ──────────────────────────────────────────────────────
    def add(self, text: str, template: str, functions: list[str], gold: dict | list | None,
            noisy: bool = False, meta: dict | None = None, accept: dict | None = None) -> None:
        if noisy:
            text = tp.disfluent(text, self.rng)
            functions = functions + ["X5"]
        if gold is not None:                       # always {"commands": [...]}
            gold = {"commands": gold if isinstance(gold, list) else [gold]}
        before = self.state
        after = apply_intent(before, gold)
        changed = route_key(after) != route_key(before) or (
            gold and any(c["command_type"] == "cancel_route" for c in gold["commands"]))
        if (gold is None) == bool(changed):
            raise AssertionError(f"gold does not do what its turn claims: {functions} {gold}")
        setup = not self.turns and self.n_turns > 1
        self.turns.append({"utterance": text, "template": template,
                           "functions": functions, "tier": tier_of(functions, setup),
                           "setup": setup, "meta": meta or {}, "accept": accept or {},
                           "state_before": before, "gold_intent": gold,
                           "state_after": after})
        self.state = after


# ─────────────────────────────────────────────
# Turn builders
# ─────────────────────────────────────────────

def plan(ep: Episode, *, idx_dest=False, stops=0, idx_stop=False, hurry=False,
         pref=False, avoid=False, need=False, selfcorr=False, noisy=False,
         nearest=False, vague=False) -> None:
    rng = ep.rng
    accept = None
    if vague:
        phrase, types = ep.vague()
        dest, accept = rng.choice(types), {"destination": types}
    else:
        dest = ep.pick(indexable=idx_dest or nearest, need=need)
    chosen = []
    for i in range(stops):
        chosen.append(ep.pick(exclude=[dest, *chosen], indexable=idx_stop and i == 0))
    di = ep.index(dest) if idx_dest else None
    si = ep.index(chosen[0]) if idx_stop else None
    p = rng.choice(sorted(tp.PREFERENCE)) if pref else None
    a = rng.choice(sorted(tp.AVOID)) if avoid else None

    slots = {"dest": ep.say(dest), "ordinal": tp.ORDINAL.get(di or si or 1, ""),
             "index": str(di or ""), "stop": ep.say(chosen[0]) if chosen else "",
             "stop2": ep.say(chosen[1]) if stops > 1 else "",
             "hurry": rng.choice(tp.HURRY), "pref": rng.choice(tp.PREFERENCE[p]) if p else "",
             "avoid": rng.choice(tp.AVOID[a]) if a else ""}
    fn = ["P2" if idx_dest else "P1"]
    used = set()                         # modifiers the template already says
    if need:
        slots["need"] = rng.choice(tp.NEED[dest])
        key, fn = "X1", fn + ["X1"]
    elif vague:
        slots["vague"] = phrase
        key, fn = "X7", fn + ["X7"]
    elif nearest:
        key, fn = "X6", fn + ["X6"]
    elif selfcorr:
        wrong = ep.pick(exclude=[dest])
        slots["wrong"] = ep.say(wrong)
        key, fn = "X4", fn + ["X4"]
    elif stops == 2:
        key, fn = "P3x2", ["P3"]
    elif stops == 1:
        key, fn = ("P4", ["P3", "P4"]) if idx_stop else ("P3", ["P3"])
    elif idx_dest:
        key = "P2"
    elif hurry and not (pref or avoid):
        key, used = "P5", {"hurry"}
    elif pref and not (hurry or avoid):
        key, used = "P6", {"pref"}
    elif avoid and not (hurry or pref):
        key, used = "P7", {"avoid"}
    else:
        key = "P1"
    text, template = ep.fill(key, **slots)
    if hurry and "hurry" not in used:
        text += " " + slots["hurry"][0].upper() + slots["hurry"][1:] + "!"
    if pref and "pref" not in used:
        text += " And " + slots["pref"] + "."
    if avoid and "avoid" not in used:
        text += " Please " + slots["avoid"] + "."
    fn += [f for f, on in (("P5", hurry), ("P6", pref), ("P7", avoid)) if on]

    gold = {"command_type": "plan_route", "destination": dest, "destination_index": di,
            "waypoints": [{"type": t, "order": i + 1, "index": si if i == 0 else None}
                          for i, t in enumerate(chosen)],
            "urgency": "high" if hurry else "normal",
            "avoid": [a] if a else [], "route_preference": p}
    ep.add(text, template, fn, gold, noisy, meta={"wrong": wrong} if selfcorr else None,
           accept=accept)


def insert(ep: Episode, position: str, *, need=False, idx=False, noisy=False,
           nearest=False, vague=False) -> None:
    if not ep.state["destination"]:
        raise Infeasible
    accept = None
    if vague:
        phrase, types = ep.vague(exclude=ep.in_route())
        new, accept = ep.rng.choice(types), {"insert_type": types}
    else:
        new = ep.pick(exclude=ep.in_route(), indexable=idx or nearest, need=need)
    ix = ep.index(new) if idx else None
    after = None
    slots = {"new": ep.say(new), "ordinal": tp.ORDINAL.get(ix or 1, "")}
    if vague:
        slots["vague"] = phrase
        key, fn, position = "X7ins", ["I3", "X7"], "enroute"
    elif nearest:
        key, fn, position = "X6ins", ["I3", "X6"], "enroute"
    elif need:
        slots["need"] = ep.rng.choice(tp.NEED[new])
        key, fn, position = "X1ins", ["I3", "X1"], "enroute"
    elif idx:
        key, fn, position = "I5", ["I3", "I5"], "enroute"
    elif position == "after":
        if not ep.state["stops"]:
            raise Infeasible
        after = ep.rng.choice(ep.state["stops"])["type"]
        slots["after"] = ep.say(after)
        key, fn, position = "I4", ["I4"], "enroute"
    else:
        key, fn = {"next": ("I1", ["I1"]), "last": ("I2", ["I2"]),
                   "enroute": ("I3", ["I3"])}[position]
    text, template = ep.fill(key, **slots)
    gold = {"command_type": "insert_stop",
            "insert": {"type": new, "position": position, "index": ix, "after": after}}
    ep.add(text, template, fn, gold, noisy, accept=accept)


def insert_missing(ep: Episode) -> None:
    if not ep.state["destination"] or not ep.town.missing:
        raise Infeasible
    missing = ep.rng.choice(ep.town.missing)
    text, template = ep.fill("X2ins", missing=ep.say(missing))
    ep.add(text, template, ["X2"], None, meta={"missing": missing})


def plan_missing(ep: Episode) -> None:
    if not ep.town.missing:
        raise Infeasible
    missing = ep.rng.choice(ep.town.missing)
    text, template = ep.fill("X2", missing=ep.say(missing))
    ep.add(text, template, ["X2"], None, meta={"missing": missing})


def remove(ep: Episode, noisy=False) -> None:
    if not ep.state["stops"]:
        raise Infeasible
    target = ep.rng.choice(ep.state["stops"])["type"]
    text, template = ep.fill("R1", target=ep.say(target))
    gold = {"command_type": "remove_stop",
            "remove": {"type": target, "fallback": "next_waypoint"}}
    ep.add(text, template, ["R1"], gold, noisy)


def new_destination(ep: Episode, mode: str, noisy=False) -> None:
    """mode: D1 replace, D2 extend, D3 indexed replace, D4 replace in a hurry,
    X3 extend by naming the current destination."""
    if not ep.state["destination"]:
        raise Infeasible
    new = ep.pick(exclude=ep.in_route(), indexable=mode == "D3")
    ix = ep.index(new) if mode == "D3" else None
    slots = {"new": ep.say(new), "ordinal": tp.ORDINAL.get(ix or 1, ""),
             "after": ep.say(ep.state["destination"]), "hurry": ep.rng.choice(tp.HURRY)}
    text, template = ep.fill(mode, **slots)
    fn = {"D1": ["D1"], "D2": ["D2"], "D3": ["D1", "D3"], "D4": ["D1", "D4"],
          "X3": ["D2", "X3"]}[mode]
    gold = {"command_type": "new_destination", "destination": new,
            "destination_index": ix, "keep_waypoints": mode in ("D2", "X3"),
            # Urgency is only gold when the passenger says it; unsaid, the
            # schema default stands (and is not scored -- see evaluate.py).
            "urgency": "high" if mode == "D4" else "normal"}
    ep.add(text, template, fn, gold, noisy)


def cancel(ep: Episode) -> None:
    if not ep.state["destination"]:
        raise Infeasible
    text, template = ep.fill("C1")
    ep.add(text, template, ["C1"], {"command_type": "cancel_route"})


def _ins(t, position="enroute", after=None, before=None, index=None) -> dict:
    return {"command_type": "insert_stop",
            "insert": {"type": t, "position": position, "index": index,
                       "after": after, "before": before}}


def _two_stops_in_order(ep: Episode) -> tuple[str, str]:
    """Two stops (a, b) with a currently before b."""
    stops = [s["type"] for s in ep.state["stops"]]
    if len(stops) < 2:
        raise Infeasible
    i, j = sorted(ep.rng.sample(range(len(stops)), 2))
    return stops[i], stops[j]


def insert_before(ep: Episode, noisy=False) -> None:
    if not ep.state["stops"]:
        raise Infeasible
    other = ep.rng.choice(ep.state["stops"])["type"]
    new = ep.pick(exclude=ep.in_route())
    text, template = ep.fill("I6", new=ep.say(new), other=ep.say(other))
    ep.add(text, template, ["I6"], _ins(new, before=other), noisy)


def move(ep: Episode, where: str) -> None:
    """where: next | last | after | before"""
    stops = [s["type"] for s in ep.state["stops"]]
    if len(stops) < 2:
        raise Infeasible
    if where == "next":
        stop = ep.rng.choice(stops[1:])
        text, template = ep.fill("O1", stop=ep.say(stop))
        cmd, fn = {"type": stop, "position": "next", "after": None, "before": None}, ["O1"]
    elif where == "last":
        stop = ep.rng.choice(stops[:-1])
        text, template = ep.fill("O2", stop=ep.say(stop))
        cmd, fn = {"type": stop, "position": "last", "after": None, "before": None}, ["O2"]
    elif where == "after":                       # a is before b: move a after b
        stop, other = _two_stops_in_order(ep)
        text, template = ep.fill("O3after", stop=ep.say(stop), other=ep.say(other))
        cmd, fn = {"type": stop, "position": None, "after": other, "before": None}, ["O3"]
    else:                                        # a is before b: move b before a
        other, stop = _two_stops_in_order(ep)
        text, template = ep.fill("O3before", stop=ep.say(stop), other=ep.say(other))
        cmd, fn = {"type": stop, "position": None, "after": None, "before": other}, ["O3"]
    ep.add(text, template, fn, {"command_type": "move_stop", "move": cmd})


def multi(ep: Episode, kind: str) -> None:
    s = ep.state
    if not s["destination"]:
        raise Infeasible
    stops = [x["type"] for x in s["stops"]]
    if kind in ("remove_next", "remove_extend", "M2") and not stops:
        raise Infeasible
    target = ep.rng.choice(stops) if stops else None
    taken = ep.in_route()
    new = ep.pick(exclude=taken)
    new2 = ep.pick(exclude=taken | {new})
    slots = {"target": ep.say(target) if target else "", "new": ep.say(new),
             "new2": ep.say(new2), "dest": ep.say(s["destination"])}
    remove_ = {"command_type": "remove_stop", "remove": {"type": target, "fallback": "next_waypoint"}}
    extend = lambda t: {"command_type": "new_destination", "destination": t,
                        "destination_index": None, "keep_waypoints": True,
                        "urgency": "normal"}
    if kind == "remove_next":
        gold, fn = [remove_, _ins(new, "next")], ["M1", "R1", "I1"]
    elif kind == "next_last":
        gold, fn = [_ins(new, "next"), _ins(new2, "last")], ["M1", "I1", "I2"]
    elif kind == "remove_extend":
        gold, fn = [remove_, extend(new)], ["M1", "R1", "D2", "X3"]
    elif kind == "move_last":
        if len(stops) < 2:
            raise Infeasible
        stop = ep.rng.choice(stops[1:])
        slots["stop"] = ep.say(stop)
        gold = [{"command_type": "move_stop",
                 "move": {"type": stop, "position": "next", "after": None, "before": None}},
                _ins(new, "last")]
        fn = ["M1", "O1", "I2"]
    else:                                        # three commands
        gold, fn = [remove_, _ins(new, "next"), extend(new2)], ["M2", "R1", "I1", "D2"]
    text, template = ep.fill("M2" if kind == "M2" else "M1" + kind, **slots)
    ep.add(text, template, fn, gold)


# ─────────────────────────────────────────────
# Recipes: what each episode does, turn by turn
# ─────────────────────────────────────────────

def _p(**kw):
    return lambda ep: plan(ep, **kw)


RECIPES: dict[str, list] = {
    # single turn -- planning
    "dest":            [_p()],
    "dest_indexed":    [_p(idx_dest=True)],
    "one_stop":        [_p(stops=1)],
    "two_stops":       [_p(stops=2)],
    "stop_indexed":    [_p(stops=1, idx_stop=True)],
    "hurry":           [_p(hurry=True)],
    "preference":      [_p(pref=True)],
    "avoid":           [_p(avoid=True)],
    "stops_hurry":     [_p(stops=1, hurry=True)],
    "hurry_pref":      [_p(hurry=True, pref=True)],
    "need":            [_p(need=True)],
    "missing":         [plan_missing],
    "self_correction": [_p(selfcorr=True)],
    "disfluent":       [_p(stops=1, noisy=True)],
    "nearest":         [_p(nearest=True)],
    "vague":           [_p(vague=True)],
    # single turn -- several slots at once (tier 2)
    "dest_indexed_hurry": [_p(idx_dest=True, hurry=True)],
    "stop_indexed_hurry": [_p(stops=1, idx_stop=True, hurry=True)],
    "two_stops_pref":     [_p(stops=2, pref=True)],
    "stops_avoid":        [_p(stops=1, avoid=True)],
    "need_hurry":         [_p(need=True, hurry=True)],
    # two turns -- one edit while driving
    "then_next":       [_p(), lambda ep: insert(ep, "next")],
    "then_last":       [_p(stops=1), lambda ep: insert(ep, "last")],
    "then_unordered":  [_p(stops=1), lambda ep: insert(ep, "enroute")],
    "then_after_stop": [_p(stops=1), lambda ep: insert(ep, "after")],
    "then_indexed":    [_p(stops=1), lambda ep: insert(ep, "enroute", idx=True)],
    "then_need":       [_p(), lambda ep: insert(ep, "enroute", need=True)],
    "then_missing":    [_p(stops=1), insert_missing],
    "then_remove":     [_p(stops=2), remove],
    "then_replace":    [_p(stops=1), lambda ep: new_destination(ep, "D1")],
    "then_extend":     [_p(stops=1), lambda ep: new_destination(ep, "D2")],
    "then_after_dest": [_p(), lambda ep: new_destination(ep, "X3")],
    "then_dest_index": [_p(), lambda ep: new_destination(ep, "D3")],
    "then_rush_elsewhere": [_p(stops=1), lambda ep: new_destination(ep, "D4")],
    "then_nearest":    [_p(stops=1), lambda ep: insert(ep, "enroute", nearest=True)],
    "then_vague":      [_p(), lambda ep: insert(ep, "enroute", vague=True)],
    "then_remove_only": [_p(stops=1), remove],
    "then_crowded":    [_p(stops=2), lambda ep: insert(ep, "enroute"),
                        lambda ep: insert(ep, "after")],
    "then_before":     [_p(stops=1), insert_before],
    "then_move_next":  [_p(stops=2), lambda ep: move(ep, "next")],
    "then_move_last":  [_p(stops=2), lambda ep: move(ep, "last")],
    "then_move_after": [_p(stops=2), lambda ep: move(ep, "after")],
    "then_move_before": [_p(stops=2), lambda ep: move(ep, "before")],
    "multi_remove_next":   [_p(stops=2), lambda ep: multi(ep, "remove_next")],
    "multi_next_last":     [_p(stops=1), lambda ep: multi(ep, "next_last")],
    "multi_remove_extend": [_p(stops=2), lambda ep: multi(ep, "remove_extend")],
    "multi_move_last":     [_p(stops=2), lambda ep: multi(ep, "move_last")],
    "multi_three":         [_p(stops=2), lambda ep: multi(ep, "M2")],
    "then_cancel":     [_p(stops=1), cancel],
    # longer -- edits that build on each other
    "long_a": [_p(stops=1), lambda ep: insert(ep, "enroute"), remove,
               lambda ep: new_destination(ep, "D2")],
    "long_b": [_p(hurry=True), lambda ep: insert(ep, "next"),
               lambda ep: new_destination(ep, "D1"),        # drops the stops
               lambda ep: insert(ep, "enroute"), lambda ep: insert(ep, "after"),
               cancel],
    "long_c": [_p(stops=2), lambda ep: new_destination(ep, "X3"), remove,
               lambda ep: insert(ep, "last", noisy=True)],
}


# Extra episodes per town: recipes that are the only source of a function, so
# none is scored on fewer than 16 turns, and easy recipes, so the scored
# turns thin out from easy to hard.
EXTRA_REPS = {"dest": 3, "one_stop": 3, "two_stops": 3,          # easy, and no setup turn
              "then_next": 2, "then_last": 2, "then_remove_only": 2,
              "then_indexed": 2, "then_dest_index": 2, "then_rush_elsewhere": 2,
              "then_cancel": 2, "self_correction": 2, "then_before": 2,
              "then_move_next": 2, "then_move_last": 2, "multi_three": 2}


def build(kb: LandmarkKnowledgeBase, towns: list[str], reps: int, seed: int) -> list[dict]:
    episodes = []
    for town_name in towns:
        town = Town(town_name, kb)
        for recipe, turns in RECIPES.items():
            for rep in range(max(reps, EXTRA_REPS.get(recipe, 0))):
                for attempt in range(20):           # re-draw places if a fill dead-ends
                    rng = random.Random(f"{seed}/{town_name}/{recipe}/{rep}/{attempt}")
                    ep = Episode(town, rng, len(turns))
                    try:
                        for turn in turns:
                            turn(ep)
                    except Infeasible:
                        continue
                    episodes.append({"id": f"{town_name}/{recipe}/{rep}", "town": town_name,
                                     "recipe": recipe, "source": "template",
                                     "turns": ep.turns})
                    break
                else:
                    print(f"[generate] {town_name}: recipe {recipe!r} cannot be filled, skipped")
    return episodes


def coverage(episodes: list[dict]) -> tuple[dict, dict]:
    per_town: dict[str, Counter] = defaultdict(Counter)
    tiers: Counter = Counter()
    for e in episodes:
        for t in e["turns"]:
            per_town[e["town"]].update(t["functions"])
            tiers[t["tier"]] += 1
    return per_town, tiers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=1, help="episodes per recipe per town")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    kb = LandmarkKnowledgeBase(str(KB_CSV))
    towns = sorted(kb.by_town)
    episodes = build(kb, towns, args.reps, args.seed)
    per_town, tiers = coverage(episodes)

    total = Counter()
    for c in per_town.values():
        total.update(c)
    print(f"[generate] {len(episodes)} episodes, "
          f"{sum(len(e['turns']) for e in episodes)} turns, towns: {', '.join(towns)}")
    print("[generate] tiers: " + ", ".join(f"T{k}={v}" for k, v in sorted(tiers.items())))
    header = "func  " + " ".join(f"{t[-4:]:>6}" for t in towns) + "   total"
    print(header)
    for f, (cmd, desc) in FUNCTIONS.items():
        row = " ".join(f"{per_town[t][f]:>6}" for t in towns)
        print(f"{f:<5} {row}   {total[f]:>5}   {desc}")
    missing = [f for f in FUNCTIONS if total[f] == 0]
    if missing:
        print(f"[generate] FAILED: functions never covered: {missing}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[generate] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
