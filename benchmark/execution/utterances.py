"""
benchmark/execution/utterances.py
=================================
Utterances for the execution scenarios, for end-to-end runs.

Each event's sentence is a reviewed language-benchmark turn with the same
command structure (commands, positions, anchors, urgency, stop count), its
place names replaced by the scenario's and re-checked for faithfulness.
Turns that never name the place, name two, say "nearest", or carry a
preference or avoid are not used.

Places are named by type; the scripted passenger answers clarifications with
the scenario's landmark ID. Places that type-only grounding cannot reach
are said with their ID ("bus stop number 3"), found by offline replay.

Adds ``utterance``, ``said``, ``source`` and ``expect_state`` per event and
``answer_ids`` per scenario.

    python3 benchmark/execution/utterances.py   # -> data/scenarios_language.jsonl
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.execution import generate as G                        # noqa: E402
from benchmark.language import templates as tp                       # noqa: E402
from benchmark.language.paraphrase import faithful, required         # noqa: E402
from core.intent_schema import commands_of                           # noqa: E402
from core.landmark_kb import LandmarkKnowledgeBase                   # noqa: E402
from core.paths import KB_CSV, section_path                          # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "scenarios_language.jsonl"
REVIEW = HERE / "data" / "utterances_review.csv"
LANGUAGE = ROOT / "benchmark" / "language" / "data" / "episodes_reviewed.jsonl"

# The words of these turns never name the place, name a second one, or
# carry something the scenario does not have.
UNUSABLE = {"X1", "X2", "X4", "X6", "X7", "P6", "P7"}


# ─────────────────────────────────────────────
# Structure: what must match between an event and a source turn
# ─────────────────────────────────────────────

def _roles(intent: dict) -> list[tuple[str, int | None]]:
    """(type, index) of every place the intent names, in a fixed order."""
    out = []
    for c in commands_of(intent):
        cmd = c["command_type"]
        if cmd == "plan_route":
            out += [(w["type"], w.get("index")) for w in c["waypoints"]]
            out.append((c["destination"], c.get("destination_index")))
        elif cmd == "insert_stop":
            i = c["insert"]
            out.append((i["type"], i.get("index")))
            out += [(i[k], None) for k in ("after", "before") if i.get(k)]
        elif cmd == "remove_stop":
            out.append((c["remove"]["type"], None))
        elif cmd == "move_stop":
            mv = c["move"]
            out.append((mv["type"], None))
            out += [(mv[k], None) for k in ("after", "before") if mv.get(k)]
        elif cmd == "new_destination":
            out.append((c["destination"], c.get("destination_index")))
    return out


def shape(intent: dict) -> tuple:
    """The intent with every place type and index taken out."""
    out = []
    for c in commands_of(intent):
        cmd = c["command_type"]
        if cmd == "plan_route":
            out.append((cmd, len(c["waypoints"]), c.get("urgency", "normal")))
        elif cmd == "insert_stop":
            i = c["insert"]
            out.append((cmd, i.get("position"), bool(i.get("after")), bool(i.get("before"))))
        elif cmd == "move_stop":
            mv = c["move"]
            out.append((cmd, mv.get("position"), bool(mv.get("after")), bool(mv.get("before"))))
        elif cmd == "new_destination":
            out.append((cmd, bool(c.get("keep_waypoints")), c.get("urgency", "normal")))
        else:
            out.append((cmd,))
    return tuple(out)


def strip_indexes(intent: dict, keep: set[str]) -> dict:
    """The intent as the passenger says it: indexes only for ``keep`` types."""
    it = copy.deepcopy(intent)
    for c in commands_of(it):
        if "destination_index" in c and c.get("destination") not in keep:
            c["destination_index"] = None
        for w in c.get("waypoints") or []:
            if w["type"] not in keep:
                w["index"] = None
        if c.get("insert") and c["insert"]["type"] not in keep:
            c["insert"]["index"] = None
    return it


# ─────────────────────────────────────────────
# Offline replay: expected states, and which IDs must be said
# ─────────────────────────────────────────────

class ById(G.Scripted):
    """Answers with the scenario's ID if offered, else the first option (recorded as a miss)."""

    def __init__(self, ids: dict) -> None:
        super().__init__()
        self.ids = ids
        self.asked = []

    def choose_place(self, req):
        want = self.ids.get(req.label)
        offered = [p.index for p in req.candidates]
        self.asked.append({"label": req.label, "want": want, "offered": offered})
        if want in offered:
            return req.candidates[offered.index(want)]
        return req.nearest()


def snapshot(m, result) -> dict:
    """Where the mission stands after an event: what scoring a parse compares."""
    if result.cancelled:
        return {"cancelled": True}
    plan = m.plan
    return {"cancelled": False, "goal": f"{plan.goal.label}#{plan.goal.index}",
            "stops": [f"{p.label}#{p.index}" for p in plan.stops],
            "urgency": m.directive.urgency}


def answer_ids(sc: dict) -> dict:
    ids = {}
    for e in sc["events"]:
        for t, i in _roles(e["intent"]):
            if i:
                ids[t] = i
    return ids


def replay(town, sc: dict, events: list[dict], inter=None) -> tuple[list, bool]:
    """Mission state after each event, and whether every event fired."""
    states: list = []
    try:
        G.drive(town, sc["start"], events, inter=inter,
                on_fire=lambda m, result: states.append(snapshot(m, result)))
    except G.Infeasible:
        return states, False
    return states, len(states) == len(events)


def ids_to_say(town, sc: dict, expect: list) -> set[str]:
    """Types that need an ID because type-only grounding picks another place."""
    ids = answer_ids(sc)
    valid = {f"{t}#{i}" for t, i in ids.items()}
    say: set[str] = set()
    for _ in range(len(ids) + 1):
        events = [dict(e, intent=strip_indexes(e["intent"], say)) for e in sc["events"]]
        inter = ById(ids)
        got, _ = replay(town, sc, events, inter)
        if got == expect:
            return say
        wrong = {a["label"] for a in inter.asked if a["want"] not in a["offered"]}
        wrong |= {k.split("#")[0] for g in got for k in [g.get("goal")] + g.get("stops", [])
                  if k and k not in valid}
        wrong = (wrong - say) or (set(ids) - say)    # nothing to blame: say one more
        if not wrong:
            break
        say |= wrong
    raise RuntimeError(f"{sc['id']}: even with every ID said the replay differs")


# ─────────────────────────────────────────────
# Place-name substitution
# ─────────────────────────────────────────────

def _names(t: str) -> list[str]:
    return [s.lower() for s in tp.SURFACE.get(t, [t.replace("_", " ")])] + [t.replace("_", " ")]


_ALL = sorted({(n, t) for t in tp.SURFACE for n in _names(t)}, key=lambda x: -len(x[0]))
_MENTION = re.compile(r"\b(" + "|".join(re.escape(n) for n, _ in _ALL) + r")\b", re.I)
_TYPE_OF = {n: t for n, t in _ALL}
_ORD_WORDS = {n: {w, str(n), f"{n}st", f"{n}nd", f"{n}rd", f"{n}th"}
              for n, w in tp.ORDINAL.items()}


def _say(t: str) -> str:
    return tp.SURFACE.get(t, [t.replace("_", " ")])[0]


def substitute(text: str, mapping: dict, said_index: dict, source_index: dict) -> str | None:
    """``text`` with every place of ``mapping`` (source type -> scenario type)
    renamed. ``said_index``: scenario type -> ID the passenger says;
    ``source_index``: source type -> the index the source sentence says.
    None when a mention cannot be mapped."""
    bad = []
    # Adjacent surface forms of one place ("university campus") are one mention.
    text = re.sub(r"\b(" + "|".join(re.escape(n) for n, _ in _ALL) + r")\s+("
                  + "|".join(re.escape(n) for n, _ in _ALL) + r")\b",
                  lambda m: m.group(1) if _TYPE_OF[m.group(1).lower()] == _TYPE_OF[m.group(2).lower()]
                  and m.group(1).lower() != m.group(2).lower() else m.group(0), text, flags=re.I)

    def rename(m: re.Match) -> str:
        old = _TYPE_OF[m.group(1).lower()]
        if old not in mapping:
            bad.append(old)
            return m.group(0)
        new = _say(mapping[old])
        head = not text[:m.start()].strip() or text[:m.start()].rstrip()[-1] in ".!?"
        if head and not new.isupper():
            new = new[0].upper() + new[1:]
        if mapping[old] in said_index and old not in source_index:
            new += f" number {said_index[mapping[old]]}"
        return new

    out = _MENTION.sub(rename, text)
    if bad:
        return None
    # An ordinal the source said is now the scenario's ID.
    for old, n_old in source_index.items():
        n_new = said_index.get(mapping[old])
        if n_new is None:
            return None
        if n_new == n_old:
            continue
        pat = r"\b(" + "|".join(re.escape(w) for w in sorted(_ORD_WORDS.get(n_old, {str(n_old)}),
                                                              key=len, reverse=True)) + r")\b"
        new_word = tp.ORDINAL.get(n_new, str(n_new))
        out, k = re.subn(pat, lambda m: new_word if not m.group(1).isdigit() else str(n_new),
                         out, count=1, flags=re.I)
        if k == 0:
            return None
    # a/an before a renamed place
    out = re.sub(r"\b([Aa])n? (?=([A-Za-z]))",
                 lambda m: (m.group(1) + ("n " if m.group(2).lower() in "aeiou" else " ")), out)
    return out


# ─────────────────────────────────────────────
# Matching events to source turns
# ─────────────────────────────────────────────

def load_sources(path: Path) -> dict:
    by_shape: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            ep = json.loads(line)
            for n, t in enumerate(ep["turns"]):
                if not t.get("gold_intent") or UNUSABLE & set(t["functions"]):
                    continue
                if (t.get("paraphrase") or {}).get("ok") is False:
                    continue
                by_shape.setdefault(shape(t["gold_intent"]), []).append(
                    {"id": f"{ep['id']}/{n}", "town": ep["town"], "turn": t})
    return by_shape


def try_source(src: dict, said: dict, state_before: dict, said_index: dict) -> str | None:
    t = src["turn"]
    roles_src, roles_new = _roles(t["gold_intent"]), _roles(said)
    if len(roles_src) != len(roles_new):
        return None
    mapping, source_index = {}, {}
    for (a, ia), (b, ib) in zip(roles_src, roles_new):
        if mapping.get(a, b) != b:
            return None
        mapping[a] = b
        if ia:
            source_index[a] = ia
    # The source sentence may name its current destination ("after the X").
    old_dest, new_dest = t["state_before"].get("destination"), state_before.get("destination")
    if old_dest and old_dest not in mapping and new_dest and new_dest not in mapping.values():
        mapping[old_dest] = new_dest
    if len(set(mapping.values())) != len(mapping):
        return None
    text = substitute(t["utterance"], mapping, said_index, source_index)
    if text is None:
        return None
    turn = {"gold_intent": said, "state_before": state_before, "functions": t["functions"],
            "meta": {}, "accept": {}}
    if faithful(text.lower(), required(turn)) is not None:
        return None
    gone = set(mapping) - set(mapping.values())
    if any(re.search(r"\b" + re.escape(n) + r"\b", text.lower()) for g in gone for n in _names(g)):
        return None
    return text


def from_template(said: dict, said_index: dict) -> str | None:
    """Three-stop plan: the two-stop template extended by one stop."""
    c = said["commands"][0]
    if len(said["commands"]) != 1 or c["command_type"] != "plan_route" or len(c["waypoints"]) < 3:
        return None
    def name(t):
        return f"the {_say(t)}" + (f" number {said_index[t]}" if t in said_index else "")
    stops = [name(w["type"]) for w in c["waypoints"]]
    return (f"Go to {stops[0]}, then " + ", then ".join(stops[1:])
            + f", and finally {name(c['destination'])}.")


def state_before_of(states: list, k: int) -> dict:
    """The route summary the parser is shown before event ``k``."""
    if k == 0 or states[k - 1].get("cancelled"):
        return {"destination": None, "destination_index": None, "stops": []}
    s = states[k - 1]
    split = lambda key: (key.split("#")[0], int(key.split("#")[1]))
    return {"destination": split(s["goal"])[0], "destination_index": split(s["goal"])[1],
            "stops": [{"type": split(x)[0], "index": split(x)[1]} for x in s["stops"]]}


REVIEW_COLS = ["id", "event", "utterance", "source", "source_turn", "functions", "added",
               "decision", "fixed_utterance", "note", "rejected"]
DECISIONS = ("ok", "fix")


def options(said: dict, before: dict, sources: dict, key: str, exclude=(), used=None):
    """Candidate sentences for ``said``, best first: (source turn or None, text, added).
    ``key`` seeds the order; ``exclude`` lists rejected source turns."""
    rng = random.Random(key)
    said_index = {t: i for t, i in _roles(said) if i}
    # Hurried plan without a source turn: match at normal urgency, append a hurry sentence.
    hurry, lookup = None, said
    if not sources.get(shape(said)) and shape(said)[0][:1] == ("plan_route",) \
            and said["commands"][0].get("urgency") == "high":
        lookup = copy.deepcopy(said)
        lookup["commands"][0]["urgency"] = "normal"
        hurry = rng.choice(tp.HURRY)
    cands = list(sources.get(shape(lookup), []))
    rng.shuffle(cands)
    used = used if used is not None else Counter()

    # Prefer matching indexes, then least used.
    def pref(src):
        same_idx = [bool(i) for _, i in _roles(src["turn"]["gold_intent"])] == \
                   [bool(i) for _, i in _roles(said)]
        return (not same_idx, used[src["id"]], src["turn"].get("setup", False))

    for src in sorted(cands, key=pref):
        if src["id"] in exclude:
            continue
        src_idx = {a for a, i in _roles(src["turn"]["gold_intent"]) if i}
        if src_idx and len(src_idx) != len(said_index):
            continue
        text = try_source(src, lookup, before, said_index)
        if not text:
            continue
        added = []
        if said_index and not src_idx:
            added.append("id")
        if hurry:
            text += " " + hurry[0].upper() + hurry[1:] + "!"
            added.append("hurry")
        yield src, text, added
    if not cands:
        text = from_template(said, said_index)
        if text:
            yield None, text, ["template"] + (["id"] if said_index else [])


def check_fix(text: str, said: dict, before: dict, functions=()) -> str | None:
    """None if a reviewer's own sentence still says ``said``, else why not."""
    turn = {"gold_intent": said, "state_before": before, "functions": list(functions),
            "meta": {}, "accept": {}}
    return faithful(text.lower(), required(turn))


def settle(ev: dict, row: dict) -> None:
    """Write a row's sentence into its event."""
    ev["utterance"] = row["utterance"]
    ev["source"] = {"turn": row["source_turn"] or None, "utterance": row["source"] or None,
                    "functions": row["functions"].split() if row["functions"] else None,
                    "added": row["added"].split(), "decision": row["decision"] or None}


def build(sc: dict, town, sources: dict, used: Counter, seed: int,
          reviewed: dict) -> tuple[dict, list]:
    expect, whole = replay(town, sc, sc["events"])
    if not whole:
        raise RuntimeError(f"{sc['id']}: the gold replay did not play out")
    say = ids_to_say(town, sc, expect)
    out = copy.deepcopy(sc)
    out["answer_ids"] = answer_ids(sc)
    out["ids_said"] = sorted(say)
    rows = []
    for k, ev in enumerate(out["events"]):
        said = strip_indexes(ev["intent"], say)
        before = state_before_of(expect, k)
        ev["said"] = said
        ev["expect_state"] = expect[k]
        ev["state_before"] = before
        old = reviewed.get((sc["id"], k), {})
        rejected = old.get("rejected", "").split()
        row = {c: "" for c in REVIEW_COLS}
        row.update({"id": sc["id"], "event": k, "rejected": " ".join(rejected),
                    "note": old.get("note", "")})
        if old.get("decision") == "fix":
            # Reviewer's own sentence, if still faithful.
            why = check_fix(old["fixed_utterance"], said, before)
            if why:
                print(f"[utterances] {sc['id']} event {k}: the reviewed fix no longer "
                      f"fits ({why}); picking a sentence again")
            else:
                row.update({k_: old.get(k_, "") for k_ in ("source", "source_turn",
                                                           "functions", "added",
                                                           "fixed_utterance")})
                row.update({"utterance": old["fixed_utterance"], "decision": "fix"})
                settle(ev, row)
                rows.append(row)
                continue
        opts = options(said, before, sources, f"{seed}/{sc['id']}/{k}", rejected, used)
        pinned = old.get("source_turn")
        pick = None
        if old:
            # Same source turn as reviewed, if it still fits.
            pick = next((o for o in options(said, before, sources, f"{seed}/{sc['id']}/{k}",
                                            rejected)
                         if (o[0]["id"] if o[0] else "") == (pinned or "")), None)
        pick = pick or next(opts, None)
        if pick is None:
            raise RuntimeError(f"{sc['id']} event {k}: no source turn fits {shape(said)}")
        src, text, added = pick
        if src:
            used[src["id"]] += 1
        row.update({"utterance": text, "source": src["turn"]["utterance"] if src else "",
                    "source_turn": src["id"] if src else "",
                    "functions": " ".join(src["turn"]["functions"]) if src else "",
                    "added": " ".join(added),
                    "decision": old.get("decision", "") if old.get("utterance") == text else ""})
        settle(ev, row)
        rows.append(row)
    return out, rows


def load_review(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {(r["id"], int(r["event"])): r for r in csv.DictReader(fh)}


def write_review(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=str(G.OUT))
    ap.add_argument("--language", default=str(LANGUAGE))
    ap.add_argument("--xodr", default=None, help="default: [covlm] opendrive_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--review", default=str(REVIEW))
    args = ap.parse_args()

    xodr = Path(args.xodr) if args.xodr else section_path("covlm", "opendrive_dir")
    kb = LandmarkKnowledgeBase(str(KB_CSV))
    sources = load_sources(Path(args.language))
    with open(args.scenarios, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]

    reviewed = load_review(Path(args.review))
    towns, out, rows, used = {}, [], [], Counter()
    for sc in scenarios:
        if sc["town"] not in towns:
            towns[sc["town"]] = G.Town(sc["town"], xodr / f"{sc['town']}.xodr", kb)
        made, r = build(sc, towns[sc["town"]], sources, used, args.seed, reviewed)
        out.append(made)
        rows += r
        print(f"[utterances] {sc['id']}: IDs said for {made['ids_said'] or 'none'}", flush=True)
        for e in made["events"]:
            print(f"    {e['utterance']}")

    with open(args.out, "w", encoding="utf-8") as fh:
        for s in out:
            fh.write(json.dumps(s) + "\n")
    write_review(Path(args.review), rows)
    n_ev = len(rows)
    added = Counter(a for r in rows for a in r["added"].split())
    print(f"[utterances] {len(out)} scenarios, {n_ev} sentences from "
          f"{len({r['source_turn'] for r in rows if r['source_turn']})} source turns; "
          f"added: {dict(added)}; "
          f"{sum(bool(s['ids_said']) for s in out)} scenarios name a place by ID; "
          f"{sum(r['decision'] in DECISIONS for r in rows)} sentences reviewed")
    print(f"[utterances] wrote {args.out} and {args.review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
