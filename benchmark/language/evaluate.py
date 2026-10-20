"""
benchmark/language/evaluate.py
==============================
Score one or more runs against the dataset.

The headline metric is **state accuracy**: push the model's intent through
``MissionPlanner.apply`` from the turn's gold starting state and compare the
route that results (destination, its index, stops in order, which stops are
still to be asked about) with the gold one. Field-level scores explain *why*
a turn failed; state accuracy is what the car would actually do.

    python3 benchmark/language/evaluate.py benchmark/runs/<run>.jsonl [more runs ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.language.run import DATA, load_episodes            # noqa: E402
from benchmark.language.state import apply_intent, route_key      # noqa: E402
from benchmark.language.taxonomy import FUNCTIONS, TIERS          # noqa: E402
from core.intent_schema import commands_of                        # noqa: E402

REJECT = "reject"


def load_run(path: Path) -> dict:
    """(id, turn, utterance) -> the last successful record, else the last
    failed one. Keyed on the sentence so an answer to an older wording of a
    turn is never scored against the current one."""
    best: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["id"], r["turn"], r.get("utterance"))
            if not r.get("error") or key not in best or best[key].get("error"):
                best[key] = r
    return best


def _cmd(intent) -> str:
    """The command sequence, e.g. 'remove_stop+insert_stop'; 'reject' if none."""
    return "+".join(c.get("command_type", "?") for c in commands_of(intent)) or REJECT


def _stops(intent) -> list:
    return [(w.get("type"), w.get("index")) for w in
            sorted(intent.get("waypoints") or [], key=lambda w: w.get("order") or 0)]


def gold_keys(turn: dict) -> set:
    """Every route that counts as right for this turn: the gold one, plus one
    per acceptable alternative type of a vague request (X7)."""
    keys = {route_key(turn["state_after"])}
    gold, accept = turn["gold_intent"], turn.get("accept") or {}
    first = commands_of(gold)[0] if commands_of(gold) else None
    for t in accept.get("destination", []):
        keys.add(route_key(apply_intent(turn["state_before"], {**first, "destination": t})))
    for t in accept.get("insert_type", []):
        alt = {**first, "insert": {**first["insert"], "type": t}}
        keys.add(route_key(apply_intent(turn["state_before"], alt)))
    return keys


def _kind(c: dict) -> str:
    """plan_route and a new_destination that drops the stops do the same
    thing -- replace the route -- so their fields are compared as one kind."""
    t = c.get("command_type")
    return "replace" if t == "plan_route" or (t == "new_destination"
                                              and not c.get("keep_waypoints")) else t


def _fields_one(g: dict, p: dict, accept: dict) -> list[tuple[str, bool]]:
    """Field checks for one gold command against the prediction's command in
    the same position."""
    cmd = g["command_type"]
    same = _kind(p) == _kind(g) or p.get("command_type") == cmd
    out = []
    if cmd in ("plan_route", "new_destination"):
        out.append(("destination", same and p.get("destination") in (
            accept.get("destination") or [g["destination"]])))
        if g.get("destination_index"):
            out.append(("destination_index", same and p.get("destination_index") == g["destination_index"]))
    if cmd == "plan_route":
        if g["waypoints"]:
            out.append(("stops", same and _stops(p) == _stops(g)))
        # urgency is scored only where the passenger said it (gold "high")
        if g["urgency"] == "high":
            out.append(("urgency", same and p.get("urgency") == "high"))
        if g["route_preference"]:
            out.append(("preference", same and p.get("route_preference") == g["route_preference"]))
        if g["avoid"]:
            out.append(("avoid", same and bool(p.get("avoid"))))
    if cmd == "insert_stop":
        gi, pi = g["insert"], (p.get("insert") or {}) if same else {}
        out.append(("insert_type", pi.get("type") in (accept.get("insert_type") or [gi["type"]])))
        out.append(("insert_position", (pi.get("position"), pi.get("after"), pi.get("before"))
                    == (gi["position"], gi.get("after"), gi.get("before"))
                    if (gi.get("after") or gi.get("before")) else pi.get("position") == gi["position"]))
        if gi.get("index"):
            out.append(("insert_index", pi.get("index") == gi["index"]))
    if cmd == "move_stop":
        gm, pm = g["move"], (p.get("move") or {}) if same else {}
        out.append(("move_type", pm.get("type") == gm["type"]))
        out.append(("move_position", all(pm.get(k) == gm.get(k) for k in ("position", "after", "before"))))
    if cmd == "remove_stop":
        out.append(("remove_type", same and (p.get("remove") or {}).get("type") == g["remove"]["type"]))
    if cmd == "new_destination":
        out.append(("keep_waypoints", same and bool(p.get("keep_waypoints")) == g["keep_waypoints"]))
        if g.get("urgency") == "high":
            out.append(("urgency", same and p.get("urgency") == "high"))
    return out


def fields(gold, pred, accept: dict | None = None) -> list[tuple[str, bool]]:
    """(field, correct) for every field the gold commands set, command by
    command in order; plus the command count when there are several."""
    gcs, pcs = commands_of(gold), commands_of(pred)
    out = []
    for i, g in enumerate(gcs):
        out += _fields_one(g, pcs[i] if i < len(pcs) else {}, accept or {})
    if len(gcs) > 1:
        out.append(("command_count", len(pcs) == len(gcs)))
    return out


def _asks(before: dict, after: dict) -> bool:
    """Does this turn add an unordered stop the passenger will be asked to
    place? Only a stop new in this turn counts, and only if there are other
    stops to place it between."""
    pending = {s["type"] for s in before["stops"] if s.get("enroute")}
    new = [s for s in after["stops"] if s.get("enroute") and s["type"] not in pending]
    return bool(new) and len(after["stops"]) > 1


def score(episodes: list[dict], run: dict, partial: bool = False) -> dict:
    turns, by_func, by_tier, by_town = [], defaultdict(list), defaultdict(list), defaultdict(list)
    field_hits = defaultdict(list)
    confusion = Counter()
    ask = {"tp": 0, "fn": 0, "fp": 0}
    episode_ok = []
    latency, tokens_in, tokens_out = [], [], []
    api_failures = 0

    unanswered = 0
    setup = []
    for e in episodes:
        ok_all, complete = True, True
        for i, t in enumerate(e["turns"]):
            r = run.get((e["id"], i, t["utterance"]))
            if r is None and partial:
                unanswered += 1
                complete = False
                continue
            if r is None:
                raise SystemExit(f"run has no answer for {e['id']} turn {i} as currently "
                                 f"worded; finish (or re-run) the run first")
            failed = bool(r.get("error"))
            api_failures += failed
            pred = None if failed else r.get("intent")
            gold = t["gold_intent"]
            pred_state = apply_intent(t["state_before"], pred)
            ok = not failed and route_key(pred_state) in gold_keys(t)
            if t.get("setup"):          # context for later turns, scored apart
                setup.append(ok)
                continue
            ok_all &= ok
            turns.append(ok)
            for f in t["functions"]:
                by_func[f].append(ok)
            by_tier[t["tier"]].append(ok)
            by_town[e["town"]].append(ok)
            confusion[(_cmd(gold), _cmd(pred))] += 1
            for name, hit in fields(gold, pred, t.get("accept")):
                field_hits[name].append(hit)
            want = _asks(t["state_before"], t["state_after"])
            got = _asks(t["state_before"], pred_state)
            if want and got:
                ask["tp"] += 1
            elif want:
                ask["fn"] += 1
            elif got:
                ask["fp"] += 1
            if not failed:
                latency.append(r.get("latency_s") or 0.0)
                u = r.get("usage") or {}
                if u.get("input_tokens") is not None:
                    tokens_in.append(u["input_tokens"])
                    tokens_out.append(u.get("output_tokens") or 0)
        if complete:
            episode_ok.append(ok_all)

    def acc(xs):
        return round(100.0 * sum(xs) / len(xs), 1) if xs else None

    def pct(xs, q):
        return round(statistics.quantiles(xs, n=100)[q - 1], 2) if len(xs) > 1 else None

    return {
        "turns": len(turns), "episodes": len(episode_ok), "api_failures": api_failures,
        "unanswered": unanswered,
        "state_acc": acc(turns), "episode_acc": acc(episode_ok), "setup_acc": acc(setup),
        "command_acc": acc([g == p for (g, p), n in confusion.items() for _ in range(n)]),
        "by_function": {f: (acc(v), len(v)) for f, v in by_func.items()},
        "by_tier": {k: (acc(v), len(v)) for k, v in sorted(by_tier.items())},
        "by_town": {k: (acc(v), len(v)) for k, v in sorted(by_town.items())},
        "fields": {k: (acc(v), len(v)) for k, v in sorted(field_hits.items())},
        "ask_recall": acc([True] * ask["tp"] + [False] * ask["fn"]),
        # Denominator: turns that should not ask.
        "ask_false_rate": acc([True] * ask["fp"]
                              + [False] * (len(turns) - ask["tp"] - ask["fn"] - ask["fp"])),
        "confusion": {f"{g} -> {p}": n for (g, p), n in sorted(confusion.items()) if g != p},
        "latency_p50": pct(latency, 50), "latency_p95": pct(latency, 95),
        "tokens_in_mean": round(statistics.mean(tokens_in)) if tokens_in else None,
        "tokens_out_mean": round(statistics.mean(tokens_out), 1) if tokens_out else None,
    }


def report(names: list[str], scores: list[dict]) -> str:
    cols = " | ".join(names)
    lines = [f"| metric | {cols} |", "|---|" + "---|" * len(names)]

    def row(label, get):
        lines.append(f"| {label} | " + " | ".join(str(get(s)) for s in scores) + " |")

    for k in ("turns", "unanswered", "api_failures", "state_acc", "episode_acc", "setup_acc",
              "command_acc",
              "ask_recall", "ask_false_rate", "latency_p50", "latency_p95",
              "tokens_in_mean", "tokens_out_mean"):
        row(k, lambda s, k=k: s[k])
    lines.append(f"| **by tier** | {' | '.join('' for _ in names)} |")
    for tier in [k for k in TIERS if k]:
        row(f"T{tier}", lambda s, tier=tier: "%s (n=%s)" % s["by_tier"].get(tier, (None, 0)))
    lines.append(f"| **by function** | {' | '.join('' for _ in names)} |")
    for f, (_, desc) in FUNCTIONS.items():
        row(f"{f} {desc}", lambda s, f=f: "%s (n=%s)" % s["by_function"].get(f, (None, 0)))
    lines.append(f"| **fields** | {' | '.join('' for _ in names)} |")
    for k in sorted({k for s in scores for k in s["fields"]}):
        row(k, lambda s, k=k: "%s (n=%s)" % s["fields"].get(k, (None, 0)))
    lines.append(f"| **by town** | {' | '.join('' for _ in names)} |")
    for k in sorted({k for s in scores for k in s["by_town"]}):
        row(k, lambda s, k=k: "%s (n=%s)" % s["by_town"].get(k, (None, 0)))
    for name, s in zip(names, scores):
        if s["confusion"]:
            lines.append(f"\n**{name}** command-type confusions: " + ", ".join(
                f"{k} ×{v}" for k, v in s["confusion"].items()))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--md", default=None, help="also write the report here")
    ap.add_argument("--partial", action="store_true",
                    help="score only the turns a run has answered so far")
    args = ap.parse_args()

    episodes = load_episodes(Path(args.data))
    names, scores = [], []
    for path in map(Path, args.runs):
        s = score(episodes, load_run(path), args.partial)
        path.with_suffix(".summary.json").write_text(json.dumps(s, indent=1))
        names.append(path.stem)
        scores.append(s)
    text = report(names, scores)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
