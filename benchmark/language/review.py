"""
benchmark/language/review.py
============================
Human verification of the paraphrased dataset.

    python3 benchmark/language/review.py export            # -> data/review.csv
    # fill in `decision` for every row: ok / fix / drop  (fix: put the sentence in fixed_utterance)
    python3 benchmark/language/review.py import            # -> data/episodes_verified.jsonl

ok   keep the paraphrase
fix  use fixed_utterance instead (re-checked; a warning if it loses meaning)
drop fall back to the template sentence

Import refuses to write while any row is undecided, so every sentence in the
verified set has been looked at by a person.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.language.paraphrase import OUT as PARAPHRASED, faithful, required  # noqa: E402
from benchmark.language.run import load_episodes                                   # noqa: E402
from core.intent_schema import commands_of                                         # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "review.csv"
VERIFIED = HERE / "data" / "episodes_verified.jsonl"
COLUMNS = ["id", "turn", "town", "functions", "route_before", "gold", "template",
           "paraphrase", "auto_check", "decision", "fixed_utterance", "note"]


def _place(t, i=None):
    return f"{t}#{i}" if i else t


def describe_state(s: dict) -> str:
    if not s["destination"]:
        return "(no route)"
    stops = [_place(x["type"], x["index"]) + ("?" if x["enroute"] else "") for x in s["stops"]]
    return (" -> ".join(stops + [_place(s["destination"], s["destination_index"])])
            + ("; hurry" if s["urgency"] == "high" else ""))


def describe_intent(g: dict | None) -> str:
    if g is None:
        return "REJECT (place not on this map; route unchanged)"
    return " ; then ".join(_describe(c) for c in commands_of(g))


def _where(x: dict) -> str:
    return (f"after {x['after']}" if x.get("after") else
            f"before {x['before']}" if x.get("before") else str(x.get("position")))


def _describe(g: dict) -> str:
    c = g["command_type"]
    if c == "plan_route":
        via = [_place(w["type"], w["index"]) for w in g["waypoints"]]
        extra = [x for x in (("hurry" if g["urgency"] == "high" else ""),
                             g["route_preference"] or "", ", ".join(g["avoid"]) and
                             "avoid " + ", ".join(g["avoid"])) if x]
        return (f"plan_route -> {_place(g['destination'], g['destination_index'])}"
                + (f" via {via}" if via else "") + (f"; {'; '.join(extra)}" if extra else ""))
    if c == "insert_stop":
        i = g["insert"]
        return f"insert_stop {_place(i['type'], i['index'])} ({_where(i)})"
    if c == "move_stop":
        return f"move_stop {g['move']['type']} ({_where(g['move'])})"
    if c == "remove_stop":
        return f"remove_stop {g['remove']['type']}"
    if c == "new_destination":
        how = "keep stops, old destination becomes a stop" if g["keep_waypoints"] else "replace route"
        return f"new_destination -> {_place(g['destination'], g['destination_index'])} ({how})"
    return "cancel_route (stop)"


def export(src: Path, out: Path) -> int:
    rows = []
    for e in load_episodes(src):
        for i, t in enumerate(e["turns"]):
            if t.get("setup"):                       # template text, context only
                continue
            p = t.get("paraphrase") or {}
            rows.append({"id": e["id"], "turn": i, "town": e["town"],
                         "functions": " ".join(t["functions"]),
                         "route_before": describe_state(t["state_before"]),
                         "gold": describe_intent(t["gold_intent"])
                         + "".join(f"  [any of {', '.join(v)}]" for v in (t.get("accept") or {}).values()),
                         "template": t.get("template_utterance", t["utterance"]),
                         "paraphrase": t["utterance"],
                         "auto_check": ("n/a" if not p else "ok" if p.get("ok")
                                        else f"FAILED: {p.get('reason')}"),
                         "decision": "", "fixed_utterance": "", "note": ""})
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:       # BOM: Excel opens UTF-8
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[review] {len(rows)} rows -> {out}")
    return 0


def import_(src: Path, review: Path, out: Path) -> int:
    with open(review, newline="", encoding="utf-8-sig") as fh:
        decisions = {(r["id"], int(r["turn"])): r for r in csv.DictReader(fh)}
    episodes = load_episodes(src)
    undecided, warnings, counts = [], [], {"ok": 0, "fix": 0, "drop": 0}
    for e in episodes:
        for i, t in enumerate(e["turns"]):
            if t.get("setup"):
                continue
            r = decisions.get((e["id"], i))
            d = (r or {}).get("decision", "").strip().lower()
            if d not in counts:
                undecided.append(f"{e['id']}#{i}")
                continue
            counts[d] += 1
            if d == "fix":
                text = r["fixed_utterance"].strip()
                why = faithful(text, required(t))
                if why:
                    warnings.append(f"{e['id']}#{i}: fixed sentence {why}: {text!r}")
                t["utterance"] = text
            elif d == "drop":
                t["utterance"] = t.get("template_utterance", t["utterance"])
            t.setdefault("paraphrase", {})["verified"] = d
        e["source"] = "verified"
    for w_ in warnings:
        print(f"[review] warning: {w_}")
    if undecided:
        print(f"[review] {len(undecided)} rows have no decision (ok/fix/drop), e.g. "
              f"{undecided[:5]}; nothing written")
        return 1
    with open(out, "w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[review] {counts} -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("--src", default=str(PARAPHRASED))
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--out", default=str(VERIFIED))
    args = ap.parse_args()
    if args.action == "export":
        return export(Path(args.src), Path(args.csv))
    return import_(Path(args.src), Path(args.csv), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
