"""
benchmark/language/rewrite_task/check_rewrites.py
=================================================
Hand the rewriting of the template utterances to another agent, and check what
comes back before it enters the pipeline.

    python3 .../check_rewrites.py export                  # -> turns.jsonl (the task)
    python3 .../check_rewrites.py check  theirs.jsonl     # what fails, and why
    python3 .../check_rewrites.py apply  theirs.jsonl     # -> data/episodes_rewritten.jsonl

``check`` runs ``paraphrase.faithful()`` -- the project's own structural check --
plus the four patterns that a turn-by-turn review of the previous rewrites
found it misses (a stop sold as a destination, a dropped "then", an invented or
lost hurry, a removal that stopped reading as one). Nothing here changes
``paraphrase.py``: this is a side door for externally produced sentences, and
``faithful()`` stays the single source of truth for the structural part.

``apply`` writes a *new* episodes file and leaves ``episodes_paraphrased.jsonl``
alone, so ``review.py export --src`` can be pointed at either one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.language.paraphrase import (STYLES, _names as _surfaces,      # noqa: E402
                                           faithful, required)
from benchmark.language.review import describe_intent, describe_state         # noqa: E402
from benchmark.language.run import load_episodes                              # noqa: E402
from core.intent_schema import commands_of                                    # noqa: E402

LANG = HERE.parent
SRC = LANG / "data" / "episodes_template.jsonl"
TURNS = HERE / "turns.jsonl"
OUT = LANG / "data" / "episodes_rewritten.jsonl"

# --- the four extra patterns -------------------------------------------------

DEST_WORD = re.compile(r"\b(final|last|next)\s+destination\b", re.I)
# names a stop properly somewhere: then "final destination" is the real destination
STOP_WORD = re.compile(r"\b(last|final|next|first)\s+(stop|spot|stand|halt)\b", re.I)
AFTER_WORD = re.compile(r"\b(then|afterwards?|after|once|when|later|second|"
                        r"following|finally|carry on|keep going|keep rolling|"
                        r"continue|next)\b", re.I)

# Urgency is asymmetric on purpose. To decide a hurry was *kept* almost any
# marker counts (WIDE); to decide one was *added* only unambiguous ones do
# (STRICT) -- "immediately prior to the bus stop" is a position and "a quick
# latte" is a drink, and neither is a passenger asking to go faster.
HURRY_WIDE = re.compile(r"\b(hurry|hurries|hurrying|rush|rushing|asap|a\.s\.a\.p|"
                        r"urgent|urgency|pronto|snappy|hustle|haste|immediate|"
                        r"immediately|prompt|promptly|quick|quickly|quicker|fast|"
                        r"faster|speedy|speed it up|right away|without delay|"
                        r"step on it|floor it|expedite)\b", re.I)
HURRY_STRICT = re.compile(r"\b(hurry|hurries|hurrying|rush|rushing|asap|a\.s\.a\.p|"
                          r"urgent|urgency|pronto|snappy|hustle|haste|quickly|"
                          r"speed it up|step on it|floor it|expedite|"
                          r"as fast as you can|as quick as you can)\b", re.I)
NO_HURRY = re.compile(r"\b(no|not in a|without a|don'?t)\s+(rush|hurry)\b|"
                      r"\btake your time\b|\bno need to (rush|hurry)\b", re.I)
PREF_PHRASE = re.compile(r"\b(fastest|quickest|shortest|scenic|prettiest|pretty)\s+"
                         r"(route|way|roads?|path)\b", re.I)

REMOVE_WORD = re.compile(r"\b(skip|skipping|forget|drop|dropping|never ?mind|"
                         r"no longer|don'?t need|do not need|cancel|omit|bypass|"
                         r"scratch|scrap|remove|leave out|lose|nix|disregard|"
                         r"ignore|call off|cut|don'?t bother|not stopping|"
                         r"no need)\b", re.I)
DROP_ME = re.compile(r"\bdrop\s+(me|us)\b", re.I)


def _urgency_of(gold: dict | None) -> bool:
    return bool(gold) and any(c.get("urgency") == "high" for c in commands_of(gold))


def _stop_to_edge(gold: dict | None) -> bool:
    """gold puts a *stop* at the front or the back of the queue."""
    if not gold:
        return False
    for c in commands_of(gold):
        for k in ("insert", "move"):
            if c.get(k) and c[k].get("position") in ("next", "last"):
                return True
    return False


def _keeps_waypoints(gold: dict | None) -> bool:
    return bool(gold) and any(c["command_type"] == "new_destination" and c.get("keep_waypoints")
                              for c in commands_of(gold))


def _removes(gold: dict | None) -> bool:
    return bool(gold) and any(c["command_type"] == "remove_stop" for c in commands_of(gold))


def _bare_new_destination(text: str, gold: dict) -> bool:
    """"...and afterwards take me to the park" compressed to "...then the park".

    Only in a multi-command turn: there the bare noun competes with the stops
    the same sentence just asked for and reads as one more of them. On its own
    ("Then the shop.") there is nothing to confuse it with.
    """
    cmds = commands_of(gold)
    if len(cmds) < 2:
        return False
    for c in cmds:
        if c["command_type"] != "new_destination" or not c.get("keep_waypoints"):
            continue
        for name in _surfaces(c["destination"]):
            if re.search(r"\b(?:then|afterwards?|after that)\s+(?:the\s+|a\s+)?"
                         + re.escape(name) + r"\b", text, re.I):
                return True
    return False


def extra_checks(text: str, turn: dict) -> list[str]:
    """Why ``text`` is not an acceptable rewrite of ``turn``, beyond faithful()."""
    g, out = turn["gold_intent"], []
    if _stop_to_edge(g) and DEST_WORD.search(text) and not STOP_WORD.search(text):
        out.append('a stop is called a "destination" (say "last stop" / "next stop")')
    if _keeps_waypoints(g) and not AFTER_WORD.search(text):
        out.append('lost "then / afterwards": the gold carries on past the old destination')
    if _bare_new_destination(text, g):
        out.append('the new destination is a bare noun after "then": it reads as '
                   "one more stop (keep the verb: \"then take me to ...\")")
    plain = PREF_PHRASE.sub("", text)
    if _urgency_of(g) and not HURRY_WIDE.search(plain):
        out.append("lost the hurry the gold has")
    if not _urgency_of(g) and HURRY_STRICT.search(NO_HURRY.sub("", plain)):
        out.append("added a hurry the gold does not have")
    if _removes(g) and not REMOVE_WORD.search(DROP_ME.sub("", text)):
        out.append("the removal no longer reads as a removal")
    return out


# --- export ------------------------------------------------------------------

def export(src: Path, out: Path) -> int:
    rows, n = [], 0
    for e in load_episodes(src):
        for i, t in enumerate(e["turns"]):
            if t.get("setup"):                       # context only, never rewritten
                continue
            req = required(t)
            extra = []
            if "X5" in t["functions"]:
                extra.append("Keep the hesitations and repeated words; it is spoken speech.")
            if "X1" in t["functions"] or "X7" in t["functions"]:
                extra.append("Describe the need only; do not name the kind of place.")
            if "X6" in t["functions"]:
                extra.append("Keep that it is the nearest / closest one.")
            if "X4" in t["functions"]:
                extra.append("Keep the self-correction: the first place said is "
                             "replaced by the second.")
            rows.append({
                "row": n, "id": e["id"], "turn": i, "town": e["town"],
                "tier": t["tier"], "functions": t["functions"],
                "template": t["utterance"],
                "route_before": describe_state(t["state_before"]),
                "gold": describe_intent(t["gold_intent"]),
                "style": STYLES[(zlib.crc32(e["id"].encode()) + i) % len(STYLES)],
                "must_name": req["places"], "must_not_name": req["absent"],
                "ordinal": req["ordinal"], "ordered": bool(req.get("ordered")),
                "keep_nearest": bool(req["any_word"]),
                "keep_disfluent": "X5" in t["functions"],
                "extra": " ".join(extra),
            })
            n += 1
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[rewrite] {len(rows)} turns -> {out}")
    return 0


# --- check / apply -----------------------------------------------------------

def _load_submission(path: Path) -> dict[int, str]:
    out = {}
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out[int(d["row"])] = str(d["rewrite"])
        except Exception as e:
            raise SystemExit(f"[rewrite] {path}:{ln} is not "
                             f'{{"row": <int>, "rewrite": "<sentence>"}}: {e}')
    return out


def _turns_by_row(src: Path) -> dict[int, tuple[str, int, dict]]:
    out, n = {}, 0
    for e in load_episodes(src):
        for i, t in enumerate(e["turns"]):
            if t.get("setup"):
                continue
            out[n] = (e["id"], i, t)
            n += 1
    return out


def check(src: Path, sub_path: Path, verbose: bool) -> tuple[dict, dict]:
    sub, turns = _load_submission(sub_path), _turns_by_row(src)
    bad, unchanged = {}, 0
    missing = sorted(set(turns) - set(sub))
    extra_rows = sorted(set(sub) - set(turns))
    for row, text in sorted(sub.items()):
        if row not in turns:
            continue
        _, _, t = turns[row]
        text = text.strip()
        why = []
        if not text:
            why.append("empty")
        else:
            w = faithful(text, required(t))
            if w:
                why.append(f"faithful(): {w}")
            why += extra_checks(text, t)
            if text == t["utterance"]:
                unchanged += 1
        if why:
            bad[row] = why
            if verbose:
                print(f"\nrow {row}  {turns[row][0]} #{turns[row][1]}  "
                      f"{' '.join(t['functions'])}")
                print(f"  route    : {describe_state(t['state_before'])}")
                print(f"  gold     : {describe_intent(t['gold_intent'])}")
                print(f"  template : {t['utterance']}")
                print(f"  rewrite  : {text}")
                for w_ in why:
                    print(f"  REJECT   : {w_}")
    ok = len(sub) - len(bad) - len(extra_rows)
    print(f"\n[rewrite] {len(turns)} turns expected, {len(sub)} submitted")
    print(f"[rewrite] accepted {ok} | rejected {len(bad)} | "
          f"identical to the template {unchanged}")
    if missing:
        print(f"[rewrite] {len(missing)} missing, e.g. {missing[:8]}")
    if extra_rows:
        print(f"[rewrite] {len(extra_rows)} unknown row numbers, e.g. {extra_rows[:8]}")
    return bad, {"missing": missing, "unknown": extra_rows}


def apply(src: Path, sub_path: Path, out: Path) -> int:
    bad, holes = check(src, sub_path, verbose=False)
    if bad or holes["missing"] or holes["unknown"]:
        print("[rewrite] nothing written; run `check` and fix the rejects first")
        return 1
    sub = _load_submission(sub_path)
    episodes, n = load_episodes(src), 0
    for e in episodes:
        for i, t in enumerate(e["turns"]):
            t["template_utterance"] = t["utterance"]
            if t.get("setup"):
                t["paraphrase"] = {"style": None, "ok": True, "verified": "setup",
                                   "reason": "setup turn, kept as template",
                                   "source": "template"}
                continue
            t["utterance"] = sub[n].strip()
            t["paraphrase"] = {"style": None, "ok": True, "reason": None,
                               "verified": None, "source": f"external:{sub_path.name}"}
            n += 1
        e["source"] = "rewritten"
    with open(out, "w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[rewrite] {n} rewrites -> {out}")
    print(f"[rewrite] next: python3 benchmark/language/review.py export --src {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["export", "check", "apply"])
    ap.add_argument("file", nargs="?", help="the agent's jsonl, for check / apply")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--turns", default=str(TURNS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quiet", action="store_true", help="counts only, no per-row detail")
    args = ap.parse_args()

    if args.action == "export":
        return export(Path(args.src), Path(args.turns))
    if not args.file:
        print(f"[rewrite] {args.action} needs the agent's jsonl file")
        return 1
    if args.action == "check":
        bad, holes = check(Path(args.src), Path(args.file), not args.quiet)
        return 1 if bad or holes["missing"] or holes["unknown"] else 0
    return apply(Path(args.src), Path(args.file), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
