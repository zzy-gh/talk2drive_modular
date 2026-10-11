"""
benchmark/language/apply_review.py
==================================
Apply the decisions made so far in ``data/review.csv`` to the paraphrased
dataset, leaving the rows nobody has looked at yet as they are.

This is *not* the verified set. ``review.py import`` writes that one, and it
refuses while any row is undecided, so the guarantee it carries -- every
sentence was looked at by a person -- stays intact. This script exists to run
a model over a partly reviewed dataset:

    python3 benchmark/language/apply_review.py   # -> data/episodes_reviewed.jsonl

ok        keep the paraphrase
fix       use fixed_utterance instead (re-checked; a warning if it loses meaning)
drop      fall back to the template sentence
undecided keep the paraphrase, and say so in paraphrase.verified
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
from benchmark.language.review import CSV                                         # noqa: E402
from benchmark.language.run import load_episodes                                   # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "episodes_reviewed.jsonl"


def apply_review(src: Path, review: Path, out: Path) -> int:
    with open(review, newline="", encoding="utf-8-sig") as fh:
        decisions = {(r["id"], int(r["turn"])): r for r in csv.DictReader(fh)}
    episodes = load_episodes(src)
    counts = {"ok": 0, "fix": 0, "drop": 0, "unreviewed": 0, "setup": 0}
    changed, warnings = 0, []
    for e in episodes:
        for i, t in enumerate(e["turns"]):
            if t.get("setup"):
                counts["setup"] += 1
                continue
            r = decisions.get((e["id"], i))
            d = (r or {}).get("decision", "").strip().lower()
            if d not in ("ok", "fix", "drop"):
                counts["unreviewed"] += 1
                t.setdefault("paraphrase", {})["verified"] = None
                continue
            counts[d] += 1
            before = t["utterance"]
            if d == "fix":
                text = r["fixed_utterance"].strip()
                why = faithful(text, required(t))
                if why:
                    warnings.append(f"{e['id']}#{i}: fixed sentence {why}: {text!r}")
                t["utterance"] = text
            elif d == "drop":
                t["utterance"] = t.get("template_utterance", t["utterance"])
            changed += t["utterance"] != before
            t.setdefault("paraphrase", {})["verified"] = d
        e["source"] = "paraphrase+review"
    for w_ in warnings:
        print(f"[apply-review] warning: {w_}")
    with open(out, "w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[apply-review] {counts}")
    print(f"[apply-review] {changed} sentences changed -> {out}")
    return 1 if warnings else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(PARAPHRASED))
    ap.add_argument("--review", default=str(CSV))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    return apply_review(Path(args.src), Path(args.review), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
