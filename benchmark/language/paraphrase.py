"""
benchmark/language/paraphrase.py
================================
LLM rewrite of every template utterance, for lexical and stylistic variety.

The meaning must not move, and that is checked against the turn's *structure*
-- never by asking a model what the sentence means, which would be grading the
benchmark with the thing it measures:

* every place the gold intent involves is still named (any of its surface
  forms) -- word order is free, a rewrite may well move the destination
  first; whether multi-stop order survived is left to the human check;
* an ordinal ("second") survives wherever the gold has an index;
* an indirect turn ("I need cash") still does not name its place;
* the off-map place, and both halves of a self-correction, survive.

A rewrite that fails is retried; after three failures the template sentence
is kept and the turn is flagged. Everything is then human-verified with
review.py before it is used as test data.

    python3 benchmark/language/paraphrase.py     # -> data/episodes_paraphrased.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.language import templates as tp                 # noqa: E402
from core.intent_schema import commands_of                     # noqa: E402
from benchmark.language.run import RateLimiter, load_episodes, retry_delay   # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "episodes_template.jsonl"
OUT = HERE / "data" / "episodes_paraphrased.jsonl"

STYLES = ["casual and relaxed", "polite and a little formal", "short and terse",
          "chatty, with a bit of extra context about why", "slightly impatient"]

PROMPT = """You rewrite what a passenger says to their self-driving car.
Rewrite the sentence below in a {style} style, as a native English speaker would say it out loud.

Hard rules:
- Keep the meaning exactly: same places, same order, same number words ("first", "second"), same urgency, same preferences.
- Name every place mentioned; you may use a natural synonym for it.
- Do not add places, stops, times or conditions that are not in the original.
- Do not add ordering words (first, then, after, before, also, too, again) that the original does not have.
- A place the passenger wants to be taken to stays a destination: never turn it into a quick stop ("swing by", "stop by", "pop into", "drop in at").
{extra}- Output only the rewritten sentence, nothing else.

Original: {text}"""


def _names(t: str) -> list[str]:
    return [s.lower() for s in tp.SURFACE.get(t, [t.replace("_", " ")])] + [t.replace("_", " ")]


def _first_pos(text: str, t: str) -> int:
    hits = [m.start() for n in _names(t) for m in re.finditer(r"\b" + re.escape(n) + r"\b", text)]
    return min(hits) if hits else -1


def required(turn: dict) -> dict:
    """What a faithful rewrite of this turn must (not) contain."""
    g, before, fn = turn["gold_intent"], turn["state_before"], turn["functions"]
    req = {"places": [], "absent": [], "ordinal": None, "any_word": []}
    if g is None:                                  # off-map: keep the missing place
        req["places"] = [turn["meta"]["missing"]]
        return req
    for c in commands_of(g):
        cmd = c["command_type"]
        if cmd == "plan_route":
            req["places"] += [w["type"] for w in c["waypoints"]] + [c["destination"]]
            req["ordinal"] = req["ordinal"] or c["destination_index"] or next(
                (w["index"] for w in c["waypoints"] if w["index"]), None)
        elif cmd == "insert_stop":
            ins = c["insert"]
            req["places"] += [x for x in (ins.get("after"), ins.get("before")) if x] + [ins["type"]]
            req["ordinal"] = req["ordinal"] or ins["index"]
        elif cmd == "remove_stop":
            req["places"] += [c["remove"]["type"]]
        elif cmd == "move_stop":
            mv = c["move"]
            req["places"] += [mv["type"]] + [x for x in (mv.get("after"), mv.get("before")) if x]
        elif cmd == "new_destination":
            req["places"] += ([before["destination"]] if "X3" in fn and c["keep_waypoints"]
                              else []) + [c["destination"]]
            req["ordinal"] = req["ordinal"] or c["destination_index"]
    first = commands_of(g)[0]
    if "X1" in fn:                                 # the need, never the place
        target = first.get("destination") or (first.get("insert") or {}).get("type")
        req["places"] = [p for p in req["places"] if p != target]
        req["absent"] = [target]
    accept = turn.get("accept") or {}
    if "X7" in fn:                                 # a category, never a type
        alts = accept.get("destination") or accept.get("insert_type") or []
        req["places"] = [p for p in req["places"] if p not in alts]
        req["absent"] = sorted(set(req["absent"]) | set(alts))
    if "X6" in fn:
        req["any_word"] = ["nearest", "closest", "nearby", "near"]
    if "X4" in fn:
        req["places"] = [turn["meta"]["wrong"]] + req["places"]
        req["ordered"] = True                      # the wrong place comes first
    return req


def faithful(text: str, req: dict) -> str | None:
    """None if ``text`` keeps the meaning, else why not."""
    low = text.lower()
    pos = []
    for t in req["places"]:
        p = _first_pos(low, t)
        if p < 0:
            return f"lost {t}"
        pos.append(p)
    if req.get("ordered") and pos != sorted(pos):
        return "self-correction order lost"
    for t in req["absent"]:
        if _first_pos(low, t) >= 0:
            return f"names {t} outright"
    if req.get("any_word") and not any(re.search(r"\b" + w + r"\b", low) for w in req["any_word"]):
        return "lost 'nearest'"
    if req["ordinal"]:
        n = req["ordinal"]
        words = {tp.ORDINAL.get(n, ""), str(n), f"{n}nd", f"{n}st", f"{n}rd", f"{n}th",
                 f"number {n}"}
        if not any(w and re.search(r"\b" + re.escape(w) + r"\b", low) for w in words):
            return f"lost the ordinal {n}"
    return None


def _openai_compatible(base_url: str, key: str, model: str):
    """chat/completions over plain HTTP. Thinking off: a rewrite needs no
    reasoning, and with it on Qwen spends ~1500 tokens and ~30 s a sentence."""
    import urllib.request

    def complete(prompt: str) -> str:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            data=json.dumps({"model": model, "temperature": 0.9, "enable_thinking": False,
                             "messages": [{"role": "user", "content": prompt}]}).encode())
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"] or ""
    return complete


def _gemini(model: str):
    from google import genai
    from google.genai import types
    client = genai.Client()

    def complete(prompt: str) -> str:
        r = client.models.generate_content(model=model, contents=prompt,
                                           config=types.GenerateContentConfig(temperature=0.9))
        return r.text or ""
    return complete


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--provider", default="qwen", choices=["qwen", "gemini"],
                    help="qwen: any OpenAI-compatible endpoint (QWEN_BASE_URL, "
                         "QWEN_API_KEY); gemini: GEMINI_API_KEY")
    ap.add_argument("--model", default=None,
                    help="default: qwen3.7-plus for qwen, $GEMINI_MODEL for gemini")
    ap.add_argument("--rpm", type=float, default=60)
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()

    limiter = RateLimiter(args.rpm)
    if args.provider == "qwen":
        model = args.model or "qwen3.7-plus"
        complete = _openai_compatible(os.environ["QWEN_BASE_URL"], os.environ["QWEN_API_KEY"], model)
    else:
        model = args.model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        complete = _gemini(model)
    print(f"[paraphrase] {args.provider} / {model}", flush=True)

    def ask(prompt: str) -> str:
        for attempt in range(10):
            limiter.wait()
            try:
                return complete(prompt).strip().strip('"')
            except Exception as exc:
                time.sleep(retry_delay(exc, attempt))
        raise RuntimeError("paraphrase request kept failing")

    out = Path(args.out)
    done = {}
    if out.exists():                                   # resume
        for e in load_episodes(out):
            done[e["id"]] = e
    episodes = load_episodes(Path(args.src))
    kept = flagged = 0
    with open(out, "w", encoding="utf-8") as fh:
        for n, e in enumerate(episodes):
            if e["id"] in done:
                e = done[e["id"]]
            else:
                for i, t in enumerate(e["turns"]):
                    t["template_utterance"] = t["utterance"]
                    if t.get("setup"):                 # context only: not rewritten
                        t["paraphrase"] = {"style": None, "ok": True, "verified": "setup",
                                           "reason": "setup turn, kept as template"}
                        continue
                    req = required(t)
                    style = STYLES[(zlib.crc32(e["id"].encode()) + i) % len(STYLES)]
                    extra = ("- Keep the hesitations and repeated words; it is spoken, disfluent speech.\n"
                             if "X5" in t["functions"] else "")
                    if "X1" in t["functions"] or "X7" in t["functions"]:
                        extra += "- Describe the need only; do not name the kind of place.\n"
                    if "X6" in t["functions"]:
                        extra += "- Keep that it is the nearest / closest one.\n"
                    if "X4" in t["functions"]:
                        extra += "- Keep the self-correction: the first place said is replaced by the second.\n"
                    why = "not tried"
                    for _ in range(args.tries):
                        cand = ask(PROMPT.format(style=style, extra=extra, text=t["utterance"]))
                        why = faithful(cand, req)
                        if why is None:
                            t["utterance"] = cand
                            break
                    t["paraphrase"] = {"style": style, "ok": why is None, "reason": why,
                                       "verified": None}
                e["source"] = "paraphrase"
            for t in e["turns"]:
                kept += t["paraphrase"]["ok"]
                flagged += not t["paraphrase"]["ok"]
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
            fh.flush()
            if (n + 1) % 25 == 0:
                print(f"[paraphrase] {n + 1}/{len(episodes)} episodes", flush=True)
    print(f"[paraphrase] {kept} turns rewritten, {flagged} kept as template (flagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
