"""
benchmark/language/run.py
=========================
Run one model over the language dataset and record every answer.

Each turn is parsed from its *gold* route state ("teacher forcing"), so one
wrong turn does not poison the rest of its episode; episode-level success is
computed from the per-turn results by evaluate.py. Runs resume: turns already
answered in the output file are skipped, so a run cut short by rate limits can
simply be started again.

    python3 benchmark/language/run.py --provider gemini --model gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.language.models import make_parser        # noqa: E402
from benchmark.language.state import route_summary       # noqa: E402
from core.landmark_kb import LandmarkKnowledgeBase       # noqa: E402
from core.paths import KB_CSV                            # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "episodes_template.jsonl"
RUNS = ROOT / "benchmark" / "runs"


def load_episodes(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class RateLimiter:
    """At most ``rpm`` requests per minute across all worker threads -- free
    API tiers cap requests per minute, and a 429 costs more than waiting."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            start = max(now, self.next)
            self.next = start + self.interval
        time.sleep(max(0.0, start - now))


def retry_delay(exc: Exception, attempt: int) -> float:
    """The provider's suggested wait (Gemini's 429 carries retryDelay), else
    exponential backoff."""
    m = re.search(r"retryDelay'?:\s*'?(\d+(?:\.\d+)?)s", str(exc))
    return float(m.group(1)) + 1.0 if m else min(60.0, 2.0 ** attempt)


def prompt_version() -> str:
    """Fingerprint of the parser prompt (schema, examples, route rules). It is
    part of the run name, so answers to an older prompt are never resumed
    into, or scored as, a run of the current one."""
    from core.llm_inference import _build_system_prompt
    route = {"destination": "cafe", "destination_index": None,
             "stops": [{"type": "park", "index": None}]}
    text = (_build_system_prompt("Town01", ["cafe", "park"], None)
            + _build_system_prompt("Town01", ["cafe", "park"], route))
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def run_name(provider: str, model: str, data: Path, context: bool, repeat: int) -> str:
    return (f"{provider}__{model}__{data.stem}__p{prompt_version()}"
            + ("" if context else "__nocontext") + (f"__r{repeat}" if repeat else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=None, help="default: benchmark/runs/<run name>.jsonl")
    ap.add_argument("--no-context", action="store_true",
                    help="ablation: do not show the model the current route")
    ap.add_argument("--repeat", type=int, default=0, help="repeat index, for variance runs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N episodes only")
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--rpm", type=float, default=12,
                    help="max requests per minute over all workers (0 = no limit)")
    args = ap.parse_args()

    data = Path(args.data)
    episodes = load_episodes(data)[: args.limit]
    context = not args.no_context
    out = Path(args.out) if args.out else RUNS / (run_name(args.provider, args.model, data,
                                                          context, args.repeat) + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["id"], r["turn"], r["utterance"]))
    # Keyed on the sentence too: a regenerated or re-verified turn is re-asked.
    jobs = [(e, i, t) for e in episodes for i, t in enumerate(e["turns"])
            if (e["id"], i, t["utterance"]) not in done]
    print(f"[run] {out.name}: {len(jobs)} turns to do, {len(done)} already done")

    kb = LandmarkKnowledgeBase(str(KB_CSV))
    local = threading.local()
    lock = threading.Lock()
    progress = {"n": 0, "failed": 0}
    limiter = RateLimiter(args.rpm)

    def parser():
        # One parser per thread: it records the last raw answer and usage.
        if not hasattr(local, "parser"):
            local.parser = make_parser(args.provider, args.model, kb)
        return local.parser

    def work(job) -> None:
        episode, i, turn = job
        route = route_summary(turn["state_before"]) if context else None
        record = {"id": episode["id"], "turn": i, "provider": args.provider,
                  "model": args.model, "prompt": prompt_version(), "context": context,
                  "repeat": args.repeat,
                  "utterance": turn["utterance"]}
        for attempt in range(args.retries):
            p = parser()
            limiter.wait()
            t0 = time.time()
            try:
                intent = p.parse(turn["utterance"], town=episode["town"], route=route)
            except Exception as exc:                 # 429 / 503 / network
                record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                time.sleep(retry_delay(exc, attempt))
                continue
            record.update(intent=intent, raw=p.last_raw, usage=p.last_usage,
                          latency_s=round(time.time() - t0, 3), error=None)
            break
        with lock:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress["n"] += 1
            progress["failed"] += bool(record.get("error"))
            if progress["n"] % 50 == 0 or progress["n"] == len(jobs):
                print(f"[run] {progress['n']}/{len(jobs)} "
                      f"({progress['failed']} failed after retries)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, jobs))
    print(f"[run] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
