"""
core/llm_worker.py
===================
JSON-line worker that runs ``GeminiCommandParser`` in its own process, so a
driving process (torch, an ancient pinned conda env, whatever) never has to
import ``google.genai`` directly -- only this worker's own environment needs
``google-genai`` installed.

Mirrors the ``simlingo_inference.worker_server`` / ``SimLingoWorker`` bridge
in ``backends/simlingo.py``: one JSON object per line on stdin, one JSON
object per line on stdout, nothing else on the wire.

    {"type": "parse", "text": "take me to the hospital", "town": "Town01"}
        -> {"status": "ok", "intent": {...} | null}

    {"type": "shutdown"}
        -> {"status": "ok"}

Run with the LLM environment's own interpreter, from the package root:

    python -m core.llm_worker [--csv path/to/special_buildings_en.csv]

Not launched directly by a human -- ``core/llm_bridge.py::LLMBridge`` spawns
this as a subprocess.
"""

from __future__ import annotations

import argparse
import json
import sys

from .landmark_kb import LandmarkKnowledgeBase
from .llm_inference import GeminiCommandParser
from .paths import KB_CSV


def _reply(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None,
                    help="override [paths] kb_csv from config.yaml")
    args = ap.parse_args()

    csv_path = args.csv or KB_CSV
    if not csv_path:
        _reply({"status": "error",
                "message": "no kb csv: pass --csv or set [paths] kb_csv"})
        return
    kb = LandmarkKnowledgeBase(str(csv_path))
    parser = GeminiCommandParser(kb)
    _reply({"status": "ok", "message": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _reply({"status": "error", "message": "request was not valid JSON"})
            continue

        kind = req.get("type")
        if kind == "shutdown":
            _reply({"status": "ok"})
            return
        if kind != "parse":
            _reply({"status": "error", "message": f"unknown request type {kind!r}"})
            continue

        try:
            intent = parser.parse(req["text"], town=req["town"])
            _reply({"status": "ok", "intent": intent})
        except Exception as exc:                      # worker must never crash silently
            _reply({"status": "error", "message": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
