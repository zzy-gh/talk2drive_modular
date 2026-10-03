"""
core/llm_bridge.py
===================
Runs ``core/llm_worker.py`` in a subprocess -- its own Python environment,
its own ``google-genai`` install -- and exposes the exact same
``.parse(text, town, route=None) -> dict | None`` method as ``GeminiCommandParser``
itself (``core/llm_inference.py``). Callers (``runtime/cli.py``,
``runtime/passenger.py``, ...) construct one or the other and never see
the difference.

Same shape as ``backends/simlingo.py::SimLingoWorker``, just for the parsing
step instead of the driving step: newline-delimited JSON over stdin/stdout,
nothing else on the wire.
"""

from __future__ import annotations

import json
import select
import subprocess
import time
from pathlib import Path
from typing import Optional


class LLMBridge:
    """Drop-in replacement for ``GeminiCommandParser`` that runs Gemini in a
    separate process/environment."""

    def __init__(self, python_bin: str, csv_path: str | None = None,
                 request_timeout_s: float = 30.0,
                 cmd: list[str] | None = None) -> None:
        """``cmd``, if given, replaces the ``python -m core.llm_worker``
        argv entirely -- only ``tools/selftest.py`` uses this, to run a
        synthetic stub worker instead of the real (google-genai-dependent)
        one."""
        self.timeout = request_timeout_s
        repo_root = Path(__file__).resolve().parents[1]
        if cmd is None:
            cmd = [python_bin, "-m", "core.llm_worker"]
            if csv_path:
                cmd += ["--csv", str(csv_path)]
        self.proc = subprocess.Popen(
            cmd, cwd=str(repo_root),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1,
        )
        print(f"[llm_bridge] worker pid={self.proc.pid} ({python_bin})")
        ready = self._read_reply(self.timeout)
        if ready.get("status") != "ok":
            raise RuntimeError(ready.get("message", "llm worker failed to start"))
        print("[llm_bridge] worker ready")

    # ── public interface: matches GeminiCommandParser.parse() ─────────────
    def parse(self, text: str, town: str, route: Optional[dict] = None) -> Optional[dict]:
        req = {"type": "parse", "text": text, "town": town}
        if route is not None:
            req["route"] = route
        try:
            response = self._request(req)
        except Exception as exc:
            print(f"[llm_bridge] request failed: {exc}")
            return None
        if response.get("status") != "ok":
            print(f"[llm_bridge] {response.get('message')}")
            return None
        return response.get("intent")

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self._request({"type": "shutdown"}, timeout_s=3.0)
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None

    # ── wire protocol ───────────────────────────────────────────────────
    def _request(self, payload: dict, timeout_s: float | None = None) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("llm worker is not running")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        return self._read_reply(timeout_s or self.timeout)

    def _read_reply(self, timeout_s: float) -> dict:
        deadline = time.time() + timeout_s
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("llm worker timed out")
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                if self.proc.poll() is not None:
                    raise RuntimeError("llm worker exited")
                continue
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("llm worker closed its pipe")
            return json.loads(line)
