"""
backends/simlingo.py
====================
SimLingo (VLA) behind the common interface.

Two things make this backend different from the PID one, and both are handled
here rather than leaking into the mission layer:

1. **It lives in another Python environment.** SimLingo needs torch +
   transformers + a checkpoint; talk2drive needs the CARLA egg and the Gemini
   SDK. So we do not import it — we speak newline-delimited JSON to
   ``simlingo_inference.worker_server`` over a pipe. That subprocess boundary
   is what makes "plug out" literal: no shared dependency graph at all.

2. **It is slow and language-native.** Inference runs at ~1-2 Hz while the sim
   ticks at 20 Hz, so the last control is held between inferences; and the
   passenger's urgency goes in as a *sentence* appended to the prompt rather
   than as a target-speed number.
"""

from __future__ import annotations

import base64
import json
import os
import select
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from core.tracking import SparseRouteTracker
from core.types import Directive, GlobalPlan, Observation

from .base import BackendCaps, DrivingBackend, brake_control, register

from core.paths import SIMLINGO_PYTHON, SIMLINGO_REPO


# ─────────────────────────────────────────────
# Worker client
# ─────────────────────────────────────────────

class SimLingoWorker:
    """Thin JSON-over-stdio client for ``simlingo_inference.worker_server``."""

    def __init__(self, checkpoint_path: str, repo_path: str | None = None,
                 python_bin: str | None = None, device: str = "cuda",
                 cache_root: str = "pretrained", use_cot: bool = True,
                 hydra_config_path: str | None = None,
                 request_timeout_s: float = 120.0) -> None:
        self.repo = Path(repo_path or SIMLINGO_REPO).expanduser().resolve()
        self.timeout = request_timeout_s
        if not (self.repo / "simlingo_inference" / "worker_server.py").is_file():
            raise FileNotFoundError(
                f"{self.repo} has no simlingo_inference/worker_server.py. The "
                f"inference worker ships with the CoVLM fork, not the official "
                f"SimLingo repo — point --simlingo-repo or [simlingo] repo "
                f"in config.yaml at "
                f"a checkout that has it.")
        python_bin = str(python_bin or SIMLINGO_PYTHON or "python3")
        python_bin = python_bin if os.path.exists(python_bin) else (shutil.which("python3") or "python3")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.repo)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        self.proc = subprocess.Popen(
            [python_bin, "-m", "simlingo_inference.worker_server"],
            cwd=str(self.repo), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1,
        )
        print(f"[simlingo] worker pid={self.proc.pid} ({python_bin})")
        self.request({"type": "init",
                      "checkpoint_path": str(checkpoint_path),
                      "hydra_config_path": hydra_config_path,
                      "device": device,
                      "cache_root": cache_root,
                      "use_cot": bool(use_cot)})
        print("[simlingo] worker initialised")

    def request(self, payload: dict, timeout_s: float | None = None) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("simlingo worker is not running")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

        deadline = time.time() + (timeout_s or self.timeout)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("simlingo worker timed out")
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                if self.proc.poll() is not None:
                    raise RuntimeError("simlingo worker exited")
                continue
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("simlingo worker closed its pipe")
            response = json.loads(line)
            if response.get("status") != "ok":
                raise RuntimeError(response.get("message", "simlingo worker error"))
            return response

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.request({"type": "shutdown"}, timeout_s=3.0)
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


# ─────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────

@register("simlingo")
class SimLingoBackend(DrivingBackend):

    caps = BackendCaps(
        plan_format="sparse_target_points",
        consumes_language=True,
        control_rate_hz=1.5,          # ~ every 15 sim ticks at 20 FPS
        needs_sensors=True,
        reports_done=False,           # a VLA has no notion of "mission complete"
    )

    CAMERA_ID = "rgb_front"

    def __init__(self, checkpoint_path: str, repo_path: str | None = None,
                 python_bin: str | None = None, device: str = "cuda",
                 hydra_config_path: str | None = None,
                 inference_interval: int = 15,
                 camera: dict | None = None,
                 min_distance: float = 7.5, max_distance: float = 50.0,
                 route_sample_factor: float = 200.0,
                 pass_language: bool = True, debug: bool = False) -> None:
        self.inference_interval = max(1, int(inference_interval))
        self.camera = {"x": -1.5, "y": 0.0, "z": 2.0, "roll": 0.0, "pitch": 0.0,
                       "yaw": 0.0, "width": 1024, "height": 512, "fov": 110.0,
                       **(camera or {})}
        self.pass_language = pass_language
        self.debug = debug

        # 200 m, not the usual 50: SimLingo's leaderboard fork calls
        # downsample_route(plan, 200) in set_global_plan, so that is the
        # sparsity its target points were trained on. min/max distance match
        # team_code/agent_simlingo.py's RoutePlanner(7.5, 50).
        self.tracker = SparseRouteTracker(min_distance=min_distance,
                                          max_distance=max_distance,
                                          sample_factor=route_sample_factor)
        self._plan: GlobalPlan | None = None
        self._instruction: str | None = None
        self._cached = None
        self._step = 0
        self.last_language: str | None = None

        self.worker = SimLingoWorker(checkpoint_path=checkpoint_path,
                                     repo_path=repo_path, python_bin=python_bin,
                                     device=device, hydra_config_path=hydra_config_path)

    # ── wiring ──────────────────────────────────────────────────────────
    def sensors(self) -> list[dict]:
        return [{"type": "sensor.camera.rgb", "id": self.CAMERA_ID, **self.camera}]

    # ── mission ─────────────────────────────────────────────────────────
    def set_plan(self, plan: GlobalPlan, directive: Directive) -> None:
        self._plan = plan
        self.tracker.set_plan(plan)
        self._instruction = directive.as_speed_instruction() if self.pass_language else None
        self._cached = None                     # force a fresh inference
        if self.debug:
            print(f"[simlingo] plan rev={plan.revision} sparse={len(self.tracker)} "
                  f"instruction={self._instruction!r}")

    # ── control ─────────────────────────────────────────────────────────
    def run_step(self, obs: Observation):
        if self._plan is None:
            return brake_control()

        loc = obs.ego_transform.location
        self.tracker.run_step(loc.x, loc.y, obs.ego_transform.rotation.yaw)
        self._step += 1

        if self._cached is not None and self._step % self.inference_interval != 0:
            return self._to_control(self._cached)

        rgb = self._front_rgb(obs)
        if rgb is None:
            return self._to_control(self._cached) if self._cached else brake_control()

        target_points = self.tracker.target_points_ego(obs.ego_transform, n=2)
        payload = {
            "type": "predict",
            "front_rgb_png": _encode_png(rgb),
            "speed_mps": float(obs.speed_mps),
            "target_points_ego": np.asarray(target_points, dtype=np.float32).tolist(),
        }
        if self._instruction:
            payload["speed_instruction"] = self._instruction

        try:
            response = self.worker.request(payload)
        except Exception as exc:
            print(f"[simlingo] inference failed: {exc}")
            return self._to_control(self._cached) if self._cached else brake_control()

        self._cached = response["control"]
        self.last_language = response.get("language")
        if self.debug:
            print(f"[simlingo] tp={np.array2string(target_points, precision=1)} "
                  f"ctrl={self._cached} lang={str(self.last_language)[:80]!r}")
        return self._to_control(self._cached)

    def cancel(self) -> None:
        self._plan = None
        self._cached = None
        self.tracker.set_plan(None)

    def destroy(self) -> None:
        self.worker.close()

    # ── helpers ─────────────────────────────────────────────────────────
    def _front_rgb(self, obs: Observation):
        data = obs.sensors.get(self.CAMERA_ID)
        if data is None:
            return None
        if isinstance(data, tuple):             # leaderboard: (frame, array)
            data = data[1]
        arr = np.asarray(data)
        if arr.ndim != 3:
            return None
        if arr.shape[2] == 4:                   # BGRA from CARLA
            arr = arr[:, :, :3][:, :, ::-1]     # -> RGB
        return np.ascontiguousarray(arr, dtype=np.uint8)

    @staticmethod
    def _to_control(control: dict | None):
        import carla
        if not control:
            return brake_control()
        return carla.VehicleControl(
            steer=float(np.clip(control["steer"], -1.0, 1.0)),
            throttle=float(np.clip(control["throttle"], 0.0, 1.0)),
            brake=float(np.clip(control["brake"], 0.0, 1.0)),
        )


def _encode_png(rgb: np.ndarray) -> str:
    """PNG-encode an RGB array the way the worker expects to decode it."""
    try:
        import cv2
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("cv2 failed to encode front RGB")
        raw = buf.tobytes()
    except ImportError:
        import io
        from PIL import Image
        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, format="PNG")
        raw = bio.getvalue()
    return base64.b64encode(raw).decode("ascii")
