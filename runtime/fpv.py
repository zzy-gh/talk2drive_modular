"""
runtime/fpv.py
==============
A window showing exactly what the driving model saw at its last inference.

The camera frame is shown as sent to the model, untouched except for a
dimmed bottom band: the
worker crops that band off before the model sees it (SimLingo's
``cut_bottom_quarter``), which is how the ego's own roof stays out of view. Below it, a strip with the
model's language output, the control it returned, the speed and any speed
instruction. Works with any backend that exposes ``last_frame`` (SimLingo).
"""

from __future__ import annotations

import textwrap

import cv2
import numpy as np

from .gui import Panel

WINDOW = "talk2drive FPV - model input"
STRIP_H = 110
WHITE, GREY, CYAN = (255, 255, 255), (170, 170, 170), (0, 220, 255)
YELLOW = (255, 220, 0)


class FpvPanel(Panel):
    window = WINDOW

    def __init__(self, backend) -> None:
        if not hasattr(backend, "last_frame"):
            raise ValueError(f"backend {backend.name!r} has no camera input to show")
        self.backend = backend
        self._shown = None

    def frame(self):
        frame = self.backend.last_frame
        if frame is None or frame is self._shown:
            return None
        self._shown = frame
        img = annotate(frame)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def annotate(frame: tuple) -> np.ndarray:
    """RGB image: the model's input over an info strip."""
    rgb, _, response, instruction, speed_mps = frame
    img = np.ascontiguousarray(rgb).copy()
    h, w = img.shape[:2]

    # Same arithmetic as simlingo_inference/model_runner.py::_build_driving_input.
    crop_h = int(h - (h * 4.8) // 16)
    img[crop_h:] = (img[crop_h:] * 0.35).astype(np.uint8)
    cv2.line(img, (0, crop_h), (w, crop_h), YELLOW, 1, cv2.LINE_AA)
    cv2.putText(img, "below: cropped off, not seen by the model", (10, crop_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

    strip = np.zeros((STRIP_H, w, 3), dtype=np.uint8)
    ctrl = response.get("control") or {}
    lines = [
        (f"steer {ctrl.get('steer', 0):+.2f}   throttle {ctrl.get('throttle', 0):.2f}   "
         f"brake {ctrl.get('brake', 0):.2f}   speed {speed_mps * 3.6:.1f} km/h", WHITE),
        (f"instruction: {instruction}" if instruction else "", GREY),
    ]
    lang = str(response.get("language") or "")
    for part in textwrap.wrap("SimLingo: " + lang, width=max(40, w // 11))[:3]:
        lines.append((part, CYAN))
    for i, (text, color) in enumerate(lines):
        cv2.putText(strip, text, (10, 22 + i * 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
    return np.vstack([img, strip])
