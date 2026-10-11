"""
runtime/fpv.py
==============
A window showing exactly what the driving model saw at its last inference.

The camera frame is shown as sent to the model, with what the backend's
:class:`~backends.base.FpvFrame` adds on top: a dimmed bottom band where the
model crops the image (SimLingo's ``cut_bottom_quarter``, which is how the
ego's own roof stays out of view), and the model's predicted waypoints
projected onto the road (TCP). Below it, a strip with the control the model
returned, the speed, any speed instruction and the model's own output.
Works with any backend that exposes ``last_frame`` (SimLingo, leaderboard
agents with a forward camera).
"""

from __future__ import annotations

import math
import textwrap

import cv2
import numpy as np

from .gui import Panel

WINDOW = "talk2drive FPV - model input"
STRIP_H = 110
WHITE, GREY, CYAN = (255, 255, 255), (170, 170, 170), (0, 220, 255)
YELLOW, GREEN = (255, 220, 0), (0, 230, 0)


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


def project(camera: dict, size: tuple[int, int], forward: float, right: float):
    """Ground point in the ego frame (metres) -> (u, v) pixel, or None when it
    is behind the camera or off the image. Pinhole camera with CARLA's
    horizontal ``fov``; the ground is taken to be at the ego's origin height."""
    h, w = size
    f = w / (2.0 * math.tan(math.radians(float(camera.get("fov", 90.0))) / 2.0))
    x = forward - float(camera.get("x", 0.0))          # camera frame: x ahead
    y = right - float(camera.get("y", 0.0))            # y right
    z = -float(camera.get("z", 0.0))                   # ground, z up
    pitch = math.radians(float(camera.get("pitch", 0.0)))
    depth = x * math.cos(pitch) + z * math.sin(pitch)
    up = -x * math.sin(pitch) + z * math.cos(pitch)
    if depth <= 0.1:
        return None
    u, v = w / 2.0 + f * y / depth, h / 2.0 - f * up / depth
    if not (0 <= u < w and 0 <= v < h):
        return None
    return int(round(u)), int(round(v))


def annotate(frame) -> np.ndarray:
    """RGB image: the model's input over an info strip."""
    img = np.ascontiguousarray(frame.rgb).copy()
    h, w = img.shape[:2]

    if frame.crop_from_row is not None:
        crop_h = frame.crop_from_row
        img[crop_h:] = (img[crop_h:] * 0.35).astype(np.uint8)
        cv2.line(img, (0, crop_h), (w, crop_h), YELLOW, 1, cv2.LINE_AA)
        cv2.putText(img, "below: cropped off, not seen by the model", (10, crop_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

    if frame.camera and frame.waypoints:
        pts = [project(frame.camera, (h, w), fwd, right) for fwd, right in frame.waypoints]
        pts = [p for p in pts if p is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, GREEN, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 4, GREEN, -1, cv2.LINE_AA)

    strip = np.zeros((STRIP_H, w, 3), dtype=np.uint8)
    ctrl = frame.control
    lines = [
        (f"steer {ctrl.get('steer', 0):+.2f}   throttle {ctrl.get('throttle', 0):.2f}   "
         f"brake {ctrl.get('brake', 0):.2f}   speed {frame.speed_mps * 3.6:.1f} km/h", WHITE),
        (f"instruction: {frame.instruction}" if frame.instruction else "", GREY),
    ]
    out = []
    for note in frame.notes:
        out += textwrap.wrap(f"{frame.model}: {note}", width=max(40, w // 11))
    for part in out[:3]:
        lines.append((part, CYAN))
    for i, (text, color) in enumerate(lines):
        cv2.putText(strip, text, (10, 22 + i * 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
    return np.vstack([img, strip])
