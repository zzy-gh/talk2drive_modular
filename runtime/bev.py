"""
runtime/bev.py
==============
A bird's-eye-view window drawn with OpenCV: the town's roads, the planned
route, the landmarks coloured by their mission state, and the ego -- the
whole town, or in a large one a window that follows the ego.

Nothing here touches ``world.debug``, so unlike the in-world drawing in
runtime/viz.py none of it can end up in a camera sensor's image.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict

import cv2
import numpy as np

from .gui import Panel
from .viz import LandmarkBoard

WINDOW = "talk2drive BEV"

# BGR
BG = (32, 32, 32)
ROAD = (88, 88, 88)
ROUTE = (0, 200, 0)
ROUTE_URGENT = (0, 120, 255)
EGO = (255, 220, 0)
TEXT = (230, 230, 230)
CANDIDATE = (0, 220, 255)


def _bgr(color) -> tuple[int, int, int]:
    return int(color.b), int(color.g), int(color.r)


# state -> BGR, from the same palette the in-world markers use
STATE_BGR = {state: _bgr(style[0]) for state, style in LandmarkBoard.STYLE.items()}
LEGEND = [("unused", STATE_BGR["idle"]), ("destination", STATE_BGR["destination"]),
          ("stop", STATE_BGR["stop"]), ("reached", STATE_BGR["visited"]),
          ("cancelled", STATE_BGR["cancelled"]), ("candidate [n]", CANDIDATE),
          ("route", ROUTE), ("ego", EGO)]


class BevRenderer:
    """World (x, y) -> pixels for one town, plus the static road layer.

    CARLA's y axis points "down" in a top view, so pixels map straight from
    (x, y) without a flip -- the same orientation as CARLA's own map images.

    A town wider than ``follow_above_m`` would be a few metres to the pixel
    in one window, its landmarks and route unreadable. There the view
    instead follows the ego, ``window_m`` across. Its roads are drawn in
    tiles as the ego reaches them, and the latest ``max_tiles`` kept.
    """

    def __init__(self, carla_map, size_px: int = 900, margin_px: int = 30,
                 step_m: float = 2.0, window_m: float = 600.0,
                 follow_above_m: float = 1500.0, max_tiles: int = 64) -> None:
        wps = carla_map.generate_waypoints(step_m)
        xs = [w.transform.location.x for w in wps]
        ys = [w.transform.location.y for w in wps]
        self.min_x, self.min_y = min(xs), min(ys)
        span = max(max(xs) - self.min_x, max(ys) - self.min_y, 1.0)
        self.follow = span > follow_above_m

        # (x0, y0, x1, y1, lane width) for every road piece
        segs = [(w.transform.location.x, w.transform.location.y,
                 n.transform.location.x, n.transform.location.y, w.lane_width)
                for w in wps for n in w.next(step_m)]
        segs = np.array(segs, dtype=np.float64).reshape(-1, 5)

        if not self.follow:
            self.scale = (size_px - 2 * margin_px) / span            # px per metre
            self.margin = margin_px
            self.w = int((max(xs) - self.min_x) * self.scale) + 2 * margin_px
            self.h = int((max(ys) - self.min_y) * self.scale) + 2 * margin_px
            self.background = np.full((self.h, self.w, 3), BG, dtype=np.uint8)
            self._draw_roads(self.background, segs, self.min_x, self.min_y, self.margin)
            return

        self.scale = size_px / window_m
        self.margin = 0
        self.w = self.h = size_px
        self.window_m = window_m
        self._tile_m = window_m / 2
        self._tile_px = int(math.ceil(self._tile_m * self.scale))
        self._segs = segs
        keys = np.floor(segs[:, :2] / self._tile_m).astype(np.int64)
        self._bucket: dict[tuple[int, int], np.ndarray] = {}
        for k in np.unique(keys, axis=0):
            self._bucket[(int(k[0]), int(k[1]))] = np.flatnonzero((keys == k).all(axis=1))
        self._tiles: OrderedDict = OrderedDict()
        self._max_tiles = max_tiles

    def _draw_roads(self, img, segs, x0: float, y0: float, margin: int) -> None:
        for ax, ay, bx, by, lane_w in segs:
            a = (int(round((ax - x0) * self.scale)) + margin,
                 int(round((ay - y0) * self.scale)) + margin)
            b = (int(round((bx - x0) * self.scale)) + margin,
                 int(round((by - y0) * self.scale)) + margin)
            cv2.line(img, a, b, ROAD, max(1, int(round(lane_w * self.scale))), cv2.LINE_AA)

    def _tile(self, ix: int, iy: int) -> np.ndarray:
        key = (ix, iy)
        if key in self._tiles:
            self._tiles.move_to_end(key)
            return self._tiles[key]
        img = np.full((self._tile_px, self._tile_px, 3), BG, dtype=np.uint8)
        # A piece starting in a neighbouring tile may reach into this one
        near = [self._bucket[k] for k in ((ix + dx, iy + dy) for dx in (-1, 0, 1)
                                          for dy in (-1, 0, 1)) if k in self._bucket]
        if near:
            self._draw_roads(img, self._segs[np.concatenate(near)],
                             ix * self._tile_m, iy * self._tile_m, 0)
        self._tiles[key] = img
        if len(self._tiles) > self._max_tiles:
            self._tiles.popitem(last=False)
        return img

    def _window(self, cx: float, cy: float) -> np.ndarray:
        """The road layer ``window_m`` across, centred on (cx, cy); sets the
        origin :meth:`px` maps from."""
        self.min_x, self.min_y = cx - self.window_m / 2, cy - self.window_m / 2
        img = np.full((self.h, self.w, 3), BG, dtype=np.uint8)
        t = self._tile_m
        for ix in range(int(math.floor(self.min_x / t)), int(math.floor((self.min_x + self.window_m) / t)) + 1):
            for iy in range(int(math.floor(self.min_y / t)), int(math.floor((self.min_y + self.window_m) / t)) + 1):
                ox = int(round((ix * t - self.min_x) * self.scale))   # tile origin in the window
                oy = int(round((iy * t - self.min_y) * self.scale))
                x0, y0 = max(ox, 0), max(oy, 0)
                x1, y1 = min(ox + self._tile_px, self.w), min(oy + self._tile_px, self.h)
                if x0 < x1 and y0 < y1:
                    img[y0:y1, x0:x1] = self._tile(ix, iy)[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        return img

    def px(self, x: float, y: float) -> tuple[int, int]:
        return (int(round((x - self.min_x) * self.scale)) + self.margin,
                int(round((y - self.min_y) * self.scale)) + self.margin)

    def render(self, ego=None, plan=None, urgency: str = "normal",
               landmarks=()) -> np.ndarray:
        """``ego`` is (x, y, yaw_deg); ``landmarks`` is LandmarkBoard.snapshot()."""
        if not self.follow:
            img = self.background.copy()
        elif ego is not None:
            img = self._window(ego[0], ego[1])
        else:
            img = self._window(self.min_x + self.window_m / 2, self.min_y + self.window_m / 2)

        if plan is not None and len(plan.points) > 1:
            pts = np.array([self.px(p.x, p.y) for p in plan.points], dtype=np.int32)
            cv2.polylines(img, [pts], False, ROUTE_URGENT if urgency == "high" else ROUTE,
                          2, cv2.LINE_AA)

        for place, state, n, cand in landmarks:
            c = self.px(place.x, place.y)
            if cand is not None:
                color, label, r = CANDIDATE, f"[{cand}] {place.label}", 7
            else:
                color, r = STATE_BGR.get(state, STATE_BGR["idle"]), 4 if state == "idle" else 7
                label = (f"{n}. " if state == "stop" and n else "") + place.tag
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c[0] + 9, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)

        if ego is not None:
            x, y, yaw = ego
            f = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
            s = (-f[1], f[0])
            size = 9.0
            tri = [(x + f[0] * size / self.scale, y + f[1] * size / self.scale),
                   (x + (-f[0] * 0.6 + s[0] * 0.6) * size / self.scale,
                    y + (-f[1] * 0.6 + s[1] * 0.6) * size / self.scale),
                   (x + (-f[0] * 0.6 - s[0] * 0.6) * size / self.scale,
                    y + (-f[1] * 0.6 - s[1] * 0.6) * size / self.scale)]
            cv2.fillPoly(img, [np.array([self.px(*p) for p in tri], dtype=np.int32)],
                         EGO, cv2.LINE_AA)

        return np.hstack([img, _legend(img.shape[0])])


def _legend(height: int, width: int = 150) -> np.ndarray:
    """A side column, so the key never covers the map."""
    col = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0, dy = 14, 22, 20
    for i, (name, color) in enumerate(LEGEND):
        y = y0 + i * dy
        cv2.circle(col, (x0, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(col, name, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1,
                    cv2.LINE_AA)
    return col


class BevPanel(Panel):
    """The BEV of a running DriveSession."""

    window = WINDOW

    def __init__(self, session, refresh_s: float = 0.1) -> None:
        self.session = session
        self.refresh_s = refresh_s
        self.renderer = None
        self._next = 0.0

    def setup(self) -> None:
        self.renderer = BevRenderer(self.session.world.get_map())   # roads, once

    def frame(self):
        now = time.time()
        if now < self._next:
            return None
        self._next = now + self.refresh_s
        s = self.session
        tf = s.vehicle.get_transform()
        s.viz.board.set_ego(tf.location.x, tf.location.y)
        return self.renderer.render(ego=(tf.location.x, tf.location.y, tf.rotation.yaw),
                                    plan=s.viz.plan, urgency=s.viz.urgency,
                                    landmarks=s.viz.board.snapshot())
