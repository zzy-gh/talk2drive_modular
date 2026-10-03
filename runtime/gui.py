"""
runtime/gui.py
==============
One thread for every OpenCV window. Qt-backed HighGUI must be driven from a
single thread, so the FPV and BEV views are panels that only produce images;
this loop owns the windows.
"""

from __future__ import annotations

import os
import threading
import time

import cv2


class Panel:
    """A window's content. ``setup`` runs once on the GUI thread; ``frame``
    returns a BGR image to show, or None to leave the window as it is."""

    window = "panel"

    def setup(self) -> None:
        pass

    def frame(self):
        raise NotImplementedError


class GuiLoop:
    def __init__(self, panels: list[Panel], tick_s: float = 0.03) -> None:
        self.panels = panels
        self.tick_s = tick_s
        self._running = True
        self._thread = None
        if not panels:
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            # Qt aborts the whole process when there is no display to open.
            print("[gui] no DISPLAY, windows disabled")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        live = []
        for p in self.panels:
            try:
                p.setup()
                cv2.namedWindow(p.window, cv2.WINDOW_NORMAL)
                live.append(p)
            except Exception as exc:
                print(f"[gui] {p.window}: setup failed: {exc}")
        while self._running:
            t0 = time.time()
            for p in live:
                try:
                    img = p.frame()
                    if img is not None:
                        cv2.imshow(p.window, img)
                except Exception as exc:
                    print(f"[gui] {p.window}: {exc}")
            cv2.waitKey(max(1, int((self.tick_s - (time.time() - t0)) * 1000)))
        for p in live:
            cv2.destroyWindow(p.window)
        cv2.waitKey(1)

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
