"""
backends/base.py
================
The plug: everything a driving stack must expose for talk2drive to steer it.

The three target stacks disagree on almost everything (plan format, control
rate, whether they read language, who owns the loop) — but they all agree on
"take a plan, return a VehicleControl". That is the whole interface;
:class:`BackendCaps` describes the disagreements so the runtime can adapt
instead of the backend pretending to be something it is not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from core.types import Directive, GlobalPlan, Observation


@dataclass(frozen=True)
class BackendCaps:
    plan_format: str                 # "dense" | "sparse_target_points" | "gps"
    consumes_language: bool = False  # can it act on the raw utterance?
    control_rate_hz: float = 20.0    # how often run_step is worth calling
    owns_loop: bool = False          # True: an external framework ticks it
    needs_sensors: bool = False      # True: runtime must spawn its sensor rig
    reports_done: bool = False       # True: done() is authoritative


class DrivingBackend(ABC):
    """One driving stack, behind one interface."""

    caps: BackendCaps = BackendCaps(plan_format="dense")
    name: str = "backend"

    # ── wiring ──────────────────────────────────────────────────────────
    def sensors(self) -> list[dict]:
        """Leaderboard-style sensor spec; empty for stacks with map access."""
        return []

    def attach(self, vehicle, world) -> None:
        """Called once the ego actor exists. Optional."""

    # ── mission ─────────────────────────────────────────────────────────
    @abstractmethod
    def set_plan(self, plan: GlobalPlan, directive: Directive) -> None:
        """Adopt a new route. Called on every replan, including mid-drive."""

    # ── control ─────────────────────────────────────────────────────────
    @abstractmethod
    def run_step(self, obs: Observation) -> Any:
        """Return a ``carla.VehicleControl`` for this tick."""

    def done(self) -> bool:
        """Arrived? Only trusted when ``caps.reports_done`` is set — a VLA has
        no idea where the mission ends, so the mission layer decides."""
        return False

    def cancel(self) -> None:
        """Drop the current plan and come to a stop."""

    def destroy(self) -> None:
        """Release subprocesses, sensors, GPU memory."""


# ─────────────────────────────────────────────
# Registry — this is what makes it plug-out
# ─────────────────────────────────────────────

_REGISTRY: dict[str, Callable[..., DrivingBackend]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(name: str, **kwargs) -> DrivingBackend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {available()}")
    return _REGISTRY[name](**kwargs)


def brake_control():
    import carla
    return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
