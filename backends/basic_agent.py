"""
backends/basic_agent.py
=======================
CARLA ``BasicAgent`` (waypoint follower + PID). The original talk2drive
controller, now behind the common interface — and the reference behaviour
every other backend is compared against.
"""

from __future__ import annotations

from core.types import Directive, GlobalPlan, Observation

from .base import BackendCaps, DrivingBackend, brake_control, register


@register("basic_agent")
class BasicAgentBackend(DrivingBackend):

    caps = BackendCaps(
        plan_format="dense",
        consumes_language=False,     # urgency has to be realised as target speed
        control_rate_hz=20.0,
        reports_done=True,           # the local planner knows when the queue is empty
    )

    def __init__(self, vehicle=None, world=None, target_speed: float = 30.0,
                 opt_dict: dict | None = None) -> None:
        self.target_speed = target_speed
        self.opt_dict = opt_dict or {}
        self.agent = None
        self.map = None
        self._active = False
        if vehicle is not None:
            self.attach(vehicle, world)

    def attach(self, vehicle, world) -> None:
        from agents.navigation.basic_agent import BasicAgent

        self.agent = BasicAgent(vehicle, target_speed=self.target_speed,
                                opt_dict=self.opt_dict)
        self.map = (world or vehicle.get_world()).get_map()

    def set_plan(self, plan: GlobalPlan, directive: Directive) -> None:
        if self.agent is None:
            raise RuntimeError("BasicAgentBackend.attach() was never called")
        speed = directive.target_speed_kph()
        if hasattr(self.agent, "set_target_speed"):
            self.agent.set_target_speed(speed)
        else:                                     # older CARLA API
            self.agent._local_planner.set_speed(speed)
        self.agent.set_global_plan(plan.to_carla_plan(self.map),
                                   stop_waypoint_creation=True, clean_queue=True)
        self._active = True

    def run_step(self, obs: Observation):
        if not self._active or self.agent is None:
            return brake_control()
        return self.agent.run_step()

    def done(self) -> bool:
        return bool(self._active and self.agent is not None and self.agent.done())

    def cancel(self) -> None:
        self._active = False

    def destroy(self) -> None:
        self._active = False
        self.agent = None
