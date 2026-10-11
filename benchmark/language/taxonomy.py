"""
benchmark/language/taxonomy.py
==============================
The functional intents the benchmark must cover. Every turn in the dataset is
tagged with the functions it exercises, and ``generate.py`` refuses to write a
dataset in which any function is missing.

IDs are stable: they appear in the dataset, the predictions and the paper's
tables, so add new ones rather than renumbering.
"""

from __future__ import annotations

# id -> (command_type, description)
FUNCTIONS: dict[str, tuple[str, str]] = {
    # plan a new route
    "P1": ("plan_route", "destination only"),
    "P2": ("plan_route", "indexed destination ('the second cafe')"),
    "P3": ("plan_route", "ordered intermediate stops"),
    "P4": ("plan_route", "indexed intermediate stop"),
    "P5": ("plan_route", "urgency: hurry"),
    "P6": ("plan_route", "route preference (scenic / fastest / shortest)"),
    "P7": ("plan_route", "something to avoid"),
    # add a stop while driving
    "I1": ("insert_stop", "insert as the next stop"),
    "I2": ("insert_stop", "insert as the last stop before the destination"),
    "I3": ("insert_stop", "insert without an order -> the passenger is asked"),
    "I4": ("insert_stop", "insert right after a named stop"),
    "I5": ("insert_stop", "insert an indexed stop"),
    "I6": ("insert_stop", "insert right before a named stop"),
    # drop a stop
    "R1": ("remove_stop", "remove an intermediate stop"),
    # reorder the stops
    "O1": ("move_stop", "move a stop to be next"),
    "O2": ("move_stop", "move a stop to be last"),
    "O3": ("move_stop", "move a stop before / after another stop"),
    # change the destination
    "D1": ("new_destination", "replace the destination, drop the stops"),
    "D2": ("new_destination", "go somewhere after the destination (it becomes a stop)"),
    "D3": ("new_destination", "indexed new destination"),
    "D4": ("new_destination", "new destination, in a hurry"),
    # stop navigating
    "C1": ("cancel_route", "cancel the route; the car stops"),
    # several commands in one utterance
    "M1": ("*", "two commands in one utterance"),
    "M2": ("*", "three commands in one utterance"),
    # language phenomena, orthogonal to the command
    "X1": ("*", "indirect reference: a need, not a place ('I need cash' -> atm)"),
    "X2": ("*", "place type not on this map -> must be rejected, route unchanged"),
    "X3": ("*", "refers to the current route ('after the cafe' when the cafe is the destination)"),
    "X4": ("*", "self-correction ('the bakery -- no, the cafe')"),
    "X5": ("*", "disfluent speech (fillers, repetitions)"),
    "X6": ("*", "'the nearest X' -> no index; the planner picks by distance"),
    "X7": ("*", "vague category ('somewhere to eat') -> any matching type on the map"),
}

TIERS = {
    0: "setup: the first turn of a multi-turn episode, there to build the route",
    1: "easy: one function, plain wording, no dependence on the current route",
    2: "medium: several slots in one utterance, or an indirect reference",
    3: "hard: depends on the current route, or rejection / self-correction / disfluency",
}


# Functions whose correct answer depends on the current route, or that ask the
# model to refuse, undo or see through what was literally said.
HARD = {"D2", "I4", "I6", "O3", "M1", "M2", "X2", "X3", "X4", "X5"}
# One command, but which one depends on the current route.
MEDIUM = {"X1", "X6", "X7", "O1", "O2"}


def tier_of(functions: list[str], setup: bool = False) -> int:
    if setup:
        return 0
    if HARD & set(functions):
        return 3
    if MEDIUM & set(functions) or len([f for f in functions if f != "P1"]) > 1:
        return 2
    return 1
