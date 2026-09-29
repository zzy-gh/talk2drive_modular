"""
Driving Intent Schema
=====================
Defines the structured output format for voice_parser.py.
Covers both static route planning (before departure) and
dynamic route updates (while driving).
"""

from typing import Literal, Optional
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# Primitive types
# ─────────────────────────────────────────────

BuildingType = Literal[
    # Residential
    "house", "apartment", "hotel",
    # Commercial
    "shopping_mall", "supermarket", "store", "post_office",
    # Food & Drink
    "restaurant", "cafe", "bar", "bakery",
    # Education
    "school", "university", "library", "research_institute",
    # Medical
    "hospital", "pharmacy",
    # Government
    "police", "fire_station", "government_office",
    # Transportation
    "parking", "gas_station", "charging_station",
    "bus_stop", "train_station", "airport",
    # Recreation
    "park", "church", "museum", "gym", "cinema", "theater",
    "sports_facility",
    # Nature
    "river", "lake", "woods", "beach", "farm",
    # Industrial
    "factory", "warehouse",
]

CommandType     = Literal["plan_route", "insert_stop", "remove_stop", "new_destination", "cancel_route"]
Urgency         = Literal["high", "normal"]
InsertPosition  = Literal["next", "enroute", "last"]
RemoveFallback  = Literal["next_waypoint", "destination"]
RoutePreference = Literal["scenic", "fastest", "shortest"]


# ─────────────────────────────────────────────
# Sub-structures
# ─────────────────────────────────────────────

class Stop(TypedDict):
    type:       BuildingType
    order:      int
    confidence: float
    index:      Optional[int]   # 1-based; set if passenger specifies "the first X" etc.


class InsertPayload(TypedDict):
    type:       BuildingType
    position:   InsertPosition
    confidence: float
    index:      Optional[int]          # 1-based building index if passenger specified one
    after:      Optional[BuildingType] # insert right after this existing stop type; null if not specified


class RemovePayload(TypedDict):
    type:     BuildingType
    fallback: RemoveFallback    # where to go after removal


# ─────────────────────────────────────────────
# Command schemas
# ─────────────────────────────────────────────

class PlanRouteIntent(TypedDict):
    command_type:       Literal["plan_route"]
    destination:        BuildingType
    destination_index:  Optional[int]         # 1-based; set if passenger specifies "the first X"
    waypoints:          list[Stop]
    urgency:            Urgency
    avoid:              list[str]
    route_preference:   Optional[RoutePreference]
    confidence:         float


class InsertStopIntent(TypedDict):
    command_type: Literal["insert_stop"]
    insert:       InsertPayload


class RemoveStopIntent(TypedDict):
    command_type: Literal["remove_stop"]
    remove:       RemovePayload


class NewDestinationIntent(TypedDict):
    command_type:      Literal["new_destination"]
    destination:       BuildingType
    destination_index: Optional[int]   # 1-based if passenger specifies which one
    keep_waypoints:    bool
    urgency:           Urgency


class CancelRouteIntent(TypedDict):
    command_type: Literal["cancel_route"]
    pull_over:    bool        # whether to pull over immediately


# Return type of voice_parser.py
DrivingIntent = (
    PlanRouteIntent
    | InsertStopIntent
    | RemoveStopIntent
    | NewDestinationIntent
    | CancelRouteIntent
)


# ─────────────────────────────────────────────
# Few-shot examples (inject into LLM prompt)
# ─────────────────────────────────────────────

EXAMPLES: list[dict] = [
    {
        "command": "Stop at the pharmacy first, then take me to the hospital.",
        "intent": {
            "command_type":     "plan_route",
            "destination":      "hospital",
            "waypoints":        [{"type": "pharmacy", "order": 1, "confidence": 0.95}],
            "urgency":          "normal",
            "avoid":            [],
            "route_preference": None,
            "confidence":       0.93,
        },
    },
    {
        "command": "Rush to the airport, terminal 2!",
        "intent": {
            "command_type":     "plan_route",
            "destination":      "airport",
            "waypoints":        [],
            "urgency":          "high",
            "avoid":            [],
            "route_preference": "fastest",
            "confidence":       0.97,
        },
    },
    {
        "command": "Take me to the mall but avoid the highway.",
        "intent": {
            "command_type":     "plan_route",
            "destination":      "shopping_mall",
            "waypoints":        [],
            "urgency":          "normal",
            "avoid":            ["highway"],
            "route_preference": None,
            "confidence":       0.89,
        },
    },
    {
        "command": "Go to the bakery, then the supermarket, then home.",
        "intent": {
            "command_type":     "plan_route",
            "destination":      "house",
            "waypoints":        [
                {"type": "bakery",      "order": 1, "confidence": 0.92},
                {"type": "supermarket", "order": 2, "confidence": 0.94},
            ],
            "urgency":          "normal",
            "avoid":            [],
            "route_preference": None,
            "confidence":       0.90,
        },
    },
    {
        "command": "Pull over at the gas station for a moment.",
        "intent": {
            "command_type": "insert_stop",
            "insert": {"type": "gas_station", "position": "next", "confidence": 0.96},
        },
    },
    {
        "command": "Stop at a cafe on the way.",
        "intent": {
            "command_type": "insert_stop",
            "insert": {"type": "cafe", "position": "enroute", "confidence": 0.88},
        },
    },
    {
        "command": "Stop at the pharmacy before we get to the hospital.",
        "intent": {
            "command_type": "insert_stop",
            "insert": {"type": "pharmacy", "position": "last", "confidence": 0.91},
        },
    },
    {
        "command": "Never mind the bakery, just take me home.",
        "intent": {
            "command_type": "remove_stop",
            "remove": {"type": "bakery", "fallback": "next_waypoint"},
        },
    },
    {
        "command": "Actually, forget the mall — take me to the museum instead.",
        "intent": {
            "command_type":      "new_destination",
            "destination":       "museum",
            "destination_index": None,
            "keep_waypoints":    False,
            "urgency":           "normal",
        },
    },
    {
        "command": "Take me to the second museum.",
        "intent": {
            "command_type":     "plan_route",
            "destination":      "museum",
            "destination_index": 2,
            "waypoints":        [],
            "urgency":          "normal",
            "avoid":            [],
            "route_preference": None,
            "confidence":       0.95,
        },
    },
    {
        "command": "Stop at the first hotel on the way.",
        "intent": {
            "command_type": "insert_stop",
            "insert": {"type": "hotel", "position": "enroute", "confidence": 0.92, "index": 1, "after": None},
        },
    },
    {
        "command": "Also stop at the train station after the hotel.",
        "intent": {
            "command_type": "insert_stop",
            "insert": {"type": "train_station", "position": "enroute", "confidence": 0.95, "index": None, "after": "hotel"},
        },
    },
    {
        "command": "After that, go to the museum.",
        "intent": {
            "command_type":   "new_destination",
            "destination":    "museum",
            "keep_waypoints": True,
            "urgency":        "normal",
        },
    },
    {
        "command": "Then go to the museum.",
        "intent": {
            "command_type":   "new_destination",
            "destination":    "museum",
            "keep_waypoints": True,
            "urgency":        "normal",
        },
    },
    {
        "command": "Stop here, I'll get out.",
        "intent": {
            "command_type": "cancel_route",
            "pull_over":    True,
        },
    },
]


# ─────────────────────────────────────────────
# JSON schema string (inject into LLM system prompt)
# ─────────────────────────────────────────────

JSON_SCHEMA_STR = """
Output ONLY valid JSON. No explanation, no markdown, no preamble.

The JSON must match one of the following five structures based on the passenger's intent:

1. plan_route — full route planned before departure
{
  "command_type":      "plan_route",
  "destination":       "<BuildingType>",
  "destination_index": <int 1-based> | null,
  "waypoints":         [ { "type": "<BuildingType>", "order": <int>, "confidence": <float>, "index": <int> | null } ],
  "urgency":           "high" | "normal",
  "avoid":             [ "<string>", ... ],
  "route_preference":  "scenic" | "fastest" | "shortest" | null,
  "confidence":        <float 0.0–1.0>
}

2. insert_stop — add a stop while already driving
{
  "command_type": "insert_stop",
  "insert": {
    "type":       "<BuildingType>",
    "position":   "next" | "enroute" | "last",
    "confidence": <float 0.0–1.0>,
    "index":      <int 1-based> | null,
    "after":      "<BuildingType>" | null
  }
}
  after: set if the passenger says "after the X" or "following the X" — insert this stop immediately after that existing stop type.
  position:
    "next"    — go there immediately (highest priority)
    "enroute" — stop if it is on the way
    "last"    — stop just before the final destination

3. remove_stop — cancel an intermediate stop while driving
{
  "command_type": "remove_stop",
  "remove": {
    "type":     "<BuildingType>",
    "fallback": "next_waypoint" | "destination"
  }
}

4. new_destination — change the final destination while driving
{
  "command_type":      "new_destination",
  "destination":       "<BuildingType>",
  "destination_index": <int 1-based> | null,
  "keep_waypoints":    true | false,
  "urgency":           "high" | "normal"
}

5. cancel_route — abort all navigation
{
  "command_type": "cancel_route",
  "pull_over":    true | false
}

Valid BuildingType values:
house, apartment, hotel,
shopping_mall, supermarket, store, post_office,
restaurant, cafe, bar, bakery,
school, university, library, research_institute,
hospital, pharmacy,
police, fire_station, government_office,
parking, gas_station, charging_station, bus_stop, train_station, airport,
park, church, museum, gym, cinema, theater, sports_facility,
river, lake, woods, beach, farm,
factory, warehouse
"""
