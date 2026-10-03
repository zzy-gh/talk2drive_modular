"""
llm_inference.py
================
Parses natural language passenger commands into structured DrivingIntent JSON
using Google Gemini Enterprise.

Prerequisites:
  1. conda activate zhiyuan_gemini
  2. Environment variables set via activate hook, either
       GEMINI_API_KEY (AI Studio), or
       GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_ENTERPRISE
     plus optionally GEMINI_MODEL to override MODEL below.

Usage:
  parser = GeminiCommandParser(kb)
  intent = parser.parse("Take me to the hospital, hurry!", town="Town01")
"""

import json
import os
from typing import Optional

from google import genai
from google.genai import types

from .intent_schema import DrivingIntent, JSON_SCHEMA_STR, EXAMPLES
from .landmark_kb import LandmarkKnowledgeBase

from .paths import KB_CSV

CSV_PATH = str(KB_CSV)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# GEMINI_MODEL overrides: gemini-2.5-flash is closed to new AI Studio keys.
MODEL        = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# All of them: the "After that, go to X" -> new_destination examples sit at
# the end, and cutting them off taught the model to answer insert_stop.
NUM_EXAMPLES = len(EXAMPLES)


# ─────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────

def _route_section(route: Optional[dict]) -> str:
    """What the car is doing right now, so edits resolve against it."""
    if route is None:                       # caller does not track a route
        return ""
    dest = route.get("destination")
    if not dest:
        return """
Current route: none, the car is idle.
- Any request to go somewhere is a plan_route (even "then go to X").
"""

    def name(t, i):
        return f"{t} (#{i})" if i else t

    lines = [f"  {n}. stop: {name(s['type'], s.get('index'))}"
             for n, s in enumerate(route.get("stops") or [], start=1)]
    lines.append(f"  -> destination: {name(dest, route.get('destination_index'))}")
    stop_types = sorted({s["type"] for s in route.get("stops") or []})
    return f"""
Current route, in driving order:
{chr(10).join(lines)}

Resolve the instruction against this route:
- Going somewhere AFTER the destination ("after that", "then", "afterwards",
  "after the {dest}") -> new_destination with keep_waypoints=true.
- "after the X" where X is one of the stops ({", ".join(stop_types) or "none"})
  -> insert_stop with after=X.
- remove_stop only names a type that is one of the stops above.
- A brand-new trip that ignores this route ("forget all that", "instead")
  -> new_destination with keep_waypoints=false, or plan_route if it lists stops.
"""


def _build_system_prompt(town: str, landmark_types: list[str],
                         route: Optional[dict] = None) -> str:
    examples_str = ""
    for ex in EXAMPLES[:NUM_EXAMPLES]:
        examples_str += f'\nInput:  "{ex["command"]}"\n'
        examples_str += f'Output: {json.dumps(ex["intent"])}\n'

    return f"""You are an in-vehicle command parser for an autonomous driving simulation in CARLA.
Convert the passenger's spoken instruction into a structured JSON object.

Current map: {town}
Known building types available in this map: {", ".join(landmark_types)}
{_route_section(route)}
Rules:
- building_type must be exactly one of the known types listed above, or null if unknown.
- Do not invent or guess building types not in the list.
- If the instruction is ambiguous, pick the most likely type and set confidence below 0.8.
- Do not output any coordinates — those are looked up separately.
- If the passenger specifies which instance of a building (e.g. "the first museum", "the second hotel", "museum number 2"), set the corresponding index field to that 1-based integer. Otherwise set it to null.

{JSON_SCHEMA_STR}

Examples:
{examples_str}
Output ONLY the JSON object, no markdown, no explanation."""


# ─────────────────────────────────────────────
# Parser class
# ─────────────────────────────────────────────

class GeminiCommandParser:
    def __init__(self, kb: LandmarkKnowledgeBase, model: str = MODEL):
        self.kb = kb
        self.model = model
        # Reads GEMINI_API_KEY (AI Studio) or GOOGLE_CLOUD_PROJECT,
        # GOOGLE_CLOUD_LOCATION and GOOGLE_GENAI_USE_ENTERPRISE (Enterprise)
        # from environment automatically
        self.client = genai.Client()

    def parse(self, text: str, town: str,
              route: Optional[dict] = None) -> Optional[DrivingIntent]:
        """``route`` is ``MissionPlanner.route_summary()``: with it the model
        knows what "after that" or "the cafe" refers to. Leave it None for a
        stand-alone utterance -- the prompt is then exactly as before."""
        landmark_types = self.kb.get_landmark_types(town)
        system_prompt  = _build_system_prompt(town, landmark_types, route)

        response = self.client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        raw = response.text.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[llm_inference] Failed to parse JSON:\n{raw[:300]}")
            return None

        # The model occasionally wraps the object in a list, or returns a bare
        # array of candidate intents. Unwrap a single-element list; refuse
        # anything else rather than crashing on `.get` -- an unusable answer is
        # a parse failure, not an exception for the caller to handle.
        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], dict):
                result = result[0]
            else:
                print(f"[llm_inference] Expected one JSON object, got a list of "
                      f"{len(result)}")
                return None
        if not isinstance(result, dict):
            print(f"[llm_inference] Expected a JSON object, got {type(result).__name__}")
            return None

        # Secondary validation: reject building_type not in known list
        bt = result.get("destination") or (result.get("insert") or {}).get("type")
        if bt and bt not in landmark_types:
            print(f"[llm_inference] Rejected unknown building_type: '{bt}'")
            return None

        return result


# ─────────────────────────────────────────────
# Town name normalizer
# ─────────────────────────────────────────────

def normalize_town(raw: str) -> str:
    """Accept '1', '10', 'town01', 'Town10', 'town10hd', 'Town10HD_Opt' etc.
    Returns canonical name like 'Town01', 'Town10HD'.
    """
    import re
    s = re.sub(r'_Opt$', '', raw.strip(), flags=re.IGNORECASE)
    s = s.lower().removeprefix("town")
    if s.endswith("hd"):
        s = s[:-2]
    if not s.isdigit():
        return raw.strip()
    n = int(s)
    return "Town10HD" if n == 10 else f"Town{n:02d}"


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    kb     = LandmarkKnowledgeBase(CSV_PATH)
    parser = GeminiCommandParser(kb)

    raw_town = input("Current town (e.g. 1, 10, Town01 — Enter for Town01): ").strip()
    town     = normalize_town(raw_town) if raw_town else "Town01"
    print(f"Selected town: {town}. Enter a command; empty line or 'quit' to exit.\n")

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cmd or cmd.lower() in {"quit", "exit"}:
            break

        intent = parser.parse(cmd, town=town)
        if not intent:
            print("-> Could not parse command.\n")
            continue

        print(json.dumps(intent, ensure_ascii=False, indent=2))

        # Look up coordinates for destination
        bt = intent.get("destination") or (intent.get("insert") or {}).get("type")
        if bt:
            coord = kb.find_coordinate(town, bt)
            if coord:
                print(f"-> coord: ({coord['x']}, {coord['y']}, {coord['z']})  [{coord['carla_name']}]\n")
            else:
                print(f"-> coord: not found for '{bt}' in {town}\n")
