"""
covlm/utterance_overrides.py
============================
Replacement phrasings for template lines the LLM does not reliably turn into a
``plan_route`` with the intended destination.

CoLMDriver's ``scripts/scenario_gen/intent_templates.py`` is their file and
stays untouched; it was written to convey *urgency* for negotiation, and a line
can do that perfectly while still being ambiguous about where the car should go.

The one that forced this module into existence:

    "My parking meter runs out shortly and I'd rather not get ticketed."

which Gemini read three different ways in three towns -- ``plan_route`` to
parking in Town07, ``new_destination`` to parking in Town06, and in Town05
``cancel_route {pull_over: true}``, with no destination at all. All three are
defensible: a meter running out is about the car you already parked, so "pull
over" is a fair reading. It is simply not a sentence that states a destination.

A replacement must keep two things and change one:

    keep   the urgency tier -- CoLMDriver scores negotiation on it, and the
           whole sweep in scripts/eval/run_intent_sweep.sh compares runs with
           and without these utterances.
    keep   the situation, so the line still reads like a passenger talking.
    change the destination into something stated outright, in the vocabulary of
           ``core/intent_schema.py``'s BuildingType.

Keyed by ``(label, tier)``. ``covlm/design_passenger_inputs.py`` reaches for
one only after the template itself has been measured to fail, so an entry here
is always backed by an observed failure rather than a hunch.
"""

OVERRIDES: dict[tuple[str, str], str] = {
    # "meter runs out" reads as pull-over, not as a destination.
    ("parking", "med"):
        "Take me to the parking garage — my meter's nearly up and I'd rather not "
        "get ticketed.",
    ("parking", "low"):
        "Just heading over to the parking lot, no rush at all.",
    ("parking", "high"):
        "Get me to that parking lot now — they're towing my car and my documents "
        "are still inside.",
}


def override_for(label: str, tier: str) -> str | None:
    return OVERRIDES.get((label, tier))
