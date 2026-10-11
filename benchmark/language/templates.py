"""
benchmark/language/templates.py
===============================
English surface forms for every functional intent. A template only ever
renders an intent it states unambiguously -- the gold label comes from the
structure the generator picked, never from reading the sentence back.

Placeholders: {dest} {stop} {stop2} {new} {ordinal} {index} {after}
{target} {need} {missing} {wrong} {hurry} {pref} {avoid}; a capitalised one
({Hurry}, {Need}) is the same value with its first letter upper-cased.
Anything that reads as "the <ordinal> <place>" must only be used with an
index; everything else must never imply one.
"""

from __future__ import annotations

import random

# How a passenger says each knowledge-base type.
SURFACE: dict[str, list[str]] = {
    "apartment": ["apartment", "apartment building"],
    "atm": ["ATM", "cash machine"],
    "bakery": ["bakery"],
    "beach": ["beach"],
    "bus_stop": ["bus stop"],
    "cafe": ["cafe", "coffee shop"],
    "church": ["church"],
    "cinema": ["cinema", "movie theater"],
    "factory": ["factory"],
    "farm": ["farm"],
    "fire_station": ["fire station"],
    "gas_station": ["gas station", "petrol station"],
    "gym": ["gym"],
    "hotel": ["hotel"],
    "house": ["house"],
    "lake": ["lake"],
    "library": ["library"],
    "museum": ["museum"],
    "park": ["park"],
    "parking": ["parking lot", "car park"],
    "pharmacy": ["pharmacy", "drugstore"],
    "playground": ["playground"],
    "post_office": ["post office"],
    "research_institute": ["research institute"],
    "restaurant": ["restaurant"],
    "river": ["river"],
    "shopping_mall": ["shopping mall", "mall"],
    "sports_facility": ["sports center", "sports facility"],
    "store": ["store", "shop"],
    "supermarket": ["supermarket", "grocery store"],
    "theater": ["theater"],
    "train_station": ["train station"],
    "university": ["university", "campus"],
    "warehouse": ["warehouse"],
    "woods": ["woods", "forest"],
}

# A need that names exactly one type. Left out on purpose: anything that could
# be two types (water -> lake/river/beach, food -> restaurant/bakery/cafe).
NEED: dict[str, list[str]] = {
    "atm": ["I need to withdraw some cash", "I have to get money out"],
    "bakery": ["I want to pick up a loaf of fresh bread"],
    "cafe": ["I could really use a cup of coffee"],
    "gas_station": ["we're running low on fuel", "the tank is almost empty"],
    "pharmacy": ["I need to pick up my prescription"],
    "supermarket": ["I have to do the weekly grocery shopping"],
    "hotel": ["I need a room for the night"],
    "library": ["I want to borrow a few books"],
    "parking": ["I need somewhere to leave the car"],
    "cinema": ["I want to catch a movie"],
    "train_station": ["I have a train to catch"],
    "post_office": ["I need to mail a package"],
    "gym": ["it's time for my workout"],
    "bus_stop": ["drop me where I can catch a bus"],
    "fire_station": ["I need to report a fire hazard in person"],
}

ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
           6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth"}

PREFERENCE = {"scenic": ["take the scenic route", "let's go the pretty way"],
              "fastest": ["take the fastest route", "whatever's quickest"],
              "shortest": ["take the shortest route"]}

AVOID = {"highway": ["avoid the highway", "stay off the highway"],
         "tolls": ["avoid toll roads"],
         "traffic": ["avoid heavy traffic"]}

HURRY = ["hurry", "I'm in a rush", "as fast as you can", "quickly please"]

# command / function -> sentence patterns
T: dict[str, list[str]] = {
    "P1": ["Take me to the {dest}.", "Drive to the {dest}, please.",
           "Can you take me to the {dest}?", "I'd like to go to the {dest}."],
    "P2": ["Take me to the {ordinal} {dest}.", "Drive to {dest} number {index}.",
           "Go to the {ordinal} {dest}, please."],
    "P3": ["Stop at the {stop} first, then take me to the {dest}.",
           "Go to the {dest}, but stop by the {stop} on the way.",
           "First the {stop}, then the {dest}."],
    "P3x2": ["Go to the {stop}, then the {stop2}, and finally the {dest}.",
             "Stop at the {stop} and then the {stop2} before we get to the {dest}."],
    "P4": ["Stop at the {ordinal} {stop}, then take me to the {dest}."],
    "P5": ["Take me to the {dest}, {hurry}.", "{Hurry}! Get me to the {dest}."],
    "P6": ["Drive to the {dest} and {pref}.", "To the {dest}, {pref}."],
    "P7": ["Take me to the {dest}, but {avoid}.", "Go to the {dest} and {avoid}."],
    "I1": ["Stop at the {new} first.", "Pull over at the {new} right now.",
           "Before anything else, let's go to the {new}."],
    "I2": ["Make the {new} the last stop before we arrive.",
           "Add the {new} as the final stop before the destination."],
    "I3": ["I also want to go to the {new}.", "Add a stop at the {new}.",
           "Could we stop at the {new} too?"],
    "I4": ["After the {after}, stop at the {new}.",
           "Once we're done at the {after}, stop by the {new} before we carry on."],
    "I5": ["I also want to go to the {ordinal} {new}.",
           "Add the {ordinal} {new} as a stop."],
    "I6": ["Stop at the {new} before the {other}.",
           "Add the {new} right before the {other}."],
    "O1": ["Let's go to the {stop} first.", "Make the {stop} our next stop."],
    "O2": ["Leave the {stop} for last.",
           "Let's do the {stop} last, right before we arrive."],
    "O3after": ["Go to the {stop} after the {other}.",
                "Visit the {stop} once we're done at the {other}."],
    "O3before": ["Stop at the {stop} before the {other}.",
                 "Let's do the {stop} before the {other}."],
    "M1remove_next": ["Skip the {target} and stop at the {new} first.",
                      "Forget the {target}, let's go to the {new} first instead."],
    "M1next_last": ["Stop at the {new} first, and add the {new2} as the last stop."],
    "M1remove_extend": ["Drop the {target}, and after the {dest} take me to the {new}."],
    "M1move_last": ["Do the {stop} first, and add the {new} as the last stop."],
    "M2": ["Skip the {target}, stop at the {new} first, and afterwards take me to the {new2}."],
    "R1": ["Never mind the {target}.", "Skip the {target}.",
           "We don't need to stop at the {target} anymore."],
    "D1": ["Actually, forget all that and take me to the {new} instead.",
           "Change of plans, just go to the {new}."],
    "D2": ["After that, take me to the {new}.", "Then let's go to the {new}.",
           "Afterwards I want to go to the {new}."],
    "D3": ["Change of plans, take me to the {ordinal} {new} instead."],
    "D4": ["Change of plans, {hurry}, take me to the {new}.",
           "Forget that, I need to get to the {new}, {hurry}."],
    "C1": ["Stop here, I'll get out.", "Pull over, I want to get out here.",
           "Cancel the route, we're stopping here."],
    # phenomena
    "X1": ["{Need}, can you take me somewhere for that?", "{Need}."],
    "X1ins": ["Also, {need}.", "Oh, and {need} on the way."],
    "X2": ["Take me to the {missing}.", "Drive to the {missing}, please."],
    "X2ins": ["Can we also stop at the {missing}?"],
    "X3": ["After the {after}, take me to the {new}.",
           "When we get to the {after}, carry on to the {new}."],
    "X4": ["Take me to the {wrong}, no wait, the {dest}.",
           "Go to the {wrong}... sorry, I mean the {dest}."],
    "X6": ["Take me to the nearest {dest}.", "Go to the closest {dest}, please."],
    "X6ins": ["Add a stop at the closest {new}.", "Can we stop at the nearest {new} too?"],
    "X7": ["{Vague}.", "{Vague}, can you take me there?"],
    "X7ins": ["Also, {vague} on the way."],
}

# A need that several types on a map satisfy equally: any of them is right.
# Only used in a town where at least two of its types exist.
VAGUE: dict[str, tuple[set, list[str]]] = {
    "eat": ({"restaurant", "cafe", "bakery"},
            ["find me somewhere to eat", "I'm hungry, I need to get some food"]),
    "shop": ({"store", "supermarket", "shopping_mall"},
             ["I need to do some shopping", "I have to buy a few things"]),
    "outdoors": ({"park", "woods", "lake", "beach", "river", "playground"},
                 ["I'd like to get some fresh air outside", "somewhere outdoors to stretch my legs"]),
    "sleep": ({"hotel", "apartment", "house"}, ["I need somewhere to sleep tonight"]),
    "transit": ({"bus_stop", "train_station"}, ["drop me where I can catch public transport"]),
    "show": ({"cinema", "theater", "museum"}, ["I want to see a show or an exhibition"]),
}

# A missing type is only a fair "not on this map" test if nothing on the map
# could reasonably be meant instead.
SIMILAR = [{"store", "supermarket", "shopping_mall"},
           {"house", "apartment", "hotel"},
           {"lake", "river", "beach"},
           {"park", "woods", "playground", "farm"},
           {"cafe", "bakery", "restaurant"},
           {"parking", "gas_station"},
           {"cinema", "theater"},
           {"factory", "warehouse"},
           {"university", "research_institute", "library"}]

FILLERS = ["uh", "um", "like", "you know"]


def surface(t: str, rng: random.Random) -> str:
    return rng.choice(SURFACE.get(t, [t.replace("_", " ")]))


def disfluent(text: str, rng: random.Random) -> str:
    """Speech-like noise that leaves the meaning intact: fillers and one
    repeated word, never a dropped or substituted content word."""
    def lower(w: str) -> str:
        return w if w == "I" or w.startswith("I'") else w[:1].lower() + w[1:]

    words = text.split()
    if len(words) < 4:
        return text
    # Only before "the" or after a comma: never inside a place name
    # ("grocery store" must not become "grocery you know, store").
    gaps = [i for i in range(1, len(words))
            if words[i].lower() == "the" or words[i - 1].endswith(",")]
    if not gaps:
        return text
    words.insert(rng.choice(gaps), rng.choice(FILLERS) + ",")
    j = rng.randrange(0, 2)
    words.insert(j + 1, lower(words[j]))            # "take take me to ..."
    prefix = rng.choice(["Uh, ", "Um, ", ""])
    if prefix:
        words[0] = lower(words[0])
    return prefix + " ".join(words)
