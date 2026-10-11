# Task: rewrite passenger utterances for the Talk2Drive language benchmark

## What this is

A passenger talks to a self-driving car. Each of the 632 turns in `turns.jsonl`
has a **template sentence** written by a program, and a **gold intent**: the
structured navigation command the car must derive from that sentence.

The template sentences are formulaic by construction. Your job is to rewrite
each one the way a real person would say it out loud, **without moving the
meaning**, so that the benchmark measures language understanding instead of
template pattern-matching.

A rewrite that changes the meaning is worse than no rewrite at all: the gold
intent stays fixed, so a drifted sentence makes a correct model look wrong.

## Input

`turns.jsonl`, one JSON object per turn:

| field | |
|---|---|
| `row` | index into `review.csv` — **return it unchanged** |
| `id`, `turn` | which conversation, which turn in it |
| `town`, `tier`, `functions` | metadata; `tier` 1–3 is difficulty |
| `template` | **the sentence to rewrite** |
| `route_before` | the route already planned when the passenger speaks |
| `gold` | the command the sentence must still produce, in words |
| `style` | the register to write in (see below) |
| `must_name` | every place that must still be named (a natural synonym is fine) |
| `must_not_name` | places the sentence must **not** name (indirect requests) |
| `ordinal` | an index word ("second") that must survive, or null |
| `ordered` | true: the two places must stay in the order given in `must_name` |
| `keep_nearest` | true: "nearest / closest" must survive |
| `keep_disfluent` | true: keep the hesitations and repeated words |
| `extra` | any per-turn instruction, already assembled |

`route_before` matters: several turns only make sense against it. "Skip the
bakery" is a removal **because** the bakery is on the route.

## Output

One JSON object per line, two fields:

```json
{"row": 63, "rewrite": "Do the coffee shop first, and make the ATM our last stop."}
```

Nothing else on the line. Write it to a file and run the checker:

```bash
python3 benchmark/language/rewrite_task/check_rewrites.py check <your_file.jsonl>
```

The checker prints every rewrite it rejects and why. **Iterate until it is
clean**, then hand the file back. It is a lexical check, not a judge: passing
it is necessary, not sufficient.

## Styles

Write each turn in the register its `style` field names:

| style | |
|---|---|
| `casual and relaxed` | everyday speech |
| `polite and a little formal` | courteous — **but still a passenger speaking, not a memo** |
| `short and terse` | clipped — **but nothing dropped** |
| `chatty, with a bit of extra context about why` | may add a reason, never a place or condition |
| `slightly impatient` | irritated tone, same request |

## Hard rules

1. **Keep the meaning exactly**: same places, same order, same index words
   ("first", "second"), same urgency, same route preferences.
2. **Name every place in `must_name`.** A natural synonym is fine (`car park`
   for `parking`, `grocery store` for `supermarket`, `cash machine` for `atm`).
3. **Never name anything in `must_not_name`.** Those turns describe a *need*
   ("I need cash", "somewhere to eat") and the car has to work out the place.
4. **Add nothing**: no places, stops, times, conditions, or ordering words
   (first, then, after, before, also, too, again) the original does not have.
5. **A destination stays a destination.** Never turn it into a quick stop
   ("swing by", "stop by", "pop into").
6. **Output only the sentence.**

## The five mistakes that actually happened

A previous model's rewrites were reviewed turn by turn. 31 of 632 had moved the
meaning. Every one fell into these five patterns — they are the rules that
matter most, because they are the ones a fluent-sounding rewrite breaks.

### A. A stop is not a destination — 13 of 31

The schema has separate commands for *adding a stop* and *changing the
destination*. English blurs them; the benchmark does not.

```
gold : insert_stop atm (last)   ← the ATM becomes the last STOP; the destination is unchanged
bad  : include the ATM as the final destination        ← now it reads as a new destination
good : make the ATM our last stop before we get there
```

**Never write `final destination`, `last destination` or `next destination`
for a stop.** Say `last stop`, `final stop`, `next stop`. This also applies to
moving an existing stop to the front or the back of the queue.

### B. "then / afterwards" carries a whole command — 8 of 31

When the gold says *the old destination becomes a stop and we carry on
somewhere new*, that relation lives entirely in one word.

```
route: restaurant -> lake        gold: new_destination supermarket, keep the stops
bad  : Let us proceed to the grocery store.       ← reads as: replace the route
good : Then let's carry on to the grocery store.
```

**Keep a word that says "after what we are already doing"**: then, afterwards,
after that, once we've..., carry on, keep going, ...second. Dropping it as a
filler word silently turns one command into a different one.

In a multi-command turn, keep the verb too: `and afterwards take me to the
park` may become `then take me to the park`, but **not** `then the park` —
without a verb it reads as one more stop.

### C. Urgency is a scored field — 2 of 31

Add a hurry only where the original has one; keep it wherever it does. Note
that a *route preference* is not urgency: `take the fastest route` is a
preference, `hurry` is urgency. A turn can have one, both, or neither, and
`ASAP` is urgency, not a preference.

### D. A removal must still read as a removal — part of the 8 below

```
gold : remove_stop parking ; then new_destination store (keep the stops)
bad  : Car park, then gas station, then shop.    ← the removal vanished; now it is an itinerary
bad  : Drop me at the research institute...      ← "drop" the stop became "drop me at" it
good : Forget the car park, and after the gas station take me to the shop.
```

Keep an explicit cue: skip, forget, drop, never mind, no longer, don't need,
cancel, leave out.

### E. Do not compress a request out of existence — 8 of 31

The `short and terse` style produced these. All are real:

```
I'd like to get some fresh air outside.   ->  Open the windows.      (not a route request at all)
I need somewhere to leave the car. I'm in a rush!  ->  Park. Hurry!  ("park" is also a landmark here)
Also, I need a room for the night.        ->  Book a room for the night.   (no travel request left)
Can you take me to the shop? As fast as you can! And take the shortest route.
                                          ->  Shop. Fastest. Shortest route.  (two conflicting preferences)
After the university, stop at the research institute.
                                          ->  University first, then the research institute.
                                                  (claims a position the route does not have)
```

Terse is fine. **Terse to the point of ambiguity is not.** If the clipped
version could be read as a different command, keep more words.

## What the checker verifies

`check_rewrites.py` runs the benchmark's own structural check plus the four
patterns above:

| | |
|---|---|
| `faithful()` | the project's own check: `must_name`, `must_not_name`, `ordinal`, `ordered`, nearest |
| `stop-vs-destination` | no "final / last / next destination" where the gold moves or inserts a stop |
| `after-the-destination` | a D2 turn keeps a "then / afterwards" word |
| `urgency` | a hurry appears exactly where the gold has one |
| `removal` | a removal keeps a removal cue, and is not "drop me at" |

It cannot tell whether a sentence sounds natural, or whether a terse rewrite is
ambiguous to a human. Those stay a judgement call — reread rule E before
handing the file back.
