# Talk2Drive benchmark

Two questions, measured separately:

* **(a) language level** — does an LLM turn what the passenger says into the
  right structured navigation intent? Offline, no simulator.
* **(b) execution level** — does a driving backend carry out a navigation
  intent that changes while it drives? Closed loop in CARLA.

## (a) Language level — `benchmark/language/`

### Pipeline

```bash
conda activate zhiyuan_gemini
python3 benchmark/language/generate.py        # templates   -> data/episodes_template.jsonl
python3 benchmark/language/paraphrase.py      # LLM rewrite -> data/episodes_paraphrased.jsonl
python3 benchmark/language/review.py export   # human check -> data/review.csv
python3 benchmark/language/review.py import   #             -> data/episodes_verified.jsonl

python3 benchmark/language/run.py --model gemini-3.5-flash-lite --data <episodes.jsonl>
python3 benchmark/language/evaluate.py benchmark/runs/<run>.jsonl [<run> ...] --data <episodes.jsonl>
```

`run.py` resumes (already answered turns are skipped), throttles to `--rpm`
requests per minute across `--workers` threads, and waits out a 429 for the
delay the provider asks for. `--no-context` hides the current route from the
model (ablation). Adding a provider is one branch in `models.py`: subclass
`core.llm_inference.CommandParser` and implement `complete()`; the prompt and
the output validation are shared, so every model is measured on the same task.

### Data

An **episode** is one passenger conversation in one town; each **turn** is

| field | |
|---|---|
| `utterance` | what the passenger says (`template_utterance` keeps the template text after paraphrasing) |
| `state_before` | the route when they say it: destination, its index, stops in order, urgency |
| `gold_intent` | `{"commands": [...]}`, applied in order (`null` = must be rejected) |
| `state_after` | `MissionPlanner.apply(gold_intent)` from `state_before` |
| `functions` | which functional intents the turn exercises (below) |
| `tier` | difficulty 1–3; 0 for a setup turn |
| `setup` | the first turn of a multi-turn episode, there to build the route: kept as template text, not paraphrased or human-checked, and scored apart from the headline numbers |
| `accept` | for a vague request (X7), every type that counts as right |
| `meta` | places the gold cannot name: the off-map place, the self-corrected one |

The dataset is built **structure first**: a recipe fixes what each turn does,
places legal in that town and state are drawn, the gold intent is built, and
only then is a sentence rendered. Gold states come from the real
`MissionPlanner.apply`, and generation aborts if any function is uncovered.
Every town with a knowledge base is used: Town01–07 and Town10HD.

### Functional intents (`taxonomy.py`)

| id | | id | |
|---|---|---|---|
| P1 | destination | I1 | insert as next stop |
| P2 | indexed destination ("the second cafe") | I2 | insert as last stop |
| P3 | ordered stops | I3 | insert, order unstated → passenger is asked |
| P4 | indexed stop | I4 | insert after a named stop |
| P5 | hurry | I5 | insert an indexed stop |
| P6 | route preference | I6 | insert before a named stop |
| P7 | something to avoid | R1 | remove a stop |
| O1 | move a stop to be next | D1 | replace the destination |
| O2 | move a stop to be last | D2 | go somewhere after the destination |
| O3 | move a stop before / after another | D3 | indexed new destination |
| M1 | two commands in one utterance | D4 | new destination, in a hurry |
| M2 | three commands in one utterance | C1 | cancel the route; the car stops |
| X1 | a need, not a place ("I need cash") | X2 | place not on this map → reject |
| X3 | refers to the current route | X4 | self-correction |
| X5 | disfluent speech | X6 | "the nearest X" → no index |
| X7 | vague category ("somewhere to eat") → any matching type | | |

Tiers: **1** one function in plain words; **2** several slots, an indirect or
vague reference, "nearest", or a move to next / last; **3** depends on the
current route in a way that changes the answer (after / before a stop, after
the destination, a move relative to another stop), several commands at once,
or rejection / self-correction / disfluency. The template set has 984 turns:
344 setup, then 264 / 192 / 184 scored in tiers 1 / 2 / 3, every function
covered by at least 16.

Scope: only what navigation over landmark types and their coordinates can
do. Nothing that needs facts the simulator lacks (opening hours, "if it is
closed"), and no speed control outside a route request.

### Paraphrase and verification

`paraphrase.py` rewrites each template sentence in one of five styles. A
rewrite is accepted only if it still names every place involved, keeps an
ordinal wherever the gold has an index, does not name the place in an
indirect turn, and keeps both the off-map place and both halves of a
self-correction — checked against the turn's structure, never by asking a
model. Three failed rewrites keep the template sentence and flag the turn.
`review.py` then puts every turn in front of a person; `import` refuses to
write the verified set while any row is undecided.

### Metrics (`evaluate.py`)

| metric | |
|---|---|
| **state accuracy** | headline: the route after `MissionPlanner.apply(prediction)` equals the gold route — destination, its index, stops in order, which stops are left for the passenger to place. Different intents with the same effect score the same. Urgency is not part of the route; it is scored as a field. Setup turns are reported separately (`setup_acc`). |
| episode accuracy | every turn of the conversation right |
| command accuracy | command type right (`reject` for rejected / unparsable) |
| field accuracy | destination, its index, stops, insert type / position (with after / before) / index, move type / position, removed stop, keep_waypoints, urgency (only where the passenger says it), preference, avoid, number of commands — each on the turns whose gold sets it, command by command. plan_route and a new_destination that drops the stops are compared as one kind. |
| ask recall / false-ask rate | an unordered stop is left for the passenger to place exactly when it should be |
| latency p50 / p95, tokens | cost |

Each turn is parsed from its **gold** starting state, so one wrong turn does
not cascade; episode accuracy is computed from the per-turn results.

`avoid` and `route_preference` are parsed and stored but the planner does not
route on them, so they are language-level only.

## (b) Execution level — `benchmark/execution/`

### Pipeline

```bash
python3 benchmark/execution/generate.py                    # offline -> data/scenarios.jsonl
python3 benchmark/execution/run.py --backend basic_agent   # CARLA running -> benchmark/runs/execution/basic_agent/
python3 benchmark/execution/run.py --backend simlingo      # (zhiyuan_gemini)
python3 benchmark/execution/run.py --backend leaderboard   # (zhiyuan_tcp)
python3 benchmark/execution/evaluate.py benchmark/runs/execution/*/
```

`run.py` takes every backend option the CLI does, skips scenarios already
logged (`--rerun` to redo them) and `--only Town03/` runs a subset. `--watch` follows the car with the spectator
and draws the route (as text, never in a camera image) at real-time pace, to
look at a scenario; leave it off for benchmark runs.

### Scenarios

A scenario is a start pose, a script of **gold** intents with triggers
(`start`, after the car has `driven_m`, or once it has `reached` a place), and
an expected outcome. Driving from gold intents keeps language errors out of
this level. Every place carries its index, so nothing is ever disambiguated,
and an unordered stop's position is scripted (`answer`).

The expected outcome is computed, not written: `generate.py` drives each
planned route offline point by point, fires the events where the runner will,
re-plans exactly as the session does, and records the places it passes. A
draw is kept only if it is well posed — first route 400–900 m, at most 2 km in
all, every plan passes its stops in order, every trigger fires, a dropped
place stays more than 15 m from the route, a "behind" destination is behind.

Thirteen templates cover the execution-level functions (P1 P3 P5, I1–I4 I6,
R1, O1–O3, D1 D2 D4, C1, M1 M2; indexed variants are implicit in every
scenario): plan with two stops and its hurry twin, insert next/last,
insert unordered/before, insert after + remove, move next/last, move
relative, replace the destination with one behind the car, extend, rush
elsewhere, cancel, two and three commands at once. Five scenarios per town.

### Running

The runner steps CARLA synchronously and as fast as the backend allows, and
fires each event from a per-step observer on the control thread, so the world
is held still while an intent is applied and re-planned: a scenario plays out
the same every time. Logged per step: pose, speed, control, and whether the
car faces against its lane; plus collision and lane-invasion events. A run
ends on arrival, on standing still after a cancel, at the time limit, or
after a minute without moving.

A stop is **visited** when the car passes within 5 m of it — no dwell.

### Metrics (`evaluate.py`)

| metric | |
|---|---|
| **success** | headline: every expected stop visited in order (each at its first approach after the previous one), arrived at the expected destination — or standing still after a cancel — no dropped place visited, within the time limit |
| completion | share of the expected stops + destination reached in order |
| detour | distance driven / distance of the ideal drive |
| collisions, lane invasions / km, wrong-way seconds | driving quality (sensor events de-duplicated over 1 s / 0.5 s) |
| hurry speed ratio | mean moving speed with / without hurry on the same route (paired scenarios) |
| rush speed ratio | mean speed after / before a hurried change of destination |
| cancel stop time | seconds from the cancel to standing still |

Hurry is realised only by `basic_agent` (target speed) and SimLingo (a prompt
sentence); TCP has no input for it.

