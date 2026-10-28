# Talk2Drive benchmark

| level | question | where |
|---|---|---|
| (a) language | does the LLM turn the passenger's words into the right navigation intent? | offline |
| (b) execution | does a backend carry out an intent that changes while it drives? | CARLA, gold intents |
| (c) end to end | (b) driven from the passenger's words: utterance → LLM → grounding → route → backend | CARLA |

Results are in the [main README](../README.md#results).

## (a) Language — `benchmark/language/`

```bash
conda activate zhiyuan_gemini
python3 benchmark/language/generate.py        # templates    -> data/episodes_template.jsonl
python3 benchmark/language/paraphrase.py      # LLM rewrite  -> data/episodes_paraphrased.jsonl
                                              # human review -> data/episodes_reviewed.jsonl
python3 benchmark/language/run.py --model gemini-3.8-flash --data benchmark/language/data/episodes_reviewed.jsonl
python3 benchmark/language/evaluate.py benchmark/runs/<run>.jsonl [...] --data benchmark/language/data/episodes_reviewed.jsonl
```

`run.py` resumes, rate-limits (`--rpm`, `--workers`) and retries 429s;
`--no-context` hides the current route (ablation). Gemini reads
`GEMINI_API_KEY` from the environment, as everywhere else in the project (see
the [main README](../README.md#api-key)).

`--provider qwen` goes through any OpenAI-compatible endpoint instead, so point
these at your own and nothing else has to change:

```bash
export QWEN_BASE_URL=http://localhost:11434/v1      # e.g. a local Ollama server
export QWEN_API_KEY=your-key-here                   # any non-empty string if the server ignores it
export QWEN_EXTRA_BODY='{"reasoning_effort": "none"}' # optional, merged into each request body
```

A new provider is a `CommandParser` subclass in `models.py`.

### Data

An episode is one conversation in one town. Each turn has `utterance`,
`state_before` (destination, stops, urgency), `gold_intent` (`null` = must be
rejected), `state_after` (from `MissionPlanner.apply`), `functions`, `tier`,
and `setup` (route-building first turns, scored separately).

Built structure first: a recipe fixes what each turn does, places are drawn
for the town, the gold intent and state are computed, then the sentence is
rendered and paraphrased (`paraphrase.py`, faithfulness checked against the
turn's structure). All turns were reviewed by hand. Towns: Town01–07, Town10HD.

| id | function | id | function |
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
| M1 | two commands at once | D4 | new destination, in a hurry |
| M2 | three commands at once | C1 | cancel; the car stops |
| X1 | a need, not a place ("I need cash") | X2 | place not on this map → reject |
| X3 | refers to the current route | X4 | self-correction |
| X5 | disfluent speech | X6 | "the nearest X" |
| X7 | vague category ("somewhere to eat") | | |

Tiers: **T1** one function in plain words; **T2** several slots, indirect or
vague reference, "nearest", move to next / last; **T3** depends on the current
route, several commands, or rejection / self-correction / disfluency. 632
scored turns (256 / 192 / 184).

### Metrics

| metric | definition |
|---|---|
| **state accuracy** | route after applying the predicted intent equals the gold route (destination, index, stops in order, stops left to ask) |
| episode / command accuracy | every turn right / command type right |
| field accuracy | per field, on turns whose gold sets it |
| ask recall / false-ask rate | over turns that should ask / turns that should not |
| latency p50 / p95, tokens | cost |

Each turn is parsed from its gold starting state, so errors do not cascade.
`avoid` and `route_preference` are scored but not used by the planner.

## (b) Execution — `benchmark/execution/`

```bash
python3 benchmark/execution/generate.py                    # -> data/scenarios.jsonl
python3 benchmark/execution/npc.py                         # -> data/npc_layout.jsonl
python3 benchmark/execution/run.py --backend basic_agent [--npc]
python3 benchmark/execution/run.py --backend simlingo --simlingo-no-cot [--npc]
python3 benchmark/execution/run.py --backend leaderboard [--npc]            # zhiyuan_tcp
python3 benchmark/execution/evaluate.py benchmark/runs/execution/<dir>
```

`run.py` takes the CLI's backend options, skips logged scenarios (`--rerun`),
filters with `--only Town03/`, and `--watch` shows the drive.

**Scenarios.** 39 (13 templates, up to 5 per town). Each is a start pose and a
script of gold intents with triggers (`start`, `driven_m`, `reached`). The
expected outcome is computed by replaying the plan offline. Kept only if well
posed: first route 400–900 m, total ≤ 2 km, every trigger fires.

**NPC.** Placed along each scenario's route: lead vehicles (30% below the
limit, following the ego's route), oncoming vehicles, and one crossing vehicle
per junction; 2–25 per scenario. Traffic Manager is synchronous and seeded,
so every backend meets the same traffic.

**Running.** Synchronous 20 Hz. The car is held with the handbrake until it
has settled; then traffic lights are reset, NPCs spawned and the first intent
issued on the same step. Events fire on the control thread, so the world is
paused while an intent is applied. A run ends on arrival, standing still after
a cancel, the time limit, or a minute without moving. A stop counts as visited
within 5 m.

| metric | definition |
|---|---|
| **success** | every expected stop visited in order and the right destination reached (or stopped after a cancel), no dropped place visited, within the time limit |
| completion | share of expected stops and destination reached in order |
| collisions | vehicle / static, de-duplicated over 1 s |
| hurry speed ratio | mean moving speed with / without hurry on the same route |
| detour, lane invasions / km, wrong-way s, rush ratio, cancel stop time | also computed |

Hurry is realised by `basic_agent` (target speed) and SimLingo (instruction);
TCP has no input for it.

## (c) End to end — `run.py --language`

```bash
python3 benchmark/execution/utterances.py                  # -> data/scenarios_language.jsonl
python3 benchmark/execution/run.py --backend basic_agent --language --llm-model gemini-3.8-flash
python3 benchmark/execution/evaluate.py benchmark/runs/execution/basic_agent__e2e__gemini-3.8-flash \
    --gold benchmark/runs/execution/basic_agent
```

**Sentences.** Each event's sentence is a reviewed language turn with the same
command structure, its places replaced by the scenario's and re-checked for
faithfulness. Passengers name places by type; the scripted passenger answers
"which one?" with the scenario's landmark ID, and places that type-only
grounding cannot reach are said with their ID (5 of 39 scenarios). All 84
sentences were reviewed by hand.

**Running.** As (b); the world is paused while the LLM parses and the route is
planned. Provider errors are retried, then logged as `api_error` and re-run.
`--llm-provider` / `--llm-model` pick the parser; `--llm-python` runs it in
another environment.

| metric | definition |
|---|---|
| **success** | as (b) |
| language loss | (b) success − (c) success, same backend |
| failure cause | **language** if an event left a wrong mission state or the first request gave no route, else **execution** |
| event state accuracy, parse latency | also computed |
