# talk2drive_modular

Talk2Drive's language and mission layer, decoupled from the controller so it
can drive any AD stack: a PID waypoint follower, a VLA (SimLingo) or a
leaderboard policy (TCP). **Talk2Drive decides where to go; the backend decides
how to drive there.** Only a `GlobalPlan` and a `Directive` go in, and a
`carla.VehicleControl` comes out.

```
utterance ──► LLM parser ──► intent ──► MissionPlanner ──► GlobalPlan ──► DrivingBackend ──► VehicleControl
                                             │
                     LandmarkKnowledgeBase, RouteProvider (CARLA GRP), Interaction
```

## Layout

| path | role |
|---|---|
| `core/` | mission layer: intent schema, LLM parser, planner, route provider, landmark KB |
| `backends/` | one adapter per driving stack (`basic_agent`, `simlingo`, `leaderboard`) |
| `runtime/` | CLI, drive loop (`session.py`), harness loop (`passenger.py`), sensors, BEV/FPV windows |
| `benchmark/` | language, execution and end-to-end evaluation |
| `covlm/` | PlugV2V: talk2drive in multi-vehicle cooperative driving |
| `data/` | landmark knowledge base CSV |
| `tools/` | `selftest.py` (offline), `smoke_carla.py` (needs a server) |
| `examples/leaderboard_agent.py` | drop-in team_code agent for Bench2Drive |

## Configuration

Copy `config.example.yaml` to `config.yaml` (git-ignored) and fill in the paths.
Relative paths resolve from the repo root; `python3 runtime/cli.py --paths`
checks them.

```yaml
paths:
  kb_csv: data/special_buildings_en.csv

simlingo:
  repo: /path/to/covlm-agent-main/simlingo-main            # must contain simlingo_inference/
  python: /path/to/miniconda3/envs/zhiyuan_simlingo/bin/python
  checkpoint: /path/to/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

leaderboard:
  agent_module: AD software/TCP/leaderboard/team_code/tcp_agent.py
  agent_config: /path/to/TCP/new.ckpt
```

- `--backend simlingo` needs a SimLingo checkout with `simlingo_inference/`
  (not in the official release). InternVL2 is looked up under
  `<repo>/pretrained/InternVL2-1B`.
- `--backend leaderboard` loads the policy in-process, so run it in that
  stack's environment. `--llm-python` moves the LLM call to another one.

### API key

The LLM call reads its credentials from the environment, so no key goes in
`config.yaml` or anywhere else in the repo. Set your own and you are ready to
run:

```bash
export GEMINI_API_KEY=your-key-here
export GEMINI_MODEL=gemini-2.5-flash     # optional, this is the default
```

To avoid retyping it, put the same lines in the conda environment's activate
hook, which runs on every `conda activate`:

```bash
# $CONDA_PREFIX/etc/conda/activate.d/gemini.sh
export GEMINI_API_KEY=your-key-here
export GEMINI_MODEL=gemini-2.5-flash
```

Do this in every environment that makes the LLM call: `zhiyuan_gemini`, plus
`zhiyuan_tcp` when `--backend leaderboard` runs it in-process. The language
benchmark can go through any OpenAI-compatible endpoint instead, with
`QWEN_BASE_URL` and `QWEN_API_KEY` -- see `benchmark/README.md`.

## Run

| env | used for |
|---|---|
| `zhiyuan_gemini` | `basic_agent`, `simlingo`; carla 0.9.15, google-genai |
| `zhiyuan_simlingo` | SimLingo's worker, started automatically |
| `zhiyuan_tcp` | `leaderboard` with TCP |

Each of these needs `GEMINI_API_KEY` in its environment; see
[API key](#api-key).

```bash
cd /path/to/carla_0.9.15 && ./CarlaUE4.sh                    # server

conda activate zhiyuan_gemini                                # zhiyuan_tcp for TCP
python3 tools/selftest.py
python3 runtime/cli.py --backend basic_agent --town 1 --bev
python3 runtime/cli.py --backend simlingo --town 1 --bev --fpv
python3 runtime/cli.py --backend leaderboard --town 1 --sync --bev
```

Type requests at the `>` prompt in English or Chinese ("take me to the second
cafe", "顺路去一下ATM"). Commands: `view` (toggle spectator), `plan` (print
state), `clear` (cancel), `quit`.

| flag | effect |
|---|---|
| `--bev` / `--fpv` | bird's-eye window / the camera frame the model last saw |
| `--sync` / `--no-sync` | synchronous 20 Hz stepping (default on for simlingo) |
| `--auto` | never prompt: nearest landmark, unordered stops last |
| `--debug` | per-step model output to `logs/` |
| `--vehicle BP` | ego blueprint; default the car the stack was trained on (MKZ 2020 for SimLingo, MKZ 2017 for TCP) |
| `--simlingo-no-cot` | SimLingo without chain of thought |
| `--simlingo-camera X,Y,Z[,PITCH]` / `--simlingo-interval N` | SimLingo camera mount / steps per inference |
| `--world-viz` / `--no-viz` | draw in the CARLA world (visible to cameras) / no drawing |
| `--no-reload` | keep the running map |

If the process is killed in sync mode, CARLA waits for a tick forever; reset it with
`python3 -c "import carla; w=carla.Client('localhost',2000).get_world(); s=w.get_settings(); s.synchronous_mode=False; s.fixed_delta_seconds=None; w.apply_settings(s)"`.

## Benchmark

See [`benchmark/README.md`](benchmark/README.md): (a) language — utterance to
intent, offline; (b) execution — each backend drives gold intents in closed
loop, optionally with NPC traffic; (c) end to end — the same scenarios driven
from the passenger's words.

### Results

CARLA 0.9.15, synchronous 20 Hz, Town01–07 and Town10HD.

**(a) Language** — 632 scored turns. State accuracy: the route after applying
the parsed intent equals the gold route. Tiers: T1 one plain function, T2
several slots or indirect references, T3 depends on the current route,
several commands, or rejection / self-correction / disfluency.

| model | state acc | T1 | T2 | T3 | latency p50 / p95 (s) |
|---|---|---|---|---|---|
| gemini-3.8-flash | **95.9** | 99.6 | 97.4 | 89.1 | 4.49 / 12.95 |
| gemma-4-26b-a4b-it | 95.1 | 99.2 | 96.9 | 87.5 | 14.01 / 46.85 |
| gemini-3.5-flash-lite | 90.2 | 96.1 | 91.7 | 80.4 | 1.21 / 2.19 |
| Qwen3.5-4B (local, Ollama Q4_K_M, thinking off) | 83.4 | 89.5 | 88.0 | 70.1 | 0.52 / 0.78 |

**(b) Execution** — 39 scenarios, gold intents, at-rest start. With NPC: 2–25
vehicles per scenario placed along the route (lead, oncoming, crossing).

| backend | NPC | success | completion | collisions (vehicle / static) | hurry speed ratio |
|---|---|---|---|---|---|
| BasicAgent (PID) | – | **87.2** | 91.2 | 0 / 22 | 1.43 |
| TCP | – | 46.2 | 63.9 | 0 / 4 | 0.99 |
| SimLingo (no CoT) | – | 23.1 | 33.7 | 0 / 9 | 1.66 |
| BasicAgent (PID) | ✓ | **76.9** | 84.3 | 3 / 10 | 1.56 |
| TCP | ✓ | 41.0 | 60.1 | 5 / 14 | 0.98 |
| SimLingo (no CoT) | ✓ | 30.8 | 41.0 | 25 / 17 | 1.66 |

**(c) End to end** — the (b) scenarios without NPC, driven by BasicAgent from
the passenger's utterances. Language loss = gold success − end-to-end success.

| parser | gold success | E2E success | language loss | failures (language / execution) |
|---|---|---|---|---|
| gemini-3.8-flash | 87.2 | **87.2** | 0.0 | 0 / 5 |
| Qwen3.5-4B (local) | 87.2 | 69.2 | 18.0 | 8 / 4 |

## Backends

```python
from backends.base import BackendCaps, DrivingBackend, register

@register("my_stack")
class MyBackend(DrivingBackend):
    caps = BackendCaps(plan_format="dense", consumes_language=False,
                       control_rate_hz=10.0, needs_sensors=True)

    def sensors(self):                     return [...]   # leaderboard format
    def attach(self, vehicle, world):      ...
    def set_plan(self, plan, directive):   ...            # every re-plan
    def run_step(self, obs):               return carla.VehicleControl(...)
```

Add it to `_LAZY` in `backends/__init__.py`. The runtime reads `BackendCaps`:
`needs_sensors` (spawn the sensor rig), `reports_done` (trust `done()` or the
mission layer), `consumes_language` (directive as a sentence or a number).

| | `basic_agent` | `simlingo` | `leaderboard` |
|---|---|---|---|
| plan | dense `(Waypoint, RoadOption)` | 2 target points (route downsampled at 200 m) | `set_global_plan(gps, world)` |
| also | — | front RGB, speed, speed instruction | sensor dict |
| process | in-process | separate env, JSON over a pipe | in-process |
| arrival | own | mission layer | mission layer |

One backend per run: a policy is never silently replaced by another mid-route.

Two loop shapes: in `runtime/session.py` talk2drive owns the loop (control
thread drives, CLI thread parses and re-plans); in `runtime/passenger.py` a
harness calls `run_step()`, and parsing and planning run on a worker thread
behind non-blocking `submit()` / `poll()`.

## Pitfalls

- **Ego frame** is `[forward, right]` from CARLA's own forward/right vectors,
  as SimLingo uses for `target_points_ego`.
- **Target points behind the car** make a VLA steer backwards;
  `SparseRouteTracker` drops them.
- **Sparsity is per stack**: 50 m for leaderboard / Bench2Drive, 200 m for
  SimLingo. Never hand a VLA dense points, and never pre-downsample for a
  leaderboard agent (its `set_global_plan` does it).
- **Never prompt on the control thread**: `build_plan` may prompt, so pass a
  non-blocking `Interaction` in real-time loops.
- **Sensor specs**: leaderboard `width`/`height` map to `image_size_x/y`
  (`runtime/sensors.py::blueprint_attributes`).
- **Debug drawing**: `draw_point`/`draw_line` appear in camera images;
  `draw_string` does not.
- **Copy sensor buffers**: `image.raw_data` is freed after the callback.
- **Re-plans** start from `CarlaRouteProvider.snap_heading` so a drifted ego
  does not plan from the oncoming lane.
- **TCP's route planner** is built once; the backend resets it on every re-plan.

## PlugV2V

[`covlm/`](covlm/README.md) brings passenger requests into multi-vehicle
cooperative driving (CoVLM / CoLMDriver on the InterDrive scenarios, CARLA
0.9.10). Each vehicle's request goes through the talk2drive chain (utterance →
LLM → landmark → route) and is exported as leaderboard route files plus driver
intents with an urgency tier (`covlm/data/export/`). The patch that feeds these
intents into the V2V negotiation is the separate `covlm-passenger-intents`
repo. `covlm/v2v_compare.py` compares runs with and without the intents.

`covlm/` may import `core/`; `core/` never imports `covlm/` (checked by
`tools/selftest.py`).
