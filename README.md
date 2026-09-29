# talk2drive_modular

Talk2Drive's language/mission layer, decoupled from the controller so it can
drive **any** AD stack: the original PID waypoint follower, a VLA such as
SimLingo, or an E2E leaderboard policy such as TCP.

The split in one line: **talk2drive decides *where to go*; a backend decides
*how to drive there*.** The only thing crossing between them is a
`GlobalPlan` plus a `Directive` in, a `carla.VehicleControl` out.

```
utterance ──► GeminiCommandParser ──► intent ──► MissionPlanner ──► GlobalPlan ──┐
                                                       │                         │
                                       LandmarkKnowledgeBase                     ▼
                                       RouteProvider (CARLA GRP)          DrivingBackend
                                       Interaction (who answers                  │
                                         "which hospital?")                      ▼
                                                                         VehicleControl
```

## Layout

| path | role |
|---|---|
| `core/` | mission layer. No controller, no carla debug drawing, no `input()`. |
| `core/types.py` | `GlobalPlan`, `Directive`, `Observation`, ego/GPS frame helpers |
| `core/mission.py` | the intent state machine and the planner |
| `core/resolver.py` | disambiguation **policy** (auto / CLI / deferred) |
| `core/route_provider.py` | `(start, goal) -> dense route`; swap for a non-CARLA map |
| `core/tracking.py` | sparse route tracking for VLA target points |
| `backends/` | one adapter per driving stack, plus a registry |
| `runtime/session.py` | loop shape **A** — talk2drive owns the loop |
| `runtime/passenger.py` | loop shape **B** — an evaluation harness owns the loop |
| `runtime/viz.py` | all CARLA debug drawing |
| `runtime/sensors.py` | sensor rig, leaderboard-shaped `input_data` |
| `config.yaml` | machine-specific paths |
| `core/paths.py` | reads `config.yaml` (~60 lines) |
| `data/` | bundled data — the knowledge base CSV |
| `covlm/` | the CoVLM / InterDrive benchmark, self-contained |
| `tools/selftest.py` | offline regression checks (no server, no GPU, no LLM) |
| `tools/smoke_carla.py` | in-the-loop check (server, but no LLM/checkpoint) |
| `examples/leaderboard_agent.py` | drop-in team_code agent for Bench2Drive |

## Configuration

All machine-specific paths live in `config.yaml`, which is git-ignored because
it is yours, not the project's. Copy
[`config.example.yaml`](config.example.yaml) once after cloning and fill it in:

```yaml
paths:
  kb_csv: data/special_buildings_en.csv       # ships with the repo
  opendrive_dir: /path/to/CARLA/.../OpenDrive

simlingo:
  repo: /path/to/simlingo                     # must contain simlingo_inference/
  python: /path/to/conda/envs/simlingo/bin/python
  checkpoint: ""

leaderboard:
  agent_module: AD software/TCP/leaderboard/team_code/tcp_agent.py
  agent_config: ""
```

It is YAML, not TOML — `section:` then two-space-indented `key: value`, no
`[brackets]` and no `=`. `core/paths.py` reads it with pyyaml when that is
installed and with a small built-in parser when it is not, because this package
runs across several conda environments and not all of them have pyyaml.

Relative paths resolve from the repo root, so the folder can be moved or
copied anywhere without editing anything. Command-line flags override the
config for one-off runs. `python -m runtime.cli --paths` prints what the
config resolved to and flags anything missing.

Two notes on what is not in the repo:

* **`--backend simlingo` needs the CoVLM fork.** `simlingo_inference/` — the
  worker the subprocess backend talks to — is not in the official SimLingo
  release under `AD software/`. Point `[simlingo] repo` at a checkout that has it.
* **`--backend leaderboard` loads the policy in-process**, so the whole process
  (Gemini SDK included) has to run in that stack's conda environment. The
  SimLingo backend avoids this by using a subprocess. An agent's own import
  roots are derived from its module path, so `PYTHONPATH` needs no setting.

## Run

```bash
conda activate zhiyuan_gemini

# offline checks first — they need nothing running
python tools/selftest.py

# the original demo, now with a --backend flag
python -m runtime.cli --backend basic_agent --town 1

# SimLingo (runs in its own conda env, over a pipe)
python -m runtime.cli --backend simlingo \
    --simlingo-ckpt /path/to/checkpoint \
    --simlingo-python /home/dellpro2/miniconda3/envs/simlingo-inference/bin/python

# any leaderboard AutonomousAgent (TCP, Bench2Drive baselines, ...)
python -m runtime.cli --backend leaderboard \
    --agent-module /path/to/tcp_agent.py --agent-config /path/to/tcp_config.py

```

**One backend per run.** There is deliberately no way to compose two stacks
into one drive: if a learned policy quietly degraded into the PID controller
mid-route, the numbers you report would belong to neither. A policy's failure
has to stay visible as that policy's failure.

`--auto` never prompts (nearest landmark wins) — use it for evaluation runs.

## The two loop shapes

**A — talk2drive drives** (`runtime/session.py`). A control thread ticks the
backend at a fixed rate; the CLI thread parses and re-plans. Disambiguation
prompts happen on the CLI thread, so the car keeps driving its previous route
while the passenger is being asked a question. The plan swap between the two is
a single locked assignment.

**B — talk2drive rides along** (`runtime/passenger.py`). Under Bench2Drive the
harness calls `agent.run_step()`, so talk2drive cannot own a loop and must
never block inside one. `submit()` only enqueues; a worker thread does the LLM
call, the landmark resolution and `trace_route`; `poll()` is non-blocking and
returns a finished plan or `None`. `make_talk2drive_agent(TCPAgent, factory)`
wraps an existing policy class with this, leaving the policy untouched.

## Adding a backend

```python
from backends.base import BackendCaps, DrivingBackend, register

@register("my_stack")
class MyBackend(DrivingBackend):
    caps = BackendCaps(plan_format="dense", consumes_language=False,
                       control_rate_hz=10.0, needs_sensors=True)

    def sensors(self):                 return [...]          # leaderboard format
    def attach(self, vehicle, world):  ...                    # ego exists now
    def set_plan(self, plan, directive): ...                  # every re-plan
    def run_step(self, obs):           return carla.VehicleControl(...)
```

Then add it to `_LAZY` in `backends/__init__.py` so it is imported on demand
and a missing dependency never breaks the other backends.

`BackendCaps` is not decoration — the runtime reads it:

| field | what the runtime does with it |
|---|---|
| `needs_sensors` | spawns the sensor rig, or skips it entirely |
| `reports_done` | trusts `done()`, or lets the mission layer decide arrival |
| `control_rate_hz` | documents the inference rate the backend caches against |
| `consumes_language` | whether `Directive` goes in as a sentence or a number |
| `owns_loop` | reserved for hosts that tick the backend themselves |

## What each backend actually eats

| | `basic_agent` | `simlingo` | `leaderboard` |
|---|---|---|---|
| plan | dense `(Waypoint, RoadOption)` | `target_points_ego` `[2,2]` | `set_global_plan(gps, world)` |
| also | — | front RGB, speed, **language** | sensor dict |
| rate | 20 Hz | ~1.3 Hz, control cached between | 20 Hz |
| process | in-process | separate conda env, JSON over a pipe | in-process |
| knows it arrived | yes | no — mission layer decides | no |

## Verified against the official repos

`AD software/TCP` and `AD software/simlingo` are the upstream checkouts the
backends were checked against. What the check settled:

| claim | source | result |
|---|---|---|
| ego frame is `[forward, right]` with CARLA yaw | `simlingo/team_code/transfuser_utils.py::inverse_conversion_2d` + `preprocess_compass` | **confirmed** — expanding their rotation gives CARLA's own forward/right vectors; asserted numerically in `tools/selftest.py` |
| sparse tracker pop rule | `simlingo/team_code/nav_planner.py::RoutePlanner.run_step` | **identical** |
| tracker bounds 7.5 / 50 m | `simlingo/team_code/agent_simlingo.py:141-145` | **match** |
| target points are route[1], route[2] | `agent_simlingo.py:442-443` | **match** |
| route sparsity | `simlingo/leaderboard/.../autonomous_agent.py:130` | **was wrong** — SimLingo's fork downsamples at **200 m**, not the usual 50. Fixed. |
| leaderboard agent construction | `TCP/leaderboard/.../autonomous_agent.py:35-45` | **was wrong** — leaderboard 1.0 takes `__init__(path_to_conf_file)` and calls `setup()` itself; only 2.0/Bench2Drive/SimLingo take `(host, port, debug)`. Fixed, dispatched by signature. |
| who downsamples the global plan | all three base classes | **was wrong** — every base class downsamples inside `set_global_plan`, so pre-downsampling compounded it. `sample_factor` now defaults to `None`. |

Two ways to run SimLingo, both valid:

* `--backend simlingo` — our worker subprocess. talk2drive stays in its own
  conda env; nothing torch-shaped is imported in-process.
* `--backend leaderboard --agent-module "AD software/simlingo/team_code/agent_simlingo.py"`
  — the official agent, unmodified. Needs the whole process to run in
  SimLingo's env, so the Gemini SDK has to live there too.

## Things that will bite you

* **Ego frame.** `world_to_ego` uses CARLA's own `get_forward_vector()` /
  `get_right_vector()` rather than a hand-rolled rotation, so the left-handed
  `y` axis cannot be signed wrong. Output is `[forward, right]` in metres,
  which is what SimLingo/CoVLM feed into `target_points_ego`.
* **Target points behind the car.** The leaderboard's pop rule only fires when
  the ego passes *within 7.5 m* of a sparse point, so a re-plan or a lane
  offset can leave a point sitting behind you — and a target point behind the
  ego is the classic way to make a VLA steer backwards. `SparseRouteTracker`
  additionally drops points behind the heading. Covered by `tools/selftest.py`.
* **Sparsity matters, and the factor is per-stack.** A VLA was trained on
  target points from a downsampled route, not on dense 1 m waypoints — and the
  factor differs: 50 m for stock leaderboard / Bench2Drive, **200 m** for
  SimLingo's fork. `GlobalPlan.downsample` reproduces the rule; the factor is a
  backend parameter. Never hand a VLA `plan.points` directly.
* **Never pre-downsample for a leaderboard agent.** Every `AutonomousAgent`
  base class downsamples inside `set_global_plan`; doing it again upstream
  compounds and stretches the spacing the policy was trained on.
* **Never prompt on the control thread.** `MissionPlanner.apply` is always
  safe; `build_plan` may prompt, so pass a non-blocking `Interaction` whenever
  a real-time loop could reach it. `Talk2DrivePassenger` refuses a blocking
  policy outright rather than deadlocking a simulation later.
* **Arrival.** Only `basic_agent` knows when the route is finished. For
  everything else `MissionPlanner.is_arrived` (goal radius) is authoritative.

## CoVLM / InterDrive

Kept out of this package, in [`covlm/`](covlm/), with its own README, its own
selftest and its own `[covlm]` config section. The dependency rule is one-way
and `tools/selftest.py` enforces it: **`covlm/` may import `core/`; `core/` may
never import `covlm/`.** Delete the directory and everything here still runs.

Two fixes that came out of that work did stay, because neither is
benchmark-specific: `CarlaRouteProvider.reset()` / `.trace_chain()` (a CARLA
`GlobalRoutePlanner` state leak that affects every re-plan) and `Place.source`.

## Parity with `talk2drive_1/`

Same behaviour, same knowledge base, same intent schema, same prompts — the
Gemini parsing path is copied verbatim into `core/`. Behavioural differences
are deliberate:

* disambiguation is a policy rather than a hard-coded `input()`;
* the planner returns a `GlobalPlan` instead of a CARLA waypoint list;
* drawing moved out of the planner into `runtime/viz.py`;
* `--auto` and headless operation are now possible at all.

`talk2drive_1/` is untouched and still runs.
