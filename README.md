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
| `runtime/cli.py` | the interactive entry point |
| `runtime/viz.py` | all CARLA debug drawing, plus plan/landmark state for the BEV |
| `runtime/camera.py` | spectator: top-down on the ego, pulls back to the whole town |
| `runtime/bev.py` | `--bev` window: roads, route, landmarks, ego — drawn outside CARLA |
| `runtime/fpv.py` | `--fpv` window: the frame SimLingo last saw |
| `runtime/gui.py` | the one thread all OpenCV windows run on |
| `runtime/sensors.py` | sensor rig, leaderboard-shaped `input_data` |
| `logs/` | `--debug` per-step logs (git-ignored) |
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

simlingo:
  repo: /path/to/covlm-agent-main/simlingo-main            # must contain simlingo_inference/
  python: /path/to/miniconda3/envs/zhiyuan_simlingo/bin/python
  checkpoint: /path/to/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

leaderboard:
  agent_module: AD software/TCP/leaderboard/team_code/tcp_agent.py
  agent_config: /path/to/TCP/new.ckpt                       # TCP: its checkpoint
```

The SimLingo checkpoint's Hydra config is found automatically at
`<ckpt>/../../../.hydra/config.yaml`. SimLingo's worker looks for InternVL2 under
`<simlingo repo>/pretrained/InternVL2-1B`; a symlink to the HuggingFace cache
snapshot avoids a 1.8 GB download.

It is YAML, not TOML — `section:` then two-space-indented `key: value`, no
`[brackets]` and no `=`. `core/paths.py` reads it with pyyaml when that is
installed and with a small built-in parser when it is not, because this package
runs across several conda environments and not all of them have pyyaml.

Relative paths resolve from the repo root, so the folder can be moved or
copied anywhere without editing anything. Command-line flags override the
config for one-off runs. `python3 runtime/cli.py --paths` prints what the
config resolved to and flags anything missing.

Two notes on what is not in the repo:

* **`--backend simlingo` needs the CoVLM fork.** `simlingo_inference/` — the
  worker the subprocess backend talks to — is not in the official SimLingo
  release under `AD software/`. Point `[simlingo] repo` at a checkout that has it.
* **`--backend leaderboard` loads the policy in-process**, so the whole process
  has to run in an environment with that stack's dependencies (for TCP:
  `zhiyuan_tcp`, below). The SimLingo backend avoids this by using a
  subprocess. An agent's own import roots are derived from its module path,
  so `PYTHONPATH` needs no setting. `--llm-python` can move just the Gemini
  call to another interpreter if the driving env cannot carry google-genai.

## Run

### Environments

| env | used for | holds |
|---|---|---|
| `zhiyuan_gemini` | `basic_agent`, `simlingo` | carla 0.9.15, google-genai, OpenCV (GUI) |
| `zhiyuan_simlingo` | SimLingo's worker only — started for you, never activated | torch 2.2, SimLingo |
| `zhiyuan_tcp` | `leaderboard` with TCP | torch 2.5, carla 0.9.15, google-genai, py_trees, pytorch-lightning |

Gemini credentials live in each activated env's conda hook
(`etc/conda/activate.d/gemini.sh`): `GEMINI_API_KEY` and `GEMINI_MODEL`
(currently `gemini-3.5-flash-lite`; the 2.5 models are closed to this key).
The hook also sets `PYTHONNOUSERSITE=1` so `~/.local` packages stay out.
CARLA's `agents/` package is put on the path by a `carla_agents.pth` in each env.

### Steps

```bash
# 1. CARLA server, in its own terminal
cd /path/to/carla_0.9.15 && ./CarlaUE4.sh

# 2. talk2drive, from the repo root
conda activate zhiyuan_gemini            # zhiyuan_tcp for TCP
cd /path/to/talk2drive_modular

python3 tools/selftest.py                # offline checks, nothing else running

# PID waypoint follower
python3 runtime/cli.py --backend basic_agent --town 1 --bev

# SimLingo (VLA, runs in zhiyuan_simlingo over a pipe)
python3 runtime/cli.py --backend simlingo --town 1 --bev --fpv

# TCP (leaderboard agent, in-process) -- needs `conda activate zhiyuan_tcp`
python3 runtime/cli.py --backend leaderboard --town 1 --sync --bev
```

Then type instructions at the `>` prompt, in English or Chinese ("take me to
the second cafe", "顺路去一下ATM", "在这之后我们去公园"). Built-in commands:

| command | effect |
|---|---|
| `view` | toggle the spectator between the ego and the whole town |
| `plan` | print the current mission state |
| `clear` | cancel the route and reset every landmark to unused |
| `quit` / `exit` | leave (restores CARLA to asynchronous mode) |

### Flags

| flag | effect |
|---|---|
| `--bev` | bird's-eye-view window: roads, route, landmark states, ego |
| `--fpv` | the exact camera frame SimLingo last saw (simlingo only); the dimmed band at the bottom is cropped off before the model |
| `--debug` | per-step model output to `logs/simlingo_debug.log` or `logs/leaderboard_debug.log`, not the terminal |
| `--sync` / `--no-sync` | synchronous 20 Hz stepping; on by default for simlingo, pass `--sync` for TCP |
| `--simlingo-camera X,Y,Z[,PITCH]` | SimLingo camera mount; default `-0.5,0,2.0`, training mount is `-1.5,0,2.0` |
| `--simlingo-interval N` | control steps per SimLingo inference (default 1 in sync mode) |
| `--vehicle BP` | ego blueprint; default Lincoln MKZ 2020 for simlingo, Tesla Model 3 otherwise |
| `--world-viz` | draw points and lines in CARLA even for a camera backend (its camera sees them) |
| `--no-viz` | no drawing, no spectator control |
| `--auto` | never prompt: nearest landmark, unordered stops appended last |
| `--no-reload` | keep the running map (and any old drawings) instead of reloading |

Watch a debug log from another terminal:

```bash
tail -F logs/simlingo_debug.log        # or logs/leaderboard_debug.log
```

### What you see

* **BEV window** — the reference view. Landmarks: grey unused, red
  destination, orange stop (numbered), green reached, purple cancelled, yellow
  `[n]` a candidate to pick.
* **CARLA window** — for `basic_agent`, route, start and landmarks as points
  and lines. For camera backends only the route (a dotted line of `o`) and an
  `EGO` label, all drawn with `draw_string`, which camera sensors do not
  capture; a cancelled route is painted grey.
* **Spectator** — top-down 50 m above the ego; pulls back to the whole town
  when a route is planned or candidates are offered, returns on arrival.

When a stop is left unordered ("I also want to go to X"), you are asked where
in the route it goes. When the passenger says where (first, last, after the
bakery, after the destination) it is placed without asking.

If talk2drive is killed hard in sync mode, CARLA waits forever for a tick.
Restore it with:

```bash
python3 -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(10); w=c.get_world(); s=w.get_settings(); s.synchronous_mode=False; s.fixed_delta_seconds=None; w.apply_settings(s)"
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
| rate | 20 Hz | every step in sync mode (sim waits for it) | 20 Hz |
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
* **Sensor specs are not blueprint attributes.** A leaderboard spec says
  `width`/`height`; the CARLA camera calls them `image_size_x`/`image_size_y`.
  `runtime/sensors.py::blueprint_attributes` mirrors the leaderboard's
  `agent_wrapper.py`, lens and noise settings included. Setting `width`
  directly is a silent no-op that leaves every camera at 800x600, FOV 90.
* **Debug drawing is in the camera image.** `draw_point`/`draw_line`/`draw_arrow`
  are scene geometry a camera sensor renders. `draw_string` is server-side
  only, but expires in *wall-clock* time and `life_time=0` is not permanent
  for it (`CarlaHUD.cpp`).
* **Copy sensor buffers.** `image.raw_data` points at memory CARLA frees when
  the callback returns; a `np.frombuffer` view read later is all zeros.
* **A drifted ego re-plans from the oncoming lane** with a plain nearest-lane
  snap. Re-plans start from `CarlaRouteProvider.snap_heading`, which keeps the
  lane running the ego's way.
* **Leaderboard agents build their route planner once.** `set_global_plan`
  alone does not reach TCP's `_route_planner`; the backend resets
  `initialized` on every re-plan.

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
