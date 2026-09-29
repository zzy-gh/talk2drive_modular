# covlm/

The CoVLM / InterDrive benchmark, kept out of the general package.

**Dependency rule, enforced by `tools/selftest.py`:** `covlm/` may import
`core/`, `backends/`, `runtime/`; **`core/` may never import `covlm/`.** Delete
this directory and talk2drive still runs. Nothing here leaks InterDrive's
vocabulary — scenario ids, r-numbers, vehicle suffixes — into the mission layer.

| file | role |
|---|---|
| `anchors.py` | `RouteAnchor` / `AnchorBook` / `ScenarioKB`, and the index-namespace guard |
| `build_route_anchors.py` | InterDrive route files → `data/route_anchors.json` |
| `verify_nav_routes.py` | does the mission layer reproduce the benchmark routes? |
| `verify_leaderboard_parity.py` | same question, under CoVLM's own carla 0.9.10 |
| `selftest.py` | offline checks (no server, no GPU, no LLM) |
| `paths.py` | the `[covlm]` config section |
| `data/` | the generated artefacts, committed |

```bash
python covlm/selftest.py                      # offline, needs nothing

python covlm/build_route_anchors.py           # -> data/route_anchors.json
PYTHONPATH=<carla>/PythonAPI/carla python covlm/verify_nav_routes.py \
    --report covlm/data/nav_route_verification.csv
```

Both tools run offline. `verify_nav_routes.py` builds `carla.Map` straight from
the town `.xodr`, so no CARLA server is contacted and nobody's running
simulation is disturbed.

## Why anchors

`covlm/anchors.py` lets the mission layer drive InterDrive's r1-r46 without
giving up route fidelity.

Those routes are short junction manoeuvres -- 15-172 m, median 40 m -- and
their end points sit a median of 36 m from the nearest real landmark, only
58 of 175 within 20 m. So "take me to the supermarket" can never reproduce
one: the planner would sail through the junction and keep going. An *anchor*
takes the destination off the route geometry instead (last keypoint = goal,
middle keypoints = vias) and keeps the passenger utterance for what it was
written to carry -- negotiation priority. About a third of the r1-r46
utterances name no destination at all ("just cruising, no particular plan"),
which is exactly why the destination must not come from the words.

| check | result |
|---|---|
| `nav_raw` -- anchor keypoints as-is | **175/175** reproduce the benchmark route |
| `nav_snap` -- same, but `snap()` first | 173/175 |
| `nav_goal` -- goal only, vias dropped | 158/175 |

Measured with whatever carla is on the path (0.9.15/0.9.16 here). CoLMDriver
runs **0.9.10**, with a different planner API (`GlobalRoutePlannerDAO` +
`grp.setup()`) and its own `register_dead_end_lanes` patch, so those numbers
mean nothing for the benchmark until they are checked there.
`verify_leaderboard_parity.py` does that, importing the patch from the
leaderboard's own source rather than reimplementing it:

| | carla 0.9.10, py3.7 (CoLMDriver's) | carla 0.9.15, py3.10 |
|---|---|---|
| reference routes traced | 175/175 | 175/175 |
| `nav_goal` | 158/175 | 158/175 |
| dead-end patch | +1 edge in Town06 | n/a |

and comparing the two dumps route by route: **175/175 identical**, within 1.1 m
and with the same RoadOption sequence. The version difference does not move a
single route.

```bash
CARLA=/path/to/colmdriver/carla
PYTHONPATH=$CARLA/PythonAPI/carla:$CARLA/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg \
python covlm/verify_leaderboard_parity.py --dump /tmp/routes_0910.json \
    --xodr $CARLA/CarlaUE4/Content/Carla/Maps/OpenDrive \
    --leaderboard /path/to/CoLMDriver-main/simulation/leaderboard

PYTHONPATH=/path/to/carla_0.9.15/PythonAPI/carla \
python covlm/verify_leaderboard_parity.py --dump /tmp/routes_0915.json \
    --xodr /path/to/carla_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive

python covlm/verify_leaderboard_parity.py --compare /tmp/routes_0910.json /tmp/routes_0915.json
```

So the mission layer reproduces all 46 scenarios exactly, and 17 routes across
11 scenarios need their via keypoints carried in the intent as
`waypoints: [Stop]` (`r4 r5 r6 r8 r9 r10 r11 r12 r20 r28 r46`). Of those 17,
5 trace the same *geometry* but a different **RoadOption** sequence -- and
RoadOption is what a backend eats as its high-level command, so they count.

### The four index namespaces

The one thing that must not get muddled. Anchors are **never inserted into the
knowledge base**:

| # | namespace | who indexes it |
|---|---|---|
| 1 | `kb.by_town[town]` row order | `find_coordinate(town, type, index)`, 0-based |
| 2 | `MissionPlanner._candidates(label)` | `destination_index` / `Stop.index`, the 1-based "the second museum" |
| 3 | `CliInteraction` menu `[1..N]` | what the human sees next to each number |
| 4 | `RouteAnchor.anchor_id` | `"r11_town05_ins_sl#veh1"` -- a key, not an ordinal |

Namespace 2 is namespace 1 filtered by building type, so appending *any* row to
the KB silently renumbers what "the second museum" means and reshuffles the CLI
menu under the passenger. `ScenarioKB` therefore delegates every KB call
through untouched and exposes anchors only via `anchor_place()`, which returns
a `Place` with `source="anchor"` and `index=None`. A mission binds that Place
directly and skips candidate resolution, so 1-3 cannot move.
`assert_kb_untouched()` snapshots every KB lookup and re-runs it through a
bound `ScenarioKB`; `covlm/selftest.py` runs it on every build.

### Two rules the sweep produced

* **Never snap an anchor.** KB landmarks sit off-road and need
  `RouteProvider.snap`; anchor coordinates are already road waypoints. Snapping
  them moved `r11_town05_ins_sl#veh0` 80.6 m onto a different road, cutting a
  159-point route to 59. Gate snapping on `Place.source == "kb"`.
* **Reset the router once per plan.** `GlobalRoutePlanner` keeps
  `_previous_decision` and `_intersection_end_node` on the *instance* and reads
  them in `_turn_decision`, so its RoadOption labels depend on which routes it
  traced before. The leaderboard never notices -- `interpolate_trajectory`
  builds a throwaway planner per call -- but we keep one for the whole session.
  Before the fix a 175-route sweep reported 17 differences that were pure
  call-order artefacts. `CarlaRouteProvider.trace_chain` resets once at the
  front of a plan and never between its legs, because the reference carries
  state across its own legs too.

