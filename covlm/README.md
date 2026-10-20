# covlm/ — PlugV2V

Passenger requests in multi-vehicle cooperative driving: every vehicle of
the InterDrive scenarios (r1–r46, 175 vehicles; CoVLM / CoLMDriver, CARLA
0.9.10) gets a passenger utterance. Talk2Drive turns it into the vehicle's
route, and its urgency tier goes to the V2V negotiation as a driver intent.

```
utterance ──► LLM ──► intent ──► landmark ──► GRP route ──► leaderboard route file  (where to drive)
     └──────────────────────────────────────────────────► driver intent + tier     (negotiation priority)
```

`covlm/` may import `core/`, `backends/`, `runtime/`; `core/` never imports
`covlm/` (checked by `tools/selftest.py`).

## Pipeline

```bash
python covlm/build_route_anchors.py          # InterDrive routes -> data/route_anchors.json
python covlm/design_passenger_inputs.py      # -> data/covlm_landmarks.csv, driver_intents_nav.yaml, passenger_missions.json
python covlm/verify_passenger_chain.py       # utterance -> LLM -> route, compared with the benchmark route
python covlm/export_nav_routes.py --install <CoLMDriver-main>   # *_nav route dirs + driver_intents_nav.yaml
```

The negotiation side (reading the driver intents into `cov2v`) is the
separate `covlm-passenger-intents` repo. After both runs:

```bash
python3 covlm/v2v_compare.py --with <results_with_intents> --without <results_without> \
    --intents covlm/data/driver_intents_nav.yaml
```

## Files

| file | role |
|---|---|
| `design_passenger_inputs.py` | one landmark per vehicle on its route end, and an utterance naming it with the original urgency tier |
| `kb.py` | landmark table served to the mission layer and the LLM prompt: the town's KB plus those landmarks |
| `utterance_overrides.py` | rephrasings for template lines the LLM reads ambiguously |
| `verify_passenger_chain.py` | full chain on all 175 vehicles (LLM answers cached in `data/llm_cache.json`) |
| `compute_min_vias.py` | smallest set of via points that reproduces each route |
| `export_nav_routes.py` | leaderboard route files and driver intents for CoLMDriver |
| `anchors.py`, `build_route_anchors.py`, `verify_nav_routes.py` | route anchors from the InterDrive route files, and planner checks against them |
| `verify_leaderboard_parity.py` | same routes under CARLA 0.9.10 and 0.9.15 |
| `v2v_compare.py` | driving score (RC, IP, DS, success by IC / LM / LC), urgency-consistent ordering, travel time |
| `selftest.py` | offline checks |
| `data/` | generated artefacts |

## Results of the checks

- The chain reproduces the benchmark route for all 175 vehicles; 17 routes in
  11 scenarios need via points, carried in the intent as stops.
- Routes traced under CARLA 0.9.10 (CoLMDriver) and 0.9.15 are identical
  (175/175, within 1.1 m, same RoadOption sequence).

## Pitfalls

- **Never snap an anchor**: anchor points are already on the road; snapping
  moved one 80.6 m onto another road. Snap only `Place.source == "kb"`.
- **Reset the router once per plan**: `GlobalRoutePlanner` keeps turn-decision
  state between calls; `CarlaRouteProvider.trace_chain` resets it at the start
  of each plan.
- **Do not add rows to the KB**: it renumbers "the second X". Anchors are
  served through `ScenarioKB.anchor_place()` instead; `assert_kb_untouched()`
  checks this.
