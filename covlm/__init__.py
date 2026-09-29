"""
covlm/
======
Everything specific to the CoVLM / InterDrive benchmark, kept out of the
general package.

The dependency rule is one-way and enforced by ``tools/selftest.py``:

    covlm/  may import  core/, backends/, runtime/
    core/   may NEVER import covlm/

So talk2drive runs with this whole directory deleted, and the benchmark work
never leaks InterDrive's vocabulary -- scenario ids, r-numbers, vehicle
suffixes -- into the mission layer.

What lives here:

    anchors.py              RouteAnchor / AnchorBook / ScenarioKB, and the
                            index-namespace guard
    build_route_anchors.py  InterDrive route files -> data/route_anchors.json
    verify_nav_routes.py    does the mission layer reproduce the benchmark
                            routes? (offline, no CARLA server)
    selftest.py             this subpackage's regression checks
    paths.py                the ``[covlm]`` config section
    data/                   the generated artefacts

What deliberately stayed in ``core/``, because it is not CoVLM-specific:

    Place.source            generic provenance on a resolved place
    CarlaRouteProvider      .reset() / .trace_chain() -- a CARLA GlobalRoute-
                            Planner state-leak fix that affects every re-plan
"""
