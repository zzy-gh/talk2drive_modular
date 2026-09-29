"""
covlm/paths.py
==============
The ``[covlm]`` section of config.yaml.

These paths belong to this subpackage, not to talk2drive. ``core/paths.py``
owns the config *file* and the resolution rules; it knows nothing about the
section this module reads, and nothing outside ``covlm/`` imports these names.
"""

from __future__ import annotations

from pathlib import Path

from core.paths import CONFIG_FILE, section_path

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"

#: Generated artefacts. They ship with the repo; rebuilding them is optional.
ROUTE_ANCHORS = DATA_DIR / "route_anchors.json"
VERIFY_REPORT = DATA_DIR / "nav_route_verification.csv"

OPENDRIVE_DIR = section_path("covlm", "opendrive_dir")
INTERDRIVE_DIR = section_path("covlm", "interdrive_dir")
DRIVER_INTENTS = section_path("covlm", "driver_intents")


def describe() -> str:
    """What the ``[covlm]`` section resolved to."""
    rows = [("covlm.opendrive_dir", OPENDRIVE_DIR),
            ("covlm.interdrive_dir", INTERDRIVE_DIR),
            ("covlm.driver_intents", DRIVER_INTENTS),
            ("data/route_anchors.json", ROUTE_ANCHORS),
            ("data/nav_route_verification.csv", VERIFY_REPORT)]
    out = [f"config: {CONFIG_FILE}", ""]
    for name, value in rows:
        if value is None:
            out.append(f"  {name:32s} (not set)")
        else:
            out.append(f"  {name:32s} {value}"
                       f"{'' if value.exists() else '   MISSING'}")
    return "\n".join(out)
