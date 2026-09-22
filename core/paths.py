"""
core/paths.py
=============
Reads config.yaml.

Paths that differ per machine live in the config file, not in the code.
Relative values resolve from the package root, so the repo keeps working
when the whole folder is moved.
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PKG_ROOT / "config.yaml"


def _parse_minimal(text: str) -> dict:
    """Parse the ``section: / two-space key: value`` subset config.yaml uses.

    This package deliberately runs across several conda environments -- the
    demo env, whichever env has a usable carla, SimLingo's own -- and pyyaml is
    not installed in all of them (``zhiyuan_gemini``, the env the README names,
    does not have it). A hard ``import yaml`` in this module would make every
    entry point unimportable there, for a config file that is two levels of
    plain strings. So pyyaml is used when present and this takes over when it
    is not.

    Handles: ``#`` comments, blank lines, one nesting level, quoted or bare
    scalar values, and empty values. Anything richer -- lists, nested maps,
    multi-line strings -- is out of scope, and ``_selftest_config_parsers``
    keeps the two parsers agreeing on the real file.
    """
    out: dict[str, dict] = {}
    section: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            key = line.split(":", 1)[0].strip()
            section = out.setdefault(key, {})
            continue
        if section is None or ":" not in line:
            continue
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        section[key.strip()] = value
    return out


def load_simple_yaml(path: Path) -> dict:
    """Load a ``section: / two-space key: value`` YAML file.

    pyyaml when it is installed, :func:`_parse_minimal` when it is not. Public
    because config.yaml is not the only file of this shape the package reads --
    CoLMDriver's ``driver_intents.yaml`` is the same two-level scalar map, and
    tools that read it must not quietly degrade in an env without pyyaml.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_minimal(text)
    return yaml.safe_load(text) or {}


_load = load_simple_yaml          # kept for the module-level read below


_cfg: dict = {}
if CONFIG_FILE.is_file():
    _cfg = _load(CONFIG_FILE)
else:
    print(f"[paths] no {CONFIG_FILE.name}; paths must be passed explicitly")


def section_path(section: str, key: str) -> Path | None:
    """One config value as a path, or None when unset.

    Relative values resolve from the package root. Public so a subpackage can
    own its own config section without core having to know it exists --
    ``covlm/paths.py`` reads ``[covlm]`` this way.
    """
    value = (_cfg.get(section) or {}).get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    return p if p.is_absolute() else PKG_ROOT / p


_path = section_path              # the original private name, still used below


KB_CSV = _path("paths", "kb_csv")

SIMLINGO_REPO = _path("simlingo", "repo")
SIMLINGO_PYTHON = _path("simlingo", "python")
SIMLINGO_CKPT = _path("simlingo", "checkpoint")

AGENT_MODULE = _path("leaderboard", "agent_module")
AGENT_CONFIG = _path("leaderboard", "agent_config")


def describe() -> str:
    """What the config resolved to, and whether each path exists.

    Walks the config file itself rather than a hard-coded list, so a section
    this module knows nothing about -- ``[covlm]``, or whatever comes next --
    still shows up in ``--paths`` instead of silently going unreported.
    """
    out = [f"config: {CONFIG_FILE}", ""]
    for section, entries in _cfg.items():
        if not isinstance(entries, dict):
            continue
        for key in entries:
            name = key if section == "paths" else f"{section}.{key}"
            value = section_path(section, key)
            if value is None:
                out.append(f"  {name:26s} (not set)")
            else:
                out.append(f"  {name:26s} {value}"
                           f"{'' if value.exists() else '   MISSING'}")
    return "\n".join(out)


def _selftest_config_parsers() -> tuple[bool, str]:
    """Do pyyaml and the fallback agree on the real config file?

    Returns (ok, detail). Reports ok when pyyaml is absent -- there is nothing
    to disagree with, and the fallback is what runs.
    """
    if not CONFIG_FILE.is_file():
        return True, "no config file"
    try:
        import yaml
    except ImportError:
        return True, "pyyaml not installed; fallback parser is the only path"
    text = CONFIG_FILE.read_text(encoding="utf-8")
    ours = _parse_minimal(text)
    theirs = yaml.safe_load(text) or {}
    norm = lambda d: {s: {k: ("" if v is None else str(v)) for k, v in (sec or {}).items()}
                      for s, sec in d.items()}
    a, b = norm(ours), norm(theirs)
    return a == b, "" if a == b else f"fallback={a}\n    pyyaml  ={b}"
