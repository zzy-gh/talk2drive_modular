"""Backend registry. Importing a backend module is what registers it.

Heavy backends (torch, a model checkpoint, a subprocess) are imported lazily
so that ``--backend basic_agent`` never pays for SimLingo being installed —
or fails because it is not.
"""

from .base import (BackendCaps, DrivingBackend, available, brake_control,
                   build as _build, register)

_LAZY = {
    "basic_agent": "backends.basic_agent",
    "simlingo":    "backends.simlingo",
    "leaderboard": "backends.leaderboard",
}


def build(name: str, **kwargs) -> DrivingBackend:
    import importlib
    if name in _LAZY:
        importlib.import_module(_LAZY[name])
    return _build(name, **kwargs)


def load_all() -> list[str]:
    """Import every known backend; returns the ones that imported cleanly."""
    import importlib
    ok = []
    for name, module in _LAZY.items():
        try:
            importlib.import_module(module)
            ok.append(name)
        except Exception as exc:
            print(f"[backends] {name} unavailable: {exc}")
    return ok


__all__ = ["DrivingBackend", "BackendCaps", "register", "build", "available",
           "load_all", "brake_control"]
