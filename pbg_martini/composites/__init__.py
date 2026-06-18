"""Martini composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — :func:`make_martinize_document` builds a PBG
   state-dict programmatically for callers that want full control over
   the input PDB + wiring. Preserved for backward compatibility with the
   existing demo / tests.

2. **Declarative ``*.composite.yaml``** — sibling files in this directory
   follow the pbg-superpowers composite-spec convention.
   :func:`build_composite` loads one by name and instantiates
   ``process_bigraph.Composite`` with parameter substitution. The
   dashboard's composite explorer discovers these automatically once
   the package is installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_martini.processes import (
    MartinizeStep,
    MembraneBuilderStep,
    MicelleBuilderStep,
    ProteinMembraneStep,
    VesicleBuilderStep,
    ParsimonySliceStep,
    MartinizeSpeciesStep,
    ParsimonyAssembleStep,
    MartiniMDStep,
)

# Re-export the legacy hand-coded factory so existing call sites
# (`from pbg_martini.composites import make_martinize_document`) keep working.
from pbg_martini.composites._legacy import make_martinize_document

__all__ = [
    'make_martinize_document',
    'register_martini',
    'list_composite_specs',
    'load_composite_spec',
    'build_composite',
]


# ---------------------------------------------------------------------------
# Core registration
# ---------------------------------------------------------------------------

def register_martini(core=None):
    """Return a core with Martini Steps, the RAM emitter, and Visualizations
    registered.
    """
    if core is None:
        core = allocate_core()
    core.register_link('MartinizeStep', MartinizeStep)
    core.register_link('MembraneBuilderStep', MembraneBuilderStep)
    core.register_link('MicelleBuilderStep', MicelleBuilderStep)
    core.register_link('ProteinMembraneStep', ProteinMembraneStep)
    core.register_link('VesicleBuilderStep', VesicleBuilderStep)
    core.register_link('ParsimonySliceStep', ParsimonySliceStep)
    core.register_link('MartinizeSpeciesStep', MartinizeSpeciesStep)
    core.register_link('ParsimonyAssembleStep', ParsimonyAssembleStep)
    core.register_link('MartiniMDStep', MartiniMDStep)
    core.register_link('ram-emitter', RAMEmitter)
    core.register_link('RAMEmitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    try:
        from pbg_martini.visualizations import MartiniBeadSummaryPlots
        core.register_link('MartiniBeadSummaryPlots', MartiniBeadSummaryPlots)
    except Exception:
        # Visualization deps (pbg-superpowers) may be missing; soft-fail so
        # the legacy hand-coded factories still work.
        pass
    return core


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every ``*.composite.yaml`` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. ``name`` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a ``*.composite.yaml`` by name and instantiate
    ``process_bigraph.Composite``.

    Parameters
    ----------
    name : str
        Spec stem (no ``.composite.yaml`` suffix).
    overrides : dict, optional
        Parameter overrides (keys must match ``spec.parameters``).
    core : Core, optional
        Pre-built core; otherwise :func:`register_martini` is used.
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_martini()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
