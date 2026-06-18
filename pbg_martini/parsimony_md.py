"""Relax + short-MD stages for the parsimony -> Martini bridge.

``relax_assembly`` (Task 8) wraps pbg-martini's WCA steepest-descent minimizer
to remove residual packing clashes from a stamped assembly. ``run_short_md``
(Task 9) is a best-effort OpenMM runner that reads the GROMACS Martini topology
directly; it degrades cleanly when OpenMM is not installed.
"""

from __future__ import annotations

import os

import numpy as np


def relax_assembly(coords_nm, n_steps=400, sigma=0.47, perturb=0.0,
                   repulsion_eps=3.0, dt=0.005, seed=42) -> np.ndarray:
    """WCA-relax a raw ``(N, 3)`` coordinate array to remove clashes.

    Adapts :func:`pbg_martini.builders.relax_structure` (which operates on a
    builder ``result`` dict) to accept and return a plain coordinate array.
    Bonds are not modelled here — this is a pure inter-molecular declash, so
    the minimum pairwise distance only increases.
    """
    from pbg_martini.builders import relax_structure

    coords = np.asarray(coords_nm, dtype=float)
    if coords.shape[0] == 0:
        return coords.copy()

    result = {
        "beads": [{"pos": list(map(float, p))} for p in coords],
        "bonds": [],
        "stats": {},
    }
    relax_structure(
        result,
        n_steps=n_steps,
        sigma=sigma,
        perturb=perturb,
        repulsion_eps=repulsion_eps,
        dt=dt,
        seed=seed,
    )
    return np.array([b["pos"] for b in result["beads"]], dtype=float)


# --------------------------------------------------------------------------
# Task 9: OpenMM short MD (best-effort)
# --------------------------------------------------------------------------

def openmm_available() -> bool:
    """Return True if OpenMM can be imported."""
    try:
        import openmm  # noqa: F401
        return True
    except Exception:
        return False


def run_short_md(gro_path, top_path, steps=200, out_traj="traj.dcd",
                 include_dir=None, temperature=310.0, dt_fs=20.0,
                 cutoff_nm=1.1, minimize=True, platform_name="CPU") -> dict:
    """Run a short Martini NVT MD via OpenMM, reading the GROMACS topology.

    Builds the system with ``GromacsGroFile`` + ``GromacsTopFile`` (the standard
    route to run Martini in OpenMM), Martini nonbonded settings
    (``CutoffPeriodic``, ~1.1 nm), a ``LangevinMiddleIntegrator`` at the Martini
    timestep, an optional local energy minimization, then ``steps`` of NVT.

    ``include_dir`` should point at the Martini force-field ``.itp`` directory
    when the ``.top`` ``#include``s external parameters (e.g. a martini-forcefields
    checkout); self-contained tops need no include dir.

    Returns ``{minimized_gro, traj, final_energy}`` (energy in kJ/mol). Raises
    ``RuntimeError`` if OpenMM is unavailable — callers gate on
    :func:`openmm_available`.
    """
    if not openmm_available():
        raise RuntimeError("OpenMM is not installed")

    from openmm import app, unit, LangevinMiddleIntegrator, Platform

    gro = app.GromacsGroFile(gro_path)
    top_kwargs = {"periodicBoxVectors": gro.getPeriodicBoxVectors()}
    if include_dir:
        top_kwargs["includeDir"] = include_dir
    top = app.GromacsTopFile(top_path, **top_kwargs)

    system = top.createSystem(
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
    )
    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picosecond,
        (dt_fs / 1000.0) * unit.picoseconds,
    )
    try:
        platform = Platform.getPlatformByName(platform_name)
    except Exception:
        platform = None
    sim = app.Simulation(top.topology, system, integrator, platform)
    sim.context.setPositions(gro.positions)

    if minimize:
        sim.minimizeEnergy(maxIterations=500)

    out_dir = os.path.dirname(os.path.abspath(top_path))
    minimized_gro = os.path.join(out_dir, "minimized.gro")
    _write_minimized_gro(minimized_gro, sim, gro)

    traj_path = out_traj
    if not os.path.isabs(traj_path):
        traj_path = os.path.join(out_dir, out_traj)
    if steps > 0:
        sim.reporters.append(app.DCDReporter(traj_path, max(1, steps // 10)))
        sim.step(steps)

    energy = sim.context.getState(getEnergy=True).getPotentialEnergy()
    final_energy = energy.value_in_unit(unit.kilojoule_per_mole)

    return {
        "minimized_gro": minimized_gro,
        "traj": traj_path if steps > 0 else None,
        "final_energy": float(final_energy),
    }


def _write_minimized_gro(path, sim, gro):
    """Write the current simulation positions back to a GROMACS ``.gro``."""
    from openmm import unit
    from pbg_martini.parsimony_assembler import write_gro

    state = sim.context.getState(getPositions=True)
    pos_nm = np.array(
        state.getPositions().value_in_unit(unit.nanometer), dtype=float
    )
    atom_names = [a.name for a in sim.topology.atoms()]
    box = gro.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    box_nm = (box[0][0], box[1][1], box[2][2])
    write_gro(path, atom_names, pos_nm, box_nm)
    return path
