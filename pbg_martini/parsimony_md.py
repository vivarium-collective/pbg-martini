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
