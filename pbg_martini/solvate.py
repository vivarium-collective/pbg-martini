"""Martini water + ion solvation of a coarse-grained system.

Fills the simulation box with Martini ``W`` water beads on a grid, removing any
that overlap the solute, then converts a fraction of waters into ``NA``/``CL``
ions to reach a target salt concentration and neutralise the system's net
charge (e.g. the chromosome's phosphate backbone). This is the standard
Martini solvation recipe (cf. ``insane`` / ``gmx solvate``), done with a KD-tree
exclusion so it scales to millions of grid points.

One ``W`` bead represents four real waters; bulk Martini water is ~8.3 W/nm^3
(0.47 nm grid).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# Martini water number density at a 0.47 nm cubic grid (~8.3 beads/nm^3).
WATER_SPACING_NM = 0.47
EXCLUSION_NM = 0.41          # drop water within this of any solute bead
AVOGADRO = 6.02214076e23


def solvate(solute_coords_nm: np.ndarray,
            box_nm: tuple[float, float, float],
            net_charge: float = 0.0,
            ion_conc_M: float = 0.15,
            spacing_nm: float = WATER_SPACING_NM,
            exclusion_nm: float = EXCLUSION_NM,
            seed: int = 0) -> dict:
    """Return Martini water + ion placements for a solute in ``box_nm``.

    Parameters
    ----------
    solute_coords_nm : (N,3) array
        Solute bead coordinates (nm), already inside the box.
    box_nm : (3,) tuple
        Box edge lengths (nm).
    net_charge : float
        Net charge of the solute; neutralising counter-ions are added.
    ion_conc_M : float
        Target NaCl concentration (mol/L).

    Returns
    -------
    dict
        ``{water (Nw,3), na (Nna,3), cl (Ncl,3), n_water, n_na, n_cl}``.
    """
    bx, by, bz = box_nm
    # grid of candidate water sites
    gx = np.arange(spacing_nm / 2, bx, spacing_nm)
    gy = np.arange(spacing_nm / 2, by, spacing_nm)
    gz = np.arange(spacing_nm / 2, bz, spacing_nm)
    grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)

    # remove sites overlapping solute
    if len(solute_coords_nm):
        tree = cKDTree(np.asarray(solute_coords_nm, dtype=float))
        d, _ = tree.query(grid, k=1, distance_upper_bound=exclusion_nm)
        grid = grid[~np.isfinite(d)]   # finite distance => within exclusion

    rng = np.random.default_rng(seed)
    rng.shuffle(grid)

    volume_L = (bx * by * bz) * 1e-24          # nm^3 -> L
    n_salt = int(round(ion_conc_M * AVOGADRO * volume_L))
    # neutralise: add |net_charge| extra counter-ions of the right sign
    n_na = n_salt + (int(round(-net_charge)) if net_charge < 0 else 0)
    n_cl = n_salt + (int(round(net_charge)) if net_charge > 0 else 0)
    n_ion = min(n_na + n_cl, len(grid))
    n_na = min(n_na, n_ion)
    n_cl = n_ion - n_na

    na = grid[:n_na]
    cl = grid[n_na:n_na + n_cl]
    water = grid[n_na + n_cl:]

    return {
        "water": water, "na": na, "cl": cl,
        "n_water": len(water), "n_na": len(na), "n_cl": len(cl),
    }
