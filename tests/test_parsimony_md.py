"""Tests for the relax + short-MD stages (Tasks 8, 9)."""

import os

import numpy as np
import pytest


# --------------------------------------------------------------------------
# Task 8: WCA relax wrapper
# --------------------------------------------------------------------------

def test_relax_separates_clashing_beads():
    from pbg_martini.parsimony_md import relax_assembly
    coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])  # 0.1 nm clash
    sigma = 0.47
    relaxed = relax_assembly(coords, n_steps=400, sigma=sigma)
    assert not np.any(np.isnan(relaxed))
    d0 = np.linalg.norm(coords[1] - coords[0])
    d1 = np.linalg.norm(relaxed[1] - relaxed[0])
    assert d1 >= d0                # min distance must not decrease
    assert d1 >= sigma * 0.99      # relaxed to >= the WCA sigma


def test_relax_empty():
    from pbg_martini.parsimony_md import relax_assembly
    out = relax_assembly(np.zeros((0, 3)))
    assert out.shape == (0, 3)


# --------------------------------------------------------------------------
# Task 9: OpenMM short MD (best-effort; skip if openmm absent)
# --------------------------------------------------------------------------

def _openmm_missing():
    from pbg_martini.parsimony_md import openmm_available
    return not openmm_available()


# A self-contained Martini-like GROMACS topology (no external ff includes), so
# the MD smoke test runs without the martini-forcefields package on disk.
MINI_TOP = """\
[ defaults ]
1 2 no 1.0 1.0

[ atomtypes ]
; name at.num mass charge ptype sigma epsilon
P4    0  72.0  0.000  A  0.47  3.5

[ moleculetype ]
; name nrexcl
W 1

[ atoms ]
; nr type resnr residue atom cgnr charge
1  P4  1  W  W  1  0.0

[ system ]
mini

[ molecules ]
W 3
"""

MINI_GRO = (
    "mini\n    3\n"
    "    1W        W    1   0.000   0.000   0.000\n"
    "    2W        W    2   1.200   0.000   0.000\n"
    "    3W        W    3   0.000   1.200   0.000\n"
    "   4.00000   4.00000   4.00000\n"
)


@pytest.mark.skipif(_openmm_missing(), reason="OpenMM not installed")
def test_run_short_md_finite_energy(tmp_path):
    from pbg_martini.parsimony_md import run_short_md
    top = tmp_path / "mini.top"
    gro = tmp_path / "mini.gro"
    top.write_text(MINI_TOP)
    gro.write_text(MINI_GRO)
    out = run_short_md(str(gro), str(top), steps=10, out_traj="traj.dcd")
    assert math_isfinite(out["final_energy"])
    assert os.path.exists(out["minimized_gro"])
    assert os.path.exists(out["traj"])


def math_isfinite(x):
    return x == x and abs(x) != float("inf")
