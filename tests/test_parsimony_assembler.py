"""Tests for the parsimony -> Martini whole-cell assembler.

The offline core (Tasks 1, 4, 5, 6, 7) is fully exercised with a tiny
synthetic pack and stub CG templates; no network, no Rust binary, no OpenMM.
"""

import json
import math
import os

import numpy as np
import pytest

from pbg_martini.parsimony_assembler import load_pack, select_slice


# --------------------------------------------------------------------------
# Task 1: pack loader + slice selection
# --------------------------------------------------------------------------

def _synthetic_pack(tmp_path):
    pack = {
        "format": "parsimony.pack.v1",
        "bounds": {"min": [-100, -100, -100], "max": [100, 100, 100]},
        "ingredients": [
            {"id": 0, "name": "groel", "color": [1, 0, 0]},
            {"id": 1, "name": "EG10367-MONOMER", "color": [0, 1, 0]},
        ],
        "placements": [
            {"ingredient": 0, "position": [10, 10, 10], "rotation": [1, 0, 0, 0], "uid": 0, "compartment": 1},
            {"ingredient": 1, "position": [20, 0, 0], "rotation": [1, 0, 0, 0], "uid": 1, "compartment": 1},
            {"ingredient": 0, "position": [90, 90, 90], "rotation": [1, 0, 0, 0], "uid": 2, "compartment": 1},
            {"ingredient": 1, "position": [-90, 0, 0], "rotation": [1, 0, 0, 0], "uid": 3, "compartment": 1},
        ],
    }
    p = tmp_path / "syn.pack.json"
    p.write_text(json.dumps(pack))
    return p


def test_load_and_slice(tmp_path):
    pack = load_pack(_synthetic_pack(tmp_path))
    assert pack.ingredients[0].name == "groel"
    assert len(pack.placements) == 4
    sl = select_slice(pack, box_min=(0, -50, -50), box_max=(50, 50, 50),
                      species_names=["groel", "EG10367-MONOMER"])
    # only the two placements at (10,10,10) and (20,0,0) are in-box
    assert sum(len(v) for v in sl.by_species.values()) == 2
    assert len(sl.by_species["groel"]) == 1


# --------------------------------------------------------------------------
# Task 4: quaternion rotation + bead stamping (pure math)
# --------------------------------------------------------------------------

def test_stamp_identity_translates_only():
    from pbg_martini.parsimony_assembler import stamp
    beads = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    out = stamp(beads, position_A=(50, 0, 0), rotation=(1, 0, 0, 0))  # A->nm: 5.0
    assert np.allclose(out, beads + np.array([5.0, 0, 0]))


def test_stamp_90deg_z():
    from pbg_martini.parsimony_assembler import stamp
    beads = np.array([[1.0, 0, 0]])
    q = (math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))  # 90 deg about z, (w,x,y,z)
    out = stamp(beads, position_A=(0, 0, 0), rotation=q)
    assert np.allclose(out, np.array([[0, 1.0, 0]]), atol=1e-6)


def test_quat_to_matrix_orthonormal():
    from pbg_martini.parsimony_assembler import quat_to_matrix
    q = (0.5, 0.5, 0.5, 0.5)
    R = quat_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
