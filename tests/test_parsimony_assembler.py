"""Tests for the parsimony -> Martini whole-cell assembler.

The offline core (Tasks 1, 4, 5, 6, 7) is fully exercised with a tiny
synthetic pack and stub CG templates; no network, no Rust binary, no OpenMM.
"""

import importlib.util
import json
import math
import os

import numpy as np
import pytest

from pbg_martini.parsimony_assembler import load_pack, select_slice

# Resolving from a cold cache falls through to the pbg-parsimony resolver; skip
# those tests where it isn't installed (the cache-first path is covered offline).
_HAS_PBG_PARSIMONY = importlib.util.find_spec("pbg_parsimony") is not None


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


# --------------------------------------------------------------------------
# Task 5: bentopy placement-JSON converter + render dispatch
# --------------------------------------------------------------------------

def test_converter_counts(tmp_path):
    from pbg_martini.parsimony_assembler import (
        to_bentopy_placements, CGTemplate, Placement, SliceSpec,
    )
    tpl = {"groel": CGTemplate("groel", "g.gro", "g.itp", np.zeros((10, 3)), 10)}
    sl = SliceSpec(by_species={"groel": [Placement(0, (10, 10, 10), (1, 0, 0, 0), 0)]})
    j = to_bentopy_placements(sl, tpl, box_nm=(20, 20, 20))
    seg = [s for s in j["placements"] if s["name"] == "groel"][0]
    assert len(seg["instances"]) == 1
    assert np.allclose(seg["instances"][0]["position"], [1.0, 1.0, 1.0])
    assert tuple(seg["instances"][0]["rotation"]) == (1, 0, 0, 0)
    assert seg["path"] == "g.gro"


def test_bentopy_available_is_bool():
    from pbg_martini.parsimony_assembler import bentopy_available
    assert isinstance(bentopy_available(), bool)


# --------------------------------------------------------------------------
# Task 6: pure-Python .gro/.top writers + stamp-all assembler
# --------------------------------------------------------------------------

def test_stamp_all_invariant(tmp_path):
    from pbg_martini.parsimony_assembler import (
        CGTemplate, Placement, SliceSpec, stamp_all, write_gro,
    )
    tpl = {"groel": CGTemplate("groel", "g.gro", "g.itp", np.zeros((10, 3)), 10),
           "EG10367-MONOMER": CGTemplate("EG10367-MONOMER", "e.gro", "e.itp", np.zeros((7, 3)), 7)}
    sl = SliceSpec(by_species={
        "groel": [Placement(0, (10, 0, 0), (1, 0, 0, 0), 0), Placement(0, (20, 0, 0), (1, 0, 0, 0), 1)],
        "EG10367-MONOMER": [Placement(1, (0, 0, 0), (1, 0, 0, 0), 2)]})
    coords, names, counts = stamp_all(sl, tpl)
    assert coords.shape[0] == 10 * 2 + 7 * 1          # n_beads invariant
    assert counts == {"groel": 2, "EG10367-MONOMER": 1}
    g = tmp_path / "s.gro"
    write_gro(g, names, coords, (30, 30, 30))
    lines = g.read_text().splitlines()
    assert int(lines[1].strip()) == coords.shape[0]  # GRO atom count line


def test_write_top(tmp_path):
    from pbg_martini.parsimony_assembler import write_top
    t = tmp_path / "s.top"
    write_top(t, itp_includes=["g.itp", "e.itp"], molecule_counts={"groel": 2, "EG10367-MONOMER": 1})
    txt = t.read_text()
    assert '#include "g.itp"' in txt
    assert "[ molecules ]" in txt
    assert "groel" in txt and "2" in txt


# --------------------------------------------------------------------------
# Task 7: assemble() orchestration against the real pack (integration)
# --------------------------------------------------------------------------

ECOLI_PACK = os.environ.get(
    "ECOLI_PACK",
    os.path.expanduser("~/code/3d-ecoli-app/data/ecoli_3d.pack.json"),
)

ALLOW_LIST = [
    "EG10367-MONOMER", "EG11036-MONOMER", "groel",
    "EG11384-MONOMER", "EG50003-MONOMER", "EG10669-MONOMER",
]


def _stub_templates(n_beads=5):
    from pbg_martini.parsimony_assembler import CGTemplate
    return {
        name: CGTemplate(name, f"{name}.gro", f"{name}.itp",
                         np.zeros((n_beads, 3)), n_beads)
        for name in ALLOW_LIST
    }


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists(ECOLI_PACK),
                    reason="real ecoli pack not present")
def test_assemble_real_pack_invariant(tmp_path):
    from pbg_martini.parsimony_assembler import assemble, load_pack, stamp
    nb = 5
    templates = _stub_templates(nb)
    # 1500 A sub-box centered at the cell center.
    box_min = (-750, -750, -750)
    box_max = (750, 750, 750)
    out = assemble(ECOLI_PACK, box_min, box_max, ALLOW_LIST,
                   templates, str(tmp_path), use_bentopy=False)

    assert out["n_molecules"] == sum(out["per_species_counts"].values())
    assert out["n_beads"] == sum(c * nb for c in out["per_species_counts"].values())
    assert out["n_molecules"] > 0
    assert os.path.exists(out["gro"])
    assert os.path.exists(out["top"])
    assert os.path.exists(out["placements_json"])

    # GRO atom-count line matches n_beads.
    lines = open(out["gro"]).read().splitlines()
    assert int(lines[1].strip()) == out["n_beads"]

    # A stamped groel instance's centroid matches its placement position / 10.
    pack = load_pack(ECOLI_PACK)
    name2id = {ing.name: ing.id for ing in pack.ingredients.values()}
    gid = name2id["groel"]
    groel = next(pl for pl in pack.placements
                 if pl.ingredient_id == gid
                 and all(box_min[a] <= pl.position[a] <= box_max[a] for a in range(3)))
    stamped = stamp(np.zeros((nb, 3)), groel.position, groel.rotation)
    centroid = stamped.mean(axis=0)
    expected = np.array(groel.position) / 10.0
    assert np.allclose(centroid, expected, atol=1e-6)


# --------------------------------------------------------------------------
# Task 2: structure resolution via pbg-parsimony (network)
# --------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(bool(os.environ.get("OFFLINE")),
                    reason="OFFLINE set; skipping network structure fetch")
@pytest.mark.skipif(not _HAS_PBG_PARSIMONY,
                    reason="pbg_parsimony not installed; cold-cache fetch unavailable")
def test_resolve_structure_groel(tmp_path):
    from pbg_martini.parsimony_assembler import resolve_structure
    path = resolve_structure("groel", cache_dir=str(tmp_path / "structures"))
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1024


@pytest.mark.network
@pytest.mark.skipif(bool(os.environ.get("OFFLINE")),
                    reason="OFFLINE set; skipping network structure fetch")
@pytest.mark.skipif(not _HAS_PBG_PARSIMONY,
                    reason="pbg_parsimony not installed; cold-cache fetch unavailable")
def test_resolve_structure_alphafold(tmp_path):
    from pbg_martini.parsimony_assembler import resolve_structure
    path = resolve_structure("EG10367-MONOMER", cache_dir=str(tmp_path / "structures"))
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1024


# --------------------------------------------------------------------------
# Task 3: CG template via martinize2 (slow, offline)
# --------------------------------------------------------------------------

# Small bundled peptide (mixed residues) — martinizes offline, fast-ish.
AAVLG_PDB = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.450   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.400   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.400   3.400   1.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       3.000   0.200   2.200  1.00  0.00           C
ATOM      6  N   ALA A   2       4.300   2.400   1.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       5.000   3.700   1.000  1.00  0.00           C
ATOM      8  C   ALA A   2       6.500   3.700   1.000  1.00  0.00           C
ATOM      9  O   ALA A   2       7.100   4.700   1.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       4.500   4.500   2.200  1.00  0.00           C
ATOM     11  N   VAL A   3       7.100   2.600   1.000  1.00  0.00           N
ATOM     12  CA  VAL A   3       8.500   2.500   1.000  1.00  0.00           C
ATOM     13  C   VAL A   3       9.100   3.800   1.000  1.00  0.00           C
ATOM     14  O   VAL A   3       8.500   4.800   1.000  1.00  0.00           O
ATOM     15  CB  VAL A   3       9.000   1.700   2.200  1.00  0.00           C
ATOM     16  CG1 VAL A   3      10.500   1.600   2.200  1.00  0.00           C
ATOM     17  CG2 VAL A   3       8.400   0.300   2.200  1.00  0.00           C
ATOM     18  N   LEU A   4      10.400   3.700   1.000  1.00  0.00           N
ATOM     19  CA  LEU A   4      11.200   4.900   1.000  1.00  0.00           C
ATOM     20  C   LEU A   4      12.700   4.800   1.000  1.00  0.00           C
ATOM     21  O   LEU A   4      13.300   5.800   1.000  1.00  0.00           O
ATOM     22  CB  LEU A   4      10.700   6.000   2.000  1.00  0.00           C
ATOM     23  CG  LEU A   4      11.200   7.400   2.000  1.00  0.00           C
ATOM     24  CD1 LEU A   4      10.600   8.100   3.200  1.00  0.00           C
ATOM     25  CD2 LEU A   4      12.700   7.500   2.000  1.00  0.00           C
ATOM     26  N   GLY A   5      13.300   3.600   1.000  1.00  0.00           N
ATOM     27  CA  GLY A   5      14.700   3.400   1.000  1.00  0.00           C
ATOM     28  C   GLY A   5      15.300   4.700   1.000  1.00  0.00           C
ATOM     29  O   GLY A   5      14.700   5.700   1.000  1.00  0.00           O
END
"""


@pytest.mark.slow
def test_martinize_species_template(tmp_path):
    from pbg_martini.parsimony_assembler import martinize_species
    pdb = tmp_path / "pep.pdb"
    pdb.write_text(AAVLG_PDB)
    tpl = martinize_species("testpep", str(pdb), str(tmp_path / "cg"), elastic=True)
    assert tpl.n_beads > 0
    assert tpl.beads_nm.shape == (tpl.n_beads, 3)
    # recentered to origin
    assert np.allclose(tpl.beads_nm.mean(axis=0), 0.0, atol=1e-6)
    assert os.path.exists(tpl.itp_path)
    itp = open(tpl.itp_path).read()
    assert "[ moleculetype ]" in itp
    assert os.path.exists(tpl.structure_path)
