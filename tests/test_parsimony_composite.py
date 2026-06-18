"""Task 10: PBG Steps + the parsimony-whole-cell composite."""

import json
import os

from process_bigraph import Composite
from process_bigraph.emitter import RAMEmitter

from pbg_martini.composites import (
    register_martini, build_composite, list_composite_specs,
)


def _synthetic_pack(tmp_path):
    pack = {
        "format": "parsimony.pack.v1",
        "bounds": {"min": [-1000, -1000, -1000], "max": [1000, 1000, 1000]},
        "ingredients": [
            {"id": 0, "name": "groel", "color": [1, 0, 0]},
            {"id": 1, "name": "EG10367-MONOMER", "color": [0, 1, 0]},
        ],
        "placements": [
            {"ingredient": 0, "position": [10, 10, 10], "rotation": [1, 0, 0, 0], "uid": 0},
            {"ingredient": 1, "position": [20, 0, 0], "rotation": [1, 0, 0, 0], "uid": 1},
            {"ingredient": 1, "position": [-30, 0, 0], "rotation": [1, 0, 0, 0], "uid": 2},
        ],
    }
    p = tmp_path / "syn.pack.json"
    p.write_text(json.dumps(pack))
    return str(p)


def test_assemble_step_direct(tmp_path):
    from pbg_martini.processes import ParsimonyAssembleStep
    pack = _synthetic_pack(tmp_path)
    core = register_martini()
    step = ParsimonyAssembleStep({
        'pack_path': pack,
        'box_min': [-100.0, -100.0, -100.0],
        'box_max': [100.0, 100.0, 100.0],
        'species': ['groel', 'EG10367-MONOMER'],
        'out_dir': str(tmp_path / "out"),
        'stub_beads': 5,
        'use_bentopy': False,
    }, core=core)
    out = step.update({})
    assert out['n_molecules'] == 3
    assert out['n_beads'] == 3 * 5
    assert os.path.exists(out['gro'])


def test_spec_is_discoverable():
    assert "parsimony-whole-cell" in list_composite_specs()


def test_composite_runs_single_update(tmp_path):
    pack = _synthetic_pack(tmp_path)
    core = register_martini()
    composite = build_composite(
        "parsimony-whole-cell",
        overrides={
            "pack_path": pack,
            "out_dir": str(tmp_path / "slice"),
            "stub_beads": 5,
            "relax_steps": 20,
        },
        core=core,
    )
    composite.update({}, 1)

    state = composite.state
    gro = state["stores"]["gro"]
    assert gro and os.path.exists(gro)
    assert state["stores"]["n_beads"] == 3 * 5
    assert state["stores"]["n_molecules"] == 3
