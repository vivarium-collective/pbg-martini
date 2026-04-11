"""Tests for membrane and micelle builders and their PBG Step wrappers."""

import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_martini.processes import (
    MembraneBuilderStep,
    MicelleBuilderStep,
    ProteinMembraneStep,
    VesicleBuilderStep,
)
from pbg_martini.builders import (
    build_bilayer,
    build_micelle,
    build_protein_in_membrane,
    build_vesicle,
)


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('MembraneBuilderStep', MembraneBuilderStep)
    c.register_link('MicelleBuilderStep', MicelleBuilderStep)
    c.register_link('ProteinMembraneStep', ProteinMembraneStep)
    c.register_link('VesicleBuilderStep', VesicleBuilderStep)
    c.register_link('ram-emitter', RAMEmitter)
    return c


# ── Builder function tests ──────────────────────────────────────────

def test_bilayer_basic():
    result = build_bilayer({'POPC': 1.0}, nx_lipids=4, ny_lipids=4)
    assert result['stats']['n_lipids'] == 32  # 4x4x2 leaflets
    assert len(result['beads']) > 0
    assert len(result['bonds']) > 0


def test_bilayer_mixed_composition():
    result = build_bilayer(
        {'POPC': 0.5, 'CHOL': 0.5}, nx_lipids=6, ny_lipids=6,
    )
    lc = result['stats']['lipid_counts']
    assert 'POPC' in lc
    assert 'CHOL' in lc
    assert sum(lc.values()) == 72


def test_micelle_basic():
    result = build_micelle('DPC', n_lipids=20, radius=1.5)
    assert result['stats']['n_lipids'] == 20
    assert result['stats']['n_beads'] == 20 * 6  # DPC has 6 beads


def test_protein_in_membrane():
    result = build_protein_in_membrane(
        {'POPC': 1.0}, nx_lipids=6, ny_lipids=6,
        n_helix_residues=10, exclusion_radius=0.5,
    )
    assert result['stats']['n_protein_residues'] == 10
    assert result['stats']['n_protein_beads'] == 20  # 2 beads per residue
    assert result['stats']['n_beads'] > 20  # protein + some lipids


def test_vesicle_basic():
    result = build_vesicle(
        {'POPC': 1.0},
        n_lipids_outer=30, n_lipids_inner=20,
        outer_radius=3.0, inner_radius=2.0,
    )
    assert result['stats']['n_lipids'] == 50
    assert result['stats']['n_outer'] == 30
    assert result['stats']['n_inner'] == 20


def test_bilayer_bead_structure():
    result = build_bilayer({'POPC': 1.0}, nx_lipids=3, ny_lipids=3)
    for bead in result['beads']:
        assert 'pos' in bead
        assert 'color' in bead
        assert len(bead['pos']) == 3
        assert len(bead['color']) == 3
        assert 'label' in bead
        assert 'resname' in bead


# ── PBG Step tests ──────────────────────────────────────────────────

def test_membrane_step(core):
    step = MembraneBuilderStep(
        config={'nx': 4, 'ny': 4, 'composition': {'POPC': 1.0}},
        core=core,
    )
    result = step.update({})
    assert len(result['beads']) > 0
    assert result['stats']['n_lipids'] == 32


def test_micelle_step(core):
    step = MicelleBuilderStep(
        config={'lipid': 'DPC', 'n_lipids': 15, 'radius': 1.5},
        core=core,
    )
    result = step.update({})
    assert result['stats']['n_lipids'] == 15


def test_protein_membrane_step(core):
    step = ProteinMembraneStep(
        config={
            'composition': {'POPC': 1.0},
            'nx': 6, 'ny': 6,
            'n_helix_residues': 10,
        },
        core=core,
    )
    result = step.update({})
    assert result['stats']['n_protein_residues'] == 10


def test_vesicle_step(core):
    step = VesicleBuilderStep(
        config={
            'composition': {'POPC': 1.0},
            'n_outer': 25, 'n_inner': 15,
            'outer_radius': 3.0, 'inner_radius': 2.0,
        },
        core=core,
    )
    result = step.update({})
    assert result['stats']['n_lipids'] == 40
