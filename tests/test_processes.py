"""Unit tests for MartinizeStep."""

import pytest
from pbg_martini.processes import MartinizeStep, run_martinize_pipeline
from tests.conftest import TRI_ALA_PDB, AAVLG_PDB


def test_instantiation(core):
    step = MartinizeStep(config={}, core=core)
    assert step.config['to_ff'] == 'martini3001'
    assert step.config['from_ff'] == 'charmm'
    assert step.config['delete_unknown'] is True


def test_custom_config(core):
    step = MartinizeStep(
        config={'to_ff': 'martini22', 'ignh': True}, core=core,
    )
    assert step.config['to_ff'] == 'martini22'
    assert step.config['ignh'] is True


def test_ports(core):
    step = MartinizeStep(config={}, core=core)
    inp = step.inputs()
    out = step.outputs()
    assert 'pdb_text' in inp
    assert 'cg_beads' in out
    assert 'cg_positions' in out
    assert 'n_beads' in out
    assert 'reduction_ratio' in out


def test_update_tri_ala(core):
    step = MartinizeStep(config={}, core=core)
    result = step.update({'pdb_text': TRI_ALA_PDB})

    assert result['n_atoms_input'] == 15
    assert result['n_beads'] == 6
    assert result['n_bonds_cg'] == 5
    assert result['reduction_ratio'] > 1.0
    assert len(result['cg_beads']) == 6
    assert len(result['cg_positions']) == 6
    assert len(result['cg_bonds']) == 5

    # All beads should be ALA
    for bead in result['cg_beads']:
        assert bead['resname'] == 'ALA'

    # Positions should be 3D
    for pos in result['cg_positions']:
        assert len(pos) == 3


def test_update_mixed_peptide(core):
    step = MartinizeStep(config={}, core=core)
    result = step.update({'pdb_text': AAVLG_PDB})

    assert result['n_beads'] == 9
    residues = {b['resname'] for b in result['cg_beads']}
    assert residues == {'ALA', 'VAL', 'LEU', 'GLY'}
    assert 'bonds' in result['interactions'] or 'constraints' in result['interactions']


def test_bead_type_counts(core):
    step = MartinizeStep(config={}, core=core)
    result = step.update({'pdb_text': TRI_ALA_PDB})
    btc = result['bead_type_counts']
    assert sum(btc.values()) == 6
    assert 'SP2' in btc  # ALA backbone type in martini3001


def test_pipeline_function_directly():
    result = run_martinize_pipeline(pdb_text=TRI_ALA_PDB)
    assert result['n_beads'] == 6
    assert result['n_bonds_cg'] == 5
    assert isinstance(result['cg_positions'], list)


def test_martini22_target(core):
    step = MartinizeStep(config={'to_ff': 'martini22'}, core=core)
    result = step.update({'pdb_text': TRI_ALA_PDB})
    assert result['n_beads'] > 0
    assert result['n_bonds_cg'] > 0
