"""Integration tests for composite assembly."""

import pytest
from process_bigraph import Composite
from pbg_martini.composites import make_martinize_document
from tests.conftest import TRI_ALA_PDB, AAVLG_PDB


def test_document_structure():
    doc = make_martinize_document(pdb_text=TRI_ALA_PDB)
    assert 'martinize' in doc
    assert 'stores' in doc
    assert 'emitter' in doc
    assert doc['martinize']['_type'] == 'step'
    assert doc['stores']['pdb_text'] == TRI_ALA_PDB


def test_composite_assembly(core):
    doc = make_martinize_document(pdb_text=TRI_ALA_PDB)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_composite_run(core):
    doc = make_martinize_document(pdb_text=TRI_ALA_PDB)
    sim = Composite({'state': doc}, core=core)
    sim.run(0)
    stores = sim.state['stores']
    assert stores['n_beads'] == 6
    assert stores['n_bonds_cg'] == 5
    assert len(stores['cg_beads']) == 6
    assert stores['reduction_ratio'] > 1.0


def test_composite_mixed_peptide(core):
    doc = make_martinize_document(pdb_text=AAVLG_PDB)
    sim = Composite({'state': doc}, core=core)
    sim.run(0)
    stores = sim.state['stores']
    assert stores['n_beads'] == 9
    assert len(stores['cg_positions']) == 9


def test_factory_params():
    doc = make_martinize_document(
        pdb_text=TRI_ALA_PDB,
        from_ff='charmm',
        to_ff='martini22',
        delete_unknown=False,
        ignh=True,
    )
    cfg = doc['martinize']['config']
    assert cfg['from_ff'] == 'charmm'
    assert cfg['to_ff'] == 'martini22'
    assert cfg['delete_unknown'] is False
    assert cfg['ignh'] is True
