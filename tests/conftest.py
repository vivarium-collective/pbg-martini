"""Shared fixtures for pbg-martini tests."""

import pytest
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_martini.processes import MartinizeStep


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('MartinizeStep', MartinizeStep)
    c.register_link('ram-emitter', RAMEmitter)
    return c


# Tri-alanine — minimal peptide (3 residues, 15 atoms → 6 CG beads)
TRI_ALA_PDB = """\
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
ATOM     11  N   ALA A   3       7.100   2.600   1.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       8.500   2.500   1.000  1.00  0.00           C
ATOM     13  C   ALA A   3       9.100   3.800   1.000  1.00  0.00           C
ATOM     14  O   ALA A   3       8.500   4.800   1.000  1.00  0.00           O
ATOM     15  CB  ALA A   3       9.000   1.700   2.200  1.00  0.00           C
END
"""

# Penta-peptide AAVLG — mixed residues (5 residues, 29 atoms → 9 CG beads)
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
