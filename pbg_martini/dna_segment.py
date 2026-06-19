"""Coarse CG model of a chromosome ``dna_segment`` for crowded Martini cells.

martinize2 does not coarse-grain nucleic acids (it returns 0 beads for a DNA
PDB), and a full Martini-2 dsDNA model via polyply mixes force-field versions
with the Martini 3 proteins. For a *crowded-environment* simulation -- where the
chromosome's role is excluded volume plus the phosphate-backbone charge that
sets up the counter-ion atmosphere -- we use a deliberately coarse model:

    one charged Martini bead per backbone phosphate of a B-DNA dodecamer (1BNA),
    held together by an elastic network.

This is **not** an atomistic Martini double helix; it is a labelled coarse
nucleoid segment. It reproduces the segment's size, shape, and ~-1/nucleotide
charge, which is what matters for crowding and ion distribution.
"""

from __future__ import annotations

import os

import numpy as np

from .parsimony_assembler import CGTemplate

# Default B-DNA dodecamer used by the parsimony ecoli pack for dna_segment.
_DEFAULT_1BNA = os.path.expanduser(
    "~/code/parsimony/examples/pdb_cache/1bna.pdb")

DNA_BEAD_TYPE = "Q5n"   # Martini 3 charged (negative) bead
DNA_BEAD_CHARGE = -1.0  # per phosphate
ELASTIC_CUTOFF_NM = 1.0
ELASTIC_K = 1250.0      # kJ/mol/nm^2


def _phosphate_coords(pdb_path):
    """Backbone phosphate (P) atom coordinates (Angstrom) from a DNA PDB."""
    coords = []
    for ln in open(pdb_path):
        if ln.startswith(("ATOM", "HETATM")) and ln[12:16].strip() == "P":
            coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return np.asarray(coords, dtype=float)


def dna_segment_template(pdb_path: str | None = None,
                         out_itp: str | None = None,
                         moltype: str = "DNA_SEG") -> CGTemplate:
    """Build a coarse CG :class:`CGTemplate` for one chromosome segment.

    One ``Q5n`` (-1) bead per backbone phosphate, origin-centred, with an
    elastic network over phosphates within :data:`ELASTIC_CUTOFF_NM`. Writes the
    ``.itp`` to ``out_itp`` if given.
    """
    pdb = pdb_path or _DEFAULT_1BNA
    p_ang = _phosphate_coords(pdb)
    if p_ang.shape[0] == 0:
        raise ValueError(f"no phosphate (P) atoms found in {pdb}")
    beads_nm = (p_ang - p_ang.mean(0)) / 10.0   # center, Angstrom -> nm
    n = beads_nm.shape[0]

    # elastic network: harmonic bonds between phosphates within the cutoff
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(beads_nm[i] - beads_nm[j]))
            if d <= ELASTIC_CUTOFF_NM:
                bonds.append((i + 1, j + 1, d))

    lines = ["[ moleculetype ]", f"{moltype} 1", "", "[ atoms ]"]
    for i in range(n):
        lines.append(
            f" {i+1} {DNA_BEAD_TYPE} 1 DNA P{i+1} {i+1} {DNA_BEAD_CHARGE}")
    lines += ["", "[ bonds ]"]
    for i, j, d in bonds:
        lines.append(f" {i} {j} 1 {d:.4f} {ELASTIC_K:.1f}")
    itp_text = "\n".join(lines) + "\n"

    if out_itp:
        with open(out_itp, "w") as fh:
            fh.write(itp_text)

    return CGTemplate(
        name=moltype,
        structure_path=out_itp or "",
        itp_path=out_itp or "",
        beads_nm=beads_nm,
        n_beads=n,
    )
