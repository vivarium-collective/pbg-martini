"""Coarse rigid-bead CG model for structures martinize2 cannot handle.

Some components either are not proteins (the ribosome is ~70% rRNA) or have
structures martinize2 maps to NaN (e.g. the GroEL 1AON assembly). For a crowded
whole-cell model their essential role is **excluded volume and overall shape**,
so we build a deliberately coarse model: voxel-downsample the heavy atoms to a
set of beads, assign a single Martini bead type, and tie them with an elastic
network. Honestly labelled as coarse (not an atomistic Martini mapping).
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from .parsimony_assembler import CGTemplate, write_gro


def _read_heavy_atoms(path):
    """Heavy-atom coordinates (Angstrom) from a PDB or mmCIF file."""
    coords = []
    if path.lower().endswith(".cif"):
        # mmCIF _atom_site loop: find the column order, then read rows.
        cols, in_loop, header = {}, False, []
        for ln in open(path):
            s = ln.strip()
            if s.startswith("loop_"):
                in_loop, header = True, []
                continue
            if in_loop and s.startswith("_atom_site."):
                header.append(s)
                continue
            if header and (s.startswith("ATOM") or s.startswith("HETATM")):
                if not cols:
                    cols = {name.split(".")[1]: i for i, name in enumerate(header)}
                p = s.split()
                try:
                    el = p[cols.get("type_symbol", 2)]
                    if el == "H":
                        continue
                    coords.append((float(p[cols["Cartn_x"]]), float(p[cols["Cartn_y"]]),
                                   float(p[cols["Cartn_z"]])))
                except (KeyError, ValueError, IndexError):
                    continue
            elif header and s and not s.startswith("_") and not (s.startswith("ATOM") or s.startswith("HETATM")):
                if cols:
                    break
    else:
        for ln in open(path):
            if ln.startswith(("ATOM", "HETATM")):
                if ln[76:78].strip() == "H" or ln[12:16].strip().startswith("H"):
                    continue
                resn = ln[17:20].strip()
                if resn == "HOH":
                    continue
                try:
                    coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
                except ValueError:
                    continue
    return np.asarray(coords, dtype=float)


def coarse_structure_template(struct_path, name, out_dir, voxel_nm=1.2,
                              bead_type="P4", net_charge=0.0,
                              elastic_cutoff_nm=2.0, elastic_k=700.0) -> CGTemplate:
    """Voxel-downsample a structure into a coarse elastic-network CG template."""
    os.makedirs(out_dir, exist_ok=True)
    atoms = _read_heavy_atoms(struct_path)
    if atoms.shape[0] == 0:
        raise ValueError(f"no heavy atoms read from {struct_path}")

    # voxel grid -> one bead per occupied cell (centroid of its atoms)
    v = voxel_nm * 10.0  # Angstrom
    keys = np.floor((atoms - atoms.min(0)) / v).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, atoms = keys[order], atoms[order]
    cell, idx = np.unique(keys, axis=0, return_inverse=True)
    beads_ang = np.zeros((cell.shape[0], 3))
    for k in range(cell.shape[0]):
        beads_ang[k] = atoms[idx == k].mean(0)
    beads_nm = (beads_ang - beads_ang.mean(0)) / 10.0
    n = beads_nm.shape[0]
    q = net_charge / n if (net_charge and n) else 0.0

    bonds = []
    if n > 1:
        for i, j in cKDTree(beads_nm).query_pairs(elastic_cutoff_nm, output_type="ndarray"):
            bonds.append((i + 1, j + 1, float(np.linalg.norm(beads_nm[i] - beads_nm[j]))))

    out = ["[ moleculetype ]", f"{name} 1", "", "[ atoms ]"]
    for i in range(n):
        out.append(f" {i+1} {bead_type} 1 {name[:3].upper()} B{i+1} {i+1} {q:.4f}")
    out += ["", "[ bonds ]"]
    for i, j, d in bonds:
        out.append(f" {i} {j} 1 {d:.4f} {elastic_k:.1f}")
    itp_path = os.path.join(out_dir, f"{name}.itp")
    open(itp_path, "w").write("\n".join(out) + "\n")

    extent = (beads_nm.max(0) - beads_nm.min(0)) + 2.0
    write_gro(os.path.join(out_dir, f"{name}.gro"), [f"B{i+1}"[:5] for i in range(n)],
              beads_nm, tuple(float(max(e, 1.0)) for e in extent))
    return CGTemplate(name, os.path.join(out_dir, f"{name}.gro"), itp_path, beads_nm, n)
