"""Martinize multi-chain protein complexes (GroEL, RNA polymerase, ...).

martinize2 emits NaN coordinates when handed a whole multi-chain assembly, so we
coarse-grain **each chain separately** (which is clean) and merge the results
into a single moleculetype: per-chain atoms + intra-chain bonds (re-indexed),
plus an **inter-chain elastic network** (harmonic bonds between nearby beads on
different chains) so the complex holds its quaternary structure under MD.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from .parsimony_assembler import CGTemplate, write_gro
from .processes import run_martinize_pipeline

INTER_CUTOFF_NM = 0.9
INTER_K = 500.0


def _chains(pdb_path):
    chains = {}
    for ln in open(pdb_path):
        if ln.startswith("ATOM"):
            chains.setdefault(ln[21], []).append(ln)
    return chains


def _parse_itp(itp_text):
    """Return (atoms, bonds): atoms=[(type,res,atom,charge)], bonds=[(i,j,rest)]."""
    atoms, bonds, sec = [], [], None
    for ln in itp_text.splitlines():
        s = ln.strip()
        if s.startswith("["):
            sec = s.strip("[] ").lower(); continue
        if not s or s.startswith(";") or s.startswith("#"):  # skip preprocessor (#ifdef)
            continue
        p = s.split()
        if sec == "atoms" and len(p) >= 7 and p[0].isdigit():
            atoms.append((p[1], p[3], p[4], p[6]))         # type, res, atom, charge
        elif sec == "bonds" and len(p) >= 2 and p[0].isdigit() and p[1].isdigit():
            bonds.append((int(p[0]), int(p[1]), " ".join(p[2:])))
    return atoms, bonds


def martinize_multichain(pdb_path, name, out_dir,
                         inter_cutoff_nm=INTER_CUTOFF_NM, inter_k=INTER_K) -> CGTemplate:
    """Coarse-grain a multi-chain complex into one elastic-network moleculetype."""
    os.makedirs(out_dir, exist_ok=True)
    all_pos, all_atoms, all_bonds, ranges, off = [], [], [], [], 0
    for ch, lines in _chains(pdb_path).items():
        txt = "".join(lines) + "END\n"
        try:
            r = run_martinize_pipeline(txt, return_itp=True,
                                       moltype_name=f"{name}_{ch}")
        except Exception:
            continue
        pos = np.asarray(r["cg_positions"], dtype=float)
        atoms, bonds = _parse_itp(r.get("itp_text") or "")
        if pos.size == 0 or not np.isfinite(pos).all() or len(atoms) != pos.shape[0]:
            continue
        all_pos.append(pos)
        all_atoms.extend(atoms)
        for i, j, rest in bonds:
            all_bonds.append((i + off, j + off, rest))
        ranges.append((off, off + pos.shape[0]))
        off += pos.shape[0]

    if not all_pos:
        raise ValueError(f"no chains martinized for {name}")
    positions = np.concatenate(all_pos)
    centroid = positions.mean(axis=0)
    beads_nm = positions - centroid
    n = positions.shape[0]

    # chain id per bead, then inter-chain elastic bonds within cutoff
    cid = np.zeros(n, dtype=int)
    for k, (s, e) in enumerate(ranges):
        cid[s:e] = k
    inter = []
    if n > 1:
        pairs = cKDTree(positions).query_pairs(inter_cutoff_nm, output_type="ndarray")
        for i, j in pairs:
            if cid[i] != cid[j]:
                d = float(np.linalg.norm(positions[i] - positions[j]))
                inter.append((i + 1, j + 1, d))

    out = ["[ moleculetype ]", f"{name} 1", "", "[ atoms ]"]
    for idx, (typ, res, atom, charge) in enumerate(all_atoms):
        out.append(f" {idx+1} {typ} {idx+1} {res} {atom} {idx+1} {charge}")
    out += ["", "[ bonds ]"]
    for i, j, rest in all_bonds:
        out.append(f" {i} {j} {rest}")
    out.append("; inter-chain elastic network")
    for i, j, d in inter:
        out.append(f" {i} {j} 1 {d:.4f} {inter_k:.1f}")
    itp_text = "\n".join(out) + "\n"

    itp_path = os.path.join(out_dir, f"{name}.itp")
    open(itp_path, "w").write(itp_text)
    extent = (beads_nm.max(0) - beads_nm.min(0)) + 2.0
    write_gro(os.path.join(out_dir, f"{name}.gro"),
              [a[2][:5] for a in all_atoms], beads_nm,
              tuple(float(max(e, 1.0)) for e in extent))
    return CGTemplate(name, os.path.join(out_dir, f"{name}.gro"), itp_path, beads_nm, n)
