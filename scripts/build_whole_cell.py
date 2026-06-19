"""Assemble the full DRY whole-cell Martini system from parsimony positions.

Stamps every resolvable species' CG template at ALL its parsimony
positions/orientations -> one ``system.gro`` (~120M beads) + ``system.top``.
Streams the .gro record-by-record so memory stays bounded (the coordinate file
is ~5-7 GB; never held in RAM at once). Intended to run on the HPC node that
will also run the MD.

Skips: lipid is kept as a 1-bead proxy; multi-chain complexes (groel,
rna_polymerase) and the ribosome (rRNA) are skipped pending proper handling.

Usage (on the HPC):
    python build_whole_cell.py --pack ecoli_3d.pack.json \
        --templates templates --out . [--max-placements N]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from pbg_martini.parsimony_assembler import load_pack, stamp, quat_to_matrix  # noqa: F401
from pbg_martini.visualizations import _read_gro
from pbg_martini.dna_segment import dna_segment_template


def _moltype_and_beads(itp_path, gro_path):
    """(moleculetype name, n_atoms, beads_nm) from a template's itp+gro."""
    name = None
    section = want = None
    for ln in open(itp_path):
        s = ln.strip()
        if s.startswith("["):
            section = s.strip("[] ").lower(); want = (section == "moleculetype"); continue
        if not s or s.startswith(";"):
            continue
        if want:
            name = s.split()[0]; break
    beads, _, _ = _read_gro(gro_path)
    return name, beads.shape[0], beads


def _gro_record(resid, resn, aname, idx, xyz):
    return "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (
        resid % 100000, resn[:5], aname[:5], idx % 100000, xyz[0], xyz[1], xyz[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--ff", default="martini_v3.0.0.itp")
    ap.add_argument("--out", default=".")
    ap.add_argument("--max-placements", type=int, default=0,
                    help="cap total stamped molecules (0 = all; for testing)")
    a = ap.parse_args()

    pack = load_pack(a.pack)
    by_species = {}
    for p in pack.placements:
        by_species.setdefault(pack.ingredients[p.ingredient_id].name, []).append(p)

    # resolve a template (moltype, natoms, beads_nm) per species
    DNA = dna_segment_template()
    templates, order, total_beads = {}, [], 0
    for name, places in sorted(by_species.items(), key=lambda kv: -len(kv[1])):
        if name == "lipid":
            continue   # envelope membrane: needs a real Martini bilayer, not a proxy
        if name == "dna_segment":
            dna_segment_template(out_itp=os.path.join(a.templates, "DNA_SEG.itp"))
            templates[name] = ("DNA_SEG", DNA.n_beads, DNA.beads_nm); order.append(name); continue
        gro = os.path.join(a.templates, f"{name}.gro")
        itp = os.path.join(a.templates, f"{name}.itp")
        if not (os.path.exists(gro) and os.path.exists(itp)):
            continue
        mt, nat, beads = _moltype_and_beads(itp, gro)
        if nat != beads.shape[0] or not np.isfinite(beads).all():
            continue   # multi-chain NaN / ribosome -> skip
        templates[name] = (mt, nat, beads); order.append(name)

    # box from pack bounds (nm) + margin
    bmin = np.array(pack.bounds["min"]) / 10.0
    bmax = np.array(pack.bounds["max"]) / 10.0
    box = tuple((bmax - bmin) + 2.0)
    shift = -bmin + 1.0

    # count total beads (respecting cap)
    counts = {}
    budget = a.max_placements or 10**18
    for name in order:
        n = min(len(by_species[name]), budget)
        if n <= 0:
            break
        counts[name] = n
        total_beads += n * templates[name][1]
        budget -= n

    os.makedirs(a.out, exist_ok=True)
    gro_path = os.path.join(a.out, "system.gro")
    print(f"streaming {total_beads:,} beads over {sum(counts.values()):,} molecules "
          f"-> {gro_path}  box {box[0]:.0f}x{box[1]:.0f}x{box[2]:.0f} nm")

    idx = 0; resid = 0
    with open(gro_path, "w") as g:
        g.write("parsimony -> Martini whole-cell E. coli (dry)\n")
        g.write(f"{total_beads}\n")
        for name in order:
            if name not in counts:
                continue
            mt, nb, beads = templates[name]
            for p in by_species[name][:counts[name]]:
                xyz = stamp(beads, p.position, p.rotation) + shift
                resid += 1
                for b in range(nb):
                    idx += 1
                    g.write(_gro_record(resid, mt, mt, idx, xyz[b]))
        g.write("%10.5f%10.5f%10.5f\n" % box)

    # topology
    itps = [a.ff]
    for name in order:
        if name in counts and name not in ("lipid",):
            itp = os.path.join(a.templates, f"{name}.itp") if name != "dna_segment" else "DNA_SEG.itp"
            base = os.path.basename(itp)
            if base not in itps:
                itps.append(base)
    top_path = os.path.join(a.out, "system.top")
    with open(top_path, "w") as t:
        for inc in itps:
            t.write(f'#include "{inc}"\n')
        t.write("\n[ system ]\nparsimony whole-cell E. coli (dry)\n\n[ molecules ]\n")
        for name in order:
            if name in counts:
                t.write(f"{templates[name][0]} {counts[name]}\n")
    print(f"wrote {top_path}  ({len(counts)} species)")


if __name__ == "__main__":
    main()
