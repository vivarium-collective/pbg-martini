"""Build the biggest tractable crowded Martini chunk of the parsimony E. coli.

Takes a cubic chunk of the parsimony pack (cytoplasm + chromosome), coarse-grains
every protein species present with martinize2 (real Martini 3), models each
``dna_segment`` as a coarse charged phosphate-backbone chain, stamps them ALL at
their parsimony positions/orientations, then solvates with Martini water + ions
(neutralised, ~150 mM NaCl). Writes ``system.gro`` + ``system.top`` + the itp set
to ``--out`` for pbg-openmm to run.

Membrane lipids (envelope) and multi-chain complexes whose stamped bead count
doesn't match martinize2's moleculetype (e.g. GroEL, RNA polymerase, ribosome)
are skipped from the runnable system and reported.

Usage::

    ECOLI_PACK=~/code/3d-ecoli-app/data/ecoli_3d.pack.json \
      .venv/bin/python scripts/build_crowded_chunk.py --edge 600 --out output/chunk
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

from pbg_martini.parsimony_assembler import (
    load_pack, resolve_structure, martinize_species, stamp, write_gro,
)
from pbg_martini.dna_segment import dna_segment_template
from pbg_martini.solvate import solvate

FF_ITP = os.path.expanduser("~/code/pbg-openmm/pbg_openmm/data/martini_v3.0.0.itp")
SOLVENT_ITP = os.path.expanduser(
    "~/code/pbg-openmm/pbg_openmm/data/martini_v3.0.0_solvents_v1.itp")
ION_ITP = os.path.expanduser(
    "~/code/pbg-openmm/pbg_openmm/data/martini_v3.0.0_ions_v1.itp")

EXCLUDE = {"lipid"}  # envelope membrane: not part of a cytoplasm+nucleoid chunk


def _load_cached_template(name, tpl_dir):
    """Reuse a previously martinized template (``<name>.gro`` + ``.itp``)."""
    from pbg_martini.parsimony_assembler import CGTemplate
    from pbg_martini.visualizations import _read_gro
    gro = os.path.join(tpl_dir, f"{name}.gro")
    itp = os.path.join(tpl_dir, f"{name}.itp")
    if os.path.exists(gro) and os.path.exists(itp):
        beads, _, _ = _read_gro(gro)
        return CGTemplate(name, gro, itp, beads, beads.shape[0])
    return None


def _itp_moltype(itp_path):
    """Return (moleculetype_name, n_atoms, total_charge) from an itp."""
    name, natoms, charge = None, 0, 0.0
    section, want_name = None, False
    for ln in open(itp_path):
        s = ln.strip()
        if s.startswith("["):
            section = s.strip("[] ").lower()
            want_name = (section == "moleculetype")
            continue
        if not s or s.startswith(";"):
            continue
        if want_name and section == "moleculetype":
            name = s.split()[0]
            want_name = False
            continue
        if section == "atoms":
            natoms += 1
            parts = s.split()
            if len(parts) >= 7:
                try:
                    charge += float(parts[6])
                except ValueError:
                    pass
    return name, natoms, charge


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=os.environ.get(
        "ECOLI_PACK", os.path.expanduser("~/code/3d-ecoli-app/data/ecoli_3d.pack.json")))
    ap.add_argument("--center", type=float, nargs=3, default=None,
                    help="cube center (A); default = placement centroid")
    ap.add_argument("--edge", type=float, default=600.0, help="cube edge (A)")
    ap.add_argument("--out", default="output/chunk")
    ap.add_argument("--ion-conc", type=float, default=0.15)
    ap.add_argument("--no-solvate", action="store_true")
    args = ap.parse_args()

    out = args.out
    tpl_dir = os.path.join(out, "templates")
    os.makedirs(tpl_dir, exist_ok=True)

    pack = load_pack(args.pack)
    pos = np.array([p.position for p in pack.placements])
    c = np.array(args.center) if args.center else pos.mean(0)
    h = args.edge / 2.0
    box_min, box_max = c - h, c + h
    box_nm = (args.edge / 10.0,) * 3

    # group in-cube placements by species
    by_species = {}
    for p in pack.placements:
        name = pack.ingredients[p.ingredient_id].name
        if name in EXCLUDE:
            continue
        if np.all(np.abs(np.array(p.position) - c) <= h):
            by_species.setdefault(name, []).append(p)
    print(f"cube edge {args.edge} A @ {c.round(0)}: "
          f"{sum(len(v) for v in by_species.values())} placements, "
          f"{len(by_species)} species")

    # build templates + itps
    templates, moltypes, used, skipped = {}, {}, [], []
    for name, places in sorted(by_species.items(), key=lambda kv: -len(kv[1])):
        if name == "dna_segment":
            itp = os.path.join(tpl_dir, "DNA_SEG.itp")
            tpl = dna_segment_template(out_itp=itp, moltype="DNA_SEG")
            templates[name] = tpl
            moltypes[name] = ("DNA_SEG", itp)
            used.append((name, len(places), tpl.n_beads))
            continue
        cached = _load_cached_template(name, tpl_dir)
        if cached is not None:
            tpl = cached
        else:
            try:
                pdb = resolve_structure(name)
                tpl = martinize_species(name, pdb, tpl_dir, elastic=True)
            except Exception as exc:
                skipped.append((name, len(places), f"resolve/martinize: {type(exc).__name__}"))
                continue
        mt, natoms, _ = _itp_moltype(tpl.itp_path)
        if natoms != tpl.n_beads:   # multi-chain mismatch -> skip from run
            skipped.append((name, len(places), f"gro/top mismatch ({tpl.n_beads}!={natoms})"))
            continue
        if not np.isfinite(tpl.beads_nm).all():   # martinize2 NaN (multi-chain complexes)
            skipped.append((name, len(places), "martinize NaN coords"))
            continue
        templates[name] = tpl
        moltypes[name] = (mt, tpl.itp_path)
        used.append((name, len(places), tpl.n_beads))

    # stamp everything at parsimony positions
    all_coords, all_names, mol_list, net_charge = [], [], [], 0.0
    for name, count, nb in used:
        tpl = templates[name]
        mt, itp = moltypes[name]
        _, _, q = _itp_moltype(itp)
        for p in by_species[name]:
            beads = stamp(tpl.beads_nm, p.position, p.rotation)
            all_coords.append(beads)
            all_names.extend([mt[:5]] * tpl.n_beads)
            net_charge += q
        mol_list.append((mt, count, itp))
    solute = np.concatenate(all_coords) if all_coords else np.zeros((0, 3))
    # Pad the box to FULLY contain the solute (molecules at the cube faces stick
    # out past edge/10); otherwise PBC wraps them into grid water -> clashes ->
    # NaN. Anchor solute at +margin so nothing sits outside [0, box].
    margin = 1.0  # nm
    mn, mx = solute.min(0), solute.max(0)
    solute = solute - mn + margin
    box_nm = tuple((mx - mn) + 2 * margin)
    print(f"solute: {solute.shape[0]} beads, net charge {net_charge:.0f}, "
          f"box {box_nm[0]:.1f}x{box_nm[1]:.1f}x{box_nm[2]:.1f} nm")
    print("  used species:", len(used), "| skipped:", len(skipped))
    for n, k, why in skipped:
        print(f"    skip {n} (x{k}): {why}")

    # solvate
    extra_mols = []
    coords = solute
    names = list(all_names)
    if not args.no_solvate:
        sol = solvate(solute, box_nm, net_charge=net_charge,
                      ion_conc_M=args.ion_conc, exclusion_nm=0.50)
        print(f"  solvent: {sol['n_water']} W, {sol['n_na']} NA, {sol['n_cl']} CL")
        for arr, resn, n in ((sol["water"], "W", sol["n_water"]),
                             (sol["na"], "NA", sol["n_na"]),
                             (sol["cl"], "CL", sol["n_cl"])):
            if n:
                coords = np.concatenate([coords, arr])
                names.extend([resn] * n)
        extra_mols = [("W", sol["n_water"], SOLVENT_ITP),
                      ("NA", sol["n_na"], ION_ITP),
                      ("CL", sol["n_cl"], ION_ITP)]

    # write gro
    gro = os.path.join(out, "system.gro")
    write_gro(gro, names, coords, box_nm)

    # collect unique itps for includes (preserve order: FF, proteins, DNA, solvent, ions)
    includes, seen = [FF_ITP], {FF_ITP}
    for _, _, itp in mol_list + extra_mols:
        if itp and itp not in seen:
            includes.append(itp); seen.add(itp)
    # copy local itps next to the top for portability
    local_inc = []
    for itp in includes:
        dst = os.path.join(out, os.path.basename(itp))
        if os.path.abspath(itp) != os.path.abspath(dst):
            shutil.copy(itp, dst)
        local_inc.append(os.path.basename(itp))

    top = os.path.join(out, "system.top")
    with open(top, "w") as f:
        for inc in local_inc:
            f.write(f'#include "{inc}"\n')
        f.write("\n[ system ]\nparsimony crowded E. coli chunk (cytoplasm + chromosome)\n\n[ molecules ]\n")
        for mt, count, _ in mol_list:
            f.write(f"{mt} {count}\n")
        for resn, n, _ in extra_mols:
            if n:
                f.write(f"{resn} {n}\n")

    print(f"total beads: {coords.shape[0]}")
    print(f"wrote {gro}\n      {top}")


if __name__ == "__main__":
    main()
