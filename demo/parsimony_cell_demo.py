"""Demo: parsimony → Martini whole-cell MD (PoC), end to end.

Pipeline (each stage degrades gracefully so the demo always produces a report):

    1. resolve   — fetch an atomistic structure per allow-list species via
                   pbg-parsimony (network); skipped per-species on failure.
    2. martinize — coarse-grain each resolved structure into a CG template
                   (vermouth/martinize2); falls back to a stub template when
                   the structure could not be resolved or martinized.
    3. assemble  — stamp every species instance at its parsimony-measured
                   position/orientation inside a small sub-box (replacing
                   Bentopy random packing). Uses real ``bentopy render`` when
                   available, else the pure-Python stamper.
    4. relax     — WCA steepest-descent declash (always runs).
    5. md        — best-effort OpenMM short NVT (skipped if OpenMM is absent or
                   the stub topology lacks force-field parameters).
    6. report    — self-contained HTML with a 3D viewer + metrics.

Run::

    ECOLI_PACK=~/code/3d-ecoli-app/data/ecoli_3d.pack.json \
      ./.venv/bin/python demo/parsimony_cell_demo.py

Set ``--edge`` small (default 1500 Å) to keep the slice light. Missing
network / openmm / bentopy never block the run — the report is always written.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from pbg_martini.parsimony_assembler import (
    CGTemplate,
    assemble,
    martinize_species,
    resolve_structure,
    write_gro,
)
from pbg_martini.parsimony_md import openmm_available, relax_assembly, run_short_md
from pbg_martini.processes import _read_gro_coords
from pbg_martini.visualizations import build_parsimony_report

ALLOW_LIST = [
    "EG10367-MONOMER", "EG11036-MONOMER", "groel",
    "EG11384-MONOMER", "EG50003-MONOMER", "EG10669-MONOMER",
]


def build_templates(species, out_dir, stub_beads=10):
    """Resolve + martinize each species; fall back to a stub CGTemplate.

    Returns ``(templates, notes)`` where ``notes`` records, per species, whether
    a real martinized template or a stub placeholder was used.
    """
    templates = {}
    notes = {}
    tpl_dir = os.path.join(out_dir, "templates")
    os.makedirs(tpl_dir, exist_ok=True)
    for name in species:
        try:
            pdb = resolve_structure(name)
            tpl = martinize_species(name, pdb, tpl_dir, elastic=True)
            templates[name] = tpl
            notes[name] = f"martinized ({tpl.n_beads} beads)"
        except Exception as exc:  # network / martinize / mapping failure
            templates[name] = CGTemplate(
                name, f"{name}.gro", f"{name}.itp",
                np.zeros((stub_beads, 3)), stub_beads,
            )
            notes[name] = f"stub {stub_beads} beads ({type(exc).__name__})"
    return templates, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=os.environ.get(
        "ECOLI_PACK",
        os.path.expanduser("~/code/3d-ecoli-app/data/ecoli_3d.pack.json")))
    ap.add_argument("--center", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--edge", type=float, default=1500.0)
    ap.add_argument("--out", default="output/parsimony_cell_demo")
    ap.add_argument("--species", nargs="*", default=ALLOW_LIST)
    ap.add_argument("--relax-steps", type=int, default=200)
    ap.add_argument("--run-md", action="store_true")
    ap.add_argument("--md-steps", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.pack):
        raise SystemExit(
            f"pack not found: {args.pack}\n"
            "Set ECOLI_PACK to a parsimony.pack.v1 JSON, or pass --pack.")

    c = np.array(args.center)
    half = args.edge / 2.0
    box_min = tuple(c - half)
    box_max = tuple(c + half)

    print("== 1-2. resolve + martinize templates (best-effort) ==")
    templates, notes = build_templates(args.species, args.out)
    for name, note in notes.items():
        print(f"   {name}: {note}")

    print("== 3. assemble at parsimony positions ==")
    summary = assemble(args.pack, box_min, box_max, args.species,
                       templates, args.out, use_bentopy=True)
    print(f"   {summary['n_beads']} beads / {summary['n_molecules']} molecules "
          f"via {summary['rendered_by']}")
    print(f"   per-species: {summary['per_species_counts']}")

    print("== 4. WCA relax ==")
    coords, box_nm = _read_gro_coords(summary["gro"])
    relaxed_gro = summary["gro"]
    if coords.shape[0]:
        relaxed = relax_assembly(coords, n_steps=args.relax_steps)
        with open(summary["gro"]) as fh:
            lines = fh.read().splitlines()
        n = int(lines[1].strip())
        names = [lines[2 + i][10:15].strip() or "BB" for i in range(n)]
        relaxed_gro = os.path.join(args.out, "relaxed.gro")
        write_gro(relaxed_gro, names, relaxed, box_nm)
        print(f"   relaxed {n} beads -> {relaxed_gro}")

    final_energy = None
    md_ran = False
    if args.run_md:
        print("== 5. OpenMM short MD (best-effort) ==")
        if openmm_available():
            try:
                md = run_short_md(summary["gro"], summary["top"],
                                  steps=args.md_steps)
                final_energy = md["final_energy"]
                relaxed_gro = md.get("minimized_gro", relaxed_gro)
                md_ran = True
                print(f"   final energy {final_energy:.2f} kJ/mol")
            except Exception as exc:
                print(f"   skipped: {exc}")
        else:
            print("   skipped: OpenMM not installed")

    print("== 6. report ==")
    report = os.path.join(args.out, "parsimony_cell_demo.html")
    build_parsimony_report(summary, relaxed_gro=relaxed_gro, out_html=report,
                           final_energy=final_energy, md_ran=md_ran)
    print(f"   report -> {report}")


if __name__ == "__main__":
    main()
