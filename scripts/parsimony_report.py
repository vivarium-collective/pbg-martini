#!/usr/bin/env python
"""CLI: assemble a parsimony slice, relax it, and render the HTML report.

Usage::

    ECOLI_PACK=~/code/3d-ecoli-app/data/ecoli_3d.pack.json \
      ./.venv/bin/python scripts/parsimony_report.py \
        --center 0 0 0 --edge 1500 --out output/slice \
        --report output/slice/report.html

Runs the offline-safe path by default (stub CG templates, WCA relax, no
OpenMM). The relaxed coordinates are written back to ``relaxed.gro`` and the
report embeds a 3D viewer of the assembled slice plus per-species counts and
the clash-before/after metric.
"""

import argparse
import os

import numpy as np

from pbg_martini.parsimony_assembler import CGTemplate, assemble, write_gro
from pbg_martini.parsimony_md import openmm_available, relax_assembly, run_short_md
from pbg_martini.processes import _read_gro_coords
from pbg_martini.visualizations import build_parsimony_report

ALLOW_LIST = [
    "EG10367-MONOMER", "EG11036-MONOMER", "groel",
    "EG11384-MONOMER", "EG50003-MONOMER", "EG10669-MONOMER",
]


def stub_templates(n_beads=5):
    return {
        name: CGTemplate(name, f"{name}.gro", f"{name}.itp",
                         np.zeros((n_beads, 3)), n_beads)
        for name in ALLOW_LIST
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=os.environ.get(
        "ECOLI_PACK",
        os.path.expanduser("~/code/3d-ecoli-app/data/ecoli_3d.pack.json")))
    ap.add_argument("--center", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--edge", type=float, default=1500.0)
    ap.add_argument("--out", default="output/slice")
    ap.add_argument("--report", default=None)
    ap.add_argument("--species", nargs="*", default=ALLOW_LIST)
    ap.add_argument("--beads", type=int, default=5)
    ap.add_argument("--relax-steps", type=int, default=200)
    ap.add_argument("--run-md", action="store_true")
    ap.add_argument("--md-steps", type=int, default=100)
    ap.add_argument("--no-bentopy", action="store_true")
    args = ap.parse_args()

    c = np.array(args.center)
    half = args.edge / 2.0
    box_min = tuple(c - half)
    box_max = tuple(c + half)

    templates = stub_templates(args.beads)
    summary = assemble(args.pack, box_min, box_max, args.species, templates,
                       args.out, use_bentopy=not args.no_bentopy)
    print(f"assembled {summary['n_beads']} beads / "
          f"{summary['n_molecules']} molecules via {summary['rendered_by']}")

    # WCA relax (always available, pure Python).
    coords, box_nm = _read_gro_coords(summary["gro"])
    relaxed_gro = summary["gro"]
    if coords.shape[0]:
        relaxed = relax_assembly(coords, n_steps=args.relax_steps)
        _, names, _ = (None, None, None)
        # Re-read residue/atom names from the assembled gro to preserve labels.
        with open(summary["gro"]) as fh:
            lines = fh.read().splitlines()
        n = int(lines[1].strip())
        names = [lines[2 + i][10:15].strip() or "BB" for i in range(n)]
        relaxed_gro = os.path.join(args.out, "relaxed.gro")
        write_gro(relaxed_gro, names, relaxed, box_nm)

    final_energy = None
    md_ran = False
    if args.run_md and openmm_available():
        try:
            md = run_short_md(summary["gro"], summary["top"], steps=args.md_steps)
            final_energy = md["final_energy"]
            relaxed_gro = md.get("minimized_gro", relaxed_gro)
            md_ran = True
        except Exception as exc:  # best-effort
            print(f"OpenMM MD skipped: {exc}")

    report_path = args.report or os.path.join(args.out, "report.html")
    out = build_parsimony_report(
        summary, relaxed_gro=relaxed_gro, out_html=report_path,
        final_energy=final_energy, md_ran=md_ran,
    )
    print(f"report written to {out}")
    return out


if __name__ == "__main__":
    main()
