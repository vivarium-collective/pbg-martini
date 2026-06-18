#!/usr/bin/env python
"""CLI: assemble a Martini CG slice from a parsimony pack sub-box.

Usage::

    ECOLI_PACK=~/code/3d-ecoli-app/data/ecoli_3d.pack.json \
      ./.venv/bin/python scripts/assemble_slice.py \
        --center 0 0 0 --edge 1500 --out output/slice

Builds stub 5-bead templates by default (so it runs fully offline); pass real
martinized templates via the library API for a runnable system. Writes
``system.gro``, ``system.top``, and ``placements.json`` to the out dir.
"""

import argparse
import json
import os

import numpy as np

from pbg_martini.parsimony_assembler import CGTemplate, assemble

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
    ap.add_argument("--center", type=float, nargs=3, default=(0, 0, 0),
                    help="sub-box center in Angstrom")
    ap.add_argument("--edge", type=float, default=1500.0,
                    help="sub-box edge length in Angstrom")
    ap.add_argument("--out", default="output/slice")
    ap.add_argument("--species", nargs="*", default=ALLOW_LIST)
    ap.add_argument("--beads", type=int, default=5,
                    help="stub template bead count")
    ap.add_argument("--no-bentopy", action="store_true")
    args = ap.parse_args()

    c = np.array(args.center)
    half = args.edge / 2.0
    box_min = tuple(c - half)
    box_max = tuple(c + half)

    templates = stub_templates(args.beads)
    out = assemble(args.pack, box_min, box_max, args.species, templates,
                   args.out, use_bentopy=not args.no_bentopy)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
