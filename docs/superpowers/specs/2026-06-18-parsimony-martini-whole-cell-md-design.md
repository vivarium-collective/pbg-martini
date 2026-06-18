# parsimony → Martini whole-cell MD (proof-of-concept)

**Date:** 2026-06-18
**Repo:** `pbg-martini`
**Status:** approved design, pending implementation plan

## Motivation

Stevens et al. 2023 ("Molecular dynamics simulation of an entire cell",
*Front. Chem.* 11:1106495) build a Martini coarse-grained (CG) model of the
minimal cell JCVI-syn3A via the Martini ecosystem: Martinize2 (atomistic
protein → Martini topology), **Bentopy** (random collision-free packing of
molecules into the cell volume), Polyply (chromosome), TS2CG (membrane), then a
GROMACS/OpenMM run of ~561 M beads. Their packing is *random* — Bentopy places
molecules with a collision-detection scheme, not from measured positions.

We already have a **measured/derived 3D structure of v2ecoli** produced by
**parsimony** (a Rust cellPACK-style packer): `ecoli_3d.pack.json`. The goal of
this work is to reproduce the paper's integrative workflow but **replace
Bentopy's random packing with parsimony's actual placements** — i.e. build a
runnable Martini CG MD system whose molecules sit at the parsimony-packed
positions and orientations.

This spec covers a **tractable proof-of-concept (PoC)** that assembles a slice
of the cell, relaxes it, and runs a short real MD, end to end on a laptop / the
Mac mini. Scaling the same code path to the full pack (files-only, HPC MD) is an
explicit later phase, out of scope here.

## Inputs (already on disk)

- `/Users/eranagmon/code/3d-ecoli-app/data/ecoli_3d.pack.json` — parsimony pack,
  `format: parsimony.pack.v1`:
  - `bounds`, `compartments` (a ~20000×12000×12000 Å box ≈ 2 µm cell)
  - `ingredients`: 52 species (`70S_ribosome`, `rna_polymerase`, `groel`, and
    EcoCyc-ID protein monomers `EG…-MONOMER`, `PD…`), each with color, an
    ellipsoid/mesh `shape`, and VdW-mesh LOD `.obj` urls.
  - `placements`: **383,124** entries, each `{ingredient, compartment,
    position: [x,y,z] (Å), rotation: [quaternion], uid}`.
- `/Users/eranagmon/code/3d-ecoli-app/data/ecoli_3d.meta.json` — per-ingredient
  `display_name`, `category`, `count`.
- `/Users/eranagmon/code/parsimony/examples/recipes/ecoli_3d.json` and
  `.../pipelines/ecoli_3d.pipeline.json` — the recipe/pipeline that built the
  pack (stages: chromosome, membrane `lipid`, fiber_proteins, interior). Lipid
  is a `single_sphere` r=12; DNA is the `1BNA` mesh.

## Existing capabilities we build on

- **pbg-martini** (this repo): `run_martinize_pipeline(pdb_text)` runs real
  vermouth/martinize2 (vermouth 0.15.0 installed) → Martini CG beads + `.itp`;
  `build_bilayer` / `build_vesicle` / `build_protein_in_membrane` membrane
  builders; `relax_structure` WCA steepest-descent minimizer. Emits
  GROMACS-suitable `.gro`/`.top`. Does **not** itself wrap a full MD engine.
- **pbg-parsimony**: `structures.py` resolves a molecule ID (EcoCyc / UniProt /
  RCSB) → atomistic PDB/mmCIF (AlphaFold DB or RCSB), cached. This is how we
  recover each species' atomistic structure to martinize (the pack/recipe only
  carry VdW meshes, not raw PDBs).
- **OpenMM** — NOT yet installed; pip-install into the pbg-martini venv. It
  reads GROMACS Martini topologies directly via `GromacsTopFile` +
  `GromacsGroFile`, the standard route to run Martini in OpenMM. (No system
  `gmx`.)

## Architecture: a four-stage PBG composite

New composite `pbg_martini/composites/parsimony-whole-cell.composite.yaml`
wiring four Steps. New module `pbg_martini/parsimony_assembler.py` holds stages
1 and 3 (and the data structures); stages 2 and 4 reuse existing pbg-martini
functions plus a thin OpenMM runner.

Each stage is independently testable with a clear interface:

### Stage 1 — Select & resolve (`parsimony_assembler.select_slice`, `.resolve_structures`)
- **Does:** read `pack.json` + `meta.json`; choose a PoC slice = (a) one
  spatial **sub-box** of the cell (configurable corner + edge length), and
  (b) a configurable list of ~5–8 cleanly-martinizable globular/membrane
  proteins (default: GAPDH `EG10367`, EF-Tu `EG11036`, GroEL `groel`, AhpC
  `EG11384`, acyl-carrier protein `EG50003`, OmpA `EG10669`). For each selected
  species recover its atomistic PDB via pbg-parsimony's resolver, cached under
  `.cache/`.
- **In:** pack path, sub-box spec, species allow-list. **Out:** `SliceSpec`
  (selected placements grouped by species) + `{species: pdb_path}`.
- **Depends on:** pbg-parsimony `structures`.
- **Excluded from PoC:** ribosome (`70S_ribosome`), DNA (`1BNA`), RNA — they need
  Martini-2 nucleic-acid params; deferred. Document the exclusion in the report.

### Stage 2 — CG templates (`MartinizeSpeciesStep`, reuses `run_martinize_pipeline`)
- **Does:** martinize each selected species **once** → a `CGTemplate`:
  origin-centered bead coordinates (reference orientation) + per-molecule
  `.itp` + bead count. Cache by species id.
- **In:** `{species: pdb_path}`. **Out:** `{species: CGTemplate}`.

### Stage 3 — Stamp & assemble (`parsimony_assembler.assemble`)
- **Does:** for every placement of each selected species in the slice: take the
  species `CGTemplate`, **rotate** its beads by the placement quaternion,
  **translate** to `position`, convert **Å → nm** (÷10), assign a unique
  molecule/residue index, append to a global bead array. Then write:
  - `system.gro` — all beads (nm), box = sub-box dimensions.
  - `system.top` — `#include` martini force-field (`martini_v3.0.0.itp`, shipped
    in repo `pbg_martini/data/` or fetched) + each species `.itp`; `[ molecules ]`
    section listing each species and its count in the slice.
  - Optional membrane: a `build_bilayer` Martini patch placed at the
    cell-envelope face of the sub-box (PoC default: small POPC/POPE/CHOL patch),
    or — later — Martini lipids stamped at parsimony's `lipid` positions.
- **In:** `SliceSpec`, `{species: CGTemplate}`, optional membrane config.
  **Out:** paths to `system.gro`, `system.top`, plus a summary
  `{n_beads, n_molecules, per_species_counts, box_nm}`.
- **Invariant (tested):** `n_beads == Σ_species (template_bead_count × slice_count)`
  (+ membrane beads).

### Stage 4 — Relax + MD (`MartiniMDStep`)
- **Does:**
  1. pbg-martini `relax_structure` WCA minimization to remove residual
     packing clashes (positions only).
  2. OpenMM: `GromacsGroFile`/`GromacsTopFile` → `System` with Martini settings
     (Verlet/Martini cutoff scheme), local energy minimization + a short NVT run
     (a few hundred steps at Martini dt) → trajectory.
  3. Render a 3D CG viewer of the result (reuse `pbg_martini/visualizations.py`
     / the 3d viewer) and an HTML report (clash before/after, bead counts,
     energy, screenshot).
- **In:** `system.gro/.top`. **Out:** minimized structure, short trajectory,
  viewer + report.
- **Caps slice size:** the MD step targets ≲ ~200k beads so min+short run
  completes on laptop/mini. Stage 3 can produce larger systems for files-only
  scaling; the composite's default config keeps the MD-runnable size.

## Data flow

```
ecoli_3d.pack.json + meta.json
        │  (Stage 1: slice + resolve)
        ▼
SliceSpec(placements by species) + {species → atomistic PDB}
        │  (Stage 2: martinize each species once)
        ▼
{species → CGTemplate(beads, itp, n)}
        │  (Stage 3: stamp at parsimony positions/orientations)
        ▼
system.gro + system.top  (+ optional membrane patch)
        │  (Stage 4: WCA relax → OpenMM min + short NVT)
        ▼
minimized structure + trajectory + 3D viewer + HTML report
```

## Testing strategy (pytest, `tests/`)

1. **Pack parse** — load `ecoli_3d.pack.json`; assert species/placement counts,
   that slice selection returns only in-box placements of allow-listed species.
2. **Single-species martinize** — one small protein PDB → non-empty
   `CGTemplate` with a valid `.itp` and bead count > 0.
3. **Stamp invariant** — assembled `n_beads` equals the analytic sum; a stamped
   instance's centroid matches its placement position (within tolerance) and its
   orientation matches the quaternion (check one bead vector).
4. **File well-formedness** — `system.gro` line/atom counts consistent with
   header; `system.top` `[ molecules ]` counts match the slice.
5. **MD smoke** — OpenMM builds a `System` from the files and runs k steps with
   finite (non-NaN) energy on a tiny slice.

Use a tiny synthetic/sub-sampled slice for fast tests; gate the full OpenMM
smoke behind a marker if too slow for CI.

## Risks & open implementation questions (resolve in the plan)

- **Force-field include:** OpenMM needs `martini_v3.0.0.itp` (+ ion/lipid itps)
  present. Decide: vendor into `pbg_martini/data/` vs fetch-and-cache. Confirm
  martinize2 here emits Martini 3 `.itp` compatible with that ff file.
- **Elastic network / bonded:** martinize2 single-protein `.itp` may include an
  elastic network (`-elastic`); decide default for PoC (likely on, to keep
  globular shape under Martini).
- **Nucleic acids deferred:** ribosome/DNA/RNA excluded from the PoC; note in
  report and as the first follow-up for a fuller model.
- **Slice/box choice:** pick a sub-box that contains a representative protein
  mix and (optionally) part of the envelope, sized so the MD stage finishes
  locally. Parameterize; default chosen in the plan after inspecting density.
- **Å → nm and periodic box:** ensure consistent unit conversion and a box that
  encloses the slice with Martini-appropriate padding.

## Out of scope (later phases)

- Full 383k-placement, 52-species whole-cell assembly (files-only, HPC MD).
- Martini nucleic-acid components (chromosome via Polyply-analog, ribosome RNA).
- Full envelope from parsimony `lipid` positions rather than a builder patch.
- Long equilibration / production MD.
