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

### Stage 3 — Convert & render via bentopy (`parsimony_assembler.to_bentopy_placements` + `render`)

The marrink-lab whole-cell workshop (`martini-workshop/05_constructing_martini_cell`)
assembles its cell in two decoupled bentopy steps:

```
bentopy pack   --rearrange --seed 5172 --rotations 3  cytosol_input.json   # random packer → placements.json
bentopy render -t topol.top  outputs/..._placements.json  cytosol.gro      # placements → gro + top
bentopy grocat chromosome_membrane.gro cytosol.gro:CYT -o cell.gro         # add envelope/chromosome
```

**This is exactly our bridge:** parsimony *is* the packer, so it **replaces
`bentopy pack`**. We convert `parsimony.pack.v1` → bentopy's **placement-list
JSON** ("which structures, at what rotations, placed where") and reuse the real
**`bentopy render`** to assemble `.gro` + `.top`, plus `bentopy grocat` to add a
membrane/envelope. We reuse the validated upstream renderer rather than
hand-rolling a stamper.

- **Does:**
  1. **Convert** — map each selected parsimony placement → a bentopy placement
     entry: segment name = species; structure file = that species' martinized CG
     `.pdb`/`.gro` (Stage 2 template); rotation = the parsimony quaternion
     (converted to bentopy's rotation convention); position = parsimony
     `position` converted **Å → nm** (÷10). Emit `placements.json` + a bentopy
     `output`/topology section listing the Martini `.itp` includes.
  2. **Render** — run `bentopy render -t system.top placements.json system.gro`.
  3. **Concatenate** — if a membrane patch is built, `bentopy grocat` it onto the
     rendered cytosol.
- **In:** `SliceSpec`, `{species: CGTemplate}` (martinized structure + `.itp`),
  optional membrane config. **Out:** `placements.json`, `system.gro`,
  `system.top`, summary `{n_beads, n_molecules, per_species_counts, box_nm}`.
- **Tool dependency:** the `bentopy` binary (Rust; `BENTOPY_BIN`/`PATH`, install
  via pip/maturin or cargo) — same pattern pbg-parsimony already uses for the
  `parsimony` binary.
- **Fallback:** if `bentopy` is unavailable, a small pure-Python stamper
  (rotate template beads by the quaternion, translate, write `.gro`/`.top`)
  produces the same artifact; kept minimal and behind a capability check so the
  PoC still runs. Primary path is real bentopy.
- **Membrane:** a `build_bilayer` Martini patch at the envelope face of the
  sub-box (PoC default: small POPC/POPE/CHOL patch), grocat'd on; later — Martini
  lipids stamped at parsimony's `lipid` positions through the same converter.
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
        │  (Stage 2: martinize each species once → CG structure + itp)
        ▼
{species → CGTemplate(structure.gro/pdb, itp, n_beads)}
        │  (Stage 3a: parsimony.pack.v1 → bentopy placements.json)
        ▼
placements.json (+ itp includes)
        │  (Stage 3b: bentopy render → assemble; grocat membrane)
        ▼
system.gro + system.top  (+ membrane patch)
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

- **Force-field include:** `system.top` and OpenMM need the Martini 3
  force-field `.itp` (+ ion/lipid itps). Prefer the **martini-forcefields** pip
  package over vendoring loose files; confirm martinize2 here emits Martini 3
  `.itp` compatible with that ff version.
- **bentopy availability:** Rust binary; decide install route (pip/maturin vs
  cargo vs `BENTOPY_BIN`) and confirm its placement-list JSON schema + rotation
  convention against parsimony quaternions (write a converter unit test). The
  pure-Python stamper is the fallback if bentopy can't be installed.
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

## Upstream tools & references (the templates to follow)

- **marrink-lab/martini-workshop** `05_constructing_martini_cell` — the canonical
  whole-cell construction tutorial (`tutorial.md`, `input.json`,
  `chromosome.gro`, `sphere.tsi`, `proteins/`). Our pipeline mirrors it with
  parsimony swapped in for `bentopy pack`. Workshop uses GROMACS 2024.1 + Martini 2
  for the cell demo; we target Martini 3 proteins + OpenMM.
- **marrink-lab/bentopy** — `pack` (random) / `render` (placements→gro/top) /
  `grocat`. We reuse `render` + `grocat`; we replace `pack` with parsimony.
- **marrink-lab/vermouth-martinize** — martinize2 (already wrapped by pbg-martini).
- **marrink-lab/martini-forcefields** — official Martini 3 `.itp` parameter
  collection; source of the force-field includes for `system.top` (pip-installable
  package — prefer this over vendoring loose itps).
- **marrink-lab/TS2CG**, **polyply_1.0** — membrane / DNA builders, for later
  envelope + chromosome phases (out of PoC scope).
- For GROMACS-based MD as an alternative to OpenMM, the workshop's `.mdp`
  minimization/equilibration files are the reference parameter set; OpenMM is the
  PoC default because no system `gmx` is present.

## Out of scope (later phases)

- Full 383k-placement, 52-species whole-cell assembly (files-only, HPC MD).
- Martini nucleic-acid components (chromosome via Polyply-analog, ribosome RNA).
- Full envelope from parsimony `lipid` positions rather than a builder patch.
- Long equilibration / production MD.
