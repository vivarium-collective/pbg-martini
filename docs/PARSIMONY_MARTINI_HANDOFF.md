# parsimony → Martini whole-cell E. coli — handoff / status

Branch: `feat/parsimony-martini-md` (pbg-martini) + `main` (pbg-openmm). Companion
PDF: `~/Desktop/ecoli_wholecell_MD_HPC_requirements.pdf`. HPC kit:
`~/Desktop/ecoli_wholecell_GH200/`.

## Goal

Take the 3-D molecular positions of a whole *E. coli* cell — produced by the
**parsimony** packer from the v2ecoli whole-cell model — and turn them into a
runnable **Martini 3 coarse-grained MD** system, à la Stevens et al. 2023
(*Front. Chem.* 11:1106495, the Martini *Mycoplasma* JCVI-syn3A cell), but with
measured WCM positions instead of Bentopy random packing.

## What works (proven, not aspirational)

- **End-to-end pipeline**: parsimony pack → martinize2 (Martini 3) → coarse DNA →
  solvate → OpenMM Martini MD (via `martini_openmm`, reaction-field, 310 K, 20 fs).
- **Real MD runs** (laptop, Apple-Silicon OpenCL, 128 GB):
  - protein slice (vacuum): 91,965 beads, PE −132k→−135k kJ/mol.
  - 40 nm crowded solvated chunk: 646k beads, stable 310 K.
  - 60 nm crowded solvated chunk: **2.58 M beads, stable 310 K, 106 s** (cytoplasm
    + chromosome + Martini water + ions).
  - Martini 3 POPE membrane: 351k beads, stable 310 K.
- **Dry whole-cell assembly**: 49 species stamped at all 383,124 parsimony
  positions → ~120–138 M beads (streaming assembler). Fits one GH200 (~55 GB).

## Components & status

| Component | Method | Status |
|---|---|---|
| Cytosolic + envelope proteins (49 species) | martinize2 Martini 3 + elastic net | ✅ real |
| Chromosome (102,763 dna_segment) | 1 `Q5n` bead/phosphate + elastic (Martini-3-compatible) | ✅ coarse (no helix) |
| RNA polymerase | per-chain martinize + inter-chain elastic | ✅ real (10,510 beads) |
| GroEL | coarse voxel model (1AON NaNs in martinize2) | ⚠️ coarse (2,206 beads) |
| 70S ribosome (3,788) | coarse voxel model (~70% rRNA) | ⚠️ coarse (3,484 beads) |
| Membrane envelope | Martini 3 POPE bilayer on the cell spherocylinder | ✅ real lipid, geometry built |
| Solvation | Martini W + Na/Cl, neutralised ~150 mM | ✅ |

## Scale / hardware (the headline)

| Target | Beads | Hardware |
|---|---|---|
| Dry cytoplasm + chromosome cell | ~138 M | **1 GH200** |
| + real membrane envelope (29.7 M lipids) | +356 M → ~494 M | multi-GH200 |
| Fully solvated whole cell | ~10⁹ | multi-node (GROMACS-MPI / ddcMD) |

## Key decisions

- **Stay Martini 3.** Polyply produces real Martini **2** dsDNA (proven: 156-bead
  dodecamer), but mixing M2 DNA with M3 proteins is FF-inconsistent. Decision:
  keep the coarse `Q5n` DNA (FF-consistent); real DNA helix awaits Martini 3 DNA
  or a full M2 rebuild.
- Membrane uses the **real `martini_v3.0.0_POPE.itp`** (not build_bilayer's
  Martini-2-ish `Qd` types). build_bilayer is used only for the 12-bead geometry,
  which matches the real itp's atom order.

## Code map

**pbg-martini** (`pbg_martini/`):
- `parsimony_assembler.py` — pack loader, slice, quaternion `stamp`, `write_gro`,
  `resolve_structure` (cache-first), `martinize_species`, `assemble`.
- `dna_segment.py` — coarse charged CG chromosome segment.
- `solvate.py` — Martini water + ion solvation (KDTree-excluded, neutralised).
- `multichain.py` — per-chain martinize + inter-chain elastic (RNA-pol, etc.).
- `coarse_structure.py` — voxel-downsample any PDB/mmCIF → coarse CG (GroEL, ribosome).
- `envelope.py` — Martini 3 POPE bilayer on the cell spherocylinder; `write_envelope_system`.
- `scripts/build_crowded_chunk.py` — cytoplasm+chromosome cube + solvate.
- `scripts/build_whole_cell.py` — streaming dry whole-cell assembler.
- `scripts/build_cell_viewer.py` — molecule-level whole-cell viewer (`--envelope`).

**pbg-openmm** (`pbg_openmm/`):
- `processes.py` — `OpenMMMartiniProcess` (real OpenMM+martini_openmm; minimize →
  equilibration ramp → NVT; `platform="CUDA"` for HPC).
- `topology.py` — `prepare_martini_top` (resolve moleculetype names, FF include,
  particle-count validation).
- `data/` — `martini_v3.0.0.itp`, solvents, ions, **POPE**.
- `scripts/make_traj_viewer.py`, `make_crowded_viewer.py` — trajectory viewers.

## Deliverables (Desktop, keep)

- `ecoli_wholecell_MD_HPC_requirements.pdf` — full technical report + GH200 analysis.
- `ecoli_wholecell_GH200/` — self-contained HPC kit (numpy-only assembler +
  templates + Martini itps + OpenMM/GROMACS run scripts + README).

## Rough edges / TODO when resuming

- **Viewers are ad-hoc** (three.js scatter; the combined cytoplasm+chromosome+
  membrane cross-section looks poor). Needs a proper renderer (Mol*/Blender/
  Molecular Nodes) for a presentable figure.
- GroEL + ribosome are **coarse** (excluded-volume only). Atomistic-CG ribosome
  needs Martini RNA and would dominate (~10⁸ beads → multi-GPU).
- DNA is **coarse** (no helix) by the Martini-3 decision above.
- Membrane envelope is built but **not yet fused into the whole-cell assembly**
  (it's the +356 M multi-GPU component); a true curved cross-section (proteins
  *inside* the bilayer) isn't rendered yet.
- **Phase 3 not started**: solvate the whole cell + multi-node production (ddcMD).

## Reproduce / inputs

- Parsimony pack: `~/code/3d-ecoli-app/data/ecoli_3d.pack.json` (383,124 placements).
- Atomistic structures (49): `~/code/3d-ecoli/build/structures/` → copy to
  `pbg-martini/.cache/structures/` to warm the cache-first resolver.
- Venv: `pbg-martini/.venv` (vermouth, openmm, martini_openmm, scipy, mdtraj;
  polyply+pysmiles+cgsmiles installed but unused per the M3 decision).
- Run MD via `pbg-openmm`: `OpenMMMartiniProcess` (needs `PYTHONPATH=…/pbg-openmm`
  when run outside the repo root).
