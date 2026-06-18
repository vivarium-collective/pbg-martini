# parsimony → Martini whole-cell MD (PoC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge parsimony's packed 3D v2ecoli structure (`ecoli_3d.pack.json`) into a runnable Martini coarse-grained MD system — assemble CG molecules at the parsimony positions/orientations (replacing Bentopy's random packing), relax, and run a short MD.

**Architecture:** Four-stage pipeline added to `pbg-martini`: (1) slice + resolve structures, (2) martinize2 each species once into a CG template, (3) convert `parsimony.pack.v1` → bentopy placement-list JSON and render `.gro`/`.top` (with a pure-Python stamper fallback), (4) WCA relax + best-effort OpenMM short MD. Exposed as PBG Steps + a composite.

**Tech Stack:** Python 3.11, vermouth/martinize2 0.15.0 (installed), numpy, pbg-parsimony (structure resolver), bentopy (optional Rust binary), OpenMM (best-effort), process-bigraph.

## Global Constraints

- Work in repo `pbg-martini`, branch `feat/parsimony-martini-md`. Use the venv `.venv/bin/python` (vermouth lives there). NEVER bare `python`.
- **Commit AND `git push origin feat/parsimony-martini-md` after every task** — the laptop is offline; pushed commits are the only progress signal.
- TDD: write the failing test, see it fail, implement minimally, see it pass, commit+push. Real code in every step — no placeholders.
- **Offline-first ordering:** Tasks 1, 4, 5, 6, 7 are fully testable with a tiny synthetic fixture and MUST be completed first. Tasks 2, 3, 8 use real tools/network. Tasks 9–12 (OpenMM MD, Steps/composite, viewer/report) are best-effort — if `openmm`/`bentopy` won't install on this machine, SKIP the run but still land the code + skipped tests; never block the deterministic core on them.
- Input pack: `~/code/3d-ecoli-app/data/ecoli_3d.pack.json` (+ `.meta.json`), synced to the mini. Reference it via env var `ECOLI_PACK` (default that path). Do NOT commit the 77 MB pack; tests use a small synthetic fixture built in code.
- Units: parsimony positions are **Ångström**; Martini/GROMACS `.gro` is **nm** → divide by 10. Rotations are quaternions `[w?,x,y,z]` — confirm convention empirically in Task 4 (see note).
- PoC species allow-list (martinizable proteins, EcoCyc/named ids from `meta.json`): `EG10367-MONOMER` (GAPDH), `EG11036-MONOMER` (EF-Tu), `groel`, `EG11384-MONOMER` (AhpC), `EG50003-MONOMER` (ACP), `EG10669-MONOMER` (OmpA). Exclude `70S_ribosome`, `1BNA`/DNA, RNA (need Martini-2 nucleic params).

---

### Task 1: Pack loader, data model, and slice selection

**Files:**
- Create: `pbg_martini/parsimony_assembler.py`
- Test: `tests/test_parsimony_assembler.py`

**Interfaces:**
- Produces: `load_pack(path) -> Pack`; `Pack.ingredients: dict[int, Ingredient]` (id→Ingredient(name, color)), `Pack.placements: list[Placement]` (Placement(ingredient_id:int, position:tuple[float,float,float], rotation:tuple[float,...], uid:int)), `Pack.bounds`. `select_slice(pack, box_min, box_max, species_names) -> SliceSpec` where `SliceSpec.by_species: dict[str, list[Placement]]` keeps only placements whose ingredient name ∈ species_names and whose position is within [box_min, box_max] (Å).

- [ ] **Step 1: Write the failing test** — synthetic pack dict written to a temp json, two species, four placements (two in-box, two out).

```python
import json, numpy as np
from pbg_martini.parsimony_assembler import load_pack, select_slice

def _synthetic_pack(tmp_path):
    pack = {
        "format": "parsimony.pack.v1",
        "bounds": {"min": [-100,-100,-100], "max": [100,100,100]},
        "ingredients": [
            {"id": 0, "name": "groel", "color": [1,0,0]},
            {"id": 1, "name": "EG10367-MONOMER", "color": [0,1,0]},
        ],
        "placements": [
            {"ingredient": 0, "position": [10,10,10], "rotation": [1,0,0,0], "uid": 0, "compartment": 1},
            {"ingredient": 1, "position": [20,0,0],  "rotation": [1,0,0,0], "uid": 1, "compartment": 1},
            {"ingredient": 0, "position": [90,90,90], "rotation": [1,0,0,0], "uid": 2, "compartment": 1},
            {"ingredient": 1, "position": [-90,0,0],  "rotation": [1,0,0,0], "uid": 3, "compartment": 1},
        ],
    }
    p = tmp_path / "syn.pack.json"; p.write_text(json.dumps(pack)); return p

def test_load_and_slice(tmp_path):
    pack = load_pack(_synthetic_pack(tmp_path))
    assert pack.ingredients[0].name == "groel"
    assert len(pack.placements) == 4
    sl = select_slice(pack, box_min=(0,-50,-50), box_max=(50,50,50),
                       species_names=["groel", "EG10367-MONOMER"])
    # only the two placements at (10,10,10) and (20,0,0) are in-box
    assert sum(len(v) for v in sl.by_species.values()) == 2
    assert len(sl.by_species["groel"]) == 1
```

- [ ] **Step 2: Run test, verify it fails** — `\.venv/bin/python -m pytest tests/test_parsimony_assembler.py::test_load_and_slice -v` → ImportError.
- [ ] **Step 3: Implement** `load_pack`/`select_slice` with `@dataclass` Ingredient/Placement/Pack/SliceSpec. `load_pack` reads json, builds `ingredients` keyed by `id`, maps each placement's `ingredient` int to keep the id. `select_slice` filters by name set and per-axis box containment.
- [ ] **Step 4: Run test, verify it passes.**
- [ ] **Step 5: Commit + push** — `git add ... && git commit -m "feat(parsimony): pack loader + slice selection" && git push origin feat/parsimony-martini-md`.

---

### Task 4: Quaternion rotation + bead stamping (pure math)

> Done before Tasks 2/3 because it is fully offline-testable and is the mathematical heart of the bridge.

**Files:**
- Modify: `pbg_martini/parsimony_assembler.py`
- Test: `tests/test_parsimony_assembler.py`

**Interfaces:**
- Produces: `quat_to_matrix(q) -> np.ndarray (3,3)`; `stamp(template_beads_nm: np.ndarray (N,3), position_A: tuple, rotation: tuple) -> np.ndarray (N,3)` — rotates origin-centered template beads, then translates to `position` converted Å→nm.

- [ ] **Step 1: Write the failing test** — identity quaternion leaves beads put (only translates); a 90° z-rotation maps +x→+y.

```python
from pbg_martini.parsimony_assembler import quat_to_matrix, stamp

def test_stamp_identity_translates_only():
    beads = np.array([[1.0,0,0],[0,1.0,0]])
    out = stamp(beads, position_A=(50,0,0), rotation=(1,0,0,0))  # Å→nm: 5.0
    assert np.allclose(out, beads + np.array([5.0,0,0]))

def test_stamp_90deg_z():
    import math
    beads = np.array([[1.0,0,0]])
    q = (math.cos(math.pi/4), 0,0, math.sin(math.pi/4))  # 90° about z, (w,x,y,z)
    out = stamp(beads, position_A=(0,0,0), rotation=q)
    assert np.allclose(out, np.array([[0,1.0,0]]), atol=1e-6)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `quat_to_matrix` (assume `(w,x,y,z)`; normalize) and `stamp` (`beads @ R.T + position/10`). **NOTE for implementer:** verify parsimony's quaternion order against a real placement in Task 7 — if stamped instances look mirrored/wrong, the pack may store `(x,y,z,w)`; add a `quat_order` param and a one-line check using `3d-ecoli-app/viewer.js` as the reference convention. Record the confirmed order in a comment.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit + push** — `"feat(parsimony): quaternion stamping math"`.

---

### Task 5: bentopy placement-JSON converter + render dispatch

**Files:**
- Modify: `pbg_martini/parsimony_assembler.py`
- Test: `tests/test_parsimony_assembler.py`

**Interfaces:**
- Produces: `to_bentopy_placements(slice_spec, templates: dict[str, CGTemplate], box_nm) -> dict` (bentopy placement-list JSON). `CGTemplate` is a dataclass `(name:str, structure_path:str, itp_path:str, beads_nm:np.ndarray, n_beads:int)`. `bentopy_available() -> bool`.

Upstream reference (martini-workshop 05): `bentopy render -t topol.top placements.json out.gro`. The placement JSON describes "which structures, at what rotations, placed where." **The exact schema is not documented publicly — at execution time, run `bentopy render --help` and inspect `bentopy`'s repo/wiki or an example `*_placements.json` to learn the field names, then map parsimony → those fields.** If `bentopy` is not installed and cannot be installed (no cargo on this host), set `bentopy_available()` False and rely on the Task 6 pure-Python stamper; STILL implement and test the converter dict shape so the bentopy path is ready when the binary exists.

- [ ] **Step 1: Write the failing test** — converter emits one entry per species with the right instance count and Å→nm-converted positions.

```python
from pbg_martini.parsimony_assembler import to_bentopy_placements, CGTemplate
def test_converter_counts(tmp_path):
    tpl = {"groel": CGTemplate("groel", "g.gro", "g.itp", np.zeros((10,3)), 10)}
    # one in-box groel placement
    from pbg_martini.parsimony_assembler import Placement, SliceSpec
    sl = SliceSpec(by_species={"groel": [Placement(0,(10,10,10),(1,0,0,0),0)]})
    j = to_bentopy_placements(sl, tpl, box_nm=(20,20,20))
    seg = [s for s in j["placements"] if s["name"]=="groel"][0]
    assert len(seg["instances"]) == 1  # adapt key to real bentopy schema at runtime
    assert np.allclose(seg["instances"][0]["position"], [1.0,1.0,1.0])
```

- [ ] **Step 2–4:** fail → implement (positions ÷10, rotation passthrough, structure_path from template) → pass. Adjust `instances`/`position` keys to the real bentopy schema discovered at runtime; keep the test asserting whatever shape you emit.
- [ ] **Step 5: Commit + push** — `"feat(parsimony): bentopy placement converter"`.

---

### Task 6: Pure-Python `.gro`/`.top` writers + stamp-all assembler

**Files:**
- Modify: `pbg_martini/parsimony_assembler.py`
- Test: `tests/test_parsimony_assembler.py`

**Interfaces:**
- Produces: `write_gro(path, bead_names, coords_nm, box_nm)`; `write_top(path, itp_includes, molecule_counts)`; `stamp_all(slice_spec, templates) -> (coords_nm: np.ndarray, bead_names: list[str], counts: dict[str,int])`. This is the fallback renderer when bentopy is absent, and the source of the assembly invariant.

- [ ] **Step 1: Write the failing test** — invariant + gro line count.

```python
def test_stamp_all_invariant(tmp_path):
    tpl = {"groel": CGTemplate("groel","g.gro","g.itp", np.zeros((10,3)), 10),
           "EG10367-MONOMER": CGTemplate("EG10367-MONOMER","e.gro","e.itp", np.zeros((7,3)), 7)}
    from pbg_martini.parsimony_assembler import Placement, SliceSpec, stamp_all, write_gro
    sl = SliceSpec(by_species={
        "groel":[Placement(0,(10,0,0),(1,0,0,0),0), Placement(0,(20,0,0),(1,0,0,0),1)],
        "EG10367-MONOMER":[Placement(1,(0,0,0),(1,0,0,0),2)]})
    coords, names, counts = stamp_all(sl, tpl)
    assert coords.shape[0] == 10*2 + 7*1          # n_beads invariant
    assert counts == {"groel":2, "EG10367-MONOMER":1}
    g = tmp_path/"s.gro"; write_gro(g, names, coords, (30,30,30))
    lines = g.read_text().splitlines()
    assert int(lines[1].strip()) == coords.shape[0]  # GRO atom count line
```

- [ ] **Step 2: fail.**
- [ ] **Step 3: Implement** GRO writer (line 1 title, line 2 atom count, `%5d%-5s%5s%5d%8.3f%8.3f%8.3f` records, last line box vectors) and TOP writer (`#include` lines + `[ molecules ]` name/count). `stamp_all` loops species×placements calling `stamp` from Task 4.
- [ ] **Step 4: pass.**
- [ ] **Step 5: Commit + push** — `"feat(parsimony): gro/top writers + stamp_all invariant"`.

---

### Task 7: `assemble()` orchestration (real pack, end-to-end offline)

**Files:**
- Modify: `pbg_martini/parsimony_assembler.py`
- Create: `scripts/assemble_slice.py` (CLI: env `ECOLI_PACK`, args box + out dir)
- Test: `tests/test_parsimony_assembler.py` (integration-marked; uses real pack if present, else skip)

**Interfaces:**
- Produces: `assemble(pack_path, box_min, box_max, species, templates, out_dir, use_bentopy=True) -> dict` with `{gro, top, placements_json, n_beads, n_molecules, per_species_counts, box_nm}`. Picks bentopy render when `bentopy_available()` and `use_bentopy`, else pure-Python writers.

- [ ] **Step 1:** integration test: `@pytest.mark.skipif(not os.path.exists(ECOLI_PACK))` — load real pack, pick a 1500 Å sub-box near the cell center, select the 6 allow-list species, build **stub** templates (e.g. 5-bead placeholders) just to exercise stamping at real positions; assert `n_beads == Σ counts×5` and that a stamped groel instance centroid ≈ its placement position/10.
- [ ] **Step 2–4:** fail → implement `assemble` (slice → to_bentopy_placements or stamp_all → write files) + the CLI script → pass.
- [ ] **Step 5: Commit + push** — `"feat(parsimony): assemble() + assemble_slice CLI"`.

---

### Task 2: Structure resolution via pbg-parsimony (real network)

**Files:**
- Modify: `pbg_martini/parsimony_assembler.py`
- Test: `tests/test_parsimony_assembler.py` (network-marked)

**Interfaces:**
- Produces: `resolve_structure(species_name, cache_dir=".cache/structures") -> str` (path to atomistic PDB/mmCIF). Map allow-list names → ids the resolver understands: prefer pbg-parsimony's `structures` module (`from pbg_parsimony import structures`); for EcoCyc `EG…-MONOMER` resolve via its EcoCyc→UniProt→AlphaFold path, for `groel` use RCSB (e.g. `1GRL`/`5W0S`). Inspect `~/code/pbg-parsimony/pbg_parsimony/structures.py` for the exact public function and reuse it; do NOT reimplement fetching.

- [ ] **Step 1:** network test (`@pytest.mark.skipif(os.environ.get("OFFLINE"))`): `resolve_structure("groel")` returns an existing file > 1 KB.
- [ ] **Step 2–4:** fail → implement (call pbg-parsimony resolver; cache) → pass. If pbg-parsimony's API differs, adapt; the deliverable is a cached atomistic structure path per species.
- [ ] **Step 5: Commit + push** — `"feat(parsimony): structure resolution via pbg-parsimony"`.

---

### Task 3: CG template via martinize2

**Files:**
- Modify: `pbg_martini/parsimony_assembler.py`
- Test: `tests/test_parsimony_assembler.py` (slow-marked)

**Interfaces:**
- Produces: `martinize_species(species_name, pdb_path, out_dir, elastic=True) -> CGTemplate`. Reuse `pbg_martini.run_martinize_pipeline(pdb_text=...)`; parse its returned CG structure into `beads_nm` (centered at centroid), write the `.itp`, set `n_beads`.

- [ ] **Step 1:** slow test: martinize a small bundled PDB (or the resolved GAPDH) → `CGTemplate.n_beads > 0`, `.itp` file exists and contains `[ moleculetype ]`.
- [ ] **Step 2–4:** fail → implement (call run_martinize_pipeline with martini3001 + elastic network; recenter beads to origin) → pass. **NOTE:** confirm `run_martinize_pipeline` returns coordinates + itp text; if it returns only stats, extend it minimally to also return the CG `.pdb`/`.gro` text and `.itp` text.
- [ ] **Step 5: Commit + push** — `"feat(parsimony): martinize species -> CG template"`.

---

### Task 8: WCA relax wrapper

**Files:**
- Modify: `pbg_martini/parsimony_assembler.py` (or new `pbg_martini/parsimony_md.py`)
- Test: `tests/test_parsimony_assembler.py`

**Interfaces:**
- Produces: `relax_assembly(coords_nm: np.ndarray, n_steps=400) -> np.ndarray` — wraps `pbg_martini.builders.relax_structure` to remove packing clashes; returns relaxed coords. Min pairwise distance must not decrease.

- [ ] **Step 1:** test: two beads at 0.1 nm separation relax to ≥ the WCA sigma; no NaN.
- [ ] **Step 2–4:** fail → implement (adapt relax_structure to accept a raw coord array) → pass.
- [ ] **Step 5: Commit + push** — `"feat(parsimony): WCA relax wrapper"`.

---

### Task 9: OpenMM short MD (best-effort)

**Files:**
- Create: `pbg_martini/parsimony_md.py` (if not already)
- Test: `tests/test_parsimony_md.py` (skip if `openmm` import fails)

**Interfaces:**
- Produces: `run_short_md(gro_path, top_path, steps=200, out_traj="traj.dcd") -> dict` with `{minimized_gro, traj, final_energy}`. Build via `openmm.app.GromacsGroFile` + `GromacsTopFile(..., includeDir=<martini ff dir>)`; Martini nonbonded (CutoffPeriodic, ~1.1 nm), `LangevinMiddleIntegrator`, dt 20 fs; local energy minimize then `steps` of NVT.

- [ ] **Step 1:** `pip install openmm` into `.venv` (best-effort; if it fails on this host, mark the test xfail/skip and move on — do NOT block). `martini-forcefields` provides the `.itp` include dir; `pip install martini-forcefields` or locate its data dir.
- [ ] **Step 2:** test (skip-if-no-openmm): build System from the Task 7 assembled files (stub or real templates) and run 10 steps → final energy is finite.
- [ ] **Step 3–4:** implement → pass (or skip cleanly).
- [ ] **Step 5: Commit + push** — `"feat(parsimony): OpenMM short MD (best-effort)"`.

---

### Task 10: PBG Steps + composite

**Files:**
- Modify: `pbg_martini/processes.py` (add Steps) and `pbg_martini/__init__.py` (exports)
- Create: `pbg_martini/composites/parsimony-whole-cell.composite.yaml`
- Test: `tests/test_parsimony_composite.py`

**Interfaces:** wrap Tasks 1–9 as `ParsimonySliceStep`, `MartinizeSpeciesStep`, `ParsimonyAssembleStep`, `MartiniMDStep`, following the existing `MembraneBuilderStep` pattern in this repo (register via `core.register_link`). The composite yaml wires slice → martinize(per species) → assemble → relax/MD.

- [ ] **Step 1:** test: `allocate_core()`, register the steps, run the composite for the synthetic/stub pack a single update; assert it emits `system.gro` path + `n_beads`.
- [ ] **Step 2–4:** fail → implement → pass (follow `composites/__init__.py` conventions already in the repo).
- [ ] **Step 5: Commit + push** — `"feat(parsimony): PBG steps + whole-cell composite"`.

---

### Task 11: Viewer + HTML report

**Files:**
- Modify: `pbg_martini/visualizations.py`
- Create: `scripts/parsimony_report.py`
- Test: smoke only.

**Interfaces:** `build_parsimony_report(assembly_summary, relaxed_gro, traj=None, out_html) -> str`. Reuse the repo's existing 3D CG viewer/visualization helpers to render the assembled slice; report shows per-species counts, n_beads, box, clash-before/after, and (if MD ran) final energy + a frame.

- [ ] **Step 1–4:** smoke test that the html is written and non-empty for the stub assembly; implement using existing visualization functions.
- [ ] **Step 5: Commit + push** — `"feat(parsimony): 3D viewer + HTML report"`.

---

### Task 12: Demo + README

**Files:**
- Create: `demo/parsimony_cell_demo.py`
- Modify: `README.md`

- [ ] **Step 1–4:** end-to-end demo over a small real sub-box (resolve→martinize→assemble→relax→[MD if available]→report), wrapped in try/except so missing openmm/bentopy degrade gracefully; README section "parsimony → Martini whole-cell MD" with the command and a note on deferred nucleic acids + full-cell scaling.
- [ ] **Step 5: Commit + push** — `"docs(parsimony): demo + README"`.

---

## Self-Review notes

- Spec coverage: Stage 1→Task 1+2, Stage 2→Task 3, Stage 3→Tasks 4/5/6/7, Stage 4→Tasks 8/9, composite→Task 10, viewer/report→Task 11, demo→Task 12. All spec sections covered.
- Tool-uncertainty is explicit and bounded: bentopy schema (Task 5) and martinize return shape (Task 3) and quaternion order (Task 4) are discovered at runtime against the real tools, each with a fallback that keeps the deterministic core green.
- If turn-capped: the offline core (Tasks 1,4,5,6,7) is the minimum shippable bridge and must be committed+pushed first.
