# pbg-martini

Process-bigraph wrapper for the [Martini](https://cgmartini.nl) coarse-grained
force field, using the [vermouth/martinize2](https://github.com/marrink-lab/vermouth-martinize)
Python library. Converts atomistic protein structures (PDB format) to
coarse-grained Martini representations as a single `Step` in the
process-bigraph framework.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_martini import MartinizeStep, make_martinize_document

core = allocate_core()
core.register_link('MartinizeStep', MartinizeStep)
core.register_link('ram-emitter', RAMEmitter)

pdb_text = open('my_protein.pdb').read()
doc = make_martinize_document(pdb_text=pdb_text, to_ff='martini3001')
sim = Composite({'state': doc}, core=core)
sim.run(0)

stores = sim.state['stores']
print(f"Beads: {stores['n_beads']}, Bonds: {stores['n_bonds_cg']}")
print(f"Reduction: {stores['reduction_ratio']}x")
```

Or use the pipeline function directly:

```python
from pbg_martini import run_martinize_pipeline

result = run_martinize_pipeline(pdb_text=open('protein.pdb').read())
print(result['cg_beads'])       # List of bead dicts
print(result['cg_positions'])   # List of [x, y, z] coordinates
print(result['bead_type_counts'])  # {'SP2': 5, 'SC3': 3, ...}
```

## API Reference

### MartinizeStep

| Config | Type | Default | Description |
|--------|------|---------|-------------|
| `from_ff` | string | `'charmm'` | Source atomistic force field |
| `to_ff` | string | `'martini3001'` | Target Martini force field |
| `delete_unknown` | boolean | `true` | Delete residues without known mapping |
| `ignh` | boolean | `false` | Ignore hydrogen atoms in input |

**Inputs:**

| Port | Type | Description |
|------|------|-------------|
| `pdb_text` | string | PDB-format text of atomistic structure |

**Outputs:**

| Port | Type | Description |
|------|------|-------------|
| `cg_beads` | list | List of bead dicts (atomname, atype, resname, resid, chain) |
| `cg_positions` | list | List of [x, y, z] CG bead coordinates (nm) |
| `cg_bonds` | list | List of [i, j] bond index pairs |
| `interactions` | map | Interaction type counts (bonds, angles, constraints, ...) |
| `bead_type_counts` | map | Count of each Martini bead type |
| `residue_counts` | map | Count of beads per residue type |
| `n_atoms_input` | integer | Number of atoms in the input PDB |
| `n_atoms_full` | integer | Number of atoms after graph repair |
| `n_beads` | integer | Number of CG beads produced |
| `n_bonds_cg` | integer | Number of CG bonds |
| `reduction_ratio` | float | Atom-to-bead reduction ratio |

### Supported Force Fields

Target: `martini3001` (default), `martini22`, `martini22p`, `martini30dev`,
`elnedyn21`, `elnedyn22`, `elnedyn22p`, `martini30b32`, `martini3IDP`

Source: `charmm` (default), `amber`, `gromos`

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Composite                        │
│                                                      │
│  ┌──────────────┐      ┌─────────────────────────┐  │
│  │ MartinizeStep│──────│         stores           │  │
│  │              │      │  pdb_text (input)        │  │
│  │  vermouth/   │─────▶│  cg_beads               │  │
│  │  martinize2  │      │  cg_positions            │  │
│  │  pipeline    │      │  cg_bonds                │  │
│  │              │      │  bead_type_counts         │  │
│  └──────────────┘      │  n_beads, reduction_ratio│  │
│                        └─────────────────────────┘  │
│  ┌──────────────┐              │                     │
│  │  RAMEmitter  │◀─────────────┘                     │
│  └──────────────┘                                    │
└──────────────────────────────────────────────────────┘
```

## Demo

```bash
python demo/demo_report.py
```

Generates `demo/report.html` — an interactive report with Three.js 3D bead
viewers, Plotly charts, bigraph-viz architecture diagrams, and PBG document
trees for three peptide configurations.

## Tests

```bash
pytest tests/ -v
```
