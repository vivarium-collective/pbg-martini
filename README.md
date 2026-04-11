# pbg-martini

Process-bigraph wrapper for the [Martini](https://cgmartini.nl) coarse-grained
force field, using [vermouth/martinize2](https://github.com/marrink-lab/vermouth-martinize)
and procedural membrane builders. Provides PBG Steps for:

- **Protein coarse-graining** (atomistic PDB → Martini CG via martinize2)
- **Lipid bilayer membranes** (POPC, POPE, CHOL, SM, DPPC)
- **Micelle self-assembly** (DPC and other detergents)
- **Protein–membrane complexes** (transmembrane helices in bilayers)
- **Vesicles / liposomes** (spherical bilayers with inner/outer leaflets)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

### Protein coarse-graining

```python
from pbg_martini import run_martinize_pipeline

result = run_martinize_pipeline(pdb_text=open('protein.pdb').read())
print(f"{result['n_beads']} CG beads, {result['reduction_ratio']}x reduction")
```

### Membrane building

```python
from pbg_martini import build_bilayer

membrane = build_bilayer(
    composition={'POPC': 0.4, 'POPE': 0.25, 'CHOL': 0.2, 'SM': 0.15},
    nx_lipids=14, ny_lipids=14,
)
print(f"{membrane['stats']['n_lipids']} lipids, {membrane['stats']['n_beads']} beads")
```

### PBG Step integration

```python
from process_bigraph import allocate_core
from pbg_martini import MembraneBuilderStep

core = allocate_core()
core.register_link('MembraneBuilderStep', MembraneBuilderStep)

step = MembraneBuilderStep(
    config={'composition': {'POPC': 0.7, 'CHOL': 0.3}, 'nx': 10, 'ny': 10},
    core=core,
)
result = step.update({})
```

## API Reference

### Steps

| Step | Description | Key Config |
|------|-------------|------------|
| `MartinizeStep` | Atomistic → CG protein mapping | `to_ff`, `from_ff` |
| `MembraneBuilderStep` | Flat lipid bilayer patch | `composition`, `nx`, `ny`, `spacing` |
| `MicelleBuilderStep` | Spherical micelle | `lipid`, `n_lipids`, `radius` |
| `ProteinMembraneStep` | TM helix in bilayer | `composition`, `n_helix_residues` |
| `VesicleBuilderStep` | Spherical vesicle | `composition`, `n_outer`, `n_inner` |

### Builder Functions

| Function | Description |
|----------|-------------|
| `run_martinize_pipeline(pdb_text, ...)` | Full martinize2 pipeline |
| `build_bilayer(composition, ...)` | Lipid bilayer with mixed composition |
| `build_micelle(lipid_name, ...)` | Spherical micelle |
| `build_protein_in_membrane(composition, ...)` | Helix embedded in bilayer |
| `build_vesicle(composition, ...)` | Liposome with two leaflets |

### Supported Lipids

| Lipid | Beads | Category |
|-------|-------|----------|
| POPC | 12 | Phospholipid |
| POPE | 12 | Phospholipid |
| DPPC | 12 | Phospholipid |
| SM | 12 | Sphingolipid |
| CHOL | 8 | Sterol |
| DPC | 6 | Detergent |

## Demo

```bash
python demo/demo_report.py
```

Generates `demo/report.html` — an interactive report with four complex systems:

1. **Asymmetric plasma membrane** — 392 lipids (POPC/POPE/CHOL/SM), 4,312 beads
2. **DPC micelle** — 80 detergent molecules, 480 beads
3. **WALP23 in bilayer** — transmembrane helix + 320 lipids, 4,202 beads
4. **Mixed-lipid vesicle** — 570 lipids across two leaflets, 6,384 beads

Each section features instanced Three.js 3D bead rendering on a dark background,
Plotly composition charts, bigraph-viz architecture diagrams, and collapsible
PBG document trees.

## Tests

```bash
pytest tests/ -v  # 23 tests
```
