"""Demo: Martini coarse-graining multi-config report with 3D viewers.

Runs three distinct martinization pipelines (polyalanine, mixed peptide,
charged peptide), generates interactive 3D bead viewers with Three.js,
Plotly charts, bigraph-viz diagrams, and navigatable PBG document trees
— all in a single self-contained HTML.
"""

import json
import os
import sys
import time
import base64
import tempfile
import subprocess
import numpy as np
from process_bigraph import allocate_core
from pbg_martini.processes import MartinizeStep, run_martinize_pipeline
from pbg_martini.composites import make_martinize_document


# ── PDB Structures ──────────────────────────────────────────────────
# Small peptides with realistic backbone geometry (phi/psi angles)
# All coordinates in Angstroms

# Config 1: Poly-alanine helix (8 residues) — regular alpha-helix
HELIX_PDB = """\
ATOM      1  N   ALA A   1       1.458   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.530   1.462   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       4.148   0.402   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.459   2.152   1.226  1.00  0.00           C
ATOM      6  N   ALA A   2       4.115   2.670   0.070  1.00  0.00           N
ATOM      7  CA  ALA A   2       5.566   2.816   0.134  1.00  0.00           C
ATOM      8  C   ALA A   2       6.115   2.069   1.347  1.00  0.00           C
ATOM      9  O   ALA A   2       5.343   1.463   2.093  1.00  0.00           O
ATOM     10  CB  ALA A   2       5.926   4.301   0.199  1.00  0.00           C
ATOM     11  N   ALA A   3       7.424   2.125   1.530  1.00  0.00           N
ATOM     12  CA  ALA A   3       8.099   1.444   2.629  1.00  0.00           C
ATOM     13  C   ALA A   3       8.003  -0.076   2.489  1.00  0.00           C
ATOM     14  O   ALA A   3       7.806  -0.574   1.376  1.00  0.00           O
ATOM     15  CB  ALA A   3       9.569   1.868   2.676  1.00  0.00           C
ATOM     16  N   ALA A   4       8.128  -0.771   3.621  1.00  0.00           N
ATOM     17  CA  ALA A   4       8.048  -2.227   3.607  1.00  0.00           C
ATOM     18  C   ALA A   4       6.605  -2.700   3.418  1.00  0.00           C
ATOM     19  O   ALA A   4       5.684  -1.879   3.514  1.00  0.00           O
ATOM     20  CB  ALA A   4       8.567  -2.800   4.931  1.00  0.00           C
ATOM     21  N   ALA A   5       6.420  -3.992   3.154  1.00  0.00           N
ATOM     22  CA  ALA A   5       5.081  -4.555   2.956  1.00  0.00           C
ATOM     23  C   ALA A   5       4.627  -4.307   1.517  1.00  0.00           C
ATOM     24  O   ALA A   5       5.432  -4.260   0.587  1.00  0.00           O
ATOM     25  CB  ALA A   5       5.099  -6.053   3.261  1.00  0.00           C
ATOM     26  N   ALA A   6       3.327  -4.120   1.346  1.00  0.00           N
ATOM     27  CA  ALA A   6       2.749  -3.857  -0.000  1.00  0.00           C
ATOM     28  C   ALA A   6       2.781  -2.356  -0.285  1.00  0.00           C
ATOM     29  O   ALA A   6       2.837  -1.525   0.620  1.00  0.00           O
ATOM     30  CB  ALA A   6       1.301  -4.339  -0.057  1.00  0.00           C
ATOM     31  N   ALA A   7       2.775  -2.021  -1.575  1.00  0.00           N
ATOM     32  CA  ALA A   7       2.801  -0.621  -1.985  1.00  0.00           C
ATOM     33  C   ALA A   7       4.186  -0.018  -1.758  1.00  0.00           C
ATOM     34  O   ALA A   7       5.125  -0.749  -1.466  1.00  0.00           O
ATOM     35  CB  ALA A   7       2.395  -0.514  -3.458  1.00  0.00           C
ATOM     36  N   ALA A   8       4.272   1.302  -1.870  1.00  0.00           N
ATOM     37  CA  ALA A   8       5.543   2.019  -1.683  1.00  0.00           C
ATOM     38  C   ALA A   8       6.018   1.868  -0.235  1.00  0.00           C
ATOM     39  O   ALA A   8       5.217   1.880   0.698  1.00  0.00           O
ATOM     40  CB  ALA A   8       5.356   3.504  -1.992  1.00  0.00           C
END
"""

# Config 2: Mixed hydrophobic peptide AVILMFP (7 residues)
MIXED_PDB = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.450   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.400   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.400   3.400   1.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       3.000   0.200   2.200  1.00  0.00           C
ATOM      6  N   VAL A   2       4.300   2.400   1.000  1.00  0.00           N
ATOM      7  CA  VAL A   2       5.000   3.700   1.000  1.00  0.00           C
ATOM      8  C   VAL A   2       6.500   3.700   1.000  1.00  0.00           C
ATOM      9  O   VAL A   2       7.100   4.700   1.000  1.00  0.00           O
ATOM     10  CB  VAL A   2       4.500   4.500   2.200  1.00  0.00           C
ATOM     11  CG1 VAL A   2       4.900   5.970   2.100  1.00  0.00           C
ATOM     12  CG2 VAL A   2       3.000   4.400   2.400  1.00  0.00           C
ATOM     13  N   ILE A   3       7.100   2.600   1.000  1.00  0.00           N
ATOM     14  CA  ILE A   3       8.500   2.500   1.000  1.00  0.00           C
ATOM     15  C   ILE A   3       9.100   3.800   1.000  1.00  0.00           C
ATOM     16  O   ILE A   3       8.500   4.800   1.000  1.00  0.00           O
ATOM     17  CB  ILE A   3       9.000   1.700   2.200  1.00  0.00           C
ATOM     18  CG1 ILE A   3      10.500   1.600   2.200  1.00  0.00           C
ATOM     19  CG2 ILE A   3       8.400   0.300   2.200  1.00  0.00           C
ATOM     20  CD1 ILE A   3      11.100   0.300   2.700  1.00  0.00           C
ATOM     21  N   LEU A   4      10.400   3.700   1.000  1.00  0.00           N
ATOM     22  CA  LEU A   4      11.200   4.900   1.000  1.00  0.00           C
ATOM     23  C   LEU A   4      12.700   4.800   1.000  1.00  0.00           C
ATOM     24  O   LEU A   4      13.300   5.800   1.000  1.00  0.00           O
ATOM     25  CB  LEU A   4      10.700   6.000   2.000  1.00  0.00           C
ATOM     26  CG  LEU A   4      11.200   7.400   2.000  1.00  0.00           C
ATOM     27  CD1 LEU A   4      10.600   8.100   3.200  1.00  0.00           C
ATOM     28  CD2 LEU A   4      12.700   7.500   2.000  1.00  0.00           C
ATOM     29  N   MET A   5      13.300   3.600   1.000  1.00  0.00           N
ATOM     30  CA  MET A   5      14.700   3.400   1.000  1.00  0.00           C
ATOM     31  C   MET A   5      15.300   4.700   1.000  1.00  0.00           C
ATOM     32  O   MET A   5      14.700   5.700   1.000  1.00  0.00           O
ATOM     33  CB  MET A   5      15.200   2.600   2.200  1.00  0.00           C
ATOM     34  CG  MET A   5      16.700   2.500   2.300  1.00  0.00           C
ATOM     35  SD  MET A   5      17.200   1.700   3.800  1.00  0.00           S
ATOM     36  CE  MET A   5      18.900   1.500   3.500  1.00  0.00           C
ATOM     37  N   PHE A   6      16.600   4.700   1.000  1.00  0.00           N
ATOM     38  CA  PHE A   6      17.300   5.900   1.000  1.00  0.00           C
ATOM     39  C   PHE A   6      18.800   5.800   1.000  1.00  0.00           C
ATOM     40  O   PHE A   6      19.400   6.800   1.000  1.00  0.00           O
ATOM     41  CB  PHE A   6      16.800   6.800   2.150  1.00  0.00           C
ATOM     42  CG  PHE A   6      17.300   8.200   2.100  1.00  0.00           C
ATOM     43  CD1 PHE A   6      18.500   8.500   2.700  1.00  0.00           C
ATOM     44  CD2 PHE A   6      16.600   9.200   1.400  1.00  0.00           C
ATOM     45  CE1 PHE A   6      19.000   9.800   2.700  1.00  0.00           C
ATOM     46  CE2 PHE A   6      17.100  10.500   1.400  1.00  0.00           C
ATOM     47  CZ  PHE A   6      18.300  10.800   2.000  1.00  0.00           C
ATOM     48  N   PRO A   7      19.400   4.600   1.000  1.00  0.00           N
ATOM     49  CA  PRO A   7      20.800   4.400   1.000  1.00  0.00           C
ATOM     50  C   PRO A   7      21.500   5.700   1.000  1.00  0.00           C
ATOM     51  O   PRO A   7      20.900   6.700   1.000  1.00  0.00           O
ATOM     52  CB  PRO A   7      21.100   3.600   2.280  1.00  0.00           C
ATOM     53  CG  PRO A   7      19.900   2.700   2.500  1.00  0.00           C
ATOM     54  CD  PRO A   7      18.800   3.500   1.800  1.00  0.00           C
END
"""

# Config 3: Charged/polar peptide KERDQN (6 residues) — includes charged + polar
CHARGED_PDB = """\
ATOM      1  N   LYS A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  LYS A   1       2.450   1.000   1.000  1.00  0.00           C
ATOM      3  C   LYS A   1       3.000   2.400   1.000  1.00  0.00           C
ATOM      4  O   LYS A   1       2.400   3.400   1.000  1.00  0.00           O
ATOM      5  CB  LYS A   1       3.000   0.200   2.200  1.00  0.00           C
ATOM      6  CG  LYS A   1       4.500   0.100   2.300  1.00  0.00           C
ATOM      7  CD  LYS A   1       5.000  -0.700   3.500  1.00  0.00           C
ATOM      8  CE  LYS A   1       6.500  -0.800   3.600  1.00  0.00           C
ATOM      9  NZ  LYS A   1       7.000  -1.600   4.800  1.00  0.00           N
ATOM     10  N   GLU A   2       4.300   2.400   1.000  1.00  0.00           N
ATOM     11  CA  GLU A   2       5.000   3.700   1.000  1.00  0.00           C
ATOM     12  C   GLU A   2       6.500   3.700   1.000  1.00  0.00           C
ATOM     13  O   GLU A   2       7.100   4.700   1.000  1.00  0.00           O
ATOM     14  CB  GLU A   2       4.500   4.500   2.200  1.00  0.00           C
ATOM     15  CG  GLU A   2       4.900   5.970   2.100  1.00  0.00           C
ATOM     16  CD  GLU A   2       4.400   6.770   3.300  1.00  0.00           C
ATOM     17  OE1 GLU A   2       3.200   6.700   3.600  1.00  0.00           O
ATOM     18  OE2 GLU A   2       5.200   7.470   3.900  1.00  0.00           O
ATOM     19  N   ARG A   3       7.100   2.600   1.000  1.00  0.00           N
ATOM     20  CA  ARG A   3       8.500   2.500   1.000  1.00  0.00           C
ATOM     21  C   ARG A   3       9.100   3.800   1.000  1.00  0.00           C
ATOM     22  O   ARG A   3       8.500   4.800   1.000  1.00  0.00           O
ATOM     23  CB  ARG A   3       9.000   1.700   2.200  1.00  0.00           C
ATOM     24  CG  ARG A   3      10.500   1.600   2.300  1.00  0.00           C
ATOM     25  CD  ARG A   3      11.000   0.800   3.500  1.00  0.00           C
ATOM     26  NE  ARG A   3      12.400   0.700   3.600  1.00  0.00           N
ATOM     27  CZ  ARG A   3      13.100   0.000   4.500  1.00  0.00           C
ATOM     28  NH1 ARG A   3      12.500  -0.700   5.400  1.00  0.00           N
ATOM     29  NH2 ARG A   3      14.400   0.000   4.400  1.00  0.00           N
ATOM     30  N   ASP A   4      10.400   3.700   1.000  1.00  0.00           N
ATOM     31  CA  ASP A   4      11.200   4.900   1.000  1.00  0.00           C
ATOM     32  C   ASP A   4      12.700   4.800   1.000  1.00  0.00           C
ATOM     33  O   ASP A   4      13.300   5.800   1.000  1.00  0.00           O
ATOM     34  CB  ASP A   4      10.700   6.000   2.000  1.00  0.00           C
ATOM     35  CG  ASP A   4      11.200   7.400   1.800  1.00  0.00           C
ATOM     36  OD1 ASP A   4      10.600   8.100   0.900  1.00  0.00           O
ATOM     37  OD2 ASP A   4      12.100   7.800   2.500  1.00  0.00           O
ATOM     38  N   GLN A   5      13.300   3.600   1.000  1.00  0.00           N
ATOM     39  CA  GLN A   5      14.700   3.400   1.000  1.00  0.00           C
ATOM     40  C   GLN A   5      15.300   4.700   1.000  1.00  0.00           C
ATOM     41  O   GLN A   5      14.700   5.700   1.000  1.00  0.00           O
ATOM     42  CB  GLN A   5      15.200   2.600   2.200  1.00  0.00           C
ATOM     43  CG  GLN A   5      16.700   2.500   2.300  1.00  0.00           C
ATOM     44  CD  GLN A   5      17.200   1.700   3.500  1.00  0.00           C
ATOM     45  OE1 GLN A   5      16.500   1.000   4.200  1.00  0.00           O
ATOM     46  NE2 GLN A   5      18.500   1.800   3.700  1.00  0.00           N
ATOM     47  N   ASN A   6      16.600   4.700   1.000  1.00  0.00           N
ATOM     48  CA  ASN A   6      17.300   5.900   1.000  1.00  0.00           C
ATOM     49  C   ASN A   6      18.800   5.800   1.000  1.00  0.00           C
ATOM     50  O   ASN A   6      19.400   6.800   1.000  1.00  0.00           O
ATOM     51  CB  ASN A   6      16.800   6.800   2.150  1.00  0.00           C
ATOM     52  CG  ASN A   6      17.300   8.200   2.000  1.00  0.00           C
ATOM     53  OD1 ASN A   6      18.400   8.400   1.500  1.00  0.00           O
ATOM     54  ND2 ASN A   6      16.500   9.200   2.400  1.00  0.00           N
END
"""

# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'helix',
        'title': 'Poly-Alanine Helix',
        'subtitle': 'Alpha-helical backbone mapped to Martini 3',
        'description': (
            'An eight-residue poly-alanine peptide in an alpha-helical '
            'conformation is coarse-grained using the Martini 3.0.0.1 force '
            'field. Each alanine maps to two CG beads (BB backbone + SC1 '
            'sidechain), producing a compact helical representation. This is '
            'the simplest test case for the martinize2 pipeline.'
        ),
        'pdb_text': HELIX_PDB,
        'to_ff': 'martini3001',
        'color_scheme': 'indigo',
    },
    {
        'id': 'mixed',
        'title': 'Hydrophobic Peptide AVILMFP',
        'subtitle': 'Seven diverse hydrophobic residues with different CG topologies',
        'description': (
            'A heptapeptide containing Ala, Val, Ile, Leu, Met, Phe, and Pro '
            'showcases the variety of CG mapping rules. Phenylalanine maps to '
            'four beads (one BB + three ring beads), while smaller residues '
            'like Ala and Gly map to fewer. The CG topology includes backbone '
            'bonds plus sidechain constraints and angles.'
        ),
        'pdb_text': MIXED_PDB,
        'to_ff': 'martini3001',
        'color_scheme': 'emerald',
    },
    {
        'id': 'charged',
        'title': 'Charged Peptide KERDQN',
        'subtitle': 'Charged and polar residues in Martini 3 vs Martini 2.2',
        'description': (
            'A hexapeptide with charged (Lys, Glu, Arg, Asp) and polar '
            '(Gln, Asn) residues demonstrates how the Martini force field '
            'handles electrostatics at coarse resolution. Charged residues '
            'like Lys and Arg map to three CG beads including a charged '
            'terminal bead. We compare Martini 3 and Martini 2.2 mappings.'
        ),
        'pdb_text': CHARGED_PDB,
        'to_ff': 'martini3001',
        'color_scheme': 'rose',
    },
]


# ── Bead color map ──────────────────────────────────────────────────
# Assign a color to each bead type category for 3D visualization.
# Martini 3 types: P (polar), N (intermediate), C (apolar), Q (charged),
# S (small), T (tiny)

BEAD_COLORS = {
    # Charged (Q) — red family
    'Q': [0.90, 0.20, 0.20],
    # Polar (P) — blue family
    'P': [0.25, 0.45, 0.90],
    # Intermediate (N) — cyan/teal
    'N': [0.20, 0.75, 0.70],
    # Apolar (C) — yellow-green
    'C': [0.65, 0.80, 0.20],
    # Small (S) — purple
    'S': [0.55, 0.35, 0.85],
    # Tiny (T) — orange
    'T': [0.95, 0.60, 0.20],
    # Default
    '_': [0.60, 0.60, 0.60],
}


def get_bead_color(atype):
    """Map a Martini bead type string to an RGB triplet."""
    if not atype:
        return BEAD_COLORS['_']
    first = atype[0].upper()
    return BEAD_COLORS.get(first, BEAD_COLORS['_'])


def run_config(cfg):
    """Run martinization for a single config and return results + runtime."""
    t0 = time.perf_counter()
    result = run_martinize_pipeline(
        pdb_text=cfg['pdb_text'],
        to_ff_name=cfg['to_ff'],
    )
    runtime = time.perf_counter() - t0

    # Also run Martini 2.2 for the charged peptide comparison
    m22_result = None
    if cfg['id'] == 'charged':
        m22_result = run_martinize_pipeline(
            pdb_text=cfg['pdb_text'],
            to_ff_name='martini22',
        )

    return result, runtime, m22_result


def generate_bigraph_image(cfg):
    """Generate a colored bigraph-viz PNG for the composite document."""
    from bigraph_viz import plot_bigraph

    doc = {
        'martinize': {
            '_type': 'step',
            'address': 'local:MartinizeStep',
            'config': {'to_ff': cfg['to_ff']},
            'inputs': {'pdb_text': ['stores', 'pdb_text']},
            'outputs': {
                'cg_beads': ['stores', 'cg_beads'],
                'cg_positions': ['stores', 'cg_positions'],
                'cg_bonds': ['stores', 'cg_bonds'],
                'n_beads': ['stores', 'n_beads'],
                'reduction_ratio': ['stores', 'reduction_ratio'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'inputs': {
                'n_beads': ['stores', 'n_beads'],
                'reduction_ratio': ['stores', 'reduction_ratio'],
            },
        },
    }

    node_colors = {
        ('martinize',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg):
    """Build the PBG composite document dict for display."""
    return make_martinize_document(
        pdb_text='<PDB text...>',
        to_ff=cfg['to_ff'],
    )


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
}


def generate_html(sim_results, output_path):
    """Generate comprehensive HTML report."""

    sections_html = []
    all_js_data = {}

    for idx, (cfg, (result, runtime, m22_result)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]

        # Build JS data for 3D viewer
        beads_js = []
        for i, bead in enumerate(result['cg_beads']):
            pos = result['cg_positions'][i]
            color = get_bead_color(bead['atype'])
            beads_js.append({
                'pos': pos,
                'color': color,
                'label': f"{bead['resname']}-{bead['atomname']}",
                'atype': bead['atype'],
                'resid': bead.get('resid', 0),
            })

        bonds_js = result['cg_bonds']

        # Bead type distribution for chart
        btc = result['bead_type_counts']
        sorted_types = sorted(btc.items(), key=lambda x: -x[1])

        # Residue distribution
        rc = result['residue_counts']
        sorted_res = sorted(rc.items(), key=lambda x: -x[1])

        # Interaction summary
        inter = result['interactions']

        # M22 comparison data (for charged config)
        m22_data = None
        if m22_result:
            m22_data = {
                'n_beads': m22_result['n_beads'],
                'n_bonds_cg': m22_result['n_bonds_cg'],
                'bead_type_counts': m22_result['bead_type_counts'],
                'interactions': m22_result['interactions'],
            }

        all_js_data[sid] = {
            'beads': beads_js,
            'bonds': bonds_js,
            'bead_types': dict(sorted_types),
            'residue_counts': dict(sorted_res),
            'interactions': inter,
            'm22': m22_data,
        }

        # Bigraph image
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        # PBG document JSON
        pbg_doc = build_pbg_document(cfg)

        # Metrics
        n_a_in = result['n_atoms_input']
        n_a_full = result['n_atoms_full']
        n_beads = result['n_beads']
        n_bonds = result['n_bonds_cg']
        ratio = result['reduction_ratio']
        n_residues = len(set(b['resid'] for b in result['cg_beads']))

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Input Atoms</span><span class="metric-value">{n_a_in}</span></div>
        <div class="metric"><span class="metric-label">Full Atoms</span><span class="metric-value">{n_a_full}</span><span class="metric-sub">after repair</span></div>
        <div class="metric"><span class="metric-label">CG Beads</span><span class="metric-value">{n_beads}</span></div>
        <div class="metric"><span class="metric-label">CG Bonds</span><span class="metric-value">{n_bonds}</span></div>
        <div class="metric"><span class="metric-label">Reduction</span><span class="metric-value">{ratio:.1f}x</span><span class="metric-sub">{n_a_full} &rarr; {n_beads}</span></div>
        <div class="metric"><span class="metric-label">Residues</span><span class="metric-value">{n_residues}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.2f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Coarse-Grained Structure</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="bead-canvas"></canvas>
        <div class="viewer-info">
          <strong>{n_beads}</strong> beads &middot; <strong>{n_bonds}</strong> bonds<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
      </div>

      <h3 class="subsection-title">Mapping Analysis</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-beadtypes-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-residues-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-interactions-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-beadcat-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    # Navigation
    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    # PBG docs for JSON viewer
    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Martini Coarse-Graining Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:700px; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .15s; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:800px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}
.viewer-wrap {{ position:relative; background:#f1f5f9; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.bead-canvas {{ width:100%; height:450px; display:block; cursor:grab; }}
.bead-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(255,255,255,.92);
                border:1px solid #e2e8f0; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#64748b; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#1e293b; }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:280px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.legend-box {{ position:absolute; bottom:.8rem; right:.8rem; background:rgba(255,255,255,.92);
               border:1px solid #e2e8f0; border-radius:8px; padding:.6rem;
               font-size:.7rem; color:#64748b; backdrop-filter:blur(4px); }}
.legend-item {{ display:flex; align-items:center; gap:.4rem; margin:.2rem 0; }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>Martini Coarse-Graining Report</h1>
  <p>Three peptide structures coarse-grained using the <strong>Martini</strong>
  force field via <strong>vermouth/martinize2</strong>, wrapped as
  <strong>process-bigraph</strong> Steps. Each configuration demonstrates a
  distinct class of amino acid chemistry with interactive 3D bead visualization.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-martini</strong> &mdash;
  Martini Force Field + vermouth/martinize2 + process-bigraph &mdash;
  Coarse-Grained Molecular Dynamics
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Three.js Bead Viewers ───
const BEAD_RADIUS = 0.025;
const BOND_RADIUS = 0.006;

function initViewer(sid) {{
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 450;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0xf1f5f9);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.01, 100);

  // Compute center and extent
  let cx=0, cy=0, cz=0;
  d.beads.forEach(b => {{ cx+=b.pos[0]; cy+=b.pos[1]; cz+=b.pos[2]; }});
  const n = d.beads.length;
  cx/=n; cy/=n; cz/=n;
  let maxR = 0;
  d.beads.forEach(b => {{
    const dx=b.pos[0]-cx, dy=b.pos[1]-cy, dz=b.pos[2]-cz;
    maxR = Math.max(maxR, Math.sqrt(dx*dx+dy*dy+dz*dz));
  }});
  const dist = Math.max(maxR * 4, 0.3);
  cam.position.set(cx + dist*0.8, cy + dist*0.6, cz + dist);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.target.set(cx, cy, cz);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.2;

  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.7);
  dl1.position.set(3,5,4); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0xcbd5e1, 0.4);
  dl2.position.set(-3,-2,-4); scene.add(dl2);

  // Add beads as spheres
  const sphereGeo = new THREE.SphereGeometry(BEAD_RADIUS, 16, 12);
  d.beads.forEach(b => {{
    const mat = new THREE.MeshPhongMaterial({{
      color: new THREE.Color(b.color[0], b.color[1], b.color[2]),
      shininess: 60,
    }});
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.position.set(b.pos[0], b.pos[1], b.pos[2]);
    scene.add(mesh);
  }});

  // Add bonds as cylinders
  const bondMat = new THREE.MeshPhongMaterial({{ color: 0x94a3b8, shininess: 20 }});
  d.bonds.forEach(([i, j]) => {{
    const a = d.beads[i].pos, b2 = d.beads[j].pos;
    const start = new THREE.Vector3(a[0], a[1], a[2]);
    const end = new THREE.Vector3(b2[0], b2[1], b2[2]);
    const dir = new THREE.Vector3().subVectors(end, start);
    const len = dir.length();
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    const cylGeo = new THREE.CylinderGeometry(BOND_RADIUS, BOND_RADIUS, len, 6);
    const cyl = new THREE.Mesh(cylGeo, bondMat);
    cyl.position.copy(mid);
    const axis = new THREE.Vector3(0,1,0);
    cyl.quaternion.setFromUnitVectors(axis, dir.normalize());
    scene.add(cyl);
  }});

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }}
  animate();
}}

Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ───
const catColors = {{
  'Q':'#e53e3e', 'P':'#3b82f6', 'N':'#14b8a6',
  'C':'#84cc16', 'S':'#8b5cf6', 'T':'#f59e0b', '_':'#94a3b8'
}};

const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:50, r:15, t:35, b:55 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];

  // Chart 1: Bead type counts (bar chart)
  const btKeys = Object.keys(d.bead_types);
  const btVals = btKeys.map(k => d.bead_types[k]);
  const btColors = btKeys.map(k => catColors[k[0]] || catColors['_']);
  Plotly.newPlot('chart-beadtypes-'+sid, [{{
    x:btKeys, y:btVals, type:'bar',
    marker:{{ color:btColors, line:{{ color:'#fff', width:1 }} }},
  }}], {{...pLayout,
    title:{{ text:'Bead Type Distribution', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'Bead Type', font:{{ size:10 }} }}, tickangle:-45 }},
  }}, pCfg);

  // Chart 2: Residue distribution (pie chart)
  const resKeys = Object.keys(d.residue_counts);
  const resVals = resKeys.map(k => d.residue_counts[k]);
  Plotly.newPlot('chart-residues-'+sid, [{{
    labels:resKeys, values:resVals, type:'pie',
    hole: 0.35,
    marker:{{ colors:['#6366f1','#10b981','#f43f5e','#f59e0b','#8b5cf6','#06b6d4','#84cc16','#ec4899'] }},
    textinfo:'label+value',
  }}], {{...pLayout,
    title:{{ text:'Bead Count per Residue', font:{{ size:12, color:'#334155' }} }},
    showlegend: false,
  }}, pCfg);

  // Chart 3: Interaction types (bar chart)
  const iKeys = Object.keys(d.interactions);
  const iVals = iKeys.map(k => d.interactions[k]);
  Plotly.newPlot('chart-interactions-'+sid, [{{
    x:iKeys, y:iVals, type:'bar',
    marker:{{ color:'#6366f1', line:{{ color:'#fff', width:1 }} }},
  }}], {{...pLayout,
    title:{{ text:'Interaction Types', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
    xaxis:{{...pLayout.xaxis, tickangle:-30 }},
  }}, pCfg);

  // Chart 4: Bead category breakdown (aggregated by first letter)
  const catMap = {{}};
  Object.keys(d.bead_types).forEach(bt => {{
    const cat = bt[0];
    const label = cat === 'Q' ? 'Charged' : cat === 'P' ? 'Polar' :
                  cat === 'N' ? 'Intermediate' : cat === 'C' ? 'Apolar' :
                  cat === 'S' ? 'Small' : cat === 'T' ? 'Tiny' : 'Other';
    catMap[label] = (catMap[label] || 0) + d.bead_types[bt];
  }});
  const catKeys = Object.keys(catMap);
  const catVals = catKeys.map(k => catMap[k]);
  const catClrs = catKeys.map(k => {{
    const first = k === 'Charged' ? 'Q' : k === 'Polar' ? 'P' :
                  k === 'Intermediate' ? 'N' : k === 'Apolar' ? 'C' :
                  k === 'Small' ? 'S' : k === 'Tiny' ? 'T' : '_';
    return catColors[first];
  }});
  Plotly.newPlot('chart-beadcat-'+sid, [{{
    labels:catKeys, values:catVals, type:'pie',
    marker:{{ colors:catClrs }},
    textinfo:'label+percent',
  }}], {{...pLayout,
    title:{{ text:'Bead Category Breakdown', font:{{ size:12, color:'#334155' }} }},
    showlegend: false,
  }}, pCfg);
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report written to {output_path}')


def main():
    print('Martini Coarse-Graining Demo Report')
    print('=' * 40)

    sim_results = []
    for cfg in CONFIGS:
        print(f'\nRunning config: {cfg["title"]}...')
        result, runtime, m22 = run_config(cfg)
        print(f'  {result["n_atoms_input"]} atoms -> {result["n_beads"]} beads '
              f'({result["reduction_ratio"]:.1f}x) in {runtime:.2f}s')
        sim_results.append((cfg, (result, runtime, m22)))

    output_path = os.path.join(os.path.dirname(__file__), 'report.html')
    print(f'\nGenerating HTML report...')
    generate_html(sim_results, output_path)

    # Open in Safari
    subprocess.run(['open', '-a', 'Safari', output_path])
    print('Done! Report opened in Safari.')


if __name__ == '__main__':
    main()
