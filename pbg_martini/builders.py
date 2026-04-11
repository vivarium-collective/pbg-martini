"""Procedural builders for Martini CG membrane and micelle structures.

Generates coarse-grained bead coordinates, types, and bonds for lipid
assemblies using known Martini 3 geometries. These are initial configurations
suitable for energy minimization and MD equilibration in GROMACS.
"""

import math
import numpy as np
from collections import OrderedDict


# ── Martini 3 Lipid Templates ────────────────────────────────────────
# Each template: list of (name, bead_type, relative_position_nm)
# Positions are for a single lipid oriented along +z (head up)

LIPID_TEMPLATES = {
    'POPC': {
        'beads': [
            ('NC3', 'Q1', [0.0, 0.0, 2.30]),
            ('PO4', 'Qa', [0.0, 0.0, 1.95]),
            ('GL1', 'N4a', [-0.15, 0.0, 1.50]),
            ('GL2', 'N4a', [0.15, 0.0, 1.50]),
            ('C1A', 'C1', [-0.15, 0.0, 1.10]),
            ('D2A', 'C4h', [-0.15, 0.0, 0.72]),
            ('C3A', 'C1', [-0.15, 0.0, 0.34]),
            ('C4A', 'C1', [-0.15, 0.0, -0.04]),
            ('C1B', 'C1', [0.15, 0.0, 1.10]),
            ('C2B', 'C1', [0.15, 0.0, 0.72]),
            ('C3B', 'C1', [0.15, 0.0, 0.34]),
            ('C4B', 'C1', [0.15, 0.0, -0.04]),
        ],
        'bonds': [
            (0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (5, 6), (6, 7),
            (3, 8), (8, 9), (9, 10), (10, 11),
        ],
        'color_head': [0.29, 0.53, 0.78],  # blue head
        'color_tail': [0.18, 0.35, 0.58],  # darker blue tail
        'category': 'phospholipid',
    },
    'POPE': {
        'beads': [
            ('NH3', 'Qd', [0.0, 0.0, 2.25]),
            ('PO4', 'Qa', [0.0, 0.0, 1.90]),
            ('GL1', 'N4a', [-0.15, 0.0, 1.48]),
            ('GL2', 'N4a', [0.15, 0.0, 1.48]),
            ('C1A', 'C1', [-0.15, 0.0, 1.08]),
            ('D2A', 'C4h', [-0.15, 0.0, 0.70]),
            ('C3A', 'C1', [-0.15, 0.0, 0.32]),
            ('C4A', 'C1', [-0.15, 0.0, -0.06]),
            ('C1B', 'C1', [0.15, 0.0, 1.08]),
            ('C2B', 'C1', [0.15, 0.0, 0.70]),
            ('C3B', 'C1', [0.15, 0.0, 0.32]),
            ('C4B', 'C1', [0.15, 0.0, -0.06]),
        ],
        'bonds': [
            (0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (5, 6), (6, 7),
            (3, 8), (8, 9), (9, 10), (10, 11),
        ],
        'color_head': [0.83, 0.47, 0.16],  # orange head
        'color_tail': [0.60, 0.33, 0.12],  # darker orange tail
        'category': 'phospholipid',
    },
    'CHOL': {
        'beads': [
            ('ROH', 'P1', [0.0, 0.0, 1.80]),
            ('R1', 'SC4', [0.15, 0.10, 1.40]),
            ('R2', 'SC3', [-0.15, 0.10, 1.20]),
            ('R3', 'SC3', [0.0, -0.10, 0.90]),
            ('R4', 'SC3', [0.10, 0.05, 0.55]),
            ('R5', 'SC3', [-0.10, -0.05, 0.20]),
            ('C1', 'C2', [0.0, 0.0, -0.15]),
            ('C2', 'C1', [0.0, 0.0, -0.50]),
        ],
        'bonds': [
            (0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        ],
        'color_head': [0.90, 0.78, 0.10],  # gold head
        'color_tail': [0.68, 0.58, 0.08],  # darker gold
        'category': 'sterol',
    },
    'SM': {
        'beads': [
            ('NC3', 'Q1', [0.0, 0.0, 2.35]),
            ('PO4', 'Qa', [0.0, 0.0, 2.00]),
            ('AM1', 'P2', [-0.10, 0.0, 1.60]),
            ('AM2', 'P5', [0.10, 0.0, 1.60]),
            ('T1A', 'C3', [-0.15, 0.0, 1.20]),
            ('C2A', 'C1', [-0.15, 0.0, 0.82]),
            ('C3A', 'C1', [-0.15, 0.0, 0.44]),
            ('C4A', 'C1', [-0.15, 0.0, 0.06]),
            ('C1B', 'C1', [0.15, 0.0, 1.20]),
            ('C2B', 'C1', [0.15, 0.0, 0.82]),
            ('C3B', 'C1', [0.15, 0.0, 0.44]),
            ('C4B', 'C1', [0.15, 0.0, 0.06]),
        ],
        'bonds': [
            (0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (5, 6), (6, 7),
            (3, 8), (8, 9), (9, 10), (10, 11),
        ],
        'color_head': [0.77, 0.31, 0.60],  # magenta head
        'color_tail': [0.55, 0.22, 0.43],  # darker magenta tail
        'category': 'sphingolipid',
    },
    'DPC': {
        'beads': [
            ('NC3', 'Q1', [0.0, 0.0, 1.80]),
            ('PO4', 'Qa', [0.0, 0.0, 1.45]),
            ('GL1', 'N4a', [0.0, 0.0, 1.10]),
            ('C1A', 'C1', [0.0, 0.0, 0.72]),
            ('C2A', 'C1', [0.0, 0.0, 0.34]),
            ('C3A', 'C1', [0.0, 0.0, -0.04]),
        ],
        'bonds': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        'color_head': [0.32, 0.70, 0.42],  # green head
        'color_tail': [0.22, 0.50, 0.30],  # darker green tail
        'category': 'detergent',
    },
    'DPPC': {
        'beads': [
            ('NC3', 'Q1', [0.0, 0.0, 2.30]),
            ('PO4', 'Qa', [0.0, 0.0, 1.95]),
            ('GL1', 'N4a', [-0.15, 0.0, 1.50]),
            ('GL2', 'N4a', [0.15, 0.0, 1.50]),
            ('C1A', 'C1', [-0.15, 0.0, 1.10]),
            ('C2A', 'C1', [-0.15, 0.0, 0.72]),
            ('C3A', 'C1', [-0.15, 0.0, 0.34]),
            ('C4A', 'C1', [-0.15, 0.0, -0.04]),
            ('C1B', 'C1', [0.15, 0.0, 1.10]),
            ('C2B', 'C1', [0.15, 0.0, 0.72]),
            ('C3B', 'C1', [0.15, 0.0, 0.34]),
            ('C4B', 'C1', [0.15, 0.0, -0.04]),
        ],
        'bonds': [
            (0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (5, 6), (6, 7),
            (3, 8), (8, 9), (9, 10), (10, 11),
        ],
        'color_head': [0.49, 0.67, 0.85],  # light blue head
        'color_tail': [0.33, 0.47, 0.62],  # darker light blue tail
        'category': 'phospholipid',
    },
}

# Protein helix bead colors
PROTEIN_COLORS = {
    'BB': [0.90, 0.25, 0.30],   # backbone: red
    'SC': [0.95, 0.55, 0.25],   # sidechain: orange
}


def _rotate_z(positions, angle):
    """Rotate positions around z-axis by angle (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return positions @ rot.T


def _jitter(n, scale=0.02):
    """Small random displacements to break grid symmetry."""
    rng = np.random.default_rng(42)
    return rng.normal(0, scale, (n, 3))


def build_bilayer(composition, nx_lipids=10, ny_lipids=10, spacing=0.65,
                  area_per_lipid=None, seed=123):
    """Build a lipid bilayer membrane patch.

    Parameters
    ----------
    composition : dict
        Mapping of lipid name to fraction, e.g. {'POPC': 0.4, 'POPE': 0.3, 'CHOL': 0.3}.
        Fractions are normalized automatically.
    nx_lipids, ny_lipids : int
        Grid dimensions (total lipids = 2 * nx * ny for both leaflets).
    spacing : float
        Grid spacing in nm (default 0.65 nm, typical for Martini).
    area_per_lipid : float or None
        If set, overrides spacing.
    seed : int
        Random seed for lipid placement.

    Returns
    -------
    dict with keys: beads, bonds, lipid_types, stats
    """
    rng = np.random.default_rng(seed)

    if area_per_lipid is not None:
        spacing = math.sqrt(area_per_lipid)

    # Normalize composition
    total = sum(composition.values())
    norm_comp = {k: v / total for k, v in composition.items()}

    # Build lipid assignment for one leaflet
    n_per_leaflet = nx_lipids * ny_lipids
    lipid_names = []
    for name, frac in norm_comp.items():
        count = round(frac * n_per_leaflet)
        lipid_names.extend([name] * count)
    # Pad or trim
    while len(lipid_names) < n_per_leaflet:
        lipid_names.append(list(norm_comp.keys())[0])
    lipid_names = lipid_names[:n_per_leaflet]
    rng.shuffle(lipid_names)

    all_beads = []
    all_bonds = []
    lipid_counts = {}
    bead_offset = 0

    for leaflet in ['upper', 'lower']:
        if leaflet == 'lower':
            rng.shuffle(lipid_names)

        for ix in range(nx_lipids):
            for iy in range(ny_lipids):
                idx = ix * ny_lipids + iy
                lip_name = lipid_names[idx]
                template = LIPID_TEMPLATES[lip_name]

                # Grid position
                cx = ix * spacing - (nx_lipids - 1) * spacing / 2
                cy = iy * spacing - (ny_lipids - 1) * spacing / 2

                # Random rotation around z
                angle = rng.uniform(0, 2 * math.pi)

                for bi, (bname, btype, bpos) in enumerate(template['beads']):
                    pos = np.array(bpos, dtype=float)
                    # Flip for lower leaflet
                    if leaflet == 'lower':
                        pos[2] = -pos[2]
                    # Rotate
                    pos = _rotate_z(pos.reshape(1, 3), angle)[0]
                    # Translate
                    pos[0] += cx
                    pos[1] += cy
                    # Small jitter
                    pos += rng.normal(0, 0.015, 3)

                    is_head = bi < 2
                    color = list(template['color_head'] if is_head else template['color_tail'])

                    all_beads.append({
                        'pos': pos.tolist(),
                        'color': color,
                        'label': f'{lip_name}-{bname}',
                        'atype': btype,
                        'resname': lip_name,
                        'leaflet': leaflet,
                        'is_head': bi < 2,
                    })

                for (a, b) in template['bonds']:
                    all_bonds.append([bead_offset + a, bead_offset + b])

                bead_offset += len(template['beads'])
                lipid_counts[lip_name] = lipid_counts.get(lip_name, 0) + 1

    n_total = nx_lipids * ny_lipids * 2
    box_x = nx_lipids * spacing
    box_y = ny_lipids * spacing

    return {
        'beads': all_beads,
        'bonds': all_bonds,
        'stats': {
            'n_lipids': n_total,
            'n_beads': len(all_beads),
            'n_bonds': len(all_bonds),
            'lipid_counts': lipid_counts,
            'box_nm': [round(box_x, 2), round(box_y, 2)],
            'area_per_lipid_nm2': round(spacing ** 2, 3),
        },
    }


def build_micelle(lipid_name='DPC', n_lipids=60, radius=2.5, seed=456):
    """Build a spherical micelle.

    Parameters
    ----------
    lipid_name : str
        Lipid type (typically a single-tail detergent like DPC).
    n_lipids : int
        Number of lipids in the micelle.
    radius : float
        Approximate micelle radius in nm.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: beads, bonds, stats
    """
    rng = np.random.default_rng(seed)
    template = LIPID_TEMPLATES[lipid_name]

    # Distribute lipids on a sphere using golden spiral
    points = _fibonacci_sphere(n_lipids)

    all_beads = []
    all_bonds = []
    bead_offset = 0

    for i in range(n_lipids):
        direction = np.array(points[i])  # unit vector pointing outward

        # Build rotation matrix to align z-axis with direction
        rot = _rotation_from_z_to(direction)

        for bi, (bname, btype, bpos) in enumerate(template['beads']):
            pos = np.array(bpos, dtype=float)
            # Scale: head at radius, tails toward center
            # Shift so head is at radius
            head_z = template['beads'][0][2][2]
            pos[2] = pos[2] - head_z + radius
            # Rotate to point outward
            pos = rot @ pos
            # Small jitter
            pos += rng.normal(0, 0.02, 3)

            is_head = bi < 2
            color = list(template['color_head'] if is_head else template['color_tail'])

            all_beads.append({
                'pos': pos.tolist(),
                'color': color,
                'label': f'{lipid_name}-{bname}',
                'atype': btype,
                'resname': lipid_name,
                'is_head': is_head,
            })

        for (a, b) in template['bonds']:
            all_bonds.append([bead_offset + a, bead_offset + b])
        bead_offset += len(template['beads'])

    return {
        'beads': all_beads,
        'bonds': all_bonds,
        'stats': {
            'n_lipids': n_lipids,
            'n_beads': len(all_beads),
            'n_bonds': len(all_bonds),
            'lipid_counts': {lipid_name: n_lipids},
            'radius_nm': radius,
        },
    }


def build_protein_helix(n_residues=23, helix_radius=0.23, rise_per_residue=0.15,
                        residues_per_turn=3.6, center_z=0.0):
    """Build a transmembrane alpha-helix (WALP-like) in Martini CG.

    Each residue → BB + SC1 beads (like Ala/Leu in Martini).
    The helix is oriented along the z-axis.

    Returns
    -------
    dict with keys: beads, bonds, stats
    """
    all_beads = []
    all_bonds = []

    for i in range(n_residues):
        angle = 2 * math.pi * i / residues_per_turn
        z = i * rise_per_residue - (n_residues - 1) * rise_per_residue / 2 + center_z

        # Backbone bead
        bb_x = helix_radius * math.cos(angle)
        bb_y = helix_radius * math.sin(angle)

        # Sidechain bead (points outward)
        sc_x = (helix_radius + 0.20) * math.cos(angle)
        sc_y = (helix_radius + 0.20) * math.sin(angle)

        # Determine residue type based on position
        if i < 2 or i >= n_residues - 2:
            resname = 'TRP'  # anchoring tryptophans
            bb_color = [0.70, 0.20, 0.85]
            sc_color = [0.85, 0.40, 0.95]
        else:
            resname = 'LEU'
            bb_color = PROTEIN_COLORS['BB']
            sc_color = PROTEIN_COLORS['SC']

        bb_idx = len(all_beads)
        all_beads.append({
            'pos': [bb_x, bb_y, z],
            'color': bb_color,
            'label': f'{resname}{i+1}-BB',
            'atype': 'P2' if resname == 'TRP' else 'SP2',
            'resname': resname,
            'resid': i + 1,
            'is_protein': True,
            'is_head': False,
        })

        sc_idx = len(all_beads)
        all_beads.append({
            'pos': [sc_x, sc_y, z],
            'color': sc_color,
            'label': f'{resname}{i+1}-SC1',
            'atype': 'SC5' if resname == 'TRP' else 'SC2',
            'resname': resname,
            'resid': i + 1,
            'is_protein': True,
            'is_head': False,
        })

        # BB-SC bond
        all_bonds.append([bb_idx, sc_idx])

        # BB-BB bond to previous residue
        if i > 0:
            all_bonds.append([bb_idx - 2, bb_idx])

    return {
        'beads': all_beads,
        'bonds': all_bonds,
        'stats': {
            'n_residues': n_residues,
            'n_beads': len(all_beads),
            'n_bonds': len(all_bonds),
        },
    }


def build_protein_in_membrane(composition, nx_lipids=12, ny_lipids=12,
                              spacing=0.65, n_helix_residues=23,
                              exclusion_radius=0.8, seed=789):
    """Build a transmembrane protein embedded in a lipid bilayer.

    Places a WALP-like helix at the center and excludes lipids within
    exclusion_radius.

    Returns
    -------
    dict with beads, bonds, stats
    """
    # Build membrane first
    membrane = build_bilayer(composition, nx_lipids, ny_lipids, spacing, seed=seed)

    # Build protein helix
    helix = build_protein_helix(n_helix_residues)

    # Remove lipids that overlap with the protein
    # Check by headgroup position distance from z-axis
    filtered_beads = []
    filtered_bonds = []
    old_to_new = {}
    excluded_lipids = set()

    # First pass: identify excluded lipids
    current_lipid_start = 0
    for i, bead in enumerate(membrane['beads']):
        if bead['is_head']:
            dist_xy = math.sqrt(bead['pos'][0]**2 + bead['pos'][1]**2)
            if dist_xy < exclusion_radius:
                # Find all beads of this lipid (from this head to next head or end)
                # Mark for exclusion
                excluded_lipids.add(i)

    # Actually, let's do this by checking each bead's xy distance
    for i, bead in enumerate(membrane['beads']):
        dist_xy = math.sqrt(bead['pos'][0]**2 + bead['pos'][1]**2)
        if dist_xy < exclusion_radius:
            old_to_new[i] = None
        else:
            old_to_new[i] = len(filtered_beads)
            filtered_beads.append(bead)

    for a, b in membrane['bonds']:
        if old_to_new.get(a) is not None and old_to_new.get(b) is not None:
            filtered_bonds.append([old_to_new[a], old_to_new[b]])

    # Add protein beads and bonds
    protein_offset = len(filtered_beads)
    for bead in helix['beads']:
        filtered_beads.append(bead)
    for a, b in helix['bonds']:
        filtered_bonds.append([protein_offset + a, protein_offset + b])

    # Count remaining lipids
    lip_counts = {}
    for b in filtered_beads:
        rn = b.get('resname', '')
        if rn in LIPID_TEMPLATES:
            if b.get('is_head'):
                lip_counts[rn] = lip_counts.get(rn, 0) + 1

    return {
        'beads': filtered_beads,
        'bonds': filtered_bonds,
        'stats': {
            'n_beads': len(filtered_beads),
            'n_bonds': len(filtered_bonds),
            'n_protein_residues': helix['stats']['n_residues'],
            'n_protein_beads': helix['stats']['n_beads'],
            'lipid_counts': lip_counts,
            'n_lipids': sum(lip_counts.values()),
        },
    }


def build_vesicle(composition, n_lipids_outer=400, n_lipids_inner=250,
                  outer_radius=8.0, inner_radius=6.0, seed=999):
    """Build a spherical vesicle (liposome) with two leaflets.

    Parameters
    ----------
    composition : dict
        Lipid composition fractions.
    n_lipids_outer, n_lipids_inner : int
        Number of lipids in each leaflet.
    outer_radius, inner_radius : float
        Radii of outer and inner leaflets in nm.
    seed : int
        Random seed.

    Returns
    -------
    dict with beads, bonds, stats
    """
    rng = np.random.default_rng(seed)

    # Normalize composition
    total = sum(composition.values())
    norm_comp = {k: v / total for k, v in composition.items()}

    all_beads = []
    all_bonds = []
    bead_offset = 0
    lipid_counts = {}

    for leaflet, n_lip, radius, outward in [
        ('outer', n_lipids_outer, outer_radius, True),
        ('inner', n_lipids_inner, inner_radius, False),
    ]:
        # Assign lipid types
        lip_names = []
        for name, frac in norm_comp.items():
            lip_names.extend([name] * round(frac * n_lip))
        while len(lip_names) < n_lip:
            lip_names.append(list(norm_comp.keys())[0])
        lip_names = lip_names[:n_lip]
        rng.shuffle(lip_names)

        # Distribute on sphere
        points = _fibonacci_sphere(n_lip)

        for i in range(n_lip):
            direction = np.array(points[i])
            if not outward:
                direction = -direction

            rot = _rotation_from_z_to(direction)
            lip_name = lip_names[i]
            template = LIPID_TEMPLATES[lip_name]

            for bi, (bname, btype, bpos) in enumerate(template['beads']):
                pos = np.array(bpos, dtype=float)
                # Place head at radius, tails inward (for outer) or outward (for inner)
                head_z = template['beads'][0][2][2]
                if outward:
                    pos[2] = pos[2] - head_z + radius
                else:
                    pos[2] = -(pos[2] - head_z) + radius

                pos = rot @ pos
                pos += rng.normal(0, 0.02, 3)

                is_head = bi < 2
                color = list(template['color_head'] if is_head else template['color_tail'])

                all_beads.append({
                    'pos': pos.tolist(),
                    'color': color,
                    'label': f'{lip_name}-{bname}',
                    'atype': btype,
                    'resname': lip_name,
                    'leaflet': leaflet,
                    'is_head': is_head,
                })

            for (a, b) in template['bonds']:
                all_bonds.append([bead_offset + a, bead_offset + b])
            bead_offset += len(template['beads'])
            lipid_counts[lip_name] = lipid_counts.get(lip_name, 0) + 1

    return {
        'beads': all_beads,
        'bonds': all_bonds,
        'stats': {
            'n_lipids': n_lipids_outer + n_lipids_inner,
            'n_beads': len(all_beads),
            'n_bonds': len(all_bonds),
            'lipid_counts': lipid_counts,
            'outer_radius_nm': outer_radius,
            'inner_radius_nm': inner_radius,
            'n_outer': n_lipids_outer,
            'n_inner': n_lipids_inner,
        },
    }


def relax_structure(result, n_steps=800, dt=0.005, sigma=0.47,
                    bond_k=3000.0, repulsion_eps=3.0,
                    perturb=0.12, constrain_z_heads=False,
                    seed=42):
    """Perturb and energy-minimize a CG structure.

    First applies a random thermal perturbation to break grid symmetry,
    then runs steepest-descent minimization with WCA repulsion and
    harmonic bond restraints to reach a lower-energy conformation.

    Operates in-place on ``result['beads']`` positions.

    Parameters
    ----------
    result : dict
        Output from a builder function (must have 'beads' and 'bonds').
    n_steps : int
        Number of minimization steps.
    dt : float
        Maximum displacement per step (nm).
    sigma : float
        Effective bead diameter for WCA repulsion (~0.47 nm for Martini).
    bond_k : float
        Harmonic spring constant for bonds (kJ/mol/nm²).
    repulsion_eps : float
        WCA repulsion strength (kJ/mol).
    perturb : float
        Standard deviation of initial random perturbation (nm).
    constrain_z_heads : bool
        If True, headgroup beads keep their z-coordinate during relaxation.
    seed : int
        Random seed for perturbation.
    """
    from scipy.spatial import cKDTree

    beads = result['beads']
    bonds = result['bonds']
    n = len(beads)
    if n == 0:
        return result

    pos = np.array([b['pos'] for b in beads], dtype=np.float64)

    # Masks
    head_mask = np.array([b.get('is_head', False) for b in beads])
    protein_mask = np.array([b.get('is_protein', False) for b in beads])

    # --- Initial perturbation to break grid symmetry ---
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, perturb, pos.shape)
    if constrain_z_heads:
        noise[head_mask, 2] = 0.0  # keep heads flat
    # Protein beads: less perturbation
    noise[protein_mask] *= 0.3
    pos += noise

    # Bond arrays + target lengths (from pre-perturbation geometry isn't ideal,
    # recompute from original template-based positions)
    bond_arr = np.array(bonds, dtype=np.int32) if bonds else np.empty((0, 2), dtype=np.int32)
    if len(bond_arr) > 0:
        # Use original positions for target bond lengths
        orig_pos = np.array([b['pos'] for b in beads], dtype=np.float64)
        # Actually we already perturbed pos, and beads still has originals at this point
        # But we modified pos in place, so recompute from the unperturbed state stored in beads
        # Wait — we haven't written back yet, so beads[i]['pos'] is still original.
        opos = np.array([b['pos'] for b in beads], dtype=np.float64)  # original
        bvecs0 = opos[bond_arr[:, 1]] - opos[bond_arr[:, 0]]
        bond_eq = np.linalg.norm(bvecs0, axis=1)
        bond_eq = np.clip(bond_eq, 0.20, 0.80)
    else:
        bond_eq = np.array([])

    # Bonded pair set for NB exclusion
    bonded_set = set()
    for a, b in bonds:
        bonded_set.add((min(a, b), max(a, b)))

    sigma2 = sigma * sigma
    wca_cut = sigma * (2.0 ** (1.0 / 6.0))  # ~0.527 nm

    actual_steps = 0
    for step in range(n_steps):
        actual_steps = step + 1
        forces = np.zeros_like(pos)

        # --- WCA repulsion via KDTree neighbor search ---
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=wca_cut + 0.02, output_type='ndarray')

        if len(pairs) > 0:
            # Exclude bonded pairs vectorized via set lookup
            keep = np.array([
                (min(int(p[0]), int(p[1])), max(int(p[0]), int(p[1]))) not in bonded_set
                for p in pairs
            ])
            pairs = pairs[keep]

        if len(pairs) > 0:
            dvec = pos[pairs[:, 1]] - pos[pairs[:, 0]]  # j - i
            dist2 = np.sum(dvec * dvec, axis=1)
            dist2 = np.clip(dist2, 0.04 * sigma2, None)  # avoid singularity
            dist = np.sqrt(dist2)

            mask_wca = dist < wca_cut
            if np.any(mask_wca):
                r = dist[mask_wca]
                r2 = dist2[mask_wca]
                inv_r2 = sigma2 / r2
                inv_r6 = inv_r2 ** 3
                inv_r12 = inv_r6 ** 2
                fmag = 24.0 * repulsion_eps * (2.0 * inv_r12 - inv_r6) / r
                fmag = np.clip(fmag, -1e4, 1e4)
                # Force direction: unit vector from i to j
                uv = dvec[mask_wca] / r[:, None]
                f = uv * fmag[:, None]  # repulsive = pushes apart

                p = pairs[mask_wca]
                # Vectorized accumulation
                np.add.at(forces, p[:, 0], -f)
                np.add.at(forces, p[:, 1], f)

        # --- Harmonic bond forces ---
        if len(bond_arr) > 0:
            bvec = pos[bond_arr[:, 1]] - pos[bond_arr[:, 0]]
            bdist = np.linalg.norm(bvec, axis=1)
            bdist = np.clip(bdist, 1e-6, None)
            uv_b = bvec / bdist[:, None]
            stretch = bdist - bond_eq
            fmag_b = bond_k * stretch
            f_b = uv_b * fmag_b[:, None]
            np.add.at(forces, bond_arr[:, 0], f_b)
            np.add.at(forces, bond_arr[:, 1], -f_b)

        # --- Constraints ---
        if constrain_z_heads:
            forces[head_mask, 2] = 0.0
        if np.any(protein_mask):
            forces[protein_mask] *= 0.2

        # --- Steepest descent step (capped displacement) ---
        fnorm = np.linalg.norm(forces, axis=1, keepdims=True)
        max_f = fnorm.max()
        if max_f < 0.5:
            break  # converged

        # Cap per-bead displacement at dt
        scale = np.minimum(dt / (fnorm + 1e-12), dt)
        pos += forces * scale

    # Write back positions
    for i in range(n):
        beads[i]['pos'] = pos[i].tolist()

    result['stats']['relax_steps'] = actual_steps
    return result


def _fibonacci_sphere(n):
    """Distribute n points on a unit sphere using the golden spiral method."""
    points = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        r = math.sqrt(1 - y * y)
        theta = golden * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append([x, y, z])
    return points


def _rotation_from_z_to(target):
    """Build a 3x3 rotation matrix that rotates [0,0,1] to target (unit vector)."""
    target = np.array(target, dtype=float)
    target = target / (np.linalg.norm(target) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])

    # Cross product gives rotation axis
    v = np.cross(z, target)
    s = np.linalg.norm(v)
    c = np.dot(z, target)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
            return np.diag([-1.0, -1.0, 1.0])

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return rot
