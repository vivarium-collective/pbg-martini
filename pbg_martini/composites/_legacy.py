"""Pre-built composite document factories for Martini coarse-graining."""


def make_martinize_document(
    pdb_text,
    from_ff='charmm',
    to_ff='martini3001',
    delete_unknown=True,
    ignh=False,
):
    """Build a PBG composite document that coarse-grains a PDB structure.

    Parameters
    ----------
    pdb_text : str
        PDB-format text of the atomistic structure.
    from_ff : str
        Source atomistic force field name.
    to_ff : str
        Target Martini force field name.
    delete_unknown : bool
        Silently delete residues without a known mapping.
    ignh : bool
        Ignore hydrogen atoms in the input.

    Returns
    -------
    dict
        A PBG composite document ready for ``Composite({'state': doc})``.
    """
    return {
        'martinize': {
            '_type': 'step',
            'address': 'local:MartinizeStep',
            'config': {
                'from_ff': from_ff,
                'to_ff': to_ff,
                'delete_unknown': delete_unknown,
                'ignh': ignh,
            },
            'inputs': {
                'pdb_text': ['stores', 'pdb_text'],
            },
            'outputs': {
                'cg_beads': ['stores', 'cg_beads'],
                'cg_positions': ['stores', 'cg_positions'],
                'cg_bonds': ['stores', 'cg_bonds'],
                'interactions': ['stores', 'interactions'],
                'bead_type_counts': ['stores', 'bead_type_counts'],
                'residue_counts': ['stores', 'residue_counts'],
                'n_atoms_input': ['stores', 'n_atoms_input'],
                'n_atoms_full': ['stores', 'n_atoms_full'],
                'n_beads': ['stores', 'n_beads'],
                'n_bonds_cg': ['stores', 'n_bonds_cg'],
                'reduction_ratio': ['stores', 'reduction_ratio'],
            },
        },
        'stores': {
            'pdb_text': pdb_text,
        },
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'n_beads': 'integer',
                    'n_bonds_cg': 'integer',
                    'reduction_ratio': 'float',
                },
            },
            'inputs': {
                'n_beads': ['stores', 'n_beads'],
                'n_bonds_cg': ['stores', 'n_bonds_cg'],
                'reduction_ratio': ['stores', 'reduction_ratio'],
            },
        },
    }
