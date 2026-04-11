"""Process-bigraph Step wrappers for Martini coarse-graining via vermouth/martinize2."""

import os
import tempfile
import numpy as np
from process_bigraph import Step

from pbg_martini.builders import (
    build_bilayer,
    build_micelle,
    build_protein_in_membrane,
    build_vesicle,
)


class MartinizeStep(Step):
    """Coarse-grain an atomistic protein structure using the Martini force field.

    This Step wraps the vermouth/martinize2 pipeline: it reads an atomistic
    structure (provided as PDB-format text or a file path), runs bond inference,
    graph repair, residue mapping, bead averaging, and link generation, then
    returns the coarse-grained bead coordinates, topology, and summary metrics.

    The Step is stateless—each call to ``update`` runs the full pipeline
    from scratch on the supplied input.
    """

    config_schema = {
        'from_ff': {'_type': 'string', '_default': 'charmm'},
        'to_ff': {'_type': 'string', '_default': 'martini3001'},
        'delete_unknown': {'_type': 'boolean', '_default': True},
        'ignh': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return {
            'pdb_text': 'string',
        }

    def outputs(self):
        return {
            'cg_beads': 'overwrite[list]',
            'cg_positions': 'overwrite[list]',
            'cg_bonds': 'overwrite[list]',
            'interactions': 'overwrite[map]',
            'bead_type_counts': 'overwrite[map]',
            'residue_counts': 'overwrite[map]',
            'n_atoms_input': 'overwrite[integer]',
            'n_atoms_full': 'overwrite[integer]',
            'n_beads': 'overwrite[integer]',
            'n_bonds_cg': 'overwrite[integer]',
            'reduction_ratio': 'overwrite[float]',
        }

    def update(self, state):
        pdb_text = state['pdb_text']
        return run_martinize_pipeline(
            pdb_text=pdb_text,
            from_ff_name=self.config['from_ff'],
            to_ff_name=self.config['to_ff'],
            delete_unknown=self.config['delete_unknown'],
            ignh=self.config['ignh'],
        )


def run_martinize_pipeline(
    pdb_text,
    from_ff_name='charmm',
    to_ff_name='martini3001',
    delete_unknown=True,
    ignh=False,
):
    """Run the martinize2 pipeline programmatically and return results as a dict.

    Parameters
    ----------
    pdb_text : str
        PDB-format text of the atomistic structure.
    from_ff_name : str
        Source force field name (default ``'charmm'``).
    to_ff_name : str
        Target CG force field name (default ``'martini3001'``).
    delete_unknown : bool
        Whether to silently delete residues without a known mapping.
    ignh : bool
        Whether to ignore hydrogen atoms in the input.

    Returns
    -------
    dict
        Keys match :meth:`MartinizeStep.outputs`.
    """
    import vermouth
    import vermouth.forcefield
    from vermouth import DATA_PATH
    from vermouth.map_input import (
        read_mapping_directory,
        generate_all_self_mappings,
        combine_mappings,
    )
    from pathlib import Path

    # Write PDB text to a temporary file
    tmp = tempfile.NamedTemporaryFile(
        suffix='.pdb', mode='w', delete=False,
    )
    try:
        tmp.write(pdb_text)
        tmp.close()

        # Load all force fields and mappings
        known_ff = vermouth.forcefield.find_force_fields(
            Path(DATA_PATH) / 'force_fields',
        )
        known_mappings = read_mapping_directory(
            Path(DATA_PATH) / 'mappings', known_ff,
        )
        combine_mappings(
            known_mappings,
            generate_all_self_mappings(known_ff.values()),
        )

        from_ff = known_ff[from_ff_name]
        to_ff = known_ff[to_ff_name]

        # Read PDB into a System
        system = vermouth.System()
        vermouth.PDBInput(
            tmp.name, exclude=(), ignh=ignh, modelidx=1,
        ).run_system(system)

        n_atoms_input = sum(len(m.nodes) for m in system.molecules)

        # Atomistic preparation
        system.force_field = from_ff
        for mol in system.molecules:
            mol._force_field = from_ff

        vermouth.MakeBonds().run_system(system)
        vermouth.RepairGraph(
            delete_unknown=delete_unknown, include_graph=False,
        ).run_system(system)
        vermouth.CanonicalizeModifications().run_system(system)
        vermouth.AttachMass(attribute='mass').run_system(system)
        vermouth.SortMoleculeAtoms().run_system(system)

        n_atoms_full = sum(len(m.nodes) for m in system.molecules)

        # CG mapping
        vermouth.DoMapping(
            mappings=known_mappings,
            to_ff=to_ff,
            delete_unknown=delete_unknown,
            attribute_keep=(
                'cgsecstruct', 'chain', 'aasecstruct', 'resname', 'stash',
            ),
            attribute_must=('resname', 'atype'),
        ).run_system(system)
        vermouth.DoAverageBead(
            ignore_missing_graphs=True,
        ).run_system(system)
        vermouth.DoLinks().run_system(system)
        vermouth.LocateChargeDummies().run_system(system)

        # Extract results
        cg_beads = []
        cg_positions = []
        cg_bonds = []
        bead_type_counts = {}
        residue_counts = {}
        interactions_summary = {}

        for mol in system.molecules:
            node_list = sorted(mol.nodes)
            node_to_idx = {n: i for i, n in enumerate(node_list)}

            for nid in node_list:
                attrs = mol.nodes[nid]
                pos = attrs.get('position')
                cg_beads.append({
                    'index': node_to_idx[nid],
                    'atomname': attrs.get('atomname', ''),
                    'atype': attrs.get('atype', ''),
                    'resname': attrs.get('resname', ''),
                    'resid': attrs.get('resid', 0),
                    'chain': attrs.get('chain', ''),
                })
                cg_positions.append(
                    pos.tolist() if pos is not None else [0.0, 0.0, 0.0],
                )

                bt = attrs.get('atype', 'unknown')
                bead_type_counts[bt] = bead_type_counts.get(bt, 0) + 1
                rn = attrs.get('resname', 'unknown')
                residue_counts[rn] = residue_counts.get(rn, 0) + 1

            for u, v in mol.edges:
                cg_bonds.append([node_to_idx[u], node_to_idx[v]])

            for itype, inters in mol.interactions.items():
                if inters:
                    interactions_summary[itype] = len(inters)

        n_beads = len(cg_beads)
        n_bonds_cg = len(cg_bonds)
        reduction = float(n_atoms_full) / n_beads if n_beads > 0 else 0.0

        return {
            'cg_beads': cg_beads,
            'cg_positions': cg_positions,
            'cg_bonds': cg_bonds,
            'interactions': interactions_summary,
            'bead_type_counts': bead_type_counts,
            'residue_counts': residue_counts,
            'n_atoms_input': n_atoms_input,
            'n_atoms_full': n_atoms_full,
            'n_beads': n_beads,
            'n_bonds_cg': n_bonds_cg,
            'reduction_ratio': round(reduction, 2),
        }
    finally:
        os.unlink(tmp.name)


class MembraneBuilderStep(Step):
    """Build a lipid bilayer membrane patch from composition and grid parameters.

    Procedurally places lipids in a flat bilayer with both leaflets,
    random rotations, and small positional jitter.
    """

    config_schema = {
        'composition': {
            '_type': 'map',
            '_default': {'POPC': 0.4, 'POPE': 0.25, 'CHOL': 0.2, 'SM': 0.15},
        },
        'nx': {'_type': 'integer', '_default': 12},
        'ny': {'_type': 'integer', '_default': 12},
        'spacing': {'_type': 'float', '_default': 0.65},
        'seed': {'_type': 'integer', '_default': 123},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'beads': 'overwrite[list]',
            'bonds': 'overwrite[list]',
            'stats': 'overwrite[map]',
        }

    def update(self, state):
        return build_bilayer(
            composition=self.config['composition'],
            nx_lipids=self.config['nx'],
            ny_lipids=self.config['ny'],
            spacing=self.config['spacing'],
            seed=self.config['seed'],
        )


class MicelleBuilderStep(Step):
    """Build a spherical micelle from a single-tail lipid/detergent."""

    config_schema = {
        'lipid': {'_type': 'string', '_default': 'DPC'},
        'n_lipids': {'_type': 'integer', '_default': 60},
        'radius': {'_type': 'float', '_default': 2.5},
        'seed': {'_type': 'integer', '_default': 456},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'beads': 'overwrite[list]',
            'bonds': 'overwrite[list]',
            'stats': 'overwrite[map]',
        }

    def update(self, state):
        return build_micelle(
            lipid_name=self.config['lipid'],
            n_lipids=self.config['n_lipids'],
            radius=self.config['radius'],
            seed=self.config['seed'],
        )


class ProteinMembraneStep(Step):
    """Build a transmembrane helix embedded in a lipid bilayer."""

    config_schema = {
        'composition': {
            '_type': 'map',
            '_default': {'POPC': 0.7, 'CHOL': 0.3},
        },
        'nx': {'_type': 'integer', '_default': 14},
        'ny': {'_type': 'integer', '_default': 14},
        'spacing': {'_type': 'float', '_default': 0.65},
        'n_helix_residues': {'_type': 'integer', '_default': 23},
        'exclusion_radius': {'_type': 'float', '_default': 0.8},
        'seed': {'_type': 'integer', '_default': 789},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'beads': 'overwrite[list]',
            'bonds': 'overwrite[list]',
            'stats': 'overwrite[map]',
        }

    def update(self, state):
        return build_protein_in_membrane(
            composition=self.config['composition'],
            nx_lipids=self.config['nx'],
            ny_lipids=self.config['ny'],
            spacing=self.config['spacing'],
            n_helix_residues=self.config['n_helix_residues'],
            exclusion_radius=self.config['exclusion_radius'],
            seed=self.config['seed'],
        )


class VesicleBuilderStep(Step):
    """Build a spherical vesicle (liposome) with inner and outer leaflets."""

    config_schema = {
        'composition': {
            '_type': 'map',
            '_default': {'POPC': 0.4, 'POPE': 0.25, 'CHOL': 0.2, 'DPPC': 0.15},
        },
        'n_outer': {'_type': 'integer', '_default': 350},
        'n_inner': {'_type': 'integer', '_default': 220},
        'outer_radius': {'_type': 'float', '_default': 7.0},
        'inner_radius': {'_type': 'float', '_default': 5.2},
        'seed': {'_type': 'integer', '_default': 999},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'beads': 'overwrite[list]',
            'bonds': 'overwrite[list]',
            'stats': 'overwrite[map]',
        }

    def update(self, state):
        return build_vesicle(
            composition=self.config['composition'],
            n_lipids_outer=self.config['n_outer'],
            n_lipids_inner=self.config['n_inner'],
            outer_radius=self.config['outer_radius'],
            inner_radius=self.config['inner_radius'],
            seed=self.config['seed'],
        )
