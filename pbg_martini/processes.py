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
    elastic=False,
    return_itp=False,
    moltype_name='molecule',
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
    elastic : bool
        Whether to apply a Martini elastic network (rubber bands across the
        backbone) to keep the protein's globular fold. Best-effort: failures
        degrade to no elastic network.
    return_itp : bool
        When True, also return ``itp_text`` (a GROMACS ``.itp`` for the CG
        molecule(s)) and ``moltype_names`` in the result dict.
    moltype_name : str
        Base name for the emitted ``[ moleculetype ]`` (suffixed per chain when
        the structure yields more than one molecule).

    Returns
    -------
    dict
        Keys match :meth:`MartinizeStep.outputs`, plus ``itp_text`` /
        ``moltype_names`` when ``return_itp`` is set.
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

        if elastic:
            # Martini elastic network with martinize2's default parameters
            # (-el 0.5 -eu 0.9 -ef 500 -ea 0 -ep 1 -em 0). Best-effort.
            try:
                from vermouth.processors import ApplyRubberBand
                ApplyRubberBand(
                    lower_bound=0.5,
                    upper_bound=0.9,
                    decay_factor=0.0,
                    decay_power=1.0,
                    base_constant=500.0,
                    minimum_force=0.0,
                ).run_system(system)
            except Exception:
                pass

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

        itp_text = None
        moltype_names = []
        if return_itp:
            import io
            from vermouth.gmx.itp import write_molecule_itp

            mols = list(system.molecules)
            buf = io.StringIO()
            for i, mol in enumerate(mols):
                name = moltype_name if len(mols) == 1 else f"{moltype_name}_{i}"
                moltype_names.append(name)
                if not hasattr(mol, 'nrexcl') or mol.nrexcl is None:
                    mol.nrexcl = 1
                write_molecule_itp(mol, outfile=buf, moltype=name)
                buf.write("\n")
            itp_text = buf.getvalue()

        result = {
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
        if return_itp:
            result['itp_text'] = itp_text
            result['moltype_names'] = moltype_names
        return result
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


# --------------------------------------------------------------------------
# parsimony -> Martini whole-cell PBG Steps (Task 10)
# --------------------------------------------------------------------------

# Default PoC allow-list of cleanly-martinizable species.
PARSIMONY_ALLOW_LIST = [
    'EG10367-MONOMER', 'EG11036-MONOMER', 'groel',
    'EG11384-MONOMER', 'EG50003-MONOMER', 'EG10669-MONOMER',
]


def _stub_templates(species, n_beads):
    """Build placeholder origin-centered CG templates (offline composite path)."""
    import numpy as np
    from pbg_martini.parsimony_assembler import CGTemplate
    return {
        name: CGTemplate(name, f'{name}.gro', f'{name}.itp',
                         np.zeros((n_beads, 3)), n_beads)
        for name in species
    }


class ParsimonySliceStep(Step):
    """Select a spatial sub-box + species allow-list from a parsimony pack."""

    config_schema = {
        'pack_path': {'_type': 'string', '_default': ''},
        'box_min': {'_type': 'list', '_default': [-750.0, -750.0, -750.0]},
        'box_max': {'_type': 'list', '_default': [750.0, 750.0, 750.0]},
        'species': {'_type': 'list', '_default': PARSIMONY_ALLOW_LIST},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'per_species_counts': 'overwrite[map]',
            'n_molecules': 'overwrite[integer]',
        }

    def update(self, state):
        from pbg_martini.parsimony_assembler import load_pack, select_slice
        pack = load_pack(self.config['pack_path'])
        sl = select_slice(
            pack,
            tuple(self.config['box_min']),
            tuple(self.config['box_max']),
            self.config['species'],
        )
        counts = {n: len(v) for n, v in sl.by_species.items() if v}
        return {
            'per_species_counts': counts,
            'n_molecules': int(sum(counts.values())),
        }


class MartinizeSpeciesStep(Step):
    """Martinize one atomistic structure into a CG template (itp + beads)."""

    config_schema = {
        'species': {'_type': 'string', '_default': ''},
        'pdb_path': {'_type': 'string', '_default': ''},
        'out_dir': {'_type': 'string', '_default': '.cache/templates'},
        'elastic': {'_type': 'boolean', '_default': True},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'itp_path': 'overwrite[string]',
            'structure_path': 'overwrite[string]',
            'n_beads': 'overwrite[integer]',
        }

    def update(self, state):
        from pbg_martini.parsimony_assembler import martinize_species
        tpl = martinize_species(
            self.config['species'],
            self.config['pdb_path'],
            self.config['out_dir'],
            elastic=self.config['elastic'],
        )
        return {
            'itp_path': tpl.itp_path,
            'structure_path': tpl.structure_path,
            'n_beads': int(tpl.n_beads),
        }


class ParsimonyAssembleStep(Step):
    """Assemble a Martini CG system at the parsimony placements.

    Uses stub templates by default (so the composite runs fully offline);
    point ``stub_beads`` at a realistic per-species bead count or supply real
    martinized templates via the library ``assemble`` API for a runnable system.
    """

    config_schema = {
        'pack_path': {'_type': 'string', '_default': ''},
        'box_min': {'_type': 'list', '_default': [-750.0, -750.0, -750.0]},
        'box_max': {'_type': 'list', '_default': [750.0, 750.0, 750.0]},
        'species': {'_type': 'list', '_default': PARSIMONY_ALLOW_LIST},
        'out_dir': {'_type': 'string', '_default': 'output/parsimony_slice'},
        'stub_beads': {'_type': 'integer', '_default': 5},
        'use_bentopy': {'_type': 'boolean', '_default': True},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'gro': 'overwrite[string]',
            'top': 'overwrite[string]',
            'placements_json': 'overwrite[string]',
            'n_beads': 'overwrite[integer]',
            'n_molecules': 'overwrite[integer]',
            'per_species_counts': 'overwrite[map]',
        }

    def update(self, state):
        from pbg_martini.parsimony_assembler import assemble
        templates = _stub_templates(self.config['species'], self.config['stub_beads'])
        out = assemble(
            self.config['pack_path'],
            tuple(self.config['box_min']),
            tuple(self.config['box_max']),
            self.config['species'],
            templates,
            self.config['out_dir'],
            use_bentopy=self.config['use_bentopy'],
        )
        return {
            'gro': out['gro'],
            'top': out['top'],
            'placements_json': out['placements_json'],
            'n_beads': out['n_beads'],
            'n_molecules': out['n_molecules'],
            'per_species_counts': out['per_species_counts'],
        }


class MartiniMDStep(Step):
    """WCA-relax the assembly, then a best-effort OpenMM short NVT run.

    Relax always runs (pure Python). The OpenMM stage is best-effort: if
    OpenMM is unavailable or the topology lacks force-field parameters, it is
    skipped and ``md_ran`` is False.
    """

    config_schema = {
        'relax_steps': {'_type': 'integer', '_default': 200},
        'md_steps': {'_type': 'integer', '_default': 0},
        'run_md': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return {
            'gro': 'string',
            'top': 'string',
        }

    def outputs(self):
        return {
            'minimized_gro': 'overwrite[string]',
            'final_energy': 'overwrite[float]',
            'md_ran': 'overwrite[boolean]',
        }

    def update(self, state):
        import numpy as np
        from pbg_martini.parsimony_md import (
            relax_assembly, run_short_md, openmm_available,
        )

        gro_path = state['gro']
        top_path = state['top']

        # WCA relax the stamped coordinates read back from the .gro.
        coords, box_nm = _read_gro_coords(gro_path)
        if coords.shape[0]:
            relaxed = relax_assembly(coords, n_steps=self.config['relax_steps'])
        else:
            relaxed = coords

        minimized_gro = gro_path
        final_energy = 0.0
        md_ran = False
        if self.config['run_md'] and openmm_available():
            try:
                md = run_short_md(gro_path, top_path,
                                  steps=self.config['md_steps'])
                minimized_gro = md['minimized_gro']
                final_energy = md['final_energy']
                md_ran = True
            except Exception:
                md_ran = False

        return {
            'minimized_gro': minimized_gro,
            'final_energy': float(final_energy),
            'md_ran': md_ran,
        }


def _read_gro_coords(gro_path):
    """Read coordinates (nm) + box from a GROMACS ``.gro`` file."""
    import numpy as np
    with open(gro_path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[1].strip())
    coords = []
    for i in range(2, 2 + n):
        ln = lines[i]
        x = float(ln[20:28]); y = float(ln[28:36]); z = float(ln[36:44])
        coords.append((x, y, z))
    box = tuple(float(v) for v in lines[2 + n].split()[:3])
    return np.array(coords, dtype=float), box
