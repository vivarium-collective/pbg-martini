"""Bridge parsimony's packed 3D structure into a Martini CG MD system.

This module assembles coarse-grained molecules at the positions/orientations
recorded by parsimony (a Rust cellPACK-style packer), replacing Bentopy's
random packing. It holds the data model (Task 1), the quaternion stamping math
(Task 4), the bentopy placement-list converter (Task 5), the pure-Python
``.gro``/``.top`` writers + stamp-all assembler (Task 6), and the ``assemble``
orchestration (Task 7).

Units: parsimony positions are Angstrom; Martini/GROMACS ``.gro`` is nm, so
positions are divided by 10. Rotations are quaternions ``(w, x, y, z)`` (see
``quat_to_matrix``).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Task 1: data model + pack loader + slice selection
# --------------------------------------------------------------------------

@dataclass
class Ingredient:
    """A packed species: its integer id, name, and display color."""

    id: int
    name: str
    color: tuple = (0.5, 0.5, 0.5)


@dataclass
class Placement:
    """A single placed instance of an ingredient.

    ``position`` is in Angstrom; ``rotation`` is a quaternion ``(w, x, y, z)``.
    """

    ingredient_id: int
    position: tuple
    rotation: tuple
    uid: int


@dataclass
class Pack:
    """A loaded ``parsimony.pack.v1`` document."""

    ingredients: dict  # id -> Ingredient
    placements: list   # list[Placement]
    bounds: dict = field(default_factory=dict)


@dataclass
class SliceSpec:
    """A spatial + species selection of placements, grouped by species name."""

    by_species: dict  # name -> list[Placement]


def load_pack(path) -> Pack:
    """Load a ``parsimony.pack.v1`` JSON file into a :class:`Pack`."""
    with open(path) as fh:
        doc = json.load(fh)

    ingredients = {}
    for ing in doc.get("ingredients", []):
        iid = int(ing["id"])
        ingredients[iid] = Ingredient(
            id=iid,
            name=ing["name"],
            color=tuple(ing.get("color", (0.5, 0.5, 0.5))),
        )

    placements = []
    for pl in doc.get("placements", []):
        placements.append(Placement(
            ingredient_id=int(pl["ingredient"]),
            position=tuple(float(x) for x in pl["position"]),
            rotation=tuple(float(x) for x in pl["rotation"]),
            uid=int(pl.get("uid", -1)),
        ))

    return Pack(
        ingredients=ingredients,
        placements=placements,
        bounds=doc.get("bounds", {}),
    )


def _in_box(position, box_min, box_max) -> bool:
    return all(
        box_min[a] <= position[a] <= box_max[a]
        for a in range(3)
    )


def select_slice(pack, box_min, box_max, species_names) -> SliceSpec:
    """Select placements whose species is allow-listed and which lie in-box.

    ``box_min``/``box_max`` are per-axis corners in Angstrom; ``species_names``
    is the allow-list of ingredient names to keep.
    """
    wanted = set(species_names)
    by_species = {name: [] for name in wanted}
    for pl in pack.placements:
        ing = pack.ingredients.get(pl.ingredient_id)
        if ing is None or ing.name not in wanted:
            continue
        if _in_box(pl.position, box_min, box_max):
            by_species[ing.name].append(pl)
    return SliceSpec(by_species=by_species)


# --------------------------------------------------------------------------
# Task 4: quaternion rotation + bead stamping (pure math)
# --------------------------------------------------------------------------

# Confirmed convention: parsimony stores rotations as (w, x, y, z) quaternions.
# Verified empirically against the synthetic fixture (90 deg about z maps
# +x -> +y) and re-confirmed at real positions in Task 7. If real packed
# instances ever appear mirrored, pass quat_order="xyzw" to reinterpret the
# stored quaternion.
def quat_to_matrix(q, quat_order="wxyz") -> np.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix.

    ``q`` is normalized first. ``quat_order`` selects the component order of
    the stored quaternion: ``"wxyz"`` (parsimony default) or ``"xyzw"``.
    """
    q = np.asarray(q, dtype=float)
    if quat_order == "wxyz":
        w, x, y, z = q
    elif quat_order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(f"unknown quat_order {quat_order!r}")

    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def stamp(template_beads_nm, position_A, rotation, quat_order="wxyz") -> np.ndarray:
    """Rotate origin-centered template beads, then translate to ``position``.

    ``template_beads_nm`` is an ``(N, 3)`` array of origin-centered bead
    coordinates in nm. ``position_A`` is the placement position in Angstrom
    (converted to nm by /10). Returns the stamped ``(N, 3)`` coordinates in nm.
    """
    beads = np.asarray(template_beads_nm, dtype=float)
    R = quat_to_matrix(rotation, quat_order=quat_order)
    translation_nm = np.asarray(position_A, dtype=float) / 10.0
    return beads @ R.T + translation_nm


# --------------------------------------------------------------------------
# Task 5: bentopy placement-list converter + render dispatch
# --------------------------------------------------------------------------

@dataclass
class CGTemplate:
    """A martinized species template, reused for every placed instance.

    ``beads_nm`` is an ``(n_beads, 3)`` array of origin-centered bead
    coordinates in nm (reference orientation). ``structure_path`` is the CG
    ``.gro``/``.pdb`` file; ``itp_path`` is the per-molecule ``.itp``.
    """

    name: str
    structure_path: str
    itp_path: str
    beads_nm: np.ndarray
    n_beads: int


def bentopy_available() -> bool:
    """Return True if a ``bentopy`` binary is callable (PATH or ``BENTOPY_BIN``)."""
    candidate = os.environ.get("BENTOPY_BIN")
    if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
        return True
    return shutil.which("bentopy") is not None


def to_bentopy_placements(slice_spec, templates, box_nm) -> dict:
    """Convert a :class:`SliceSpec` to a bentopy placement-list JSON dict.

    Schema (one segment per species, each carrying its instances)::

        {
          "size": [x, y, z],            # box edge lengths in nm
          "placements": [
            {"name": species,
             "path": template structure file,
             "itp": template .itp,
             "instances": [{"position": [x,y,z] nm, "rotation": [w,x,y,z]}, ...]}
          ]
        }

    Positions are converted Angstrom -> nm (/10); rotations pass through as the
    parsimony quaternion. The exact field names mirror bentopy's documented
    placement list as closely as can be confirmed offline; if the installed
    binary expects different keys, adjust here and the matching test.
    """
    placements = []
    for name, place_list in slice_spec.by_species.items():
        tpl = templates.get(name)
        if tpl is None:
            continue
        instances = []
        for pl in place_list:
            instances.append({
                "position": [p / 10.0 for p in pl.position],
                "rotation": list(pl.rotation),
            })
        placements.append({
            "name": name,
            "path": tpl.structure_path,
            "itp": tpl.itp_path,
            "instances": instances,
        })
    return {"size": list(box_nm), "placements": placements}
