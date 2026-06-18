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
from dataclasses import dataclass, field


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
