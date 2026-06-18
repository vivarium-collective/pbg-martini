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
import subprocess
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


# --------------------------------------------------------------------------
# Task 6: pure-Python .gro/.top writers + stamp-all assembler
# --------------------------------------------------------------------------

def stamp_all(slice_spec, templates):
    """Stamp every placement of every species at its parsimony pose.

    Returns ``(coords_nm, bead_names, counts)`` where ``coords_nm`` is an
    ``(N, 3)`` array, ``bead_names`` is a length-N list of per-bead atom names,
    and ``counts`` maps species name -> instance count. This is the pure-Python
    fallback renderer and the source of the assembly invariant
    ``N == sum(template.n_beads * instance_count)``.
    """
    chunks = []
    bead_names = []
    counts = {}
    for name, place_list in slice_spec.by_species.items():
        tpl = templates.get(name)
        if tpl is None or not place_list:
            continue
        per_bead_names = getattr(tpl, "bead_names", None)
        if not per_bead_names:
            per_bead_names = [_bead_label(name)] * tpl.n_beads
        counts[name] = len(place_list)
        for pl in place_list:
            stamped = stamp(tpl.beads_nm, pl.position, pl.rotation)
            chunks.append(stamped)
            bead_names.extend(per_bead_names)
    if chunks:
        coords = np.concatenate(chunks, axis=0)
    else:
        coords = np.zeros((0, 3))
    return coords, bead_names, counts


def _bead_label(species_name) -> str:
    """A short (<=5 char) atom/residue label for a species' beads."""
    return species_name.replace("-", "")[:5] or "BB"


def write_gro(path, bead_names, coords_nm, box_nm):
    """Write a GROMACS ``.gro`` file (nm units).

    Line 1 is a title, line 2 the atom count, then one fixed-width record per
    bead, and a final box-vector line.
    """
    coords = np.asarray(coords_nm, dtype=float)
    n = coords.shape[0]
    lines = ["parsimony->Martini assembled slice", f"{n:5d}"]
    for i in range(n):
        atom_name = bead_names[i] if i < len(bead_names) else "BB"
        res_name = atom_name[:5]
        resid = (i % 99999) + 1
        atomid = (i % 99999) + 1
        x, y, z = coords[i]
        lines.append(
            f"{resid:5d}{res_name:<5s}{atom_name:>5s}{atomid:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    bx, by, bz = box_nm
    lines.append(f"{bx:10.5f}{by:10.5f}{bz:10.5f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


def write_top(path, itp_includes, molecule_counts, system_name="parsimony cell slice"):
    """Write a GROMACS ``.top`` file: ``#include`` lines + ``[ molecules ]``."""
    lines = []
    for inc in itp_includes:
        lines.append(f'#include "{inc}"')
    lines.append("")
    lines.append("[ system ]")
    lines.append(system_name)
    lines.append("")
    lines.append("[ molecules ]")
    for name, count in molecule_counts.items():
        lines.append(f"{name} {count}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


# --------------------------------------------------------------------------
# Task 7: assemble() orchestration (slice -> render -> files)
# --------------------------------------------------------------------------

def render_bentopy(placements_json_path, top_path, gro_path):
    """Dispatch to the real ``bentopy render`` binary to assemble gro + top.

    Mirrors the marrink-lab workflow: ``bentopy render -t topol.top
    placements.json out.gro``. Raises if the binary is unavailable or fails;
    callers fall back to the pure-Python writers (``stamp_all`` + ``write_gro``).
    """
    binary = os.environ.get("BENTOPY_BIN") or shutil.which("bentopy")
    if not binary:
        raise RuntimeError("bentopy binary not available")
    subprocess.run(
        [binary, "render", "-t", top_path, placements_json_path, gro_path],
        check=True,
    )
    return gro_path


def assemble(pack_path, box_min, box_max, species, templates, out_dir,
             use_bentopy=True, ff_includes=None):
    """Assemble a Martini CG system from a parsimony pack sub-box.

    Slices the pack, converts to a bentopy placement list (always written to
    ``placements.json`` for reproducibility / the bentopy path), then renders
    via real ``bentopy render`` when available + requested, else the pure-Python
    stamper. Returns a summary dict with file paths and assembly invariants.
    """
    os.makedirs(out_dir, exist_ok=True)
    pack = load_pack(pack_path)
    sl = select_slice(pack, box_min, box_max, species)
    box_nm = tuple((box_max[a] - box_min[a]) / 10.0 for a in range(3))

    placements_json = os.path.join(out_dir, "placements.json")
    bentopy_doc = to_bentopy_placements(sl, templates, box_nm)
    with open(placements_json, "w") as fh:
        json.dump(bentopy_doc, fh, indent=2)

    counts = {n: len(v) for n, v in sl.by_species.items() if v}
    n_beads_expected = sum(counts[n] * templates[n].n_beads for n in counts)

    gro = os.path.join(out_dir, "system.gro")
    top = os.path.join(out_dir, "system.top")

    itp_includes = list(ff_includes or [])
    itp_includes += [templates[n].itp_path for n in counts]

    rendered_by = "stamper"
    if use_bentopy and bentopy_available():
        write_top(top, itp_includes, counts)
        try:
            render_bentopy(placements_json, top, gro)
            rendered_by = "bentopy"
            n_beads = n_beads_expected
        except Exception:
            rendered_by = "stamper"

    if rendered_by == "stamper":
        coords, names, _ = stamp_all(sl, templates)
        write_gro(gro, names, coords, box_nm)
        write_top(top, itp_includes, counts)
        n_beads = int(coords.shape[0])

    return {
        "gro": gro,
        "top": top,
        "placements_json": placements_json,
        "n_beads": n_beads,
        "n_molecules": int(sum(counts.values())),
        "per_species_counts": counts,
        "box_nm": box_nm,
        "rendered_by": rendered_by,
    }
