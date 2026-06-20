"""Martini 3 lipid membrane envelope on the cell's spherocylinder surface.

The parsimony E. coli is a capsule (cylinder + two hemispherical caps, axis along
x, radius ~500 nm, length ~2 um). This builds a Martini 3 bilayer over that
surface: a POPE lipid template (from build_bilayer) is stamped on the inner and
outer leaflets, oriented along the local surface normal, at the Martini
area-per-lipid. ``density_scale`` < 1 thins the membrane for tractable demos /
viewers; the full envelope is enormous (see :func:`envelope_bead_count`).
"""

from __future__ import annotations

import numpy as np

from .parsimony_assembler import CGTemplate, write_gro
from . import build_bilayer

APL_NM2 = 0.423          # Martini area per lipid (POPE)
LEAFLET_OFFSET_NM = 2.0  # half bilayer thickness

# Martini 3 POPE bead order (matches build_bilayer geometry AND the real
# martini_v3.0.0_POPE.itp): head -> phosphate -> glycerols -> two tails.
POPE_BEADS = ["NH3", "PO4", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A",
              "C1B", "C2B", "C3B", "C4B"]


def pope_template() -> CGTemplate:
    """One POPE lipid, 12 beads, head->tail along +z, origin-centered."""
    m = build_bilayer(composition={"POPE": 1.0}, nx_lipids=2, ny_lipids=2)
    n_per = m["stats"]["n_beads"] // m["stats"]["n_lipids"]
    lip = np.array([b["pos"] for b in m["beads"][:n_per]], dtype=float)
    lip = lip - lip.mean(0)
    return CGTemplate("POPE", "", "", lip, n_per)


def _orient(template_nm, axis):
    """Rotate a +z-aligned template so its long axis points along `axis`."""
    z = np.array([0.0, 0.0, 1.0])
    a = axis / (np.linalg.norm(axis) + 1e-12)
    v = np.cross(z, a)
    s = np.linalg.norm(v)
    c = float(np.dot(z, a))
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return template_nm @ R.T


def _surface_points(R_nm, L_nm, apl_nm2, density_scale):
    """Sample (point, outward_normal) over a spherocylinder surface (nm)."""
    spacing = np.sqrt(apl_nm2 / max(density_scale, 1e-6))
    pts, normals = [], []
    # cylinder body: x in [-L/2, L/2]
    n_ax = max(int(L_nm / spacing), 1)
    n_ph = max(int(2 * np.pi * R_nm / spacing), 1)
    for ix in range(n_ax):
        x = -L_nm / 2 + (ix + 0.5) * L_nm / n_ax
        for ip in range(n_ph):
            phi = 2 * np.pi * (ip + 0.5) / n_ph
            pts.append(np.array([x, R_nm * np.cos(phi), R_nm * np.sin(phi)]))
            normals.append(np.array([0.0, np.cos(phi), np.sin(phi)]))
    # two hemispherical caps at x = +/- L/2
    n_th = max(int((np.pi / 2) * R_nm / spacing), 1)
    for sign in (+1.0, -1.0):
        cx = sign * L_nm / 2
        for it in range(n_th):
            theta = (it + 0.5) * (np.pi / 2) / n_th         # 0=equator..pi/2=pole
            ring_r = R_nm * np.cos(theta)
            xoff = sign * R_nm * np.sin(theta)
            n_ph2 = max(int(2 * np.pi * ring_r / spacing), 1)
            for ip in range(n_ph2):
                phi = 2 * np.pi * (ip + 0.5) / n_ph2
                nrm = np.array([sign * np.sin(theta), np.cos(theta) * np.cos(phi),
                                np.cos(theta) * np.sin(phi)])
                pts.append(np.array([cx + xoff, ring_r * np.cos(phi), ring_r * np.sin(phi)]))
                normals.append(nrm / np.linalg.norm(nrm))
    return np.array(pts), np.array(normals)


def envelope_bead_count(R_nm=500.0, L_nm=1000.0, apl_nm2=APL_NM2):
    """(n_lipids, n_beads) for the FULL Martini 3 envelope of this cell."""
    area = 2 * np.pi * R_nm * L_nm + 4 * np.pi * R_nm ** 2   # cylinder + sphere
    n_lipids = int(area / apl_nm2) * 2                       # two leaflets
    return n_lipids, n_lipids * 12


def build_envelope(R_nm=500.0, L_nm=1000.0, density_scale=1.0, apl_nm2=APL_NM2):
    """Stamp a Martini 3 POPE bilayer over the spherocylinder surface.

    Returns (coords_nm (N,3), n_lipids). `density_scale`<1 thins it for demos.
    """
    tpl = pope_template().beads_nm
    pts, nrm = _surface_points(R_nm, L_nm, apl_nm2, density_scale)
    coords = []
    for leaflet, off in (("outer", +LEAFLET_OFFSET_NM), ("inner", -LEAFLET_OFFSET_NM)):
        # outer: head outward (+normal); inner: head inward (-normal)
        axis = nrm if leaflet == "outer" else -nrm
        base = pts + nrm * off
        for k in range(len(pts)):
            coords.append(_orient(tpl, axis[k]) + base[k])
    return (np.concatenate(coords) if coords else np.zeros((0, 3))), 2 * len(pts)


def write_envelope_system(coords_nm, n_lipids, out_dir, box_nm,
                          pope_itp="martini_v3.0.0_POPE.itp"):
    """Write the envelope as a Martini 3 system: envelope.gro + envelope.top.

    Each lipid is 12 beads in :data:`POPE_BEADS` order, matching the real
    Martini 3 ``martini_v3.0.0_POPE.itp`` (shipped). Bead order in the .gro
    therefore lines up with the itp's ``[ atoms ]`` for a valid MD topology.
    """
    import os
    names = (POPE_BEADS * n_lipids)[:coords_nm.shape[0]]
    gro = os.path.join(out_dir, "envelope.gro")
    write_gro(gro, names, coords_nm, box_nm)
    top = os.path.join(out_dir, "envelope.top")
    with open(top, "w") as f:
        f.write('#include "martini_v3.0.0.itp"\n')
        f.write(f'#include "{pope_itp}"\n')
        f.write(f"\n[ system ]\nMartini 3 cell envelope\n\n[ molecules ]\nPOPE {n_lipids}\n")
    return gro, top
