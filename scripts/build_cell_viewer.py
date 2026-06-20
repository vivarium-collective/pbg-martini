"""Build a self-contained 3D viewer of a parsimony whole-cell pack.

Renders *every* placement in a ``parsimony.pack.v1`` file as an instanced
sphere, sized by the ingredient's van-der-Waals enclosing radius and colored by
species (cellPACK-style). Unlike the per-bead Martini viewer (which is only
practical for a small slice), this shows the whole crowded cell -- all species,
all instances -- as the recognizable rod-shaped E. coli.

The output is one self-contained ``.html`` (three.js from CDN; geometry embedded
inline as base64 Float32 position arrays) that opens in any browser and can be
shared/hosted as-is.

Usage::

    .venv/bin/python scripts/build_cell_viewer.py \
        --pack ~/code/3d-ecoli-app/data/ecoli_3d.pack.json \
        --meta ~/code/3d-ecoli-app/data/ecoli_3d.meta.json \
        --out output/ecoli_cell_viewer.html
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import json
import os
from collections import defaultdict

import numpy as np

# Fallback radii (angstrom) for ingredients whose shape carries no
# enclosing_radius (e.g. the single-sphere lipid / dna proxies).
_DEFAULT_RADIUS = 14.0


def _ingredient_meta(pack, meta):
    ings = {i["id"]: i for i in pack["ingredients"]}
    meta_ings = (meta or {}).get("ingredients", {})
    info = {}
    for iid, ing in ings.items():
        name = ing["name"]
        shape = ing.get("shape") or {}
        radius = shape.get("enclosing_radius") or _DEFAULT_RADIUS
        color = ing.get("color") or [0.7, 0.7, 0.7]
        info[iid] = {
            "name": name,
            "display": meta_ings.get(name, {}).get("display_name", name),
            "category": meta_ings.get(name, {}).get("category", "Other"),
            "radius": float(radius),
            "color": [float(c) for c in color[:3]],
        }
    return info


def build_cell_viewer(pack_path, meta_path, out_html,
                      title="parsimony 3D E. coli (whole cell)",
                      background="#05070d", envelope=False, envelope_density=0.006):
    with open(pack_path) as fh:
        pack = json.load(fh)
    meta = None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

    info = _ingredient_meta(pack, meta)

    # Group placement positions (angstrom) by ingredient.
    by_ing = defaultdict(list)
    for p in pack["placements"]:
        by_ing[p["ingredient"]].append(p["position"])

    species = []
    total = 0
    for iid, positions in by_ing.items():
        meta_i = info.get(iid)
        if meta_i is None:
            continue
        arr = np.asarray(positions, dtype="<f4").reshape(-1, 3)
        total += arr.shape[0]
        species.append({
            "name": meta_i["display"],
            "category": meta_i["category"],
            "radius": meta_i["radius"],
            "color": meta_i["color"],
            "count": int(arr.shape[0]),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        })

    # Optional Martini 3 membrane envelope: a lipid shell on the cell capsule.
    if envelope:
        from pbg_martini.envelope import build_envelope
        env_nm, n_lip = build_envelope(R_nm=500.0, L_nm=1000.0,
                                       density_scale=envelope_density)
        env_ang = (env_nm * 10.0)[::12].astype("<f4")   # ~1 marker per lipid
        species.append({
            "name": f"Membrane envelope ({n_lip:,} lipids)",
            "category": "Membrane",
            "radius": 22.0,
            "color": [0.30, 0.85, 0.45],
            "count": int(env_ang.shape[0]),
            "b64": base64.b64encode(env_ang.tobytes()).decode("ascii"),
        })
        total += env_ang.shape[0]

    # Sort largest-radius first so big species (ribosomes) draw first.
    species.sort(key=lambda s: -s["radius"])
    bounds = pack.get("bounds", {})

    species_json = json.dumps(species)
    # Compact legend grouped by category (color from the first species seen).
    cat_color, cat_count = {}, defaultdict(int)
    for s in species:
        cat_count[s["category"]] += s["count"]
        cat_color.setdefault(s["category"], s["color"])
    legend_rows = "".join(
        f'<div class="lrow"><span class="sw" style="background:rgb('
        f'{int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"></span>'
        f'{_html.escape(cat)} <span class="n">{cat_count[cat]:,}</span></div>'
        for cat, c in sorted(cat_color.items(), key=lambda kv: -cat_count[kv[0]])
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
  html,body{{margin:0;height:100%;overflow:hidden;background:{background};color:#dfe6f3;
    font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
  #bar{{position:fixed;top:0;left:0;right:0;padding:10px 16px;z-index:10;pointer-events:none;
    background:linear-gradient(180deg,rgba(5,7,13,.92),rgba(5,7,13,0))}}
  #bar h1{{margin:0;font-size:15px;font-weight:600}}
  #bar p{{margin:2px 0 0;font-size:12px;opacity:.7}}
  #legend{{position:fixed;left:12px;bottom:12px;z-index:10;font-size:12px;
    background:rgba(5,7,13,.6);padding:8px 10px;border-radius:8px;max-width:230px}}
  .lrow{{display:flex;align-items:center;gap:6px;margin:2px 0}}
  .sw{{width:11px;height:11px;border-radius:2px;display:inline-block;flex:0 0 auto}}
  .n{{margin-left:auto;opacity:.6}}
  #c{{position:absolute;inset:0}}
</style>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head><body>
  <div id="bar"><h1>{_html.escape(title)}</h1>
    <p>{total:,} molecules &middot; {len(species)} species &middot; parsimony pack &middot; drag to rotate &middot; scroll to zoom</p></div>
  <div id="legend">{legend_rows}</div>
  <canvas id="c"></canvas>
  <script>
    const SPECIES = {species_json};
    function b64ToF32(b64) {{
      const bin = atob(b64); const len = bin.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
      return new Float32Array(bytes.buffer);
    }}
    const renderer = new THREE.WebGLRenderer({{canvas: document.getElementById('c'), antialias: true}});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("{background}");
    const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 1, 200000);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.8); key.position.set(1,1,1); scene.add(key);
    const sphere = new THREE.SphereGeometry(1, 8, 6);
    const dummy = new THREE.Object3D();
    const box = new THREE.Box3();
    for (const sp of SPECIES) {{
      const pos = b64ToF32(sp.b64); const n = pos.length / 3;
      const mat = new THREE.MeshLambertMaterial({{
        color: new THREE.Color(sp.color[0], sp.color[1], sp.color[2]) }});
      const mesh = new THREE.InstancedMesh(sphere, mat, n);
      for (let i = 0; i < n; i++) {{
        const x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
        dummy.position.set(x, y, z);
        dummy.scale.setScalar(sp.radius);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        box.expandByPoint(dummy.position);
      }}
      mesh.instanceMatrix.needsUpdate = true;
      scene.add(mesh);
    }}
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(0, 0, size * 0.7));
    camera.near = size / 1000; camera.far = size * 10; camera.updateProjectionMatrix();
    addEventListener("resize", () => {{
      camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    }});
    (function loop() {{ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); }})();
  </script>
</body></html>
"""
    with open(out_html, "w") as fh:
        fh.write(html)
    return out_html, total, len(species)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=os.path.expanduser(
        "~/code/3d-ecoli-app/data/ecoli_3d.pack.json"))
    ap.add_argument("--meta", default=os.path.expanduser(
        "~/code/3d-ecoli-app/data/ecoli_3d.meta.json"))
    ap.add_argument("--out", default="output/ecoli_cell_viewer.html")
    ap.add_argument("--envelope", action="store_true",
                    help="overlay the Martini 3 membrane envelope (lipid shell)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out, total, nspec = build_cell_viewer(args.pack, args.meta, args.out,
                                          envelope=args.envelope)
    size_mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({total:,} molecules, {nspec} species, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
