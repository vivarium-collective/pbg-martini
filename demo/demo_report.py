"""Demo: Martini CG membrane systems — publication-quality interactive report.

Builds four complex molecular assemblies:
1. Asymmetric plasma membrane (POPC/POPE/CHOL/SM)
2. DPC detergent micelle
3. WALP23 transmembrane helix in POPC/CHOL bilayer
4. POPC/POPE/CHOL/DPPC vesicle (liposome)

Each section features an interactive Three.js 3D viewer with instanced
sphere rendering, Plotly composition charts, bigraph-viz architecture
diagrams, and collapsible PBG document trees.
"""

import json
import os
import time
import base64
import tempfile
import subprocess
import numpy as np

from process_bigraph import allocate_core
from pbg_martini.builders import (
    build_bilayer,
    build_micelle,
    build_protein_in_membrane,
    build_vesicle,
    LIPID_TEMPLATES,
)
from pbg_martini.composites import make_martinize_document


# ── Configurations ──────────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'plasma',
        'title': 'Asymmetric Plasma Membrane',
        'subtitle': '392 lipids — POPC / POPE / Cholesterol / Sphingomyelin',
        'description': (
            'A realistic plasma membrane patch with four lipid species at '
            'physiological proportions. POPC and POPE provide the phospholipid '
            'bilayer core, cholesterol modulates fluidity and thickness, and '
            'sphingomyelin contributes to lipid raft formation. The 14×14 grid '
            'produces ~392 lipids across two leaflets with random orientations.'
        ),
        'builder': 'bilayer',
        'params': {
            'composition': {'POPC': 0.35, 'POPE': 0.25, 'CHOL': 0.25, 'SM': 0.15},
            'nx_lipids': 14, 'ny_lipids': 14, 'spacing': 0.65, 'seed': 42,
        },
        'color_scheme': 'indigo',
        'camera_dist': 1.8,
    },
    {
        'id': 'micelle',
        'title': 'DPC Detergent Micelle',
        'subtitle': '80 DPC molecules — spherical self-assembly',
        'description': (
            'A dodecylphosphocholine (DPC) micelle with 80 detergent molecules '
            'arranged on a sphere. DPC is the most widely used detergent for '
            'NMR studies of membrane proteins. Each DPC maps to 6 Martini beads '
            '(headgroup + single tail). The micelle radius of 2.8 nm matches '
            'experimental estimates from SAXS measurements.'
        ),
        'builder': 'micelle',
        'params': {'lipid_name': 'DPC', 'n_lipids': 80, 'radius': 2.8, 'seed': 77},
        'color_scheme': 'emerald',
        'camera_dist': 2.5,
    },
    {
        'id': 'protein',
        'title': 'WALP23 in POPC/Cholesterol Bilayer',
        'subtitle': 'Transmembrane helix embedded in 320+ lipids',
        'description': (
            'A WALP23-like model peptide (23 residues: Trp-Ala/Leu₁₉-Trp) '
            'embedded in a POPC/cholesterol bilayer. WALP peptides are the '
            'canonical model system for studying hydrophobic mismatch and '
            'lipid-protein interactions. The helix spans the bilayer along the '
            'z-axis with anchoring tryptophans at the water-lipid interface.'
        ),
        'builder': 'protein_membrane',
        'params': {
            'composition': {'POPC': 0.70, 'CHOL': 0.30},
            'nx_lipids': 14, 'ny_lipids': 14, 'spacing': 0.65,
            'n_helix_residues': 23, 'exclusion_radius': 0.8, 'seed': 314,
        },
        'color_scheme': 'rose',
        'camera_dist': 1.8,
    },
    {
        'id': 'vesicle',
        'title': 'Mixed-Lipid Vesicle',
        'subtitle': '570 lipids — spherical liposome with two leaflets',
        'description': (
            'A small unilamellar vesicle (SUV) with 350 lipids in the outer '
            'leaflet and 220 in the inner leaflet, reflecting the natural area '
            'asymmetry of curved bilayers. The four-component composition '
            '(POPC/POPE/CHOL/DPPC) models a simplified eukaryotic membrane '
            'vesicle used in drug delivery and membrane biophysics studies.'
        ),
        'builder': 'vesicle',
        'params': {
            'composition': {'POPC': 0.40, 'POPE': 0.25, 'CHOL': 0.20, 'DPPC': 0.15},
            'n_lipids_outer': 350, 'n_lipids_inner': 220,
            'outer_radius': 7.0, 'inner_radius': 5.2, 'seed': 2024,
        },
        'color_scheme': 'amber',
        'camera_dist': 3.0,
    },
]

COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
    'amber': {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#d97706',
              'bg': '#fffbeb', 'accent': '#fbbf24', 'text': '#78350f'},
}

# Lipid legend colors (match builders.py)
LIPID_COLORS_HEX = {
    'POPC': '#4d8de6', 'POPE': '#d97333', 'CHOL': '#f2d933',
    'SM': '#cc4da6', 'DPC': '#66cc80', 'DPPC': '#80b3f2',
}


def run_config(cfg):
    """Run a builder for one config and return results + runtime."""
    t0 = time.perf_counter()
    if cfg['builder'] == 'bilayer':
        result = build_bilayer(**cfg['params'])
    elif cfg['builder'] == 'micelle':
        result = build_micelle(**cfg['params'])
    elif cfg['builder'] == 'protein_membrane':
        result = build_protein_in_membrane(**cfg['params'])
    elif cfg['builder'] == 'vesicle':
        result = build_vesicle(**cfg['params'])
    else:
        raise ValueError(f'Unknown builder: {cfg["builder"]}')
    runtime = time.perf_counter() - t0
    return result, runtime


def generate_bigraph_image(cfg):
    """Generate a colored bigraph-viz PNG."""
    from bigraph_viz import plot_bigraph

    step_name = {
        'bilayer': 'MembraneBuilder',
        'micelle': 'MicelleBuilder',
        'protein_membrane': 'ProteinMembrane',
        'vesicle': 'VesicleBuilder',
    }[cfg['builder']]

    doc = {
        step_name: {
            '_type': 'step',
            'address': f'local:{step_name}Step',
            'outputs': {
                'beads': ['stores', 'beads'],
                'bonds': ['stores', 'bonds'],
                'stats': ['stores', 'stats'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'inputs': {
                'stats': ['stores', 'stats'],
            },
        },
    }

    node_colors = {
        (step_name,): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc, out_dir=outdir, filename='bigraph',
        file_format='png', remove_process_place_edges=True,
        rankdir='LR', node_fill_colors=node_colors,
        node_label_size='16pt', port_labels=False, dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def generate_html(sim_results, output_path):
    """Generate the full HTML report."""

    sections_html = []
    all_js_data = {}

    for idx, (cfg, (result, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        stats = result['stats']

        # Prepare JS data — only send positions and colors as flat arrays for performance
        positions_flat = []
        colors_flat = []
        labels = []
        is_head_flags = []
        is_protein_flags = []
        for b in result['beads']:
            positions_flat.extend(b['pos'])
            colors_flat.extend(b['color'])
            labels.append(b.get('label', ''))
            is_head_flags.append(b.get('is_head', False))
            is_protein_flags.append(b.get('is_protein', False))

        # Lipid composition for charts
        lip_counts = stats.get('lipid_counts', {})

        all_js_data[sid] = {
            'positions': positions_flat,
            'colors': colors_flat,
            'bonds': result['bonds'],
            'n_beads': len(result['beads']),
            'labels': labels,
            'is_head': is_head_flags,
            'is_protein': is_protein_flags,
            'lipid_counts': lip_counts,
            'camera_dist': cfg['camera_dist'],
        }

        # Bigraph image
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        # PBG document
        pbg_doc = {
            'builder': cfg['builder'],
            'params': cfg['params'],
            'stats': stats,
        }

        # Build metrics row
        n_beads = stats.get('n_beads', len(result['beads']))
        n_bonds = stats.get('n_bonds', len(result['bonds']))
        n_lipids = stats.get('n_lipids', 0)

        extra_metrics = ''
        if cfg['builder'] == 'bilayer':
            box = stats.get('box_nm', [0, 0])
            extra_metrics = f'''
        <div class="metric"><span class="metric-label">Patch Size</span><span class="metric-value">{box[0]:.1f} × {box[1]:.1f}</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Area/Lipid</span><span class="metric-value">{stats.get("area_per_lipid_nm2", 0):.3f}</span><span class="metric-sub">nm²</span></div>'''
        elif cfg['builder'] == 'micelle':
            extra_metrics = f'''
        <div class="metric"><span class="metric-label">Radius</span><span class="metric-value">{stats.get("radius_nm", 0):.1f}</span><span class="metric-sub">nm</span></div>'''
        elif cfg['builder'] == 'protein_membrane':
            extra_metrics = f'''
        <div class="metric"><span class="metric-label">Helix</span><span class="metric-value">{stats.get("n_protein_residues", 0)} res</span></div>
        <div class="metric"><span class="metric-label">Prot. Beads</span><span class="metric-value">{stats.get("n_protein_beads", 0)}</span></div>'''
        elif cfg['builder'] == 'vesicle':
            extra_metrics = f'''
        <div class="metric"><span class="metric-label">Outer R</span><span class="metric-value">{stats.get("outer_radius_nm", 0):.1f}</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Inner R</span><span class="metric-value">{stats.get("inner_radius_nm", 0):.1f}</span><span class="metric-sub">nm</span></div>'''

        # Lipid legend HTML
        legend_items = ''
        for lip_name, count in sorted(lip_counts.items()):
            hex_color = LIPID_COLORS_HEX.get(lip_name, '#999')
            legend_items += (
                f'<div class="legend-item">'
                f'<div class="legend-dot" style="background:{hex_color};"></div>'
                f'<span>{lip_name} ({count})</span></div>'
            )

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
        <div class="metric"><span class="metric-label">Lipids</span><span class="metric-value">{n_lipids}</span></div>
        <div class="metric"><span class="metric-label">CG Beads</span><span class="metric-value">{n_beads:,}</span></div>
        <div class="metric"><span class="metric-label">Bonds</span><span class="metric-value">{n_bonds:,}</span></div>
        {extra_metrics}
        <div class="metric"><span class="metric-label">Build Time</span><span class="metric-value">{runtime*1000:.0f}ms</span></div>
      </div>

      <h3 class="subsection-title">3D Molecular Structure</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mol-canvas"></canvas>
        <div class="viewer-info">
          <strong>{n_beads:,}</strong> beads &middot; <strong>{n_bonds:,}</strong> bonds<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="legend-box">
          {legend_items}
        </div>
      </div>

      <h3 class="subsection-title">Composition Analysis</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-comp-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-beads-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Builder Configuration</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    pbg_docs = {}
    for cfg, (result, runtime) in sim_results:
        pbg_docs[cfg['id']] = {
            'builder': cfg['builder'],
            'params': cfg['params'],
            'stats': result['stats'],
        }

    html = _build_html(nav_items, sections_html, all_js_data, pbg_docs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report written to {output_path}')


def _build_html(nav_items, sections_html, all_js_data, pbg_docs):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Martini CG Membrane Systems Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}

/* Header */
.page-header {{
  background:linear-gradient(135deg,#0f172a 0%,#1e293b 40%,#334155 100%);
  padding:3.5rem 3rem 3rem; color:#fff; position:relative; overflow:hidden;
}}
.page-header::after {{
  content:''; position:absolute; top:0; right:0; bottom:0; left:0;
  background:radial-gradient(ellipse at 70% 20%, rgba(99,102,241,0.15) 0%, transparent 60%),
             radial-gradient(ellipse at 30% 80%, rgba(244,63,94,0.10) 0%, transparent 50%);
}}
.page-header h1 {{ font-size:2.4rem; font-weight:800; position:relative; z-index:1; margin-bottom:.4rem;
  background:linear-gradient(135deg,#fff 0%,#e0e7ff 100%); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; }}
.page-header p {{ color:#94a3b8; font-size:.95rem; max-width:750px; position:relative; z-index:1; }}

/* Nav */
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100;
        backdrop-filter:blur(8px); background:rgba(248,250,252,0.92); }}
.nav-link {{ padding:.45rem 1.1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .2s; }}
.nav-link:hover {{ transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,.1); }}

/* Sections */
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem; padding-left:1rem; }}
.sim-number {{ width:40px; height:40px; border-radius:12px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.2rem; }}
.sim-title {{ font-size:1.6rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:850px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155; margin:1.8rem 0 .8rem; }}

/* Metrics */
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
                gap:.7rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.7rem; text-align:center; transition:transform .15s; }}
.metric:hover {{ transform:translateY(-2px); box-shadow:0 2px 8px rgba(0,0,0,.06); }}
.metric-label {{ display:block; font-size:.68rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.15rem; }}
.metric-value {{ display:block; font-size:1.25rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.68rem; color:#94a3b8; }}

/* 3D Viewer */
.viewer-wrap {{ position:relative; background:#0f172a; border:1px solid #334155;
                border-radius:16px; overflow:hidden; margin-bottom:1rem;
                box-shadow:0 4px 20px rgba(0,0,0,0.15); }}
.mol-canvas {{ width:100%; height:550px; display:block; cursor:grab; }}
.mol-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:1rem; left:1rem; background:rgba(15,23,42,0.85);
                border:1px solid #334155; border-radius:10px; padding:.6rem 1rem;
                font-size:.75rem; color:#94a3b8; backdrop-filter:blur(6px); }}
.viewer-info strong {{ color:#e2e8f0; }}
.legend-box {{ position:absolute; top:1rem; right:1rem; background:rgba(15,23,42,0.85);
               border:1px solid #334155; border-radius:10px; padding:.7rem .9rem;
               font-size:.72rem; color:#cbd5e1; backdrop-filter:blur(6px); }}
.legend-item {{ display:flex; align-items:center; gap:.5rem; margin:.25rem 0; }}
.legend-dot {{ width:11px; height:11px; border-radius:50%; border:1px solid rgba(255,255,255,0.2); }}

/* Charts */
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; overflow:hidden; }}
.chart {{ height:300px; }}

/* PBG row */
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:12px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}

/* JSON tree */
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
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

.footer {{ text-align:center; padding:2.5rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; background:#f8fafc; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>Martini CG Membrane Systems</h1>
  <p>Four complex molecular assemblies built with the <strong>Martini</strong>
  coarse-grained force field, wrapped as <strong>process-bigraph</strong> Steps.
  Interactive 3D visualization of plasma membranes, micelles, protein-lipid
  complexes, and vesicles — from hundreds to thousands of CG beads.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-martini</strong> &mdash;
  Martini Force Field &middot; vermouth/martinize2 &middot; process-bigraph
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// ─── JSON Tree Viewer ───────────────────────────────────────────────
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + (Number.isInteger(obj) ? obj : obj.toFixed(4)) + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      return '<span class="jt-bracket">[</span>' + obj.map(x => renderJson(x, depth+1)).join(', ') + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const col = depth >= 1;
    let h = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' + (col?'&#9654;':'&#9660;') + '</span>';
    h += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + '</span>';
    h += '<div id="' + id + '"' + (col?' class="jt-collapsed"':'') + ' style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ h += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    h += '</div><span class="jt-bracket">]</span>';
    return h;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const col = depth >= 2;
    let h = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' + (col?'&#9654;':'&#9660;') + '</span>';
    h += '<span class="jt-bracket">{{</span>';
    h += '<div id="' + id + '"' + (col?' class="jt-collapsed"':'') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      h += '<div><span class="jt-key">' + k + '</span>: ' + renderJson(obj[k], depth+1) + (i<keys.length-1?',':'') + '</div>';
    }});
    h += '</div><span class="jt-bracket">}}</span>';
    return h;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  const isCol = el.classList.contains('jt-collapsed');
  el.classList.toggle('jt-collapsed');
  const p = el.previousElementSibling;
  if (p && p.previousElementSibling && p.previousElementSibling.classList.contains('jt-toggle'))
    p.previousElementSibling.innerHTML = isCol ? '&#9660;' : '&#9654;';
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Three.js Viewers (Instanced Spheres) ───────────────────────────

function initViewer(sid) {{
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 550;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true, alpha:false}});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.setClearColor(0x0f172a);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0f172a, 0.012);

  const cam = new THREE.PerspectiveCamera(50, W/H, 0.01, 200);

  // Compute center and extent
  const n = d.n_beads;
  let cx=0, cy=0, cz=0;
  for (let i=0; i<n; i++) {{ cx+=d.positions[i*3]; cy+=d.positions[i*3+1]; cz+=d.positions[i*3+2]; }}
  cx/=n; cy/=n; cz/=n;
  let maxR = 0;
  for (let i=0; i<n; i++) {{
    const dx=d.positions[i*3]-cx, dy=d.positions[i*3+1]-cy, dz=d.positions[i*3+2]-cz;
    maxR = Math.max(maxR, Math.sqrt(dx*dx+dy*dy+dz*dz));
  }}
  const dist = maxR * d.camera_dist;
  cam.position.set(cx + dist*0.6, cy + dist*0.5, cz + dist*0.7);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.target.set(cx, cy, cz);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;
  controls.maxDistance = dist * 4;
  controls.minDistance = dist * 0.3;

  // Lighting — warm key + cool fill + rim
  scene.add(new THREE.AmbientLight(0x334155, 0.4));
  const key = new THREE.DirectionalLight(0xfff5e6, 0.9);
  key.position.set(5, 8, 6); scene.add(key);
  const fill = new THREE.DirectionalLight(0xc7d2fe, 0.35);
  fill.position.set(-4, -2, -5); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xa78bfa, 0.25);
  rim.position.set(-2, 5, -3); scene.add(rim);

  // Determine bead radius based on system size
  const beadR = maxR > 5 ? 0.12 : maxR > 2 ? 0.06 : 0.04;
  const bondR = beadR * 0.22;

  // Instanced sphere rendering
  const sphereGeo = new THREE.SphereGeometry(1, 14, 10);
  const sphereMat = new THREE.MeshStandardMaterial({{
    roughness: 0.35, metalness: 0.05,
  }});
  const mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, n);
  const dummy = new THREE.Object3D();
  const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);

  for (let i = 0; i < n; i++) {{
    dummy.position.set(d.positions[i*3], d.positions[i*3+1], d.positions[i*3+2]);
    // Larger beads for heads and proteins
    let r = beadR;
    if (d.is_head[i]) r *= 1.3;
    if (d.is_protein[i]) r *= 1.5;
    dummy.scale.set(r, r, r);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    colorAttr.setXYZ(i, d.colors[i*3], d.colors[i*3+1], d.colors[i*3+2]);
  }}
  mesh.instanceColor = colorAttr;
  scene.add(mesh);

  // Bonds — use line segments for performance with large systems
  if (d.bonds.length < 8000) {{
    const bondPositions = [];
    const bondColors = [];
    for (const [a, b] of d.bonds) {{
      bondPositions.push(d.positions[a*3], d.positions[a*3+1], d.positions[a*3+2]);
      bondPositions.push(d.positions[b*3], d.positions[b*3+1], d.positions[b*3+2]);
      // Use dimmed colors
      bondColors.push(d.colors[a*3]*0.5, d.colors[a*3+1]*0.5, d.colors[a*3+2]*0.5);
      bondColors.push(d.colors[b*3]*0.5, d.colors[b*3+1]*0.5, d.colors[b*3+2]*0.5);
    }}
    const bondGeo = new THREE.BufferGeometry();
    bondGeo.setAttribute('position', new THREE.Float32BufferAttribute(bondPositions, 3));
    bondGeo.setAttribute('color', new THREE.Float32BufferAttribute(bondColors, 3));
    const bondMat = new THREE.LineBasicMaterial({{ vertexColors: true, transparent: true, opacity: 0.4 }});
    scene.add(new THREE.LineSegments(bondGeo, bondMat));
  }}

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }}
  animate();
}}

Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ──────────────────────────────────────────────────
const lipColors = {json.dumps(LIPID_COLORS_HEX)};
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:50, r:25, t:40, b:50 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];
  const lc = d.lipid_counts;
  const names = Object.keys(lc);
  const counts = names.map(n => lc[n]);
  const colors = names.map(n => lipColors[n] || '#94a3b8');

  // Donut chart: composition
  Plotly.newPlot('chart-comp-'+sid, [{{
    labels: names, values: counts, type: 'pie', hole: 0.45,
    marker: {{ colors: colors, line: {{ color: '#fff', width: 2 }} }},
    textinfo: 'label+value',
    textfont: {{ size: 12 }},
  }}], {{...pLayout,
    title: {{ text: 'Lipid Composition', font: {{ size: 13, color: '#1e293b' }} }},
    showlegend: false,
    annotations: [{{ text: counts.reduce((a,b)=>a+b, 0) + '<br>lipids',
                     font: {{ size: 14, color: '#64748b' }}, showarrow: false }}],
  }}, pCfg);

  // Bar chart: beads per lipid type
  const beadsPerType = {{}};
  names.forEach(n => {{
    // Approximate beads per lipid from template sizes
    const templates = {json.dumps({k: len(v['beads']) for k, v in LIPID_TEMPLATES.items()})};
    beadsPerType[n] = lc[n] * (templates[n] || 10);
  }});
  const bpNames = Object.keys(beadsPerType);
  const bpValues = bpNames.map(n => beadsPerType[n]);
  Plotly.newPlot('chart-beads-'+sid, [{{
    x: bpNames, y: bpValues, type: 'bar',
    marker: {{ color: bpNames.map(n => lipColors[n] || '#94a3b8'),
               line: {{ color: '#fff', width: 1 }} }},
    text: bpValues.map(v => v.toLocaleString()),
    textposition: 'auto',
  }}], {{...pLayout,
    title: {{ text: 'CG Beads by Lipid Type', font: {{ size: 13, color: '#1e293b' }} }},
    yaxis: {{...pLayout.yaxis, title: {{ text: 'Bead count', font: {{ size: 10 }} }} }},
    xaxis: {{...pLayout.xaxis, title: {{ text: 'Lipid', font: {{ size: 10 }} }} }},
  }}, pCfg);
}});
</script>
</body>
</html>"""


def main():
    print('Martini CG Membrane Systems — Demo Report')
    print('=' * 50)

    sim_results = []
    for cfg in CONFIGS:
        print(f'\n  Building: {cfg["title"]}...')
        result, runtime = run_config(cfg)
        n = result['stats'].get('n_beads', len(result['beads']))
        print(f'    {n:,} beads, {len(result["bonds"]):,} bonds in {runtime*1000:.0f}ms')
        sim_results.append((cfg, (result, runtime)))

    output_path = os.path.join(os.path.dirname(__file__), 'report.html')
    print(f'\nGenerating HTML report...')
    generate_html(sim_results, output_path)

    subprocess.run(['open', '-a', 'Safari', output_path])
    print('Done! Report opened in Safari.')


if __name__ == '__main__':
    main()
