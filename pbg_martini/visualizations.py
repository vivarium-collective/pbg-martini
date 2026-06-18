"""Visualization Step subclasses for pbg-martini.

Visualizations follow the pbg-superpowers convention (v0.4.15+):
each subclass overrides ``update()`` to consume per-step state via wires
(like an Emitter), accumulates history internally, and returns
``{'html': '<rendered figure>'}`` each step. The composite spec wires
the input ports to store paths.

See ``pbg_superpowers.visualization`` for the base-class contract.
"""
from __future__ import annotations

import html as _html

import numpy as np

from pbg_superpowers.visualization import Visualization


class MartiniBeadSummaryPlots(Visualization):
    """Time-series HTML plot of Martini CG mapping summary scalars.

    Consumes the four scalar outputs from ``MartinizeStep`` (n_atoms_input,
    n_atoms_full, n_beads, n_bonds_cg, reduction_ratio) per step,
    accumulates them across calls, and emits a Plotly HTML figure on
    every update. Downstream consumers (dashboards, notebook viewers)
    read the latest ``html`` from the wired store.

    Because ``MartinizeStep`` is a stateless Step that re-runs the full
    pipeline each update, repeated calls produce identical scalars unless
    the input changes — but the trace is still useful as a sanity-check
    panel in the report.
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'Martini CG mapping'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times: list[float] = []
        self.history: dict[str, list[float]] = {
            'n_atoms_input': [],
            'n_atoms_full': [],
            'n_beads': [],
            'n_bonds_cg': [],
            'reduction_ratio': [],
        }

    def inputs(self):
        return {
            'n_atoms_input': 'integer',
            'n_atoms_full': 'integer',
            'n_beads': 'integer',
            'n_bonds_cg': 'integer',
            'reduction_ratio': 'float',
            'time': 'float',
        }

    def update(self, state, interval=1.0):
        self.times.append(float(state.get('time', len(self.times) * (interval or 1.0))))
        for key in self.history:
            v = state.get(key)
            self.history[key].append(float(v) if v is not None else 0.0)

        title = (self.config or {}).get('title', 'Martini CG mapping')
        traces = []
        for key, ys in self.history.items():
            traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines+markers","name":"' + key + '"}'
            )
        html = (
            f'<div id="mbsp" style="height:380px"></div>'
            f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            f'<script>Plotly.newPlot("mbsp",[{",".join(traces)}],'
            f'{{title:"{title}",margin:{{l:55,r:15,t:35,b:40}},'
            f'xaxis:{{title:"time"}},'
            f'legend:{{orientation:"h",y:-0.2}}}},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )
        return {'html': html}


# ---------------------------------------------------------------------------
# Task 11: parsimony assembly 3D viewer + HTML report
# ---------------------------------------------------------------------------

def _read_gro(gro_path):
    """Read ``(coords_nm (N,3), res_names list[str], box_nm tuple)`` from a .gro."""
    with open(gro_path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[1].strip())
    coords = np.zeros((n, 3), dtype=float)
    res_names = []
    for i in range(n):
        ln = lines[2 + i]
        res_names.append(ln[5:10].strip())
        coords[i] = (float(ln[20:28]), float(ln[28:36]), float(ln[36:44]))
    box = tuple(float(v) for v in lines[2 + n].split()[:3]) if len(lines) > 2 + n else (0.0, 0.0, 0.0)
    return coords, res_names, box


def _min_pairwise(coords, cap=2000):
    """Minimum pairwise distance (nm). Subsamples above ``cap`` beads for speed."""
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 2:
        return float("nan")
    if n > cap:
        idx = np.linspace(0, n - 1, cap).astype(int)
        coords = coords[idx]
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


def _scatter3d_html(coords, res_names, div_id="parsimony3d", height=460):
    """Build a self-contained Plotly scatter3d HTML block colored by species."""
    coords = np.asarray(coords, dtype=float)
    # Group bead indices by residue/species label so each becomes a colored trace.
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(res_names):
        groups.setdefault(name or "bead", []).append(i)
    traces = []
    for name, idx in groups.items():
        xs = coords[idx, 0].tolist()
        ys = coords[idx, 1].tolist()
        zs = coords[idx, 2].tolist()
        traces.append(
            '{"x":' + repr(xs) + ',"y":' + repr(ys) + ',"z":' + repr(zs) +
            ',"mode":"markers","type":"scatter3d","name":"' + _html.escape(name) +
            '","marker":{"size":4}}'
        )
    return (
        f'<div id="{div_id}" style="height:{height}px"></div>'
        f'<script>Plotly.newPlot("{div_id}",[{",".join(traces)}],'
        f'{{margin:{{l:0,r:0,t:0,b:0}},'
        f'scene:{{aspectmode:"data",xaxis:{{title:"x (nm)"}},'
        f'yaxis:{{title:"y (nm)"}},zaxis:{{title:"z (nm)"}}}},'
        f'legend:{{orientation:"h",y:-0.05}}}},'
        f'{{responsive:true,displayModeBar:false}});</script>'
    )


def build_parsimony_report(
    assembly_summary: dict,
    relaxed_gro: str,
    out_html: str,
    traj: str | None = None,
    final_energy: float | None = None,
    md_ran: bool = False,
    title: str = "parsimony → Martini whole-cell slice",
) -> str:
    """Render a self-contained HTML report for a parsimony->Martini assembly.

    Shows per-species instance counts, total bead count, the box dimensions, the
    minimum pairwise distance before/after the WCA relax (the "clash" metric),
    a 3D viewer of the relaxed slice colored by species, and the OpenMM final
    energy when the best-effort MD stage ran.

    Parameters
    ----------
    assembly_summary : dict
        The return value of :func:`pbg_martini.parsimony_assembler.assemble`
        (``gro``, ``top``, ``n_beads``, ``n_molecules``, ``per_species_counts``,
        ``box_nm``, ...).
    relaxed_gro : str
        Path to the post-relax ``.gro`` (often equal to ``assembly_summary['gro']``
        when relax wrote back in place; clash-after is computed from this).
    out_html : str
        Destination path. Returned on success.
    traj : str, optional
        Trajectory path (recorded in the report header when present).
    final_energy : float, optional
        OpenMM potential energy (kJ/mol) when the MD stage ran.
    md_ran : bool
        Whether the OpenMM short MD stage actually executed.
    """
    counts = assembly_summary.get("per_species_counts", {}) or {}
    n_beads = assembly_summary.get("n_beads", 0)
    n_molecules = assembly_summary.get("n_molecules", 0)
    box_nm = assembly_summary.get("box_nm", (0.0, 0.0, 0.0))
    rendered_by = assembly_summary.get("rendered_by", "stamper")

    pre_gro = assembly_summary.get("gro")
    coords_before = res_before = None
    if pre_gro:
        try:
            coords_before, res_before, _ = _read_gro(pre_gro)
        except Exception:
            coords_before = None
    coords_after, res_after, box_read = (None, None, box_nm)
    try:
        coords_after, res_after, box_read = _read_gro(relaxed_gro)
    except Exception:
        pass
    if box_read and any(box_read):
        box_nm = box_read

    clash_before = _min_pairwise(coords_before) if coords_before is not None else float("nan")
    clash_after = _min_pairwise(coords_after) if coords_after is not None else float("nan")

    viewer_coords = coords_after if coords_after is not None else coords_before
    viewer_res = res_after if res_after is not None else res_before
    viewer_html = (
        _scatter3d_html(viewer_coords, viewer_res)
        if viewer_coords is not None and viewer_coords.shape[0]
        else "<em>no beads to display</em>"
    )

    rows = "".join(
        f"<tr><td><code>{_html.escape(str(name))}</code></td>"
        f"<td style='text-align:right'>{count}</td></tr>"
        for name, count in sorted(counts.items())
    )

    def _fmt(v):
        return f"{v:.3f} nm" if v == v else "n/a"  # NaN-safe

    energy_block = ""
    if md_ran and final_energy is not None:
        energy_block = (
            f'<div class="metric"><div class="k">OpenMM final energy</div>'
            f'<div class="v">{final_energy} kJ/mol</div></div>'
        )
    elif final_energy is not None:
        energy_block = (
            f'<div class="metric"><div class="k">OpenMM final energy</div>'
            f'<div class="v">{final_energy} kJ/mol (stage skipped flag)</div></div>'
        )

    traj_note = (
        f"<p class='lead'>Trajectory: <code>{_html.escape(traj)}</code></p>"
        if traj else ""
    )
    md_status = "ran" if md_ran else "skipped (best-effort)"

    bx, by, bz = (list(box_nm) + [0.0, 0.0, 0.0])[:3]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{_html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family:-apple-system,sans-serif; max-width:1000px; margin:2rem auto;
         color:#1e293b; line-height:1.55; padding:0 1rem; }}
  h1 {{ margin-bottom:.2rem }}
  .lead {{ color:#64748b }}
  h2 {{ margin-top:2rem; padding-bottom:.3rem; border-bottom:1px solid #e2e8f0 }}
  .metrics {{ display:flex; flex-wrap:wrap; gap:1rem; margin:1rem 0 }}
  .metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
             padding:.7rem 1rem; min-width:140px }}
  .metric .k {{ color:#64748b; font-size:.78rem }}
  .metric .v {{ font-size:1.25rem; font-weight:600 }}
  table {{ border-collapse:collapse; margin-top:.5rem }}
  td,th {{ border:1px solid #e2e8f0; padding:.35rem .8rem }}
  th {{ background:#f1f5f9 }}
  code {{ background:#f1f5f9; padding:.05rem .3rem; border-radius:4px }}
</style></head>
<body>
  <h1>{_html.escape(title)}</h1>
  <p class="lead">Martini CG molecules assembled at the parsimony-measured
     positions/orientations (replacing Bentopy random packing), then WCA-relaxed.
     Renderer: <code>{_html.escape(str(rendered_by))}</code>. MD stage: {md_status}.</p>
  {traj_note}

  <div class="metrics">
    <div class="metric"><div class="k">n_beads</div><div class="v">{n_beads}</div></div>
    <div class="metric"><div class="k">n_molecules</div><div class="v">{n_molecules}</div></div>
    <div class="metric"><div class="k">box</div>
      <div class="v">{bx:.1f}×{by:.1f}×{bz:.1f} nm</div></div>
    <div class="metric"><div class="k">min pairwise (pre-relax)</div>
      <div class="v">{_fmt(clash_before)}</div></div>
    <div class="metric"><div class="k">min pairwise (post-relax)</div>
      <div class="v">{_fmt(clash_after)}</div></div>
    {energy_block}
  </div>

  <h2>Per-species instance counts</h2>
  <table><tr><th>species</th><th>instances</th></tr>{rows}</table>

  <h2>3D view of the assembled slice</h2>
  <p class="lead">One trace per species (bead positions, nm).</p>
  {viewer_html}
</body></html>
"""

    with open(out_html, "w") as fh:
        fh.write(html)
    return out_html


# ---------------------------------------------------------------------------
# Shareable interactive 3D viewer (NGL) of the assembled Martini CG slice
# ---------------------------------------------------------------------------

def build_ngl_viewer(
    gro_path: str,
    out_html: str,
    title: str = "parsimony -> Martini E. coli (CG slice)",
    bead_radius: float = 2.5,
    color_scheme: str = "resname",
    background: str = "#0b0f1a",
) -> str:
    """Write a self-contained, shareable HTML that renders a Martini CG ``.gro``
    in 3D as space-filling beads via NGL (loaded from CDN).

    Unlike :func:`build_parsimony_report`'s Plotly scatter, this is a true
    molecular viewer: every CG bead is a sphere, rotatable/zoomable, and the
    whole structure is embedded inline so the single ``.html`` file can be
    shared or hosted as-is (open it in any browser -- no server, no local files).

    Parameters
    ----------
    gro_path : str
        Path to the assembled / relaxed Martini ``.gro`` (coordinates in nm).
    out_html : str
        Output HTML path.
    title : str
        Heading shown over the viewer.
    bead_radius : float
        Sphere radius (angstrom) for each CG bead.
    color_scheme : str
        NGL color scheme (e.g. ``"resname"``, ``"chainindex"``, ``"atomindex"``).
    background : str
        Viewport background color.
    """
    with open(gro_path) as fh:
        gro_text = fh.read()
    n_beads = 0
    try:
        n_beads = int(gro_text.splitlines()[1].strip())
    except (IndexError, ValueError):
        pass
    gro_block = _html.escape(gro_text)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
  html,body{{margin:0;height:100%;background:{background};color:#dfe6f3;
    font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
  #bar{{position:fixed;top:0;left:0;right:0;padding:10px 16px;z-index:10;
    background:linear-gradient(180deg,rgba(11,15,26,.92),rgba(11,15,26,0));}}
  #bar h1{{margin:0;font-size:15px;font-weight:600}}
  #bar p{{margin:2px 0 0;font-size:12px;opacity:.7}}
  #viewport{{position:absolute;inset:0}}
</style>
<script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
</head><body>
  <div id="bar"><h1>{_html.escape(title)}</h1>
    <p>{n_beads:,} Martini CG beads &middot; drag to rotate &middot; scroll to zoom &middot; NGL spacefill</p></div>
  <div id="viewport"></div>
  <script type="text/plain" id="grodata">{gro_block}</script>
  <script>
    document.addEventListener("DOMContentLoaded", function () {{
      var stage = new NGL.Stage("viewport", {{ backgroundColor: "{background}" }});
      window.addEventListener("resize", function () {{ stage.handleResize(); }}, false);
      var groText = document.getElementById("grodata").textContent;
      var blob = new Blob([groText], {{ type: "text/plain" }});
      stage.loadFile(blob, {{ ext: "gro" }}).then(function (comp) {{
        comp.addRepresentation("spacefill", {{
          radius: {bead_radius}, colorScheme: "{color_scheme}" }});
        comp.autoView();
      }});
    }});
  </script>
</body></html>
"""
    with open(out_html, "w") as fh:
        fh.write(html)
    return out_html
