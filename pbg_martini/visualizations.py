"""Visualization Step subclasses for pbg-martini.

Visualizations follow the pbg-superpowers convention (v0.4.15+):
each subclass overrides ``update()`` to consume per-step state via wires
(like an Emitter), accumulates history internally, and returns
``{'html': '<rendered figure>'}`` each step. The composite spec wires
the input ports to store paths.

See ``pbg_superpowers.visualization`` for the base-class contract.
"""
from __future__ import annotations

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
