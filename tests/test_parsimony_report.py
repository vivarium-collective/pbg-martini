"""Task 11: parsimony assembly 3D viewer + HTML report (smoke)."""

import os

import numpy as np

from pbg_martini.parsimony_assembler import (
    CGTemplate,
    Placement,
    SliceSpec,
    assemble,
)
from pbg_martini.parsimony_md import relax_assembly
from pbg_martini.visualizations import build_parsimony_report


def _stub_pack(tmp_path):
    import json
    pack = {
        "format": "parsimony.pack.v1",
        "bounds": {"min": [-1000, -1000, -1000], "max": [1000, 1000, 1000]},
        "ingredients": [
            {"id": 0, "name": "groel", "color": [1, 0, 0]},
            {"id": 1, "name": "EG10367-MONOMER", "color": [0, 1, 0]},
        ],
        "placements": [
            {"ingredient": 0, "position": [10, 10, 10], "rotation": [1, 0, 0, 0], "uid": 0},
            {"ingredient": 1, "position": [20, 0, 0], "rotation": [1, 0, 0, 0], "uid": 1},
            {"ingredient": 1, "position": [-30, 0, 0], "rotation": [1, 0, 0, 0], "uid": 2},
        ],
    }
    p = tmp_path / "syn.pack.json"
    p.write_text(json.dumps(pack))
    return str(p)


def test_build_report_smoke(tmp_path):
    pack = _stub_pack(tmp_path)
    templates = {
        "groel": CGTemplate("groel", "groel.gro", "groel.itp", np.zeros((5, 3)), 5),
        "EG10367-MONOMER": CGTemplate(
            "EG10367-MONOMER", "e.gro", "e.itp", np.zeros((5, 3)), 5
        ),
    }
    summary = assemble(
        pack,
        (-100.0, -100.0, -100.0),
        (100.0, 100.0, 100.0),
        ["groel", "EG10367-MONOMER"],
        templates,
        str(tmp_path / "out"),
        use_bentopy=False,
    )

    out_html = str(tmp_path / "report.html")
    path = build_parsimony_report(
        summary,
        relaxed_gro=summary["gro"],
        out_html=out_html,
        final_energy=-123.4,
        md_ran=True,
    )

    assert path == out_html
    assert os.path.exists(out_html)
    text = open(out_html).read()
    assert len(text) > 500
    # core report content
    assert "groel" in text
    assert "EG10367-MONOMER" in text
    assert "-123.4" in text  # final energy surfaced when MD ran
    assert "n_beads" in text or "beads" in text.lower()
    # a 3D viewer payload is present
    assert "scatter3d" in text.lower() or "3dmol" in text.lower()
