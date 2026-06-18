"""Tests for the relax + short-MD stages (Tasks 8, 9)."""

import os

import numpy as np
import pytest


# --------------------------------------------------------------------------
# Task 8: WCA relax wrapper
# --------------------------------------------------------------------------

def test_relax_separates_clashing_beads():
    from pbg_martini.parsimony_md import relax_assembly
    coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])  # 0.1 nm clash
    sigma = 0.47
    relaxed = relax_assembly(coords, n_steps=400, sigma=sigma)
    assert not np.any(np.isnan(relaxed))
    d0 = np.linalg.norm(coords[1] - coords[0])
    d1 = np.linalg.norm(relaxed[1] - relaxed[0])
    assert d1 >= d0                # min distance must not decrease
    assert d1 >= sigma * 0.99      # relaxed to >= the WCA sigma


def test_relax_empty():
    from pbg_martini.parsimony_md import relax_assembly
    out = relax_assembly(np.zeros((0, 3)))
    assert out.shape == (0, 3)
