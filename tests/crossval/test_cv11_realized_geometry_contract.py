"""Build-only geometry contracts for cv11's analytic slab comparison.

The cv11 external artifact once hid a one-node dielectric-length mismatch:
``center + L/2`` landed one float64 ulp above the intended hi node after the
#802 coordinate fix.  These tests exercise the builder's actual material
array, so a later coordinate rewrite cannot silently change the board again.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "validation/crossval/11_waveguide_port_wr90.py"


def _load_cv11():
    spec = importlib.util.spec_from_file_location("cv11_geometry_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cv11_slab_bounds_are_exactly_ten_lattice_cells() -> None:
    cv11 = _load_cv11()
    lo, hi = cv11._lattice_slab_bounds(10.0e-3)
    assert lo == 95 * cv11.DX_M
    assert hi == 105 * cv11.DX_M
    assert np.isclose(hi - lo, 10.0e-3, rtol=0.0, atol=1e-15)


def test_cv11_material_realization_matches_declared_slab_length() -> None:
    cv11 = _load_cv11()
    lo_x, hi_x = cv11._lattice_slab_bounds(10.0e-3)
    sim = cv11._build_sim(
        cv11.FREQS_HZ,
        obstacles=[
            ((lo_x, 0.0, 0.0), (hi_x, cv11.DOMAIN_Y, cv11.DOMAIN_Z), 2.0)
        ],
    )
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(
        grid, pec_sheets=[], pec_wires=[]
    )
    eps = np.asarray(materials.eps_r)
    j, k = grid.ny // 2, grid.nz // 2
    occupied = np.flatnonzero(eps[:, j, k] > 1.01)
    assert occupied.size == 10
    assert occupied[-1] - occupied[0] + 1 == 10
