"""The nonuniform patch demo solves its drawn x/y geometry (#1383).

Read the realized PEC edges independently of the demo's build-time check;
removing edge-aware registration must also make the builder refuse the board.
Build only, no time step.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "examples" / "tutorials" / "nonuniform_patch_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("_nonuniform_patch_1383", DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(mod):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.build_simulation()


def test_committed_board_is_solved_at_its_drawn_size():
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    from rfx.mesh_edges import EDGE_OFFSET
    from rfx.nonuniform import position_to_index
    from tests._realized_geometry import realized

    mod = _load()
    sim = _build(mod)
    rz = realized(sim)
    ex, ey, _ez = (np.asarray(m) for m in rz.edge_masks)
    coords = coords_from_nonuniform_grid(rz.grid)
    x, y = np.asarray(coords.x, float), np.asarray(coords.y, float)

    def solved(nodes, a0, a1):
        return (nodes[a0] - EDGE_OFFSET * (nodes[a0] - nodes[a0 - 1]),
                nodes[a1] + EDGE_OFFSET * (nodes[a1 + 1] - nodes[a1]))

    drawn = {"ground": (mod.gx_lo, mod.gx_hi, mod.gy_lo, mod.gy_hi),
             "patch": (mod.patch_x_lo, mod.patch_x_hi, mod.patch_y_lo, mod.patch_y_hi)}
    planes = sorted(int(sp.plane) for sp in rz.sheets)
    assert len(planes) == 2, f"expected two sheets, got planes {planes}"
    for name, k in zip(("ground", "patch"), planes):
        i = np.flatnonzero(ex[:, :, k].any(axis=1))
        j = np.flatnonzero(ey[:, :, k].any(axis=0))
        got = solved(x, i[0], i[-1] + 1) + solved(y, j[0], j[-1] + 1)
        assert np.allclose(got, drawn[name], rtol=0.0, atol=1e-6 * mod.dx), (
            name, got, drawn[name])

    (src,) = [p for p in sim._ports if float(p.impedance) == 0.0]
    i_f, j_f, _k = position_to_index(rz.grid, tuple(src.position))
    assert np.allclose((x[i_f], y[j_f]), (mod.feed_x, mod.feed_y),
                       rtol=0.0, atol=1e-6 * mod.dx)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        materials, *_rest = sim._assemble_materials_nu(rz.grid, pec_sheets=[])
    cells = np.argwhere(np.asarray(materials.eps_r) > 1.0)
    lo, hi = cells.min(axis=0), cells.max(axis=0)
    substrate = (x[lo[0]], x[hi[0] + 1], y[lo[1]], y[hi[1] + 1])
    assert np.allclose(substrate, drawn["ground"], rtol=0.0, atol=mod.dx)
    assert hi[2] - lo[2] + 1 == mod.n_sub


def test_the_uniform_lattice_is_refused_at_build():
    mod = _load()
    mod.edge_aware_profiles = lambda *a, **k: {}  # the pre-#1383 x/y mesh
    with pytest.raises(RuntimeError, match="not solved where it is drawn") as exc:
        _build(mod)
    msg = str(exc.value)
    assert ("patch solved: drawn (27.25, 56.75, 20.50, 58.50), "
            "realized (27.65, 56.35, 20.65, 58.35) mm") in msg, msg
    assert ("ground solved: drawn (12.00, 72.00, 12.00, 67.00), "
            "realized (11.65, 72.35, 11.65, 67.35) mm") in msg, msg
    assert ("feed Ez edge: drawn (35.25, 39.50), "
            "realized (35.00, 39.00) mm") in msg, msg
