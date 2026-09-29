"""The patch tutorial solves the board it draws, in-plane (#1375).

``examples/tutorials/patch_antenna_demo.py`` draws the openEMS 32 x 40 mm
patch on a 60 x 60 mm ground.  A PEC sheet's edge is solved
``rfx.mesh_edges.EDGE_OFFSET`` (0.35) of a cell beyond its last node, so on
the uniform 2 mm lattice the tutorial used to build, with every edge
mid-cell, the patch was solved 31.4 x 39.4 mm and the feed sat 1 mm off in x
and y.  The tutorial now grades x and y with ``edge_aware_profiles`` and
asserts the result at build time.  These tests pin both halves: the
committed board is solved at its drawn size with the feed at its drawn point
(read here from the realized PEC edges, not from the tutorial's own check),
and the uniform lattice is refused.  Build only, no time step.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = REPO_ROOT / "examples" / "tutorials" / "patch_antenna_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("_patch_tutorial_1375", TUTORIAL)
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

    def solved(n, a0, a1):
        return (n[a0] - EDGE_OFFSET * (n[a0] - n[a0 - 1]),
                n[a1] + EDGE_OFFSET * (n[a1 + 1] - n[a1]))

    centre = (mod.GP_SIZE + 2 * mod.MARGIN_XY) / 2
    drawn = {"ground": (mod.GP_SIZE, mod.GP_SIZE),
             "patch": (mod.PATCH_W, mod.PATCH_L)}
    planes = sorted(int(sp.plane) for sp in rz.sheets)
    assert len(planes) == 2, f"expected two sheets, got planes {planes}"
    for name, k in zip(("ground", "patch"), planes):
        i = np.flatnonzero(ex[:, :, k].any(axis=1))
        j = np.flatnonzero(ey[:, :, k].any(axis=0))
        got = solved(x, i[0], i[-1] + 1) + solved(y, j[0], j[-1] + 1)
        w, l = drawn[name]
        want = (centre - w / 2, centre + w / 2, centre - l / 2, centre + l / 2)
        assert np.allclose(got, want, rtol=0.0, atol=1e-9), (
            f"{name}: solved x {got[0] * 1e3:.3f}-{got[1] * 1e3:.3f} mm, "
            f"y {got[2] * 1e3:.3f}-{got[3] * 1e3:.3f} mm; drawn x "
            f"{want[0] * 1e3:.3f}-{want[1] * 1e3:.3f}, y "
            f"{want[2] * 1e3:.3f}-{want[3] * 1e3:.3f} mm")

    (src,) = [p for p in sim._ports if float(p.impedance) == 0.0]
    i_f, j_f, _k = position_to_index(rz.grid, tuple(src.position))
    feed = (centre + mod.FEED_OFFSET_X, centre)
    assert np.allclose((x[i_f], y[j_f]), feed, rtol=0.0, atol=1e-9), (
        f"feed Ez edge at ({x[i_f] * 1e3:.3f}, {y[j_f] * 1e3:.3f}) mm, "
        f"drawn at ({feed[0] * 1e3:.3f}, {feed[1] * 1e3:.3f}) mm")


def test_the_uniform_lattice_is_refused_at_build():
    mod = _load()
    mod.edge_aware_profiles = lambda *a, **k: {}   # the pre-#1375 mesh
    with pytest.raises(RuntimeError, match="not solved where it is drawn") as exc:
        _build(mod)
    msg = str(exc.value)
    assert ("patch solved: drawn (99.00, 131.00, 95.00, 135.00), "
            "realized (99.30, 130.70, 95.30, 134.70) mm") in msg, msg
    assert ("feed Ez edge: drawn (109.00, 115.00), "
            "realized (108.00, 114.00) mm") in msg, msg
