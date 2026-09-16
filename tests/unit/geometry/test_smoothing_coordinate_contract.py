"""Coordinate ownership checks for the opt-in Kottke path (#833).

Three things are pinned here:

1. the uniform smoother samples the shared host-float64 node spine
   (``_yee_coords`` == ``coords_from_uniform_grid`` after the cast);
2. the NU smoother's cell centres are ``node + d/2`` with ``d`` from the
   float64 cell-size spine, formed in host float64 and cast once;
3. a measured DRIFT PIN for the uniform smoothed eps across the x64 flags.

(3) is NOT the #833 acceptance. The issue's acceptance line is "smoothed-eps
x64-invariance on a boundary voxel"; ab280a38 (PR #998) did not reach it and
cannot by coordinate routing alone (the coordinates are the correctly rounded
float32 of the spine at x64=0, and roughly half of the remaining flag
dependence is float32 SDF/Kottke arithmetic). The pin records the measured
state after ab280a38 so it cannot drift silently while the PI decides between
option (a) — host-float64 fill fractions and normals for concrete shapes,
which would take the pin to zero and let it be tightened — and option (b) —
ratifying this envelope as the weaker acceptance. Either outcome is recorded
on #833; this file only measures.

Both flags are exercised from any pytest session by running the measurement
in a subprocess with ``JAX_ENABLE_X64`` set (x64 scoped per process, never
flipped at module level).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.geometry.smoothing import _yee_coords
from rfx.grid import Grid
from tests._gate_policy import gate_from_envelope

REPO_ROOT = Path(__file__).resolve().parents[3]

# Auditor's boundary-voxel fixture (#833, 2026-09-16): Grid + one Box whose
# faces sit off-node so every interface voxel carries a partial fill.
UNIFORM_FIXTURE = dict(freq_max=1.0e9, domain=(7.0e-3, 5.0e-3, 3.0e-3),
                       dx=2.54e-4, cpml_layers=2)
BOX_LO = (1.3e-3, 1.1e-3, 0.7e-3)
BOX_HI = (4.7e-3, 3.9e-3, 2.3e-3)
BOX_EPS = 4.0

# Graded fixture the #833 audit measured the NU centres on: WR-90-class
# dz profile 20xD | 10xD/2 | 20xD, D = 0.3048 mm, all-CPML, cpml_layers=8.
NU_D = 0.3048e-3
NU_DZ = np.concatenate([np.full(20, NU_D), np.full(10, NU_D / 2), np.full(20, NU_D)])
NU_GRADING_ADVISORY = "dz_profile has max adjacent cell ratio 2.000"

# Measured cross-flag envelope of the uniform smoothed eps on the fixture
# above, main 6d721a56 (after ab280a38), CPU host, max rel |eps(x64=0) -
# eps(x64=1)| over all voxels: ex 1.131e-06 (15 f32 ulps), ey 6.467e-07
# (5 ulps), ez 1.025e-06 (11 ulps). Pre-fix 3f3ed7e0 was 1.019e-06 /
# 5.661e-06 / 5.661e-06. Gate = envelope x ENVELOPE_GATE_MULTIPLIER, rounded
# up to 1e-7.
UNIFORM_DRIFT_ENVELOPE_REL = {"ex": 1.131e-06, "ey": 6.467e-07, "ez": 1.025e-06}
UNIFORM_DRIFT_GATE_REL = {k: gate_from_envelope(v, quantum=1e7)
                          for k, v in UNIFORM_DRIFT_ENVELOPE_REL.items()}

_WORKER = r"""
import json, sys, warnings
import numpy as np, jax, jax.numpy as jnp
import rfx
from rfx.grid import Grid
from rfx.geometry.csg import Box
from rfx.geometry import smoothing as sm
from rfx.geometry.rasterize_grid import (coords_from_nonuniform_grid,
                                         cell_sizes_from_nonuniform_grid)
from rfx.runners.nonuniform import build_nonuniform_grid
cfg = json.loads(sys.argv[1]); out = sys.argv[2]
res = {"x64": np.array(bool(jax.config.jax_enable_x64)),
       "rfx_file": np.array(rfx.__file__)}
# --- uniform smoothed eps on the boundary-voxel fixture
grid = Grid(**cfg["uniform"])
ex, ey, ez = sm.compute_smoothed_eps(grid, [(Box(tuple(cfg["lo"]), tuple(cfg["hi"])), cfg["eps"])],
                                     background_eps=1.0)
for k, a in (("ex", ex), ("ey", ey), ("ez", ez)):
    res["u_" + k] = np.asarray(a, dtype=np.float64)
res["u_dtype"] = np.array(str(ex.dtype))
# --- NU centres on the graded fixture
D = cfg["D"]; dz = np.asarray(cfg["dz"], dtype=np.float64)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    g = build_nonuniform_grid(12e9, (30 * D, 20 * D, float(dz.sum())), D, 8, dz)
res["nu_warnings"] = np.array([str(ww.message) for ww in w])
res["nu_shape"] = np.array(g.shape)
coords = coords_from_nonuniform_grid(g)
from rfx.geometry.rasterize_grid import centres_from_nonuniform_grid, _uniform_axis_centres
_c = centres_from_nonuniform_grid(g, coords)
# cast exactly as compute_smoothed_eps_nonuniform does (host f64 -> active JAX dtype)
got = tuple(jnp.asarray(a) for a in (_c.x, _c.y, _c.z))
for name, node, d, c in zip("xyz", (coords.x, coords.y, coords.z),
                            cell_sizes_from_nonuniform_grid(g), got):
    # Canonical primal-cell centre: on a uniform-valued axis the #807 closed
    # form (i - pad + 1/2) * dx that the UNIFORM lane samples at (so both
    # lanes centre-sample a uniform axis bit-identically); on a graded axis
    # node + d_exact/2 in host float64.
    _dd = np.asarray(d, np.float64); _nn = np.asarray(node, np.float64)
    if _dd.size and bool(np.all(_dd == _dd[0])):
        _dx = float(_dd[0]); _pad = int(round(-_nn[0] / _dx))
        res["nu_exact_" + name] = _uniform_axis_centres(_nn.size, _pad, _dx)
    else:
        res["nu_exact_" + name] = _nn + _dd / 2.0
    res["nu_got_" + name] = np.asarray(c, dtype=np.float64)
    res["nu_dtype_" + name] = np.array(str(jnp.asarray(c).dtype))
np.savez(out, **res)
"""


def _measure_under_flag(flag: int, tmp_path: Path):
    """Run the worker with ``JAX_ENABLE_X64=flag`` and return its arrays."""
    cfg = dict(uniform=UNIFORM_FIXTURE, lo=BOX_LO, hi=BOX_HI, eps=BOX_EPS,
               D=NU_D, dz=NU_DZ.tolist())
    out = tmp_path / f"x64_{flag}.npz"
    env = dict(os.environ, JAX_ENABLE_X64=str(flag),
               PYTHONPATH=os.pathsep.join(
                   p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p))
    proc = subprocess.run([sys.executable, "-c", _WORKER, json.dumps(cfg), str(out)],
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    data = np.load(out)
    assert bool(data["x64"]) is bool(flag), (flag, data["x64"])
    assert Path(str(data["rfx_file"])).resolve().is_relative_to(REPO_ROOT), (
        f"worker imported rfx from {data['rfx_file']}, not {REPO_ROOT}")
    return data


@pytest.fixture(scope="module")
def both_flags(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("x64_flags")
    return {flag: _measure_under_flag(flag, tmp) for flag in (0, 1)}


def test_smoothing_uses_shared_uniform_node_spine():
    grid = Grid(**UNIFORM_FIXTURE)
    expected = coords_from_uniform_grid(grid)
    actual = _yee_coords(grid)

    # The smoothing path may use the active JAX storage precision, but it
    # must consume the same host-built node values as the binary rasterizer.
    # This certifies "correctly rounded f32 of the spine" at x64=0 — NOT that
    # the sampled positions are flag-independent (they are not; see the
    # drift pin below).
    for got, want in zip(actual, (expected.x, expected.y, expected.z)):
        got_array = np.asarray(got)
        np.testing.assert_array_equal(got_array, np.asarray(want, dtype=got_array.dtype))


@pytest.mark.parametrize("flag", [0, 1])
def test_nu_smoothing_centres_come_from_the_f64_spine(both_flags, flag):
    """``centres_from_nonuniform_grid`` (the shared producer the NU smoother now calls) == the canonical
    primal-cell centre exactly at x64=1 and its correctly rounded float32 at
    x64=0, on the graded fixture (#833). Canonical = the #807 closed form
    (i - pad + 1/2)*dx on the uniform-valued x/y axes (what the uniform lane
    samples at, so the two lanes cannot centre-sample a uniform axis
    differently) and node + d_exact/2 in host float64 on the graded z axis.

    Before this fix the half-cell offset came from the float32 solver store
    (``node + f32(store)/2``): 3.704e-12 m off the spine at x64=1 on all
    three axes, and 5.18e-10 / 4.44e-10 / 1.10e-09 m (x/y/z) at x64=0 —
    not the f32 rounding of the exact centre either.
    """
    data = both_flags[flag]
    assert tuple(int(n) for n in data["nu_shape"]) == (47, 37, 67), data["nu_shape"]
    # Preflight-class output is part of the result: the only warning this
    # fixture may raise is the known grading advisory (ratio 2.0 > 1.4).
    for msg in data["nu_warnings"].tolist():
        assert NU_GRADING_ADVISORY in msg, msg
    for name in "xyz":
        exact = data["nu_exact_" + name]
        got = data["nu_got_" + name]
        if flag:
            assert str(data["nu_dtype_" + name]) == "float64"
            np.testing.assert_array_equal(got, exact, err_msg=f"axis {name}, x64=1")
        else:
            assert str(data["nu_dtype_" + name]) == "float32"
            np.testing.assert_array_equal(
                got, exact.astype(np.float32).astype(np.float64),
                err_msg=f"axis {name}, x64=0: not the f32 rounding of the exact centre")


def test_uniform_smoothed_eps_cross_flag_drift_pin(both_flags):
    """MEASURED DRIFT PIN, not the #833 acceptance.

    Records the cross-flag state of the uniform Kottke path after ab280a38
    (PR #998): on the boundary-voxel fixture the smoothed eps at x64=0 and
    x64=1 still differ on every interface voxel (576/603/616 for ex/ey/ez),
    max rel 1.131e-06 / 6.467e-07 / 1.025e-06 (15 / 5 / 11 f32 ulps). The
    issue's acceptance — x64 invariance — is NOT met; ab280a38 made the
    sample coordinates the correctly rounded float32 of the shared spine,
    which explains about half of the pre-fix flag dependence, and the rest
    is float32 SDF / Kottke arithmetic that coordinate routing cannot touch.

    Pending the PI's decision on #833: option (a) host-float64 fill fractions
    and normals for concrete shapes (would take this drift to <= 0.5 f32 ulp
    and the pin should then be tightened to that), or option (b) ratifying
    this envelope as the weaker acceptance. Gate = measured envelope x
    ENVELOPE_GATE_MULTIPLIER (tests/_gate_policy), rounded up to 1e-7.
    """
    lo, hi = both_flags[0], both_flags[1]
    assert str(lo["u_dtype"]) == "float32" and str(hi["u_dtype"]) == "float64"
    for k in ("ex", "ey", "ez"):
        a, b = lo["u_" + k], hi["u_" + k]
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
        worst = float(rel.max())
        assert worst <= UNIFORM_DRIFT_GATE_REL[k], (
            f"{k}: cross-flag drift {worst:.3e} rel exceeds the pinned gate "
            f"{UNIFORM_DRIFT_GATE_REL[k]:.3e} (measured envelope "
            f"{UNIFORM_DRIFT_ENVELOPE_REL[k]:.3e}); do not loosen — find the cause")
