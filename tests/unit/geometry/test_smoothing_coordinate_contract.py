"""Coordinate and precision ownership checks for the opt-in Kottke path (#833).

Three things are pinned here:

1. the uniform smoother samples the shared host-float64 node spine
   (``_yee_coords`` == ``coords_from_uniform_grid`` after the cast);
2. the NU smoother's cell centres are ``node + d/2`` with ``d`` from the
   float64 cell-size spine, formed in host float64 and cast once;
3. the #833 ACCEPTANCE: smoothed eps is x64-invariant on every voxel --
   ``eps(x64=0) == float32(eps(x64=1))`` bitwise -- for
   ``compute_smoothed_eps``, ``compute_inv_eps_tensor_diag`` (dielectric AND
   PEC branches) and ``compute_smoothed_eps_nonuniform``, on a Box, a Sphere
   and a Cylinder so every SDF family in ``smoothing.py`` is covered.

(3) replaced the measured DRIFT PIN PR #1088 left here (max rel 1.131e-06 /
6.467e-07 / 1.025e-06 for ex/ey/ez, 15 / 5 / 11 f32 ulps on the Box fixture,
every interface voxel differing): with concrete shapes the smoothing chain now
runs in host float64 and is cast to the active JAX dtype once (option (a),
PI decision 2026-09-16), so the two flags agree exactly and the lock is
bitwise. A traced shape parameter still takes the ``jax.numpy`` path, which
is exercised separately below.

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

# Sphere / Cylinder siblings on the same grid (all three SDF families), and
# two PEC bodies for the ``compute_inv_eps_tensor_diag`` PEC-limit branch.
SPHERE = dict(center=(3.5e-3, 2.5e-3, 1.5e-3), radius=1.1e-3)
CYLINDER = dict(center=(3.5e-3, 2.5e-3, 1.5e-3), radius=1.3e-3, height=1.7e-3, axis="z")
PEC_SPHERE = dict(center=(1.2e-3, 1.0e-3, 0.8e-3), radius=0.55e-3)
PEC_CYLINDER = dict(center=(5.8e-3, 4.0e-3, 2.1e-3), radius=0.45e-3, height=1.1e-3, axis="y")
# NU shapes on the graded fixture, in units of D; the Box has voxels exactly
# equidistant from its x and z faces (rx == rz), where the pre-fix float32
# path broke the nearest-face tie differently per flag (|d eps| ~ 1).
NU_BOX = dict(lo=(2.3, 1.7, 5.6), hi=(20.4, 13.3, 17.2))
NU_SPHERE = dict(center=(15.0, 10.0, 12.0), radius=5.3)
NU_CYLINDER = dict(center=(15.0, 10.0, 12.0), radius=6.1, height=9.7, axis="z")

_WORKER = r"""
import json, sys, warnings
import numpy as np, jax, jax.numpy as jnp
import rfx
from rfx.grid import Grid
from rfx.geometry.csg import Box, Sphere, Cylinder
from rfx.geometry import smoothing as sm
from rfx.geometry.rasterize_grid import (coords_from_nonuniform_grid,
                                         cell_sizes_from_nonuniform_grid)
from rfx.runners.nonuniform import build_nonuniform_grid
cfg = json.loads(sys.argv[1]); out = sys.argv[2]
res = {"x64": np.array(bool(jax.config.jax_enable_x64)),
       "rfx_file": np.array(rfx.__file__)}
# --- uniform smoothed eps on the boundary-voxel fixture
grid = Grid(**cfg["uniform"])
def put(prefix, arrs):
    for k, a in zip(("ex", "ey", "ez"), arrs):
        res[prefix + "_" + k] = np.asarray(a, dtype=np.float64)
        res[prefix + "_" + k + "_dtype"] = np.array(str(a.dtype))
box = Box(tuple(cfg["lo"]), tuple(cfg["hi"]))
sph = Sphere(tuple(cfg["sphere"]["center"]), cfg["sphere"]["radius"])
cyl = Cylinder(tuple(cfg["cylinder"]["center"]), cfg["cylinder"]["radius"],
               cfg["cylinder"]["height"], cfg["cylinder"]["axis"])
pec_sph = Sphere(tuple(cfg["pec_sphere"]["center"]), cfg["pec_sphere"]["radius"])
pec_cyl = Cylinder(tuple(cfg["pec_cylinder"]["center"]), cfg["pec_cylinder"]["radius"],
                   cfg["pec_cylinder"]["height"], cfg["pec_cylinder"]["axis"])
put("u_box", sm.compute_smoothed_eps(grid, [(box, cfg["eps"])], background_eps=1.0))
put("u_sphere", sm.compute_smoothed_eps(grid, [(sph, cfg["eps"])], background_eps=1.0))
put("u_cylinder", sm.compute_smoothed_eps(grid, [(cyl, cfg["eps"])], background_eps=1.0))
put("inv_box", sm.compute_inv_eps_tensor_diag(grid, dielectric_shapes=[(box, cfg["eps"])]))
put("inv_box_pec", sm.compute_inv_eps_tensor_diag(
    grid, dielectric_shapes=[(box, cfg["eps"])], pec_shapes=[pec_sph, pec_cyl]))
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
# --- NU smoothed eps on the graded fixture, all three SDF families
nb = cfg["nu_box"]; ns = cfg["nu_sphere"]; nc = cfg["nu_cylinder"]
nu_box = Box(tuple(v * D for v in nb["lo"]), tuple(v * D for v in nb["hi"]))
nu_sph = Sphere(tuple(v * D for v in ns["center"]), ns["radius"] * D)
nu_cyl = Cylinder(tuple(v * D for v in nc["center"]), nc["radius"] * D, nc["height"] * D, nc["axis"])
put("nu_box", sm.compute_smoothed_eps_nonuniform(g, [(nu_box, cfg["eps"])], background_eps=1.0))
put("nu_sphere", sm.compute_smoothed_eps_nonuniform(g, [(nu_sph, cfg["eps"])], background_eps=1.0))
put("nu_cylinder", sm.compute_smoothed_eps_nonuniform(g, [(nu_cyl, cfg["eps"])], background_eps=1.0))
np.savez(out, **res)
"""


def _measure_under_flag(flag: int, tmp_path: Path):
    """Run the worker with ``JAX_ENABLE_X64=flag`` and return its arrays."""
    cfg = dict(uniform=UNIFORM_FIXTURE, lo=BOX_LO, hi=BOX_HI, eps=BOX_EPS,
               sphere=SPHERE, cylinder=CYLINDER,
               pec_sphere=PEC_SPHERE, pec_cylinder=PEC_CYLINDER,
               D=NU_D, dz=NU_DZ.tolist(),
               nu_box=NU_BOX, nu_sphere=NU_SPHERE, nu_cylinder=NU_CYLINDER)
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


# (case prefix, dtype at x64=1). ``compute_smoothed_eps*`` return the active
# default float dtype once SDF arithmetic reached the output; the dielectric
# inverse is float32 by contract, and the PEC-limit branch promotes it to the
# active dtype (a pre-existing x64=1 dtype quirk this lock records, not fixes).
INVARIANCE_CASES = [
    ("u_box", "float64"), ("u_sphere", "float64"), ("u_cylinder", "float64"),
    ("inv_box", "float32"), ("inv_box_pec", "float64"),
    ("nu_box", "float64"), ("nu_sphere", "float64"), ("nu_cylinder", "float64"),
]


@pytest.mark.parametrize("case,dtype_x64", INVARIANCE_CASES)
def test_smoothed_eps_is_x64_invariant(both_flags, case, dtype_x64):
    """The #833 ACCEPTANCE: ``eps(x64=0) == float32(eps(x64=1))`` on EVERY
    voxel, bitwise, for the three public smoothing entry points and all three
    SDF families (Box / Sphere / Cylinder), dielectric and PEC branches.

    With concrete inputs the SDF, fill fraction, analytic normal and Kottke
    averaging run in host float64 and are cast to the active JAX dtype once,
    so the flag can only change the final rounding. Before option (a) the
    x64=0 lane ran that chain in float32: on the Box fixture every interface
    voxel differed by up to 15 f32 ulps (PR #1088's drift pin), and on the
    graded NU Box 12 voxels equidistant from two faces flipped their
    nearest-face normal with the flag (|d eps| 0.99).

    ``assert_array_equal`` -- no tolerance: a nonzero difference here means
    some part of the chain runs in the active JAX precision again.
    """
    lo, hi = both_flags[0], both_flags[1]
    for k in ("ex", "ey", "ez"):
        assert str(lo[f"{case}_{k}_dtype"]) == "float32", (case, k)
        assert str(hi[f"{case}_{k}_dtype"]) == dtype_x64, (case, k)
        a, b = lo[f"{case}_{k}"], hi[f"{case}_{k}"]
        # The fixture must actually exercise interface voxels, or the lock is
        # vacuous: a value strictly between background and bulk (or, for the
        # inverse, strictly between their reciprocals / above zero for PEC).
        partial = (b != b.min()) & (b != b.max())
        assert int(partial.sum()) > 0, f"{case}/{k}: no interface voxel in the fixture"
        np.testing.assert_array_equal(
            a, b.astype(np.float32).astype(np.float64),
            err_msg=f"{case}/{k}: eps(x64=0) is not the float32 image of eps(x64=1)")


def test_traced_shape_parameter_takes_the_jax_path():
    """A traced Sphere radius (``jax.jit`` / ``jax.grad`` over geometry) must
    still work: the host-float64 path is for concrete inputs only, and the
    ``jax.numpy`` body it falls back to is the same formula. Agreement is to
    float32 tolerance in-process (this test runs under whatever flag the
    session has); the bitwise lock above is for the concrete path.
    """
    import jax
    from rfx.geometry.csg import Sphere
    from rfx.geometry.smoothing import compute_smoothed_eps

    grid = Grid(**UNIFORM_FIXTURE)
    centre, radius = SPHERE["center"], SPHERE["radius"]
    concrete = compute_smoothed_eps(grid, [(Sphere(centre, radius), BOX_EPS)])
    traced = jax.jit(lambda r: compute_smoothed_eps(grid, [(Sphere(centre, r), BOX_EPS)]))(radius)
    for k, c, t in zip(("ex", "ey", "ez"), concrete, traced):
        c, t = np.asarray(c, np.float64), np.asarray(t, np.float64)
        assert np.isfinite(t).all(), k
        np.testing.assert_allclose(t, c, rtol=2e-5, atol=0.0, err_msg=k)
