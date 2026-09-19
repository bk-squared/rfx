"""Shape-parameter gradients through the opt-in Kottke smoother (#1085).

``jax.grad`` of ``compute_smoothed_eps`` w.r.t. a **Box** shift (and, by the
same form, a **Cylinder** radius / height / shift) was **NaN at both
``jax_enable_x64`` flags** while the central finite difference was finite:
the clamped SDFs build their exterior distance as ``sqrt(ox**2 + oy**2 +
oz**2)`` with all three overhangs exactly ``0`` at every INTERIOR sample
point, and ``0 * d/du sqrt(u)|_0 == 0 * inf == nan`` propagates through the
smoother's ``jnp.where`` into every interface voxel. ``_exterior_norm``
(``smoothing.py``) replaces that with the double-``where`` safe norm, which
leaves the FORWARD value bit-for-bit unchanged and makes the derivative 0
at the origin.

What is pinned here:

1. **finiteness** -- the regression lock for #1085: every gradient below is
   finite at both flags. On ``04c98da7`` (parent of the fix) the five Box /
   Cylinder cases are all ``nan`` and only the Sphere control is finite.
2. **central-FD agreement**, per flag, gated through
   ``tests._gate_policy.gate_from_envelope`` from the MEASURED envelopes
   below.
3. **cross-flag agreement** of the two gradients -- the traced path is the
   same ``jax.numpy`` formula at both flags, so at x64=0 it is the float32
   arithmetic of the x64=1 result.

Measured envelopes (2026-09-16, this fixture, worst over the six cases):

  * x64=1, central FD ``h = 1e-8`` : worst rel **1.208e-08** (``box_shift_x``;
    ``sphere_radius`` 1.8e-09, ``cylz_radius`` 2.8e-09, ``cylz_shift_x``
    4.0e-09, ``cylx_radius`` 1.3e-10, ``cylx_height`` 1.6e-09).
    -> ``FD_GATE_REL_X64`` = round-up(env x 1.5, 1e-9) = 1.9e-08.
  * x64=0, central FD ``h = 1e-7`` : worst rel **1.152e-03** (``box_shift_x``;
    ``sphere_radius`` 8.6e-05, ``cylz_shift_x`` 8.9e-05, ``cylz_radius``
    4.2e-05, ``cylx_radius`` 8.0e-06, ``cylx_height`` 5.2e-06).
    This is NOT a gradient error -- it is float32 cancellation in the
    difference quotient itself: the objective is O(1.5e3) with an f32 ulp of
    ~1.2e-4, so ``(f(s+h) - f(s-h)) / 2h`` at ``h = 1e-7`` carries ~6e2
    absolute noise on a derivative of ~7e5. Halving the flag's FD step makes
    it worse (1.105e-02 at ``h = 1e-8``), which is the signature of
    cancellation, not of a wrong derivative. Item 3 below is the check that
    actually constrains the x64=0 gradient.
    -> ``FD_GATE_REL_F32`` = round-up(env x 1.5, 1e-4) = 1.8e-03.
  * cross-flag ``|g(x64=0) - g(x64=1)| / |g(x64=1)|`` : worst **1.151e-05**
    (``cylz_shift_x``; ``box_shift_x`` 4.5e-06, ``cylx_height`` 5.6e-06,
    ``cylz_radius`` 1.5e-06, ``sphere_radius`` 6.3e-07, ``cylx_radius``
    2.6e-08). -> ``CROSS_FLAG_GATE_REL`` = round-up(env x 1.5, 1e-6) = 1.8e-05.

Both flags are exercised by running the measurement in a subprocess with
``JAX_ENABLE_X64`` set -- x64 is process-global and must never be flipped at
module level.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._gate_policy import gate_from_envelope

REPO_ROOT = Path(__file__).resolve().parents[3]

# Same boundary-voxel fixture as test_smoothing_coordinate_contract.py: every
# interface voxel carries a partial fill, and the Box interior supplies the
# all-zero-overhang sample points that produced the NaN.
UNIFORM_FIXTURE = dict(freq_max=1.0e9, domain=(7.0e-3, 5.0e-3, 3.0e-3),
                       dx=2.54e-4, cpml_layers=2)
BOX_LO = (1.3e-3, 1.1e-3, 0.7e-3)
BOX_HI = (4.7e-3, 3.9e-3, 2.3e-3)
BOX_EPS = 4.0
SPHERE = dict(center=(3.5e-3, 2.5e-3, 1.5e-3), radius=1.1e-3)
CYL_Z = dict(center=(3.5e-3, 2.5e-3, 1.5e-3), radius=1.3e-3, height=1.7e-3)
CYL_X = dict(center=(3.5e-3, 2.5e-3, 1.5e-3), radius=1.0e-3, height=2.6e-3)

# Central-FD step per flag (see the module docstring for why they differ).
FD_STEP = {0: 1.0e-7, 1: 1.0e-8}

FD_ENVELOPE_REL_X64 = 1.208e-08
FD_GATE_REL_X64 = gate_from_envelope(FD_ENVELOPE_REL_X64, quantum=1e9)
FD_ENVELOPE_REL_F32 = 1.152e-03
FD_GATE_REL_F32 = gate_from_envelope(FD_ENVELOPE_REL_F32, quantum=1e4)
CROSS_FLAG_ENVELOPE_REL = 1.151e-05
CROSS_FLAG_GATE_REL = gate_from_envelope(CROSS_FLAG_ENVELOPE_REL, quantum=1e6)

# The Sphere control was already finite and FD-validated before the fix
# (issue #1085): it is carried here so a future change that breaks it is
# caught by the same lock.
CASES = ["box_shift_x", "sphere_radius", "cylz_radius", "cylz_shift_x",
         "cylx_radius", "cylx_height"]
NAN_BEFORE_THE_FIX = {"box_shift_x", "cylz_radius", "cylz_shift_x",
                      "cylx_radius", "cylx_height"}

_WORKER = r"""
import json, sys
import numpy as np, jax, jax.numpy as jnp
import rfx
from rfx.grid import Grid
from rfx.geometry.csg import Box, Sphere, Cylinder
from rfx.geometry import smoothing as sm

cfg = json.loads(sys.argv[1]); out = sys.argv[2]
EPS = cfg["eps"]; h = cfg["h"]
LO, HI = tuple(cfg["lo"]), tuple(cfg["hi"])
SPH, CZ, CX = cfg["sphere"], cfg["cyl_z"], cfg["cyl_x"]
grid = Grid(**cfg["uniform"])
res = {"x64": bool(jax.config.jax_enable_x64), "rfx_file": rfx.__file__, "cases": {}}

def objective(build, s0):
    # Fixed interface-voxel mask, chosen at s0 so the objective the gradient
    # and the finite difference see is literally the same function.
    e0 = np.asarray(sm.compute_smoothed_eps(grid, [(build(s0), EPS)],
                                            background_eps=1.0)[0], np.float64)
    mask = jnp.asarray((e0 > 1.0 + 1e-9) & (e0 < EPS - 1e-9))
    def f(s):
        ex = sm.compute_smoothed_eps(grid, [(build(s), EPS)], background_eps=1.0)[0]
        return jnp.sum(jnp.where(mask, ex, 0.0))
    return f, int(np.asarray(mask).sum())

builders = {
    "box_shift_x": (lambda s: Box((LO[0] + s, LO[1], LO[2]),
                                  (HI[0] + s, HI[1], HI[2])), 0.0),
    "sphere_radius": (lambda r: Sphere(tuple(SPH["center"]), r), SPH["radius"]),
    "cylz_radius": (lambda r: Cylinder(tuple(CZ["center"]), r, CZ["height"], "z"),
                    CZ["radius"]),
    "cylz_shift_x": (lambda s: Cylinder((CZ["center"][0] + s, CZ["center"][1],
                                         CZ["center"][2]), CZ["radius"],
                                        CZ["height"], "z"), 0.0),
    "cylx_radius": (lambda r: Cylinder(tuple(CX["center"]), r, CX["height"], "x"),
                    CX["radius"]),
    "cylx_height": (lambda hh: Cylinder(tuple(CX["center"]), CX["radius"], hh, "x"),
                    CX["height"]),
}
for name, (build, s0) in builders.items():
    f, nvox = objective(build, s0)
    try:
        grad = float(jax.grad(f)(s0))
        err = ""
    except Exception as exc:          # pragma: no cover - the pre-fix path raised nothing
        grad, err = float("nan"), repr(exc)
    fd = (float(f(s0 + h)) - float(f(s0 - h))) / (2.0 * h)
    res["cases"][name] = dict(grad=grad, fd=fd, f0=float(f(s0)),
                              nvox=nvox, h=h, err=err)

# Direct probe of the mechanism: the SDF itself at an interior / exterior point.
def sdf_probe(pt):
    def g(s):
        b = Box((LO[0] + s, LO[1], LO[2]), (HI[0] + s, HI[1], HI[2]))
        return sm._sdf_box(jnp.asarray(pt[0]), jnp.asarray(pt[1]),
                           jnp.asarray(pt[2]), b, xp=jnp)
    return [float(g(0.0)), float(jax.grad(g)(0.0))]

res["probe_interior"] = sdf_probe((2.0e-3, 2.0e-3, 1.5e-3))
res["probe_exterior"] = sdf_probe((5.0e-3, 2.0e-3, 1.5e-3))
Path = __import__("pathlib").Path
Path(out).write_text(json.dumps(res))
"""


def _measure_under_flag(flag: int, tmp_path: Path):
    cfg = dict(uniform=UNIFORM_FIXTURE, lo=BOX_LO, hi=BOX_HI, eps=BOX_EPS,
               sphere=SPHERE, cyl_z=CYL_Z, cyl_x=CYL_X, h=FD_STEP[flag])
    out = tmp_path / f"grad_x64_{flag}.json"
    env = dict(os.environ, JAX_ENABLE_X64=str(flag),
               PYTHONPATH=os.pathsep.join(
                   p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p))
    proc = subprocess.run([sys.executable, "-c", _WORKER, json.dumps(cfg), str(out)],
                          env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    data = json.loads(out.read_text())
    assert data["x64"] is bool(flag), (flag, data["x64"])
    assert Path(data["rfx_file"]).resolve().is_relative_to(REPO_ROOT), (
        f"worker imported rfx from {data['rfx_file']}, not {REPO_ROOT}")
    return data


@pytest.fixture(scope="module")
def both_flags(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shape_grad")
    return {flag: _measure_under_flag(flag, tmp) for flag in (0, 1)}


@pytest.mark.parametrize("flag", [0, 1])
@pytest.mark.parametrize("case", CASES)
def test_shape_gradient_through_smoothing_is_finite(both_flags, flag, case):
    """#1085 regression lock: ``jax.grad`` of the smoothed eps w.r.t. a shape
    parameter is finite. Before ``_exterior_norm`` the five Box / Cylinder
    cases returned ``nan`` at both flags (the Sphere control did not -- its
    SDF has no clamped-overhang norm)."""
    rec = both_flags[flag]["cases"][case]
    assert rec["err"] == "", f"{case}: gradient raised {rec['err']}"
    assert rec["nvox"] > 0, f"{case}: fixture has no interface voxel; lock is vacuous"
    assert np.isfinite(rec["grad"]), (
        f"{case} at x64={flag}: grad = {rec['grad']} (central FD {rec['fd']!r} is "
        f"finite) -- the clamped-SDF exterior norm is being differentiated at 0 again")
    assert rec["grad"] != 0.0, f"{case}: gradient is identically zero, lock is vacuous"


@pytest.mark.parametrize("flag", [0, 1])
@pytest.mark.parametrize("case", CASES)
def test_shape_gradient_matches_central_finite_difference(both_flags, flag, case):
    """AD vs central FD, per flag, at the gate derived from the measured
    envelope in the module docstring (1.208e-08 at x64=1 with h=1e-8;
    1.152e-03 at x64=0 with h=1e-7, where the difference quotient itself is
    float32-cancellation-limited)."""
    rec = both_flags[flag]["cases"][case]
    gate = FD_GATE_REL_X64 if flag else FD_GATE_REL_F32
    rel = abs(rec["grad"] - rec["fd"]) / abs(rec["fd"])
    assert rel <= gate, (
        f"{case} at x64={flag}: |AD - FD| / |FD| = {rel:.3e} exceeds the gate "
        f"{gate:.3e} (AD {rec['grad']!r}, central FD {rec['fd']!r}, h = {rec['h']:.0e}, "
        f"{rec['nvox']} interface voxels) -- do not loosen, find the cause")


@pytest.mark.parametrize("case", CASES)
def test_shape_gradient_is_the_same_at_both_flags(both_flags, case):
    """The traced path runs the same ``jax.numpy`` formula at both flags, so
    the x64=0 gradient is the float32 arithmetic of the x64=1 one. This is
    the check that constrains the x64=0 gradient (its own central FD is
    cancellation-limited); measured envelope 1.151e-05."""
    lo = both_flags[0]["cases"][case]["grad"]
    hi = both_flags[1]["cases"][case]["grad"]
    rel = abs(lo - hi) / abs(hi)
    assert rel <= CROSS_FLAG_GATE_REL, (
        f"{case}: grad(x64=0) {lo!r} vs grad(x64=1) {hi!r} -> rel {rel:.3e} exceeds "
        f"the gate {CROSS_FLAG_GATE_REL:.3e}")


@pytest.mark.parametrize("flag", [0, 1])
def test_sdf_box_derivative_is_finite_inside_and_correct_outside(both_flags, flag):
    """The mechanism itself, one sample point at a time. Inside the box every
    clamped overhang is 0, so the exterior norm is differentiated at the
    origin -- ``nan`` before the fix, ``+1`` after (shifting the box by ``+s``
    moves the interior point toward the lo face, and the interior branch
    ``min(max(dx, dy, dz), 0)`` is what carries the derivative there). Outside
    the +x face the gradient is the unchanged ``-1``."""
    sdf_in, grad_in = both_flags[flag]["probe_interior"]
    sdf_out, grad_out = both_flags[flag]["probe_exterior"]
    assert sdf_in < 0.0 and sdf_out > 0.0, (sdf_in, sdf_out)
    assert np.isfinite(grad_in), f"interior d(sdf)/d(shift) = {grad_in} at x64={flag}"
    assert grad_in == pytest.approx(1.0, abs=1e-6), grad_in
    assert grad_out == pytest.approx(-1.0, abs=1e-6), grad_out


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_exterior_norm_forward_is_bit_identical_to_plain_sqrt(backend):
    """The safe norm must not move a single forward bit on either backend --
    that is what lets the #833 x64-invariance lock stay bitwise. Covers the
    all-zero argument (the NaN case) and a spread of positive magnitudes."""
    import jax.numpy as jnp
    from rfx.geometry.smoothing import _exterior_norm

    xp = np if backend == "numpy" else jnp
    rng = np.random.default_rng(1085)
    a = np.concatenate([[0.0, 0.0, 1e-30, 1.0], rng.exponential(1e-3, 400)])
    b = np.concatenate([[0.0, 1e-4, 0.0, 2.0], rng.exponential(1e-3, 400)])
    c = np.concatenate([[0.0, 0.0, 0.0, 3.0], rng.exponential(1e-3, 400)])
    A, B, C = (xp.asarray(v) for v in (a, b, c))
    got = np.asarray(_exterior_norm(xp, A, B, C))
    want = np.asarray(xp.sqrt(A**2 + B**2 + C**2))
    np.testing.assert_array_equal(
        got.view(np.uint32 if got.dtype == np.float32 else np.uint64),
        want.view(np.uint32 if want.dtype == np.float32 else np.uint64),
        err_msg=f"{backend}: _exterior_norm moved a forward bit")
