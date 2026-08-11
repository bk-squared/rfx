"""Gate tests for `rfx.observables` (issue #579).

`rfx.observables` is a thin accessor/objective layer over the DFT-plane
accumulator mechanism that already existed (``Simulation.add_dft_plane_probe``
+ ``result.dft_planes``) and was already end-to-end JAX-differentiable; the
new surface here is `dft_field` / `field_energy` / `field_softmax`
(rfx/observables.py) plus two fail-loud distributed fences.

Coverage:
    (1) FD-vs-AD gates for BOTH acceptance-criteria legs (material eps_r
        scalar; one topology-density pixel via
        ``rfx.topology.density_to_material_fields``), for both built-in
        functionals.
    (2) A run()-vs-forward() physics witness: the two independent code
        paths that populate ``result.dft_planes`` (rfx/runners/uniform.py
        vs rfx/api/_execute.py) must agree on the SAME config -- this
        catches shared-forward-code defects the FD-vs-AD gate (which only
        exercises forward()) is blind to.
    (3) `dft_field` error-path and dict/stack behavior (fast, no FDTD).
    (4) Regression tests for the two distributed fail-loud fences.

FALSIFIER RITUAL: before trusting the FD-vs-AD gates below, a throwaway
`jax.lax.stop_gradient` was inserted on the DFT-plane accumulator inside a
scratch copy of the material-leg objective (field_energy, alpha0=2.0) and
the gate was re-run: AD grad went to EXACTLY 0.0 while the FD reference
stayed finite at -7.7177063137973e-09 -- i.e. the gate DOES go red when
the tape is actually severed (see the #579 PR body for the full captured
output). The throwaway change was reverted immediately after and verified
clean (grep + git status) before this file's tests were trusted green.

BETA CALIBRATION NOTE (the softmax gates below): an initial beta=1.0 on
these two tests passed by COINCIDENCE, not physics -- see
`_SOFTMAX_BETA`'s comment for the measured counter-example (h=0.01 read
18.9% relative error with the AD gradient unchanged) and the fix.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.observables import dft_field, field_energy, field_softmax
from rfx.topology import density_to_material_fields

try:
    from jax import enable_x64
except ImportError:  # older JAX (< ~0.4.31) -- see tests/_x64_compat.py
    from tests._x64_compat import enable_x64


# ---------------------------------------------------------------------------
# Shared small CPU-fast sim: vacuum domain, one soft source, one DFT plane.
# ---------------------------------------------------------------------------

_DX = 1.0e-3
_NX, _NY, _NZ = 30, 20, 20
_PLANE = "probe_plane"
_FREQS = jnp.asarray([5.0e9])
# Physical positions (engineering principle #1: physical absolute
# coordinates, not raw cell-relative indices) — converted to grid indices
# via Grid.position_to_index() once the actual (CPML-padded) grid.shape is
# known, since it is larger than (_NX, _NY, _NZ).
_SLAB_LO_M = (12 * _DX, 0.0, 0.0)  # material-leg design slab
_SLAB_HI_M = (18 * _DX, _NY * _DX, _NZ * _DX)
_PIXEL_POS_M = (14 * _DX, 10 * _DX, 10 * _DX)  # geometry-leg single design pixel
_N_STEPS = 160
_FD_H = 0.02
# field_softmax is now auto-scaled (rfx.observables.field_softmax computes
# beta_eff = beta / stop_gradient(max(vals)) internally), so beta below is
# DIMENSIONLESS -- it no longer needs hand-matching to this sim's raw
# max|field|^2 (~5e-10 to 5.4e-10 on both legs). The original footgun this
# fixture caught (#619): at the OLD unscaled beta=1.0, with
# count = n_freqs*n1*n2 = 1*31*31 = 961 elements on the single registered
# plane, log(count)/beta ~ 6.87 totally dominated the objective (it sat at
# ~100.000002% of log(961)) -- both AD and FD differentiated rounding
# noise in that constant rather than physics, and the two coincidentally
# agreed at h=0.02 (< 5% rel_err) while DISAGREEING at h=0.01 (18.9%
# rel_err, same AD value) -- proof the original green was luck, not a
# real measurement. Post-auto-scale, beta_eff*max(vals) == beta ALWAYS by
# construction, so this failure mode cannot recur at any beta -- but
# stop_gradient on the normalizer means FD (which re-evaluates the WHOLE
# renormalization at each perturbed point) and AD (which correctly omits
# that renormalization's own derivative, by design) are not evaluating
# quite the same thing, and disagree by a few percent at low-to-moderate
# beta_eff. Re-measured sweep (geometry leg, same sim, three h values):
# beta=10 -> up to 34.0% rel_err, beta=30 -> up to 7.8%, beta=50 -> up to
# 3.5%, beta=100 -> up to 1.2%, beta=200 -> up to 0.7% -- monotonically
# tightening as beta grows (the stop_gradient-omitted term's relative
# contribution shrinks as the softmax concentrates on fewer terms near the
# true max), comfortably below the existing 5% gate at beta=200 with
# similar headroom to the pre-auto-scale calibration. beta=200 is used
# below, re-verified across THREE h values (not one) so a future
# coincidental-rounding regression cannot slip back in unnoticed.
_SOFTMAX_BETA = 200.0
_SOFTMAX_FD_H_VALUES = (0.01, 0.02, 0.04)
# A single-Yee-cell material perturbation (the geometry/topology leg) has a
# proportionally tiny effect on a bulk plane-energy sum -- real,
# FD-matching signal, not noise, but small. The tightest margin measured
# across all four FD-vs-AD gates is field_softmax's geometry leg at
# beta=_SOFTMAX_BETA (~4.9e-12) -- about 2.7 orders of magnitude above
# this floor (field_energy's geometry leg, ~2.1e-10, and both material-leg
# gradients sit further above it still). The floor only needs to catch an
# actually-severed tape (which reads exactly/near 0.0), so it sits
# comfortably below that measured minimum.
_GRAD_FLOOR = 1e-14


def _build_sim() -> Simulation:
    sim = Simulation(
        freq_max=8e9, domain=(_NX * _DX, _NY * _DX, _NZ * _DX), dx=_DX,
        boundary="cpml", cpml_layers=5,
    )
    sim.add_source(
        position=(8 * _DX, 10 * _DX, 10 * _DX), component="ez",
        amplitude_kind="current",
    )
    sim.add_dft_plane_probe(
        axis="x", coordinate=20 * _DX, component="ez",
        freqs=_FREQS, name=_PLANE,
    )
    return sim


def _grid_and_eps_base(sim: Simulation):
    """The actual grid is larger than (_NX, _NY, _NZ): CPML padding is
    added OUTSIDE the requested domain, not carved out of it. Always
    derive shapes/indices from the built Grid, never from the domain
    cell counts directly."""
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    return grid, eps_base


def _material_objective(sim, functional):
    """Material leg: alpha scales eps_r of a dielectric slab between the
    source and the DFT plane. functional is field_energy(_PLANE) or
    field_softmax(_PLANE)."""
    grid, eps_base = _grid_and_eps_base(sim)
    lo = grid.position_to_index(_SLAB_LO_M)
    hi = grid.position_to_index(_SLAB_HI_M)
    slab = (slice(lo[0], hi[0] + 1), slice(None), slice(None))

    def objective(alpha):
        eps = eps_base.at[slab].set(alpha)
        result = sim.forward(
            eps_override=eps, n_steps=_N_STEPS, checkpoint=False,
            skip_preflight=True,
        )
        return functional(result)

    return objective


def _geometry_objective(sim, functional, *, eps_bg=1.0, eps_fg=4.0):
    """Geometry leg: ONE topology-density pixel through
    density_to_material_fields -> eps_override (topology.py:209-221)."""
    grid, eps_base = _grid_and_eps_base(sim)
    i, j, k = grid.position_to_index(_PIXEL_POS_M)

    def objective(rho_scalar):
        rho = jnp.reshape(rho_scalar, (1, 1, 1))
        fields = density_to_material_fields(rho, eps_bg, eps_fg)
        eps = eps_base.at[i, j, k].set(fields.eps[0, 0, 0])
        result = sim.forward(
            eps_override=eps, n_steps=_N_STEPS, checkpoint=False,
            skip_preflight=True,
        )
        return functional(result)

    return objective


def _central_fd_f64(objective, x0: float, h: float):
    """Central finite difference with the comparison arithmetic in float64
    (#527 discipline: the FDTD fields stay float32; only the loss/FD
    comparison gains precision, scoped via enable_x64()).

    The design-variable input is deliberately kept float32 (its natural
    dtype, matching eps_base — test_coax_two_port_ad.py established this:
    forcing the input to float64 is unnecessary AND scatters a float64
    value into a float32 eps_override array). Under enable_x64() the
    accumulator/comparison math promotes to float64 on its own (the DFT
    accumulator dtype follows jax.config.x64_enabled, independent of the
    FDTD field precision), which is what the dtype assertion below checks.
    """
    with enable_x64():
        fp = objective(jnp.asarray(x0 + h, dtype=jnp.float32))
        fm = objective(jnp.asarray(x0 - h, dtype=jnp.float32))
        assert fp.dtype == jnp.float64 and fm.dtype == jnp.float64, (
            f"FD reference did not run in float64 (got {fp.dtype}); JAX "
            "truncates a float64 request to float32 when x64 is off."
        )
        return (float(fp) - float(fm)) / (2.0 * h)


def _assert_ad_matches_fd(g_ad: float, g_fd: float, *, label: str, tol: float = 0.05):
    assert np.isfinite(g_ad), f"{label}: AD gradient not finite: {g_ad}"
    assert np.isfinite(g_fd), f"{label}: FD reference not finite: {g_fd}"
    assert abs(g_ad) > _GRAD_FLOOR, (
        f"{label}: AD gradient ~0 ({g_ad:.3e}, floor {_GRAD_FLOOR:.0e}) -- "
        "tape may be severed (see the module docstring falsifier ritual)."
    )
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < tol, (
        f"{label}: AD grad {g_ad:.6e} vs FD {g_fd:.6e}, rel_err "
        f"{rel * 100:.2f}% >= {tol * 100:.0f}%"
    )


# ---------------------------------------------------------------------------
# (1) FD-vs-AD gates: both AC legs x both built-in functionals.
# ---------------------------------------------------------------------------

def test_field_energy_material_leg_ad_matches_fd():
    sim = _build_sim()
    functional = field_energy(_PLANE)
    obj = _material_objective(sim, functional)

    alpha0 = 2.0
    val = float(obj(jnp.asarray(alpha0, dtype=jnp.float32)))
    assert val > 0.0, f"field_energy did not couple to the source (val={val:.3e})"

    g_ad = float(jax.grad(obj)(jnp.asarray(alpha0, dtype=jnp.float32)))
    g_fd = _central_fd_f64(obj, alpha0, _FD_H)
    _assert_ad_matches_fd(g_ad, g_fd, label="field_energy material leg")


def test_field_energy_geometry_leg_ad_matches_fd():
    sim = _build_sim()
    functional = field_energy(_PLANE)
    obj = _geometry_objective(sim, functional)

    rho0 = 0.5
    val = float(obj(jnp.asarray(rho0, dtype=jnp.float32)))
    assert val > 0.0, f"field_energy did not couple to the source (val={val:.3e})"

    g_ad = float(jax.grad(obj)(jnp.asarray(rho0, dtype=jnp.float32)))
    g_fd = _central_fd_f64(obj, rho0, _FD_H)
    _assert_ad_matches_fd(g_ad, g_fd, label="field_energy geometry (1-pixel) leg")


def test_field_softmax_material_leg_ad_matches_fd():
    sim = _build_sim()
    functional = field_softmax(_PLANE, beta=_SOFTMAX_BETA)
    obj = _material_objective(sim, functional)

    alpha0 = 2.0
    val = float(obj(jnp.asarray(alpha0, dtype=jnp.float32)))
    assert np.isfinite(val)

    g_ad = float(jax.grad(obj)(jnp.asarray(alpha0, dtype=jnp.float32)))
    # h-robustness, not a single h: see _SOFTMAX_BETA's comment -- a
    # coincidental rounding-lattice match does NOT hold consistently
    # across multiple step sizes, which is exactly what caught the
    # beta=1.0 false green (18.9% at h=0.01 vs <5% at h=0.02, same g_ad).
    for h in _SOFTMAX_FD_H_VALUES:
        g_fd = _central_fd_f64(obj, alpha0, h)
        _assert_ad_matches_fd(g_ad, g_fd, label=f"field_softmax material leg (h={h})")


def test_field_softmax_geometry_leg_ad_matches_fd():
    sim = _build_sim()
    functional = field_softmax(_PLANE, beta=_SOFTMAX_BETA)
    obj = _geometry_objective(sim, functional)

    rho0 = 0.5
    val = float(obj(jnp.asarray(rho0, dtype=jnp.float32)))
    assert np.isfinite(val)

    g_ad = float(jax.grad(obj)(jnp.asarray(rho0, dtype=jnp.float32)))
    for h in _SOFTMAX_FD_H_VALUES:
        g_fd = _central_fd_f64(obj, rho0, h)
        _assert_ad_matches_fd(
            g_ad, g_fd, label=f"field_softmax geometry (1-pixel) leg (h={h})"
        )


# ---------------------------------------------------------------------------
# (1b) Issue #619 regression: the DEFAULT beta must be safe at ANY field
# magnitude, not merely at values a caller happened to hand-tune against.
# Synthetic fixtures (no FDTD -- fast), same _fake_result pattern as (3).
# ---------------------------------------------------------------------------

def test_field_softmax_default_beta_is_scale_invariant():
    """The pre-fix implementation computed logsumexp(beta*vals)/beta
    directly against raw vals, so at a physically tiny field magnitude and
    the default beta=1.0, the design-independent constant log(count)/beta
    dominated: the output collapsed toward THAT CONSTANT regardless of
    the field's actual scale (measured, #619 -- see _SOFTMAX_BETA's
    comment above). field_softmax now auto-scales internally
    (beta_eff = beta / stop_gradient(max(vals))), which makes it exactly
    homogeneous of degree 1 in vals: rescaling every field value by a
    positive constant c rescales the output by the SAME c, at ANY beta
    including the default -- a clean, exact invariant the pre-fix
    implementation violated at small c. Verified at field-magnitude
    scales spanning rfx's own documented realistic DFT-accumulator range
    (vals ~1e-22 to ~1e-8, per field_softmax's and this file's docstrings)
    plus an O(1) and a large point for symmetry.
    """
    rng = np.random.default_rng(0)
    base_field = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(3, 3, 2)), dtype=jnp.float32
    ).astype(jnp.complex64)
    softmax_default = field_softmax("p")  # beta left at its default (1.0)

    def value_at_vals_scale(vals_scale: float) -> float:
        # vals = |field|**2, so scaling vals by `vals_scale` means scaling
        # the field itself by sqrt(vals_scale).
        field = base_field * jnp.asarray(vals_scale, dtype=jnp.float32) ** 0.5
        result = _fake_result(p=field)
        return float(softmax_default(result))

    v_ref = value_at_vals_scale(1.0)
    assert v_ref > 0.0

    for vals_scale in (1e-22, 1e-10, 1e2, 1e8):
        v = value_at_vals_scale(vals_scale)
        expected = v_ref * vals_scale
        assert np.isfinite(v) and v > 0.0, (
            f"field_softmax(default beta) not finite/positive at vals_scale="
            f"{vals_scale:.0e}: {v}"
        )
        rel = abs(v - expected) / expected
        assert rel < 1e-3, (
            f"field_softmax(default beta) is not scale-invariant at "
            f"vals_scale={vals_scale:.0e}: got {v:.6e}, expected "
            f"{expected:.6e} (ratio={v / expected:.6f}). Pre-fix, this "
            "ratio collapsed toward a magnitude-independent constant "
            "instead of tracking vals_scale -- the exact #619 footgun."
        )


def test_field_softmax_default_beta_gradient_not_rounding_noise():
    """Direct FD-vs-AD check at the DEFAULT beta and a physically tiny
    field magnitude (vals ~1e-20, inside rfx's own documented realistic
    DFT-accumulator range) -- the exact operating point at which the
    pre-fix implementation's gradient measured rounding noise in
    log(count)/beta rather than physics (#619)."""
    rng = np.random.default_rng(1)
    base = jnp.asarray(rng.uniform(0.5, 1.5, size=(3, 3)), dtype=jnp.float32)
    field_scale = 1e-10  # |field|**2 ~ 1e-20, matching the realistic range
    softmax_default = field_softmax("p")

    def obj(alpha):
        arr = base.at[1, 1].set(alpha) * field_scale
        result = _fake_result(p=arr.astype(jnp.complex64))
        return softmax_default(result)

    alpha0 = 1.0
    g_ad = float(jax.grad(obj)(jnp.asarray(alpha0, dtype=jnp.float32)))
    assert np.isfinite(g_ad) and g_ad != 0.0, (
        f"gradient collapsed at tiny field magnitude: {g_ad} -- the exact "
        "#619 rounding-noise failure mode."
    )
    for h in (0.01, 0.02, 0.04):
        with enable_x64():
            fp = obj(jnp.asarray(alpha0 + h, dtype=jnp.float32))
            fm = obj(jnp.asarray(alpha0 - h, dtype=jnp.float32))
            g_fd = float((fp - fm) / (2.0 * h))
        rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-300)
        assert rel < 0.05, (
            f"AD vs FD mismatch at tiny field magnitude (h={h}): "
            f"g_ad={g_ad:.6e} g_fd={g_fd:.6e} rel_err={rel * 100:.2f}%"
        )


# ---------------------------------------------------------------------------
# (2) Physics witness: run() and forward() must agree on the accumulator.
# ---------------------------------------------------------------------------

def test_run_forward_dft_plane_parity():
    """The two independent code paths that populate result.dft_planes
    (rfx/runners/uniform.py for run(), rfx/api/_execute.py for forward())
    must produce the SAME accumulator on an identical vacuum config. This
    is the guard the FD-vs-AD gates above are blind to (they only ever
    exercise forward())."""
    sim_run = _build_sim()
    sim_fwd = _build_sim()

    run_result = sim_run.run(n_steps=_N_STEPS, skip_preflight=True)
    fwd_result = sim_fwd.forward(n_steps=_N_STEPS, skip_preflight=True)

    assert run_result.dft_planes and _PLANE in run_result.dft_planes
    assert fwd_result.dft_planes and _PLANE in fwd_result.dft_planes

    run_acc = np.asarray(run_result.dft_planes[_PLANE].accumulator)
    fwd_acc = np.asarray(fwd_result.dft_planes[_PLANE].accumulator)

    assert run_acc.shape == fwd_acc.shape
    peak = np.max(np.abs(run_acc)) + 1e-30
    assert peak > 1e-20, "accumulator is empty -- source did not couple"
    rel_err = np.max(np.abs(run_acc - fwd_acc)) / peak
    assert rel_err < 1e-5, (
        f"run() vs forward() dft_planes accumulator mismatch: rel_err={rel_err:.2e} "
        f"(run peak={np.max(np.abs(run_acc)):.4e}, forward peak={np.max(np.abs(fwd_acc)):.4e})"
    )


# ---------------------------------------------------------------------------
# (3) dft_field error-path and dict/stack behavior — no FDTD, fake results.
# ---------------------------------------------------------------------------

def _fake_result(**planes):
    return types.SimpleNamespace(
        dft_planes={name: types.SimpleNamespace(accumulator=arr) for name, arr in planes.items()}
    )


def test_dft_field_single_name_returns_array_not_dict():
    arr = jnp.ones((2, 3, 4), dtype=jnp.complex64)
    result = _fake_result(p1=arr)
    out = dft_field("p1")(result)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(np.asarray(out), np.asarray(arr))


def test_dft_field_list_stacks_when_shapes_match():
    a = jnp.ones((2, 3, 4), dtype=jnp.complex64)
    b = jnp.full((2, 3, 4), 2.0, dtype=jnp.complex64)
    result = _fake_result(a=a, b=b)
    out = dft_field(["a", "b"])(result)
    assert out.shape == (2, 2, 3, 4)
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(a))
    np.testing.assert_allclose(np.asarray(out[1]), np.asarray(b))


def test_dft_field_mismatched_shapes_raise_with_stack_false_hint():
    a = jnp.ones((2, 3, 4), dtype=jnp.complex64)
    b = jnp.ones((2, 3, 5), dtype=jnp.complex64)
    result = _fake_result(a=a, b=b)
    with pytest.raises(ValueError, match="stack=False"):
        dft_field(["a", "b"])(result)


def test_dft_field_stack_false_returns_dict():
    a = jnp.ones((2, 3, 4), dtype=jnp.complex64)
    b = jnp.ones((2, 3, 5), dtype=jnp.complex64)
    result = _fake_result(a=a, b=b)
    out = dft_field(["a", "b"], stack=False)(result)
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (2, 3, 4)
    assert out["b"].shape == (2, 3, 5)


def test_dft_field_missing_result_planes_raises():
    result = types.SimpleNamespace(dft_planes=None)
    with pytest.raises(ValueError, match="add_dft_plane_probe"):
        dft_field("p1")(result)


def test_dft_field_missing_name_raises():
    result = _fake_result(p1=jnp.ones((1, 2, 2), dtype=jnp.complex64))
    with pytest.raises(ValueError, match="not found in result.dft_planes"):
        dft_field("p2")(result)


def test_field_energy_missing_plane_raises_naming_the_call():
    result = types.SimpleNamespace(dft_planes=None)
    with pytest.raises(ValueError, match="field_energy"):
        field_energy("p1")(result)


def test_field_softmax_rejects_nonpositive_beta():
    with pytest.raises(ValueError, match="beta"):
        field_softmax("p1", beta=0.0)


def test_dft_field_and_field_energy_accept_raw_array_planes():
    """Cross-PR robustness (#578): a dft_planes dict may carry BARE arrays
    instead of DFTPlaneProbe-like objects -- e.g. a vmap-batched sweep
    result. _select_planes must duck-type (`getattr(v, "accumulator", v)`)
    rather than assume every value has an `.accumulator` attribute, or this
    combination crashes AttributeError."""
    raw = jnp.full((2, 3, 4), 1.5, dtype=jnp.complex64)  # NOT wrapped in a probe-like object
    result = types.SimpleNamespace(dft_planes={"p1": raw})

    out = dft_field("p1")(result)
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(np.asarray(out), np.asarray(raw))

    energy = float(field_energy("p1")(result))
    assert np.isclose(energy, float(jnp.sum(jnp.abs(raw) ** 2)))


# ---------------------------------------------------------------------------
# (4) Fail-loud distributed fences (issue #579 item 2) — regression pins.
# ---------------------------------------------------------------------------

def _nu_sim_with_dft_plane() -> Simulation:
    nx, ny, nz = 16, 12, 12
    dx = 1e-3
    sim = Simulation(
        freq_max=8e9, domain=(nx * dx, ny * dx, nz * dx), dx=dx,
        boundary="cpml", cpml_layers=4,
    )
    sim._dx_profile = np.full(nx, dx)
    sim._dy_profile = np.full(ny, dx)
    sim._dz_profile = np.full(nz, dx)
    sim.add_source(
        position=(4e-3, 6e-3, 6e-3), component="ez", amplitude_kind="current",
    )
    sim.add_dft_plane_probe(
        axis="x", coordinate=10e-3, component="ez", freqs=_FREQS,
    )
    return sim


def test_forward_distributed_nu_dft_planes_fence_raises():
    sim = _nu_sim_with_dft_plane()
    with pytest.raises(NotImplementedError, match="add_dft_plane_probe"):
        sim.forward(distributed=True, n_steps=2, skip_preflight=True)


def test_run_distributed_dft_planes_fence_raises():
    sim = _build_sim()
    devices = [jax.devices()[0], jax.devices()[0]]
    with pytest.raises(NotImplementedError, match="add_dft_plane_probe"):
        sim.run(devices=devices, n_steps=2, skip_preflight=True)
