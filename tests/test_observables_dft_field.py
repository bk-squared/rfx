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
scratch copy of the material-leg objective and the gate was re-run -- AD
grad collapsed to ~0.0 while the FD reference stayed finite, i.e. the gate
DOES go red when the tape is actually severed. The red output is saved at
scratchpad/i579_falsifier_red.txt (not committed here -- see the PR body);
the throwaway change was reverted immediately after.
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
# A single-Yee-cell material perturbation (the geometry/topology leg) has a
# proportionally tiny effect on a bulk plane-energy sum (~1e-10 measured) —
# real, FD-matching signal, not noise. The floor only needs to catch an
# actually-severed tape (which reads exactly/near 0.0), so it sits several
# orders below that measured single-pixel gradient, not just below the
# larger material-leg (whole-slab) gradient.
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
    functional = field_softmax(_PLANE, beta=1.0)
    obj = _material_objective(sim, functional)

    alpha0 = 2.0
    val = float(obj(jnp.asarray(alpha0, dtype=jnp.float32)))
    assert np.isfinite(val)

    g_ad = float(jax.grad(obj)(jnp.asarray(alpha0, dtype=jnp.float32)))
    g_fd = _central_fd_f64(obj, alpha0, _FD_H)
    _assert_ad_matches_fd(g_ad, g_fd, label="field_softmax material leg")


def test_field_softmax_geometry_leg_ad_matches_fd():
    sim = _build_sim()
    functional = field_softmax(_PLANE, beta=1.0)
    obj = _geometry_objective(sim, functional)

    rho0 = 0.5
    val = float(obj(jnp.asarray(rho0, dtype=jnp.float32)))
    assert np.isfinite(val)

    g_ad = float(jax.grad(obj)(jnp.asarray(rho0, dtype=jnp.float32)))
    g_fd = _central_fd_f64(obj, rho0, _FD_H)
    _assert_ad_matches_fd(g_ad, g_fd, label="field_softmax geometry (1-pixel) leg")


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
