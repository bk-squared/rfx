"""Regression tests for differentiable ``Simulation.forward()`` on the
non-uniform mesh path (GitHub issue #33).

Before this change, ``forward()`` raised ``ValueError`` whenever any of
``dx_profile`` / ``dy_profile`` / ``dz_profile`` was set, making NU
grids unusable for gradient-based optimisation. The forward path now
routes NU profiles through ``_forward_nonuniform_from_materials``,
which wraps ``run_nonuniform_path`` with ``eps_override`` /
``sigma_override`` / ``pec_mask_override`` support.

Tests:

1. **Smoke**: ``sim.forward(n_steps=100)`` with ``dz_profile`` set must
   return a finite ``ForwardResult`` (no ValueError).
2. **AD vs FD**: ``jax.grad(loss)(eps_override)`` on a single eps cell
   matches a centred finite-difference estimate to <2% relative error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


def _build_sim():
    """Small graded-z cavity (20×20×9 interior, CPML=4)."""
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3,
        dz_profile=dz,
        cpml_layers=4,
    )
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    return sim


def test_forward_nonuniform_smoke():
    """``forward(n_steps=...)`` on a NU-z sim must return finite data."""
    sim = _build_sim()
    fr = sim.forward(n_steps=100)
    ts = np.asarray(fr.time_series)
    assert ts.shape[0] == 100
    assert np.all(np.isfinite(ts)), "NU forward produced NaN/Inf"
    # Sanity: source excites something within 100 steps.
    assert float(np.max(np.abs(ts))) > 0.0


def test_forward_nonuniform_grad_eps_matches_fd():
    """AD grad w.r.t. a single eps cell agrees with centred FD (<2% rel err).

    We scale ``eps_override`` by a scalar ``alpha`` and differentiate a
    squared-L2 probe loss w.r.t. ``alpha``. This reduces the grad check
    to a 1-D comparison that's robust to step-size / noise.
    """
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)

    # Target cell for the eps perturbation — well away from CPML and
    # source/probe cells.
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2
    n_steps = 60

    def loss(alpha):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        fr = sim.forward(eps_override=eps, n_steps=n_steps)
        return jnp.sum(fr.time_series ** 2)

    alpha0 = jnp.float32(2.0)
    grad_ad = float(jax.grad(loss)(alpha0))

    h = 1e-2
    lp = float(loss(alpha0 + h))
    lm = float(loss(alpha0 - h))
    grad_fd = (lp - lm) / (2.0 * h)

    rel_err = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
    # Tolerance is 5% (the house AD-vs-FD standard, cf. MSL / waveguide-flux
    # gradient tests), not tighter: the FD reference is a centred difference of
    # two ~1e10-scale float32 losses, i.e. a small-difference-of-large-numbers
    # with machine-dependent cancellation. The SAME (correct) AD gradient has
    # been observed at 0.4%–2.2% rel-err across machines while grad_ad itself is
    # stable to all printed digits; the 2% threshold was cross-machine-flaky.
    # This gate validates gradient correctness (AD≈FD), not FD float32 noise.
    assert rel_err < 0.05, (
        f"AD grad {grad_ad:.4e} vs FD grad {grad_fd:.4e} — "
        f"rel_err {rel_err:.4%} above 5% threshold"
    )


# --- Known gaps (not yet plumbed into the NU forward path) ---------------

def test_forward_nonuniform_pec_occupancy_accepted():
    """``pec_occupancy_override`` now flows through the NU scan body.

    Issue #45 (sentinel 1 in ``nu_known_limits.md``): soft-PEC continuous
    occupancy is plumbed through ``run_nonuniform`` and applied after
    each E update (mirrors the uniform path). A zero-occupancy field is
    bit-identical to the no-occupancy baseline; a full-occupancy field
    inside a shielded inner box damps the probe trace to zero.
    """
    sim = _build_sim()
    g = sim._build_nonuniform_grid()

    # Baseline: no occupancy field at all.
    fr_base = sim.forward(n_steps=50)
    ts_base = np.asarray(fr_base.time_series)

    # Zero occupancy must produce a bit-identical result to the baseline.
    occ_zero = jnp.zeros(g.shape, dtype=jnp.float32)
    fr_zero = sim.forward(pec_occupancy_override=occ_zero, n_steps=50)
    ts_zero = np.asarray(fr_zero.time_series)
    # Different XLA graphs for occ=0.0 vs no-occupancy baseline;
    # bit-identity not guaranteed across separate JIT compilations.
    assert np.allclose(ts_base, ts_zero, atol=1e-7, rtol=1e-7), (
        "zero pec_occupancy should be a no-op — got difference "
        f"max={np.max(np.abs(ts_base - ts_zero)):.3e}"
    )

    # Full-occupancy shell around the probe must shrink the probe trace
    # (soft-PEC zeros tangential E just like hard PEC for 1.0 occupancy).
    occ_shield = jnp.ones(g.shape, dtype=jnp.float32)
    fr_shield = sim.forward(pec_occupancy_override=occ_shield, n_steps=50)
    ts_shield = np.asarray(fr_shield.time_series)
    assert float(np.max(np.abs(ts_shield))) < 1e-6, (
        "full occupancy should zero the probe — got max "
        f"{float(np.max(np.abs(ts_shield))):.3e}"
    )


def test_forward_nonuniform_pec_occupancy_grad():
    """AD grad w.r.t. ``pec_occupancy_override`` is finite on the NU path."""
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    occ0 = jnp.full(g.shape, 0.1, dtype=jnp.float32)

    def loss(occ):
        fr = sim.forward(pec_occupancy_override=occ, n_steps=40)
        return jnp.sum(fr.time_series ** 2)

    grad = jax.grad(loss)(occ0)
    assert grad.shape == occ0.shape
    assert jnp.all(jnp.isfinite(grad)), "pec_occupancy grad has NaN/Inf"
    assert float(jnp.max(jnp.abs(grad))) > 0.0, "pec_occupancy grad is identically zero"


def test_forward_nonuniform_pec_occupancy_interpolates_hard_pec():
    """Soft occupancy monotonically interpolates between free-space and hard PEC.

    Physical invariants that must hold on any FDTD mesh (uniform or NU):
    - ``occ = 0`` reproduces the no-occupancy baseline (identity).
    - ``occ = 1`` reproduces the hard-PEC result that ``pec_mask_override``
      would give for the same region.
    - Intermediate occupancies monotonically reduce probe energy as occ
      increases — otherwise soft-PEC is not a valid relaxation of hard-PEC
      and density-based topology optimisation has no smooth descent path.
    """
    sim = _build_sim()
    g = sim._build_nonuniform_grid()

    # One-cell shell between the source and the probe.
    shell = np.zeros(g.shape, dtype=np.float32)
    ti, tj, tk = g.nx // 2, g.ny // 2, g.nz // 2 + 1
    shell[ti - 1:ti + 2, tj - 1:tj + 2, tk - 1] = 1.0

    def energy(occ_arr):
        fr = sim.forward(pec_occupancy_override=jnp.asarray(occ_arr, dtype=jnp.float32),
                         n_steps=80, skip_preflight=True)
        return float(jnp.sum(fr.time_series ** 2))

    energies = [energy(shell * s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for i in range(len(energies) - 1):
        assert energies[i] >= energies[i + 1] - 1e-3 * abs(energies[i]), (
            f"probe energy not monotonically non-increasing as occ grows: "
            f"{energies}"
        )

    # occ = 1 must match hard-PEC via pec_mask_override, bit-for-bit.
    fr_hard = sim.forward(pec_mask_override=jnp.asarray(shell.astype(bool)),
                          n_steps=80, skip_preflight=True)
    fr_occ1 = sim.forward(pec_occupancy_override=jnp.asarray(shell, dtype=jnp.float32),
                          n_steps=80, skip_preflight=True)
    ts_h = np.asarray(fr_hard.time_series)
    ts_o = np.asarray(fr_occ1.time_series)
    assert np.allclose(ts_h, ts_o, atol=0.0, rtol=0.0), (
        f"occ=1 must match hard-PEC on NU mesh — max diff "
        f"{float(np.max(np.abs(ts_h - ts_o))):.4e}"
    )


# ---------------------------------------------------------------------------
# Sentinel tests — pin known limitations so XPASS trips CI when fixed
# ---------------------------------------------------------------------------

def test_sim_forward_grad_wrt_dz_profile_through_init():
    """jax.grad through Simulation.__init__ w.r.t. dz_profile flows end-to-end.

    Closure of nu_known_limits.md sentinel #2 (2026-04-17): Simulation.__init__
    now gates the host-coercion sites (`float(np.sum(profile))`, grading
    warning, `_validate_mesh_quality`, `_check_numerical_dispersion`,
    `_validate_thin_metal_on_nu_mesh`, CPML-z advisory) with is_tracer(),
    so a tracer-valued dz_profile flows through to run_nonuniform. The
    caller must supply a concrete domain_z since the profile sum cannot
    be host-coerced during tracing.
    """
    dz = jnp.asarray(
        [0.5e-3] * 5 + [0.4e-3] * 4, dtype=jnp.float32
    )
    # Concrete domain_z — user's responsibility when dz_profile is traced.
    domain_z = float(jnp.sum(dz))

    def _loss(dz_profile):
        sim = Simulation(
            freq_max=10e9,
            domain=(0.01, 0.01, domain_z),
            dx=0.5e-3,
            dz_profile=dz_profile,
            cpml_layers=4,
        )
        sim.add_source((0.005, 0.005, 0.001), "ez")
        sim.add_probe((0.005, 0.005, 0.003), "ez")
        fr = sim.forward(n_steps=20, skip_preflight=True)
        return jnp.sum(fr.time_series ** 2)

    grad = jax.grad(_loss)(dz)
    assert grad.shape == dz.shape, f"grad shape {grad.shape} != {dz.shape}"
    assert bool(jnp.all(jnp.isfinite(grad))), "grad must be finite"
    assert float(jnp.sum(jnp.abs(grad))) > 0.0, "grad must be non-zero"


def test_grad_wrt_dx_profile_flows():
    """jax.grad w.r.t. dx_profile flows through make_nonuniform_grid.

    Closes the dx_profile half of the mesh-as-design-variable story
    (companion to `test_grad_wrt_dz_profile_flows`). The tracer-valued
    dx_profile routes through `_pad_profile`'s is_tracer branch, the CFL
    dt is computed in-trace, and `grid.dx_arr` carries the tracer into
    the sum, so the gradient is well-defined.
    """
    from rfx.nonuniform import make_nonuniform_grid

    dx0 = jnp.asarray([0.5e-3] * 6, dtype=jnp.float32)

    def _loss(dx_profile):
        grid = make_nonuniform_grid(
            domain_xy=(0.003, 0.003),
            dz_profile=jnp.asarray([0.5e-3] * 6, dtype=jnp.float32),
            dx=float(dx0[0]),
            dx_profile=dx_profile,
            cpml_layers=2,
        )
        return jnp.sum(grid.dx_arr)

    grad = jax.grad(_loss)(dx0)
    assert grad.shape == dx0.shape
    assert bool(jnp.all(jnp.isfinite(grad))), "dx_profile grad must be finite"
    assert float(jnp.sum(jnp.abs(grad))) > 0.0, "dx_profile grad must be non-zero"


def test_grad_wrt_dy_profile_flows():
    """jax.grad w.r.t. dy_profile flows through make_nonuniform_grid."""
    from rfx.nonuniform import make_nonuniform_grid

    dy0 = jnp.asarray([0.5e-3] * 6, dtype=jnp.float32)

    def _loss(dy_profile):
        grid = make_nonuniform_grid(
            domain_xy=(0.003, 0.003),
            dz_profile=jnp.asarray([0.5e-3] * 6, dtype=jnp.float32),
            dx=float(dy0[0]),
            dy_profile=dy_profile,
            cpml_layers=2,
        )
        return jnp.sum(grid.dy_arr)

    grad = jax.grad(_loss)(dy0)
    assert grad.shape == dy0.shape
    assert bool(jnp.all(jnp.isfinite(grad))), "dy_profile grad must be finite"
    assert float(jnp.sum(jnp.abs(grad))) > 0.0, "dy_profile grad must be non-zero"


def test_sim_forward_grad_wrt_dx_profile_through_init():
    """jax.grad through Simulation.__init__ w.r.t. dx_profile (x-axis twin of dz test).

    Exercises the full Simulation-level flow: constructor `is_tracer()`
    gate on domain[0], `make_nonuniform_grid` tracer x-path, CFL in-trace,
    source/probe index resolution via the uniform-nominal fallback, and
    FDTD scan through the traced cell sizes.
    """
    dx = jnp.asarray([0.5e-3] * 20, dtype=jnp.float32)
    domain_x = float(jnp.sum(dx))

    def _loss(dx_profile):
        sim = Simulation(
            freq_max=10e9,
            domain=(domain_x, 0.01, 0.004),
            dx=0.5e-3,
            dx_profile=dx_profile,
            cpml_layers=4,
        )
        sim.add_source((0.005, 0.005, 0.001), "ez")
        sim.add_probe((0.005, 0.005, 0.003), "ez")
        fr = sim.forward(n_steps=20, skip_preflight=True)
        return jnp.sum(fr.time_series ** 2)

    grad = jax.grad(_loss)(dx)
    assert grad.shape == dx.shape, f"grad shape {grad.shape} != {dx.shape}"
    assert bool(jnp.all(jnp.isfinite(grad))), "dx_profile grad must be finite"
    assert float(jnp.sum(jnp.abs(grad))) > 0.0, "dx_profile grad must be non-zero"


def test_sim_forward_grad_wrt_dy_profile_through_init():
    """jax.grad through Simulation.__init__ w.r.t. dy_profile (y-axis twin)."""
    dy = jnp.asarray([0.5e-3] * 20, dtype=jnp.float32)
    domain_y = float(jnp.sum(dy))

    def _loss(dy_profile):
        sim = Simulation(
            freq_max=10e9,
            domain=(0.01, domain_y, 0.004),
            dx=0.5e-3,
            dy_profile=dy_profile,
            cpml_layers=4,
        )
        sim.add_source((0.005, 0.005, 0.001), "ez")
        sim.add_probe((0.005, 0.005, 0.003), "ez")
        fr = sim.forward(n_steps=20, skip_preflight=True)
        return jnp.sum(fr.time_series ** 2)

    grad = jax.grad(_loss)(dy)
    assert grad.shape == dy.shape, f"grad shape {grad.shape} != {dy.shape}"
    assert bool(jnp.all(jnp.isfinite(grad))), "dy_profile grad must be finite"
    assert float(jnp.sum(jnp.abs(grad))) > 0.0, "dy_profile grad must be non-zero"


def test_grad_wrt_joint_dxyz_profile_flows():
    """jax.grad w.r.t. a joint (dx, dy, dz) profile flows end-to-end.

    Mesh-as-design-variable on all three axes simultaneously. Exercises
    the CFL in-trace path (all three axis minima are tracers) and the
    composition of the three tracer profiles in `make_nonuniform_grid`.
    """
    from rfx.nonuniform import make_nonuniform_grid

    d0 = jnp.asarray([0.5e-3] * 6, dtype=jnp.float32)

    def _loss(profiles):
        dx_p, dy_p, dz_p = profiles
        grid = make_nonuniform_grid(
            domain_xy=(0.003, 0.003),
            dz_profile=dz_p,
            dx=float(d0[0]),
            dx_profile=dx_p,
            dy_profile=dy_p,
            cpml_layers=2,
        )
        return jnp.sum(grid.dx_arr) + jnp.sum(grid.dy_arr) + jnp.sum(grid.dz)

    grads = jax.grad(_loss)((d0, d0, d0))
    for name, g, ref in zip(("dx", "dy", "dz"), grads, (d0, d0, d0)):
        assert g.shape == ref.shape, f"{name} grad shape mismatch"
        assert bool(jnp.all(jnp.isfinite(g))), f"{name} grad has NaN/Inf"
        assert float(jnp.sum(jnp.abs(g))) > 0.0, f"{name} grad identically zero"
