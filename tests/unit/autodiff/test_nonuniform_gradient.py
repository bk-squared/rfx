"""Gradient regression for the NonUniformGrid run lane.

Case (a) pins what already works empirically (gap-probe 2026-04-15):
``jax.grad`` through ``run_nonuniform`` w.r.t. a scalar source amplitude
agrees with a finite-difference estimate.

Case (b) pins the known gap: differentiating w.r.t. ``dz_profile`` hits
a host boundary inside ``make_nonuniform_grid`` (``np.asarray`` /
``float(np.min)``) and raises ``TracerArrayConversionError``.  Marked
``xfail(strict=True)`` so that a future grid-construction refactor that
stays in trace flips this test to XPASS and fails loudly — the hole
self-announces when it closes.

See docs/research_notes/2026-04-15_nonuniform_completion_handoff.md
(Step 2 + Step 5) for the scope behind this test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import EPS_0, MaterialArrays
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform


def _base_waveform(n_steps: int, dt: float) -> jnp.ndarray:
    t = jnp.arange(n_steps, dtype=jnp.float32) * jnp.float32(dt)
    t0 = 15.0 * jnp.float32(dt)
    width = 5.0 * jnp.float32(dt)
    return jnp.exp(-((t - t0) / width) ** 2).astype(jnp.float32)


def _build():
    """Small but non-trivial graded-z sim with CPML on all sides."""
    dz = np.array([0.5e-3] * 5 + [0.3e-3] * 4, dtype=np.float64)
    grid = make_nonuniform_grid(
        domain_xy=(0.005, 0.005),
        dz_profile=dz,
        dx=0.5e-3,
        cpml_layers=4,
    )
    nx, ny, nz = grid.shape
    shape = (nx, ny, nz)
    materials = MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
    )
    return grid, materials


def _loss_from_amplitude(amplitude, grid, materials, n_steps):
    """Run nonuniform sim with a single Ez source scaled by `amplitude`
    and return the L2 energy of the probe trace (a scalar loss)."""
    nx, ny, nz = grid.shape
    base = _base_waveform(n_steps, grid.dt)
    src_i, src_j, src_k = nx // 2, ny // 2, nz // 2 + 1
    prb_i, prb_j, prb_k = nx // 2, ny // 2, nz // 2 - 1
    sources = [(src_i, src_j, src_k, "ez", amplitude * base)]
    probes = [(prb_i, prb_j, prb_k, "ez")]
    out = run_nonuniform(grid, materials, n_steps,
                         sources=sources, probes=probes)
    ts = out["time_series"][:, 0]
    return jnp.sum(ts * ts)


def test_grad_wrt_source_amplitude_matches_fd():
    """AD through the nonuniform scan agrees with centered FD to <1%."""
    grid, materials = _build()
    n_steps = 60
    amp0 = jnp.float32(1.0)

    grad_ad = float(jax.grad(
        _loss_from_amplitude)(amp0, grid, materials, n_steps))

    h = 1e-2
    loss_plus  = float(_loss_from_amplitude(amp0 + h, grid, materials, n_steps))
    loss_minus = float(_loss_from_amplitude(amp0 - h, grid, materials, n_steps))
    grad_fd = (loss_plus - loss_minus) / (2.0 * h)

    rel_err = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
    assert rel_err < 0.01, (
        f"AD grad {grad_ad:.4e} vs FD grad {grad_fd:.4e} — "
        f"rel_err {rel_err:.4%} above 1% threshold"
    )


def _loss_from_dz(dz):
    """Loss function closure that threads dz_profile into grid + scan."""
    grid = make_nonuniform_grid(
        domain_xy=(0.005, 0.005), dz_profile=dz,
        dx=0.5e-3, cpml_layers=4,
    )
    nx, ny, nz = grid.shape
    shape = (nx, ny, nz)
    materials = MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
    )
    base = _base_waveform(40, grid.dt)
    sources = [(nx // 2, ny // 2, nz // 2, "ez", base)]
    probes = [(nx // 2, ny // 2, nz // 2 - 1, "ez")]
    out = run_nonuniform(grid, materials, 40,
                         sources=sources, probes=probes)
    return jnp.sum(out["time_series"] ** 2)


def test_grad_wrt_dz_profile_flows():
    """AD grad w.r.t. ``dz_profile`` is finite and non-trivial.

    Previously ``xfail(strict=True)`` because ``make_nonuniform_grid`` and
    ``_cpml_profile`` had host boundaries (``np.asarray`` /
    ``float(np.min)`` / ``float(dz_arr[0])``) that broke the JAX trace.
    Issue #45 removed those boundaries so ``dz_profile`` now flows as a
    tracer through grid construction, CFL / ``dt`` derivation, and the
    CPML profile.  This unblocks mesh-as-design-variable inverse design.
    """
    dz0 = jnp.asarray([0.5e-3] * 5 + [0.3e-3] * 4, dtype=jnp.float32)
    grad_ad = jax.grad(_loss_from_dz)(dz0)
    assert grad_ad.shape == dz0.shape
    assert jnp.all(jnp.isfinite(grad_ad)), "dz_profile grad contains NaN/Inf"
    # At least one cell moves the loss non-trivially — verifies the
    # tracer actually propagates (constant-zero would also be finite).
    assert float(jnp.max(jnp.abs(grad_ad))) > 1.0


def test_grad_wrt_dz_profile_matches_fd():
    """AD↔FD agreement on dominant cells with unique dz values.

    ``jnp.min(dz_full)`` appears in the CFL / ``dt`` derivation, so ties in
    ``dz_profile`` produce non-smooth subgradients at the min.  AD picks
    one tied cell (argmin convention); centered FD spreads the
    contribution across all tied cells.  Both are legitimate; neither is
    a bug.  The test uses a strictly-monotone ``dz_profile`` so ``min`` is
    uniquely attained and AD↔FD agreement on dominant cells (>5 % of
    max magnitude) is a meaningful regression guard.
    """
    # Strictly-monotone profile — no ties in min(dz_full).
    dz0 = jnp.asarray(
        [0.50e-3, 0.51e-3, 0.52e-3, 0.53e-3, 0.54e-3,
         0.30e-3, 0.31e-3, 0.32e-3, 0.33e-3],
        dtype=jnp.float32,
    )
    grad_ad = np.asarray(jax.grad(_loss_from_dz)(dz0))

    h = 1e-7
    grad_fd = np.zeros_like(grad_ad)
    for i in range(len(dz0)):
        e = np.zeros(len(dz0), dtype=np.float32); e[i] = h
        lp = float(_loss_from_dz(dz0 + jnp.asarray(e)))
        lm = float(_loss_from_dz(dz0 - jnp.asarray(e)))
        grad_fd[i] = (lp - lm) / (2.0 * h)

    max_mag = float(np.max(np.abs(grad_fd)))
    big = np.abs(grad_fd) > 0.05 * max_mag
    assert np.all(np.sign(grad_ad[big]) == np.sign(grad_fd[big])), (
        f"sign disagreement on dominant cells: AD={grad_ad[big]}, FD={grad_fd[big]}"
    )
    rel = np.max(
        np.abs(grad_ad[big] - grad_fd[big]) / (np.abs(grad_fd[big]) + 1e-30)
    )
    assert rel < 0.15, (
        f"dominant-cell AD↔FD rel_err {rel:.4f} above 15 % threshold; "
        f"AD={grad_ad[big]}, FD={grad_fd[big]}"
    )


def test_grad_wrt_dz_profile_reduces_loss():
    """One step of steepest descent along the AD gradient reduces loss.

    This is the strongest physical check available for the mesh-as-
    design-variable direction — it verifies that the gradient is usable
    as an optimisation signal, not just a first-order match to FD.
    """
    dz0 = jnp.asarray(
        [0.50e-3, 0.51e-3, 0.52e-3, 0.53e-3, 0.54e-3,
         0.30e-3, 0.31e-3, 0.32e-3, 0.33e-3],
        dtype=jnp.float32,
    )
    loss0 = float(_loss_from_dz(dz0))
    grad = jax.grad(_loss_from_dz)(dz0)
    # Tiny step — grads are ~1e5, we want a ~1e-7 change in dz (well
    # below the cell spacing).  Clamp to physical range.
    dz1 = jnp.clip(dz0 - 1e-12 * grad, 0.1e-3, 1.0e-3)
    loss1 = float(_loss_from_dz(dz1))
    reduction = loss0 - loss1
    # Floor is 1e-6 * |loss0|: three orders of magnitude below the observed
    # ~0.046% reduction signal, tight enough to catch noise-dominated passes.
    floor = max(1e-6 * abs(loss0), 1e-12)
    assert reduction > floor, (
        f"gradient step must produce meaningful loss reduction: "
        f"got {reduction:.3e}, floor={floor:.3e} (loss0={loss0:.3e})"
    )
