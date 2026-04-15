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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Geometry gradient blocked: make_nonuniform_grid calls np.asarray "
        "and float(np.min) on the dz_profile input before any JAX op "
        "touches it — a host boundary that breaks AD. Requires refactoring "
        "grid construction to stay in trace (see Step 5 of "
        "docs/research_notes/2026-04-15_nonuniform_completion_handoff.md). "
        "When this flips to XPASS the hole has closed."
    ),
)
def test_grad_wrt_dz_profile_blocked():
    """Differentiating w.r.t. dz_profile must fail loudly until Step 5."""
    def loss_fn(dz):
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

    dz0 = jnp.asarray([0.5e-3] * 5 + [0.3e-3] * 4, dtype=jnp.float32)
    # Expected to raise TracerArrayConversionError / TypeError —
    # xfail(strict=True) captures both positive-grad and raised outcomes
    # as a failure.  We call jax.grad so any raise counts as xfail.
    _ = jax.grad(loss_fn)(dz0)
