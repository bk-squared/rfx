"""Issue #70 — NU s_params extraction tracer-safe under jax.grad.

Before the fix: ``rfx/nonuniform.py:1140`` did
``S[j, k, :] = _np.array(b_j / safe_a_k)``. When ``sim.forward`` is
wrapped in ``jax.grad`` and a WirePort with ``extent=`` is registered,
``b_j / safe_a_k`` is a JAX tracer, and ``_np.array(tracer)`` raises
``TracerArrayConversionError``, blocking differentiable inverse design
on non-uniform grids (the IEEE T-MTT paper's ex3_series_fed_array_nu
scenario).

After the fix: ``S`` is a jnp array built via
``.at[j, k, :].set(...)``; the extractor is tracer-safe and the
concrete-path output remains readable by numpy consumers via
``np.asarray(s_params)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.sources.sources import GaussianPulse


# Minimal NU sim with a WirePort spanning a 3-cell "substrate" so the
# wire_sparams extractor has multi-cell data to accumulate.
F0 = 5e9
DX = 2e-3
NX, NY, NZ = 20, 10, 10
H_SUB = 3 * DX   # 3 cells — minimum for a meaningful WirePort


def _build_nu_sim_with_wireport(*, sub_eps: float = 2.5) -> Simulation:
    sim = Simulation(
        freq_max=2 * F0,
        domain=(NX * DX, NY * DX, NZ * DX),
        dx=DX,
        dx_profile=np.full(NX, DX, dtype=np.float64),
        dy_profile=np.full(NY, DX, dtype=np.float64),
        dz_profile=np.full(NZ, DX, dtype=np.float64),
        boundary="pec",
    )
    sim.add_material("sub", eps_r=sub_eps)
    # Thin dielectric slab the WirePort will span.
    from rfx import Box
    feed_x, feed_y = NX // 2 * DX, NY // 2 * DX
    sub_z0 = 3 * DX
    sim.add(
        Box((0.0, 0.0, sub_z0), (NX * DX, NY * DX, sub_z0 + H_SUB)),
        material="sub",
    )
    # WirePort spans the substrate thickness along z.
    sim.add_port(
        (feed_x, feed_y, sub_z0),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=F0, bandwidth=0.5),
        extent=H_SUB,
    )
    sim.add_probe((feed_x, feed_y, sub_z0 + H_SUB / 2), "ez")
    return sim


def test_nu_wireport_grad_does_not_crash():
    """jax.grad on a NU sim with a WirePort extent= must NOT raise
    TracerArrayConversionError (issue #70).

    sim.forward auto-enables wire_sparams when a WirePort is registered
    (rfx/runners/nonuniform.py:436-440 defaults sp_freqs via linspace
    when compute_s_params is None), so the s_params extractor fires
    automatically under jax.grad — previously this is the site that
    called _np.array on a tracer and crashed.
    """

    def loss(eps_scale: jax.Array) -> jax.Array:
        sim = _build_nu_sim_with_wireport(sub_eps=2.5)
        grid = sim._build_nonuniform_grid()
        eps = jnp.ones(grid.shape, dtype=jnp.float32) * eps_scale
        fr = sim.forward(eps_override=eps, n_steps=30)
        return jnp.sum(fr.time_series ** 2)

    alpha0 = jnp.float32(1.0)
    grad = float(jax.grad(loss)(alpha0))
    assert np.isfinite(grad), f"non-finite grad: {grad}"


def test_nu_wireport_concrete_path_s_params_still_numpy_compatible():
    """Post-fix concrete-path S-matrix must still be consumable by numpy
    (jnp arrays implement __array__ — pin backward compatibility)."""
    freqs = np.linspace(0.5 * F0, 1.5 * F0, 11).astype(np.float32)
    sim = _build_nu_sim_with_wireport()
    fr = sim.run(n_steps=30, compute_s_params=True, s_param_freqs=freqs)
    s = fr.s_params
    # np.asarray accepts jnp and returns a usable ndarray.
    s_np = np.asarray(s)
    assert s_np.shape == (1, 1, len(freqs)), (
        f"unexpected S-matrix shape {s_np.shape}"
    )
    assert np.all(np.isfinite(s_np)), "concrete-path S has non-finite values"
