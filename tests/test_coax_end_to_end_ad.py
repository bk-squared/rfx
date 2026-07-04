"""End-to-end AD gate for ``compute_coaxial_line_reflection(eps_scale=...)``.

The coaxial moat gate (composition-level, per the G2 lesson that unit AD tests do
not protect compositions): ``grad(|S11|**2)`` w.r.t. a dielectric perturbation
flows through the FULL method —

    eps_scale -> Yee update -> DFT plane accumulators -> modal voltage
              -> matrix-pencil reflection -> Gamma

— is finite and matches a central finite difference. Passing this is what makes
``coaxial_port`` end-to-end AD-traceable (the auditor's ``ad_fd_test``); the
extractor-only property is separately gated in ``test_coaxial_line_extraction.py``.

Marked ``slow_physics`` (FDTD forward + reverse-mode tape); deselected by default.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse

# A small, resolved coax line: annulus ~3.8 cells at dx from freq_max=40 GHz,
# short-terminated. Kept minimal so the reverse-mode tape fits in memory.
N_STEPS = 1500
FREQ = jnp.asarray([8.0e9], dtype=jnp.float32)


def _build_sim():
    sim = Simulation(domain=(0.008, 0.008, 0.020), freq_max=40e9, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.010), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


def _s11_mag2(deps):
    """|S11|^2 with a permittivity bump ``deps`` multiplied into a mid-line slab."""
    sim = _build_sim()
    grid = sim._build_grid()
    nz = grid.shape[2]
    eps_scale = jnp.ones(grid.shape, dtype=jnp.float32).at[:, :, nz // 2 - 3: nz // 2 + 3].add(deps)
    res = sim.compute_coaxial_line_reflection(
        termination="short", n_steps=N_STEPS, freqs=FREQ, eps_scale=eps_scale,
    )
    return jnp.abs(res.s11[0]) ** 2


@pytest.mark.slow_physics
def test_coax_reflection_grad_finite_and_fd_consistent():
    """Gate: the full-method reflection is differentiable w.r.t. the dielectric and
    the AD gradient matches a finite difference (the AD moat for coaxial_port)."""
    val, g = jax.value_and_grad(_s11_mag2)(0.0)
    assert np.isfinite(float(val)), f"|S11|^2 is not finite: {val}"
    assert np.isfinite(float(g)), f"gradient is not finite: {g}"

    h = 0.02  # small enough that the matrix-pencil stays well-conditioned
    fd = (float(_s11_mag2(h)) - float(_s11_mag2(-h))) / (2 * h)
    assert np.isfinite(fd) and fd != 0.0, "FD slope not finite/nonzero — rebuild fixture"
    rel = abs(float(g) - fd) / max(abs(fd), 1e-12)
    assert rel <= 0.05, f"AD={float(g):+.6e} vs FD={fd:+.6e} (rel diff {rel:.3f} > 5%)"


@pytest.mark.slow_physics
def test_coax_eps_scale_unity_matches_concrete_path():
    """The AD path does not change the physics: ``eps_scale=1.0`` (jnp path) matches
    ``eps_scale=None`` (validated numpy path) in |S11|."""
    sim_a = _build_sim()
    res_a = sim_a.compute_coaxial_line_reflection(
        termination="short", n_steps=N_STEPS, freqs=FREQ)
    sim_b = _build_sim()
    grid = sim_b._build_grid()
    res_b = sim_b.compute_coaxial_line_reflection(
        termination="short", n_steps=N_STEPS, freqs=FREQ,
        eps_scale=jnp.ones(grid.shape, dtype=jnp.float32))
    sa = np.asarray(res_a.s11)
    sb = np.asarray(res_b.s11)
    assert np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))
    np.testing.assert_allclose(np.abs(sb), np.abs(sa), rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-m", "slow_physics"])
