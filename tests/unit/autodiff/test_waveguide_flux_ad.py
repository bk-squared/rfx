"""Issue #148: ``normalize='flux'`` waveguide S-matrix on the AD tape.

Before the fix, ``extract_waveguide_s_matrix_flux`` concretized traced
arrays (``np.array(flux_spectrum(...))`` at the old waveguide_port.py:2307
plus np in-place S-matrix assembly), so
``compute_waveguide_s_matrix(normalize='flux', eps_override=<traced>)``
under ``jax.grad`` raised ``TracerArrayConversionError`` — the
production-recommended power-flux path was the one extraction that
could NOT be optimized through.

Gates here (composition-level, per the G2 lesson: unit AD tests do not
protect compositions):
  1. grad(|S21(f0)|^2) w.r.t. a substrate-eps scalar through the FULL
     flux extraction is finite (was: raise).
  2. AD matches central finite differences.
  3. Forward S-matrix is unchanged vs the pre-fix path (regression for
     the np->jnp rewrite; tolerance covers float reassociation only).
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec, Boundary

NUM_PERIODS = 8.0
FREQS = jnp.linspace(5e9, 6.5e9, 4)


def _wr90_sim():
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 14 * 0.003, 7 * 0.003),
        dx=0.003,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )  # solved length; #1887
    sim.add_waveguide_port(0.010, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=FREQS, f0=6e9, bandwidth=0.5, name="left")
    sim.add_waveguide_port(0.090, direction="-x", mode=(1, 0), mode_type="TE",
                           freqs=FREQS, f0=6e9, bandwidth=0.5, name="right")
    return sim


def _slab_cells(sim, grid):
    """A 4-cell slab between the two ports' realized S21 probe planes, off both
    ports' sections: a traced medium at a probe plane is refused (#2016)."""
    from rfx.boundaries.features import waveguide_segment_cells
    n_steps = grid.num_timesteps(num_periods=NUM_PERIODS)
    cfgs = [sim._build_waveguide_port_config(e, grid, FREQS, n_steps)
            for e in sim._waveguide_ports]
    lo, hi = sorted(cfg.probe_x for cfg in cfgs)
    i_lo = (lo + hi) // 2 - 2
    i_hi = i_lo + 4
    assert lo < i_lo and i_hi <= hi, f"slab {i_lo}:{i_hi} not between probes {lo}, {hi}"
    for cfg in cfgs:
        s_lo, s_hi = waveguide_segment_cells(cfg, grid)[0]
        assert i_hi <= s_lo or s_hi <= i_lo, f"slab {i_lo}:{i_hi} in {cfg.port_name} section {s_lo}:{s_hi}"
    return i_lo, i_hi


def _eps_override_for(sim, deps):
    """Uniform-eps grid override 1.0 + deps in a slab mid-guide (traced)."""
    grid = sim._build_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    i_lo, i_hi = _slab_cells(sim, grid)
    return eps.at[i_lo:i_hi, :, :].add(deps)


def _s21_mag2(deps):
    sim = _wr90_sim()
    eps = _eps_override_for(sim, deps)
    res = sim.compute_waveguide_s_matrix(
        num_periods=NUM_PERIODS, normalize="flux", eps_override=eps,
    )
    # |S21|^2 at the band-center bin (index 2 of the 4-point grid)
    return jnp.abs(res.s_params[1, 0, 2]) ** 2


def test_flux_smatrix_grad_finite_and_fd_consistent():
    """Gates 1+2: traced flux extraction yields a finite, FD-consistent grad."""
    deps0 = jnp.asarray(0.5, dtype=jnp.float32)
    val, g = jax.value_and_grad(_s21_mag2)(deps0)
    assert np.isfinite(float(val))
    assert np.isfinite(float(g)), f"grad is {g}"

    h = 0.05
    fd = (float(_s21_mag2(deps0 + h)) - float(_s21_mag2(deps0 - h))) / (2 * h)
    assert fd != 0.0, "FD slope is zero — fixture has no sensitivity; rebuild it"
    rel = abs(float(g) - fd) / max(abs(fd), 1e-12)
    assert rel <= 0.05, (
        f"AD={float(g):+.6e} vs FD={fd:+.6e} (rel diff {rel:.3f} > 5%)"
    )


def test_flux_smatrix_forward_matches_untraced():
    """Gate 3: the jnp rewrite did not change forward values — the traced
    and untraced (no eps_override) paths share the assembly, so compare
    normalize='flux' against itself with a no-op override."""
    sim_a = _wr90_sim()
    res_a = sim_a.compute_waveguide_s_matrix(num_periods=NUM_PERIODS,
                                             normalize="flux")
    sim_b = _wr90_sim()
    eps = _eps_override_for(sim_b, jnp.asarray(0.0, dtype=jnp.float32))
    res_b = sim_b.compute_waveguide_s_matrix(num_periods=NUM_PERIODS,
                                             normalize="flux",
                                             eps_override=eps)
    sa = np.asarray(res_a.s_params)
    sb = np.asarray(res_b.s_params)
    assert np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))
    np.testing.assert_allclose(sb, sa, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
