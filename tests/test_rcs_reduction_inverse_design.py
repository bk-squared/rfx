"""Worked RCS-reduction inverse design on the differentiable compute_rcs_jax (#421).

A broadband (9/10/11 GHz) backscatter RCS objective is minimized over a lossy COATING's
conductivity σ_coat (the RAM — radar-absorbing-material — DoF) via jax.grad through
run(tfsf, ntff) + compute_rcs_jax. Ground truth is the RAM impedance-matching physics:

  • RCS(σ_coat) has an INTERIOR minimum (too little loss = transparent; too much = conductive/
    reflective), so the gradient BRACKETS it: d(RCS)/dσ < 0 below the optimum, > 0 above. This
    pins that the gradient points the physically-correct way (not just "nonzero").
  • gradient descent (keeping the best iterate) reduces the band RCS by ~47% and lands at the
    RAM optimum σ_coat*≈1.9 S/m.

Builds on tests/test_rcs_jax_differentiable.py (equivalence to Mie-validated numpy + FD gradient).
Harness: docs/research_notes/experiments/i404_oblique_20260720/rcs_reduction_diag.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.farfield import NTFFBox
from rfx.sources.tfsf import init_tfsf
from rfx.simulation import run
from rfx.rcs import compute_rcs_jax, _incident_spectrum_amplitude

F0, BW = 10e9, 0.5
CPML, N_STEPS = 8, 220
DOMAIN = (0.09, 0.09, 0.09)
DX = 0.003
FREQS = np.array([9.0, 10.0, 11.0]) * 1e9  # broadband objective
TH_B, PH_B = np.array([np.pi / 2]), np.array([np.pi])


def _setup():
    grid = Grid(freq_max=15e9, domain=DOMAIN, dx=DX, cpml_layers=CPML)
    e_inc = _incident_spectrum_amplitude(F0, BW, FREQS, grid.dt, N_STEPS)
    sx, sy, sz = grid.shape
    blk = np.zeros(grid.shape, np.float32)
    blk[sx // 2 - 3:sx // 2 + 3, sy // 2 - 3:sy // 2 + 3, sz // 2 - 3:sz // 2 + 3] = 1.0
    return grid, e_inc, jnp.asarray(blk)


def _band_rcs(grid, e_inc, blk, sigma_scale):
    mb = init_materials(grid.shape)
    mats = mb._replace(eps_r=mb.eps_r + 3.0 * blk,               # εr=4 core
                       sigma=mb.sigma + sigma_scale * blk)        # lossy coating DoF
    cfg, st = init_tfsf(nx=grid.nx, dx=grid.dx, dt=grid.dt, cpml_layers=CPML, tfsf_margin=3,
                        f0=F0, bandwidth=BW, amplitude=1.0, polarization="ez",
                        direction="+x", angle_deg=0.0)
    fl = {k: CPML for k in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")}
    box = NTFFBox.from_grid(
        grid, i_lo=max(cfg.x_lo - 1, 1), i_hi=min(cfg.x_hi + 2, grid.nx - 2),
        j_lo=fl["y_lo"] + 1, j_hi=grid.ny - fl["y_hi"] - 1,
        k_lo=fl["z_lo"] + 1, k_hi=grid.nz - fl["z_hi"] - 1,
        freqs=jnp.array(FREQS, jnp.float32))
    nd = run(grid, mats, N_STEPS, boundary="cpml", tfsf=(cfg, st), ntff=box).ntff_data
    return jnp.sum(compute_rcs_jax(nd, box, grid, TH_B, PH_B, e_inc)[:, 0, 0])


@pytest.mark.slow
def test_rcs_reduction_gradient_brackets_ram_optimum():
    """d(RCS)/dσ_coat < 0 below the RAM optimum and > 0 above it (physically-correct sign)."""
    grid, e_inc, blk = _setup()
    grad = jax.grad(lambda s: _band_rcs(grid, e_inc, blk, s))
    g_lo = float(grad(0.0))
    g_hi = float(grad(10.0))
    assert np.isfinite(g_lo) and np.isfinite(g_hi)
    assert g_lo < 0.0 < g_hi, (
        f"gradient must bracket the interior RAM optimum: g(0)={g_lo:.3e} (want<0), "
        f"g(10)={g_hi:.3e} (want>0)"
    )


@pytest.mark.slow
def test_rcs_reduction_inverse_design_reduces_backscatter():
    """Gradient descent on σ_coat reduces the broadband backscatter RCS by a large margin."""
    grid, e_inc, blk = _setup()
    vg = jax.value_and_grad(lambda s: _band_rcs(grid, e_inc, blk, s))

    s = 0.0
    r0, g0 = (lambda v: (float(v[0]), float(v[1])))(vg(s))
    assert g0 < 0.0, "at σ_coat=0 more loss must reduce RCS (descent increases σ)"
    best = r0
    m = v = 0.0
    lr = 0.3
    for _ in range(20):
        val, g = vg(s)
        best = min(best, float(val))
        g = float(g)
        m = 0.8 * m + 0.2 * g
        v = 0.999 * v + 0.001 * g * g
        s = float(np.clip(s - lr * m / (np.sqrt(v) + 1e-12), 0.0, 25.0))
    best = min(best, float(vg(s)[0]))
    assert best < 0.6 * r0, f"RCS reduction too small: {r0:.4e} -> {best:.4e} ({(1-best/r0)*100:.1f}%)"
