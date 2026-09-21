"""Worked RCS-reduction inverse design on the differentiable compute_rcs_jax (#421).

A broadband (9/10/11 GHz) backscatter RCS objective is minimized over a lossy COATING's
conductivity σ_coat (the RAM — radar-absorbing-material — DoF) via jax.grad through
run(tfsf, ntff) + compute_rcs_jax. Ground truth is the RAM impedance-matching physics:

  • RCS(σ_coat) has an INTERIOR minimum (too little loss = transparent; too much = conductive/
    reflective), so the gradient BRACKETS it: d(RCS)/dσ < 0 below the optimum, > 0 above. This
    pins that the gradient points the physically-correct way (not just "nonzero").
  • gradient descent (keeping the best iterate) reduces the band RCS by ~41% and lands at the
    RAM optimum σ_coat*≈0.3 S/m.

RE-PINNED 2026-09-22 (issue #1172), with the root cause. This cube scatters FORWARD 10-20 dB harder
than backward. On the far face of the Huygens box the electric and magnetic equivalent currents of
that forward wave must cancel toward the backscatter direction. Before #1159 the far-field rule read
E and H a full timestep apart (omega*dt = 20.6 deg at 10 GHz on this mesh) and half a cell apart
(k*dx/2 = 18 deg), so the cancellation failed and about a third of the forward-scattered field leaked
into the backscatter. Measured on this fixture, old rule -> second-order rule: forward scatter
1.13e-2 / 8.66e-3 / 8.47e-3 -> 1.08e-2 / 8.86e-3 / 9.38e-3 m^2 (unchanged within 0.4 dB), backscatter
1.03e-3 / 1.41e-3 / 2.71e-3 -> 1.84e-4 / 7.81e-5 / 9.42e-4 m^2 (4.6-12.6 dB lower). Mesh refinement of
the band backscatter at sigma = 0, dx = 3 / 1.5 / 1 mm: second-order rule 1.205e-3 / 9.95e-4 / 9.38e-4
(limit ~8.9e-4), old rule 5.15e-3 / 2.17e-3 / 1.65e-3 — the old rule was heading to the same value from
7.6 dB away. The old expectations here ("~47 %", "sigma* ~ 1.9") were read off the leakage-dominated
curve, and 1.9 is also exactly this optimizer's first step (lr*0.2/sqrt(0.001) = 6.3*lr), whatever the
gradient. Scan at this mesh with the corrected rule: RCS/RCS(0) = 0.617 / 0.591 / 0.599 / 0.619 / 0.726
at sigma = 0.2 / 0.3 / 0.4 / 0.5 / 1.0. jax.grad agrees with central differences on the corrected
rule (1.0001 at sigma = 1.9, 1.0095 at 0.5 with the difference still converging in h).

Builds on tests/unit/autodiff/test_rcs_jax_differentiable.py (equivalence to Mie-validated numpy + FD gradient).
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
    best, s_best = r0, 0.0
    m = v = 0.0
    # The normalized first step is 6.3*lr whatever the gradient; 0.05 keeps it (0.32) on the near
    # side of the optimum instead of jumping over it (module docstring).
    lr = 0.05
    for _ in range(20):
        val, g = vg(s)
        if float(val) < best:
            best, s_best = float(val), s
        g = float(g)
        m = 0.8 * m + 0.2 * g
        v = 0.999 * v + 0.001 * g * g
        s = float(np.clip(s - lr * m / (np.sqrt(v) + 1e-12), 0.0, 25.0))
    last = float(vg(s)[0])
    if last < best:
        best, s_best = last, s
    # Measured: best/r0 = 0.591 at sigma = 0.32 (scan minimum 0.591 at 0.3).
    assert best < 0.65 * r0, f"RCS reduction too small: {r0:.4e} -> {best:.4e} ({(1-best/r0)*100:.1f}%)"
    # The optimum's LOCATION is the physics this fixture carries: with the pre-#1159 far-field rule the
    # backscatter was mostly leaked forward scatter and the minimum sat near sigma = 1.0.
    assert 0.15 <= s_best <= 0.6, f"RAM optimum moved: best iterate at sigma_coat = {s_best:.2f} S/m"
