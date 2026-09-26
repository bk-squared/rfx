"""A dielectric interface sits where it was drawn, to second order (#1210).

A 24 mm PEC cube, the lower half (z < 12 mm) filled with eps_r = 4. The lowest
mode with E transverse to z and one half-wave across x -- TE(1,0)-to-z -- has a
closed form. E_t is tangential to the z = 12 mm plane, so E_t and H_t are both
continuous there; with one mu that makes f(z) and df/dz continuous, and with
f(0) = f(d) = 0 at the electric walls the transverse resonance is

    k1*cot(k1*H) + k2*cot(k2*(d-H)) = 0,   k_i = sqrt(eps_i*(w/c)^2 - k_t^2)

with k_t = pi/a. Between the dielectric cutoff (3.13 GHz) and the air cutoff
(6.25 GHz) the air half is EVANESCENT: k2 = -j*alpha and k2*cot(k2*L)
becomes +alpha*coth(alpha*L). The root is 5.2177 GHz.

WHAT THIS PINS, and what it read before. The tangential E on the dielectric's
top face used to take whichever single cell owned the edge -- the air cell --
instead of the mean of the four cells the edge touches. On main this fixture
reads 5.3069 GHz at dx = 1 mm, 1.71e-2 relative: the interface is effectively
half a cell out of place, and the error halves with the mesh (8.69e-3 at
dx = 0.5 mm) because the treatment is first order. With the edge average it
reads 5.2162 GHz (2.94e-4) and 5.2174 GHz (5.87e-5) -- quartering, second
order. The 5e-3 bound below is between the two with an order of margin on
each side; main fails it, this branch passes it by a factor of 17.

Measured with scripts/diagnostics/half_filled_cavity_interface_order.py, which
also carries the cell-owned arm.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest
from scipy.optimize import brentq

from rfx import GaussianPulse, Simulation
from rfx.harminv import harminv

C0 = 299792458.0
L = 24e-3          # cube side, a = b = d
H = 12e-3          # fill height
EPSR = 4.0
M_T, N_T = 1, 0    # transverse mode indices: TE(1,0)-to-z

BOUND = 5e-3       # main reads 1.71e-2 here; this branch 2.94e-4


def _residual(freq, kt):
    k0 = 2 * np.pi * freq / C0

    def branch(eps, length):
        ksq = eps * k0 ** 2 - kt ** 2
        if ksq > 0:
            k = np.sqrt(ksq)
            return k / np.tan(k * length)          # k*cot(k*L)
        alpha = np.sqrt(-ksq)                       # k = -j*alpha
        return alpha / np.tanh(alpha * length)      # +alpha*coth(alpha*L)

    return branch(EPSR, H) + branch(1.0, L - H)


def _analytic():
    kt = np.sqrt((M_T * np.pi / L) ** 2 + (N_T * np.pi / L) ** 2)
    f_lo = C0 * kt / (2 * np.pi * np.sqrt(EPSR)) * 1.0001
    fs = np.linspace(f_lo, 12e9, 20000)
    vals = np.array([_residual(f, kt) for f in fs])
    for i in range(len(fs) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0 and abs(a - b) < 1e4:
            return brentq(_residual, fs[i], fs[i + 1], args=(kt,))
    raise AssertionError("no root of the transverse resonance was found")


def _measure(dx, steps, window, debye_block=False):
    f0 = 0.5 * (window[0] + window[1])
    sim = Simulation(freq_max=3 * f0, domain=(L, L, L), dx=dx, boundary="pec")
    if debye_block:
        # #1260: a 1 mm Debye cube of delta_eps 1e-6 in the far corner of the
        # air half -- electrically nothing, but it puts the whole grid on the
        # dispersive E update.
        from rfx import Box
        from rfx.materials.debye import DebyePole
        sim.add_material("dbl", eps_r=1.0,
                         debye_poles=[DebyePole(delta_eps=1e-6, tau=1e-11)])
        sim.add(Box((L - 2e-3,) * 3, (L - 1e-3,) * 3), material="dbl")
    sim.add_source((L * 0.23, L * 0.31, L * 0.41), "ex",
                   amplitude_kind="field",
                   waveform=GaussianPulse(f0=f0, bandwidth=1.0))
    sim.add_probe((L * 0.73, L * 0.64, L * 0.29), "ex")
    probe = sim.forward(n_steps=2, skip_preflight=True)
    shape, dt = tuple(probe.grid.shape), float(probe.grid.dt)
    z_centres = (np.arange(shape[2]) + 0.5) * dx
    eps = jnp.broadcast_to(
        jnp.asarray(np.where(z_centres < H, EPSR, 1.0).astype(np.float32))[None, None, :],
        shape)
    res = sim.forward(eps_override=eps, n_steps=steps, skip_preflight=True,
                      checkpoint=False)
    modes = [m for m in harminv(np.asarray(res.time_series[:, 0])[steps // 4:],
                                dt, *window) if m.Q > 30]
    assert modes, f"no mode with Q > 30 in {window} at dx = {dx}"
    return max(modes, key=lambda m: m.amplitude).freq


def test_the_half_filled_cube_reads_its_analytic_te10_and_converges():
    f_an = _analytic()
    assert f_an == pytest.approx(5.2177e9, rel=1e-4), (
        "the transverse-resonance root moved; check the evanescent branch "
        "(+alpha*coth, not -alpha*coth)")
    window = (f_an * 0.94, f_an * 1.10)

    f_coarse = _measure(1e-3, 6000, window)
    err_coarse = abs(f_coarse - f_an) / f_an
    assert err_coarse < BOUND, (
        f"TE(1,0) of the half-filled cube reads {f_coarse / 1e9:.4f} GHz "
        f"against the analytic {f_an / 1e9:.4f} GHz at dx = 1 mm "
        f"({err_coarse:.3e} relative). 5.3069 GHz (1.71e-2) is the cell-owned "
        f"rule: the tangential E on the dielectric's top face taking the air "
        f"cell's permittivity instead of the mean of its four cells.")

    f_fine = _measure(0.5e-3, 12000, window)
    err_fine = abs(f_fine - f_an) / f_an
    assert err_fine < err_coarse, (
        f"halving the mesh did not reduce the error: {err_coarse:.3e} at "
        f"dx = 1 mm, {err_fine:.3e} at dx = 0.5 mm "
        f"({f_coarse / 1e9:.4f} -> {f_fine / 1e9:.4f} GHz)")


def test_a_negligible_debye_block_leaves_the_interface_where_it_was_drawn():
    """#1260: with any Debye or Lorentz pole present the dispersive update
    replaces the E update over the whole grid, and it used to build its
    coefficients from the cell that owns each edge -- this fixture then read
    5.3069 GHz, the cell-owned number, with a delta_eps = 1e-6 cube in a far
    corner. Its coefficients are the edge mean now: the block changes nothing
    (measured: 5.2161968 against 5.2161967 GHz)."""
    f_an = _analytic()
    window = (f_an * 0.94, f_an * 1.10)
    f_plain = _measure(1e-3, 6000, window)
    f_block = _measure(1e-3, 6000, window, debye_block=True)
    assert abs(f_block - f_plain) / f_plain < 1e-5, (
        f"a negligible Debye block moves TE(1,0) from {f_plain / 1e9:.4f} to "
        f"{f_block / 1e9:.4f} GHz; 5.3069 GHz is the interface edge taking the "
        f"air cell alone on the dispersive update (#1260)")
