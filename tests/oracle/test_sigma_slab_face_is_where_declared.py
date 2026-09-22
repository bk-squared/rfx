"""A volume conductor's plus face sits on the boundary it was drawn on (#1210).

A 6 mm slab of sigma = 1e5 S/m lies on the floor of a 24 mm PEC cube. Its skin
depth at 10 GHz is 50 um, a twentieth of a cell, so the slab is a short: the
cavity above it is 18 mm tall and its lowest mode with one half-wave along z
and one along x sits at c/2 * sqrt(1/L^2 + 1/(L-h)^2) = 10.4095 GHz.

Until #1210 the E coefficients came from the single cell that owns each edge,
so the tangential E on the slab's top face took the air cell's material and the
slab shorted one cell lower than drawn. The cavity read 10.0585 GHz -- the
answer for a 5 mm slab (analytic 10.0623), 3.3 % low. With the edge average the
same fixture reads the PEC occupancy lane's number.

The PEC lane is run here, in the same process and on the same build, rather
than quoted: it is the reference that makes this a statement about WHERE the
face is, not about how well a lossy slab imitates a PEC one.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx import GaussianPulse, Simulation
from rfx.harminv import harminv

C0 = 299792458.0
DX, N = 1e-3, 24
L = N * DX
H = 6e-3          # declared slab height
F0 = 10.5e9
STEPS = 3000
SIGMA = 1e5       # S/m -- a short at 10 GHz, below the 1e6 PEC-mask threshold


def _fixture():
    sim = Simulation(freq_max=2 * F0, domain=(L, L, L), dx=DX, boundary="pec")
    sim.add_source((5.5e-3, 7.5e-3, 15.5e-3), "ex", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.6))
    sim.add_probe((17.5e-3, 15.5e-3, 19.5e-3), "ex")
    return sim


def _strongest(ts, dt):
    modes = [m for m in harminv(np.asarray(ts)[STEPS // 4:], dt, 8e9, 12e9)
             if m.Q > 30]
    assert modes, "no mode with Q > 30 in 8-12 GHz -- the fixture rang out"
    return max(modes, key=lambda m: m.amplitude).freq


def _analytic(h):
    return C0 / 2 * np.sqrt(1.0 / L ** 2 + 1.0 / (L - h) ** 2)


def test_sigma_slab_reads_the_pec_occupancy_lane_and_the_analytic_height():
    sim = _fixture()
    probe = sim.forward(n_steps=2, skip_preflight=True)
    shape = tuple(probe.grid.shape)
    dt = float(probe.grid.dt)

    z_centres = (np.arange(shape[2]) + 0.5) * DX
    slab = (z_centres < H).astype(np.float32)
    plane = lambda a: jnp.broadcast_to(jnp.asarray(a)[None, None, :], shape)

    f_sigma = _strongest(
        sim.forward(sigma_override=plane(slab * SIGMA), n_steps=STEPS,
                    skip_preflight=True, checkpoint=False).time_series[:, 0],
        dt)
    f_pec = _strongest(
        sim.forward(pec_occupancy_override=plane(slab), n_steps=STEPS,
                    skip_preflight=True, checkpoint=False).time_series[:, 0],
        dt)

    rel_lane = abs(f_sigma - f_pec) / f_pec
    rel_an = abs(f_sigma - _analytic(H)) / _analytic(H)
    assert rel_lane < 2e-3, (
        f"the sigma slab and the PEC occupancy lane disagree by {rel_lane:.2e} "
        f"({f_sigma / 1e9:.4f} vs {f_pec / 1e9:.4f} GHz). The two rules realize "
        f"the same face: the PEC lane's noisy-OR and the material edge average "
        f"(#1210) sum over the same four incident cells. A value near "
        f"{_analytic(H - DX) / 1e9:.4f} GHz is the pre-#1210 cell-owned rule, "
        f"one cell of slab missing.")
    assert rel_an < 5e-3, (
        f"the sigma slab reads {f_sigma / 1e9:.4f} GHz against the analytic "
        f"{_analytic(H) / 1e9:.4f} GHz for an {(L - H) * 1e3:.0f} mm cavity "
        f"(relative {rel_an:.2e}).")
