"""A series RLC load on a line reads its own closed-form impedance (#1163).

The physics
-----------
A one-cell-wide parallel-plate line (PEC plates on z, magnetic walls on y and
behind the port) carries an exact TEM wave with Zc = eta0 * h / w = eta0 for a
one-cell gap and width. A lumped port (Zref = Zc) drives it at node 1; a
series R + C, R + L or R + L + C sits across the gap 120 cells away, with the
magnetic end wall's zeroed half-cell right behind it, so nothing follows the
load. The load's own reflection is then ``Gamma_L = (Z_L - Zc)/(Z_L + Zc)``
with ``Z_L = R + j*w*L + 1/(j*w*C)``, and the element must reproduce it.

The reading does not trust the port or the lattice: a short (PEC) placed on
the SAME node as the load gives ``S11_short = -exp(-2j*beta*L)``, so
``Gamma_L = -S11_load / S11_short`` divides out the line, the port and the
numerical dispersion. ``Z_L = Zc (1 + Gamma_L)/(1 - Gamma_L)``.

The bar is +-5 % of |Z_L| on every bin, for the real and the imaginary part
separately (pre-declared for #1163). The extraction's own error on this
fixture, measured with a plain folded resistor where nothing but the
extraction can be wrong, is 3.1 % of |Z| at worst on these ten bins (2 % over
the 91-bin band of the lumped/wire battery, #1215).

Before #1163 the series element's current was taken from the field before
the element had acted and subtracted afterwards, which removes the edge's own
impedance d/(D0*A) = 215 ohm from R: 300 ohm + 1 pF read 84.7 ohm + j*X_C, and
50 ohm (a negative resistance) did not return a finite S11.

Geometry and extraction follow ``scripts/diagnostics/lumped_wire_chain_battery_measure.py``
(lumped kind, 250 um rung) and the #1163 measurements; this file keeps its own
copy of the few lines it needs so the gate does not depend on a script.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

ETA0 = 376.730313668
DX = 0.25e-3
N_LINE = 120                 # cells from the port node to the load node
PORT_NODE = 1
LOAD_NODE = PORT_NODE + N_LINE
FREQS = np.linspace(1e9, 10e9, 10)
NUM_PERIODS = 40.0
BAR = 0.05


def _channel(n_nodes: int, x_hi: str) -> Simulation:
    sim = Simulation(
        freq_max=10e9, domain=((n_nodes - 1) * DX, DX, DX), dx=DX,
        boundary=BoundarySpec(x=Boundary(lo="pmc", hi=x_hi),
                              y=Boundary(lo="pmc", hi="pmc"),
                              z=Boundary(lo="pec", hi="pec")))
    sim.add_port(position=(PORT_NODE * DX, 0.0, 0.0), component="ez",
                 impedance=ETA0, waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    return sim


def _s11(sim: Simulation) -> np.ndarray:
    res = sim.forward(port_s11_freqs=jnp.asarray(FREQS), num_periods=NUM_PERIODS,
                      skip_preflight=True)
    return np.asarray(res.s_params).reshape(-1)


@pytest.fixture(scope="module")
def s11_short():
    # PEC on the x_hi node zeroes the tangential E there: the short sits ON
    # node LOAD_NODE, the same node the loads below occupy.
    sim = _channel(LOAD_NODE + 1, "pec")
    grid = sim._build_grid()
    assert int(grid.shape[0]) - 1 == LOAD_NODE, "the short is not on the load node"
    assert "x_hi" in (grid.pec_faces or set())
    s = _s11(sim)
    assert np.all(np.isfinite(s))
    assert np.allclose(np.abs(s), 1.0, atol=1e-4), "a lossless short must reflect fully"
    return s


@pytest.mark.parametrize("r_ohm,l_h,c_f", [
    (300.0, 0.0, 1e-12),        # series R + C
    (300.0, 1e-9, 1e-12),       # series R + L + C (resonance at 5.03 GHz)
    (300.0, 1e-9, 0.0),         # series R + L
    (50.0, 0.0, 1e-12),         # R below d/(D0*A): used to be a negative resistance
], ids=["RC-300", "RLC-300", "RL-300", "RC-50"])
def test_series_load_reads_its_closed_form(s11_short, r_ohm, l_h, c_f):
    sim = _channel(LOAD_NODE + 2, "pmc")
    sim.add_lumped_rlc(position=(LOAD_NODE * DX, 0.0, 0.0), component="ez",
                       R=r_ohm, L=l_h, C=c_f, topology="series")
    grid = sim._build_grid()
    el = sim._lumped_rlc[0]
    assert tuple(int(v) for v in grid.position_to_index(el.position)) == (LOAD_NODE, 0, 0)
    assert (el.R, el.L, el.C, el.topology) == (r_ohm, l_h, c_f, "series")
    assert "x_hi" in (grid.pmc_faces or set())

    s = _s11(sim)
    assert np.all(np.isfinite(s)), f"series R={r_ohm} L={l_h} C={c_f}: non-finite S11"
    gamma = -s / s11_short
    z_meas = ETA0 * (1 + gamma) / (1 - gamma)

    w = 2 * math.pi * FREQS
    z_true = r_ohm + (1j * w * l_h if l_h else 0.0) + (1.0 / (1j * w * c_f) if c_f else 0.0)
    tol = BAR * np.abs(z_true)
    bad_re = np.abs(z_meas.real - z_true.real) > tol
    bad_im = np.abs(z_meas.imag - z_true.imag) > tol
    rows = "\n".join(
        f"  {f / 1e9:5.1f} GHz  Z_meas {zm.real:9.2f} {zm.imag:+9.2f}j   "
        f"closed form {zt.real:9.2f} {zt.imag:+9.2f}j   tol {t:6.2f}"
        for f, zm, zt, t in zip(FREQS, z_meas, z_true, tol))
    assert not (bad_re.any() or bad_im.any()), (
        f"series R={r_ohm} L={l_h} C={c_f}: {int(bad_re.sum())} bin(s) off in Re and "
        f"{int(bad_im.sum())} in Im by more than {BAR:.0%} of |Z|\n{rows}")
