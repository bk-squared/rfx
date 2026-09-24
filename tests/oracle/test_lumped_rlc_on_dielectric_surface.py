"""A lumped element on a dielectric surface realizes its declared value (#1163).

The physics
-----------
A plane wave in a one-cell periodic cross-section (every x-directed edge of
the plane z = z_s loaded by the same lumped element, so the elements form a
uniform sheet of impedance Z_L per square on a cubic cell) meets a
dielectric half-space eps2 that starts AT the sheet. The field at the sheet,
divided by the field an empty vacuum run has there, is the transmission

    T = 2 Z_par / (Z_par + eta0),   Z_par = eta2 || Z_L,   eta2 = eta0/sqrt(eps2)

so Z_L is read back from T without any rfx helper: Z_par = T eta0 / (2 - T),
Z_L = Z_par eta2 / (eta2 - Z_par). Only the first 520 steps are kept, which
excludes every reflection from the domain ends (a time gate).

The element's edge sits on the eps 1 | eps2 interface, so its own
permittivity is the four-cell mean (1 + eps2)/2 while its cell reads eps2.
An element that reads its update denominator D0 from the cell rather than
the edge realizes the wrong value: a parallel 2 nH read 3.2 nH (40.4 ohm
against 25.3 ohm at 2 GHz) on an eps 1|4 surface before #1163 took D0 from
the edge for the parallel topology too.

Geometry and extraction follow the #1163 review's sheet script; the band is
1-8 GHz (37 cells per wavelength at 8 GHz). A folded 100 ohm on the same
surface is the extraction's own witness (it must read 100 ohm, a value no
ADE is involved in).

MEASURED over 1-8 GHz with the edge D0: folded 100 ohm within 0.35 %;
parallel 2 nH reactance within 0.74 % (surface) and 0.84 % (vacuum) before
#1245, 0.16 % on both after it; series 100 ohm + 1 nH within 0.43 % of |Z|.
With the single-cell D0 restored: the
parallel 2 nH off by 60 % (21.09 against 13.18 ohm at 1.05 GHz) and the
series element reading 160 ohm; the vacuum row unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

ETA0 = 376.730313668
DX = 1e-3
LZ = 0.4
N_STEPS = 600
GATE = 520
N_FFT = 6000
BAND = (1e9, 8e9)
EPS2 = 4.0
#: Sheet and source planes, in cells from the domain's z = 0 (16 CPML cells
#: sit below it, so the nodes are 236 and 176).
K_SHEET = 220
K_SOURCE = 160


def _sheet_run(load, eps2):
    sim = Simulation(
        freq_max=10e9, domain=(DX, DX, LZ), dx=DX, cpml_layers=16,
        boundary=BoundarySpec(x=Boundary(lo="periodic", hi="periodic"),
                              y=Boundary(lo="periodic", hi="periodic"),
                              z=Boundary(lo="cpml", hi="cpml")))
    grid = sim._build_grid()
    nx, ny = int(grid.shape[0]), int(grid.shape[1])
    z_s, z_src = K_SHEET * DX, K_SOURCE * DX
    if eps2 is not None:
        # Wider than the declared cross-section on purpose: the periodic axes
        # realize TWO cells for a one-cell domain (the #1223 boundary
        # campaign records it), and a Box of the declared width fills only
        # one of the four cells. Asserted below on every cell.
        sim.add_material("half_space", eps_r=eps2)
        sim.add(Box((-2 * DX, -2 * DX, z_s), (4 * DX, 4 * DX, LZ)),
                material="half_space")
    wf = GaussianPulse(f0=8e9, bandwidth=1.0 / (math.pi * 25e-12 * 8e9), cutoff=5.0)
    for i in range(nx):
        for j in range(ny):
            sim.add_source((i * DX, j * DX, z_src), "ex", waveform=wf,
                           amplitude_kind="current")
            if load is not None:
                load(sim, (i * DX, j * DX, z_s))
    sim.add_probe((0.0, 0.0, z_s), "ex")
    k_s = int(grid.position_to_index((0.0, 0.0, z_s))[2])
    if load is not None:
        # Realized, not declared: one element on every x edge of the plane.
        idx = sorted(tuple(int(v) for v in grid.position_to_index(e.position))
                     for e in sim._lumped_rlc)
        assert idx == sorted((i, j, k_s) for i in range(nx) for j in range(ny))
    if eps2 is not None:
        eps = np.asarray(sim._build_materials(grid)[0].eps_r)
        assert np.all(eps[:, :, k_s - 1] == 1.0) and np.all(eps[:, :, k_s:k_s + 150] == eps2), (
            "the half-space does not fill the periodic cell from the sheet plane up")
    ts = np.asarray(sim.run(n_steps=N_STEPS, skip_preflight=True).time_series,
                    dtype=np.float64).reshape(-1)
    assert np.all(np.isfinite(ts))
    return ts, float(grid.dt)


@pytest.fixture(scope="module")
def incident():
    return _sheet_run(None, None)


def _z_load(ts, incident, eps2):
    ref, dt = incident
    a, b = ts.copy(), ref.copy()
    a[GATE:] = 0.0
    b[GATE:] = 0.0
    f = np.fft.rfftfreq(N_FFT, dt)
    t = np.fft.rfft(a, N_FFT) / np.fft.rfft(b, N_FFT)
    eta2 = ETA0 / math.sqrt(eps2)
    z_par = t * ETA0 / (2.0 - t)
    z_l = z_par * eta2 / (eta2 - z_par)
    sel = (f >= BAND[0]) & (f <= BAND[1])
    return f[sel], z_l[sel]


def test_folded_resistor_on_the_surface_reads_its_value(incident):
    """The extraction's own witness: a folded resistor needs no ADE and no D0."""
    ts, _ = _sheet_run(lambda s, p: s.add_lumped_rlc(
        position=p, component="ex", R=100.0, topology="parallel"), EPS2)
    f, z = _z_load(ts, incident, EPS2)
    err = np.abs(z - 100.0) / 100.0
    assert err.max() < 0.01, f"folded 100 ohm read {z[np.argmax(err)]:.2f} ohm at {f[np.argmax(err)] / 1e9:.2f} GHz"


#: A lumped inductor is lossless: |Re Z_L| and |Im Z_L - w L| within 1 % of
#: w L on every bin (#1245).
BAR_L = 0.01


@pytest.mark.parametrize("eps2, l_h", [(EPS2, 2e-9), (1.0, 2e-9), (1.0, 5e-9)],
                         ids=["eps-1-4-surface-2nH", "vacuum-2nH", "vacuum-5nH"])
def test_parallel_inductor_is_lossless_and_reads_its_reactance(incident, eps2, l_h):
    """A pure parallel inductor: Re Z_L = 0 and Im Z_L = w L, each within 1 %
    of w L on every bin of 1-8 GHz.

    The inductor update it replaced applied I^{n+1} over the step n -> n+1
    (backward Euler, half a step late against the centred field update) and
    so carried a series resistance w^2 L dt: 9.5 ohm at 8 GHz for 2 nH on
    these 1 mm cells, 9.5 % of w L (Q ~ 10). #1245 solves the inductor with
    its edge field in one trapezoidal step.

    MEASURED with the trapezoidal inductor (1-8 GHz): |Re Z_L|/wL <= 7.8e-7 in
    vacuum and 0.56 % on the surface, |Im Z_L - wL|/wL <= 0.16 %; with the old
    update restored, 9.5 % and 0.84 %. The surface
    Re is the extraction's own floor, not the element: at the eps 1|4 interface
    the vacuum and dielectric sides of the discrete jump condition carry
    different dispersion factors cos(b1) and cos(b2), which the continuous
    inversion reads as a spurious conductance (cos b2/cos b1 - 1)/eta2
    (-5.6e-5 S at 8 GHz) and so as Re Z_L = G |Z_L|^2. That closed form
    reproduces the surface readings of the folded 100 ohm and of both
    inductor updates to 1e-6 (#1245). It grows with |Z_L|^2, so the 5 nH runs
    in vacuum only: on the surface it reads 1.4 % of wL at 8 GHz.
    """
    ts, _ = _sheet_run(lambda s, p: s.add_lumped_rlc(
        position=p, component="ex", L=l_h, topology="parallel"), eps2)
    f, z = _z_load(ts, incident, eps2)
    x_true = 2.0 * math.pi * f * l_h
    loss = np.abs(z.real) / x_true
    err = np.abs(z.imag - x_true) / x_true
    k, m = int(np.argmax(loss)), int(np.argmax(err))
    assert loss.max() <= BAR_L, (
        f"parallel {l_h * 1e9:.0f} nH on eps2={eps2}: Re Z_L {z.real[k]:.3f} ohm at "
        f"{f[k] / 1e9:.2f} GHz, {loss[k]:.2%} of w L = {x_true[k]:.2f} ohm (bar {BAR_L:.0%})")
    assert err.max() <= BAR_L, (
        f"parallel {l_h * 1e9:.0f} nH on eps2={eps2}: reactance {z.imag[m]:.2f} ohm "
        f"against {x_true[m]:.2f} ohm at {f[m] / 1e9:.2f} GHz ({err[m]:.2%}; bar {BAR_L:.0%})")


def test_series_element_on_the_surface_reads_its_impedance(incident):
    """Series 100 ohm + 1 nH on the eps 1|4 surface: Re and Im of Z_L within
    5 % of |Z_L| on every bin (the #1163 series bar). With the single-cell D0
    the same element read 160 ohm."""
    r_ohm, l_h = 100.0, 1e-9
    ts, _ = _sheet_run(lambda s, p: s.add_lumped_rlc(
        position=p, component="ex", R=r_ohm, L=l_h, topology="series"), EPS2)
    f, z = _z_load(ts, incident, EPS2)
    z_true = r_ohm + 2j * math.pi * f * l_h
    tol = 0.05 * np.abs(z_true)
    bad = (np.abs(z.real - z_true.real) > tol) | (np.abs(z.imag - z_true.imag) > tol)
    k = int(np.argmax(np.maximum(np.abs(z.real - z_true.real), np.abs(z.imag - z_true.imag)) / tol))
    assert not bad.any(), (
        f"series {r_ohm:.0f} ohm + {l_h * 1e9:.0f} nH on eps2={EPS2}: {int(bad.sum())} bin(s) "
        f"off; worst {z[k].real:.2f}{z[k].imag:+.2f}j against {z_true[k].real:.2f}"
        f"{z_true[k].imag:+.2f}j at {f[k] / 1e9:.2f} GHz")
