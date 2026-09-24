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

The continuous inversion has one discrete term of its own on the surface: a
real interface conductance G (:func:`interface_conductance`), which reads
as Re Z = G |Z|^2 on a reactive load (1.4 % of |Z| at 250 ohm, 8 GHz). Every
row is judged with G removed (PI decision, #1245); in vacuum G = 0.

MEASURED over 1-8 GHz with the edge D0, raw -> with G removed: folded
100 ohm within 0.33 % -> 0.23 %; series 100 ohm + 1 nH within 0.46 % ->
0.22 % of |Z|; parallel 2 nH reactance within 0.74 % (surface) and 0.84 %
(vacuum) before #1245, 0.16 % on both after it.
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
C0 = 299792458.0
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


def interface_conductance(f, dt, eps2):
    """The conductance the continuous inversion reads at the eps 1|eps2 plane.

    One-dimensional Yee along z (the fields are uniform across the periodic
    cross-section), sheet on the interface E node, the H nodes either side in
    vacuum and in eps2. With b_m = beta_m dz/2 from each medium's own discrete
    dispersion, sin(b_m) = (dz sqrt(eps_m)/(c dt)) sin(w dt/2), the jump
    condition at the sheet reads (K the sheet current referred to the half
    step)

        E_T [cos b1/eta1 + cos b2/eta2 + K/E_T] = 2 E_i cos b1/eta1,

    and the continuous formula T = 2/eta1 / (1/eta1 + 1/eta2 + 1/Z) inverts that
    to 1/Z = (K/E_T)/cos b1 + G with

        G = (cos b2 / cos b1 - 1) / eta2,

    a real conductance of the extraction, not of the element: -5.6e-5 S at
    8 GHz on 1 mm cells with eps2 = 4, and exactly 0 in vacuum (b2 = b1).
    It reads as Re Z = G |Z|^2 on a reactive load. MEASURED (#1245) as
    Re(1/Z_raw) of four lossless loads on this surface -- folded 0.0796 pF and
    0.32 pF, parallel 2 nH and 5 nH: -5.5976e-5 to -5.5979e-5 S at 7.95 GHz
    against -5.5978e-5 S from this formula, within 7.6e-8 S on every bin of
    1-8 GHz.
    """
    s = DX / (C0 * dt) * np.sin(np.pi * f * dt)
    b1 = np.arcsin(s)
    b2 = np.arcsin(math.sqrt(eps2) * s)
    return (np.cos(b2) / np.cos(b1) - 1.0) / (ETA0 / math.sqrt(eps2))


def _z_load_raw(ts, incident, eps2):
    """Z_L from the continuous inversion alone (no interface term)."""
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


def _z_load(ts, incident, eps2):
    """Z_L with the interface conductance removed (PI decision, #1245): the
    extraction every row is judged on. In vacuum G = 0 and this is the raw
    inversion."""
    f, z_raw = _z_load_raw(ts, incident, eps2)
    g = interface_conductance(f, incident[1], eps2)
    return f, 1.0 / (1.0 / z_raw - g)


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


@pytest.mark.parametrize("eps2, l_h", [(EPS2, 2e-9), (EPS2, 5e-9), (1.0, 2e-9), (1.0, 5e-9)],
                         ids=["eps-1-4-surface-2nH", "eps-1-4-surface-5nH",
                              "vacuum-2nH", "vacuum-5nH"])
def test_parallel_inductor_is_lossless_and_reads_its_reactance(incident, eps2, l_h):
    """A pure parallel inductor: Re Z_L = 0 and Im Z_L = w L, each within 1 %
    of w L on every bin of 1-8 GHz.

    The inductor update it replaced applied I^{n+1} over the step n -> n+1
    (backward Euler, half a step late against the centred field update) and
    so carried a series resistance w^2 L dt: 9.5 ohm at 8 GHz for 2 nH on
    these 1 mm cells, 9.5 % of w L (Q ~ 10). #1245 solves the inductor with
    its edge field in one trapezoidal step.

    Judged on the extraction with the interface conductance removed
    (:func:`interface_conductance`; the folded-capacitor test below is its
    witness at the same |Z|).

    MEASURED with the trapezoidal inductor (1-8 GHz): |Re Z_L|/wL <= 1.0e-6
    (2 nH) and 5.3e-7 (5 nH) on the surface, 7.8e-7 in vacuum; |Im Z_L - wL|/wL
    <= 0.16 % everywhere. The raw inversion on the surface, before the
    interface term is removed, reads 0.56 % (2 nH) and 1.39 % (5 nH) of wL:
    G |Z_L|^2 of the extraction. With the old update restored, Re Z_L is
    9.5 % of wL in vacuum.
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


#: The folded capacitor with the 5 nH row's |Z| at 8 GHz (249.9 ohm against
#: 251.3 ohm): no inductor code, no ADE, only the edge permittivity.
C_WITNESS = 0.0796e-12
#: Re Z_raw = G |Z|^2 per bin, relative to |Z|. MEASURED residual 4.0e-6; the
#: term it checks is 1.4e-2 of |Z| at 8 GHz.
IDENTITY_BAR = 1e-4


def test_folded_capacitor_witnesses_the_interface_term(incident):
    """The interface term belongs to the extraction, not to any element.

    A folded 0.0796 pF on the same surface is lossless and involves no
    inductor code. Its raw reading must carry exactly the loss the interface
    term predicts, Re Z_raw = G |Z|^2 on every bin, and with the term removed
    it must read a lossless capacitor within the inductor rows' 1 % bar.
    MEASURED (1-8 GHz): raw Re Z = -1.398 % of |Z| at 7.95 GHz (the 5 nH row
    reads -1.394 % there); identity residual 4.0e-6 of |Z|; corrected
    |Re Z|/|Z| <= 4.0e-6 and |Im Z - X_C|/|X_C| <= 0.31 %. G read back from
    this load, Re(1/Z_raw), is -5.5977e-5 S at 7.95 GHz against the closed
    form's -5.5978e-5 S (2 nH: -5.5978e-5, 5 nH: -5.5979e-5).
    """
    ts, _ = _sheet_run(lambda s, p: s.add_lumped_rlc(
        position=p, component="ex", C=C_WITNESS, topology="parallel"), EPS2)
    f, z_raw = _z_load_raw(ts, incident, EPS2)
    _, z = _z_load(ts, incident, EPS2)
    g = interface_conductance(f, incident[1], EPS2)
    x_true = -1.0 / (2.0 * math.pi * f * C_WITNESS)
    mag = np.abs(x_true)

    predicted = g * np.abs(z) ** 2
    assert np.max(np.abs(predicted[-1])) / mag[-1] > 1e-2, (
        "the witness no longer exercises the interface term at the band top")
    resid = np.abs(z_raw.real - predicted) / np.abs(z)
    k = int(np.argmax(resid))
    assert resid.max() <= IDENTITY_BAR, (
        f"folded {C_WITNESS * 1e12:.4f} pF on eps2={EPS2}: raw Re Z {z_raw.real[k]:.4f} ohm "
        f"at {f[k] / 1e9:.2f} GHz, the interface term predicts {predicted[k]:.4f} ohm "
        f"(residual {resid[k]:.2e} of |Z|; bar {IDENTITY_BAR:.0e})")

    loss = np.abs(z.real) / mag
    err = np.abs(z.imag - x_true) / mag
    k, m = int(np.argmax(loss)), int(np.argmax(err))
    assert loss.max() <= BAR_L, (
        f"folded {C_WITNESS * 1e12:.4f} pF on eps2={EPS2}: Re Z {z.real[k]:.3f} ohm at "
        f"{f[k] / 1e9:.2f} GHz, {loss[k]:.2%} of |X_C| (bar {BAR_L:.0%})")
    assert err.max() <= BAR_L, (
        f"folded {C_WITNESS * 1e12:.4f} pF on eps2={EPS2}: reactance {z.imag[m]:.2f} ohm "
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
