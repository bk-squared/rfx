"""W3.2 — Lossy-material validation oracle (E1->E2).

Validates rfx's conductive-loss FDTD against exact analytics:

  (1) PRIMARY — plane-wave attenuation alpha(f) of a homogeneous conductive
      medium vs the exact analytic
          alpha(w) = (w/c) * |Im( sqrt( eps_r - j*sigma/(w*eps0) ) )|
      (e^{+jwt} convention: loss => Im(eps)<0 => Im(n)<0).  Measured by
      sampling the per-frequency field magnitude at several interior depths of
      a thick lossy region and log-linear-fitting the spatial decay of the
      forward wave.  No probe-FFT (time-window artifact, forbidden); the
      per-frequency |E(x,f)| comes from add_dft_plane_probe DFT accumulators.

  (2) SECONDARY — a PEC cavity fully filled with a low-loss dielectric has a
      dielectric-limited quality factor Q ~= 1/tan(delta).  Q is measured by
      Harminv ring-down and compared to 1/tan(delta) at the measured resonance.

R5 (curves recorded before the gates were set):
docs/research_notes/20260702_w32_lossy_material.md.  Measured margins are far
inside the gates (alpha mean rel-err 0.5-0.9%; Q rel-err <0.2%); the gates are
set at 5% (alpha, mirroring the dispersive-Fresnel oracle) and 5% (Q, generous
vs the 0.12% measured — Harminv Q is noisier than a decay-rate fit).

Marked slow_physics (opt-in release-tag physics gate).
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.harminv import harminv

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
EPS_R = 4.0

# --- attenuation-case geometry (thick lossy region, planes deep inside) ---
_DX = 0.5e-3
_DOM_X = 340e-3
_DOM_Y = 5e-3
_FREQ_MAX = 18e9
_F0 = 10e9
_BW = 0.7
_X_ENTRY = 30e-3
_X_EXIT = 320e-3
_PLANES = np.array([45e-3, 55e-3, 65e-3, 75e-3, 85e-3, 95e-3])
_FREQS_A = np.linspace(6e9, 14e9, 9)

ALPHA_GATE = 0.05          # mean |alpha_fdtd/alpha_analytic - 1| over the band
ALPHA_FIT_RESID_GATE = 0.05   # log-linear fit RMS residual: forward-wave purity
Q_GATE = 0.05             # |Q_meas / (1/tan(delta)) - 1| at the measured resonance


def _eps_c(freqs, sigma):
    w = 2.0 * np.pi * np.asarray(freqs)
    return EPS_R - 1j * sigma / (w * EPS0)      # e^{+jwt}: loss -> Im<0


def _alpha_analytic(freqs, sigma):
    w = 2.0 * np.pi * np.asarray(freqs)
    n = np.sqrt(_eps_c(freqs, sigma))           # principal sqrt -> Im(n)<0
    return (w / C0) * np.abs(np.imag(n))


def _measure_alpha(sigma):
    """Return (alpha_fit(f), fit_resid(f), alpha_analytic(f)) for the band."""
    sim = Simulation(freq_max=_FREQ_MAX, domain=(_DOM_X, _DOM_Y, _DX), dx=_DX,
                     boundary="cpml", cpml_layers=20, mode="2d_tmz")
    sim.add_material("lossy", eps_r=EPS_R, sigma=sigma)
    x_hi = _X_EXIT - _DX / 2.0
    sim.add(Box((_X_ENTRY, -1.0, -1.0), (x_hi, 1.0, 1.0)), material="lossy")
    sim.add_tfsf_source(f0=_F0, bandwidth=_BW, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    for i, xc in enumerate(_PLANES):
        sim.add_dft_plane_probe(axis="x", coordinate=float(xc), component="ez",
                                freqs=_FREQS_A, name=f"p{i}")
    res = sim.run(until_decay=1e-4, decay_monitor_component="ez",
                  decay_monitor_position=(70e-3, _DOM_Y / 2.0, 0.0))
    # |E(x,f)| = mean over y of |DFT accumulator|
    mag = np.zeros((len(_PLANES), len(_FREQS_A)))
    for i in range(len(_PLANES)):
        acc = np.asarray(res.dft_planes[f"p{i}"].accumulator)  # (nf, ny, nz)
        mag[i] = np.mean(np.abs(acc[:, :, 0]), axis=1)
    logmag = np.log(mag)
    A = np.vstack([_PLANES, np.ones_like(_PLANES)]).T
    a_fit = np.zeros(len(_FREQS_A))
    resid = np.zeros(len(_FREQS_A))
    for fi in range(len(_FREQS_A)):
        sol, *_ = np.linalg.lstsq(A, logmag[:, fi], rcond=None)
        a_fit[fi] = -sol[0]
        resid[fi] = np.sqrt(np.mean((logmag[:, fi] - A @ sol) ** 2))
    return a_fit, resid, _alpha_analytic(_FREQS_A, sigma)


@pytest.mark.slow_physics
@pytest.mark.parametrize("tand", [0.1, 0.3])   # low + moderate loss
def test_lossy_slab_attenuation(tand):
    sigma = tand * (2.0 * np.pi * _F0) * EPS0 * EPS_R   # tan(delta) at F0
    a_fit, resid, a_an = _measure_alpha(sigma)

    assert np.all(np.isfinite(a_fit)), "alpha_fdtd has non-finite entries"
    assert np.all(a_fit > 0), "attenuation must be positive for a lossy medium"
    # forward-wave purity: the interior field is a single decaying exponential
    # (a standing wave from a leaking far boundary would inflate the fit RMS).
    assert np.max(resid) < ALPHA_FIT_RESID_GATE, (
        f"log-linear fit residual {np.max(resid):.4f} too large — "
        "interior wave is not a clean single exponential (reflection leak?)"
    )
    mean_rel = float(np.mean(np.abs(a_fit / a_an - 1.0)))
    assert mean_rel < ALPHA_GATE, (
        f"tan(delta)={tand}: mean |alpha_fdtd/alpha_analytic - 1| = "
        f"{mean_rel:.4f} exceeds {ALPHA_GATE}"
    )


@pytest.mark.slow_physics
def test_lossy_cavity_Q_vs_tandelta():
    """PEC cavity filled with a low-loss dielectric: Q ~= 1/tan(delta)."""
    tand = 0.1                       # Q ~ 10 -> short ring-down, cheap
    Lx = Ly = 20e-3
    dx = 0.5e-3
    f11 = (C0 / (2.0 * np.sqrt(EPS_R))) * np.sqrt((1.0 / Lx) ** 2 + (1.0 / Ly) ** 2)
    sigma = tand * (2.0 * np.pi * f11) * EPS0 * EPS_R
    fmax = 3.0 * f11                 # keeps >=15 cells/lambda_eff (clears advisory)

    sim = Simulation(freq_max=fmax, domain=(Lx, Ly, dx), dx=dx,
                     boundary="pec", mode="2d_tmz")
    sim.add_material("diel", eps_r=EPS_R, sigma=sigma)
    sim.add(Box((-1.0, -1.0, -1.0), (Lx + 1.0, Ly + 1.0, 1.0)), material="diel")
    # off-centre source + probe couple to the TM11 mode
    sim.add_source((0.3 * Lx, 0.35 * Ly, 0.0), component="ez",
                   waveform=GaussianPulse(f0=f11, bandwidth=0.6))
    sim.add_probe((0.62 * Lx, 0.57 * Ly, 0.0), component="ez")
    res = sim.run(n_steps=9000)

    dt = float(res.grid.dt)
    ez = np.asarray(res.time_series[:, 0])
    modes = harminv(ez, dt, 0.5 * f11, 2.5 * f11, min_Q=2.0)
    assert modes, "Harminv found no cavity mode"

    m0 = max(modes, key=lambda m: m.amplitude)     # dominant TM11
    assert abs(m0.freq / f11 - 1.0) < 0.05, (
        f"dominant mode {m0.freq/1e9:.3f} GHz not near TM11 {f11/1e9:.3f} GHz"
    )
    # 1/tan(delta) at the MEASURED resonance (sigma fixed => tan(delta) is
    # frequency-dependent; evaluate the theory where the mode actually sits).
    tand_at_f = sigma / (2.0 * np.pi * m0.freq * EPS0 * EPS_R)
    q_theory = 1.0 / tand_at_f
    rel = abs(m0.Q / q_theory - 1.0)
    assert rel < Q_GATE, (
        f"Q_meas={m0.Q:.2f} vs 1/tan(delta)={q_theory:.2f} at "
        f"{m0.freq/1e9:.3f} GHz — rel-err {rel:.4f} exceeds {Q_GATE}"
    )
