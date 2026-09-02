"""#677 G2: loss-is-real witness + perturbation-Q band on a patch cavity.

Fixture: single rectangular patch (5.5 x 4.75 mm) over a full ground plane,
dz = 31.4 um cells, sigma_bulk = 1e5 in f0 mode (Rs0 = 1.0700 ohm/sq at
f0 = 29 GHz) so the conductor-loss FWHM is spectrally resolvable on CPU.

Measured 2026-08-19 (120000 steps, df = 0.0816 GHz, JAX CPU float32;
preflight on the fixture class: "All checks passed"):

  f0 run:   f_mode = 24.7530 GHz, FWHM = 0.9339 GHz  -> Q_f0 = 26.5
  PEC run:  f_mode = 24.7673 GHz, FWHM = 0.1339 GHz  (window-limited)
  Q_pred = omega*W/P = 13.9, with W = 1/4 int(eps|E|^2 + mu|H|^2) dV over
  the full stack and P = 1/2 Rs0 sum |dHt|^2 dA from the PEC-run
  tangential-H jump across each sheet plane (both from the same PEC-run
  DFT, so the phasor normalization cancels).

Two teeth, split by what the evidence supports:

* LOSS-IS-REAL (green): FWHM_f0 / FWHM_pec measured 6.98 — the f0 sheet
  broadens the mode far beyond the identical processing window, i.e.
  Q_f0 << Q_PEC, and only when asked (the PEC control is the same cells).
* PERTURBATION BAND (xfail, strict): the contract band
  Q_f0 in [0.7, 1.3] * Q_pred measured Q_f0/Q_pred = 1.905, and 1.885 on
  the stacked-pair variant with a filtered-ring-down Q extractor reading
  33.5 vs spectral 34.4 — a stable ~1.9x across two fixtures and two Q
  extractors. Comparator-first caveat (workspace rule): the Q_pred
  integral chain (DFT-plane energy + H-jump quadrature) has NOT itself
  reproduced a known-Q analytic case, so this red cannot be attributed to
  the operator — the independent free-standing-sheet transmission oracle
  (tests/oracle/test_leontovich_alpha_oracle.py) pins the delivered Rs to +-4.4%
  frequency-flat. Next step is validating the Q_pred comparator on an
  analytic lossy-wall cavity, not touching the operator (R2).
"""

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation
from rfx.core.yee import EPS_0, MU_0
from rfx.materials.thin_conductor import leontovich_rs
from rfx.runners.nonuniform import assemble_materials_nu

DZ = 31.4e-6
DX = 0.25e-3
NZ = 30
L_PATCH = 5.5e-3
W_PATCH = 4.75e-3
K_GND, K_P1 = 6, 11
DOMAIN = (12e-3, 12e-3, 0.0)
F0_SHEET = 29e9
SIGMA_BULK = 1e5
N_STEPS = 120000
X0 = (12e-3 - L_PATCH) / 2
Y0 = (12e-3 - W_PATCH) / 2

_cache = {}


def _build(mode, dft_freqs=None):
    sim = Simulation(freq_max=40e9, domain=DOMAIN, dx=DX,
                     dz_profile=[DZ] * NZ, boundary="pec")
    kw = dict(sigma_bulk=SIGMA_BULK if mode == "f0" else 5.8e7)
    if mode == "f0":
        kw["surface_impedance_f0"] = F0_SHEET
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((0.0, 0.0, K_GND * DZ), (12e-3, 12e-3, K_GND * DZ)), **kw)
        sim.add_thin_conductor(
            Box((X0, Y0, K_P1 * DZ),
                (X0 + L_PATCH, Y0 + W_PATCH, K_P1 * DZ)), **kw)
    xs, ys = X0 + 0.5e-3, Y0 + 0.5e-3
    for k in range(K_GND, K_P1):
        sim.add_source((xs, ys, (k + 0.5) * DZ), "ez",
                       waveform=GaussianPulse(f0=29e9, bandwidth=0.6),
                       amplitude_kind="field")
    sim.add_probe((X0 + L_PATCH - 0.5e-3, Y0 + W_PATCH - 0.5e-3,
                   (K_GND + 2) * DZ), "ez")
    if dft_freqs is not None:
        for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
            for k in range(NZ):
                sim.add_dft_plane_probe(
                    axis="z", coordinate=k * DZ, component=comp,
                    freqs=jnp.asarray(dft_freqs), name=f"{comp}{k}")
    return sim


def _fwhm_and_peak(ts, dt, lo=20e9, hi=27e9):
    w = np.hanning(len(ts))
    X = np.abs(np.fft.rfft(ts * w))
    f = np.fft.rfftfreq(len(ts), dt)
    sel = np.nonzero((f >= lo) & (f <= hi))[0]
    k = int(sel[np.argmax(X[sel])])
    df = float(f[1] - f[0])
    a, b, c = np.log(X[k - 1]), np.log(X[k]), np.log(X[k + 1])
    delta = 0.5 * (a - c) / (a - 2 * b + c)
    half = X[k] / np.sqrt(2.0)
    li = k
    while li > 1 and X[li] > half:
        li -= 1
    ri = k
    while ri < len(X) - 2 and X[ri] > half:
        ri += 1
    fl = f[li] + (half - X[li]) / (X[li + 1] - X[li]) * df
    fr = f[ri - 1] + (half - X[ri - 1]) / (X[ri] - X[ri - 1]) * df
    return float(f[k] + delta * df), float(fr - fl), df


def _measured():
    if "out" in _cache:
        return _cache["out"]
    sim_f0 = _build("f0")
    grid = sim_f0._build_nonuniform_grid()
    dt = float(grid.dt)
    r = sim_f0.run(n_steps=N_STEPS, skip_preflight=True)
    ts = np.asarray(r.time_series)[:, 0]
    f_mode, fw_f0, df = _fwhm_and_peak(ts, dt)

    sim_pec = _build("pec", dft_freqs=(f_mode,))
    r2 = sim_pec.run(n_steps=N_STEPS, skip_preflight=True)
    ts2 = np.asarray(r2.time_series)[:, 0]
    _, fw_pec, _ = _fwhm_and_peak(ts2, dt)

    planes = {name: np.asarray(p.accumulator)[0]
              for name, p in r2.dft_planes.items()}
    dV = DX * DX * DZ
    W = 0.0
    for comp, c0 in (("ex", EPS_0), ("ey", EPS_0), ("ez", EPS_0),
                     ("hx", MU_0), ("hy", MU_0), ("hz", MU_0)):
        for k in range(NZ):
            W += 0.25 * c0 * float(
                np.sum(np.abs(planes[f"{comp}{k}"]) ** 2)) * dV
    rs0 = float(leontovich_rs(F0_SHEET, SIGMA_BULK))
    specs = []
    assemble_materials_nu(sim_f0, grid, sheet_specs=specs)
    dA = DX * DX
    P = 0.0
    for sp in specs:
        m = np.asarray(sp.mask)
        k = sorted({int(i[2]) for i in np.argwhere(m)})[0]
        foot = m[:, :, k]
        for comp in ("hx", "hy"):
            dH = planes[f"{comp}{k}"] - planes[f"{comp}{k - 1}"]
            P += 0.5 * rs0 * float(np.sum(np.abs(dH[foot]) ** 2)) * dA
    q_f0 = f_mode / fw_f0
    q_pred = 2 * np.pi * f_mode * W / P
    _cache["out"] = dict(f_mode=f_mode, fw_f0=fw_f0, fw_pec=fw_pec, df=df,
                         q_f0=q_f0, q_pred=q_pred)
    return _cache["out"]


@pytest.mark.slow_physics
def test_g2_loss_is_real():
    out = _measured()
    # measured 2026-08-19: fw_f0/fw_pec = 6.98; the f0 FWHM must stay far
    # above the identically processed PEC (window-limited) width
    assert out["fw_f0"] / out["fw_pec"] > 2.0, out
    # regression pins on the measured envelope (loose: mode tracking)
    assert abs(out["f_mode"] - 24.753e9) < 0.3e9, out["f_mode"]
    assert abs(out["q_f0"] / 26.5 - 1.0) < 0.25, out["q_f0"]


@pytest.mark.slow_physics
@pytest.mark.xfail(
    strict=True,
    reason="#677 G2 band (2026-08-19): Q_f0/Q_pred measured 1.905 here and "
           "1.885 on the stacked-pair variant — a stable ~1.9x outside the "
           "contract band [0.7, 1.3], reproduced by two fixtures and two Q "
           "extractors (spectral FWHM 34.4 vs filtered ring-down 33.5 on "
           "the pair). Comparator-first: the Q_pred integral chain "
           "(DFT-plane energy + H-jump quadrature) has never reproduced a "
           "known-Q analytic case, while the independent transmission "
           "oracle pins the operator's delivered Rs to +-4.4% "
           "frequency-flat — so validate the Q_pred comparator on an "
           "analytic lossy-wall cavity BEFORE attributing this to the "
           "operator. A future pass must remove this marker explicitly.",
)
def test_g2_perturbation_q_band():
    out = _measured()
    ratio = out["q_f0"] / out["q_pred"]
    assert 0.7 <= ratio <= 1.3, out
