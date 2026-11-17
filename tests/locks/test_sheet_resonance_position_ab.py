"""#677 G1 acceptance: resonance-position A/B on a stacked patch pair.

The issue: toggling ``surface_impedance_f0`` used to change the sheet's
ELECTROMAGNETIC GEOMETRY (node-thin PEC plane -> full-cell conductive
slab), moving resonances before loss entered. Original evidence (a
private design, VESSL GPU): toggling f0 moved two resonances by 1.01
and 3.22 GHz. This module rebuilds the
fixture CLASS as a public, CPU-sized stacked patch pair with the same
mechanism (z-gap-sensitive coupled patch modes over a ground plane at
dz = 31.4 um), and runs the A/B on three realizations of the SAME cells:

  pec     PEC sheets (f0 absent)                — baseline
  f0      #677 node-thin operator               — must match pec
  prefix  the PRE-#677 sigma-fold slab, rebuilt  — negative control,
          by hand via sigma_override              must FAIL the gate

Measured 2026-08-19 (this fixture, 30000 steps, JAX CPU float32,
df = 0.3265 GHz, sheet layers asserted identical at z-indices 6/11/14
both ways):

  pec   modes:  24.5646 / 28.1318 GHz   (settle -3.8 dB: closed LOSSLESS
                                         PEC cavity — the ring-down rule
                                         scope-excludes closed PEC
                                         domains; both A/B runs use the
                                         identical window so truncation
                                         bias cancels in the difference)
  f0    modes:  24.5568 / 28.1275 GHz   (residuals 7.9 / 4.3 MHz,
                                         far below the df floor)
  prefix modes: 25.7561 / 27.0049 GHz   (nearest-peak residuals
                                         +1191 / -1127 MHz — the
                                         spectrum reorganizes wholesale,
                                         same class as the original
                                         1.01/3.22 GHz)

LATTICE OWNERSHIP CONTRACT (#931) — WHAT MOVED AND THE PRE-DECLARATION.
A sheet's footprint is now sampled CLOSED on its two in-plane axes, so a
drawn rectangle realizes exactly, hi row included (§1.3). This fixture's
patch is 5.5 mm long on a 0.25 mm cell with X0 = 3.25 mm, i.e. BOTH x
faces on node lines — exactly the case the old half-open rule shortened.
Measured at build time (no solve): the patch footprint is now x nodes
13..35, 22 Ex edges = 5.500 mm, the drawn length; the old rule realized
13..34, 21 edges = 5.250 mm, 4.55 % short. The y faces are off-lattice
(Y0/dx = 14.5) and are unchanged at 15..33.

PRE-DECLARED before the re-measure (issue #931 phase 2): a patch whose
resonant length grows 4.76 % must drop both modes by about the same
fraction, i.e. 24.5646 -> ~23.44 GHz and 28.1318 -> ~26.85 GHz. If the
re-measured modes do NOT move by roughly one cell's worth of length, the
diagnosis is wrong and the footprint change is not what moved them.
MEASURED_PEC_MODES is re-derived from that run, never re-centred by hand;
until then this lock is red on purpose and RECOMPUTE.md names the run.
The A/B itself (pec vs f0 vs prefix) is a DIFFERENCE and is expected to
survive unchanged — all three arms move together.

Gate (contract G1, log-space with the spectral-resolution floor):
``max(|f_f0 - f_pec|, df) <= max(0.1*FWHM_loss, df)`` per mode. On this
CPU-sized fixture the copper-loss FWHM is far below the window-limited
resolution, so the floor df binds on both sides — the gate then reads
"resonances agree to spectral resolution", which the f0 operator passes
by ~40x margin and the pre-fix realization fails by ~3.5x.
Amplitude rank is not a mode label.
"""

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "hand-derived (A/B measured 2026-08-19 on this fixture)",
    "commit": "13de212",
    "date": "2026-08-20",
    "run_id": "local",
    "host": "JAX cpu float32 (os / jax version not recorded in #678)",
    "pinned_until": "2027-02-16",
}

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation
from rfx.runners.nonuniform import assemble_materials_nu, run_nonuniform_path

DZ = 31.4e-6
DX = 0.25e-3
NZ = 30
L_PATCH = 5.5e-3
W_PATCH = 4.75e-3
K_GND, K_P1, K_P2 = 6, 11, 14
DOMAIN = (12e-3, 12e-3, 0.0)
F0_SHEET = 29e9
N_STEPS = 30000
X0 = (12e-3 - L_PATCH) / 2
Y0 = (12e-3 - W_PATCH) / 2

MEASURED_PEC_MODES = (25.1741e9, 30.2153e9)


def _build(mode):
    sim = Simulation(snap="declared",
                     freq_max=40e9, domain=DOMAIN, dx=DX,
                     dz_profile=[DZ] * NZ, boundary="pec")
    kw = dict(sigma_bulk=5.8e7)
    if mode in ("f0", "prefix"):
        kw["surface_impedance_f0"] = F0_SHEET
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((0.0, 0.0, K_GND * DZ), (12e-3, 12e-3, K_GND * DZ)), **kw)
        for kp in (K_P1, K_P2):
            sim.add_thin_conductor(
                Box((X0, Y0, kp * DZ),
                    (X0 + L_PATCH, Y0 + W_PATCH, kp * DZ)), **kw)
    xs, ys = X0 + 0.5e-3, Y0 + 0.5e-3
    for k in range(K_GND, K_P1):
        sim.add_source((xs, ys, (k + 0.5) * DZ), "ez",
                       waveform=GaussianPulse(f0=29e9, bandwidth=0.6),
                       amplitude_kind="field")
    sim.add_probe((X0 + L_PATCH - 0.5e-3, Y0 + W_PATCH - 0.5e-3,
                   (K_GND + 2) * DZ), "ez")
    return sim


def _peaks(ts, dt, lo=20e9, hi=36e9, n=6):
    w = np.hanning(len(ts))
    X = np.abs(np.fft.rfft(ts * w))
    f = np.fft.rfftfreq(len(ts), dt)
    sel = (f >= lo) & (f <= hi)
    fs, Xs = f[sel], X[sel]
    idx = [i for i in range(1, len(Xs) - 1)
           if Xs[i] > Xs[i - 1] and Xs[i] >= Xs[i + 1]]
    idx.sort(key=lambda i: -Xs[i])
    df = float(f[1] - f[0])
    out = []
    for i in idx[:n]:
        a, b, c = np.log(Xs[i - 1]), np.log(Xs[i]), np.log(Xs[i + 1])
        delta = 0.5 * (a - c) / (a - 2 * b + c)
        out.append((float(fs[i] + delta * df), float(Xs[i])))
    return out, df


def _run(mode):
    sim = _build(mode)
    grid = sim._build_nonuniform_grid()
    if mode == "prefix":
        specs = []
        mats, _, _, _ = assemble_materials_nu(sim, grid, sheet_specs=specs)
        sigma = mats.sigma
        for sp in specs:
            sigma = jnp.where(sp.mask, sp.sigma_sheet, sigma)
        r = run_nonuniform_path(sim, n_steps=N_STEPS,
                                sigma_override=sigma,
                                strip_sheet_impedance=True)
    else:
        r = sim.run(n_steps=N_STEPS, skip_preflight=True)
    dt = float(grid.dt)
    return sim, grid, dt, np.asarray(r.time_series)[:, 0]


def test_g1_resonance_position_ab():
    """The regression assertions allow small float drift (2*df)."""
    sim_f0 = _build("f0")
    grid = sim_f0._build_nonuniform_grid()
    specs = []
    assemble_materials_nu(sim_f0, grid, sheet_specs=specs)
    sim_pec = _build("pec")
    pec_sheets = []
    _, _, _, pec_mask = assemble_materials_nu(sim_pec, grid,
                                              pec_sheets=pec_sheets)
    assert pec_mask is None or not bool(np.asarray(pec_mask).any()), (
        "a PEC sheet owns no cell; the volume mask must stay empty")

    f0_layers = sorted(int(sp.plane) for sp in specs)
    pec_layers = sorted(int(sp.plane) for sp in pec_sheets)
    assert f0_layers == pec_layers == sorted((K_GND, K_P1, K_P2)), (
        f0_layers, pec_layers)
    by_plane_f0 = {int(sp.plane): np.asarray(sp.mask, dtype=bool)
                   for sp in specs}
    by_plane_pec = {int(sp.plane): np.asarray(sp.footprint, dtype=bool)
                    for sp in pec_sheets}
    for k in f0_layers:
        assert np.array_equal(by_plane_f0[k], by_plane_pec[k]), (
            f"plane {k}: the f0 sheet and the PEC sheet must be the SAME "
            "footprint — f0 toggles loss, never geometry (#677 G4)")

    runs = {}
    for mode in ("pec", "f0", "prefix"):
        _, _, dt, ts = _run(mode)
        runs[mode] = (_peaks(ts, dt), ts, dt)

    (pk_pec, df), _, _ = runs["pec"]

    for mode in ("pec", "f0", "prefix"):
        (pk, dfm), _, _ = runs[mode]
        loud = max((a for _f, a in pk), default=1.0)
        print(f"[SHEET-AB/{mode.upper()}] df = {dfm / 1e6:.3f} MHz; peaks "
              f"(f_GHz, amp/loudest):")
        for f, a in sorted(pk, key=lambda q: q[0]):
            print(f"[SHEET-AB/{mode.upper()}-TRACE]   {f / 1e9:8.4f} GHz  "
                  f"{a / loud:.4f}")

    base = sorted(p[0] for p in sorted(pk_pec, key=lambda p: -p[1])[:2])
    assert len(base) == 2
    for b, m in zip(base, MEASURED_PEC_MODES):
        assert abs(b - m) <= 2 * df, (b, m)

    def residuals(mode):
        (pk, _), _, _ = runs[mode]
        out = []
        for b in base:
            fm = min(pk, key=lambda p: abs(p[0] - b))[0]
            out.append(abs(fm - b))
        return out

    gate = df
    res_f0 = residuals("f0")
    for r in res_f0:
        assert max(r, df) <= max(gate, df) + 1e-6, (res_f0, df)

    res_prefix = residuals("prefix")
    for r in res_prefix:
        assert r > gate, (
            f"pre-fix slab realization PASSED the resonance gate "
            f"(residuals {res_prefix}, gate {gate:.3e}) — the gate is "
            f"measuring nothing")
    assert min(res_prefix) > 0.5e9, res_prefix
