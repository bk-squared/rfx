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

Gate (contract G1, log-space with the spectral-resolution floor):
``max(|f_f0 - f_pec|, df) <= max(0.1*FWHM_loss, df)`` per mode. On this
CPU-sized fixture the copper-loss FWHM is far below the window-limited
resolution, so the floor df binds on both sides — the gate then reads
"resonances agree to spectral resolution", which the f0 operator passes
by ~40x margin and the pre-fix realization fails by ~3.5x.
"""

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

# measured provenance (2026-08-19); the regression assertions below allow
# small float drift around these, the GATE is computed live per contract
MEASURED_PEC_MODES = (24.5646e9, 28.1318e9)


def _build(mode):
    sim = Simulation(freq_max=40e9, domain=DOMAIN, dx=DX,
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
        # Reconstruct the PRE-#677 realization: fold each sheet into
        # materials.sigma as a full-cell slab (sigma_eff = 1/(Rs0*d_dual),
        # exactly the emitted spec's sigma_sheet) and strip the operator
        # ctx. This is the G1 NEGATIVE CONTROL — a realization the gate
        # must FAIL, or the gate is measuring nothing.
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


@pytest.mark.slow_physics
def test_g1_resonance_position_ab():
    # --- assembly-identity witness: same cells both ways -----------------
    sim_f0 = _build("f0")
    grid = sim_f0._build_nonuniform_grid()
    specs = []
    assemble_materials_nu(sim_f0, grid, sheet_specs=specs)
    f0_layers = sorted(
        int(k) for sp in specs
        for k in {int(i[2]) for i in np.argwhere(np.asarray(sp.mask))})
    sim_pec = _build("pec")
    _, _, _, pec_mask = assemble_materials_nu(sim_pec, grid)
    pec_layers = sorted(
        {int(i[2]) for i in np.argwhere(np.asarray(pec_mask))})
    assert f0_layers == pec_layers == sorted((K_GND, K_P1, K_P2)), (
        f0_layers, pec_layers)

    # --- three realizations, identical processing ------------------------
    runs = {}
    for mode in ("pec", "f0", "prefix"):
        _, _, dt, ts = _run(mode)
        runs[mode] = (_peaks(ts, dt), ts, dt)

    (pk_pec, df), _, _ = runs["pec"]
    base = sorted(p[0] for p in sorted(pk_pec, key=lambda p: -p[1])[:2])
    assert len(base) == 2
    # provenance pin: the fixture's PEC modes stay where they were measured
    for b, m in zip(base, MEASURED_PEC_MODES):
        assert abs(b - m) <= 2 * df, (b, m)

    def residuals(mode):
        (pk, _), _, _ = runs[mode]
        out = []
        for b in base:
            fm = min(pk, key=lambda p: abs(p[0] - b))[0]
            out.append(abs(fm - b))
        return out

    # --- the G1 gate (df floor binds on both sides; log-space compare) ---
    # FWHM_loss for copper sheets is far below the window-limited
    # resolution on this fixture, so gate = df.
    gate = df
    res_f0 = residuals("f0")
    for r in res_f0:
        assert max(r, df) <= max(gate, df) + 1e-6, (res_f0, df)

    # --- NEGATIVE CONTROL: the pre-fix realization must FAIL ------------
    res_prefix = residuals("prefix")
    for r in res_prefix:
        assert r > gate, (
            f"pre-fix slab realization PASSED the resonance gate "
            f"(residuals {res_prefix}, gate {gate:.3e}) — the gate is "
            f"measuring nothing")
    # and it fails big (the original evidence was 1.01/3.22 GHz; this
    # public rebuild measured ~1.19/1.13 GHz)
    assert min(res_prefix) > 0.5e9, res_prefix
