"""T7-E Phase 2 — PMC λ/4 cavity mode-ladder oracle (T7 Phase 2, 2026-04).

Closes critic blocker #5: the shipped PMC tests (tangential-H=0,
dual-boundary Hx sample) prove the PMC hook fires and is a distinct
code path from PEC, but they do not prove PMC is a magnetic wall
(reflection sign convention, quarter-wave resonance).

## Analytic basis

In a 1D-like cavity of length L with PEC at z = L (``E_tan = 0``) and
PMC at z = 0 (``H_tan = 0``, which forces ``∂E_tan/∂z = 0`` at the
wall), the transverse-electric standing-wave modes satisfy

    E_x(z, t) = cos(k_n z) · cos(ω_n t)
    boundary conditions: cos(k_n · L) = 0  =>  k_n · L = (2n + 1) π / 2

giving resonance frequencies

    f_n = c · (2n + 1) / (4 L),     n = 0, 1, 2, ...

So the PEC-PMC ladder is f_0 = c/(4L), f_1 = 3c/(4L), f_2 = 5c/(4L),
spaced at ratio 1 : 3 : 5 : 7. By contrast a PEC-PEC cavity supports

    f_n^{PEC-PEC} = n · c / (2 L),  ladder 1 : 2 : 3 : 4.

The spacing ratio f_1 / f_0 is **3 for PEC-PMC** and **2 for PEC-PEC**;
a pure 2% frequency check on either peak alone cannot distinguish the
two, but the SPACING RATIO does. This test asserts the ladder AND
includes a negative PEC-PEC control.

## Test configuration

- Narrow transverse xy (periodic) to suppress non-axial modes.
- Cavity z-length L locked so f_0 = c/(4L) lies well below the
  source frequency cap and f_1 = 3 f_0 is still covered.
- Gaussian impulse E_x source near the centre of the interior.
- Probe E_x at z = 0.7 L (clear of the n=0 and n=1 nodes).
- Ez DFT over a long run (2048 steps) to resolve the ladder.
- Peak detection finds the two strongest DFT peaks in the
  [0.5 f_0, 4 f_0] band.

## What this oracle pins

1. **Spacing ratio f_1 / f_0 ∈ [2.5, 3.5]**: distinguishes PEC-PMC
   (analytic 3.0) from PEC-PEC (analytic 2.0). Tolerance is wide
   enough to absorb the finite-cavity + discretization frequency
   drift but tight enough to separate the two ladders.
2. **f_0 within 10 % of analytic c/(4L)**: confirms the quarter-wave
   resonance is where PEC-PMC physics predicts.
3. **Negative control**: the same cavity with PMC swapped for PEC
   on z_lo fails the 3.0 spacing ratio (lands at 2.0, PEC-PEC ladder).
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


_C0 = 299_792_458.0
_L_CAVITY = 0.02              # 20 mm z-axis interior
_DX = 0.5e-3                  # 0.5 mm cells → 40 interior cells
_N_STEPS = 2048
_F0_ANALYTIC = _C0 / (4.0 * _L_CAVITY)  # quarter-wave for PEC-PMC


def _run_cavity(z_lo_token: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (freqs, |E(f)|, dt) for the probe spectrum of a
    z-axis cavity with ``z_hi='pec'`` and ``z_lo=z_lo_token``.
    """
    # cpml_layers=0 produces a closed grid — the PMC / PEC walls sit
    # at the grid edge and no CPML is allocated on any axis.
    spec = BoundarySpec(
        x="periodic", y="periodic",
        z=Boundary(lo=z_lo_token, hi="pec"),
    )
    sim = Simulation(
        freq_max=40e9,                          # covers through f_2 = 5 f_0
        domain=(0.002, 0.002, _L_CAVITY),
        dx=_DX, boundary=spec, cpml_layers=0,
    )
    # Source and probe on the z-axis, clear of analytic nodes.
    sim.add_source((0.001, 0.001, 0.3 * _L_CAVITY), "ex")
    sim.add_probe((0.001, 0.001, 0.7 * _L_CAVITY), "ex")
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")  # skip preflight advisories on this tiny cavity
        res = sim.run(n_steps=_N_STEPS)
    ts = np.asarray(res.time_series)[:, 0]
    dt = float(res.dt)
    # Hann window to reduce sidelobes while the cavity rings.
    window = np.hanning(_N_STEPS)
    spec_mag = np.abs(np.fft.rfft(ts * window))
    freqs = np.fft.rfftfreq(_N_STEPS, dt)
    return freqs, spec_mag, dt


def _two_strongest_peaks(freqs: np.ndarray, spec: np.ndarray,
                         band: tuple[float, float]) -> tuple[float, float]:
    """Find the two strongest spectral peaks inside ``band``."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_freqs = freqs[mask]
    band_spec = spec[mask]
    # Sort by amplitude descending; pick the top two peaks with >= 2 bins
    # separation to reject neighbouring-bin spillover.
    order = np.argsort(-band_spec)
    chosen = []
    for idx in order:
        if all(abs(idx - c) >= 2 for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == 2:
            break
    f_sorted = sorted(band_freqs[c] for c in chosen)
    return float(f_sorted[0]), float(f_sorted[1])


def test_pmc_lambda_quarter_two_peak_ladder():
    """PEC-PMC cavity produces a 1:3 spacing ratio (quarter-wave ladder).
    First mode within 10 % of analytic c/(4L)."""
    freqs, spec, _ = _run_cavity("pmc")
    band = (0.5 * _F0_ANALYTIC, 4.0 * _F0_ANALYTIC)
    f0, f1 = _two_strongest_peaks(freqs, spec, band)
    assert abs(f0 - _F0_ANALYTIC) / _F0_ANALYTIC < 0.10, (
        f"PMC-PEC f_0 must land within 10 % of analytic c/(4L) = "
        f"{_F0_ANALYTIC:.3e} Hz; got {f0:.3e} Hz"
    )
    ratio = f1 / f0
    assert 2.5 < ratio < 3.5, (
        f"PEC-PMC cavity ladder spacing f_1/f_0 must be near 3.0 "
        f"(quarter-wave); got f_0={f0:.3e}, f_1={f1:.3e}, "
        f"ratio={ratio:.3f}. PEC-PEC half-wave gives ratio 2.0."
    )


def test_pec_cavity_fails_pmc_ladder_negative_control():
    """Negative control: a PEC-PEC cavity has half-wave ladder
    f_1/f_0 = 2.0, which MUST fail the PMC quarter-wave check."""
    freqs, spec, _ = _run_cavity("pec")
    # PEC-PEC f_0 = c/(2L) = 2 · _F0_ANALYTIC (the quarter-wave analytic).
    band = (0.5 * _F0_ANALYTIC, 4.0 * _F0_ANALYTIC)
    f0, f1 = _two_strongest_peaks(freqs, spec, band)
    ratio = f1 / f0
    assert ratio <= 2.5 or ratio >= 3.5, (
        f"PEC-PEC control must NOT satisfy the PMC quarter-wave spacing "
        f"check (test would otherwise pass on any reflector). Got "
        f"f_0={f0:.3e}, f_1={f1:.3e}, ratio={ratio:.3f}; need outside "
        f"[2.5, 3.5]."
    )


# ---------------------------------------------------------------------------
# V173-B — full Harminv mode ladder (multi-mode)
# ---------------------------------------------------------------------------


def test_pmc_full_harminv_mode_ladder():
    """Four PEC-PMC modes at f_n = (2n+1)·c/(4L) for n=0..3.

    Extracted via Harminv Matrix-Pencil decomposition on the ringdown
    portion of the probe trace. Each mode must land within 3% of its
    analytic frequency, and the set of modes must cover the 1:3:5:7
    ladder (not merely 1:3 as the two-peak test pins).
    """
    from rfx.harminv import harminv as _harminv

    spec = BoundarySpec(
        x="periodic", y="periodic",
        z=Boundary(lo="pmc", hi="pec"),
    )
    sim = Simulation(
        freq_max=60e9,  # cover up to f_3 = 7 f_0 with headroom
        domain=(0.002, 0.002, _L_CAVITY),
        dx=_DX, boundary=spec, cpml_layers=0,
    )
    sim.add_source((0.001, 0.001, 0.3 * _L_CAVITY), "ex")
    sim.add_probe((0.001, 0.001, 0.7 * _L_CAVITY), "ex")
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        res = sim.run(n_steps=4096)
    ts = np.asarray(res.time_series)[:, 0]
    dt = float(res.dt)

    # Use the ringdown tail (skip the first 25% where the source is still
    # injecting). Matrix Pencil needs a clean decay to estimate decay rates.
    ringdown = ts[len(ts) // 4 :]
    modes = _harminv(
        ringdown, dt,
        f_min=0.5 * _F0_ANALYTIC,
        f_max=7.5 * _F0_ANALYTIC,
        min_Q=2.0,
    )
    assert len(modes) >= 4, (
        f"Harminv must resolve at least 4 PMC-PEC modes in "
        f"[{0.5 * _F0_ANALYTIC:.2e}, {7.5 * _F0_ANALYTIC:.2e}] Hz; "
        f"got {len(modes)}"
    )

    expected = [(2 * n + 1) * _F0_ANALYTIC for n in range(4)]
    # Match each analytic mode to the nearest Harminv peak. A 3%
    # tolerance accommodates numerical dispersion on a 40-cell cavity.
    found_freqs = np.asarray([m.freq for m in modes])
    for n, f_analytic in enumerate(expected):
        nearest = float(found_freqs[int(np.argmin(np.abs(found_freqs - f_analytic)))])
        rel_err = abs(nearest - f_analytic) / f_analytic
        assert rel_err < 0.03, (
            f"PMC-PEC mode n={n}: analytic f_{n} = "
            f"(2·{n}+1)·c/(4L) = {f_analytic:.3e} Hz, "
            f"nearest Harminv = {nearest:.3e} Hz (rel err {rel_err:.3%})."
        )


# ---------------------------------------------------------------------------
# V173-C — energy conservation in a closed PMC box
# ---------------------------------------------------------------------------


_EPS_0 = 8.854187817e-12
_MU_0 = 1.2566370614e-6


def _field_energy(state, dx: float) -> float:
    """Sum ½ε₀|E|² + ½μ₀|H|² over all interior cells."""
    ex = np.asarray(state.ex)
    ey = np.asarray(state.ey)
    ez = np.asarray(state.ez)
    hx = np.asarray(state.hx)
    hy = np.asarray(state.hy)
    hz = np.asarray(state.hz)
    e2 = float(np.sum(ex ** 2) + np.sum(ey ** 2) + np.sum(ez ** 2))
    h2 = float(np.sum(hx ** 2) + np.sum(hy ** 2) + np.sum(hz ** 2))
    cell_vol = dx ** 3
    return 0.5 * _EPS_0 * e2 * cell_vol + 0.5 * _MU_0 * h2 * cell_vol


def test_closed_pmc_cavity_energy_drift_bounded():
    """PEC-PMC cavity: total field energy during the ringdown phase
    stays bounded (< 30 % drift) over 1000 steps — no radiation path
    (PEC + PMC + periodic xy), so energy loss is limited to
    numerical dispersion on a 40-cell axis.

    The all-PMC cube variant is numerically degenerate (source has no
    radiation resistance; deposited energy falls into machine-zero
    regimes that look like "decay"). The PEC-PMC cavity that V172-C /
    V173-B already rely on is the cleanest closed-box energy oracle.
    """
    spec = BoundarySpec(
        x="periodic", y="periodic",
        z=Boundary(lo="pmc", hi="pec"),
    )

    def _run_to(n_steps):
        sim = Simulation(
            freq_max=40e9,
            domain=(0.002, 0.002, _L_CAVITY),
            dx=_DX, boundary=spec, cpml_layers=0,
        )
        sim.add_source((0.001, 0.001, 0.3 * _L_CAVITY), "ex")
        sim.add_probe((0.001, 0.001, 0.7 * _L_CAVITY), "ex")
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            return sim.run(n_steps=n_steps)

    n_early = 1000     # well after source shut-off; cavity ringing
    n_late = 2000      # 1000 steps later; energy drift due to dispersion only
    res_early = _run_to(n_early)
    res_late = _run_to(n_late)

    energy_early = _field_energy(res_early.state, _DX)
    energy_late = _field_energy(res_late.state, _DX)
    assert energy_early > 1e-30, (
        f"cavity early-energy must be above numerical-zero floor; "
        f"got {energy_early:.3e} J"
    )
    assert np.isfinite(energy_early) and np.isfinite(energy_late)
    drift = abs(energy_late - energy_early) / energy_early
    assert drift < 0.30, (
        f"PEC-PMC cavity energy drift over {n_early}→{n_late} steps "
        f"must stay < 30% (numerical dispersion only — no radiation "
        f"path). got energy_early={energy_early:.3e} J, "
        f"energy_late={energy_late:.3e} J, drift={drift:.3%}"
    )
