"""NTFF absolute-power calibration + directivity validation battery.

Validation-campaign lane: verifies the POST-PROCESSING / calibration layer
on top of the (already-validated) core FDTD update — the NTFF
Huygens-surface transform (``rfx/farfield.py``), the directivity integral
(``directivity`` / ``rfx/antenna.py`` sphere quadrature), and the pattern
fidelity of the far-field chain — using a staircase-free Hertzian z-dipole
so that NO geometry-discretization error contaminates the postproc gates.

Honest, additive posture: changes/relaxes NO existing gate (the legacy
``test_farfield.py::test_dipole_directivity`` +/-0.75 dB gate is untouched;
this battery adds a TIGHTER, measured-provenance lock on a separate,
flux-cross-checked fixture).

Key structural point: directivity D = 4*pi*U_max / P_rad is a RATIO — any
overall scale error in the NTFF transform cancels in it. Only the
P_ntff / P_flux cross-check in this battery pins the ABSOLUTE power scale
of the far-field chain against an independent in-domain observable (the
closed 6-face Poynting flux box). Predicted ratio (derived in the
2026-07-10 convention audit, not fitted): the flux path integrates the
UN-halved ``Re(E x H*) . n_hat`` (``rfx/probes/probes.py::flux_spectrum``)
while the NTFF power path uses U = |E|^2 / (2*eta_0) (``rfx/antenna.py``),
so P_ntff / P_flux = 1/2 exactly in the continuum, independent of dt,
source spectrum, and frequency bin.

Fixture (mirrors the 2026-07-10 diagnostic lane script EXACTLY; all
coordinates are integer multiples of every rung dx so snapping is exact and
each flux-face quadrature area is exactly (0.030 m)^2):
  - domain 0.06 m cube, boundary="cpml", cpml_layers=6, freq_max=5 GHz
  - single-cell soft ez source at center (0.03, 0.03, 0.03),
    GaussianPulse(f0=3 GHz, bandwidth=0.5); DFT bin at f0
  - NTFF box [0.021, 0.039]^3
  - closed 6-face flux box [0.015, 0.045]^3, strictly BETWEEN the NTFF box
    and the CPML; net outward power = (x_hi + y_hi + z_hi) -
    (x_lo + y_lo + z_lo) per the +axis-positive sign convention of
    ``flux_spectrum`` (probes.py: "Positive = power flowing in +axis
    direction" — lo faces need a minus sign for OUTWARD power)
  - dense sphere: theta 73 pts on [0.01, pi-0.01], phi 48 pts over 2*pi
    endpoint=False (task-recipe pitfall 6: a sparse phi grid under-samples
    the P_rad integral and corrupts D by dB-level amounts)

Measured baseline (R5 measure-before-gate; 2026-07-10 diagnostic lane +
same-session base-rung remeasure, bit-reproducible on this CPU box):
  base rung dx=3.0 mm, 400 steps (~2.3 s):
    D = 1.8000060791 dBi   (theory 10*log10(1.5) = 1.7609; err +0.0391 dB)
    P_ntff/P_flux = 0.4947778515   (predicted 0.5; off by -0.0052)
    bright-region [20,160] deg max |dev| vs sin^2(theta) = 0.0586 dB
    cross-pol max|E_phi|/max|E_theta| over the sphere = 0.003882
    jnp flux_spectrum vs float64 accumulator recompute: per-face rel
    difference <= 3.0e-7 (net 4.0e-9)
  dx-ladder (slow_physics): err_db 0.0391 -> 0.0147 -> 0.0064 (monotone);
    flux ratio 0.4948 / 0.5128 / 0.5215 (all within 5% of 0.5)

Preflight (quoted per feedback_never_ignore_preflight): explicit
``sim.preflight()`` on this fixture returns six lambda/4 advisories of the
form "NTFF face x_lo is 9.00mm from port/source at (0.03, 0.03, 0.03) —
below λ/4 = 24.98mm at f_max=3.00GHz. NTFF will integrate reactive
near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2
from any radiating/scattering structure (Huygens-equivalence rule)."
(one per NTFF face; the x_hi one is triggered by a passive witness probe
sitting ON that face). The measured 0.006-0.039 dB directivity error shows
the advisory is conservative for a point dipole 9 mm from the box; the
fixture keeps the run non-blocking (run()'s auto-preflight passes) and the
battery does NOT use skip_preflight.

Known numerics caveats carried by this battery (documented, not patched):
  (1) float32 flush-to-zero in ``flux_spectrum`` at fine dx — at the
      dx=0.75 mm rung the jnp flux path returns exactly 0.0 on all six
      faces because per-cell integrand*dA (~5.5e-39 W) is below the float32
      minimum normal 1.18e-38 with x64 disabled; the DFT accumulators are
      healthy, and a NumPy float64 recompute of the identical sum gives
      finite, correctly-signed flux (rung-2 witness: recompute matches the
      non-zero jnp value to <= 3e-7 rel). The ladder test therefore uses
      the float64 recompute helper for ALL rungs, and the fast suite pins
      recompute==jnp on the base rung where both are healthy.
  (2) the measured ratio drifts away from 0.5 with refinement (0.4948 ->
      0.5128 -> 0.5215); the coarsest rung matches the E-time-label
      prediction 0.5*cos^2(omega*dt/2) = 0.4986. The ladder gate is a
      +/-5% band around 0.5 at every rung (measured max off 4.3%), i.e. a
      stability lock, NOT a convergence-to-0.5 claim.

No network, no external solver; deterministic (fixed geometry, fixed step
counts, rect DFT window). Tighten gates only with a fresh measured
baseline; a failure here marks a postproc/calibration regression.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Simulation, GaussianPulse
from rfx.antenna import _radiation_intensity, _total_radiated_power
from rfx.farfield import compute_far_field, directivity
from rfx.probes.probes import flux_spectrum

# ===========================================================================
# Fixture geometry (all lengths in metres; multiples of every rung dx)
# ===========================================================================
F0_HZ = 3.0e9            # DFT bin / pulse centre
FREQ_MAX_HZ = 5.0e9      # Simulation freq_max (sets default mesh checks)
DOMAIN_M = 0.06          # cube edge
CENTER_M = 0.03          # dipole position on every axis
NTFF_LO_M, NTFF_HI_M = 0.021, 0.039    # NTFF box corners
FLUX_LO_M, FLUX_HI_M = 0.015, 0.045    # closed flux box, outside NTFF box
CPML_LAYERS = 6
BASE_DX_M = 3.0e-3       # base rung (fast suite)
BASE_N_STEPS = 400       # probe tail/peak <= 1.8e-4 at all rungs (settled)

D_THEORY_DBI = 10.0 * np.log10(1.5)    # Hertzian dipole: 1.7609 dBi

# Dense observation sphere (recipe pitfall 6: phi >= 24 samples over 2*pi,
# endpoint=False, REQUIRED for the P_rad integral).
THETA_GRID = np.linspace(0.01, np.pi - 0.01, 73)
PHI_GRID = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)

# ===========================================================================
# Gate constants (R5: every gate = measured value + honest margin)
# ===========================================================================
# Measured base rung 2026-07-10: ratio = 0.4947778515 -> |off| = 0.0052.
# Gate 0.05 (~10x measured). Predicted 0.5 derived from source (see module
# docstring), not fitted.
_RATIO_PREDICTED = 0.5
_RATIO_TOL_BASE = 0.05

# Measured base rung: D = 1.8000060791 dBi, |err| = 0.0391 dB vs theory.
# Gate 0.25 dB (~6x measured) — tighter than the legacy +/-0.75 dB gate in
# test_farfield.py (which is untouched); the base-rung measurement supports
# the tighter bound, and the flux cross-check above pins the absolute scale
# that D (a ratio) cannot see.
_D_ERR_MAX_DB = 0.25

# Measured base rung: bright-region [20, 160] deg max |10*log10(U/sin^2)|
# = 0.0586 dB (finest rung: 0.0056 dB). Gate 0.30 dB (~5x measured base).
_BRIGHT_DEV_MAX_DB = 0.30

# Measured base rung: max|E_phi|/max|E_theta| over the full sphere
# = 0.003882 (finest rung: 0.000984). Gate 0.02 (~5x measured base).
_CROSSPOL_MAX = 0.02

# Measured base rung: per-face |jnp - f64 recompute| / |f64| <= 3.0e-7.
# Gate 1e-4 — validates the float64 recompute helper against the shipped
# flux_spectrum on a rung where float32 does NOT flush.
_FLUX_RECOMPUTE_RTOL = 1e-4

# dx-ladder (slow_physics). Measured |err_db|: 0.0391, 0.0147, 0.0064.
# Per-rung caps ~3-6x measured; plus a strict finest < coarsest witness
# (6x apart in measurement, robust to cross-machine float noise).
_LADDER_RUNGS = (
    # (dx_m, n_steps, err_cap_db)
    (3.0e-3, 400, 0.15),
    (1.5e-3, 800, 0.08),
    (0.75e-3, 1600, 0.04),
)
# Measured ladder ratios: 0.4948, 0.5128, 0.5215 (max off 0.0215 = 4.3%).
# Gate +/-0.05 around 0.5 at every rung — a stability lock; the drift AWAY
# from 0.5 with refinement is a documented anomaly (module docstring), so
# this deliberately does NOT assert convergence to 0.5.
_RATIO_TOL_LADDER = 0.05


# ===========================================================================
# Shared helpers
# ===========================================================================
def _build_sim(dx: float) -> Simulation:
    """Hertzian-dipole calibration fixture at cell size ``dx``.

    Mirrors the 2026-07-10 diagnostic lane script exactly: 0.06 m cube,
    CPML(6 layers), single-cell soft ez GaussianPulse dipole at the centre,
    NTFF box [0.021, 0.039]^3, and six finite flux monitors forming the
    closed box [0.015, 0.045]^3 (each face a 0.030 x 0.030 m plane centred
    on the domain axis). Two ez witness probes (source cell + NTFF x_hi
    face) provide the pulse-decay witness.
    """
    sim = Simulation(
        freq_max=FREQ_MAX_HZ, domain=(DOMAIN_M,) * 3, dx=dx,
        boundary="cpml", cpml_layers=CPML_LAYERS,
    )
    sim.add_source((CENTER_M,) * 3, "ez",
                   waveform=GaussianPulse(f0=F0_HZ, bandwidth=0.5))
    sim.add_probe((CENTER_M,) * 3, "ez")
    sim.add_probe((NTFF_HI_M, CENTER_M, CENTER_M), "ez")
    sim.add_ntff_box(corner_lo=(NTFF_LO_M,) * 3, corner_hi=(NTFF_HI_M,) * 3,
                     freqs=jnp.array([F0_HZ]))
    span = FLUX_HI_M - FLUX_LO_M
    for ax in "xyz":
        for coord, side in ((FLUX_LO_M, "lo"), (FLUX_HI_M, "hi")):
            sim.add_flux_monitor(axis=ax, coordinate=coord,
                                 freqs=jnp.array([F0_HZ]),
                                 size=(span, span),
                                 center=(CENTER_M, CENTER_M),
                                 name=f"{ax}_{side}")
    return sim


def _flux_f64(mon) -> float:
    """Float64 NumPy recompute of ``flux_spectrum`` from the accumulators.

    Identical sum to probes.py::flux_spectrum (integrand = e1*conj(h2) -
    e2*conj(h1), times dA), evaluated in float64 so it cannot flush to zero
    when per-cell integrand*dA drops below the float32 minimum normal
    (1.18e-38) — which the dx=0.75 mm rung measurably does (module
    docstring, caveat 1). Validated against the jnp path on the base rung
    by test_flux_f64_recompute_matches_jnp_path.
    """
    e1 = np.asarray(mon.e1_dft, dtype=np.complex128)
    e2 = np.asarray(mon.e2_dft, dtype=np.complex128)
    h1 = np.asarray(mon.h1_dft, dtype=np.complex128)
    h2 = np.asarray(mon.h2_dft, dtype=np.complex128)
    dA = np.asarray(mon.dA, dtype=np.float64)
    return float(np.real(np.sum((e1 * np.conj(h2) - e2 * np.conj(h1)) * dA)))


def _net_outward_power(face_flux: dict) -> float:
    """Closed-box outward power from the six +axis-positive face fluxes."""
    return ((face_flux["x_hi"] + face_flux["y_hi"] + face_flux["z_hi"])
            - (face_flux["x_lo"] + face_flux["y_lo"] + face_flux["z_lo"]))


def _run_rung(dx: float, n_steps: int) -> dict:
    """Run one fixture rung; return the derived calibration observables."""
    sim = _build_sim(dx)
    res = sim.run(n_steps=n_steps)

    # Decay witness: the rect-window DFT is only trustworthy if the pulse
    # has left the domain within the window (measured tail/peak <= 1.8e-4).
    ts = np.asarray(res.time_series)
    tail = ts[int(0.9 * ts.shape[0]):]
    tail_over_peak = float(np.max(
        np.max(np.abs(tail), axis=0) / np.max(np.abs(ts), axis=0)))

    face_jnp = {name: float(np.asarray(flux_spectrum(mon))[0])
                for name, mon in res.flux_monitors.items()}
    face_f64 = {name: _flux_f64(mon)
                for name, mon in res.flux_monitors.items()}

    ff = compute_far_field(res.ntff_data, res.ntff_box, res.grid,
                           THETA_GRID, PHI_GRID)
    return {
        "dx": dx,
        "tail_over_peak": tail_over_peak,
        "face_jnp": face_jnp,
        "face_f64": face_f64,
        "p_flux_f64": _net_outward_power(face_f64),
        "ff": ff,
        "p_ntff": float(_total_radiated_power(ff)[0]),
        "d_dbi": float(directivity(ff)[0]),
    }


@pytest.fixture(scope="module")
def base_rung():
    """Base rung (dx=3 mm, 400 steps, ~2.3 s) — shared by all fast tests."""
    rung = _run_rung(BASE_DX_M, BASE_N_STEPS)
    # R5 breadcrumb for -s runs.
    print(f"\n[ntff-battery base rung] D={rung['d_dbi']:.6f} dBi "
          f"(theory {D_THEORY_DBI:.4f}), "
          f"P_ntff={rung['p_ntff']:.6e} W, "
          f"P_flux(f64)={rung['p_flux_f64']:.6e} W, "
          f"ratio={rung['p_ntff'] / rung['p_flux_f64']:.6f}, "
          f"tail/peak={rung['tail_over_peak']:.2e}")
    return rung


# ===========================================================================
# FAST battery (default suite)
# ===========================================================================

def test_pulse_settled_within_window(base_rung):
    """Rect-window validity witness: probe tail/peak <= 1e-3.

    Measured 1.3e-4 on the base rung (both probes; ladder max 1.8e-4).
    If this fails, every DFT-derived gate below is measuring an unsettled
    ringdown, not the radiated pulse — fix the window before reading gates.
    """
    assert base_rung["tail_over_peak"] < 1e-3, (
        f"pulse not settled: tail/peak = {base_rung['tail_over_peak']:.3e} "
        f"(measured 1.3e-4) — rect-window DFT observables are unreliable")


def test_flux_f64_recompute_matches_jnp_path(base_rung):
    """The float64 recompute helper reproduces shipped flux_spectrum.

    Measured per-face rel difference <= 3.0e-7 on the base rung (where
    float32 does not flush); gate 1e-4. This pins the helper the ladder
    test relies on at the dx=0.75 mm rung, where the jnp path measurably
    returns exactly 0.0 (float32 flush-to-zero, module docstring caveat 1).
    """
    for name in base_rung["face_jnp"]:
        jnp_val = base_rung["face_jnp"][name]
        f64_val = base_rung["face_f64"][name]
        assert f64_val != 0.0, f"face {name}: f64 recompute is zero"
        rel = abs(jnp_val - f64_val) / abs(f64_val)
        assert rel < _FLUX_RECOMPUTE_RTOL, (
            f"face {name}: jnp {jnp_val:.6e} vs f64 recompute {f64_val:.6e} "
            f"(rel {rel:.2e}, measured <= 3.0e-7, gate {_FLUX_RECOMPUTE_RTOL})")


def test_ntff_absolute_power_calibration_vs_flux_box(base_rung):
    """P_ntff / P_flux pins the ABSOLUTE scale of the NTFF chain: 0.5 +/- 0.05.

    The only test in the far-field suite sensitive to an overall NTFF scale
    error (directivity is a ratio and cancels it). Predicted 0.5 from
    source conventions (un-halved flux integrand vs |E|^2/(2*eta_0) far-
    field intensity — module docstring); measured 0.4947778515 on the base
    rung (off by 0.0052; gate ~10x that). A factor-2 or eta_0 slip anywhere
    in farfield.py/antenna.py normalization moves this by 3 dB-class
    amounts and fails loudly.
    """
    p_ntff = base_rung["p_ntff"]
    p_flux = base_rung["p_flux_f64"]
    assert p_flux > 0.0, f"closed-box outward power not positive: {p_flux:.3e}"
    assert p_ntff > 0.0, f"NTFF radiated power not positive: {p_ntff:.3e}"
    ratio = p_ntff / p_flux
    assert abs(ratio - _RATIO_PREDICTED) < _RATIO_TOL_BASE, (
        f"P_ntff/P_flux = {ratio:.6f}, predicted {_RATIO_PREDICTED} "
        f"(measured 0.494778, gate +/-{_RATIO_TOL_BASE}) — absolute-scale "
        f"calibration of the NTFF chain has moved")


def test_dipole_directivity_regression_lock(base_rung):
    """Hertzian-dipole directivity: |D - 1.7609 dBi| < 0.25 dB.

    Measured 1.8000060791 dBi on the base rung (err +0.0391 dB); gate ~6x
    the measured error, deliberately tighter than the legacy +/-0.75 dB
    gate in test_farfield.py (untouched). The base-rung measurement
    supports the tighter bound; treat a failure as a regression in the
    NTFF transform or the sphere quadrature, not as licence to widen.
    """
    err_db = abs(base_rung["d_dbi"] - D_THEORY_DBI)
    assert err_db < _D_ERR_MAX_DB, (
        f"D = {base_rung['d_dbi']:.4f} dBi vs theory {D_THEORY_DBI:.4f} "
        f"(|err| {err_db:.4f} dB, measured 0.0391, gate {_D_ERR_MAX_DB})")


def test_bright_region_pattern_matches_sin2_theta(base_rung):
    """Phi-averaged normalized pattern vs sin^2(theta), [20, 160] deg.

    Measured max |deviation| 0.0586 dB on the base rung (0.0056 dB at
    dx=0.75 mm); gate 0.30 dB (~5x measured base). A point dipole has no
    staircase, so this isolates pure NTFF surface-integration error.
    """
    U = _radiation_intensity(base_rung["ff"])[0]        # (n_theta, n_phi)
    U_phi_mean = U.mean(axis=1)
    U_norm = U_phi_mean / U_phi_mean.max()
    sin2 = np.sin(THETA_GRID) ** 2
    bright = ((THETA_GRID >= np.radians(20.0))
              & (THETA_GRID <= np.radians(160.0)))
    dev_db = 10.0 * np.log10(U_norm[bright]) - 10.0 * np.log10(sin2[bright])
    max_dev = float(np.max(np.abs(dev_db)))
    assert max_dev < _BRIGHT_DEV_MAX_DB, (
        f"bright-region pattern deviation {max_dev:.4f} dB vs sin^2(theta) "
        f"(measured 0.0586, gate {_BRIGHT_DEV_MAX_DB})")


def test_crosspol_stays_numerical_noise(base_rung):
    """z-dipole cross-pol max|E_phi|/max|E_theta| over the sphere < 0.02.

    An ideal z-dipole radiates E_theta only; measured 0.003882 on the base
    rung (0.000984 at dx=0.75 mm); gate ~5x measured base. Complements the
    looser phi=0-cut gate in test_farfield.py::test_dipole_e_phi_small
    (0.3) by covering the FULL sphere at a numerically-tight level.
    """
    ff = base_rung["ff"]
    e_th_peak = float(np.max(np.abs(ff.E_theta[0])))
    e_ph_max = float(np.max(np.abs(ff.E_phi[0])))
    assert e_th_peak > 0.0
    crosspol = e_ph_max / e_th_peak
    assert crosspol < _CROSSPOL_MAX, (
        f"cross-pol {crosspol:.6f} (measured 0.003882, gate {_CROSSPOL_MAX})")


# ===========================================================================
# slow_physics battery (opt-in: -m slow_physics)
# ===========================================================================

@pytest.mark.slow_physics
def test_dx_ladder_directivity_converges_and_ratio_stable(base_rung):
    """dx-ladder witness: D error shrinks with refinement; ratio stays 0.5-ish.

    Measured (2026-07-10): |err_db| 0.0391 -> 0.0147 -> 0.0064 dB across
    dx = 3.0 / 1.5 / 0.75 mm (strictly monotone); per-rung caps 0.15 /
    0.08 / 0.04 dB (~3-6x measured) plus a strict finest < coarsest
    assertion (6x apart in measurement — robust to float noise).

    Flux ratio uses the float64 recompute at EVERY rung because at
    dx=0.75 mm the jnp flux path measurably flushes to exactly 0.0
    (float32, caveat 1). Measured ratios 0.4948 / 0.5128 / 0.5215; gate
    0.5 +/- 0.05 per rung. This is a STABILITY lock: the measured drift
    AWAY from 0.5 with refinement (-1.0% -> +2.6% -> +4.3%) is a documented
    anomaly (suspected fixed-cpml_layers confound, module docstring caveat
    2), so no convergence-to-0.5 is asserted. ~35 s CPU for the two finer
    rungs (the base rung is reused from the module fixture).
    """
    rungs = [base_rung]
    for dx, n_steps, _cap in _LADDER_RUNGS[1:]:
        rungs.append(_run_rung(dx, n_steps))

    errs_db = []
    for rung, (dx, _n, cap_db) in zip(rungs, _LADDER_RUNGS):
        err_db = abs(rung["d_dbi"] - D_THEORY_DBI)
        errs_db.append(err_db)
        print(f"[ntff-battery ladder] dx={dx * 1e3:.2f} mm: "
              f"D={rung['d_dbi']:.6f} dBi (|err| {err_db:.4f} dB, cap {cap_db}), "
              f"ratio={rung['p_ntff'] / rung['p_flux_f64']:.6f}, "
              f"tail/peak={rung['tail_over_peak']:.2e}")
        assert err_db < cap_db, (
            f"dx={dx * 1e3:.2f} mm: |D err| {err_db:.4f} dB exceeds cap "
            f"{cap_db} (measured 0.0391/0.0147/0.0064 across the ladder)")

        assert rung["tail_over_peak"] < 1e-3, (
            f"dx={dx * 1e3:.2f} mm: pulse not settled "
            f"(tail/peak {rung['tail_over_peak']:.3e})")

        p_flux = rung["p_flux_f64"]
        assert p_flux > 0.0, (
            f"dx={dx * 1e3:.2f} mm: f64 closed-box power not positive "
            f"({p_flux:.3e})")
        ratio = rung["p_ntff"] / p_flux
        assert abs(ratio - _RATIO_PREDICTED) < _RATIO_TOL_LADDER, (
            f"dx={dx * 1e3:.2f} mm: P_ntff/P_flux = {ratio:.6f} outside "
            f"0.5 +/- {_RATIO_TOL_LADDER} "
            f"(measured 0.4948/0.5128/0.5215 across the ladder)")

    assert errs_db[-1] < errs_db[0], (
        f"directivity error did not shrink across the ladder: "
        f"{errs_db[0]:.4f} dB (coarsest) -> {errs_db[-1]:.4f} dB (finest); "
        f"measured 0.0391 -> 0.0064")
