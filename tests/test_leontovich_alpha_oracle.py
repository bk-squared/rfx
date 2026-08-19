"""Parallel-plate conductor-attenuation oracle for the Leontovich sheet
(issue #669) — the PRIMARY FDTD physics gate for surface_impedance_f0.

Fixture (constants fixed by the #669 implementation contract):
  Vacuum parallel-plate TEM guide, plate separation b = 5 mm (10 cells at
  dx = 0.5 mm uniform), two f0-mode sheets as the plates (z = 0.5 and
  5.5 mm, spanning the full guide), sigma_bulk = 1e4 S/m, thickness =
  2e-4 m (>= 3*delta = 151 um, so the thin-film advisory stays off;
  thickness does not enter sigma_eff in f0 mode), f0 = 10 GHz = source
  centre. Rs0 = 1.9869 ohm/sq, alpha_analytic = Rs0/(eta0*b) =
  1.05482 Np/m (0.916 dB over the 100 mm fit span). Operating point
  x = sigma_eff*dt/(2*eps0) ~ 55.

Guide construction (fixture engineering, not contract constants):
  x: CPML on the source side, PEC wall on the far side (see below); y:
  PMC walls (2 mm); z: PEC outer faces at 0/6 mm, one cell behind each
  sheet (backing-stub reactance eta0*tan(beta*g) = 39.6 ohm >> Rs0,
  Re(Zs_eff)/Rs0 = 0.9975). If the sheets simply truncated at a CPML pad
  (thin conductors are applied AFTER pad extension, issue #642) the wave
  would see a 5 mm -> 6 mm guide-height step reflecting |Gamma| = 1/11 =
  0.0909; a 1-D transfer-matrix comparator (run at design time; table in
  docs/research_notes/) showed that ripple biases the 100 mm log-linear
  fit by up to 15.5% worst-case over phase — the naive fixture would fail
  its own gates for reasons that are FIT contamination, not sheet physics.
  The guide is therefore terminated by an in-guide graded absorber:
  60 mm (120 cells) of eps_r = 1 material with sigma(x) = 2.0 *
  ((i+0.5)/120)**2 S/m ending on the hi-x PEC wall. The absorber MUST end
  on a reflecting wall, not run into a CPML pad: geometry sigma is
  pad-extended (#582/#627), and conductive material replicated into a
  CPML pad is the #642 instability class — measured here as a slow
  exponential blow-up (fields 1e16 by step 4000) present with the
  absorber alone and absent without it, at ONLY 2 S/m. With the PEC back
  wall the transfer-matrix return (grading reflection + wall reflection
  attenuated by 2 x 5.9 Np) is |Gamma| = 0.0026 -> worst-case fit bias
  0.5% (lossy) and 0.005 Np/m (PEC control); an 8000-step witness run is
  clean (probe flat, final fields at -80.6 dB).

Measurement: log-linear least-squares fit of ln|Ez(x)| at f0 over the
DFT-plane accumulator (z = 3 mm midplane) along x in [25, 125] mm, with
the fit RMS residual as the forward-wave-purity witness. NEVER probe
time-series + FFT (repo Never-list).

Comparator-first (contract graft from C): before any FDTD number is
gated, `test_comparator_quadrature_reproduces_closed_form` re-derives
alpha from the analytic TEM field and Rs0 by hand quadrature over the
actual grid sampling and asserts it reproduces Rs0/(eta0*b) to rtol 1e-6,
and `test_fit_chain_recovers_synthetic_alpha` shows the fitting chain
recovers a synthetic exponential's alpha to rtol 1e-6.

Envelope provenance and R2-STOP record (measured 2026-08-19, this file's
fixture, JAX CPU float32):
  Attempt 1 (both-ends-CPML fixture) never evaluated the sheet physics:
  its own witnesses flagged it (settle 0 dB, alpha identical across all
  frequency bins) — the absorber-sigma-in-CPML-pad instability above, an
  identified implementation defect of the FIXTURE, diagnosed
  comparator-first and fixed by the PEC-wall termination.
  Attempt 2 (this fixture): alpha_meas(10 GHz) = 0.85779 Np/m,
  |alpha_meas/alpha_analytic - 1| = 0.18679 — ABOVE the pre-declared
  0.10 R2-STOP bar, so the O3 physics gate is RED and stays red
  (no-silent-gate-loosening): gate_from_envelope(0.18679) = 0.29 exceeds
  the 0.15 contract hard cap, so the committed gate is the cap and
  ``test_alpha_oracle_o3`` is marked xfail(strict=True) — the red is
  loud, and a future pass is loud too. Every witness was clean at
  the time (fit ln-RMS residual 0.00206, settle -79.8 dB, comparator
  chain validated — but see the COMPARATOR CAVEAT below, which the
  ln-RMS residual is blind to), and the discriminators PASS (O4a
  sqrt-sigma ratio 0.517, O4b thickness-invariance 7e-6, O4c PEC control
  0.0071 Np/m), so the fold implements exactly the contracted algebra and
  loss moves only when asked — the deficit is in the delivered sheet
  DYNAMICS, not in the wiring. Mechanism fingerprint: alpha(f) =
  0.688/0.769/0.858/0.943/1.025 Np/m at 8/9/10/11/12 GHz — linear in f
  through the origin, while an ideal Rs-flat sheet predicts a FLAT 1.055.
  A 10x-smaller-Rs0 case (sigma_bulk = 1e6) shows the SAME fractional
  deficit (0.174), so the deficit is Rs-scale-independent. Named
  alternatives per the R2-STOP protocol (not built here):
  exponential-stepping sheet-current coefficient; true multi-pole SIBC
  boundary.

Mesh-refinement falsifier — pre-declared, run 2026-08-19, gate untouched,
  and NOT resolving: the same oracle at dx = 0.500 / 0.250 / 0.125 mm
  measures alpha(10 GHz) = 0.85777 / 0.84685 / 0.95601 Np/m, i.e.
  deficits 0.18681 / 0.19716 / 0.09367. That is not a systematic shrink
  with dx — it RISES at the first refinement — so the pre-declared "O(dx)
  one-cell sheet discretization" hypothesis is not supported; a flat
  constant offset is not supported either. Note that the operating point
  x = sigma_eff*dt/(2*eps0) measures 54.19 on ALL THREE meshes:
  sigma_eff = 1/(Rs0*dx) grows as 1/dx while dt falls as dx, so refining
  THIS fixture cannot move x and cannot probe an x-driven mechanism at
  all. (An earlier note recorded dx = 0.25 mm as "essentially unchanged,
  0.847" and concluded mesh-independence; 0.84685 reproduces, but the
  quarter-resolution point shows two points did not establish a trend.)

COMPARATOR CAVEAT — why that sweep does not settle it, and what to fix
  first: the fitted alpha is a SPAN AVERAGE of a profile that is not a
  single exponential. The local decay rate over 20 mm sub-windows falls
  monotonically across the 25-125 mm fit span on every mesh: 1.043 ->
  0.757 Np/m at dx = 0.5 mm, 1.077 -> 0.706 at 0.25 mm, 1.229 -> 0.637 at
  0.125 mm; the two half-span fits differ by 19.3% / 8.3% / 36.9% of
  alpha. The ln-RMS residual does NOT surface this (0.00206 / 0.00201 /
  0.00313) — a slow curvature over a 100 mm span leaves tiny per-point
  residuals while moving the slope a lot, so a small residual is not a
  purity witness for a DECAY-RATE fit. The mesh-to-mesh differences above
  therefore sit inside the measurement's own window sensitivity. The next
  step on O3 is the COMPARATOR (a fit that does not assume one
  exponential, or a two-plane amplitude ratio at fixed separation), not
  another mechanism attempt on the sheet.
Full per-bin dump, alpha(f) trace, verbatim preflight output, the
mesh-refinement sweep with its per-window decay-rate table, and the
sigma_bulk = 1e6 deep-screened evidence case: docs/research_notes/
2026-08-19_leontovich_sheet_envelope.md (local-only).
"""

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.materials.thin_conductor import leontovich_rs
from rfx.core.yee import EPS_0, MU_0

from tests._gate_policy import gate_from_envelope

ETA_0 = float(np.sqrt(MU_0 / EPS_0))

# ---- contract-fixed fixture constants ----
DX = 0.5e-3
B_PLATE = 5e-3                     # plate separation (10 cells)
SIGMA_BULK = 1e4
THICKNESS = 2e-4
F0 = 10e9
FIT_X = (0.025, 0.125)             # 100 mm fit span
ALPHA_ANALYTIC = float(leontovich_rs(F0, SIGMA_BULK)) / (ETA_0 * B_PLATE)

# ---- fixture engineering (see module docstring) ----
Z_SHEET_LO = 0.5e-3
Z_SHEET_HI = 5.5e-3
DOMAIN = (0.190, 0.002, 0.006)
ABSORBER_X0 = 0.130
ABSORBER_N = 120                   # cells (60 mm)
ABSORBER_SIGMA_MAX = 2.0           # S/m, quadratic profile
SRC_X = 0.010
N_STEPS = 4000

# ---- measured envelope (see module docstring R2-STOP record) ----
MEASURED_ALPHA = 0.85779           # Np/m at f0, this fixture, 2026-08-19
MEASURED_ENVELOPE = 0.18679        # |alpha/alpha_analytic - 1| — R2-STOP
# gate_from_envelope(MEASURED_ENVELOPE) = 0.29 would EXCEED the 0.15
# contract hard cap, so the committed gate is the cap itself and the O3
# test is xfail(strict=True): red today, loudly, and a future pass is
# loud too. The derivation below is kept so the cap-binding is checked.
ALPHA_GATE = min(gate_from_envelope(MEASURED_ENVELOPE, quantum=100), 0.15)
assert ALPHA_GATE == 0.15, "cap must bind while the envelope exceeds 0.10"


def _build_guide(sigma_bulk=SIGMA_BULK, *, f0_mode=True,
                 thickness=THICKNESS, freqs=(F0,)):
    sim = Simulation(
        freq_max=10e9, domain=DOMAIN, dx=DX,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="pec"),
                              y="pmc", z="pec"),
        cpml_layers=10,
    )
    # graded in-guide absorber (full cross-section, ends on hi-x PEC)
    for i in range(ABSORBER_N):
        s = ABSORBER_SIGMA_MAX * ((i + 0.5) / ABSORBER_N) ** 2
        x0 = ABSORBER_X0 + i * DX
        sim.add_material(f"abs{i}", eps_r=1.0, sigma=s)
        sim.add(Box((x0, 0.0, 0.0), (x0 + DX, DOMAIN[1], DOMAIN[2])),
                material=f"abs{i}")
    # the two plates
    kw = dict(sigma_bulk=sigma_bulk, thickness=thickness)
    if f0_mode:
        kw["surface_impedance_f0"] = F0
    for zs in (Z_SHEET_LO, Z_SHEET_HI):
        sim.add_thin_conductor(
            Box((0.0, 0.0, zs), (DOMAIN[0], DOMAIN[1], zs)), **kw)
    # TEM launch: Ez column spanning the gap, uniform in z (single y is
    # fine: PMC walls + y-uniform mode; TE-y modes cut off at 75 GHz)
    for k in range(9):
        z = 1.0e-3 + k * DX
        sim.add_source((SRC_X, 0.001, z), "ez",
                       waveform=GaussianPulse(f0=F0, bandwidth=0.5),
                       amplitude_kind="field")
    sim.add_dft_plane_probe(axis="z", coordinate=3.0e-3, component="ez",
                            freqs=jnp.asarray(freqs), name="midplane")
    sim.add_probe((0.060, 0.001, 3.0e-3), "ez")   # settling witness
    return sim


def _fit_alpha(xs, mag):
    """Log-linear LSQ fit; returns (alpha, rms_residual_ln)."""
    y = np.log(mag)
    A = np.vstack([np.ones_like(xs), -xs]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(coef[1]), float(np.sqrt(np.mean(resid ** 2)))


def _run_guide(sigma_bulk=SIGMA_BULK, *, f0_mode=True, thickness=THICKNESS,
               freqs=(F0,), n_steps=N_STEPS):
    """One guide run. Returns dict with per-freq alpha, residuals, the
    captured preflight/warning text, and the settling witness in dB."""
    sim = _build_guide(sigma_bulk, f0_mode=f0_mode, thickness=thickness,
                       freqs=freqs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = sim.run(n_steps=n_steps, compute_s_params=False)
    grid = sim._build_grid()
    acc = np.asarray(result.dft_planes["midplane"].accumulator)
    j = int(round(0.001 / DX)) + grid.pad_y_lo
    i0 = int(round(FIT_X[0] / DX)) + grid.pad_x_lo
    i1 = int(round(FIT_X[1] / DX)) + grid.pad_x_lo
    xs = (np.arange(i0, i1 + 1) - grid.pad_x_lo) * DX
    out = {"freqs": list(freqs), "alpha": [], "resid": [], "profile": [],
           "xs": xs}
    for fi in range(len(freqs)):
        mag = np.abs(acc[fi, i0:i1 + 1, j])
        a, r = _fit_alpha(xs, mag)
        out["alpha"].append(a)
        out["resid"].append(r)
        out["profile"].append(mag)
    ts = np.abs(np.asarray(result.time_series)[:, 0])
    tail = ts[int(0.95 * len(ts)):].max()
    out["settle_db"] = float(20 * np.log10(max(tail, 1e-300) / ts.max()))
    out["warnings"] = [str(w.message) for w in rec]
    return out


# ---------------------------------------------------------------------------
# Comparator-first: validate the measurement chain with no FDTD involved
# ---------------------------------------------------------------------------

def test_comparator_quadrature_reproduces_closed_form():
    """alpha from the analytic TEM field + Rs0 by hand quadrature over the
    actual grid sampling == Rs0/(eta0*b) to rtol 1e-6 (contract)."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    rs0 = float(leontovich_rs(F0, SIGMA_BULK))
    ny, nz = 5, 11                      # grid sampling of the cross-section
    ys = np.linspace(0.0, DOMAIN[1], ny)
    zs = np.linspace(0.0, B_PLATE, nz)
    E0 = 1.0                            # uniform TEM E between the plates
    H0 = E0 / ETA_0
    # forward power: integral of |E||H|/2 over the cross-section
    integrand = np.full((nz, ny), 0.5 * E0 * H0)
    p_flow = trapezoid(trapezoid(integrand, ys, axis=1), zs, axis=0)
    # loss per length: two plates, Rs0*|H|^2/2 per unit area, width-integrated
    p_loss = 2.0 * trapezoid(np.full(ny, 0.5 * rs0 * H0 ** 2), ys)
    alpha_quad = p_loss / (2.0 * p_flow)
    assert abs(alpha_quad - ALPHA_ANALYTIC) / ALPHA_ANALYTIC < 1e-6
    # and the closed form itself matches the contract's number
    assert abs(ALPHA_ANALYTIC - 1.05482) < 5e-5


def test_fit_chain_recovers_synthetic_alpha():
    """The log-linear fit recovers a synthetic exponential to rtol 1e-6,
    and flags ripple through the residual witness (purity channel)."""
    xs = np.arange(FIT_X[0], FIT_X[1] + 1e-12, DX)
    a, r = _fit_alpha(xs, np.exp(-ALPHA_ANALYTIC * xs))
    assert abs(a - ALPHA_ANALYTIC) / ALPHA_ANALYTIC < 1e-6
    assert r < 1e-12
    # counter-propagating ripple shows up in the residual, not silently
    beta = 2 * np.pi * F0 / 299792458.0
    e = np.exp(-ALPHA_ANALYTIC * xs) * np.abs(
        1 + 0.05 * np.exp(2j * beta * xs))
    _, r2 = _fit_alpha(xs, e)
    assert r2 > 1e-3


# ---------------------------------------------------------------------------
# FDTD gates (slow_physics lane)
# ---------------------------------------------------------------------------

_cache = {}


def _cached(key, **kw):
    if key not in _cache:
        _cache[key] = _run_guide(**kw)
    return _cache[key]


@pytest.mark.slow_physics
def test_alpha_envelope_regression_lock():
    """DIAGNOSTIC pin (not a physics pass): the measured envelope itself.
    alpha at f0 stays within +-5% of the recorded MEASURED_ALPHA, and the
    run witnesses stay clean — so any regression OR improvement of the
    one-cell sheet realization surfaces instead of drifting silently
    behind the xfail'd contract gate below. Provenance: 2026-08-19
    envelope run, fit ln-RMS resid 0.00206, settle -79.8 dB."""
    out = _cached("base")
    alpha = out["alpha"][0]
    assert abs(alpha / MEASURED_ALPHA - 1.0) <= 0.05, (
        f"measured alpha moved: {alpha:.5f} vs recorded {MEASURED_ALPHA}")
    # forward-wave-purity witness (envelope run measured 0.00206 ln-RMS)
    assert out["resid"][0] < 0.02, out["resid"][0]
    # ring-down settling witness (repo rule; envelope run: -79.8 dB)
    assert out["settle_db"] < -40.0, out["settle_db"]


@pytest.mark.slow_physics
@pytest.mark.xfail(
    strict=True,
    reason="R2-STOP (issue #669, 2026-08-19): measured envelope 0.18679 > "
           "0.10 at the contract resolution — a one-cell volumetric sigma "
           "sheet under-delivers Leontovich loss at dx = lambda/60 "
           "(alpha(f) linear in f instead of flat; Rs-scale-independent). "
           "Pre-declared mesh falsifier RUN 2026-08-19 and NOT resolving: "
           "deficit 0.18681/0.19716/0.09367 at dx = 0.500/0.250/0.125 mm "
           "(non-monotone), with x = sigma_eff*dt/(2*eps0) = 54.19 fixed "
           "on all three by construction; and the fitted alpha is a span "
           "average of a non-exponential profile whose half-span fits "
           "differ by 19.3-36.9%. Fix the COMPARATOR before the mechanism "
           "— see module docstring. "
           "Named alternatives: exponential-stepping sheet coefficient; "
           "multi-pole SIBC. Do NOT loosen the 0.15 cap; a future pass "
           "must remove this marker explicitly.",
)
def test_alpha_oracle_o3():
    """O3 (contract gate, currently RED — see xfail reason): |alpha_meas/
    alpha_analytic - 1| <= ALPHA_GATE (0.15 contract hard cap)."""
    out = _cached("base")
    assert not any("PreflightError" in w for w in out["warnings"])
    alpha = out["alpha"][0]
    err = abs(alpha / ALPHA_ANALYTIC - 1.0)
    assert err <= ALPHA_GATE, (
        f"alpha={alpha:.5f} Np/m vs analytic {ALPHA_ANALYTIC:.5f} "
        f"(err {err:.3%} > gate {ALPHA_GATE:.2f}); resid={out['resid'][0]:.4f}")


@pytest.mark.slow_physics
def test_sqrt_sigma_discriminator_o4a():
    """O4a: sigma_bulk x4 => alpha ratio in [0.40, 0.60] (Leontovich
    predicts 0.50; the DC sheet model predicts 0.25 — outside the gate,
    so wrong-model wiring fails here)."""
    a1 = _cached("base")["alpha"][0]
    a4 = _cached("sig4", sigma_bulk=4e4)["alpha"][0]
    ratio = a4 / a1
    assert 0.40 <= ratio <= 0.60, ratio


@pytest.mark.slow_physics
def test_thickness_invariance_o4b():
    """O4b: thickness x2 => |delta alpha|/alpha <= 0.02 (f0 mode is
    thickness-independent; the DC model would halve alpha)."""
    a1 = _cached("base")["alpha"][0]
    a2 = _cached("thick2", thickness=2 * THICKNESS)["alpha"][0]
    assert abs(a2 - a1) / a1 <= 0.02, (a1, a2)


@pytest.mark.slow_physics
def test_pec_control_o4c():
    """O4c: same geometry, sigma_bulk = 5.8e7 and f0 ABSENT (true PEC
    sheets) => alpha_PEC <= 0.05 * alpha_analytic — loss must move only
    when asked."""
    a_pec = _cached("pec", sigma_bulk=5.8e7, f0_mode=False)["alpha"][0]
    assert abs(a_pec) <= 0.05 * ALPHA_ANALYTIC, a_pec
