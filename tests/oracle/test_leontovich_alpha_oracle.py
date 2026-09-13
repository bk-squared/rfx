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
DFT plane registered at z = 3 mm (actual Ez samples at z = 3.25 mm)
along x in [25, 125] mm, with
the fit RMS residual as the forward-wave-purity witness. NEVER probe
time-series + FFT (repo Never-list).

Comparator-first (contract graft from C): before any FDTD number is
gated, `test_comparator_quadrature_reproduces_closed_form` re-derives
alpha from the analytic TEM field and Rs0 by hand quadrature over the
actual grid sampling and asserts it reproduces Rs0/(eta0*b) to rtol 1e-6,
and `test_fit_chain_recovers_synthetic_alpha` shows the fitting chain
recovers a synthetic exponential's alpha to rtol 1e-6.

#947 correction (2026-09-13): the historical records below retain their
original comparisons; the current gate predicts Ez separately from Hy,
using each mode's Ez/Hy = -eta0*kx/k0 and each component's Yee coordinates.
Only Hy amplitudes are fitted. This checks model adequacy and predicts the
unfitted Ez observable; it is not an independent prediction of source launch.
The one-cell absorber boxes now use canonical integer-node bounds. The old
decimal additions left sigma=0 at x=144 mm and displaced eight ramp values.
A controlled repair on current main restores the old profiles and the
0.72494 endpoint pin without changing the sheet operator or either gate.
Evidence: docs/research_notes/2026-09-13_issue947_oracle_repair.md.

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

#677 RE-MEASURE (node-thin exponential-stepping operator, run 2026-08-19,
  this fixture, JAX CPU float32): the R2-STOP record above measured the
  PRE-#677 realization (full-cell volumetric sigma fold). #677 replaced it
  with the node-thin sheet operator (the R2-named alternative
  architecture), and the G6 re-measure on THIS fixture reads
  alpha_fit(10 GHz) = 0.69823 Np/m (envelope 0.33806), two-plane-ratio
  alpha = 0.72494 (envelope 0.31273) — WORSE than the slab's 0.85779, with
  a steeper frequency slope (alpha(f) = 0.231/0.464/0.698/0.874/1.014 at
  8..12 GHz). Witnesses: fit ln-RMS resid 0.00245 (< 0.02), settle
  -71.0 dB, preflight advisory = the known y-axis cpml_layers clamp only.
  ATTRIBUTION (comparator-first, independent oracle): the same operator on
  a free-standing sheet reproduces the closed-form normal-incidence
  transmission T = 2Rs/(2Rs+eta0) to within 4.4 percent, FREQUENCY-FLAT,
  across 8-12 GHz (test_sheet_transmission_matches_closed_form below) —
  so the delivered sheet dynamics are Rs-flat and correct to a few
  percent, and the remaining guide envelope belongs to THIS FIXTURE'S
  measurement (the COMPARATOR CAVEAT below, now confirmed: the 20 mm
  sub-window decay rates still fall monotonically 0.807 -> 0.582 across
  the span at f0 — a 32 percent half-span disagreement that a 0.002
  ln-RMS residual is blind to). Per the #671 gate policy the envelope
  stays > 0.10, so the O3 gate stays RED (cap 0.15 binds,
  xfail(strict=True) kept with the new fingerprint) and the next O3 step
  remains the guide COMPARATOR, not another sheet mechanism.
  [SUPERSEDED by #700 — see O3 MODEL RE-PAIR below: the comparator step
  was taken; the closed-form pairing is retired as the enforced gate.]

  O4a under the same re-measure: the sqrt-law discriminator SPLIT. Read on
  the clean free-standing-sheet transmission oracle the x4 ratio is 0.5025
  — inside ``O4A_BAND`` [0.40, 0.60], and that leg is the green physics
  tooth. Read on THIS guide fixture the ratio is 0.6089 — outside the same
  band, and it ships as ``test_sqrt_sigma_discriminator_o4a_guide_leg``,
  xfail(strict=True) against the UNWIDENED band, with the measured 0.6089
  regression-locked green in a third test so drift cannot hide under the
  expected failure. The pre-#677 slab's 0.517 quoted in the R2-STOP block
  above is the OLD realization's number and is not the current state.

Mesh-refinement falsifier — pre-declared, run 2026-08-19 (PRE-#677
  realization), gate untouched,
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
  another mechanism attempt on the sheet. [DONE in #700: the caveat's
  non-exponential profile IS the two-mode beat — see O3 MODEL RE-PAIR
  below.]
O3 MODEL RE-PAIR (#700, measured 2026-08-24, this fixture, JAX CPU
  float32): the #700 scout solved the fixture's EXACT transverse spectrum
  with no discretization anywhere and showed the closed form
  alpha = Rs/(eta0*b) is NOT an eigenvalue of the structure as built: the
  1-cell PEC backing stubs (the #642-avoidance choice above) make the
  z-stack PEC | g air | sheet | b air | sheet | g air | PEC a 4-conductor
  line whose spectrum holds a strictly lossless z-uniform TEM supermode
  (uniform Hy -> zero current jump at each sheet -> zero dissipation), a
  symmetric lossy supermode at 6.326 Np/m (f0) and an antisymmetric one
  at 5.273 Np/m — nothing near 1.055. The gap-only launch beats the
  lossless mode against the symmetric lossy one, and the fitted alpha is
  that transient's slope. This one mechanism owns every recorded symptom:
  the 0.33806 envelope, the linear-in-f fingerprint (the launch split
  |c_lossy/c_lossless| moves 0.41 -> 1.35 across 8-12 GHz — reproduced
  by this file's model fit), the non-exponential sub-window profile
  (0.807 -> 0.582), and the dx-immunity of the deficit (0.3381 -> 0.3323
  at half the cell). The sheet operator is exonerated independently by
  energy closure: measured dissipation over Rs|J|^2/2 = 0.992-1.016 at
  every bin (scout s700_c). The enforced O3 gate is therefore RE-PAIRED
  with the exact 4-conductor model (tests/_transverse_resonance_o3.py):
  per-bin measured alpha vs the model's two-mode-transient prediction on
  this probe span, both extraction routes (Ez midplane fit, Hy midplane
  fit), gate O3_MODEL_GATE = gate_from_envelope(0.05677) = 0.09 (repo
  x1.5 rule, quantum=100) — an envelope in the few-percent class where
  the closed-form pairing needed a 0.15 cap and still sat 2.3x outside
  it. Measured re-pair table (settle -71.0 dB; preflight = the known
  y-axis cpml_layers clamp advisory, #647 class, quoted in the run log):
    f (GHz)               8        9       10       11       12
    model fit rel rms   0.0056   0.0026   0.0033   0.0034   0.0038
    alpha_model         0.21865  0.44116  0.66653  0.85158  0.99350
    alpha_Ez-fit (err)  0.23106  0.46383  0.69823  0.87370  1.01433
                        (5.68%)  (5.14%)  (4.76%)  (2.60%)  (2.10%)
    alpha_Hy-fit (err)  0.23103  0.45586  0.67940  0.86829  1.00813
                        (5.66%)  (3.33%)  (1.93%)  (1.96%)  (1.47%)
  The closed form STAYS in this file as the documented LIMIT ANCHOR, not
  as the gate: it is what the model's symmetric lossy supermode converges
  to as the stub term Rs/(2*eta0*g) vanishes (limit-reduction self-check,
  rel err b/(2g), gated), and the measured f0 distance to it (0.33806)
  stays pinned through MEASURED_ALPHA in the regression lock — a true
  statement about the fixture geometry, not about the sheet. The O4a
  guide leg (0.6089, xfail) shares this mechanism qualitatively but is
  NOT quantified by #700 and its gate is untouched.
  MUTATION RECORD (2026-08-24; each mutation applied alone in a copy of
  this tree, the named tests re-run, then reverted — the committed state
  is the green direction, 12 passed + 1 xfailed):
    M1 model stub term deleted (z_top: through(Z, g) -> through(Z, 0.0)):
       test_o3_model_limit_reduces_to_closed_form RED ("RuntimeError:
       symmetric lossy supermode not found near alpha ~ 6.3289");
       test_o3_model_fits_measured_field RED ("model fit not trustworthy
       at 8 GHz: 0.1356 > 0.01"); test_alpha_oracle_o3 RED.
    M2 closed-form pairing reintroduced (a_model = ALPHA_ANALYTIC):
       test_alpha_oracle_o3 RED ("Ez-fit alpha at 8 GHz: 0.23106 vs
       model 1.05482 (err 78.095% > gate 0.09)"); at f0 the deficit is
       the recorded 33.8% (#677-era; the pre-#677 slab measured 18.7%).
    M3 envelope tightened below the measurement (0.05677 -> 0.03, pin
       0.05): test_alpha_oracle_o3 RED ("Ez-fit alpha at 8 GHz: 0.23106
       vs model 0.21865 (err 5.677% > gate 0.05)").
    M4 Yee z-registration broken (z_nodes (k+1/2)*DX -> k*DX):
       test_o3_model_fits_measured_field RED ("model fit not trustworthy
       at 8 GHz: 0.1242 > 0.01") and test_alpha_oracle_o3 RED — the
       field-fit gate senses the half-cell Hy stagger, so it is a real
       registration witness, not a rubber stamp.

Full per-bin dump, alpha(f) trace, verbatim preflight output, the
mesh-refinement sweep with its per-window decay-rate table, and the
sigma_bulk = 1e6 deep-screened evidence case: docs/research_notes/
2026-08-19_leontovich_sheet_envelope.md (local-only). #700 scout report
(exact-spectrum solve, energy closure, blind-predicted g/dx/b sweeps):
issue #700 verdict comment, 2026-08-24.
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
from tests import _transverse_resonance_o3 as _trm

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

# ---- measured envelope (see module docstring: #677 RE-MEASURE) ----
MEASURED_ALPHA = 0.69823           # Np/m at f0, node-thin operator (#677)
MEASURED_ALPHA_TWO_PLANE = 0.72494  # original #677 endpoint-ratio record
# Restored by #947's absorber-only comparison: 0.87333 -> 0.72494797,
# alpha_fit 0.71565 -> 0.69823741. The 2026-09-07 re-pin attributed the
# change to sheet rims without isolating the malformed absorber. Current
# sheet ownership is retained; correcting nine ramp cells recovers the
# historical profile. The diagnostic tolerance remains 5%.
MEASURED_ENVELOPE = 0.33806        # |alpha_fit/alpha_analytic - 1| — the
#   closed-form pairing's envelope, kept as the documented LIMIT-ANCHOR
#   record (#700): the fixture's alpha_fit is 34% below Rs/(eta0*b)
#   because the closed form is not an eigenvalue of this 4-conductor
#   fixture, not because the sheet under-dissipates.
MEASURED_GUIDE_SQRT_RATIO = 0.6089  # alpha(4*sigma)/alpha(sigma), guide fit

# ---- #700 re-pair: exact 4-conductor comparator (tests/_transverse_resonance_o3) ----
O3_FREQS = (8e9, 9e9, 10e9, 11e9, 12e9)
G_STUB = Z_SHEET_LO                # PEC backing-stub depth = 0.5 mm (1 cell)
RS0 = float(leontovich_rs(F0, SIGMA_BULK))
# measured 2026-08-24 (this fixture, JAX CPU float32): per-bin
# |alpha_meas/alpha_model - 1| worst case over both extract routes
# (Ez midplane fit and Hy midplane fit) and all five bins:
O3_MODEL_ENVELOPE = 0.05677        # worst bin (8 GHz, Ez route)
O3_MODEL_GATE = gate_from_envelope(O3_MODEL_ENVELOPE, quantum=100)
assert O3_MODEL_GATE == 0.09, O3_MODEL_GATE
# The historical 0.26-0.56% Hy fit envelope sets a 1% trust gate.
# #947 restores that envelope by fixing the absorber declaration, not by
# widening this gate. Analytic eigenvalues/shapes stay fixed; amplitudes
# ARE data-fitted, so the resulting alpha prediction can change with the
# measured field. Hy and Ez also have distinct modal wave impedances.
O3_FIELD_FIT_RMS_GATE = 0.01

# O4a band: Leontovich predicts alpha ~ sqrt(1/sigma), so a x4 in
# sigma_bulk halves the loss -> 0.50. A DC (thickness-fold) sheet would
# predict 0.25. The [0.40, 0.60] window separates the two models and is
# the HISTORICAL band — it is not widened here. See the two O4a tests: the
# clean transmission oracle passes it (0.5025) and the guide fit does not
# (0.6089), which is why the guide leg ships xfail(strict=True).
O4A_BAND = (0.40, 0.60)
# The closed-form pairing (|alpha/alpha_analytic - 1| vs a 0.15 cap,
# xfail(strict=True) since the 2026-08-19 R2-STOP) is RETIRED as the
# enforced O3 gate by #700: the closed form is not an eigenvalue of this
# fixture (see O3 MODEL RE-PAIR in the module docstring), so gating
# against it measured the stub geometry, not the sheet. The closed form
# stays in this file as the documented limit anchor
# (test_comparator_quadrature_reproduces_closed_form and the model's
# limit-reduction self-check); the enforced gate is now
# test_alpha_oracle_o3 against the exact 4-conductor model with
# O3_MODEL_GATE above.


def _build_guide(sigma_bulk=SIGMA_BULK, *, f0_mode=True,
                 thickness=THICKNESS, freqs=(F0,)):
    sim = Simulation(
        freq_max=10e9, domain=DOMAIN, dx=DX,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="pec"),
                              y="pmc", z="pec"),
        cpml_layers=10,
    )
    # graded in-guide absorber (full cross-section, ends on hi-x PEC)
    # Use the grid's node arithmetic for BOTH faces of every slab. Adding
    # DX to a decimal x0 can move a half-open face by one float64 ulp,
    # leaving a hole in this one-cell ramp after material rasterization.
    absorber_i0 = int(round(ABSORBER_X0 / DX))
    for i in range(ABSORBER_N):
        s = ABSORBER_SIGMA_MAX * ((i + 0.5) / ABSORBER_N) ** 2
        x0 = (absorber_i0 + i) * DX
        x1 = (absorber_i0 + i + 1) * DX
        sim.add_material(f"abs{i}", eps_r=1.0, sigma=s)
        sim.add(Box((x0, 0.0, 0.0), (x1, DOMAIN[1], DOMAIN[2])),
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
    # #700 model comparator instrumentation: full Hy(x, z) at the y
    # midplane. A DFT accumulator is a passive observer — adding it does
    # not perturb the time stepping (scout-verified: the instrumented run
    # reproduces MEASURED_ALPHA = 0.69823 bit-for-bit at f0).
    sim.add_dft_plane_probe(axis="y", coordinate=0.001, component="hy",
                            freqs=jnp.asarray(freqs), name="yhy")
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
    # #700: complex Hy(x, z) over the fit span for the model comparator.
    # Hy z-nodes sit at (k + 1/2)*DX on the Yee grid (12 nodes across the
    # 6 mm stack, z = 0 at the bottom PEC).
    hy = np.asarray(result.dft_planes["yhy"].accumulator)
    kz0 = grid.pad_z_lo
    nz = int(round(DOMAIN[2] / DX))
    out["hy_plane"] = hy[:, i0:i1 + 1, kz0:kz0 + nz]
    out["hy_xs"] = xs + 0.5 * DX
    out["z_nodes"] = (np.arange(nz) + 0.5) * DX
    out["ez_z"] = (
        result.dft_planes["midplane"].index - grid.pad_z_lo + 0.5) * DX
    ts = np.abs(np.asarray(result.time_series)[:, 0])
    tail = ts[int(0.95 * len(ts)):].max()
    out["settle_db"] = float(20 * np.log10(max(tail, 1e-300) / ts.max()))
    out["warnings"] = [str(w.message) for w in rec]
    return out


# ---------------------------------------------------------------------------
# Comparator-first: validate the measurement chain with no FDTD involved
# ---------------------------------------------------------------------------

def test_absorber_realizes_the_declared_ramp_without_gaps_or_overwrites():
    """No solve: all 120 distinct conductivities tile the declared volume.

    Check the assembled full array, including zero conductivity outside
    the absorber and its half-open y/z extents. Checking only Box widths
    missed the decimal-bound singleton/two-node ambiguity behind #947.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _build_guide()
        grid = sim._build_grid()
        materials, *_ = sim._assemble_materials(grid)
    sigma = np.asarray(materials.sigma)
    expected = np.zeros_like(sigma)
    start = int(round(ABSORBER_X0 / DX)) + grid.pad_x_lo
    ny, nz = (int(round(length / DX)) for length in DOMAIN[1:])
    values = np.asarray([
        ABSORBER_SIGMA_MAX * ((i + 0.5) / ABSORBER_N) ** 2
        for i in range(ABSORBER_N)], dtype=sigma.dtype)
    expected[start:start + ABSORBER_N,
             grid.pad_y_lo:grid.pad_y_lo + ny,
             grid.pad_z_lo:grid.pad_z_lo + nz] = values[:, None, None]
    np.testing.assert_array_equal(sigma, expected)


def test_ez_prediction_recovers_tem_impedance_and_physical_phase_reference():
    """A prescribed TEM wave has Ez=-eta0*Hy at every sample location."""
    k0 = 2 * np.pi * F0 / _trm.C0
    xs = np.array([0.017, 0.029, 0.045])
    zs = np.array([0.00025, 0.00325, 0.00575])
    amplitude, x_reference = 0.7 + 0.3j, 0.02625
    fit = {"modes": [k0], "amps": [amplitude], "x_reference": x_reference}
    actual = _trm.predict_ez_from_hy_fit(
        fit, xs, zs, F0, B_PLATE, G_STUB, RS0, ETA_0)
    expected = -ETA_0 * amplitude * np.exp(-1j * k0 * (xs - x_reference))
    np.testing.assert_allclose(actual, np.broadcast_to(expected[:, None], actual.shape),
                               rtol=1e-13, atol=1e-13)


def test_multimode_ez_prediction_does_not_reuse_the_hy_alpha():
    """An exact prescribed mixture falsifies a shared Ez/Hy alpha gate.

    The lossy/TEM ratio at z_E is chosen as 0.25j without a data search.
    Fit only the synthetic Hy plane, then predict Ez at its half-cell
    shifted x samples. Its exact alpha differs from Hy by >9%, despite
    both belonging to the same analytic solution of Maxwell's equations.
    """
    f = O3_FREQS[0]
    k0 = 2 * np.pi * f / _trm.C0
    ks = _trm.find_symmetric_lossy_mode(f, B_PLATE, G_STUB, RS0, ETA_0)
    xs_e = np.arange(201) * DX + FIT_X[0]
    xs_h = xs_e + DX / 2
    zs = (np.arange(12) + 0.5) * DX
    z_e = 3.25e-3
    phi = _trm.hy_profile(ks, f, B_PLATE, G_STUB, RS0, ETA_0, zs)
    phi_e = _trm.hy_profile(ks, f, B_PLATE, G_STUB, RS0, ETA_0, [z_e])[0]
    q = 0.25j
    distance_h = xs_h - xs_h[0]
    hy = (np.exp(-1j * k0 * distance_h)[:, None]
          + q / phi_e * np.exp(-1j * ks * distance_h)[:, None] * phi[None, :])
    fit = _trm.fit_hy_field(xs_h, zs, hy, f, B_PLATE, G_STUB, RS0, ETA_0)
    assert fit["rel_resid"] < 1e-10
    predicted = _trm.predict_ez_from_hy_fit(
        fit, xs_e, [z_e], f, B_PLATE, G_STUB, RS0, ETA_0)[:, 0]
    distance_e = xs_e - xs_h[0]
    expected = -ETA_0 * (np.exp(-1j * k0 * distance_e)
                        + ks / k0 * q * np.exp(-1j * ks * distance_e))
    np.testing.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)
    alpha_e = _trm._fit_alpha_loglin(xs_e, np.abs(expected))
    assert abs(alpha_e / fit["alpha_model"] - 1) > O3_MODEL_GATE
    assert abs(_trm._fit_alpha_loglin(xs_e, np.abs(predicted)) - alpha_e) < 1e-9


def test_the_guide_plates_realize_two_planes_spanning_the_cross_section():
    """Build-time (no solve): where the plates are, and how wide.

    Every alpha here is a per-unit-length quantity measured between two
    surface-impedance plates, so the plates' realized planes and footprint
    ARE the fixture. The declaration is a zero-extent Box on each of
    z = 0.5 mm and 5.5 mm spanning the full cross-section; this reads back
    what the lattice ownership contract realizes for it (#931 §1.3).

    The closed sheet footprint includes both y rims and the hi-x node.
    The y-PMC condition zeros Hx/Hz, not Hy, so boundary type alone cannot
    establish zero Ex-sheet-current dissipation there. This build check
    asserts placement only. #947's controlled absorber repair retains these
    sheet masks and recovers the historical field profile.

    A note for whoever generalizes ``tests/_realized_geometry.realized``:
    it cannot be used here. This fixture declares f0 sheets and dielectric
    Boxes only, so there is no cell mask, no PEC sheet and no wire, and
    ``realized_pec_edge_masks`` refuses an empty realization by design.
    """
    import warnings as _w

    sim = _build_guide()
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        grid = sim._build_grid()
        specs = []
        _m, _d, _l, pec, *_rest = sim._assemble_materials(
            grid, sheet_specs=specs)
    assert pec is None or not bool(np.asarray(pec).any()), (
        "the plates are f0 sheets and the graded absorber is dielectric; "
        "nothing here is a conductor VOLUME")
    assert len(specs) == 2, len(specs)

    z_nodes = np.asarray(grid.z if hasattr(grid, "z") else None, dtype=float) \
        if getattr(grid, "z", None) is not None else None
    planes = sorted(int(sp.plane) for sp in specs)
    if z_nodes is not None:
        expect = sorted(int(np.argmin(np.abs(z_nodes - z)))
                        for z in (Z_SHEET_LO, Z_SHEET_HI))
        assert planes == expect, (planes, expect)
    assert planes[1] - planes[0] == int(round((Z_SHEET_HI - Z_SHEET_LO) / DX)), \
        planes

    for sp in specs:
        m = np.asarray(sp.mask, dtype=bool)
        occ = np.argwhere(m)
        assert int(occ[:, 1].max() - occ[:, 1].min() + 1) == m.shape[1], (
            "a plate must span the whole y cross-section; the closed "
            "footprint includes both PMC rim rows")
        assert int(occ[:, 2].max()) == int(occ[:, 2].min()) == sp.plane, (
            "a sheet occupies exactly its own plane")


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


_F0_IDX = O3_FREQS.index(F0)


def _base():
    """The shared base run — all five O3_FREQS bins in one run (#700);
    single-frequency consumers index the F0 bin via _F0_IDX."""
    return _cached("base", freqs=O3_FREQS)


def _alpha_two_plane(xs, mag):
    """Two-plane amplitude-ratio alpha at fixed separation — the #669
    COMPARATOR CAVEAT's named alternative extractor (#677 G6): no
    single-exponential assumption, just the endpoint amplitude ratio."""
    return float(np.log(mag[0] / mag[-1]) / (xs[-1] - xs[0]))


def test_two_plane_comparator_recovers_synthetic_alpha():
    """G6 comparator-first: the two-plane extractor recovers a synthetic
    exponential's alpha within 1% (exactly, in fact) BEFORE any FDTD
    number is attributed through it."""
    xs = np.arange(FIT_X[0], FIT_X[1] + 1e-12, DX)
    a = _alpha_two_plane(xs, np.exp(-ALPHA_ANALYTIC * xs))
    assert abs(a / ALPHA_ANALYTIC - 1.0) < 0.01
    assert abs(a / ALPHA_ANALYTIC - 1.0) < 1e-9   # measured: exact


@pytest.mark.slow_physics
def test_alpha_envelope_regression_lock():
    """DIAGNOSTIC pin (not a physics pass): the measured envelope itself.
    alpha at f0 stays within +-5% of the recorded MEASURED_ALPHA, and the
    run witnesses stay clean — so any regression OR improvement of the
    node-thin sheet realization surfaces instead of drifting silently
    behind the xfail'd contract gate below. Provenance: 2026-08-19 #677
    re-measure, fit ln-RMS resid 0.00245, settle -71.0 dB, two-plane
    alpha 0.72494."""
    out = _base()
    alpha = out["alpha"][_F0_IDX]
    a2 = _alpha_two_plane(out["xs"], out["profile"][_F0_IDX])

    # R5 trace, added 2026-09-07 (#931). The two extractors disagreed about
    # whether this fixture moved — the span-average fit stayed inside its 5 %
    # pin while the endpoint ratio went 20 % — and neither the profile nor the
    # endpoints it reads were ever printed, so the disagreement could not be
    # read. The endpoint ratio uses only profile[0] and profile[-1]; the fit
    # uses all of it. A profile whose ENDS move relative to its middle moves
    # one and not the other, which is the "non-exponential profile / two-mode
    # beat" this module's docstring already describes.
    prof = np.asarray(out["profile"][_F0_IDX], dtype=float)
    xs = np.asarray(out["xs"], dtype=float)
    print(f"\n[LEONTOVICH/ENVELOPE] alpha_fit={alpha:.5f} (pin "
          f"{MEASURED_ALPHA}), alpha_two_plane={a2:.5f} (pin "
          f"{MEASURED_ALPHA_TWO_PLANE}), ln-RMS resid="
          f"{out['resid'][_F0_IDX]:.5f}, settle={out['settle_db']:.1f} dB")
    print(f"[LEONTOVICH/ENVELOPE] endpoints used by the two-plane extractor: "
          f"|E|(x={xs[0] * 1e3:.2f} mm)={prof[0]:.6g}, "
          f"|E|(x={xs[-1] * 1e3:.2f} mm)={prof[-1]:.6g}, "
          f"ratio={prof[0] / prof[-1]:.6f}")
    for i in range(0, len(xs), max(1, len(xs) // 25)):
        print(f"[LEONTOVICH/PROFILE] {xs[i] * 1e3:8.3f} mm  {prof[i]:.6g}  "
              f"ln={np.log(prof[i]):.5f}")

    assert abs(alpha / MEASURED_ALPHA - 1.0) <= 0.05, (
        f"measured alpha moved: {alpha:.5f} vs recorded {MEASURED_ALPHA}")
    # Endpoint ratio is a distinct extractor of the same record. The
    # absorber-only A/B recovers the original pin; this is not a claim
    # that a multimode guide has one exponential attenuation constant.
    assert abs(a2 / MEASURED_ALPHA_TWO_PLANE - 1.0) <= 0.05, a2
    # forward-wave-purity witness (re-measure run: 0.00245 ln-RMS)
    assert out["resid"][_F0_IDX] < 0.02, out["resid"][_F0_IDX]
    # ring-down settling witness (repo rule; re-measure run: -71.0 dB)
    assert out["settle_db"] < -40.0, out["settle_db"]


def test_o3_model_limit_reduces_to_closed_form():
    """LIMIT-REDUCTION self-check for the #700 model comparator (house
    comparator rule, research/CLAUDE.md: a new comparator must reproduce
    a known-good limit as a checkable artifact in its own tests).

    As the PEC backing stub deepens the stub-current loss term vanishes
    and the exact symmetric lossy supermode's alpha must converge to the
    closed form Rs/(eta0*b) = 1.05482 Np/m, following the rate law
    rel_err = b/(2g) (see tests/_transverse_resonance_o3 docstring).
    Gated, not printed: per-step the residual must shrink (log-space,
    ratio <= 0.55; measured ~0.50 per g-doubling), track b/(2g) to 2%
    relative, and end below 0.10 (measured 0.0784 at g = 32 mm).
    Measured ladder 2026-08-24: rel err 4.99747 / 2.49953 / 1.24989 /
    0.62497 / 0.31250 / 0.15628 / 0.07842 for g = 0.5..32 mm.
    """
    rel_prev = None
    for g in (0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3, 16e-3, 32e-3):
        kx = _trm.find_symmetric_lossy_mode(F0, B_PLATE, g, RS0, ETA_0)
        rel = -kx.imag / ALPHA_ANALYTIC - 1.0
        assert rel > 0.0, (g, rel)
        assert abs(rel / (B_PLATE / (2.0 * g)) - 1.0) < 0.02, (g, rel)
        if rel_prev is not None:
            assert rel <= 0.55 * rel_prev, (g, rel, rel_prev)
        rel_prev = rel
    assert rel_prev < 0.10, rel_prev
    # and the base fixture's spectrum has NO mode at the closed form:
    # the nearest lossy mode sits at ~6x it (6.326 Np/m at f0)
    modes = _trm.find_modes(F0, B_PLATE, G_STUB, RS0, ETA_0)
    lossy = [-m.imag for m in modes if -m.imag > 0.1]
    assert lossy and min(lossy) > 4.0 * ALPHA_ANALYTIC, modes


def _model_fits(out):
    """Per-bin exact-model fits to the measured Hy(x, z) plane (cached)."""
    if "model_fits" not in _cache:
        _cache["model_fits"] = [
            _trm.fit_hy_field(out["hy_xs"], out["z_nodes"],
                              out["hy_plane"][fi], f, B_PLATE, G_STUB,
                              RS0, ETA_0)
            for fi, f in enumerate(O3_FREQS)]
        for f, fit in zip(O3_FREQS, _cache["model_fits"]):
            ez = _trm.predict_ez_from_hy_fit(
                fit, out["xs"], [out["ez_z"]], f, B_PLATE, G_STUB, RS0, ETA_0)
            fit["alpha_model_ez"] = _trm._fit_alpha_loglin(out["xs"], np.abs(ez[:, 0]))
    return _cache["model_fits"]


def _print_model_fit_census(out):
    """R5 census for the O3 pair — every bin, both routes, printed BEFORE
    any assertion (2026-09-07, #931).

    The two O3 tests both trip on the trust precondition at ONE bin and
    say so with one number, which is not enough to tell "the model got
    worse everywhere" from "one bin moved". This prints the whole table
    so the next reader classifies by inspection instead of by argument.
    Printed once per session; the fits themselves are cached."""
    if _cache.get("census_printed"):
        return
    _cache["census_printed"] = True
    fits = _model_fits(out)
    print(f"\n[LEONTOVICH/O3-CENSUS] gate: fit trust <= "
          f"{O3_FIELD_FIT_RMS_GATE}, model err <= {O3_MODEL_GATE}; "
          f"settle {out['settle_db']:.1f} dB")
    print("[LEONTOVICH/O3-CENSUS]  f(GHz)  fit_rel_rms  model_Ez  "
          "alpha_Ez  err_Ez   model_Hy  alpha_Hy  err_Hy")
    for fi, (f, ft) in enumerate(zip(O3_FREQS, fits)):
        a_model = ft["alpha_model"]
        a_model_ez = ft["alpha_model_ez"]
        a_ez = out["alpha"][fi]
        a_hy = ft["alpha_meas"]
        print(f"[LEONTOVICH/O3-CENSUS]  {f/1e9:5.1f}   {ft['rel_resid']:10.5f}  "
              f"{a_model_ez:8.5f}  {a_ez:8.5f}  {abs(a_ez/a_model_ez-1):6.2%}  "
              f"{a_model:8.5f}  {a_hy:8.5f}  {abs(a_hy/a_model-1):6.2%}")


@pytest.mark.slow_physics
def test_o3_model_fits_measured_field():
    """FIELD-FIT self-check for the #700 model comparator (house rule,
    part 2): on THIS committed fixture the 3-supermode expansion (three
    complex amplitudes fitted, mode shapes and kx exact) must reproduce
    the measured 2-D Hy(x, z) over the whole fit span to <= 1% relative
    rms at every bin (O3_FIELD_FIT_RMS_GATE; scout measured 0.26-0.56%).
    A field this well described by {lossless TEM + symmetric lossy
    supermode} IS the mechanism statement of #700: the fitted alpha is a
    two-mode transient, not an eigenvalue."""
    out = _base()
    _print_model_fit_census(out)
    for f, ft in zip(O3_FREQS, _model_fits(out)):
        assert ft["rel_resid"] <= O3_FIELD_FIT_RMS_GATE, (
            f"model fit degraded at {f/1e9:.0f} GHz: rel rms "
            f"{ft['rel_resid']:.4f} > {O3_FIELD_FIT_RMS_GATE}")


@pytest.mark.slow_physics
def test_alpha_oracle_o3():
    """O3 (contract gate, RE-PAIRED by #700 — GREEN): per-bin measured
    alpha vs the exact 4-conductor model's multimode prediction
    for this probe span, |alpha_meas/alpha_model - 1| <= O3_MODEL_GATE,
    on BOTH extraction routes (Ez midplane fit — the fixture's canonical
    extractor — and Hy midplane fit, the field the model is fitted on).

    #947 predicts each component at its actual Yee samples. Ez uses the
    same Hy-fitted amplitudes and mode-specific impedance, with no E fit.
    The Hy comparison measures model adequacy, not independent source
    prediction; a multimode Ez/Hy pair need not share an alpha.

    The pre-#700 pairing gated |alpha_meas/ALPHA_ANALYTIC - 1| against a
    0.15 cap and was strict-xfail RED at envelope 0.33806. #700 showed
    the closed form Rs/(eta0*b) is not an eigenvalue of this fixture (the
    1-cell PEC backing stubs make it a 4-conductor line whose spectrum
    holds a lossless supermode and a 6.33 Np/m lossy one — nothing at
    1.055), so that red measured stub geometry, not sheet physics. The
    model prediction uses the same probe span; only the three complex
    launch amplitudes are fitted (the trust gate above bounds the fit).
    Envelope provenance for O3_MODEL_GATE: see O3 MODEL RE-PAIR in the
    module docstring."""
    out = _base()
    _print_model_fit_census(out)
    assert not any("PreflightError" in w for w in out["warnings"])
    assert out["settle_db"] < -40.0, out["settle_db"]
    for fi, (f, ft) in enumerate(zip(O3_FREQS, _model_fits(out))):
        assert ft["rel_resid"] <= O3_FIELD_FIT_RMS_GATE, (
            f"model fit not trustworthy at {f/1e9:.0f} GHz: "
            f"{ft['rel_resid']:.4f}")
        for route, a_meas, a_model in (
                ("Ez-fit", out["alpha"][fi], ft["alpha_model_ez"]),
                ("Hy-fit", ft["alpha_meas"], ft["alpha_model"])):
            err = abs(a_meas / a_model - 1.0)
            assert err <= O3_MODEL_GATE, (
                f"{route} alpha at {f/1e9:.0f} GHz: {a_meas:.5f} vs "
                f"model {a_model:.5f} (err {err:.3%} > gate "
                f"{O3_MODEL_GATE}); fit resid={out['resid'][fi]:.4f}, "
                f"settle={out['settle_db']:.1f} dB")


def _guide_sqrt_ratio():
    """alpha(4*sigma)/alpha(sigma) from the guide fit (cached runs)."""
    a1 = _base()["alpha"][_F0_IDX]
    a4 = _cached("sig4", sigma_bulk=4e4)["alpha"][0]
    return float(a4 / a1)


@pytest.mark.slow_physics
def test_sqrt_sigma_discriminator_o4a():
    """O4a (PHYSICS tooth, GREEN): sigma_bulk x4 => loss ratio inside the
    historical ``O4A_BAND`` (Leontovich predicts 0.50; a DC thickness-fold
    sheet predicts 0.25 — outside the band, so wrong-model wiring fails
    here).

    #677 retarget: the discriminant reads the clean free-standing-sheet
    transmission oracle, where the x4 ratio measured 0.5025. The guide-fit
    ratio moved to 0.6089 under the node-thin operator, OUTSIDE the same
    band, and is asserted against that same unwidened band in the
    xfail(strict=True) sibling below rather than being folded into this
    test's pass. Moving the tooth to the uncontaminated observable is a
    retarget with a named root cause (the guide profile is
    non-exponential — see the #677 RE-MEASURE record and the O3 xfail
    reason); it is not a relaxation of the band, which is byte-identical
    to what shipped before #677."""
    t1 = _transmission_spectra(SIGMA_BULK)
    t4 = _transmission_spectra(4 * SIGMA_BULK)
    ratio_t = float(t4[2] / t1[2])          # f0 bin
    lo, hi = O4A_BAND
    assert lo <= ratio_t <= hi, ratio_t


@pytest.mark.slow_physics
@pytest.mark.xfail(
    strict=True,
    reason="#677 re-measure (2026-08-19): the GUIDE-fit sqrt-law ratio is "
           "0.6089, outside the unwidened historical O4A_BAND "
           "[0.40, 0.60] — RED, and kept red rather than accommodated. "
           "Same attribution as the O3 xfail above: the independent "
           "free-standing-sheet transmission oracle reproduces the x4 "
           "ratio at 0.5025 (test_sqrt_sigma_discriminator_o4a) and the "
           "closed form to 4.4% frequency-flat, so the excursion is the "
           "guide fixture's span-average contamination, not the sheet's "
           "sigma scaling. Fix the guide comparator/fixture before "
           "touching the operator. Do NOT widen O4A_BAND; a future pass "
           "must remove this marker explicitly. The measured value itself "
           "is regression-locked GREEN in "
           "test_o4a_guide_ratio_regression_lock, so drift away from "
           "0.6089 surfaces there rather than hiding under this xfail.",
)
def test_sqrt_sigma_discriminator_o4a_guide_leg():
    """O4a guide leg (contract band, currently RED — see xfail reason).

    #700 scope note: the stub-supermode mechanism that the O3 re-pair
    quantified (see O3 MODEL RE-PAIR in the module docstring)
    qualitatively explains this excursion too — the guide-fit ratio reads
    a two-mode transient whose launch split moves with sigma — but it is
    NOT quantified for this observable; #700 deliberately leaves this
    gate untouched."""
    ratio = _guide_sqrt_ratio()
    lo, hi = O4A_BAND
    assert lo <= ratio <= hi, ratio


@pytest.mark.slow_physics
def test_o4a_guide_ratio_regression_lock():
    """DIAGNOSTIC pin (not a physics pass): the guide-fit sqrt-law ratio
    stays within +-5% of ``MEASURED_GUIDE_SQRT_RATIO``.

    Same +-5% relative convention as ``test_alpha_envelope_regression_lock``
    on this fixture's other measured quantities — a lock on what WAS
    measured, deliberately not a two-sided physics window centred anywhere
    convenient. It is GREEN on purpose: an assertion buried inside the
    xfail(strict=True) sibling would be swallowed by the expected failure,
    so drift in either direction has to be checked from outside it."""
    ratio = _guide_sqrt_ratio()
    assert abs(ratio / MEASURED_GUIDE_SQRT_RATIO - 1.0) <= 0.05, (
        f"guide sqrt-law ratio moved: {ratio:.5f} vs recorded "
        f"{MEASURED_GUIDE_SQRT_RATIO}")


@pytest.mark.slow_physics
def test_thickness_invariance_o4b():
    """O4b: thickness x2 => |delta alpha|/alpha <= 0.02 (f0 mode is
    thickness-independent; the DC model would halve alpha)."""
    a1 = _base()["alpha"][_F0_IDX]
    a2 = _cached("thick2", thickness=2 * THICKNESS)["alpha"][0]
    assert abs(a2 - a1) / a1 <= 0.02, (a1, a2)


@pytest.mark.slow_physics
def test_pec_control_o4c():
    """O4c: same geometry, sigma_bulk = 5.8e7 and f0 ABSENT (true PEC
    sheets) => alpha_PEC <= 0.05 * alpha_analytic — loss must move only
    when asked."""
    a_pec = _cached("pec", sigma_bulk=5.8e7, f0_mode=False)["alpha"][0]
    assert abs(a_pec) <= 0.05 * ALPHA_ANALYTIC, a_pec

# ---------------------------------------------------------------------------
# #677: independent closed-form oracle for the sheet OPERATOR itself
# ---------------------------------------------------------------------------

_TRANS_FREQS = (8e9, 9e9, 10e9, 11e9, 12e9)
_trans_cache = {}


def _transmission_spectra(sigma_bulk):
    """|T|(f) through a free-standing f0 sheet — two-run reference ratio.

    Cached per sigma_bulk; the sheet-free reference run is shared. The
    ring-down settle witness (< -40 dB) is asserted on every run.
    """
    from rfx.boundaries.spec import BoundarySpec

    x_sheet, domain = 40e-3, (80e-3, 2e-3, 2e-3)

    def _build(sb):
        sim = Simulation(freq_max=12e9, domain=domain, dx=DX,
                         boundary=BoundarySpec(x="cpml", y="pmc", z="pec"),
                         cpml_layers=10)
        if sb is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim.add_thin_conductor(
                    Box((x_sheet, 0.0, 0.0),
                        (x_sheet, domain[1], domain[2])),
                    sigma_bulk=sb, thickness=THICKNESS,
                    surface_impedance_f0=F0)
        for k in range(4):
            sim.add_source((10e-3, 1e-3, (k + 0.5) * DX), "ez",
                           waveform=GaussianPulse(f0=F0, bandwidth=0.5),
                           amplitude_kind="field")
        sim.add_dft_plane_probe(axis="x", coordinate=60e-3, component="ez",
                                freqs=jnp.asarray(_TRANS_FREQS),
                                name="trans")
        sim.add_probe((60e-3, 1e-3, 1e-3), "ez")
        return sim

    def _spectrum(sb):
        key = "ref" if sb is None else float(sb)
        if key not in _trans_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = _build(sb).run(n_steps=3000, compute_s_params=False)
            spec = np.abs(np.asarray(
                r.dft_planes["trans"].accumulator)).mean(axis=(1, 2))
            ts = np.abs(np.asarray(r.time_series)[:, 0])
            tail = ts[int(0.95 * len(ts)):].max()
            settle = 20 * np.log10(max(tail, 1e-300) / ts.max())
            assert settle < -40.0, (key, settle)   # ring-down witness
            _trans_cache[key] = spec
        return _trans_cache[key]

    return _spectrum(sigma_bulk) / _spectrum(None)


@pytest.mark.slow_physics
def test_sheet_transmission_matches_closed_form():
    """Normal-incidence |T| through a free-standing Rs0 sheet equals the
    closed form T = 2Rs/(2Rs + eta0), frequency-FLAT, at every probed bin.

    This is the attribution witness for the #677 G6 re-measure: it shares
    NOTHING with the guide fixture above (different domain, termination,
    observable and normalization — a two-run reference ratio), so it
    separates "the operator delivers the wrong Rs" from "the guide
    measurement is contaminated". Measured envelope 2026-08-19: worst
    |T/T_analytic - 1| = 0.0444 over 8-12 GHz -> gate
    gate_from_envelope(0.0444) = 0.07 (repo x1.5 rule, quantum=100).
    """
    TRANS_GATE = gate_from_envelope(0.0444, quantum=100)
    assert TRANS_GATE == 0.07   # derivation pin, not a hand-tuned number

    rs0 = float(leontovich_rs(F0, SIGMA_BULK))
    t_analytic = 2 * rs0 / (2 * rs0 + ETA_0)
    T = _transmission_spectra(SIGMA_BULK)
    for f, t in zip(_TRANS_FREQS, T):
        assert abs(t / t_analytic - 1.0) <= TRANS_GATE, (
            f"|T| off closed form at {f/1e9:.0f} GHz: {t:.6f} vs "
            f"{t_analytic:.6f}")
