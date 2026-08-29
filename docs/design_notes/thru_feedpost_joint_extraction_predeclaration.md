# THRU feed-post de-embedding, attempt 2 — joint (L, Zc, l_eff) inversion pre-declaration

Status: BINDING. Committed BEFORE any new measurement. Sections 1-7 are
frozen at commit time; results are APPENDED in later sections, never
edited back. No window below is widened after any measurement; a fired
falsifier is a STOP for its arm. A justified STOP is a valid outcome.

Lane: `agent/thru-deembed-r2` (branched from `agent/thru-deembed`).
Author: thru-diagonal de-embedding author agent (attempt 2), 2026-08-29 KST.
Parent pre-declaration (attempt 1, settled context):
`docs/design_notes/thru_feedpost_deembed_predeclaration.md` including its
post-review correction. Nothing in that note is re-litigated here; its
sections 1-2 (de-embed algebra: exact 2-port wave-cascade removal of
BOTH posts, `rfx.deembed.deembed_series_inductance`; naive Z_in - jwL
refuted; refplane machinery inapplicable and `rfx/probes/refplane.py`
untouched together with the #764 byte-INEQUALITY pin) carry over verbatim.

## 1. Why attempt 2 exists, and what may be reused from attempt 1

Attempt 1's extraction arm STOPped on F-L1: with ASSUMED line constants
(Zc = 48.25 ohm, beta = 1.055 w/c, l = 16 mm) the per-bin single-flat-L
inversion returned a systematic monotone decline (0.282 -> 0.194 nH over
1.4-2.6 GHz, < 1% reproduced across both driven ports). The apparatus
was verified exact on synthetics; the NAMED mechanism is
ill-conditioning in the assumed constants: dL/dZc ~ 0.065 nH/ohm at the
low-f bins (a ~1 ohm effective-Zc error swings L by ~25% and reproduces
the falling signature; +2.25 ohm reverses it) plus 1-cell trace-overhang
electrical-length uncertainty. Under that method, constant bias and true
post dispersion are indistinguishable.

Discipline for attempt 2: attempt-1 DATA inform the DESIGN (which
parameters must be freed, the sensitivity scales, the noise classes) but
its extracted NUMBERS are not targets. No window below is chosen to make
the attempt-1 reading pass; every window is derived from prior physics,
ledger witness classes, or measured noise classes, and each derivation is
shown.

Remedy chosen (attempt-1 follow-up option (b), the CPU-scale one):
**joint (L, Zc, l_eff) inversion of the full complex out-of-band 2-port
data**. Instead of assuming the line constants the inversion measures
them simultaneously from observables that pin them independently of L
(identifiability argument in section 3). Options (a) dedicated
single-post fixture and (c) in-situ #313 two-plane Zc(f) remain the
named next-scale remedies if THIS design's identifiability falsifier
(F-J5) or adequacy falsifiers fire; the VESSL sketch
`validation/research/thru_feedpost_singlepost_vessl.yaml` stays proposal-only.

## 2. Model (frozen)

Fixture: battery-verbatim THRU (identical builder to attempt 1's
harness, byte-shared by import). Between the two lumped port terminal
planes (reference Z0 = 50 ohm):

    port1 -- series x -- lossless line(Zc, theta) -- series x -- port2

with x = j*omega*L (symmetric posts; the fixture is geometrically
symmetric and attempt 1 measured 0.47% port asymmetry — symmetry is
ASSERTED by the model and re-CHECKED by falsifier F-J4, not assumed
silently), and theta = omega * tau the line's electrical angle.

Parameters (3): L (post inductance, henries), Zc (line characteristic
impedance, ohm), tau (one-way electrical delay, seconds). Reported
l_eff = c * tau / 1.055 (the #313 beta-factor center; beta-factor and
length are inseparable in theta, so l_eff is a reporting convention and
its window below folds the beta corners in).

Forward model (exact ABCD cascade, reference Z0):

    ABCD = [[1, x],[0,1]] . [[cos theta, j Zc sin theta],
                             [j sin(theta)/Zc, cos theta]] . [[1, x],[0,1]]
    Delta = A + B/Z0 + C*Z0 + D
    S11m = S22m = (A + B/Z0 - C*Z0 - D)/Delta   (A = D by symmetry)
    S21m = S12m = 2/Delta

This is the same physical model as attempt 1 (its quadratic Z_in
inversion is the S11-only, constants-fixed special case) — the model is
NOT being changed to fit the data; the change is which quantities are
measured vs assumed.

## 3. Identification data and identifiability (which observable pins which parameter)

Data: one driven-fixture run per port (battery-verbatim geometry,
GaussianPulse(f0=2.0 GHz, bandwidth=0.8), n_steps = 12000 — attempt-1
extraction-arm parameters verbatim) with s_param_freqs =
[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6] GHz (7 bins; attempt 1's 4 bins
plus 3 midpoints, all inside the pulse's 1.2-2.8 GHz band and all BELOW
the 3 GHz gate-band edge). The full complex 2x2 matrix at every bin is
fitted: 7 bins x 4 entries x (Re, Im) = 56 real observations, 3
parameters.

Held-out structure (frozen): the identification band is 1.4-2.6 GHz;
the gate band is the battery's 9 bins linspace(3, 7) GHz. They are
DISJOINT by construction — no gate-band bin enters the fit, and no
identification bin enters any gate. The extracted L* crosses to the
gate band only through the quasi-static flat-L model, whose adequacy is
gated in the identification band (F-J1, F-J3) and whose in-gate-band
consequence is bounded by B(L*) (section 5).

Identifiability (small-signal, at the identification bins;
omega = 0.88e10..1.63e10 rad/s, theta ~ 0.77..1.43 rad):

- **tau <- arg S21.** d(arg S21)/d(tau) = -omega to leading order; the
  signal omega*tau is 0.8-1.4 rad against a phase-residual class of
  ~0.02 rad (section 4 budget / |S21| ~ 1), so tau is pinned to
  ~2 ps ~ 0.6 mm of l_eff — 30-80x finer than the +-1 mm overhang
  uncertainty that poisoned attempt 1. L contributes to arg S21 only
  through arctan-class terms ~ x/(2*Z0) ~ 0.02-0.04 rad with a KNOWN
  omega-linear signature — separable over the near-octave span.
- **Zc <- the S11/S22 standing-wave ripple.** The line-mismatch part of
  S11 is Gamma*(1 - e^(-2j*theta)) + O(Gamma^2), Gamma = (Zc-Z0)/(Zc+Z0):
  amplitude dS11/dZc = |dGamma/dZc|*|1 - e^(-2j*theta)| ~ 0.0104 * (1.4..2.0)
  ~ 0.015-0.02 per ohm, with a theta-PERIODIC frequency signature.
- **L <- the omega-LINEAR component of the diagonal.** Each post adds
  ~ j*omega*L/(2*Z0) class terms: d|S11|/dL ~ omega/Z0 ~ 0.18-0.33 per
  nH, growing linearly in omega with a phase distinct from the line
  ripple.
- **Why the attempt-1 degeneracy is lifted**: attempt 1 inverted ONE
  complex number per bin (Z_in from S11 alone) with two constants
  assumed — at any single bin dL and dZc trade against each other
  (0.065 nH/ohm). The joint fit adds arg S21 (pins tau independently of
  L and Zc) and uses the DIFFERENT frequency signatures of the L term
  (linear in omega) and the Zc term (periodic in theta) across seven
  bins spanning nearly an octave, so the Jacobian columns decorrelate.
  This argument is verified NUMERICALLY, not just asserted: F-J5 gates
  the linearized parameter covariance actually achieved.

Fit (frozen): least squares on the 56 stacked real residuals
Re/Im(S_model - S_meas), uniform weights, scipy.optimize.least_squares,
multi-start over the 27-point grid L0 in {0.2, 0.3, 0.4} nH x Zc0 in
{46, 48.25, 50.5} ohm x l0 in {15.5, 16.5, 17.5} mm (best final cost
wins; degenerate multi-minima at comparable cost = STOP via F-J5).
Parameter covariance: cov = s^2 (J^T J)^{-1}, s^2 = SSR/(56-3), J the
final Jacobian.

## 4. Extraction-arm falsifiers (frozen; any firing = STOP, no band arm, xfail stays)

- **F-X5 (Re(V/I) first, binding lane-wide)**: Re(Z_in) > 0 at every
  bin of every driven sweep in this lane, checked BEFORE interpreting
  any |S| >= 1 anywhere.
- **F-J1 (joint-fit residual — model adequacy)**: over all 4 entries x
  7 bins, max |S_fit - S_meas| <= 0.025 and rms <= 0.012.
  Derivation (known unmodeled physics in the identification band, all
  prior/measured classes, linear sum): junction shunt C <= 12 fF
  (attempt-1 frozen class) -> omega*C*Z0/2 <= 0.005 at 2.6 GHz; Zc
  dispersion WITHIN the 1.2 GHz identification span around the fitted
  value (#313 measured 0.7 ohm across 4 GHz -> <= ~0.35 ohm class here)
  -> 0.007; loss/radiation not in the lossless model (attempt-1
  report-only Re(x) class <= 0.8 ohm) -> 0.008; DFT-settling/float
  class 0.005. Sum 0.025; rms class ~ half.
- **F-J2 (parameter physicality — the falsifiable prediction, declared
  before extraction)**:
  - L* in [0.20, 0.50] nH — identical derivation to attempt-1 F-L2
    (#318 ledger witness ~0.26 nH minus 25%; quasi-static thin-wire
    estimate 0.40 nH plus 25%). Independent of attempt-1's extracted
    number.
  - Zc* in [44.0, 53.0] ohm — #313 mid-line measured class 47.9-48.6
    (3-7 GHz) plus ~10% for plane-definition offset and low-f
    dispersion; outside this the "same quasi-TEM line" premise is
    refuted.
  - l_eff* in [15.0, 18.0] mm — port-to-port 16 mm; trace overhang
    +0.5 mm (1 cell) per side gives geometric [16, 17]; +-1 mm for
    fringing and the beta-factor corners (1.048/1.062 vs the 1.055
    reporting convention, +-0.11 mm equivalent).
- **F-J3 (cross-bin flatness under MEASURED constants — the direct
  re-test of attempt-1's F-L1)**: run the attempt-1 per-bin quadratic
  inversion (apparatus verified exact in that lane) with the FITTED
  Zc*, tau* held fixed; on driven port 1,
  max_bin |L_bin - median| / median <= 0.20 — the SAME window as F-L1,
  not widened. If the attempt-1 decline was constant-conditioning bias,
  measured constants flatten it; if the post reactance is genuinely
  dispersive it persists and fires. Not circular: Zc*/tau* are pinned
  by arg S21 and the ripple signature (section 3), not by the per-bin
  flatness being tested, and a dispersive truth cannot be absorbed into
  (Zc, tau) without violating F-J1's residual gate (verified on a
  synthetic, section 6 V3).
- **F-J4 (port symmetry)**: per-port per-bin inversion with fitted
  constants; |median_p1 - median_p2| / mean <= 0.10 (attempt-1 window).
- **F-J5 (identifiability achieved, not just argued)**: from the fit
  covariance, sigma_L/L* <= 0.10, sigma_Zc <= 1.5 ohm,
  sigma_tau/tau* <= 0.05; and the multi-start must converge to a single
  basin (all starts within 3*sigma of the best, or best-cost basin
  strictly dominant with second basin cost >= 2x). If F-J5 fires, the
  fixture's out-of-band data cannot pin the parameters and the lane
  STOPs; the named next-scale remedy is the single-post fixture
  (VESSL sketch, section 1).

Adopted values on ALL-PASS: (L*, Zc*, tau*) from the best joint fit;
L* is the ONLY parameter that crosses into the band arm (Zc*, tau* are
identification nuisances; the band budget's line-mismatch term uses the
#313 IN-BAND measured class, not the extrapolated Zc*).

## 5. Band arm (only if section 4 all-holds) — re-derived budget and falsifiers

Frozen budget formula, numerically frozen the moment L* is adopted
(before the band measurement runs):

    B(L*) = 0.0430                       line mismatch: #313 in-band measured
                                         worst Zc = 47.9 ohm -> |Gamma| = 0.0215,
                                         |S11_line| <= 2|Gamma|/(1-Gamma^2 e) = 0.0430
                                         (attempt-1 derivation, unchanged — the
                                         in-band Zc class is MEASURED at the gate
                                         bins by #313, so no extrapolation)
          + 2 * (0.10 * omega_7GHz * L*) / (2 * Z0)
                                         two posts, coherent worst case;
                                         delta-L/L = 0.10 per the F-J5 gate
                                         (TIGHTER than attempt 1's 0.15 —
                                         the joint fit must demonstrate it
                                         or the lane stops at F-J5)
          + 0.012                        non-L post parasitic (shunt C <= 12 fF
                                         at 7 GHz; attempt-1 class)
          + 0.005                        complex64/DFT float class
    with absolute ceiling B <= 0.13.

Orientation (not targets): L* = 0.25 nH -> B = 0.0820; 0.35 nH ->
0.0908; the F-J2 edges give B(0.20) = 0.0776, B(0.50) = 0.1040.

Band measurement: battery-verbatim run (linspace(3,7,9) GHz, n_steps
4000, GaussianPulse(f0=5 GHz, bandwidth=0.8)); de-embed offline with
`deembed_series_inductance(S, f, [L*, L*])`. Falsifiers:

- **F-D1 (floor)**: max in-band (|S11_dut|, |S22_dut|) < B(L*).
- **F-D2 (reduction)**: max in-band de-embedded diagonal <
  0.5 * 0.2910 = 0.1455.
- **F-X1 (passivity)**: de-embedded per-bin sv_max <= 1.01, interpreted
  only after F-X5.
- **F-X2 (reciprocity preserved)**: max in-band |S21d - S12d| <= 1e-3.
- **F-X3 (off-diagonal magnitude)**: per-bin |S21d| in [0.93, 1.005].
- **F-X4 (raw paths untouched)**: no shipped extraction/scan/decomposer
  code edited; wire/lumped batteries (marker override), refplane and
  dump/replay suites pass unchanged; de-embed remains opt-in
  post-processing.
- De-embedded S21 phase vs analytic line delay: REPORT-ONLY (no
  baseline exists to gate).

## 6. Apparatus verification (synthetics, run AFTER this commit and BEFORE any FDTD)

All on the section-2 forward model with independent ABCD arithmetic
(the harness fit must not be verified against its own forward code
alone; the synthetic generator is written separately in the harness and
cross-checked against `rfx.deembed`'s embed direction where applicable):

- **V1 (exactness)**: truth (0.25 nH, 48.25 ohm, 16 mm) -> joint fit
  recovers all three to <= 1e-6 relative.
- **V2 (attempt-1 bias injected and correctly absorbed)**: truth
  (flat 0.25 nH, Zc = 46.0 ohm, l_eff = 17 mm). (a) The attempt-1
  fixed-constant per-bin inversion on this synthetic MUST show the
  declining-L signature (reproducing the failure mode); (b) the joint
  fit MUST recover the truth to <= 1% in every parameter and the F-J3
  flatness with fitted constants must be <= 0.01 — the bias lands in
  the constants where it belongs, not in L(f).
- **V3 (falsifier has teeth)**: truth genuinely dispersive,
  L(f) = 0.25 nH * (1 - 0.3 * f / 2.6 GHz), constants at center. The
  joint fit + falsifier evaluation MUST fire F-J1 or F-J3 — a real
  model violation must NOT be silently absorbed into (Zc, tau).
- **V4 (uncertainty machinery)**: V1 plus iid complex Gaussian noise of
  scale 0.005 per entry -> recovered parameters within 3 sigma of
  truth with sigma from the F-J5 covariance, and F-J5's windows hold.

Any V-check failing = apparatus bug; fix the apparatus, never the
windows, and re-run all V-checks before any FDTD.

## 7. Dispositions (frozen)

- All of section 4 and section 5 hold -> replace the strict xfail on
  `test_thru_s11_floor` with the measured physical de-embedded floor
  lock. In-file provenance: raw 0.2910 -> measured de-embedded worst;
  the full post model (L* both ports, joint-extraction method, this
  note); the held-out structure (identification 1.4-2.6 GHz disjoint
  from the 3-7 GHz gate). Gate value = measured de-embedded worst with
  >= 25% headroom, REQUIRED <= B(L*). Plus: raw fixture-physics
  envelope pin (raw worst in [0.20, 0.35], measured 0.2910) so a raw
  physical-channel regression stays loud; raw alive floor (> 0.02) kept
  on the RAW diagonals only (the de-embedded diagonal is legitimately
  near-null).
- Any section-4 falsifier fires -> STOP before the band arm; record in
  an appended results section; xfail stays byte-untouched; if the
  firing is F-J5-class (identifiability), emit the corrected VESSL yaml
  for the single-post fixture as the named next-scale remedy.
- F-D1 or any F-X* fires -> STOP; record honestly; xfail stays.
- Movers beyond the one xfail replacement: NONE planned; any unexpected
  mover is STOP-and-report, not re-pin. No window above is widened
  after any measurement, under any rationale.

## 8. PRE-MEASUREMENT apparatus finding (appended 2026-08-29, BEFORE any FDTD run; sections 1-7 unchanged)

V-check results on the frozen synthetics (harness `--verify`):

- V1 PASS exact (max rel err 2.2e-16); generator cross-check vs the
  `rfx.deembed` inverse at 5.6e-16.
- V2a PASS: the attempt-1 fixed-constant method on the biased truth
  (Zc=46, l_eff=17, flat 0.25 nH) reproduces the declining-L signature
  (flatness 0.80, monotone).
- V2b PASS exact: the joint fit recovers the biased truth to 2.2e-16
  and F-J3 flatness with fitted constants is ~3e-15.
- **V3 as frozen FAILED — recorded, not hidden.** The 30% linear
  dispersive truth is absorbed by the joint fit into
  (L, Zc, l_eff) = (0.289 nH, 46.81 ohm, 15.40 mm) — inside all F-J2
  windows — at resid_max 0.0044 (below the honest F-J1 noise budget)
  and F-J3 flatness 0.022. This is a MATHEMATICAL property of the
  design discovered before any measurement: over 1.4-2.6 GHz a smooth
  linear L(f) is near-degenerate with a flat L plus shifted constants.
  No out-of-band falsifier on this data can detect it without gating
  below the physical noise class, which would be dishonest.

Consequence, derived and verified numerically (no window of sections
4-5 is touched; this appendix only relocates where the smooth-dispersion
teeth are VERIFIED to live):

- The smooth-dispersion falsification is carried by the HELD-OUT band
  arm's F-D1 with the frozen B(L*). End-to-end synthetic check: the
  same 30%-dispersion law extended over 3-7 GHz, de-embedded with the
  out-of-band fitted flat L*, leaves a worst in-band diagonal 0.1293
  vs B = 0.0854 -> **F-D1 FIRES**. A 10%-dispersion law leaves 0.0406
  vs B = 0.0832 -> inside the budget, i.e. within the locked claim's
  own stated resolution (the lock asserts the de-embedded floor under
  the FROZEN flat-L* post model is < B; dispersion whose in-band
  consequence is below B is inside that claim's tolerance by
  construction).
- **V3' (supersedes V3, frozen now, still pre-measurement)**: the
  30%-dispersion synthetic must (a) be absorbed out-of-band (recording
  the degeneracy) and (b) fire F-D1 on the synthetic band arm with the
  fitted flat L*. V3' passing = the teeth exist where the held-out
  structure puts them.
- F-J1/F-J3 retain their verified teeth against the attempt-1
  constant-bias class (V2) and against model-breaking residual
  structure; they are no longer claimed to detect smooth dispersion.

No FDTD measurement had been run in this lane when this section was
committed; the amendment is append-only design correction, not
post-measurement tuning.

### 8.1 V4 finding (appended 2026-08-29, still BEFORE any FDTD run)

V4 as frozen: pulls PASS (0.15-0.21 sigma — the covariance machinery is
correct), but the clause "F-J5's windows hold at the 0.005 noise class"
is REFUTED numerically: at iid complex noise of scale 0.005 per entry
the fit returns sigma_L/L = 0.20 (sigma_Zc = 0.74 ohm and
sigma_leff/leff = 0.019 pass). The L column of the Jacobian is the
weakest — consistent with the section-8 degeneracy finding.

Consequence (no window touched; F-J5 stands exactly as frozen):
sigma scales linearly with the residual class, so **F-J5's
sigma_L/L <= 0.10 gate passes only if the measured joint-fit residual
class is <= ~0.0025** (complex rms per entry). That is the quantitative
bar the FDTD data must clear; if the real fixture's residual class is
larger, F-J5 FIRES and the lane STOPs — which is precisely the gate
doing its job ("identifiability achieved, not just argued"). The
next-scale remedy in that case remains the single-post fixture.

**V4' (supersedes V4, frozen now, pre-measurement)**: (a) pulls within
3 sigma at the 0.005 noise class; (b) linear sigma scaling verified: at
noise scale 0.002 the F-J5 windows must hold (sigma_L/L <= 0.10),
demonstrating the gate is passable by clean-enough data and fires
otherwise.
