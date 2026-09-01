# THRU feed-post de-embedding, attempt 4 — analytic junction-parameter windows (closing lane)

Status: BINDING. Committed BEFORE any attempt-4 verdict is evaluated.
Sections 1-8 are frozen at commit time; results are APPENDED in later
sections, never edited back. No window below is widened after any
verdict; a fired falsifier is a STOP for its arm. A justified STOP is a
valid outcome.

Lane: `agent/thru-deembed-r4` (branched from `agent/thru-deembed-r3`).
Author: thru-diagonal de-embedding author agent (attempt 4), 2026-08-29 KST.
Parents (settled, not re-litigated): attempts 1-3
(`thru_feedpost_deembed_predeclaration.md`,
`thru_feedpost_joint_extraction_predeclaration.md`,
`thru_feedpost_twoseg_predeclaration.md`). Settled and carried verbatim:
the exact 2-port cascade de-embed algebra (`rfx/deembed.py` incl.
`deembed_line_segment`, round-trips 1e-12); the two-segment post model
(section 2 of the attempt-3 note: each post a lossless segment
(Zp, tau_p) + `l_trace` = 16.0 mm pinned by geometry); the
identification apparatus (I1 in-situ #313 two-plane Zc(f)/beta(f), I2
repaired single-post 1-port fixture with measured-load channel and the
section-12-corrected T observable, I3 thru 2-parameter refit) — VALLEY
BROKEN and reviewer-reproduced (corr 0.169/0.401, cond 1.31/1.61 vs the
3-parameter apparatus's -0.99/~170 on the same data).

## 1. Standing after attempt 3, and the binding data-hygiene rule

Attempt 3 STOPped on F-P: the I2 single-post L_p = 0.5089 +- 0.0066 nH
fired the frozen [0.20, 0.50] nH window by 1.3 sigma (I3
0.4976 +- 0.0033 sat 0.7 sigma inside). The recorded mechanism: that
window's upper edge (thin-wire point estimate 0.40 nH + 25%) was
derived in attempt 1 for a POINT series element — the bare post's
self-inductance — while the segment parameter L_p = Zp*tau_p is the
effective series inductance of the FULL terminal-to-line JUNCTION
(1 mm post + 0.5 mm trace-overhang stub + junction fringe). The named
remedy, which THIS lane executes: an ANALYTIC re-derivation of the
segment-parameter windows from junction geometry + prior-provenance
classes, as a FORWARD calculation.

**Binding hygiene rule.** The attempt-1/2/3 thru and single-post
extraction NUMBERS are BURNED for window-setting: L class
0.3819/0.4976/0.5089 nH, tau class 6.70/7.33 ps, Zp class 68-76 ohm,
C_p class 88-108 fF, the in-situ Zc(1.4-2.6 GHz) class 48.60-48.74 ohm
and beta-factor class 1.052-1.059, l_eff* 18.211 mm, and every residual
those windows judged. Every term below comes from a FORMULA + committed
GEOMETRY + a CITED prior-provenance class that existed before this
attempt's fixtures were measured, each on its own line with inputs and
result, so an adversary can recompute every term blind and verify none
was reverse-engineered to cover the burned numbers. Where a fresh
analytic edge and a frozen prior edge disagree, the rule declared here
is: a physicality window may only be UNIONED with a prior MEASURED
witness class (never intersected against it), and all roundings go
INWARD (tightening) — both rules bite against passing, not for it.

## 2. Committed geometry and prior-provenance classes (the only inputs)

Geometry (from the committed builders
`validation/research/thru_feedpost_deembed.py::build_thru` and
`.../thru_feedpost_twoseg_extraction.py::build_singlepost` — pure
geometry, no measured numbers):

| symbol | value | source |
|---|---|---|
| dx | 0.5 mm | `DX` |
| post height h | 1.0 mm | port `extent=H`, trace bottom at z = H |
| thin-wire equiv. radius r0 | 0.135*dx = 67.5 um | FDTD grid-edge natural radius (Taflove/Umashankar thin-wire class; the radius attempts 1-3 already used) |
| overhang l_ov | 0.5 mm (1 cell) | trace spans X1-dx .. X2+dx; ports at X1, X2 |
| trace width w | 5.0 mm | `W` |
| trace thickness t | 0.5 mm (1 cell) | Box z = H .. H+dx |
| l_trace | 16.0 mm | X2 - X1 (pinned, attempt-3 sec. 2) |
| dielectric | vacuum (eps_r = 1) | builder |

Prior-provenance classes (each pre-existing this lane, none from the
burned set):

| class | value | provenance |
|---|---|---|
| mid-line Zc class | 47.9 .. 48.6 ohm | #313 Phase-0 measured plane class (ledger; attempt-1 F-L5 corners) |
| beta-factor class n | 1.048 .. 1.062 | #313 measured "dx-stable physical class" (ledger: "measured β here is 1.05-1.06×ω/c"); attempt-1 corners |
| per-length L' = Zc*n/c | 167.4 .. 172.2 nH/m | derived from the two rows above |
| per-length C' = n/(Zc*c) | 71.9 .. 74.0 pF/m | derived from the two rows above |
| #318 witness class | ~0.26 nH per post | ledger 2026-07-11 #318 entry (fixture-physics attribution, MEASURED witness) |
| discretization class | +-25% on the post-L point estimate | attempt-1 F-L2 frozen class (prior provenance, pre-existing) |
| corrected equiv. radius class | r_eff = 0.23*dx | Noda-Yokoyama thin-vertical-wire FDTD equivalent-radius result (literature class, pre-existing) |
| bare-junction shunt-C class | 0 .. 20 fF | attempt-1 (<= ~12 fF quasi-static) / attempt-3 named 12-20 fF class |
| per-fixture systematic classes (L, tau) | see sec. 4/6 | attempt-3 section-4 F-C design-phase injection table (committed pre-measurement; reproduced by V5) |

## 3. L_junction window — term by term (frozen)

The windowed quantity is the SEGMENT series inductance
L_p = Zp*tau_p: the effective series inductance of the whole
terminal-to-line junction (post + overhang environment + fringe), NOT
the bare post. Terms:

**T-L1 — post partial self-inductance (Goldfarb-Pucel via formula —
the same cited formula attempt 1 used for its point estimate):**

    L_post = (mu0/2pi) * [ h*ln((h + sqrt(h^2+r0^2))/r0)
                           + 1.5*(r0 - sqrt(h^2+r0^2)) ]
    inputs h = 1.0 mm, r0 = 67.5 um
    = 2e-7 * [1.0e-3*ln(2.002275e-3/6.75e-5) + 1.5*(6.75e-5 - 1.002275e-3)]
    = 2e-7 * (3.3899e-3 - 1.4022e-3) = 0.3975 nH.

The gap V integral spans exactly the 2 live cells (= 1.0 mm), so the
terminal pair pins h; residual staircase/terminal ambiguity is carried
by T-L4, not by an h spread (the attempt-3 tau window's +-1-cell h_eff
class was a TRANSIT-PATH class including fringe path-lengthening; it
does not apply to the series-L terminal definition).

**T-L2 — trace-overhang / launch-plane series contribution
(sign +, bound):** l_trace = 16.0 mm is pinned between the port
COLUMNS, but the physical launch of line-mode current has a one-cell
rasterization ambiguity toward the overhang side (the post column's
trace cell is shared with the overhang), and the stub's own charging
current traverses part of the stub's series inductance (open-stub
input-impedance expansion: -j*Zc*cot(beta*l) = 1/(jwC'l) + jw(L'l)/3 +
O((bl)^4) — the linear current taper gives exactly 1/3 as the point
estimate; the full-current-path bound — the lesson of attempt-2's F-J2
— is 1). Bound with the interval arithmetic maximum:

    T-L2 in [0, L'_hi * l_ov] = [0, 172.2 nH/m * 0.5 mm] = [0, 0.0861] nH.

**T-L3 — junction fringe/mutual (sign -, bound):** the post-to-trace
right-angle junction: orthogonal filaments have ZERO partial mutual
inductance (Grover), so the junction mutual cannot ADD series L; the
two real fringe mechanisms — corner current redistribution (microstrip
bend inductance-deficit class) and top-loading of the post by the wide
trace (larger effective top radius) — both REDUCE L. Bound: one
half-cell of line-class series inductance,

    T-L3 in [-L'_hi * dx/2, 0] = [-0.0430, 0] nH.

**T-L4 — discretization class (both signs, prior provenance):** the
envelope of two pre-existing classes for the same rasterization
uncertainty (envelope, not sum — they describe the same physics, so
summing would double-count):

  - equivalent-radius spread r_eff in [0.135*dx, 0.23*dx]:
    L_post(r=0.23*dx) = 0.3044 nH -> delta = -0.0932 nH (one-sided:
    a larger effective radius only reduces L);
  - the attempt-1 frozen +-25% class on the point estimate:
    +-0.25 * 0.3975 = +-0.0994 nH.

    T-L4 in [-0.0994, +0.0994] nH  (lower edge: max magnitude of
    {-0.0932, -0.0994}; upper edge from the +-25% class).

**Sum (interval arithmetic), then the section-1 union/rounding rules:**

    L_lo = 0.3975 + 0 - 0.0430 - 0.0994 = 0.2551 nH
    L_hi = 0.3975 + 0.0861 + 0 + 0.0994 = 0.5830 nH

Lower edge UNIONED with the prior measured #318 witness class edge
(0.26 nH - 25% = 0.195 -> the frozen 0.20 of attempts 1-3, which no
attempt refuted): min(0.2551, 0.20) = 0.20. Upper edge rounded INWARD:
0.5830 -> 0.58.

**FROZEN: L_p in [0.20, 0.58] nH** (applies to BOTH the I3 and I2
fitted values, as before).

## 4. tau_p and C_p windows (frozen)

At the identification band the segment is electrically short
(w*tau ~ 0.1 rad), so tau_p = sqrt(L_p * C_p) with C_p the junction's
total shunt capacitance. C_p derivation, term by term:

**T-C1 — overhang line capacitance:** C' * l_ov =
[71.9, 74.0] pF/m * 0.5 mm = **[36.0, 37.0] fF**.

**T-C2 — open-end fringe (Hammerstad open-end length formula, cited;
eps_eff = 1, w/h = 5):**

    dl/h = 0.412 * (eps_eff + 0.3)(w/h + 0.264)
                 / ((eps_eff - 0.258)(w/h + 0.8))
         = 0.412 * (1.3 * 5.264)/(0.742 * 5.8) = 0.6551 -> dl = 0.655 mm
    C_end = C' * dl = [47.1, 48.4] fF  (zero-thickness formula)

Class: -25% formula-accuracy on the lower edge (the formula is a
zero-thickness fit outside its calibration corpus at eps_r = 1); on the
upper edge the finite trace thickness t = dx adds end-face capacitance
bounded by the parallel-plate limit of the end face over the ground,
eps0*w*t/h = 8.854e-12 * 5e-3 * 0.5e-3 / 1e-3 = 22.1 fF:

    T-C2 in [0.75*47.1, 48.4 + 22.1] = [35.3, 70.5] fF.

**T-C3 — post/port junction capacitance:** prior bare-junction class
**[0, 20] fF** (section-2 table).

**Sum:** C_p in [36.0+35.3+0, 37.0+70.5+20] = [71.3, 127.5] fF ->
**C_p class [71, 128] fF, REPORT-ONLY** (C_p = tau_p^2/L_p is a
deterministic function of the two gated parameters; a third gate on the
same two numbers adds no independent physics — it is quoted for
orientation and used as the input to the tau window below).

**tau_p window:** geometric interval

    tau_geo = [sqrt(0.20 nH * 71 fF), sqrt(0.58 nH * 128 fF)]
            = [3.77, 8.62] ps

widened by the honest per-fixture SYSTEMATIC class from the attempt-3
design-phase injection table (prior provenance, reproduced by V5;
this is the class the measured 0.627 ps cross-fixture tau disagreement
lives in — it is treated as a real fixture-level systematic and
carried at full value into the band budget in section 6, never
shrunk): largest per-fixture tau systematic = I2 plane-channel side
1.11 + 0.06 + 0.05 = 1.22 ps, plus 3x the design-phase Fisher sigma_tau
class (~0.027 ps -> 0.08 ps) = 1.30 ps:

    tau_lo = 3.77 - 1.30 = 2.47 -> 2.5 (inward)
    tau_hi = 8.62 + 1.30 = 9.92 -> 9.9 (inward)

**FROZEN: tau_p in [2.5, 9.9] ps** (both fixtures). For orientation
(not a window input): the bare geometric transit class of attempt 3,
h_eff/c in [1.67, 8.34] ps, overlaps this interval; the sqrt(L*C)
derivation supersedes it because the segment tau is the junction's
LC delay, not a light-transit time.

## 5. F-A loss-term repair (the review-flagged impurity) + frozen extraction falsifiers

Attempt-3's F-A1 budget included a 0.008 loss/radiation term whose
provenance was attempt-1's MEASURED report-only Re(x) class (<= 0.8 ohm)
at the burned bins. Re-derivation from first principles (replacing it;
independent of every burned number): the fixture is PEC + vacuum +
CPML — the only physical loss channel is radiation (absorbed by the
CPML). The radiating discontinuities are the two posts and the open
overhang ends, each of electrical size h/lambda <= 1.0 mm / 115 mm at
2.6 GHz. Monopole-class radiation resistance:

    R_rad = 40*pi^2*(h/lambda)^2 = 40*pi^2*(1/115.3)^2 = 0.030 ohm
    |dS| ~ R_rad/(2*Z0) = 3.0e-4 per radiator

Bound for the fixture (2 posts + 2 open ends, coherent worst case):
<= 4 * 3.0e-4 = 0.0012 -> **0.001** (identification band; the term is
a residual-budget line, and the same derivation at 7 GHz gives 0.002
per radiator — noted for the band-arm passivity reading, which has its
own gate). Budget consequence, restated with the surviving attempt-3
terms (their derivations unchanged and burned-free: junction parasitic
beyond the segment model 0.005; measured-constant bin-scatter 0.006;
DFT/float 0.005):

    F-A1 (thru refit): max |S_fit - S_meas| <= 0.005+0.006+0.005+0.001
                       = 0.017 ; rms <= 0.008   (was 0.025 / 0.012 —
                       the repair TIGHTENS the gate)
    F-A2 (single-post): 0.025 + 0.010 + 0.017 = 0.052 ; rms <= 0.026
                       (was 0.06 / 0.03 — tightened; plane-path
                       amplitude 0.025 and phase 0.010 classes are #313
                       instrument provenance, unchanged)

Frozen extraction falsifiers for attempt 4 (any firing = STOP, no band
arm, xfail stays):

- **F-X5** (Re(V/I) first, lane-wide) — carried verbatim.
- **F-I1/F-I2/F-I3/F-I4** — carried verbatim (instrument gates;
  their derivations never involved burned numbers).
- **F-A1/F-A2** — the REPAIRED windows above (0.017/0.008 and
  0.052/0.026).
- **F-P (attempt-4 windows, sections 3-4)**: BOTH the I3 and I2 fitted
  parameters in L_p in [0.20, 0.58] nH and tau_p in [2.5, 9.9] ps.
  C_p and Zp report-only.
- **F-V1/F-V2** (valley broken; identifiability) — carried verbatim.
- **F-C** (cross-fixture consistency 0.11 nH / 2.6 ps) — carried
  verbatim; the tau component's measured value is ALSO carried at full
  value into the band budget (section 6) — the honest systematic
  treatment.

**Verdict rule (frozen, carried from attempt 3):** on ALL-PASS the
adopted post model is (L*, tau_p*) from I3 — the best joint fit on the
object actually being de-embedded — with I2 the mandatory independent
witness via F-C. Zp* = L*/tau_p*, C_p* = tau_p*^2/L*.

**Data-reuse statement (explicit, per the deterministic-fixture rule):**
the fixtures and harness are byte-identical to the attempt-3 final tree
(commit f72eabc); attempt 4 RE-RUNS
`thru_feedpost_twoseg_extraction.py --extract` to reproduce the
deterministic raw measurements and evaluates the attempt-4 verdicts on
that output with the attempt-4 harness
(`validation/research/thru_feedpost_junction_windows.py`, which imports
the attempt-3 apparatus and changes ONLY the window constants named in
this note). The committed attempt-3 numbers are the expectation; a
reproduction deviating beyond the float class (|dL| > 0.001 nH,
|dtau| > 0.01 ps, any instrument channel > 0.1%) is apparatus drift =
STOP.

## 6. Band arm — re-derived budget (frozen formula; numbers freeze at adoption, before the band run)

Structure carried verbatim from attempts 1/3 (reusable structure per
the lane brief); NUMBERS re-derived for the adopted model:

    B(L*, tau*) = 0.0430                       line mismatch (#313 in-band
                                               worst Zc 47.9 -> |Gamma| =
                                               0.0215, x2/(1-..) — attempt-1
                                               derivation unchanged)
        + delta_L * omega_7GHz / Z0            two posts, series-error,
                                               coherent worst case
        + omega_7GHz * delta_C * Z0            two posts, shunt-error
        + 0.012                                beyond-model junction
                                               dispersion: 5% of the C_p*
                                               reactance at 7 GHz
                                               (omega_7*0.05*C_p**Z0 =
                                               0.012 at the section-4
                                               C class midpoint ~108 fF;
                                               the open-end fringe and
                                               stub cot-expansion
                                               dispersion classes are
                                               0.2-few % over 3-7 GHz)
        + 0.005                                complex64/DFT float class

    delta_L   = 0.035 nH + 3*sigma_L(I3) + |L*_I3 - L*_I2|
    delta_tau = 0.74 ps  + 3*sigma_tau(I3) + |tau*_I3 - tau*_I2|
    delta_C   = (2*delta_tau/tau* + delta_L/L*) * C_p*

(the 0.035 nH / 0.74 ps are the thru-side measured-constant injection
classes from the attempt-3 design-phase table — prior provenance; the
CROSS-FIXTURE terms carry the measured fixture-level systematic at full
value, per the section-4/5 honesty rule). Absolute ceiling carried:
**B_eff = min(B, 0.13)** — the ceiling binds whenever the formula
exceeds it (at the burned-number orientation class the formula reads
~0.23, so the effective pre-declared gate is expected to be 0.13; the
formula is still computed and frozen at adoption).

Band falsifiers (frozen; battery-verbatim band run linspace(3,7,9) GHz,
n_steps 4000, GaussianPulse(f0=5 GHz, bandwidth=0.8), de-embed via
`rfx.deembed.deembed_line_segment` with (Zp*, tau*) at both ports):

- **F-D1 (floor)**: max in-band (|S11_dut|, |S22_dut|) < B_eff.
- **F-D2 (reduction)**: max in-band de-embedded diagonal < 0.1455.
- **F-X1 (passivity)**: de-embedded per-bin sv_max <= 1.01, read only
  after F-X5 on the raw run.
- **F-X2 (reciprocity)**: max |S21d - S12d| <= 1e-3.
- **F-X3**: per-bin |S21d| in [0.93, 1.005].
- **F-X4 (raw paths untouched)**: no shipped extraction/scan/decomposer
  code edited; `git diff agent/thru-deembed-r3 -- rfx/` EMPTY for this
  lane; batteries + refplane + dump/replay suites pass unchanged;
  de-embed remains opt-in post-processing. Pre-declared additive
  movers: ONE example-fidelity classification entry for the attempt-4
  harness (`no_simulation` — it builds nothing of its own), and the
  single test_thru_s11_floor pin replacement (section 7). Nothing else
  moves.
- De-embedded S21 phase vs analytic line delay: REPORT-ONLY (attempt-1
  declared treatment, carried: reflective insertion loss removal makes
  the de-embedded S21 phase a derived, non-gated diagnostic).

## 7. Dispositions (frozen)

- ALL of section 5 and section 6 hold -> replace the strict xfail on
  `test_thru_s11_floor` with the measured physical de-embedded floor
  lock:
    * gate value = measured de-embedded worst diagonal * 1.25 (>= 25%
      headroom), rounded UP at the 3rd decimal, REQUIRED <= B_eff —
      if 1.25*worst > B_eff the lock is NOT placeable and the lane
      STOPs (budget too tight for an honest pin = not a pass);
    * in-file provenance: raw 0.2910 -> per-bin de-embedded values ->
      pinned floor; the full adopted post model (L*, tau*, Zp*, C_p*,
      both ports, with uncertainties and the cross-fixture systematic);
      the fixture chain attempts 1-4 and the held-out structure
      (identification 1.4-2.6 GHz, gate band 3-7 GHz, no shared bins);
    * plus the raw fixture-physics envelope pin (raw worst in
      [0.20, 0.35], measured 0.2910) and the raw alive floor (> 0.02)
      on the RAW diagonals (attempt-1 disposition, carried);
    * the de-embed inside the test uses the pinned (Zp*, tau*)
      constants and `rfx.deembed.deembed_line_segment` — opt-in
      post-processing inside the test, no shipped path touched.
- Any section-5 falsifier fires -> STOP before the band arm; with the
  windows now derived for the correct physical quantity, a fired F-P is
  a GENUINE physicality violation of the segment model, not a
  derivation slip; record and stop; xfail stays byte-untouched.
- F-D1/F-D2 or any F-X* fires -> STOP; record honestly; xfail stays.
- No window above is widened after any verdict, under any rationale.
- Cluster arm: none (the remedy is analytic + CPU-scale; the fine-dx
  sweep stays CORROBORATIVE per attempt-3 section 8; the VESSL yaml
  stays a proposal-only artifact, untouched by this lane).

## 8. Adversary-verification recipe

Every window above is recomputable blind from this note: T-L1 from the
Goldfarb-Pucel formula with (h, r0) from the committed builder; T-L2/
T-L3 from L' = Zc*n/c over the #313 ledger classes and (l_ov, dx) from
the builder; T-L4 from the two cited pre-existing discretization
classes; T-C1/T-C2 from C' = n/(Zc*c), the Hammerstad open-end formula
at (w/h = 5, eps_eff = 1), and the parallel-plate end-face bound;
T-C3 from the attempt-1 class; the tau window from sqrt(L*C) over the
frozen intervals plus the attempt-3 injection-table per-fixture class;
the F-A repair from the monopole radiation-resistance formula. The only
measured inputs are the #313/#318 ledger classes and the attempt-3
DESIGN-PHASE injection table, all committed before any attempt-3/4
measurement existed. No term uses the burned extraction numbers of
section 1.

## 9. PRE-VERDICT apparatus finding — stale attempt-3 F-P rows leaked into the attempt-4 verdict table (appended 2026-08-29, before the corrected table is adopted; sections 1-8 unchanged)

The first `--extract` run of the attempt-4 harness reproduced every
attempt-3 deterministic number to the 1e-5 class (all 7 reproduction
gates PASS) and every falsifier IN THE FROZEN SECTION-5 LIST passed —
but the printed verdict table also contained two rows per fit labeled
"F-P L*/tau*" carrying the SUPERSEDED attempt-3 windows
([0.20, 0.50] nH / [1.67, 8.34] ps), one of which read FIRED on the I2
L (0.5089 — the very number whose window this lane re-derived).
Mechanism (verified in source, not conjectured): the imported attempt-3
helper `fit_verdicts` hardcodes its module-level `F_P_L_NH`/`F_P_TAU_PS`
constants into its output rows; the attempt-4 harness passed its own
F-A windows as parameters but could not parameterize F-P, so the
attempt-3 rows rode along. The attempt-3 F-P window is NOT an attempt-4
falsifier (section 5 froze the attempt-4 F-P as the sections-3/4
windows; re-deriving that window is this lane's entire mandate, per the
attempt-3 STOP's named remedy) — the stale rows contradict the
committed pre-declaration. Per the frozen V-check discipline this is an
apparatus bug: FIX THE APPARATUS, NEVER THE WINDOWS. Repair (committed
before the corrected table is adopted): the harness filters the two
hardcoded "F-P " rows out of the imported helper's output; its own
F-P4 rows (already present and already evaluated in the same run:
I3 0.4976/7.3302, I2 0.5089/6.7028 — ALL PASS) are the binding ones.
No window, model, fixture, bin, or gate value changes; the fixtures are
deterministic and the re-run is a reproduction.

## 10. RESULTS — extraction + band arms (appended 2026-08-29; sections 1-9 unchanged)

Run: `validation/research/thru_feedpost_junction_windows.py --extract
--band`, branch `agent/thru-deembed-r4` (section-9 repair tree),
JAX_PLATFORMS=cpu. F-X5 CLEAN on every driven sweep before any
interpretation (thru in-situ drives Re(Z_in) 50.2..51.8; band raw
37.3..49.1 ohm, all > 0).

**Reproduction (deterministic fixtures, sec.-5 data-reuse gates):** all
7 gates PASS at the 1e-5 class (|dL| <= 5e-5 nH, |dtau| <= 2e-5 ps,
Zc ranges to 4e-4 ohm) — the attempt-3 measurements are reproduced
exactly; the windows are what changed.

**Attempt-4 extraction verdicts: ALL PASS (33/33 incl. reproduction).**
- I3 thru: L* = 0.4976 +- 0.0033 nH, tau* = 7.3302 +- 0.0399 ps
  (Zp* = 67.89 ohm, C_p* = 107.97 fF); resid max/rms 0.01310/0.00469 vs
  the TIGHTENED F-A1 0.017/0.008; corr 0.169, cond 1.31; 9/9 one basin.
- I2 single-post: L = 0.5089 +- 0.0066 nH, tau = 6.7028 +- 0.1077 ps
  (C_p = 88.29 fF); resid 0.00859/0.00429 vs tightened F-A2
  0.052/0.026; corr 0.401, cond 1.61; 9/9 one basin.
- **F-P4: PASS on both fixtures** — L 0.4976/0.5089 in [0.20, 0.58] nH
  (10.8 sigma inside the analytic edge for I2); tau 7.3302/6.7028 in
  [2.5, 9.9] ps; report-only C_p 107.97/88.29 fF inside the [71, 128]
  orientation class. THE ATTEMPT-3 STOP MECHANISM IS CONFIRMED
  RESOLVED: the forward-derived full-junction windows contain both
  independent fixtures' parameters with wide margin, with no term
  informed by the burned numbers.
- F-I1..4, F-V1/2, F-C all PASS (F-C 0.0112 nH / 0.6274 ps).

**Adoption + frozen budget (before the band run):**
(L*, tau*) = (0.4976 nH, 7.3302 ps) from I3 per the frozen rule;
delta_L = 0.0562 nH (11.3%), delta_tau = 1.4871 ps (20.3% — carries
the measured 0.627 ps cross-fixture systematic at full value),
delta_C = 56.0 fF; B_formula = 0.2326 -> **B_eff = min(B, 0.13) =
0.1300** (the carried ceiling binds, as the pre-declaration expected).

**Band arm (battery-verbatim, 9 bins 3-7 GHz):**

    raw   |S11| 0.0093..0.2896, |S22| 0.0176..0.2910 (worst 0.2910 —
                                          the held-gate provenance value)
    deemb |S11| 0.0391..0.1237 (peak at 6.5 GHz), |S22| 0.0429..0.1273
    deemb |S21| 0.9708..0.9952 ; sv_max 0.9861..1.00349
    [report-only] deemb S21 phase dev vs c-line delay -0.060..-0.128 rad

- F-D1 floor: worst = 0.1273 < B_eff = 0.1300 -> **PASS** (2.1% margin)
- F-D2 reduction: 0.1273 < 0.1455 -> PASS (2.29x reduction of the raw)
- F-X1 passivity: 1.00349 <= 1.01 -> PASS (read after F-X5)
- F-X2 reciprocity: 2.71e-4 <= 1e-3 -> PASS
- F-X3 |S21d|: [0.9708, 0.9952] in [0.93, 1.005] -> PASS
- F-X4: `git diff agent/thru-deembed-r3 -- rfx/` EMPTY; battery suite
  8 passed + 1 xfailed (the held floor xfail, byte-untouched);
  wire-port suites 25 passed; deembed + refplane + dump/replay 65
  passed; example-fidelity: 83 passed + the 33 PRE-EXISTING base
  snapshot failures, count unchanged (34 -> 33 with the one additive
  attempt-4 entry, which repairs exactly the discovery row for the new
  harness). All with `-o addopts="" -m "not gpu"`. [^33-correction]

[^33-correction]: **Correction (appended 2026-08-30, PR #779 review).**
  "33 PRE-EXISTING base snapshot failures" above is wrong and is
  withdrawn. Those rows are not base-branch failures independent of
  this lane. Running `tests/test_example_fidelity_contract.py`
  standalone off `agent/thru-deembed-r4` (no `main` merge) gives
  **6 failed, 99 passed**, not 33/83. All 6 failures are snapshot rows
  captured against a stale rfx build earlier in this same stacked
  branch, not a pre-existing base condition: 3 were added by this
  lane's own commit `83ac4a7d` (`thru_feedpost_deembed.py`
  band-pulse/insitu-refplane, `thru_feedpost_twoseg_extraction.py`
  refplane-n10), the other 3 were inherited unregenerated from the
  #777/#778 legs earlier in the stack
  (`issue683_sampling_order_decision.py`,
  `issue770_offdiag_adjudication.py` x2). Merging `origin/main` (which
  does not touch either fidelity file) and re-running
  `scripts/capture_example_fidelity_snapshot.py` against the branch's
  own current rfx regenerates exactly those 6 rows — the diff is 18
  additive lines adding a `mesh_extent_um` field the current schema
  now emits, no other content moves — and the full contract suite
  runs **105 passed, 0 failed** (as of PR head `e557f69e`). There is
  no snapshot debt attached to this PR that pre-dates it. The branch
  was later rebased onto `main` after #778 squash-merged (`4d8c2fc9`)
  and re-captured again by a separate pass (`1f39b054`,
  "re-capture the de-embed harness snapshot keys for the upstream
  mesh_extent_um field") — an independent 9-line delta across three
  variants x three axes, for a `mesh_extent_um` field an earlier-merged
  PMC-plane-convention change added to `fidelity_report`'s domain rows.
  Both re-captures were verified byte-identical to a fresh run of
  `scripts/capture_example_fidelity_snapshot.py` at the commit they
  were made on; at the final rebased tip the suite now collects 46
  variants (more validation harnesses landed via the #777/#778 stack)
  and runs **153 passed, 0 failed**.

**Disposition (frozen section 7): STOP — the pin is NOT placeable.**
The candidate lock gate = 1.25 * 0.1273 rounded up = 0.160 > B_eff =
0.130. Per the frozen rule ("budget too tight for an honest pin = not
a pass") the lane STOPs: `test_thru_s11_floor` stays byte-untouched as
the held strict xfail; NO lock anywhere moves.

**Named mechanism (recorded, not excused).** The de-embedded floor
0.1273 at 6.5 GHz is ~3x the 0.0430 bare-line-mismatch class: its
dominant content is residual post-model error — the flat (L*, tau*)
segment vs the junction's true in-band dispersion, exactly the
smooth-dispersion class the V3 synthetic showed lives in the held-out
band. The honest error budget for that class (whose delta_tau carries
the measured cross-fixture systematic at full value, per the
section-4/5 honesty rule) is the same size as the measurement, so a
>= 25%-headroom pin cannot fit inside it. Both pressures are honest and
they collide: shrinking the systematic class to make the pin fit is
precisely what the frozen rule forbids. A 2%-headroom pin at 0.13 would
be a flaky cross-machine gate enshrining model error as physics; the
STOP is the physically accurate outcome.

## 11. CLOSING SUMMARY — the four-attempt chain and the final physical answer

What each attempt eliminated:

1. **Attempt 1 (F-L1 STOP)** — point series-L inversion under ASSUMED
   line constants: refuted the assumed-constants frame (31% apparent L
   decline = dL/dZc ~ 0.065 nH/ohm conditioning). Left settled: the
   exact wave-cascade de-embed algebra, the budget structure B(...),
   the Re(V/I)-before-|S| discipline, the raw envelope provenance
   0.2910.
2. **Attempt 2 (F-J2 STOP)** — joint (L, Zc, l_eff) fit: CONFIRMED the
   attempt-1 mechanism (measured constants flatten L to 1%) and refuted
   the point-element model (l_eff* = 18.211 mm absorbed the posts' own
   transit; measured identifiability valley corr -0.991/-0.996, cond
   ~170). Left settled: the thru-only fit cannot identify the junction.
3. **Attempt 3 (F-P STOP)** — two-segment post model + independent
   fixtures (in-situ two-plane line constants; single-post 1-port with
   measured load): BROKE the valley (corr 0.169/0.401, cond 1.31/1.61)
   and measured the junction TWICE independently, agreeing to 0.011 nH
   / 0.63 ps — then fired a window that had been derived for the bare
   post, not the full junction it measured. Left settled: the
   apparatus, the parameters, and the diagnosis.
4. **Attempt 4 (this lane, STOP at the pin)** — analytic forward
   re-derivation of the junction windows (Goldfarb-Pucel post 0.3975 nH
   + overhang <= 0.0861 + fringe/discretization classes; C_p [71, 128]
   fF from overhang + Hammerstad open-end + junction class; tau =
   sqrt(LC) + honest systematic): the windows CONTAIN both fixtures'
   parameters with margin — attempt 3's numbers were physics, its edge
   was the error. The adopted model de-embeds the battery diagonal
   0.2910 -> 0.1273 (2.29x) and passes every band falsifier including
   the floor-vs-budget gate; the >= 25%-headroom lock does not fit
   inside the honest budget, so the gate is not moved.

**The final physical answer.** The thru's raw in-band diagonal (max
0.2910 at 7 GHz) is quantitatively the fixture's own two feed-post
JUNCTIONS — each the full terminal-to-line transition (1 mm post +
0.5 mm overhang stub + fringe), a short line segment with
Zp ~ 68-76 ohm, tau_p ~ 6.7-7.3 ps, i.e. L_p ~ 0.50 nH and
C_p ~ 88-108 fF — identified by two independent fixtures that agree to
0.011 nH / 0.63 ps, and accounted analytically term-by-term from
geometry and prior classes with no tuned quantity. Cascade-removing the
identified junctions leaves a de-embedded floor of 0.127 (6.5 GHz),
2.9x above the bare line-mismatch class 0.043; that remainder is
bounded by — and comparable to — the honest flat-segment model-error
budget, so it cannot be pinned as fixture physics at better than the
0.13 class. The truthful gate state therefore remains the held strict
xfail with the raw 0.2910 provenance; the de-embedded floor lives here
as a MEASUREMENT with its full model provenance, not as a lock. A
future lane that wants the pin needs a dispersive junction
identification (frequency-dependent segment parameters, or the
corroborative fine-dx sweep of attempt-3 section 8) to shrink the
delta_tau/delta_C classes below the ceiling — an apparatus improvement,
not a re-derivation of any window in this chain.
