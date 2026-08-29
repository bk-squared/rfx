# THRU feed-post de-embedding, attempt 3 — two-segment post model, identification from independent fixtures

Status: BINDING. Committed BEFORE any new measurement and BEFORE the
attempt-3 harness exists. Sections 1-9 are frozen at commit time;
results are APPENDED in later sections, never edited back. No window
below is widened after any measurement; a fired falsifier is a STOP for
its arm. A justified STOP is a valid outcome.

Lane: `agent/thru-deembed-r3` (branched from `agent/thru-deembed-r2`).
Author: thru-diagonal de-embedding author agent (attempt 3), 2026-08-29 KST.
Parents (settled context, not re-litigated):
`docs/design_notes/thru_feedpost_deembed_predeclaration.md` (attempt 1,
incl. its post-review correction) and
`docs/design_notes/thru_feedpost_joint_extraction_predeclaration.md`
(attempt 2). The exact 2-port wave-cascade de-embed algebra, the
refplane-inapplicability-to-lumped-removal derivation, and the untouched
`rfx/probes/refplane.py` module + #764 byte-INEQUALITY pin all carry
over verbatim.

## 1. Standing after attempt 2, and the binding data-hygiene rule

Attempt 1 STOPped on F-L1: per-bin flat-L inversion under ASSUMED line
constants declines 31% across 1.4-2.6 GHz (mechanism: dL/dZc ~ 0.065
nH/ohm conditioning + overhang length uncertainty). Attempt 2's joint
(L, Zc, l_eff) fit CONFIRMED that mechanism (measured constants flatten
L to 1.05%) but STOPped on F-J2: l_eff* = 18.211 +- 0.197 mm vs the
frozen [15, 18] mm window, whose trace-only derivation omitted the two
1 mm posts' own transit. The reviewer additionally measured the
identifiability soft valley of the 3-parameter thru-only fit:
corr(L, Zc) = -0.991, corr(L, l_eff) = -0.996, Jacobian condition ~170.

Binding hygiene rule for THIS attempt: the attempt-1/2 THRU-fixture raw
data at the extraction bins is BURNED for window-setting. Every window
below is derived from geometry/first principles, from prior ledger /
#313 instrument provenance, or from the NEW independent fixtures —
never from the numbers those windows will judge. Attempt-2 numbers
(L* class ~0.38 nH, low-f Zc class ~47.3 ohm, per-post transit class
~3.9 ps) are used ONLY as design-scale inputs (e.g. to evaluate an
expected condition number from the MODEL); no window is placed to make
them pass, and each derivation below is shown.

## 2. The two-segment post model (frozen)

The feed post is a 1 mm vertical thin-wire (Ez column; 3 cells with the
top cell dead inside the PEC trace; thin-wire equivalent radius
0.135*dx = 67.5 um) connecting the lumped port terminal pair to the
16 mm air-microstrip trace. Attempt 2 measured that a point series
element under-describes it: the post is a SHORT TRANSMISSION-LINE
SEGMENT — it has both series inductance and shunt capacitance, i.e. a
delay, which the 3-parameter model could only absorb into l_eff.

Model (minimal parameterization, 2 free parameters):

    port1 --[seg(Zp, tau_p)]--[line(Zc(f), theta_t(f))]--[seg(Zp, tau_p)]-- port2

* Each post is a lossless uniform line segment with characteristic
  impedance Zp and one-way delay tau_p, identical at both ports (the
  fixture is geometrically symmetric; symmetry is re-checked by the
  instrument cross-checks, not assumed silently).
* Free parameters: L_p (reported, henries) and tau_p (seconds), with
  Zp = L_p / tau_p. In the electrically-short limit the segment is a
  series L_p = Zp*tau_p plus shunt C_p = tau_p/Zp = tau_p^2/L_p — the
  quasi-static thin-wire L class plus the junction/post capacitance the
  attempt-1/2 budgets could only bound. NOTE (exact model symmetry):
  (Zp, tau_p) -> (-Zp, -tau_p) leaves the segment ABCD invariant, so
  every fit below is bounded to tau_p > 0.
* The trace is a line segment with PER-BIN MEASURED constants: Zc(f)
  and beta(f) from the in-situ two-plane instrument (section 3), and
  theta_t(f) = beta(f) * l_trace with l_trace = 16.0 mm FROM GEOMETRY
  (the port columns sit at x = 8 and 24 mm; the post delay is a model
  parameter, so l_trace is a pinned geometric constant, not a fit
  parameter — the attempt-2 l_eff absorption channel is closed by
  construction). The 1-cell trace overhangs beyond each port column are
  part of the junction environment and are ABSORBED into the identified
  post parameters identically in both fixtures (both fixtures carry the
  same 0.5 mm overhang on the port side).

Prior physical bounds on the parameters (window derivations, frozen):

* **L_p in [0.20, 0.50] nH** — identical derivation to attempt-1 F-L2 /
  attempt-2 F-J2: #318 ledger witness class ~0.26 nH minus 25%;
  quasi-static thin-wire estimate for a 1 mm post at r = 67.5 um
  (0.40 nH) plus 25%. Independent of any burned number.
* **tau_p in [1.67, 8.34] ps** — full-current-path derivation (the
  correction attempt 2's l_eff window lacked): nominal electrical
  height h = 1.0 mm -> h/c = 3.34 ps; staircase/dead-cell rasterization
  ambiguity +-1 cell on the effective height (the top post cell is dead
  inside the trace; the gap V integral spans 2 live cells) gives
  h_eff in [0.5, 1.5] mm; junction fringing can lengthen the effective
  path by <= 2 cells (1.0 mm) more. Window = [0.5 mm/c, 2.5 mm/c].
* Implied Zp = L_p/tau_p in [24, 300] ohm — report-only sanity, no
  separate gate (redundant given the two above).

## 3. Identification design — which parameter comes from which INDEPENDENT observable

Three instruments, all evaluated at the 7 identification bins
[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6] GHz (all below the 3 GHz gate-band
edge; section 7 holds the 3-7 GHz gate bins out of everything here):

**I1 — in-situ #313 two-plane instrument on the THRU (new observable
channels).** Battery-verbatim thru opted into
`add_port(reference_plane_cells=10)` (N = 10: both planes >= 10 cells
from every port — the measured clean-plane class, Im/Re(Zc) <= 1.2%,
vs 8.2% at N = 3). Per driven port the production sparam driver's
refplane diagnostics return the MEASURED Zc(f) and beta(f) at every
extraction bin from the exact load-independent two-plane invariant
Zc^2 = (V1^2 - V2^2)/(I1^2 - I2^2) and the outgoing-wave phase between
the planes. The line constants are thereby measured, not fitted or
assumed — the attempt-1 conditioning channel and the attempt-2
(L, Zc) / (L, l_eff) valley are removed at the source. Beta-wrap guard:
beta*N*dx = 0.287 rad max at 2.6 GHz vs the 0.9*pi guard — clear.
Adopted per-bin constants: mean over the two driven ports (symmetry
gated by F-I2). The S-matrix returned by the refplane path carries
byte-frozen LEGACY diagonals (refuted frame) and is DISCARDED — only
the diagnostics are consumed; the physical S comes from the default
battery path (below).

**I2 — dedicated single-post 1-port fixture (new fixture).** Same
domain/dx/boundary class as the battery, same port + post + trace
cross-section and same port-side overhang, but the trace CONTINUES from
x = X1 - dx through the +x CPML (matched termination) and there is NO
far post: the near post is the only discontinuity. One raw driven run
(refplane opted, N = 10) yields, at each bin, four measured channels
from the same solve:

  - whole-port S11 (the #764 physical diagonal, (V_port - Z0*I)/(V_port + Z0*I));
  - the whole-port incident wave a = (V_port + Z0*I)/(2*sqrt(Z0));
  - its own line constants Zc_sp(f), beta_sp(f) (two-plane invariant);
  - the plane wave pair (out, in), translated to the post-top column
    with the MEASURED beta_sp: out_top = out1*exp(+j*beta_sp*N*dx),
    in_top = in1*exp(-j*beta_sp*N*dx). Gamma_top = in_top/out_top is
    the measured reflection of everything downstream (line + CPML
    termination) at the post top — so the line-termination quality
    DROPS OUT of the model exactly (the load is measured per bin, not
    assumed matched).

  Fit observables per bin (complex): S11_wp and the trans-post launch
  T = (out_top/sqrt(Re Zc_sp)) / (a/sqrt(Z0)); model:
  S11_model = s11_seg + s21_seg^2*Gamma_top/(1 - s22_seg*Gamma_top),
  T_model = s21_seg/(1 - s22_seg*Gamma_top), where (s11, s21, s22)_seg
  is the segment (Zp, tau_p) junction S between reference Z0 (port
  side) and Zc_sp(f) (line side). 7 bins x 2 channels x Re/Im = 28 real
  observations, 2 parameters. Model-derived identifiability at the
  design classes (L = 0.38 nH, tau = 4 ps, Zc = 47.3, noise 0.005):
  corr(L, tau) = 0.25, cond(value-scaled J) = 2.1,
  sigma_L/L = 2.6%, sigma_tau/tau = 5.3% — the transmission channel's
  phase pins tau_p with no far-post/line degeneracy (S11-only would be
  corr 0.998: numerically verified in the design phase and the reason
  this channel is declared).

**I3 — thru 2-parameter refit (primary identification).**
Battery-verbatim thru run WITHOUT the refplane opt-in (byte-identical
production path to attempts 1/2: physical whole-port S, #770
off-diagonals), n_steps = 12000, GaussianPulse(f0=2.0 GHz,
bandwidth=0.8) — attempt-2 run parameters verbatim. Least squares of
(L_p, tau_p) on the full complex 2x2 S at the 7 bins (56 real
observations) with (Zc(f), beta(f)) FIXED at the I1 measured values and
l_trace = 16.0 mm fixed by geometry. Multi-start grid (frozen):
L0 in {0.25, 0.35, 0.45} nH x tau0 in {2.5, 4.0, 6.0} ps; bounds
L in [0.05, 1.0] nH, tau in [0.5, 12] ps (kills the exact
(-Zp, -tau_p) mirror); scipy least_squares trf, xtol/ftol/gtol 1e-14.
Covariance cov = s^2 (J^T J)^-1, s^2 = SSR/(56-2), on the
value-scaled Jacobian for corr/cond. Model-derived identifiability at
the design classes: corr(L, tau) = -0.09, cond = 1.6,
sigma_L/L = 1.2%, sigma_tau/tau = 2.0% at the 0.005 noise class — the
valley is broken because arg S21's delay excess is attributable to
tau_p alone once the line constants are measured.

**Adopted post model** on ALL-PASS: (L*, tau_p*) from I3 (the best
joint fit on the object actually being de-embedded), with I2 the
mandatory independent witness via F-C. Zp* = L*/tau_p*;
C_p* = tau_p*^2/L*.

Why this satisfies "identification from independent data": the two
quantities that poisoned attempts 1-2 (line constants) are MEASURED by
I1's new observable channels, and the post parameters are additionally
measured on I2, a fixture with no far post and no line-length
dependence; I3's remaining 2-parameter fit is gated on achieving the
valley break (F-V1) and on agreeing with I2 (F-C). The thru S data at
the extraction bins re-enters only through I3's fit — never through
any window.

## 4. Extraction-arm falsifiers (frozen; any firing = STOP, no band arm, xfail stays)

- **F-X5 (Re(V/I) first, binding lane-wide)**: Re(Z_in) > 0 at every
  bin of every driven sweep in this lane (thru runs, single-post run,
  band run), checked BEFORE interpreting any |S| >= 1 anywhere.

Instrument gates (I1/I2 line measurements):

- **F-I1 (clean planes)**: max_bin |Im(Zc)/Re(Zc)| <= 0.03 on every
  measured Zc set (thru port 1, thru port 2, single-post). Derivation:
  the refplane module's own measured class boundary (8.2% near-field
  N=3 class vs <= 1.2% clean N=10 class; constant
  `_ZC_IM_RE_WARN_RATIO`).
- **F-I2 (thru port symmetry)**: max_bin |Zc_p1 - Zc_p2| <= 1.2 ohm and
  max_bin |beta_p1/beta_p2 - 1| <= 0.02. Derivation: the fixture is
  mirror-symmetric; 1.2 ohm = 2x the #313 Phase-0 measured plane-to-
  plane spread class (47.9-48.6 ohm -> 0.6 ohm class + the same again
  for the two independent drives).
- **F-I3 (line physicality)**: every measured Re(Zc) in [44, 53] ohm
  (attempt-2 F-J2 derivation: #313 mid-line class +-10% for plane
  definition and low-f dispersion) and every measured beta/(omega/c) in
  [1.00, 1.10] (air quasi-TEM floor 1.00; slow-wave measured class
  1.048-1.062 at 3-7 GHz plus low-f headroom).
- **F-I4 (cross-fixture line consistency)**: max_bin
  |mean(Zc_thru) - Zc_sp| <= 1.2 ohm and |beta ratio - 1| <= 0.02 —
  identical extruded cross-section must read the same line constants on
  both fixtures (same derivation as F-I2).

Fit adequacy and parameter physicality:

- **F-A1 (thru refit residual — model adequacy)**: over 4 entries x 7
  bins, max |S_fit - S_meas| <= 0.025 and rms <= 0.012. Derivation
  (re-done for the new model; NOT copied from attempt-2's measured
  residuals): loss/radiation absent from the lossless model (prior
  Re(x) class <= 0.8 ohm -> 0.008) + junction parasitic beyond the
  segment model (<= 12 fF prior class -> omega*C*Z0/2 = 0.005 at
  2.6 GHz) + per-bin scatter of the measured constants entering as
  residual (0.3 ohm bin-scatter class -> 0.006) + DFT/float class
  0.005. Sum 0.024 -> 0.025; rms class ~half. (Smooth constant BIAS is
  absorbed into (L, tau) — bounded by F-C and the band budget, not by
  this residual gate; verified numerically in the design phase:
  +0.6 ohm bias leaves resid_max 0.0015.)
- **F-A2 (single-post fit residual)**: over 2 channels x 7 bins, max
  complex residual <= 0.06 and rms <= 0.03. Derivation: the plane
  transmission channel carries the #313 plane-path amplitude class
  (referee residual -1.18..-2.54% -> 0.025) + phase class 0.010 rad ->
  0.010 + the F-A1 terms (0.008 + 0.005 + 0.006 + 0.005). Sum 0.059 ->
  0.06; rms half.
- **F-P (parameter physicality — the falsifiable prediction, declared
  before extraction)**: BOTH the I3 and I2 fitted parameters must land
  in L_p in [0.20, 0.50] nH and tau_p in [1.67, 8.34] ps (derivations
  in section 2).
- **F-V1 (valley broken on the thru — the attempt-3 core claim)**: from
  the I3 fit covariance: |corr(L, tau)| <= 0.90 (attempt-2 measured
  pairs -0.991/-0.996), cond(value-scaled Jacobian) <= 60 (materially
  below the attempt-2 ~170; model-derived expectation 1.6, window =
  ~35x expectation but still < half the refuted class),
  sigma_L/L* <= 0.10, sigma_tau/tau* <= 0.15, and the multi-start
  converges to a single basin (all starts within 3 sigma of the best,
  or second-basin cost >= 2x). Any clause failing = the valley is NOT
  broken -> STOP (this is the task's explicit stop condition).
- **F-V2 (single-post identifiability)**: from the I2 fit covariance:
  |corr(L, tau)| <= 0.90 (expected 0.25), cond <= 15 (expected 2.1),
  sigma_L/L <= 0.10, sigma_tau/tau <= 0.20, single basin (same rule).
- **F-C (cross-fixture consistency — the independence witness)**:
  |L*_I3 - L*_I2| <= 0.11 nH and |tau*_I3 - tau*_I2| <= 2.6 ps.
  Derivation (linear sum of design-phase INJECTED systematic classes +
  3 sigma statistics, computed from the model before any measurement):
  plane-channel phase class 0.010 rad -> (0.037 nH, 1.11 ps); amplitude
  class 2% -> (0.004 nH, 0.06 ps); Zc_sp class 0.6 ohm ->
  (0.0003 nH, 0.05 ps); thru-side measured-constant classes (0.6 ohm,
  1% beta) -> (0.035 nH, 0.74 ps); 3 sigma statistical
  (0.033 nH, 0.68 ps). Sums: 0.109 nH -> 0.11; 2.64 ps -> 2.6. Wide but
  honest — it still refutes the attempt-2 failure class (a 7.8 ps
  delay-misassignment is 3x this window).

## 5. Band arm (only if section 4 all-holds) — re-derived budget and falsifiers

De-embed: `rfx.deembed.deembed_line_segment` (NEW, section 6) removes
the identified segment (Zp*, tau_p*) at both ports of the
battery-verbatim band run (linspace(3, 7, 9) GHz, n_steps 4000,
GaussianPulse(f0=5 GHz, bandwidth=0.8)) — the post model AT the battery
dx, as required (the discrete thin-wire parameters are dx-dependent by
construction; a fine-dx sweep is corroborative only, section 8).
The gate band 3-7 GHz sits INSIDE the span where the segment model's
frequency behaviour is anchored (quasi-static parameters identified at
1.4-2.6 GHz; the segment form itself supplies the frequency
dependence), and the identified model interpolates— it never fits —
these bins.

Frozen budget formula, numerically frozen the moment (L*, tau_p*) are
adopted (before the band measurement runs):

    B(L*, tau_p*) = 0.0430                    line mismatch (#313 in-band
                                              measured worst Zc = 47.9 ->
                                              |Gamma| = 0.0215, x2/(1-..) —
                                              attempt-1 derivation unchanged)
        + 2 * (delta_L * omega_7GHz) / (2*Z0)         series-error term
        + 2 * (omega_7GHz * delta_C * Z0 / 2)         shunt-error term
        + 0.012                               junction parasitic beyond the
                                              segment model (12 fF class)
        + 0.005                               complex64/DFT float class
    with delta_L = 0.13 * L*   (thru-side systematic classes 0.6 ohm Zc +
                                1% beta -> 9.3% worst, design-phase
                                injection + 3 sigma stat 3.7%)
         delta_C = 0.60 * C_p* (delta_C/C = 2*delta_tau/tau + delta_L/L
                                with delta_tau/tau = 0.23 same derivation)
    and the absolute ceiling B <= 0.13 (binds whenever the formula
    exceeds it; at the design classes the formula reads 0.159, so the
    EFFECTIVE PRE-DECLARED GATE IS 0.13 unless the adopted parameters
    are materially smaller than the design classes).

Falsifiers:

- **F-D1 (floor)**: max in-band (|S11_dut|, |S22_dut|) < B(L*, tau_p*).
- **F-D2 (reduction)**: max in-band de-embedded diagonal <
  0.5 * 0.2910 = 0.1455.
- **F-X1 (passivity)**: de-embedded per-bin sv_max <= 1.01, interpreted
  only after F-X5.
- **F-X2 (reciprocity preserved)**: max in-band |S21d - S12d| <= 1e-3.
- **F-X3 (off-diagonal magnitude)**: per-bin |S21d| in [0.93, 1.005].
- **F-X4 (raw paths untouched)**: no shipped extraction/scan/decomposer
  code edited; wire/lumped batteries (marker override), refplane and
  dump/replay suites pass unchanged; de-embed remains opt-in
  post-processing. (One pre-declared additive repair rides along:
  the example-fidelity discovery gate currently FAILS on the base
  branch because attempts 1/2 never classified their harnesses; this
  lane adds the missing classification entries + surgical snapshot
  keys, the #770-precedent fix. No other gate moves.)
- De-embedded S21 phase vs analytic line delay: REPORT-ONLY.

## 6. Code plan (additive only; default paths byte-identical)

- `rfx/deembed.py` gains `deembed_line_segment(s_matrix, freqs,
  segments, z0)` — exact wave-cascade removal of a lossless line
  segment (zc_seg, tau_seg) per port via the existing `_s_to_t`/
  `_t_to_s` helpers. NEW function only; nothing shipped calls it. Unit
  tests in `tests/test_deembed.py` (embed with INDEPENDENT ABCD
  arithmetic written in the test): round-trip 1e-12 class, zero-length
  identity, reduction to `deembed_series_inductance` in the tau->0,
  Zp*tau = L limit, input validation.
- Measurement harness
  `validation/research/thru_feedpost_twoseg_extraction.py`: imports the
  attempt-1 builder (`build_thru`, byte-shared fixture), adds the
  SEPARABLE single-post builder `build_singlepost(pulse,
  reference_plane_cells=None)` (returns a Simulation, no solve call —
  classified `audited` in the example-fidelity table), plus arms
  `--verify` (section 9 synthetics), `--insitu`, `--singlepost`,
  `--extract`, `--band`. Raw plane phasors for I2 come from the same
  `_forward_from_materials(..., _return_raw_port_sparams=True)` hook
  the refplane plumbing tests use; extraction math reuses the module's
  own public functions (`refplane_centered_current`,
  `refplane_zc_two_plane`, `refplane_split`, `refplane_beta`).
- Example-fidelity classification entries for all three feed-post
  harnesses (attempt-1 `audited`, attempt-2 `no_simulation`, attempt-3
  `audited`) + surgical snapshot keys; every pre-existing key
  byte-untouched.

## 7. Held-out structure (frozen)

The identification band is 1.4-2.6 GHz on BOTH fixtures; the gate band
is the battery's 9 bins linspace(3, 7) GHz on the thru. No gate-band
bin enters I1, I2, or I3; no identification bin enters any gate. The
identified (L*, tau_p*) cross into the band arm only through the frozen
segment model, whose adequacy is gated in the identification band
(F-A1/F-A2) and whose in-gate-band consequence is bounded by
B(L*, tau_p*). Smooth parameter dispersion undetectable at 1.4-2.6 GHz
is caught by the held-out F-D1 (verified on synthetics, V3).

## 8. Cluster-scale arm (NOT run in this lane)

The fine-dx single-post convergence sweep is CORROBORATIVE, not
decisive: the band de-embed must use the battery-dx post model by
construction (the discrete thin-wire L is dx-dependent), so a fine-dx
study characterizes rasterization convergence of the post parameters
but cannot replace the battery-dx identification.
`validation/research/thru_feedpost_singlepost_vessl.yaml` is updated to
the attempt-3 harness as a proposal for the orchestrator; nothing in
this lane runs vessl.

## 9. Apparatus verification (synthetics, run AFTER this commit and BEFORE any FDTD)

Independent ABCD arithmetic in the harness (cross-checked once against
`rfx.deembed.deembed_line_segment`'s inverse at 1e-12):

- **V0 (plane-channel math)**: synthetic two-wave uniform line phasors
  (the refplane test pattern): Gamma_top, Zc, beta, and the T-channel
  translation recovered exactly (1e-9 class).
- **V1 (exactness)**: segment truth (0.38 nH, 4.0 ps, Zc 47.3,
  beta-factor 1.055): I3-style thru fit with truth constants AND
  I2-style single-post fit (with a nonzero synthetic Gamma_top class
  0.02) each recover (L, tau) to <= 1e-6 relative.
- **V2 (attempt-2-style l_eff absorption RESOLVED by the new design)**:
  on the same segment truth, (a) the attempt-2 3-parameter flat-L
  apparatus (imported verbatim) must ABSORB the post transit — fitted
  l_eff pulled >= 0.5 mm off the 16 mm geometry and L biased >= 10%
  low, at small residuals (the failure signature; model-derived
  expectation: l_eff -> 17.0 mm, L -> 0.31 nH); (b) the attempt-3
  pipeline on the same data must recover the truth to <= 1% in both
  parameters. This is the required demonstration that the independent
  data resolves what the thru-only fit cannot.
- **V3 (held-out teeth)**: dispersive truth L_p(f) = 0.38 nH *
  (1 - 0.3 f/2.6 GHz) with tau_p fixed: absorbed at the identification
  bins (small residuals), then the synthetic band arm de-embedded with
  the fitted flat parameters must FIRE F-D1 against the frozen budget —
  the smooth-dispersion teeth live in the held-out band, as attempt 2
  established.
- **V4 (uncertainty machinery / Fisher bar)**: V1 truth + iid complex
  noise 0.005 per entry: parameter pulls <= 3 sigma on both fits, and
  the F-V1/F-V2 sigma windows must HOLD at that noise class (the
  model-derived expectation says they hold with >= 3x margin — if the
  synthetic shows otherwise the apparatus or the derivation is wrong
  and must be fixed BEFORE any FDTD, windows untouched).
- **V5 (systematic-injection reproduction)**: the design-phase F-C
  derivation table (T-phase 0.010 rad -> ~(0.037 nH, 1.11 ps) etc.) is
  reproduced by the committed harness code to 10% — pinning the F-C
  window's provenance into the apparatus itself.

Any V-check failing = apparatus/derivation bug; fix the apparatus,
never the windows, and re-run all V-checks before any FDTD.

## 10. Dispositions (frozen)

- ALL of section 4 and section 5 hold -> replace the strict xfail on
  `test_thru_s11_floor` with the measured physical de-embedded floor
  lock. In-file provenance: raw 0.2910 -> measured de-embedded worst;
  the full post model (L*, tau_p*, Zp*, both ports, this note); the
  fixtures (single-post + in-situ two-plane) and the held-out
  structure. Gate value = measured de-embedded worst with >= 25%
  headroom, REQUIRED <= B(L*, tau_p*). Plus: raw fixture-physics
  envelope pin (raw worst in [0.20, 0.35], measured 0.2910) and the
  raw alive floor (> 0.02) on the RAW diagonals only.
- Any section-4 falsifier fires -> STOP before the band arm; record in
  an appended results section; xfail stays byte-untouched; the
  corrected VESSL yaml is emitted for the orchestrator ONLY if a
  coarser-cause remedy is named (e.g. rasterization-scale ambiguity ->
  fine-dx sweep).
- F-D1 or any F-X* fires -> STOP; record honestly; xfail stays.
- Movers beyond the one xfail replacement and the pre-declared
  fidelity-table repair (F-X4): NONE planned; any unexpected mover is
  STOP-and-report, not re-pin. No window above is widened after any
  measurement, under any rationale.

## 11. PRE-RESULT apparatus finding — the I2 fixture as declared is UNREALIZABLE; repaired realization (appended 2026-08-29, BEFORE the repaired fixture is measured; sections 1-10 unchanged)

The first `--extract` run (harness commit a1ec264's tree; full log kept)
realized I1 and I3 exactly as declared — their measurements stand and
are quoted in the results section below — but the I2 single-post
fixture as BUILT did not realize the section-3 design. Measured
witness: |Gamma_top| = 0.989..0.997 across the identification bins and
port Re(Z_in) = 0.34 ohm — the line is terminated nearly TOTALLY
reflectively, not matched. Mechanism (verified in source, not
conjectured): `extend_cpml_pad_materials` extends only
eps_r/sigma/mu_r into the CPML padding; the PEC mask is never extended
(`rfx/api/_compile.py` pad-extension block; `pec_mask` comes solely
from interior rasterization). A PEC trace drawn to the domain edge
therefore ENDS at the pad interface as an open circuit — "a PEC trace
continuing through the CPML" does not exist in this codebase. With
|Gamma_top| ~ 1 the fixture is resonant: the trans-post channel's
1/(1 - s22*Gamma_top) denominator amplifies every systematic, the
measured |T| reached 7.2, and the I2 fit hit its bounds with residual
6.1 — an apparatus-invalid reading of a fixture that was not the
declared one. Per the frozen V-check discipline this is an apparatus
bug: FIX THE APPARATUS, NEVER THE WINDOWS. The F-A2/F-V2/F-C
evaluations from that run are void as instrument readings (recorded,
not interpreted); every I1/I3 number is from declared fixtures and
stands.

Repaired realization (committed BEFORE the repaired fixture is run):

- The single-post fixture's line is terminated in the VALIDATED
  matched-termination class this stack actually has: a passive
  (excite=False) 50-ohm wire port — the battery thru's own far
  termination, validated through #683/#764 — placed at x = 28 mm on a
  20 mm line (trace X1 - dx .. 28.5 mm; a genuinely distinct fixture
  from the 16 mm thru). The port-side geometry (port, post, overhang,
  cross-section) is byte-identical to the declared builder.
- NOTHING else changes: the observables (S11_wp, T with measured
  Gamma_top, Zc_sp, beta_sp), the model, the identification bins, and
  every window of sections 4-5 are untouched. The declared design
  already handles an arbitrary DOWNSTREAM network exactly — Gamma_top
  is measured at the post-top plane, between the near post and
  everything else — so the far termination's own post sits inside the
  measured load and never enters the model: the near-post separation is
  achieved by wave-splitting construction, as section 3 declared.
  Expected |Gamma_top| class ~0.03..0.08 (far-post mismatch + Z0-vs-Zc
  mismatch at 1.4-2.6 GHz), i.e. the conditioning regime of the V1
  synthetic.
- New apparatus VALIDITY PRECONDITION (an assert, not a falsifier — it
  can only stop the harness, never pass anything): the realized
  single-post fixture must measure max_bin |Gamma_top| <= 0.5, else the
  fixture again failed to provide a measurable load and the harness
  stops loudly instead of feeding a resonant channel to the fit.
