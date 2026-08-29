# THRU feed-post de-embedding — pre-declaration (PI disposition (b) on the held #683 gate)

Status: BINDING. Committed BEFORE any new measurement. Sections 1-6 are
frozen at commit time; results are APPENDED in later sections, never
edited back into these. No window below may be widened after a
measurement; a fired falsifier is a STOP for its arm.

Lane: `agent/thru-deembed` (branched from `agent/issue-770-offdiag`).
Author: thru-diagonal de-embedding author agent, 2026-08-29 KST.

## 1. Context (verified standing; not re-litigated here)

The wire-port whole-port stack is physical through #764 (driven diagonal
S_kk = (V_port - Z0 I)/(V_port + Z0 I) = Gamma_L; matched->0, short->-1,
known loads exact), #683 (uniform POST-injection flip + decomposer
recalibration, PR #777) and #770 (whole-port off-diagonals, PR #778).
The one remaining non-physical gate on the THRU battery is
`test_thru_s11_floor` (strict xfail): the physical diagonal measures max
in-band 0.2910 vs the restored < 0.12 floor.

Measured diagnosis (PR #777 design note section 6, P1 gate 5): driven
whole-port Z_in = 49.1 - j0.05 ohm at 3 GHz rising to 42.9 + j27.1 ohm
at 7 GHz; on the symmetric fixture the far port's identical feed-post
series reactance alone predicts |Gamma| ~ 27/104 = 0.26. The 0.2910 is
the fixture's TRUE un-de-embedded feed-post reflection (#318 ledger:
two 1 mm vertical feed posts, ~0.26 nH class each, interfering across
the 16 mm line). The old "0.09 feed-post class" was an extrapolation
from the legacy frame-mismatch artifact and is refuted.

PI disposition (b): DE-EMBED the feed posts via the #313
reference-plane path family, then pin the physical de-embedded floor.

## 2. Derivation: what de-embedding is correct for THIS fixture topology

Fixture (canonical THRU battery, verbatim geometry of
`tests/test_lumped_twoport_vi_validation_battery.py`): at each port a
1 mm vertical wire feed post (Ez column, 3 cells, top cell dead inside
the PEC trace) connects the lumped port terminal pair (ground plane to
trace) to the 16 mm air-microstrip line (measured Zc = 47.9-48.6 ohm,
measured beta = 1.048-1.062 x omega/c — #313 Phase-0 constants, not
re-derived).

Network topology between the two port terminal planes:

    port1 [Z0=50] -- series x1 -- line(Zc, beta, l=16mm) -- series x2 -- port2 [Z0=50]

with x_k = j*omega*L_k the post's series reactance (quasi-static class;
the post is electrically short: 1 mm << lambda/10 up to > 15 GHz, so a
single lumped series element is the correct leading-order model).

**Why the naive diagonal subtraction Z_in - j*omega*L is NOT the
de-embed**: it removes only the NEAR post. The FAR post sits behind the
line and still reflects; with the near post removed the residual is the
line-transformed far-post mismatch, |Gamma_far| = omega*L /
sqrt((2*Z0)^2 + (omega*L)^2) ~ 0.15 at 7 GHz for L = 0.35 nH — the gate
would still fail for fixture (not extraction) reasons.

**Correct de-embed — full 2-port cascade removal.** The measured 2-port
S (whole-port frame, PR #778) is the wave-cascade

    T_meas = T_post(x1) . T_line . T_post(x2)

so the bare-line ("DUT") S-parameters follow exactly from

    T_dut = T_post(x1)^-1 . T_meas . T_post(x2)^-1,

where T_post(x) is the wave-transfer matrix at reference Z0 of the
series element with S11 = x/(x + 2*Z0), S21 = 2*Z0/(x + 2*Z0). This is
standard, exact network algebra; no line model enters the de-embed
itself (only L does). Equivalent ABCD form: ABCD_dut =
[[1,-x1],[0,1]] . ABCD_meas . [[1,-x2],[0,1]].

**Where the #313 refplane phase machinery fits**: it does not apply
here, by derivation. The refplane path shifts reference planes ALONG
the line by N*dx with the MEASURED beta (phase-only, magnitude-neutral)
— a transmission-line plane translation for wave quantities sampled
outboard. The feed post sits between the port's lumped terminal plane
and the line's wave frame at the SAME x-column; there is no line length
between the port plane and the post, so no exp(+/- j*beta*N*dx) factor
belongs in this operation. Post removal is a lumped, magnitude-active
series cascade; the two operations are orthogonal and composable.
Consequence: `rfx/probes/refplane.py` is NOT touched by this lane; the
#764 open-question-4 byte-INEQUALITY pin
(`test_refplane_port_waves.py::test_run_short_diagonals_byte_frozen_offdiagonals_move`)
stays valid and untouched. Re-unifying refplane diagonals onto the
whole-port channel remains a deliberately open, separate decision.

**Effect on S21 (derived, stated up front)**: the cascade multiplies
the off-diagonals by (x + 2*Z0)/(2*Z0) per side (exact for the
symmetric series element), i.e. |S21| rises slightly (the post's
reflective insertion loss is removed) and the S21 phase LOSES the
feed-post group-delay excess. The raw-S battery gates (|S21| band,
signed phase-deviation band, reciprocity, sv_max) gate the RAW matrix
and are untouched; the de-embedded off-diagonals get their own
cross-check windows in section 5 and a report-only phase comparison
(no pre-measured de-embedded phase baseline exists, so no phase GATE is
declared — reported, not gated).

**Reciprocity is preserved exactly**: for equal posts both off-diagonal
entries acquire the same factor, so S21d/S12d = S21/S12 in exact
arithmetic.

## 3. Code plan (additive only; default paths byte-identical)

- `rfx/deembed.py` (post-processing module) gains
  `deembed_series_impedance(s_matrix, freqs, series_z, z0=50)` and the
  convenience `deembed_series_inductance(s_matrix, freqs, inductances,
  z0=50)` implementing section 2 via the existing `_s_to_t`/`_t_to_s`
  helpers. NEW functions only; no shipped extraction/scan/decomposer
  code is edited anywhere in this lane, so every existing path is
  byte-identical by construction. Unit tests (synthetic, fast): exact
  round-trip embed->de-embed of an analytic mismatched line at 1e-10
  class, L=0 identity, input validation. The embed side of the unit
  test is computed with INDEPENDENT ABCD arithmetic written in the
  test, not with the module's own T helpers.
- Measurement harness `validation/research/thru_feedpost_deembed.py`
  (measurement-only, offline algebra on run() outputs; battery-verbatim
  geometry), arms `--extract` (section 4) and `--band` (section 5).

## 4. Pre-declared L extraction — INDEPENDENT of the gate it feeds

The gate (section 5) is the de-embedded floor over the 3-7 GHz battery
bins. L is therefore extracted from OUT-OF-GATE-BAND bins on the same
driven fixture (the task's candidate (i), adapted: the post is the same
physical object at every frequency; the extraction bins share no data
with the gate bins).

Method (frozen):

- Fixture: battery-verbatim THRU geometry; pulse GaussianPulse(f0=2.0
  GHz, bandwidth=0.8) (in-band 1.2-2.8 GHz); n_steps = 12000 (the #770
  DC-arm precedent for low-f DFT settling); s_param_freqs =
  [1.4, 1.8, 2.2, 2.6] GHz. All four bins are BELOW the 3 GHz gate-band
  edge and below the tan(beta*l) singularity at ~4.44 GHz.
- Driven whole-port Z_in per bin and per driven port j from the
  whole-port diagonal: Z_in = Z0 (1 + S_jj)/(1 - S_jj) (exact inverse
  of the #764 definition).
- **Re(V/I) protocol (binding)**: Re(Z_in) > 0 is checked at every bin
  of every driven sweep in this lane BEFORE any interpretation; any
  |S_jj| > 1 reading is interpreted only after that check (F-X5).
- Model inversion (exact, per bin): with t = tan(beta*l),
  Zc = 48.25 ohm, beta = 1.055 * omega/c (centres of the #313 measured
  classes), l = 16 mm, Z0 = 50 ohm, and the SYMMETRIC two-post model

      Z_in = x + Zc*(Z0 + x + j*Zc*t)/(Zc + j*(Z0 + x)*t),

  x solves  a*x^2 + b*x + c = 0  with
      a = -j*t
      b = j*t*(Z_in - Z0) - 2*Zc
      c = Zc*(Z_in - Z0) + j*t*(Z_in*Z0 - Zc^2).
  Root selection: the root with Im(x) > 0 and the smaller |Re(x)|
  (physical series inductance). L_bin = Im(x)/omega. Re(x) is REPORTED
  per bin (a series post is reactive; Re is the loss/radiation
  residual) but not gated — model adequacy is gated by F-L1 and by the
  section-5 floor itself.
- Adopted value: L* = median over the 4 bins of the DRIVEN-PORT-1 arm;
  port 2 is the symmetry witness.
- Corner sensitivity (reported and gated by F-L5): re-solve L* at the
  four corners Zc in {47.9, 48.6} x beta-factor in {1.048, 1.062}.

Falsifiers (frozen; any firing = STOP for the lane, record, leave the
xfail in place):

- **F-L1 (quasi-static flatness)**: max_bin |L_bin - L*| / L* <= 0.20.
  A real series inductance is frequency-flat in a 1.4-2.6 GHz
  quasi-static band; larger spread refutes the single-series-L model.
- **F-L2 (prediction — the falsifiable class, declared before
  extraction)**: L* in [0.20, 0.50] nH. Lower edge: the #318 ledger
  witness class ~0.26 nH minus 25%. Upper edge: quasi-static via/wire
  estimate for a 1 mm post at FDTD thin-wire equivalent radius
  0.135*dx = 67.5 um — L = (mu0/2pi)*[h*ln((h+sqrt(h^2+r^2))/r) +
  1.5*(r - sqrt(h^2+r^2))] = 0.40 nH — plus 25%. Outside this window
  the "series feed-post inductance" hypothesis is refuted at
  extraction, before the gate is touched.
- **F-L3 (symmetry)**: |L*_port1 - L*_port2| / mean <= 0.10 (the
  fixture is geometrically symmetric; measured raw diagonals differ by
  ~0.5%).
- **F-L5 (constant sensitivity honesty)**: max corner |L*_corner - L*|
  / L* <= 0.15. This backs the 15% delta-L term used in B(L*) below;
  if it fires, the budget is dishonest and the lane STOPs rather than
  re-deriving after the fact.

## 5. Pre-declared de-embedded floor — prediction, tolerance, cross-checks

Expected class (derived, not inherited): after exact removal of both
posts the bare line terminated in Z0 both sides remains. Its reflection
is |S11_line| = |Gamma*(1 - e^(-2j*beta*l))/(1 - Gamma^2*e^(-2j*beta*l))|
with Gamma = (Zc - Z0)/(Zc + Z0); worst measured Zc = 47.9 gives
|Gamma| = 0.0215 and |S11_line| <= 2*0.0215/(1 - 4.6e-4) = 0.0430.
So the EXPECTED de-embedded diagonal class is <= ~0.04.

Falsifier bound (worst-case linear budget, frozen as a FORMULA now and
frozen NUMERICALLY the moment L* is adopted — i.e. before the band
measurement is run):

    B(L*) = 0.0430                      (line mismatch, above)
          + 2 * (0.15 * omega_7GHz * L*) / (2 * Z0)
                                        (two posts, coherent worst case,
                                         delta-L/L = 0.15 per F-L5)
          + 0.012                       (non-L post parasitic: quasi-static
                                         junction shunt C <= ~12 fF;
                                         |dGamma| = omega*C*Z0/2 at 7 GHz)
          + 0.005                       (complex64 extraction float class)

  For orientation: L* = 0.35 nH gives B = 0.106; the F-L2 edges give
  B(0.20 nH) = 0.086 and B(0.50 nH) = 0.126. Absolute ceiling
  regardless of L*: B <= 0.13 (from the F-L2 upper edge; if the
  computed B exceeds it, 0.13 binds).

Falsifiers on the de-embedded battery matrix (9 bins, 3-7 GHz,
battery-verbatim fixture and run parameters, de-embed applied offline
with the single adopted L* at both ports):

- **F-D1 (floor)**: max in-band (|S11_dut|, |S22_dut|) < B(L*).
- **F-D2 (reduction)**: max in-band de-embedded diagonal <
  0.5 * 0.2910 = 0.1455 — the de-embed must actually remove the post
  reflection, not reshuffle it (diagnostic granularity if F-D1 fires
  marginally).
- **F-X1 (passivity)**: de-embedded per-bin max singular value <= 1.01
  (the raw gate's class; removal of a lossless reciprocal embedding
  preserves the passivity bound in exact arithmetic). Interpreted only
  after F-X5.
- **F-X2 (reciprocity preserved)**: max in-band |S21d - S12d| <= 1e-3
  (raw measured 2.6678e-4; the symmetric cascade preserves the ratio
  exactly, the window is ~4x the raw class for the |factor| ~ 1.03
  scaling plus float).
- **F-X3 (off-diagonal magnitude)**: per-bin |S21d| in
  [0.93, 1.0 + 5e-3]. Lower edge: de-embedding removes reflective
  insertion loss, so |S21d| must not drop below the raw per-bin min
  class 0.9341 minus float; upper edge: passivity + float headroom.
- **F-X4 (raw paths untouched)**: no shipped extraction code is edited;
  the full wire/lumped battery (marker override), the refplane module
  battery and the dump/replay modules must pass unchanged, and the #683
  known-load harness is unaffected (its fixture has no post).
- **F-X5 (Re(V/I) first)**: Re(Z_in) > 0 at all bins, both ports, on
  every driven sweep in this lane, checked before interpreting any
  |S| >= 1.
- De-embedded S21 phase vs analytic line delay: REPORT-ONLY (expected:
  the -0.81..-0.35 rad feed-post group-delay excess shrinks toward 0);
  no gate is declared because no de-embedded baseline exists to pin.

## 6. Dispositions (frozen)

- All of F-L*, F-D*, F-X* hold -> replace the strict xfail on
  `test_thru_s11_floor` with the measured physical de-embedded floor
  lock: in-file provenance (raw 0.2910 -> measured de-embedded value,
  L* used, extraction method + this note), gate = measured de-embedded
  worst + honest margin, REQUIRED <= B(L*); plus a raw fixture-physics
  envelope pin (raw worst in [0.20, 0.35], measured 0.2910) so a
  regression of the raw physical channel stays loud; raw alive floor
  (> 0.02) kept on the RAW diagonals (the de-embedded diagonal is
  legitimately near-null and gets no alive floor).
- Any F-L* fires -> STOP before the band arm; record in the appended
  results section; the xfail stays byte-untouched.
- F-D1 or any F-X* fires -> STOP; record honestly; the xfail stays.
  A justified STOP is a valid outcome of this lane.
- No window above is widened after any measurement, under any
  rationale. Movers beyond the xfail replacement: NONE planned; any
  unexpected mover is a STOP-and-report, not a re-pin.

## 7. RESULTS — extraction arm (appended 2026-08-29; sections 1-6 unchanged)

Run: `validation/research/thru_feedpost_deembed.py --extract`, branch
`agent/thru-deembed` (harness commit 7282e08), JAX_PLATFORMS=cpu,
battery-verbatim fixture, preflight code set pinned and matched
(`pec_faces_finite_pec` + 2x `wire_port_dead_extent_cells`, quoted
verbatim in the log).

Measured, bins [1.4, 1.8, 2.2, 2.6] GHz:

- F-X5 CLEAN both ports: Re(Z_in) = 51.499/51.838/51.647/50.734 (port 1)
  and 51.227/51.448/51.174/50.242 (port 2), all > 0.
- Port 1 Z_in = 51.499+2.189j, 51.838+1.862j, 51.647+1.178j,
  50.734+0.421j ohm.
- Port 1 per-bin L = [0.2821, 0.2636, 0.2358, 0.1939] nH,
  median L* = 0.2497 nH; port 2 = [0.2828, 0.2645, 0.2372, 0.1968] nH,
  median 0.2508 nH. Report-only Re(x): port 1 +0.05..+0.18 ohm,
  port 2 -0.07..-0.82 ohm.
- Corner medians (Zc x beta corners): [0.2741, 0.2785, 0.2218,
  0.2250] nH.

Verdicts (windows verbatim from section 4):

- **F-L1 flatness: FIRED** — max|L_bin - L*|/L* = 0.2233 vs <= 0.20.
  The per-bin L is a smooth MONOTONE decline (0.282 -> 0.194 nH,
  ~31% across 1.4-2.6 GHz), reproduced to < 1% between the two
  independently driven ports — systematic, not noise.
- F-L2 PASS: L* = 0.2497 nH inside [0.20, 0.50] nH (and squarely the
  #318 ledger witness class ~0.26 nH).
- F-L3 PASS: port symmetry 0.47% vs <= 10%.
- F-L5 PASS: corner deviation 11.5% vs <= 15%.

**Disposition (per the frozen section 6): STOP before the band arm.**
The band arm was NOT run; no de-embedded in-band measurement exists in
this lane. `test_thru_s11_floor` stays byte-untouched as the held
strict xfail. No lock anywhere is moved. The additive
`rfx.deembed.deembed_series_impedance/_inductance` code and its
synthetic unit tests are kept (no shipped path calls them; they are the
verified instrument for a follow-up lane).

### Apparatus verification (comparator-bug discipline, 12/12 ledger class)

Before accepting the firing, the inversion was verified on synthetics
(no FDTD; same harness functions):

1. Exact constants, flat L_true = 0.25 nH -> recovered
   [0.25, 0.25, 0.25, 0.25] nH exactly. The quadratic inversion is
   bug-free; the firing is a property of the measurement, not the
   instrument.
2. Mechanism witnesses (flat L_true = 0.25 nH, one constant biased):
   - Zc_true = 46.0 vs assumed 48.25: recovered [0.100, 0.089, 0.071,
     0.044] nH — a steep monotone DECLINE, the measured signature.
   - Zc_true = 50.5: [0.394, 0.407, 0.429, 0.465] nH — trend sign
     REVERSES. Sensitivity dL/dZc ~ 0.065 nH per ohm at these bins:
     the low-f Im(Z_in) signal (0.4-2.4 ohm) is comparable to the
     line-transformer term t*(Zc - Z0^2/Zc), so a ~1 ohm effective-Zc
     error swings the extraction by ~25%.
   - l_eff = 17 mm vs assumed 16 mm: [0.239, 0.235, 0.230, 0.221] nH —
     mild decline (the trace overhangs each port column by 1 cell, so
     the effective electrical length is genuinely uncertain at the
     +-1 mm scale).
   - beta corners (1.048/1.062 vs 1.055): < 1.3% per-bin — negligible.
   - post shunt C = 10 fF: ~2% decline — weak.

**Named mechanism**: the out-of-band low-frequency arm is
ILL-CONDITIONED in the frozen line constants — chiefly the effective
Zc (the #313 measured 47.9-48.6 ohm class was measured at mid-line
planes over 3-7 GHz and does not transfer to this model at the
+-0.35 ohm fidelity the inversion needs) and secondarily the effective
electrical length (1-cell trace overhangs). A ~1 ohm Zc deficit plus
the overhang length reproduces the measured decline with a flat true L
in the 0.30-0.35 nH range — but selecting those constants AFTER seeing
the trend would be exactly the post-hoc fitting the falsifier exists to
prevent. Under the frozen method, constant-conditioning bias and true
post-reactance dispersion are indistinguishable, so the
single-frequency-flat-series-L extraction is REFUTED as pre-declared
and the STOP stands.

### Follow-up (requires a FRESH pre-declaration; nothing here executes it)

A conditioning-robust L extraction needs constants measured, not
assumed: (a) a dedicated 1-port single-post fixture (post + matched
line into CPML, no far post) whose Im(Z_in) is post-dominated, or
(b) joint (L, Zc, l_eff) inversion using the full complex 2-port
out-of-band data (S21 phase pins beta*l_eff independently), or (c) the
#313 refplane two-plane machinery run AT the extraction bins to measure
Zc(f) in-situ. Any of these must commit its own falsifiers before
running; the B(L*) budget formula of section 5 remains valid for that
future lane. A cluster-scale arm (fine-dx single-post study) is
sketched in `validation/research/thru_feedpost_singlepost_vessl.yaml`
(proposal only; DO NOT run before its pre-declaration exists).


## Post-review correction (2026-08-29, appended — adversarial review finding)

Section 2's parenthetical "(exact for the symmetric series element)" on the
per-side off-diagonal factor (x+2Z0)/(2Z0) is numerically refuted: the true
S21_dut/S21_meas ratio is DUT-dependent (measured 1.0223 vs the claimed
1.0128 on the battery THRU). The cascade removal itself (T-matrix form) is
unaffected — only that parenthetical's exactness claim is withdrawn. The
S21 treatment stays report-only as declared; raw gates stay raw.
