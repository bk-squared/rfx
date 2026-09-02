# Issue #770 — off-diagonal receive channel adjudicated against external physics: derivation + pre-declaration

Status: DERIVATION AND PRE-DECLARATION, written and committed BEFORE the
harness exists and BEFORE any adjudication measurement is run.  Branch
`agent/issue-770-offdiag`, based on `agent/issue-683-decomposer-flip`
(41002b8 — the POST flip + decomposer recalibration, validated P2/P3/P4/P5).
Falsifiers in §4 are binding and are never widened.  Append-only.

Prior measured facts consumed here (NOT re-derived, NOT re-measured):

- #683 verdict (2026-08-29): POST-injection V/I/V_port sampling is the
  terminal-consistent wire-port order at an excited port; at the driven
  port `V_port + Z0·I = Z0·Î0` is a drive-only constant measured to
  `n·a = +0.999/+0.996` against the known-load circuit law, and
  `V_port = +Z_L·I`, `I = V_src/(Z0+Z_L)` (load law slope 0.9996).
- #764: the driven diagonal `S_jj = (V_port − Z0·I)/(V_port + Z0·I)` is
  load-tracking physics on BOTH lanes (matched → 0.001-0.0125, short →
  −1 class, Γ_L regression slope 0.9996).  Passive per-cell law:
  `−v_c = +Z0c·i_c` with `Z0c = Z0/n_live` at a matched port cell
  (measured −V/I = 16.6672 Ω vs 16.667 Ω); per-bin live-cell |I| spread
  ≤ 0.0062 (F9); per-cell V collapse ratio V7/V8 = 1.40 class on the
  short (F8 mechanism — cells move together).
- #308 (2026-07-10): the per-cell receive channel is the orthogonal
  combination `b_i = (v − Z0c·i)/(2√Z0c)`; the direct combination
  `(−v − Z0c·i)` structurally cancels at a matched receive cell; sign
  pinned by the DC witness S21(DC) → +1 on the canonical thru.
- #313 (2026-07-10, RESOLVED via the opt-in reference-plane path): the
  default port-cell |S21| magnitude is deflated κ(f) = 1.49–1.86
  (frequency-dependent) on the canonical thru; the flux referee measured
  a net transmitted power fraction 0.959–0.998 across 3–7 GHz (implied
  flux-true |S21| = 0.971–0.997, lossless closure gap 0.002–0.040); the
  EXTERNAL openEMS referee (VESSL 369367246600) measured |S21|
  0.973–1.034 on coordinate-identical geometry with the rfx refplane
  path inside it at 0.983–0.998.  Ledger do-not-repeat: "the port-cell
  |S21| envelope (0.52–0.67 on the canonical thru) is a REGRESSION LOCK,
  not physics."
- #683 x #764 flip results (§6–7 of the flip pre-declaration): the
  uniform-lane off-diagonals are bit-identical through the flip (P2 = 0.0),
  reciprocity 7.5277e-3 unchanged; the physical thru diagonal is
  |S11| 0.0093–0.2896 / |S22| 0.0176–0.2910 with the whole-port
  Z_in(3 GHz) = 49.1 − j0.05 Ω rising to 42.9 + j27.1 Ω at 7 GHz (the
  un-de-embedded feed-post reflection, +j27 Ω series class).
- Canonical THRU battery locks (tests/unit/sparams/test_lumped_twoport_vi_validation_battery.py):
  shipped per-cell |S21| = 0.54606–0.60974 (band lock [0.35, 0.85]);
  phase-dev band [−1.1, −0.1] rad; reciprocity gates 1.5e-2 abs / 0.10
  rel (measured 7.53e-3 / 0.0136); sv_max gate 0.85 (measured 0.6934
  post-flip); DC anchor dev band (−0.25, +0.10) rad at 0.5/1.0 GHz.

## 1. The two frames, written on the POST-consistent channels

Drive run `j`; FDTD signs (`v = −E_mid·d_par` at the port midpoint cell,
`v_port = Σ_live(−E_c·d_par,c)`, `i` = Ampère loop at the midpoint cell);
`v`, `i`, `v_port` sampled POST-injection (the true field level `E^{n+1}`,
#683); `v_ref` = the PRE-injection drive sample carried as the #308/#313
calibration reference (uniform lane; bit-identical to the pre-flip drive
sample).  `Z0c = Z0/n_live`.

**Frame P (per-cell #308, the shipped off-diagonal):**

    a_j^P = (−v_ref[j,j] + Z0c_j·i[j,j]) / (2√Z0c_j)
    b_i^P = (v[j,i] − Z0c_i·i[j,i]) / (2√Z0c_i)          (i ≠ j)
    S_ij^P = b_i^P / a_j^P

**Frame W (whole-port, the #764-consistent completion):**

    a_j^W = (v_port[j,j] + Z0_j·i[j,j]) / (2√Z0_j)
    b_i^W = s · (v_port[j,i] − Z0_i·i[j,i]) / (2√Z0_i)   (i ≠ j)
    b_j^W = (v_port[j,j] − Z0_j·i[j,j]) / (2√Z0_j)
    S_ij^W = b_i^W / a_j^W

with `s ∈ {+1, −1}` a single global receive sign for component-homogeneous
multiports, pinned empirically by the DC witness exactly as #308 pinned
frame P's sign (the pin is part of the frame definition, declared here; it
is not a tuning knob — a π-scale phase residual AFTER the pin refutes the
frame, §4 F-A4).  Note `S_jj^W ≡ (v_port − Z0·i)/(v_port + Z0·i)` — the
diagonal of frame W IS the validated #764 physical diagonal, by algebra,
with no additional freedom.

## 2. Derivation of frame W's terminal claims (algebra on measured laws)

(i) **The incident wave.**  At the driven port the measured #683 circuit
law gives `v_port + Z0·i = Z0·Î0` — a drive-only constant, independent of
the load (measured invariance: F0 per-bin spread ≤ 0.0142 across
matched/short/open).  So `|a_j^W|² = Z0·|Î0|²/4` is exactly the available
power of the Norton drive `(Î0, Z0)` — the physically correct incident
normalization.  Frame P's `a_j^P` is instead a PRE-sample blend: with
`v_ref = v_post + cb_mid·v̂_src/n_live` (G2 identity, exact) and
`v_post = Z_Lc·i`, `a_j^P = ((Z0c − Z_Lc)·i − cb_mid·v̂_src/n_live)/(2√Z0c)`
— a fixture- and drive-coefficient-dependent quantity that is not the
incident wave of any transmission-line model.  It was never claimed to
be: it is the #308/#313 calibration reference, and its measured composite
deflation on the canonical thru is κ(f) = 1.49–1.86 (#313, drive-side).

(ii) **The receive wave.**  At a passive port every quantity is
slot-invariant (injection writes no cell there).  The port is n_live
series-terminated cells realizing the whole-port Z0 (#318).  With the
per-cell law `−v_c = Z0c·i_c` at a matched cell, the DIRECT whole-port
combination cancels structurally — `v_port + Z0·i ≈ Σ(v_c + Z0c·i_c) → 0`
— which is the whole-port image of the #308 cancellation: the vanishing
combination is the wave RE-INCIDENT from the matched termination, and it
vanishes precisely because the termination is matched.  The surviving
orthogonal combination `v_port − Z0·i` is the wave delivered by the DUT
into the termination — the outgoing wave b_i.  Frame W is therefore the
whole-port completion of the #308 selection, not a different channel:
under the measured per-cell uniformity (F9) `v_port ≈ n_live·v_mid` and

    b_i^W ≈ √n_live · b_i^P.

(iii) **Power meaning.**  For any terminal pair,
`|a|² − |b|² = Re(V·conj(I))·(sign conv.)` — the waves of frame W conserve
power against the SAME whole-port terminal pair (V_port, I) that #764
validated as physical.  Frame P's per-cell waves reference Z0c against a
single-cell v and do not close power at the port plane (the #313
mechanism finding).  This is what makes the #770 adjudication external:
`|S_jj|² + |S_ij|²` computed in frame W is a statement about physical
power, checkable against the flux/openEMS referees.

(iv) **Predicted frame ratio** (reported, never a gate):
`S_ij^W / S_ij^P = √n_live,i · (a_j^P / a_j^W)` per drive column — on the
canonical thru (n_live = 2) the √2 receive-side factor times the
drive-side reference ratio must land on κ(f) = 1.49–1.86 if both #313's κ
decomposition and this derivation are right.  Reported per bin.

(v) **The ledger do-not-repeat clause** ("never anchor a drive-side wave
at the port cell") is addressed, not overridden: its recorded mechanism
was the PRE-injection sampling factor plus the per-cell frame mismatch.
Both are exactly what #683 (POST samples are the true field level) and
#764 (whole-port terminal pair, measured Γ_L-exact) removed.  Whether the
port-plane wave pair is NOW physical is precisely what §4 measures; it is
not assumed.

## 3. Fixture and lanes (declared)

**FIX-T** — the canonical THRU battery geometry, verbatim
(`tests/unit/sparams/test_lumped_twoport_vi_validation_battery.py::_build_thru`):
32×20×10 mm, dx = 0.5 mm, CPML(x, y, z_hi) + PEC z_lo ground, 16 mm
air microstrip w/h = 5 (Zc ~ 50 Ω), two ez wire ports extent 1 mm
(n_live = 2, one dead extent cell each — the pinned advisory set), Z0 =
50 Ω, GaussianPulse(f0 = 5 GHz, bw = 0.8), 4000 steps, 9 bins 3–7 GHz
(all in-band; no band-edge gate).  NU-lane realization: identical
geometry with a uniform-valued `dz_profile` (the lane-parity module's
construction) — bit-identical discretization, so any frame difference is
the extractor.  Uniform-lane realization: the same Simulation without
`dz_profile`, driven through `compute_lumped_wire_s_matrix_via_scan(...,
return_vi_dump=True)` for the all-port raw channels (v, i, v_port,
v_ref).

**FIX-T-DC** — same geometry, GaussianPulse(f0 = 2.5 GHz, bw = 1.0),
12000 steps, bins 0.5/1.0 GHz (the committed DC-anchor arm).  Drive
port 1 only (the S21 column is what the witness reads).

NU lane first: the NU drive-1 and drive-2 runs are the primary
adjudication measurements; the uniform bundle supplies frame P's shipped
values (v_ref lives only there) and the frame-W lane-parity check.

Preflight runs and is quoted verbatim on every fixture build; the pinned
advisory set is `{pec_faces_finite_pec, wire_port_dead_extent_cells ×2}`.
Any other set, n_live ≠ 2, or a non-finite accumulator = FIXTURE INVALID
(fix the fixture, gates untouched) — distinct from a falsifier verdict,
per the #764 precedent.  One run per arm; no tuning, no step-count or
bin changes after the first measurement.

## 4. Pre-declared falsifiers (binding; committed before the harness;
##    never widened; any failure outside them = STOP and report)

All magnitude gates evaluate the 9 in-band bins 3–7 GHz only; the phase
anchor evaluates 0.5/1.0 GHz only.  `S_jj` in every closure expression is
the #764 physical whole-port diagonal (frame P defines no physical
diagonal of its own; the shipped matrix is mixed-frame).  External
anchors, committed as the targets BEFORE measurement (no self-consistency
values anywhere): flux-referee net-through fraction 0.959–0.998, openEMS
|S21| 0.973–1.034, rfx refplane 0.983–0.998, all on this geometry class.

- **F-A1 — passivity ceiling (each frame, each drive j, per bin):**
  `|S_1j|² + |S_2j|² ≤ 1.02` with the physical diagonal and the frame's
  off-diagonal.  Any in-band violation refutes THAT frame (inflation
  error), localized by construction to its receive/incident channel.
- **F-A2 — power-closure adjudication (each frame, each drive j, per
  bin):** net-through fraction `T ≡ |S_ij|² / (1 − |S_jj|²)` must lie in
  `[0.90, 1.02]` for the frame to be magnitude-physical.  The window is
  derived from the external anchors (0.959–0.998 measured net fraction,
  −0.06 honest low-side margin for the port-plane vs de-embedded-plane
  difference and float32; +0.02 float headroom), NOT from any rfx port
  reading.  Pre-registered separation: frame P's committed regression
  class predicts T ≈ 0.30–0.41; if frame P nevertheless measures inside
  [0.90, 1.02] it SURVIVES and outcome (a) applies.
- **F-A3 — reciprocity (each frame):** max in-band |S21 − S12| ≤ 1.5e-2
  absolute OR max |S21 − S12|/|S21| ≤ 0.10 (the battery's own committed
  gate pair, applied unchanged).  Frame W failing BOTH arms is refuted.
  Report both frames' residuals against the locked 7.5277e-3 class
  (shrink or grow — the #770 item-2 question) — reported, not gated
  beyond the committed pair.
- **F-A4 — DC/phase anchor (frame W, FIX-T-DC):** after the single
  global sign pin `s`, wrapped `arg(S21^W) − (−2πfL/c)` at 0.5 and
  1.0 GHz must lie in `(−0.25, +0.10)` rad (the committed _DCA band,
  unchanged), and the sign-flipped channel must leave the band (the
  committed π-discrimination witness).  A π-scale residual after the pin
  refutes frame W.
- **F-A5 — lane parity (frame W):** computed from the NU raw
  accumulators and from the uniform bundle on the same geometry:
  per-bin max |S^W_NU − S^W_uniform| ≤ 1e-3 over all four entries
  (the flip's measured excited-port parity is 1.8e-7 class; 1e-3 is
  cross-channel float headroom, still 15× under any gate above).
- **F-A6 — no passive mover, no code change during measurement:** the
  harness is measurement-only (spy capture + offline algebra); the
  shipped extractors are not edited before the verdict.  Frame values
  are computed from raw accumulators; the shipped S must be reproduced
  from the same accumulators to ≤ 1e-5 (wiring identity, #764 F-extraction
  precedent) before any frame number is trusted.

**Verdict rule (binding, committed now):**

1. Frame P passes F-A1 + F-A2 + F-A3 → outcome (a): the per-cell frame
   survives external physics; document + lock it as physical; close the
   question; NO code change.
2. Frame P fails F-A2 (or F-A1/F-A3) AND frame W passes F-A1–F-A5 →
   outcome (b): the frame error is confirmed and localized; implement
   the whole-port receive channel with the pre-declared scope of §5,
   every mover re-pinned only from THIS harness's measurements.
3. Frame W fails any of F-A1–F-A5 → STOP: the shipped mixed frame stays,
   the measured design is documented, disposition to review.  No gate
   widening, no re-aim, no third frame invented after the fact.

## 5. Outcome-(b) implementation scope (pre-declared so movers are
##    classified before they move)

Code: `decompose_wire_s_matrix` off-diagonal channel (frame P → frame W;
`v_ref` consumption drops from the off-diagonal; the byte-frozen legacy
diagonal and the #313 refplane decomposer keep `v_ref` and the per-cell
frame VERBATIM — the refplane path is documented byte-frozen legacy) and
the NU `_ab` off-diagonal in `rfx/nonuniform.py` (whose current
whole-port-Z0-against-per-cell-v mix is the open defect #764 §6 named).
The all-passive NU fallback and every excite=False channel stay
byte-frozen (#764 scope note).  Dump/replay: the wire replay bundle
gains an off-diagonal frame marker so pre-#770 dumps replay their
recorded frame (the #764 `raw_port_voltages_fdt=None` precedent).

Admissible mover classes (each re-pinned only with a measured value from
this harness or held xfail with the firing documented): (1) THRU-battery
|S21| band, phase-dev band, sv_max, reciprocity measured quotes, DC-anchor
measured quotes; (2) `test_twoport_wire_port.py` |S21| class quotes;
(3) default-path off-diagonal byte-pins (e.g. refplane-vs-default
inequality/equality pins) in `tests/locks/test_refplane_port_waves.py`;
(4) replay/dump schema pins; (5) NU off-diagonal value pins if any test
carries them.  ANY mover outside these classes = STOP and report.  A
full wire/lumped/sparam/twoport battery WITH the slow_physics marker
override is mandatory before the verdict is recorded.

Interim CPU budget: every run ≤ ~20 min (measured classes: 70 s/drive
uniform, minutes/drive NU, 12000-step DC arm the longest).  No GPU/VESSL
run is required by this plan; if one becomes decisive it is emitted as
YAML, not run.
