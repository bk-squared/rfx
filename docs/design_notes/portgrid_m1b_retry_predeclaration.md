# SPEC-02 portgrid — F-M1b RETRY pre-declaration (append-only)

Lane: `agent/portgrid-m0m1` · Tracker: #781 · Date: 2026-08-29 (KST)
Author agent session: https://claude.ai/code/session_016E2cSPq3RYrGrrJvS5TLaH
Prior notes (binding, read in full): `portgrid_m0m1_predeclaration.md` (incl. Correction 1),
`portgrid_m0m1_results.md` (incl. Correction 2 — which frames this retry).

This note is committed BEFORE any retry measurement. All windows in §5 are frozen at this
commit and may not be widened after measurement (SPEC-00 §0.2-2). The phase-1 F-M1b verdict
(FIRED) is untouched; this is a NEW falsifier on a paper-faithful fixture, adjudicating
scheme-vs-fixture as Correction 2 item 4 requires.

## 1. Sources re-obtained for this retry (not from memory, not from the phase-1 review)

- arXiv:1606.08761 PDF re-downloaded 2026-08-29 (502587 bytes, 12 PDF pages) and re-read:
  Sec. V-B (material traverse), Sec. V-C (four-rod reflection), Fig. 8 (fixture layout),
  Fig. 9 (both panels), Table I.
- Fig. 9 curve values re-extracted from the PDF's VECTOR path data by this lane's own
  instrument (`validation/research/portgrid/fig9_extract.py`), independently of the phase-1
  review's extraction. Output committed: `portgrid_fig9_extraction.json` (this directory).
  Axis calibration from tick-mark path coordinates; frame top of the bottom panel is
  −30.05 dB (confirming Correction 2: −40 dB is only the topmost tick LABEL; nothing is
  clipped). Extracted anchors (worst curve = r=6, red):

  | quantity | r=2 | r=4 | r=6 | worst |
  |---|---|---|---|---|
  | max dB over [2,20] GHz | −53.46 | −51.57 | −51.24 | **−51.24** |
  | max dB over [2,30] GHz (≡ value @30 GHz) | −36.51 | −34.63 | −34.29 | **−34.29** |

  These agree with Correction 2's review numbers (−51.23 / −34.28) to 0.01 dB.
- Line-width of the printed curves is 0.90 pt ≈ 0.8 dB of reading uncertainty — absorbed by
  the ≥5 dB rule below, as originally declared.

## 2. Paper-faithful fixture (Fig. 8, decoded from the figure's vector data)

Scale check: drawn domain rectangle 149.67 × 90.71 pt with labels "66 mm" / "40 mm" gives
2.2678 pt/mm in both axes; every element below reproduces its dimension label exactly.

- Domain 66 mm × 40 mm, coarse Δx = Δy = 1 mm (λ/10 at 30 GHz, per the text). PEC at
  y = 0 and y = 40 mm. The two x-ends are absorbing (paper: 15 mm PML, x ∈ [0,15] and
  [51,66] mm).
- Jy (electric) line current source spanning all y at x = 17 mm (drawn red line; 2 mm
  inside the left PML's inner face).
- Probe LINE spanning all y at x = 19 mm (drawn gray line; the "20 mm" dimension runs from
  the probe line to the subgrid's left face). Observable: the y-AVERAGE of Ey on the
  x = 19 mm column — the y-average projects out every cos(nπy/H) mode, n ≥ 1, leaving the
  TEM (y-uniform) mode that the y-uniform source launches. (Phase-1 used a single-point Hz
  probe; a point probe also records non-TEM scattered modes — the parallel-plate n = 2
  even mode propagates above 7.5 GHz — one of the fixture-class differences being removed.)
- Fine subgrid: coarse cells [39,47) × [16,24) — the 8 × 8 mm box, centered in y, left face
  20 mm from the probe line. Strictly interior. Ratios: paper runs r ∈ {2,4,6}; this lane
  runs r ∈ {2,3,4,5,6} (odd lane included), all against the same windows — justified by the
  paper's near-overlapping curves (extracted spread ≤ 2.3 dB).
- Scatterer (rod arm only): four COPPER rods, radius 1 mm, centers (41,18), (41,22),
  (45,18), (45,22) mm (2 mm surface-to-surface gaps, 1 mm clearance to the subgrid
  boundary on every side; all decoded from the drawing). Copper is modeled volumetrically
  as conductivity σ_Cu = 5.8e7 S/m on the edges whose centers fall inside a rod — the same
  treatment class as the paper's Sec. V-B "copper" arm. **The rods are NOT PEC**, so the
  routed alternative (PEC-in-fine-region support) is not needed; what is needed — and
  implemented for this retry — is the eq. (58)/(61) σ̂ path and per-edge material
  coefficients (see §4).
- dt = 0.99 × fine CFL of each run's finest grid (paper: "1% below the CFL limit of the
  refined region"); uniform reference of each run uses the SAME dt as that run.
- Excitation waveform: compact-support modulated Gaussian, f0 = 16 GHz, HWHM bandwidth
  10 GHz (spectral magnitude ≥ 25% of peak across [2,30] GHz; the paper does not print the
  V-C waveform — any common waveform cancels in the S11 ratio).
- Record length: 4.0 ns (interface arms) / 6.0 ns (rod arms, extra ring-down for the rod
  resonances); ≥ 18 transits of the 66 mm domain; no time gating (termination absorbs).
- S11 chain (paper Sec. V-C method): reference run = identical grid and termination with
  no subgrid and no rods; incident I(f) = FFT(reference probe trace); reflected
  R(f) = FFT(probe_run − probe_reference); |S11| = |R/I|. No gates.

## 3. Termination: 2D split-field PML (the paper's termination class)

Choice: split-field Berenger PML graded along x only (the guide's PEC walls need no y
absorption), 15 coarse cells (= the paper's 15 mm) at both x-ends, cubic (m = 3) grading,
design reflection R0 = 1e-5, σx on Ey planes and matched σ*x = σx·μ0/ε0 on Hz planes,
outer wall PEC. Hz is split Hz = Hzx + Hzy only in the terminated steppers (the
M1a-verified PEC stepper is untouched). σmax = −(m+1)·ε0·c·ln(R0)/(2·d) ≈ 4.07 S/m·(σmax
scale), giving σmax·dt/(2ε0) ≤ 0.28 for every dt used here (coefficient stability trivial).

Why 1st-order Mur was rejected: its discrete reflection floor, derived exactly from the
scheme's own dispersion relation (plane-wave ansatz on the Mur update; derivation kept in
`m1b_retry.py`), is |R(30 GHz)| ≈ −32 dB at dt = 0.99×fine-CFL(r=6). In this DIFFERENTIAL
chain a termination error only multiplies the island-scattered signal (the common incident
cancels exactly in probe_run − probe_reference — linearity; chain-null re-verified below),
so Mur would bias |S11| by ≤ 4·|R_Mur| ≈ 10% ≈ 0.9 dB — acceptable for the interface arm
(margin 5 dB) but NOT cleanly for the rod cross-check arm, where 4·|R_Mur|·|S11_rod| ≈
0.04 linear is ~40% of that arm's window. The PML floor removes the concern for both.

Pre-declared PML floor requirement (its OWN reflection floor, so it cannot confound):
- **F-M1b-abc**: direct floor measurement — 400 mm × 40 mm uniform coarse guide, PML both
  ends, source column x = 100 mm, probe column x = 150 mm; direct-pulse gate [0, 0.32] ns,
  left-PML echo gate [0.35, 0.70] ns (echo path +100 mm ⇒ +0.333 ns; right-PML echo
  arrives ≥ 1.6 ns — outside the gate); measured at dt(r=2) and dt(r=6).
  Window: measured |R_PML(f)| ≤ −50 dB for all f ∈ [2,30] GHz. Fires ⇒ fix the
  termination before ANY retry arm is judged (the retry is not run over a broken floor).
  Derived confound bound at the floor window: interface arm ≤ 4·10^(−50/20) = 1.3% ≤
  0.11 dB; rod arm ≤ 4·10^(−50/20)·max|S11_fine| ≈ 0.005 linear ≈ 5% of that window.
- Chain-null re-verification: r = 1 island through the FULL retry chain (PML, Jy source,
  column-mean probe) must give max|S11| ≤ −200 dB over [2,30] GHz (pipeline contributes
  nothing; phase-1 PEC/gated chain gave −296 dB).

## 4. Implementation additions declared for this retry (task 2 of the routing)

- `sim2d.make_stepper` (PEC, M1a-verified): gains OPTIONAL per-edge material maps —
  σ on fine edges (σ_fx, σ_fy), and ε,σ on coarse host edges (eps_cx, eps_cy, σ_cx, σ_cy).
  Interface update (61) gets its full material coefficients: with ε̂,σ̂ the r-segment means
  of the fine boundary-row per-edge values (eq. (58)),
  Dp = (ε_c + ε̂/r)/dt + (σ_c + σ̂/r)/2, Dm = (ε_c + ε̂/r)/dt − (σ_c + σ̂/r)/2,
  ca = Dm/Dp, cb = (2/δn)/Dp. Defaults (vacuum, lossless) reproduce the current
  coefficients EXACTLY (σ = 0 ⇒ ca = 1, cb unchanged), so F-M1a/grad/vjp regressions bind.
- New terminated steppers (separate builders; the verified PEC step body is not edited):
  two-region + PML strips, and uniform + PML (used for references, all-fine, all-coarse,
  floor arm). Jy column source and column-mean Ey probe live only in these.
- Wiring falsifiers (pytest, windows frozen now):
  * r = 1 island with LOSSY nonuniform maps (host ≡ island material) must equal the
    uniform lossy stepper to ≤ 1e-12 rel — validates ε̂/σ̂ wiring: at r = 1 eq. (61)
    reduces algebraically to the standard lossy Yee coefficient.
  * Sec. V-B-class traverse: a lossy slab (εr = 2, σ = 5 S/m — the paper's V-B dielectric)
    CROSSING the south interface, PEC box: after source-off the staggered storage (25)
    must be non-increasing step-to-step within +1e-13·E_ref roundoff slack (dissipativity
    with the σ̂ terms active), and finite.
  * Vacuum-maps-vs-default equivalence: explicit vacuum/lossless maps reproduce the
    default stepper trace to ≤ 1e-14 rel (coefficient wiring is exact, not approximate).

## 5. Retry falsifiers (windows FROZEN NOW; provenance = §1 extraction + ≥5 dB rule)

Burned-data compliance: every number below derives from the paper-figure extraction
(committed instrument + JSON, §1) or from first-principles derivations (§3) — none from
phase-1 measurements. Phase-1's measured plateau (−43.7..−45.9 dB) was NOT consulted in
choosing windows; the ≥5 dB rule and anchor set were declared in phase 1 before any
measurement existed.

### F-M1b-r2 (primary): interface-only reflection, paper-faithful fixture
For EVERY r ∈ {2,3,4,5,6}, on the §2 fixture with no rods:
  max|S11| over [2,20] GHz ≤ **−46.24 dB** AND max|S11| over [2,30] GHz ≤ **−29.29 dB**
(worst extracted curve + 5.0 dB exactly). Exceeding either at any r → FIRE.
Verdict semantics (routing task 3): PASS ⇒ M1 recorded complete (the phase-1 FIRE stands
as a fixture-class finding, now superseded by the faithful fixture). FIRE ⇒ STOP: the
fixture excuse is removed, the discrepancy is scheme-vs-paper (our eq. (58)/(61)
implementation or the scheme itself) — characterize, no further window motion.

### F-M1b-rod (secondary): with-scatterer accuracy cross-check
Subgrid r = 6 run with the four copper rods vs OUR all-fine (r = 6 uniform, same dt)
reference pair, both |S11| via §2's chain:
  max over f ∈ [2,30] GHz of | |S11_subgrid(f)| − |S11_allfine(f)| | (LINEAR) ≤ **0.0941**
(paper's extracted r=6-vs-all-fine class 0.0529 + 5 dB, from the committed JSON; linear
metric because dB differences explode on resonance-dip flanks for immaterial frequency
shifts). Exceeding → FIRE (accuracy-with-materials defect).
Context arms (recorded, non-falsifier): r ∈ {2,4} subgrid mismatch, all-coarse mismatch
(paper class 0.1563 — ours should be the same order, and larger than the subgrid runs').

### F-M1b-abc (gate, must pass before r2/rod are judged): §3 floor window.

### Regression (unchanged windows, re-run after implementation):
Full portgrid battery green; F-M1a 10⁶-step arms r=4/r=5 stay ≤ +1e-8 (expected roundoff
class ~1e-15 as measured in phase 1); F-M1-grad ≤ 1e-6; F-M1-vjp ≤ 1e-12.

### Runtime rule
Every arm is CPU-sized (largest: all-fine 396×240, ~15k steps). If any arm's pilot
extrapolates past 20 min CPU, emit a VESSL yaml instead and report partial_gpu_pending;
windows unchanged.

## 6. M2 checklist pre-seed (2022 Corrections, IEEE TAP 70(4):3132 — expanded from
`portgrid_m0m1_predeclaration.md` §4 for the M2 lane to copy into its pre-declaration)

The Corrections replace the 3D paper's STRONG equalization of fine hanging variables with
weaker averaged/signed conditions and re-derive the supply-rate cancellation. Concretely
for the M2 implementation (arXiv:1705.02274 numbering unless tagged [Corr]):

- [ ] C1. Planar hanging-variable interpolation must impose only the AVERAGED condition
      [Corr eq. (1)] on each coarse face's r² fine H samples — NOT the per-sample
      equalities (arXiv (39a)-(39c) / TAP (43a)-(43c)). Implementation: face-restriction
      operator = (1/r²)·1ᵀ (P-weighted mean), replication stays per-sample.
- [ ] C2. Edge (line) hanging variables: use the signed-circulation conditions
      [Corr eqs. (3)-(5)] in place of TAP (61a)-(61c)/(67)/(68) equalities; signs follow
      the outward-normal convention per face pair — enumerate all 12 edge orientations in
      tests (no sign tuning; derive once, test each).
- [ ] C3. Corner variables: the Corrections leave corners to the edge-rule composition —
      implement by composing C2 rules and VERIFY numerically (energy audit on a fixture
      where three faces meet, SPEC-02 M2 arm ⓒ) rather than trusting the composed proof.
- [ ] C4. Everything else of the 3D paper is unchanged ("All the other equations,
      theorems and numerical results in [1] are correct as stated") — in particular the
      region matrices, the odd-r restriction, and eq.-class (58)/(61) material averages.
- [ ] C5. Because the original proof needed correcting, EVERY M2 stability claim carries
      a ≥10⁶-step numerical energy audit (SPEC-02 §6 discipline; M2 arms ⓐⓑⓒ) — the
      certificate calculator must be extended to the corrected interconnect conditions
      and cross-checked against the audit, not the other way around.
- [ ] C6. The 2D pieces (this lane) are NOT affected by the Corrections (2D has a single
      hanging-H average per coarse edge — no equalization to weaken); re-confirmed against
      the re-downloaded paper this retry.

## 7. Do-not-repeat compliance (unchanged)

Single global dt = 0.99×fine CFL; no SBP derivation; no Huygens/filter stabilization;
unmodified Yee inside regions; the PML is measurement-fixture termination (as in the
paper), not a stabilization device, and its floor is measured before use (§3).
