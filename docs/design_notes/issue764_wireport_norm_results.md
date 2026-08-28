# Issue #764 — measurement record, verdicts, and deviations

Companion to `issue764_wireport_norm_predeclaration.md` (the binding
pre-declaration, committed BEFORE any measurement, unmodified since).
This note records what was measured, every falsifier verdict, and every
deviation from the pre-declaration with its provenance. Harness:
`validation/research/issue764_wireport_norm_falsifiers.py`.

All NU-lane runs: worktree @ agent/issue-764-wireport-norm, CPU JAX,
float32, JAX_PLATFORMS=cpu, preflight ON.

## 1. Fixture revision (G0 FIXTURE INVALID on the first build — not a
falsifier verdict)

The pre-declared FIX-A used one-cell-thick (0.5 mm) electrode plates.
Measured on the first run (short fixture, per-cell samplers): the
driven column's Ez at the bottom-plate interior cell (k=6) was LIVE and
carried `V(k6) = -(V7+V8)` — `apply_pec_mask`'s thin-sheet rule
preserves the normal E of a 1-cell-thick PEC (surface charge), so the
bottom plate conducted at z=3.0 mm, not the intended z=3.5 mm gap face.
The preflight sheet-cavity advisory reported the same fact (+50%
electrical gap, "geometry[0]'s OWN cell" live). This fails the
pre-declared G0 sanity clause ("the plates rasterize PEC at the intended
k-planes" — electrically they did not), which the pre-declaration
explicitly classifies as FIXTURE INVALID, distinct from a falsifier
verdict. Repair: plates thickened to 1 mm (>= 2 cells at both FIX-A and
FIX-A' resolutions: z in [2.5, 3.5] and [4.5, 5.5] mm), which zeroes the
interior normal edges and realizes the declared terminals. No gate was
touched. (First-build readings, for the record: matched |S11| ~ 0.080,
short ~ 0.667 — both explained by the live series plate-interior layer.)

## 2. Falsifier verdicts (repaired FIX-A, gates verbatim)

- G0 fixture sanity: PASS (n_live = 2 every port, preflight ON).
- Extraction wiring (production `S[k,k]` vs harness formula on the raw
  accumulators): max delta 9.3e-9. PASS.
- F0 a-invariance across {matched, short, open}: per-bin relative
  spread [0.0015, 0.0037, 0.0074, 0.0142] vs gate 0.05. PASS.
- F1 matched: |S11| = [0.0010, 0.0026, 0.0055, 0.0125] (gates <0.05
  everywhere, <0.02 at <=1 GHz). PASS. (Pre-#764 defect value: +0.35426.)
- F2 PEC short: |S11| = [0.9998, 0.9989, 0.9958, 0.9844];
  |arg-180| = [1.4, 3.4, 6.7, 13.2] deg (gates 10/25); Re < 0 at every
  bin. PASS. (Pre-#764 defect value: +0.26780.)
- F3 open: |S11| = [1.0000, 0.9999, 0.9996, 0.9982]; arg(lowest bin)
  -1.5 deg. PASS.
- F4 passivity order-of-operations: Re(+V_port/I) >= 0 on every passive
  load at every in-band bin (min 0.004 on the short); no |S11| > 1.02
  anywhere. PASS.
- F5 #683 circuit law + load law: per-load |I| within 0.9% of
  V_src/(Z0+Z_L) (gate 5%); S11-vs-(Z_L-50)/(Z_L+50) regression slope
  0.9996, intercept -0.0000 (gates [0.9, 1.1], 0.05). PASS.
- F6 n_live invariance (dx 0.5 -> 0.25 mm, n_live 2 -> 4): per-bin
  |S11| move [0.0004, 0.0009, 0.0016, 0.0022] vs gate 0.05; the defect
  frame predicted a ~0.267 jump. PASS.
- F7 power bookkeeping: wiring identity max residual 3.2e-27 (gate
  2.3e-16); short 1-|S11|^2 = [0.0003, 0.0021, 0.0084, 0.031] vs gate
  0.10. PASS. F7b flux-box referee: NOT-RUN — the NU lane implements
  only full-plane flux monitors (`runners/nonuniform.py` raises
  NotImplementedError for finite-region `add_flux_monitor(size=...)`),
  so the pre-declared closed box around the driven column is not
  expressible; per the pre-declaration the sub-check is reported NOT-RUN
  with the plumbing gap named, not re-aimed.
- F8 KVL witness: **FIRED AS WRITTEN.** |V_port|/|V_mid| = 2.40 at the
  quasi-static bins vs gate < 0.1. Mechanism (named, gate NOT widened):
  the criterion's premise was that a short forces sum(V_c) -> 0 while
  V_mid stays finite — the ledger fixture class, where the shorting PEC
  intersects the port's own extent cells. On FIX-A's clean EXTERNAL
  short the per-cell relation V_c = (Z0/n)(I0 - I) makes every live-cell
  voltage collapse together with the sum (measured V7/V8 = 1.40), so the
  ratio tends to ~n_live regardless of how well KVL holds; the criterion
  cannot read below ~n_live on this fixture class for ANY correct
  extractor. The physical claim F8 guards — the SUM being the
  KVL-constrained gap voltage — is witnessed independently by the
  wave-scale collapse |V_port|/(Z0|I|) = 1.2e-2 / 3.0e-2 at the
  quasi-static bins (reported, not gated) and by F2's S11 -> -1.
  Verdict stands recorded as a miss with this mechanism; it is a
  criterion-derivation error against this fixture class, not evidence
  against the whole-port V definition.
- F9 current-uniformity premise: per-bin live-cell |I| spread
  [0.0001, 0.0004, 0.0016, 0.0062] vs gate 0.05. PASS.
- F10 extraction-only guard: all six field components bit-identical to
  main @ b29f9de after 400 steps on FIX-A matched. PASS.
- F11 frozen-channel guard: lane-parity closed forms, sigma ORACLE 1/2,
  off-diagonal locks — green (see section 4 for the two mechanical
  edits that are NOT lock moves).

FIX-C thru (reported prediction, never a gate): |S11| = [0.034, 0.086,
0.168, 0.316] at [0.2, 0.5, 1, 2] GHz. The two quasi-static bins sit
inside the measured 0.033-0.086 class. The rise at 1-2 GHz is fixture
physics of THIS FIX-C realization: the 1 mm x 2 mm parallel-plate line
has Zc ~ 377*d/w ~ 188 ohm against 50-ohm terminations (recomputed
model: |Gamma| ~ sin(beta*l)*(Zc/Z0 - Z0/Zc)/2 ~ 0.33 at 2 GHz,
matching the measured 0.316), unlike the #318 canonical w/h=5 microstrip
(~50 ohm) the 0.033-0.086 class was measured on. Reported with the
recomputed model per the pre-declaration; not folded into any gate.

## 3. Harness deviations from the pre-declaration (both with measured
provenance; no gate touched)

1. **F5a units conversion.** The pre-declaration bound What_cell to "the
   rect-DFT of the captured per-cell table", calling the table "a
   CURRENT (amperes) per make_current_source". Measured: the table is in
   E-add units — the actual per-cell injected current implied by the
   scan's own update is
   `I0_eff(t) = -table(t) * A * eps * (1 + loss)/dt` with
   `loss = sigma_port*dt/(2 eps)` (the port cell's update coefficient
   carries the FOLDED sigma; make_current_source normalized with
   sigma=0). Verified against the pinned per-cell discrete law
   `I_loop = -(G+jwC)V + I0` (exact: I0_eff identical at both live cells
   to <0.1%, drive-only constant — the (*) invariance). The conversion
   uses only the captured table, grid metrics and folded sigma — still
   independent of the measured V/I, so F5a remains non-circular. Side
   observation for a separate issue: the injection normalization is off
   by (1+loss)/d_par from its own "amperes" docstring; drive-amplitude
   only, cancels in S-parameters.
2. **F9 realization.** Sampling-only wire-port spec entries are injected
   via the `run_nonuniform` spy exactly as pre-declared; with n_live = 2
   on FIX-A "first/mid/last" degenerates to the two live cells (both
   sampled).

## 4. Lock moves (pre-declared list) and mechanical edits

Moved WITH written provenance (each carries it in-file):
- `tests/test_twoport_wire_port.py::test_two_port_s_envelope_on_matched_line`
  — envelope 28.511 -> 4.84444; the +0.35426 / +0.26780 witnesses
  rewritten into historical provenance (|S11| now [0.115, 0.728],
  load-tracking; |S21| class unchanged — #308/#318 off-diagonal defect
  untouched).
- `tests/test_nu_wire_port_lane_parity.py::test_excited_port_lane_ordering_disagreement_is_open_683`
  — residuals re-measured: vacuum 1.983e-01 -> 4.975e-02, pec_plates
  6.109e-01 -> 1.051e-01 (normalization half removed; #683 ordering half
  remains, near-conjugate signature preserved). Stays xfail.
- `tests/test_refplane_port_waves.py::test_run_short_diagonals_byte_frozen_offdiagonals_move`
  — the default-path and refplane-path diagonals now intentionally
  DIFFER (default = #764 whole-port driven; refplane = byte-frozen #313
  legacy); the byte-equality pin became a byte-inequality pin so a
  silent "unification" reds loudly (open question 4 of the design).
- `tests/test_sparam_driver_dump_parity.py` +
  `tests/test_port_dump_replay.py` — dump schema grows
  `raw_port_voltages_fdt` (None marks a pre-#764 dump; replay falls back
  to the legacy diagonal for those).

Mechanical edit, NOT a lock move:
- `tests/test_nu_wire_port_lane_parity.py::test_raw_port_ratio_matches_the_analytic_admittance`
  — unpack widened (`accs[0], accs[1]`); every asserted value
  byte-identical.

## 5. Uniform lane: what is keyed to the #683 flip

The whole-port driven-diagonal FORMULA lands on the uniform lane
(runners/uniform.py fast path, api/_execute.py forward extraction,
sparam_driver + decompose_wire_s_matrix v_port channel), but the lane
samples PRE-injection (issue #72 contract), which contaminates the
driven V at order 1 (sigma*dt/eps ~ 0.96 on the canonical cell). Per the
adjudicated design and the 2026-08-29 adversarial review of the #683
uniform-flip attempt:
- NO uniform-lane measurement before the flip can falsify or validate
  this design; the F0-F9 gates re-run on the uniform lane only after the
  flip lands.
- The decomposer's sampling-order contract is documented in
  `decompose_wire_s_matrix` (PRE for every quantity it consumes on the
  uniform lane; the NU lane does not route through it).
- The passive/legacy diagonal (excite=False, and the all-passive NU
  fallback) is byte-frozen; the #313 refplane decomposer diagonals are
  byte-frozen; the vinc channel and `extract_s11_normalised` are
  untouched.
- Uniform-lane driven-diagonal value locks re-pinned in this branch
  (the slow_physics THRU battery diagonals, if moved) are regression
  locks on the PRE-ordered interim values, not validated physics, until
  the #683 flip lands — each carries that label in-file.

## 6. Open items forwarded

- The NU multi-port off-diagonal defect (full Z0 in `_ab` for the
  receive port) remains open and out of scope; the S-matrix is now
  mixed-frame (whole-port driven diagonal, per-cell off-diagonal) and
  loudly documented at the extraction sites.
- The wire-port injection normalization deviation ((1+loss)/d_par from
  its "amperes" docstring, section 3) deserves its own issue; it cancels
  in S-parameters.
- F7b's finite-region NU flux monitors remain unimplemented (#544-class
  plumbing gap).
- L_short of the built FIX-A short: committed geometric estimate 0.1 nH;
  the measured F2 phase walk (13.2 deg at 2 GHz) corresponds to
  L_short ~ 0.18 nH via 2*atan(wL/Z0) — inside the <= 0.25 nH envelope
  the phase gates were derived for. No re-derivation needed.
