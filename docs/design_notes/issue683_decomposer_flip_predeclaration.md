# Issue #683 x #764: the uniform-lane POST flip DECIDED TOGETHER with the wire decomposer recalibration — derivation + pre-declaration

Status: DERIVATION AND PRE-DECLARATION, written and committed BEFORE the
implementation and BEFORE any acceptance measurement is run.  Branch
`agent/issue-683-decomposer-flip`, based on `agent/issue-764-wireport-norm`
(bc88f1c).  Falsifiers in §5 are binding and are never widened.

Prior measured facts consumed here (NOT re-derived, NOT re-measured):

- The #683 verdict (`issue683_sampling_order_decision_protocol.md` §9,
  2026-08-29, one run): POST-injection V/I sampling is the
  terminal-consistent, physically correct wire-port sampling order at an
  excited port (n·a = +0.9987/+0.9950, n·|b| = 0.08/0.32 Ω against the
  known-load circuit law; Ampère-identity residual 2.3e-7 vs 3.25).
  PRE-injection sampling is refuted at an excited port (slope −0.62,
  intercept −81 Ω; its E is not any field time level of the update).
- Gate G2 of the same run: on bit-identical geometry the PRE and POST
  samples differ EXACTLY by the same-step injection increment at the
  sampled cell — `V_pre(t_n) = V_post(t_n) + d_par·W(t_n)` with `W` the
  per-step injected ΔE (measured `G_PRE − G_POST = d_par` to rel err
  ≤ 0.001).  This identity is exact by construction of the update: the
  soft-source loop is the last write to E in the step, so the pre-slot
  field is `E^{n+1} − W^n` at driven cells and `E^{n+1}` everywhere else.
- The raw-flip attempt (branch `agent/issue-683-uniform-flip`, adversarial
  review recorded in the `decompose_wire_s_matrix` docstring on the #764
  base): flipping the samples WITHOUT touching the decomposer breaks the
  two-port THRU locks catastrophically — |S21 − S12| max 76.79 vs gate
  0.015, thru diagonal 2.93, receive phase ~π.
- #764 (this base): the driven diagonal is the whole-port reflection
  `S_jj = (V_port − Z0·I)/(V_port + Z0·I)` on BOTH lanes, with six uniform
  physical gates keyed to the #683 flip and in-file restore instructions.

## 1. The step algebra (exact, no new measurement)

Per timestep the scan does: E-update + PEC → [old wire DFT slot] →
soft-source loop (last E write) → [TFSF aux only].  With `v = −E_mid·dx`
sampled at the driven port's midpoint cell:

    v_pre^(n)  = v_post^(n) + d_par·W^(n)         (G2, exact)
    W^(n)      = cb_mid · v_src(t_n)/(n_live·d_par)   (apply_wire_port)

Both slots stamp the same `t = n·dt`, so the identity carries to the DFT
accumulators verbatim: `V̂_pre = V̂_post + cb_mid·V̂inc`.  At every
PASSIVE port (every receive port `i ≠ j` of drive run `j`) the injection
writes no cell of the port, and I reads H only, so pre/post samples are
bit-identical there.  The ONLY sampled quantities the flip moves are the
driven port's own `v[j,j]` and `v_port[j,j]`.

## 2. Why the raw flip detonated the THRU battery (mechanism, quantitative)

The #683 circuit law fixes the driven POST samples: per cell
`v_post = +Z_L·i/n_live ≡ Z_Lc·i` (driven-branch sense, #764 provenance).
The old incident-wave formula applied to POST samples is then

    a_raw = (−v_post + Z0c·i)/(2√Z0c) = (Z0c − Z_Lc)·i/(2√Z0c)

which STRUCTURALLY CANCELS at a matched drive — the exact mirror of the
#308 receive-channel cancellation, now at the drive port.  On the thru
(|Γ| ~ 0.09 class) the denominator survives only at the −Γ_cell scale, so
every off-diagonal inflates by ~1/0.09 ≈ 11× with a sign flip — which is
precisely the measured wreckage (|S21 − S12| 76.79, receive phase ~π).
The raw flip was not "slightly off"; it divided by a residual.

## 3. What the decomposition must become (the derivation)

Write the calibrated PRE-frame drive reference as `v_ref ≡ v_pre[j,j]`.
The three requirements and what each forces:

(i) **Driven diagonal stays Γ_L-exact.**  The #764 formula
    `S_jj = (V_port − Z0·I)/(V_port + Z0·I)` is already the correct
    whole-port terminal reflection — the #683 harness validated exactly
    this pair (`V_port = +Z_L·I`, `I = V_src/(Z0+Z_L)`) ON POST SAMPLES.
    Therefore: formula unchanged, samples flip to POST.  No other change
    can achieve (i): on PRE samples the pair is not a terminal pair at all
    (slope −0.62, intercept −81 Ω).

(ii) **Receive channel keeps its #313-calibrated magnitude.**
    The receive numerator `b_i = (v − Z0c·i)/(2√Z0c)` at passive port `i`
    is slot-invariant (§1) — its #308 sign pin (S21(DC) → +1, 2026-07-10)
    and its #313-documented deflated magnitude remain valid measurements
    with no change.  The DENOMINATOR is the part the flip touches.  The
    #308 receive-wave sign and the per-cell `Z0c = Z0/n_cells`
    normalization were CALIBRATED against the PRE-injection drive sample
    `a_cal = (−v_ref + Z0c·i)/(2√Z0c)`.  The physically "clean"
    alternative — re-referencing to the true incident wave
    `(v_post + Z0c·i)/(2√Z0c)` (= the per-cell drive-only constant
    `Z0c·i0`) — would rescale every off-diagonal by the fixture-dependent
    factor `(Γ_cell − cb_mid)` per drive column: it moves the
    #313-locked magnitudes by port-material-dependent amounts and moves
    S21 and S12 by DIFFERENT factors on any asymmetric fixture.  That
    renormalization is the #313 flux-referee program (reference planes),
    not a sampling fix, and is REFUSED here.  Therefore: the off-diagonal
    drive reference must remain the PRE-injection drive sample, carried
    as its own accumulator channel `v_ref_dft` (accumulated at the old
    slot, bit-identical to the historical `v_dft` drive diagonal by
    construction), while `v_dft`/`i_dft`/`v_port_dft` flip to POST.

(iii) **Reciprocity and passivity hold on the THRU battery.**  Follows
    from (i)+(ii): off-diagonals are preserved bit-near-exactly (so
    reciprocity stays in the measured 7.53e-3 class), and the diagonal
    becomes the measured-physical Γ_L (|Γ| ≤ 1 for every passive load —
    the half-plane argument now applied to a terminal-consistent pair,
    same as the NU lane's validated |S11| ≤ 1.02).

The same reference (`v_ref`) also feeds the byte-frozen LEGACY diagonal
`z_in = −v_ref/i` (used only by the #313 reference-plane decomposer's
diagonals and by pre-#764 dump replay), keeping those documented-frozen
readings frozen through the flip.

**Sign convention: unchanged.**  **Z0 normalization: unchanged**
(`Z0c = Z0/n_cells` off-diagonal, whole-port Z0 on the diagonal).
**Drive-sample reference: new** — the only recalibration the flip needs.

## 4. Why the NU lane needs NO change

- It already samples POST (the measured-correct slot; #683 verdict).
- Its off-diagonal split `a,b = (−v ± Z0·i)/2` uses the WHOLE-port Z0
  against the per-cell midpoint v, so at a driven port on POST samples
  `−v + Z0·i = (Z0 − Z_L/n_live)·i` — Z0-dominated, no structural
  cancellation; its locks were pinned in that frame on POST samples.
- Its diagonal is already the #764 whole-port formula on POST samples.
Only its "#683 OPEN" comment blocks are rewritten to record the verdict.

Scope: WIRE-port family only.  The LUMPED accumulation keeps the #72
PRE contract pending its own decision run (the #683 measurement was made
on wire ports; a ride-along flip is the align-first-decide-later mistake
the ledger's #673/#672 entry warns about).  MSL/waveguide untouched.

## 5. Pre-declared falsifiers (binding; committed before implementation;
##    never widened; any failure outside them = STOP and report)

- **P1 — the six keyed gates are restored to their physical values per
  their own in-file instructions, and pass** (the restoration IS the
  acceptance; no interim envelope survives):
  1. `test_run_forward_s11_contract.py::test_run_s11_is_passive_both_boundaries`
     — xfail removed; `|S11| ≤ 1 + 1e-3`, both boundaries.
  2. `test_wire_sparam.py::test_wire_port_jit_scan_s11_passivity` — xfail
     removed; its passivity assert live.
  3. `test_lumped_wire_sparam_cpml_dielectric.py` — restore
     `max|S11| ≤ 1.0 + 1e-3`.
  4. `test_wire_port_sparams_forward.py` — restore
     `np.all(mag < 1.20)`.
  5. `test_lumped_twoport_vi_validation_battery.py::test_thru_s11_floor` —
     restore the physical floor gate: max in-band |S11|,|S22| < 0.12
     (expected 0.09 feed-post class; exact re-pin value from the
     measurement, provenance recorded).
  6. same module, thru singular-value gate — restore `sv_max < 0.85`
     (expected 0.63 class; exact re-pin from measurement).
- **P2 — off-diagonal preservation (the exactness claim of §3(ii))**: on
  the THRU battery, max over off-diagonal entries and bins of
  |S_flipped − S_base| ≤ 1e-5 (expected bit-identical; 1e-5 is float
  re-association headroom, still 3 orders under the 0.015 reciprocity
  gate).  Reciprocity `max|S21 − S12| ≤ 0.015` with the existing
  in-battery gate, expected to stay in the measured 7.53e-3 class.
- **P3 — physics acceptance**: `validation/research/issue683_flip_acceptance.py`
  (the #683 harness's uniform-lane arm re-run on the flipped lane, same
  §6 rule, one run, no tuning): G1 coupling gates pass; `n·a ∈ [0.90,
  1.10]` and `n·|b| ≤ 10 Ω` at BOTH f1 = 0.05 GHz and f2 = 0.1 GHz.
- **P4 — lane parity at the excited port**: the strict-xfail witness
  `test_excited_port_lane_ordering_disagreement_is_open_683` converts to
  a locked parity test and passes: `max|S_NU − S_uniform| ≤
  LANE_PARITY_ATOL = 1e-4` per bin, both loads (vacuum n_live=4,
  pec_plates n_live=6).
- **P5 — no passive mover**: every passive-port reading and every passive
  lock (`(1−n_live)/(1+n_live)` self-consistency locks, NU envelope
  locks, vinc channel) unchanged.  A passive mover of any size = STOP.
- **P6 — locked-value sweep** (wire/lumped/sparam/twoport battery WITH the
  slow_physics marker override): the only admissible movers are (a) the
  six P1 gates, (b) the P4 witness conversion, (c) DRIVEN uniform-lane
  wire diagonal readings and their docstring quotes, (d) run↔forward
  parity deltas within their existing gates (both paths flip together).
  Any other mover = STOP and report; no re-pin without measured
  provenance inside a pre-declared class.

Interim CPU budget: every local battery ≤ ~20 min (the THRU fixture is
~70 s/run, module-scoped).  No GPU/VESSL run is required by this plan; if
one becomes decisive it is emitted as YAML, not run.
