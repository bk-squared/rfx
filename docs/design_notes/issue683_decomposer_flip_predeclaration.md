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

---

## 6. RESULTS (appended after the measurements of 2026-08-29; sections 1-5 unchanged)

All runs on branch `agent/issue-683-decomposer-flip` (implementation commit
bf78bf9), JAX_PLATFORMS=cpu, marker override active so slow_physics ran.

- **P3 — physics acceptance: PASS.**
  `validation/research/issue683_flip_acceptance.py`, one run: G1 pass
  (monotone, ratio_I 7.19, ratio_Vload 4.45); fits n·a = +0.9990 /
  n·b = +0.066 Ω at f1 and n·a = +0.9960 / n·b = +0.265 Ω at f2.
  Corroborating lane difference at R_L = 50 collapsed from exactly the
  injection increment to max|dV| = 5.6e-6 (float noise).  The same
  known-false-positive preflight advisories as the §9 decision run fired
  (node-attached-column fixture), contradicted by G1 as before.

- **P2 — off-diagonal preservation: PASS, exact.**  THRU battery fixture
  S captured on the base (bc88f1c, scratch worktree, measurement only)
  and on the flipped tip: max over off-diagonal entries and bins
  |S_flipped − S_base| = 0.0 (bit-identical; gate ≤ 1e-5).  Reciprocity
  max|S21 − S12| = 7.5277e-3 on both — the measured 7.53e-3 class,
  unchanged to the last digit.  Base run also reproduced the keyed
  interim diagonal 2.8068 exactly (fixture identity check).

- **P4 — lane parity at the excited port: PASS.**  Witness converted to
  a locked parity test; measured max|S_NU − S_uniform| = 6.839e-8
  (vacuum) / 1.767e-7 (pec_plates) vs gate 1e-4 (pre-flip: 4.975e-2 /
  1.051e-1).

- **P1 — keyed-gate restoration: 5 of 6 PASS; gate 5 FIRED.**
  1. run_s11 passivity both boundaries — RESTORED, green.
  2. jit-scan s11 passivity — RESTORED (xfail removed), green.
  3. cpml-dielectric max|S11| ≤ 1 + 1e-3 — RESTORED, green.
  4. forward PEC cavity mag < 1.20 — RESTORED, green.
  5. **thru diagonal physical floor < 0.12 — FIRED.**  Measured physical
     diagonal on the THRU battery fixture: |S11| 0.0093-0.2896,
     |S22| 0.0176-0.2910 over the 3-7 GHz bins, worst 0.2910 at 7 GHz.
     Diagnosis (measurement, not tuning): driven whole-port
     Z_in(3 GHz) = 49.1 − j0.05 Ω (matched, S11 = 0.009) rising to
     42.9 + j27.1 Ω at 7 GHz; on the symmetric fixture the far port's
     identical +j27 Ω feed-post series reactance alone predicts
     |Γ| ≈ 27/104 = 0.26 — quantitatively the measured 0.27-0.29.  The
     reading is the fixture's true un-de-embedded feed-post reflection;
     the "0.09 feed-post class" expectation was an extrapolation from the
     LEGACY per-cell artifact reading (which measured the frame mismatch,
     not the load) and is REFUTED by the physical channel.  Per the STOP
     discipline the gate was NOT widened: the physical floor assert is
     restored verbatim and held under strict xfail whose reason documents
     the firing (`test_thru_s11_floor`).  Disposition belongs to review.
  6. thru sv_max < 0.85 — RESTORED, green; measured 0.6934 (the
     predicted 0.63 class).

- **P5 — passive movers: none observed.**  Passive parity/closed-form
  locks in test_nu_wire_port_lane_parity.py all green (13 passed
  besides the converted witness); NU lane untouched.

- **Schema follow-through** (declared in §3's implementation paragraph):
  the wire dump/replay round-trip stays green with the new
  `raw_drive_ref_voltages_fdt` channel (test_sparam_driver_dump_parity:
  2 passed; the in-test schema pin re-pinned with written provenance).

- **P6 — locked-value sweep**: recorded in the final commit after the
  full `-k "wire or lumped or sparam or twoport"` battery with the
  slow_physics override.

Standing after these results: the flip + recalibration is validated by
P2/P3/P4/P5 and restores five of the six keyed gates; the sixth
(thru-floor < 0.12 class) is a pre-declared falsifier that FIRED against
a physically-explained measurement and is held, visibly, for review —
a justified STOP outcome for that gate, not a silent re-pin.

## 7. P6 addendum (appended after the full sweep of 2026-08-29; all prior sections unchanged)

Full battery, `-k "wire or lumped or sparam or twoport"`,
`-m "not gpu and (slow_physics or not slow_physics)"` (marker override
active — the slow_physics THRU battery ran, not skipped):
**326 passed, 7 failed, 4 skipped, 1 xfailed** (14:43 CPU).

- The 7 failures are ALL PRE-EXISTING on the unflipped base bc88f1c
  (each re-run there individually, in a measurement-only scratch
  checkout): six `test_example_matches_snapshot` grid-realization
  snapshot drifts (ports_and_sparams_101 x5, lumped_port_gradient_check)
  plus `test_not_auditable_classifications_are_machine_checked
  [issue764_wireport_norm_falsifiers.py]`.  Untouched by this change.
- The 1 xfailed is `test_thru_s11_floor` — the documented FIRED P1
  gate 5 (section 6).
- Two movers surfaced by the sweep were code-following schema
  consequences of the new reference channel, fixed and re-run green,
  not lock moves: (a) the wire dump/replay savez mirrors in
  tests/test_port_dump_replay.py and
  scripts/diagnostics/report_wire_replay_sweep.py gained
  `raw_drive_ref_voltages_fdt` (without it a post-#683 dump replays the
  POST samples as the calibration reference — exactly the raw-flip
  catastrophe in miniature, caught by the replay gate as designed);
  (b) the mixed-lane fill in rfx/api/_sparams.py guards accumulator
  tuples shorter than 5 (the bisecting-mesh path's lane did not flip;
  its vi[0] still IS the pre-injection reference).
- tests/locks/test_refplane_port_waves.py (not matched by the -k selection)
  run explicitly: 32 passed — the #313 reference-plane path's
  byte-frozen diagonals and non-opted off-diagonals held through the
  flip via the same v_ref reference.

Final standing: unchanged from section 6 — the flip + decomposer
recalibration is validated end to end; the single fired falsifier
(thru-floor < 0.12 restore class vs measured physical 0.2910) is held
visibly as a strict xfail for review disposition.
