# Issue #770 — adjudication record, verdict, and the whole-port receive channel

Companion to `issue770_offdiag_adjudication_predeclaration.md` (the binding
pre-declaration, committed BEFORE the harness existed, unmodified since).
Harness: `validation/research/issue770_offdiag_adjudication.py`.  All runs
2026-08-29/30, branch `agent/issue-770-offdiag`, CPU JAX, float32,
JAX_PLATFORMS=cpu, preflight ON and quoted verbatim, one run per arm, no
tuning.

## 1. Fixture sanity (G0) and recorded deviation

FIX-T built on both lanes from the verbatim battery constants.  Uniform
lane preflight advisory set matched the pinned battery set exactly
(`pec_faces_finite_pec` + 2x `wire_port_dead_extent_cells`, n_live = 2 per
port).  RECORDED DEVIATION (lane advisory, not geometry drift): the NU
realization substitutes the two dead-cell advisories with the #544
lane-specific `wire_port_dead_cell_classification_unavailable` pair (the
shared classification primitive covers only the uniform grid) plus the
uncoded shard_map deprecation notice; the runner's own live-cell fold
still measured n_live = 2 per port (asserted in-harness), and the F-A5
lane parity result (3.97e-6) independently confirms both lanes realized
the same electrical fixture.

F-A6 wiring identities (all PASS): NU shipped diagonal reproduced from
raw accumulators to 3.1e-8; uniform bundle S reproduced through
`decompose_wire_s_matrix` to 0.0; frame-P off-diagonals reproduced from
bundle raw channels to 7.6e-8 (gate 1e-5 each).

## 2. Falsifier verdicts (gates verbatim from the pre-declaration)

Frequencies: 9 bins, 3–7 GHz.  S_jj = the #764 physical diagonal
(measured |S11| 0.0093–0.2896, |S22| 0.0176–0.2910 — the #683-run class
reproduced).

- **F-A1 passivity ceiling (1.02):**
  frame W column power 0.9564–0.9912 — PASS both drives.
  frame P column power 0.3334–0.4118 — PASS (trivially, deflated).
- **F-A2 power-closure adjudication (T = |S_ij|²/(1−|S_jj|²) in
  [0.90, 1.02]):**
  frame W: T = 0.9524–0.9912 (drive 0) / 0.9535–0.9912 (drive 1) — PASS.
  frame P: T = 0.3121–0.3718 / 0.3187–0.3700 — **FAIL, both drives, all
  bins** — the pre-registered separation (predicted class 0.30–0.41)
  measured exactly; the per-cell frame is REFUTED as magnitude physics.
- **F-A3 reciprocity (abs 1.5e-2 OR rel 0.10):**
  frame W: max|S21−S12| = 2.6675e-4, rel 2.78e-4 — PASS, and the #770
  item-2 question is answered: the residual SHRINKS 28x from the locked
  7.5277e-3 class (the per-cell residual was the PRE-referenced a_j's
  per-column asymmetry, not fixture physics).
  frame P: 7.5277e-3 (rel 1.3711e-2) — PASS (the locked class, to the
  last digit — fixture identity witness).
- **F-A4 DC/phase anchor (frame W):** pinned global receive sign s = +1;
  |S21^W| = 0.9999/0.9997 at 0.5/1.0 GHz; dev = −0.0575/−0.1152 rad
  inside (−0.25, +0.10); flipped channel +3.0841/+3.0264 rad (leaves the
  band) — PASS with pi-discrimination.  The DC thru limit now holds in
  magnitude AND phase.
- **F-A5 lane parity (frame W, 1e-3):** max|S^W_NU − S^W_uniform| =
  3.969e-6 over all four entries — PASS.

Measured frame-W absolute transmission (the #770 item-3 witness):
|S21| = 0.9341–0.9954, monotone, against the external anchors
flux-implied 0.971–0.997 / openEMS 0.973–1.034 with the feed-post
reflection separated the #764 way (at 7 GHz |S11| = 0.29 caps
|S21| ≤ 0.957; measured 0.9341 with the 0.9–4.4% closure deficit
matching the 0.2–4.0% flux closure gap).  Measured frame ratio
|S^W/S^P| = 1.63–1.75 per bin — inside the #313 κ(f) = 1.49–1.86 class,
closing the κ decomposition: κ = √n_live x (calibrated-reference /
physical-incident ratio), exactly as §2(iv) of the pre-declaration
derived.

For the record, the pre-#770 NU shipped mixed `_ab` off-diagonal
measured |S21| 0.6533–0.9684 (drifting, neither frame) — the #764 §6
open defect, now retired.

## 3. Verdict (rule 2 of the pre-declaration, applied as committed)

Frame P fails F-A2; frame W passes F-A1–F-A5 → **outcome (b): the frame
error is confirmed and localized to the off-diagonal wave pair's frame
(both the per-cell receive normalization and the PRE-referenced
calibrated incident wave), and the whole-port receive channel is
implemented** with the §5 scope of the pre-declaration:

- `decompose_wire_s_matrix`: with `v_port` provided, off-diagonals are
  the whole-port pair `b_i/a_j = (V_port,i − Z0·I_i)/(V_port,j +
  Z0·I_j)` (x √(Z0_j/Z0_i)); `v_port=None` keeps the FULL legacy
  per-cell decomposition byte-for-byte (#313 refplane decomposer and
  pre-#770 dump replay — `v_ref` is consumed only there).
- NU lane (`rfx/nonuniform.py`): every genuinely excited column is the
  same whole-port pair; the all-passive diagnostic fallback keeps the
  frozen per-cell `_ab` verbatim.
- Global receive sign +1, pinned by the F-A4 DC witness (the #308
  witness class); the mixed-component ±1 fence is unchanged.
- Replay schema: the wire dump metadata gains `"offdiag_frame":
  "wholeport"`; a dump without the tag replays the per-cell frame
  (pre-#770), mirroring the #764 `raw_port_voltages_fdt=None` marker
  precedent.  The replay diagnostic refuses `wholeport` without the
  `raw_port_voltages_fdt` channel.

## 4. Lock moves (all inside the pre-declared mover classes, each with
## in-file provenance)

Post-implementation shipped-path provenance run
(`--battery-provenance`): |S21| = 0.9341–0.9954, |S12| = 0.9342–0.9955,
phase dev −0.3516..−0.8125 rad, reciprocity 2.6678e-4 (rel 2.78e-4),
per-bin sv 0.9874–1.003227, column power 0.9563–0.9908, DC anchor
−0.0575/−0.1152 rad with |S21| = 0.9999/0.9997.

1. THRU battery `_THRU_S21_BAND` (0.35, 0.85) → (0.90, 1.001): edges
   physics-derived (external-anchor closure window / passivity bound +
   float headroom), measured 0.9341–0.9954.  The old band was the #313
   regression lock whose own docstring prescribed exactly this
   re-baseline ("when the kappa item lands ... re-baseline in the same
   PR").
2. `_THRU_MAX_SINGULAR_VALUE` 0.85 → 1.01: measured 1.003227 — the
   matrix is now near-unitary; the 0.32% excess over the physical bound
   is complex64-extraction noise (the repo's 1.02 column-power ceiling
   class).  A strictly-below-1 gate cannot bind a physically ~unity
   singular value; 0.85 was bindable only against the deflated frame.
3. Phase-dev band (−1.1, −0.1) KEPT (measured −0.3516..−0.8125, margins
   0.29/0.25); reciprocity gates 1.5e-2/0.10 KEPT (measured 2.67e-4);
   DC-anchor band (−0.25, +0.10) KEPT (measured −0.0575/−0.1152).
   Measured quotes updated in-file; per-cell history preserved verbatim
   as legacy-frame history.
4. `test_twoport_wire_port.py::test_two_port_s_envelope_on_matched_line`
   — re-pinned per its own in-file instruction: envelope 4.84444 (the
   known-wrong mixed column) → passivity ceiling 1.0 + 1e-2 with alive
   floor 0.5; measured max column power 0.98453 (min 0.94379), |S21|
   0.64631–0.98558, |S11| 0.11471–0.72794 (diagonal untouched —
   unchanged from #764 to 5 decimals).  Independent confirmation on a
   graded NU mesh at n_live = 4.
5. Dump/replay schema re-pins: `offdiag_frame` metadata tag in
   `tests/test_sparam_driver_dump_parity.py`,
   `tests/test_port_dump_replay.py`,
   `scripts/diagnostics/generate_wire_port_vi_dump.py`,
   `scripts/diagnostics/report_wire_replay_sweep.py`; replay diagnostic
   frame dispatch in `scripts/diagnostics/replay_wire_port_vi_dump.py`.
6. NOT moved: every diagonal gate (incl. the #683 thru-floor strict
   xfail, which stays FIRED-and-held), the refplane byte-frozen pins
   (module re-run green — the legacy path is byte-frozen through this
   change), the all-passive NU locks, lumped ports, MSL/waveguide.

## 5. Battery results (module-level, post-implementation)

- THRU battery + twoport module (`-m "not gpu and (slow_physics or not
  slow_physics)"` — the slow_physics THRU locks RAN): 11 passed,
  1 xfailed (the #683 thru-floor gate 5, still FIRED-and-held; the #770
  frame change does not touch the diagonal it fired on).
- Dump-parity + port-dump-replay + refplane modules: 44 passed (the
  refplane byte-frozen legacy pins held through the frame change; the
  wholeport-tagged dumps replay faithfully; untagged synthetic dumps
  replay the per-cell frame).
- `test_extract_lumped_s11_is_the_decompose_diagonal` +
  `test_sparam_driver_matches_eager` (driver-vs-eager, wire included):
  6 passed — the eager and driver paths moved together through the
  shared decomposer, atol 2e-3 held.
- Example-fidelity contract (discovery + machine-check + snapshot keys
  for the three newly classified research harnesses): 56 passed; the
  single failure is the PRE-EXISTING
  `test_not_auditable_classifications_are_machine_checked[issue764...]`
  recorded in the #683 P6 addendum (base-branch item, untouched here).

## 6. Full-sweep record (appended after the sweep of 2026-08-30;
## sections 1–5 unchanged)

Full battery, `-k "wire or lumped or sparam or twoport"`,
`-m "not gpu and (slow_physics or not slow_physics)"` (marker override
active — the slow_physics THRU battery ran, not skipped):
**326 passed, 7 failed, 4 skipped, 1 xfailed** (13:43 CPU) — the same
counts and the SAME seven failures as the #683 P6 baseline on the
unchanged base (six `test_example_matches_snapshot` grid-realization
snapshot drifts: ports_and_sparams_101 x5 + lumped_port_gradient_check;
plus `test_not_auditable_classifications_are_machine_checked
[issue764_wireport_norm_falsifiers.py]`).  Zero new movers.  The
1 xfailed is the held #683 thru-floor gate.

Recorded deviation: three doc-only commits (fidelity-table
classification + surgical snapshot keys, CHANGELOG entry, stale-comment
retirements) landed while the sweep was running; none changes runtime
behaviour, and the affected fidelity tests were additionally run
standalone on the final tree (the 56-passed line in section 5).

The example-fidelity classification additions surfaced a base-branch
omission: the two #683 research harnesses had no classification entry,
so `test_discovery_matches_classification_table` (outside the sweep's
`-k` selection) was failing on the base.  Fixed here by classifying
them (`issue683_sampling_order_decision` audited with its separable
`build()`; `issue683_flip_acceptance` no_simulation — it imports that
builder) and extending the snapshot additively (581 lines; every
pre-existing key byte-untouched).  The pre-existing #764 harness
misclassification failure is left exactly as the #683 P6 addendum
recorded it.

Final standing: the whole-port receive channel is validated end to end
by the pre-declared external falsifiers and locked at its measured
physical values; the per-cell frame survives only as the byte-frozen
legacy/replay path; no VESSL run was required (every decisive
measurement fit the local budget).
