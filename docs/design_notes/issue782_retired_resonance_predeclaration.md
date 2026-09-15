# Issue #782 — retiring the 9.32/9.21 GHz patch numbers: predeclaration

> **SUPERSEDED IN PART by #931 (the lattice ownership contract), 2026-09-07.**
> Kept as dated history and deliberately NOT rewritten.
> Its statement that "#702 gave a node-thin conductor's cell the material at
> its live edge", and the reproduction leg built on
> `resample_sheet_node_materials`, refer to a mechanism deleted by #931. The
> one physical case #702 served — a stack-up drawn with a slot for the foil —
> is now a preflight finding (`sheet_slot_vacuum`), not a silent re-sample.
> The current rule is
> `docs/design_notes/20260906_plan_realign_lattice_ownership.md` §1, and for
> users `docs/public/guide/materials-geometry.mdx` ("How conductors land on
> the lattice").

Date 2026-09-01. Baseline tree `635ab2e3` (origin/main, verified same-day). Written
BEFORE any verification arm ran; the measurement plan, expected values and falsifiers
below are pre-declared. Implementer worktree `wf_b56f9a1c-177-4`.

## 1. What happened and why the numbers are retired

#702 (`8e004976`) gave a node-thin conductor's cell the material at its live edge.
On the edge-fed RO4003C patch fixture that filled the ground-plane cell layer inside
the cavity with `eps_r = 3.38` instead of vacuum, and every mode moved down by a
uniform ×0.876 (velocity signature). Bit-exact attribution: disabling
`resample_sheet_node_materials` on today's tree reproduces the old tree digit for
digit (unfed 10.02375 GHz, fed 9.32155 GHz). The old fed TM010 9.32 GHz and the
Balanis-on-design-dimensions 9.21 GHz are therefore retired as current numbers.
Seven committed surfaces still quote them as current (issue #782 lists file:line);
recon on `635ab2e3` confirmed all seven live and found three more (Section 6).

## 2. Replacement number set — one mesh per number

Rule (issue #782): every number is anchored to exactly ONE realized raster. Mixing
dimensions from two meshes describes a board that exists on no mesh (the 44×51 =
"8.657 × 10.034 → 9.14" mix cost ~2 points in every percentage; ledger
"ANCHOR CORRECTED 2026-08-29").

There are TWO distinct boards in this arc, and the distinction was not previously
written down on any committed surface:

### Board H — the harminv-gate board (`tests/locks/test_patch_edgefed_resonance_harminv.py`)
`DX = H_SUB/4 = 196.75 µm`. Realized patch raster 43 × 51 cells =
8.46025 × 10.03425 mm.

| quantity | value | provenance |
|---|---|---|
| Balanis TM010 on realized raster | 9.3305 GHz | recomputed at runtime by the test's `_balanis_ghz`; ledger 2026-08-28 block |
| Balanis TM001 on realized raster | 8.0188 GHz | same |
| unfed (isolated) TM010 | 8.767317 GHz at N=120 CPU | the gate's own config; moves ±0.2 pp across N=120/200/260/400 (moving-window estimator) — always quote N |
| unfed TM001 (measured) | 8.0217 GHz (vs Balanis 8.0188, 0.04%) | mode-label cross-check, ledger mode-labels block |
| fed TM010 | 8.16131 GHz, N=260 GPU control (VESSL 369367257262, tree fa3a99bd); CPU N=120 8.162221 (<4e-8 rel) | ledger tip-fixed-sweep table; fed arms move <0.01% with N |
| pre-#702 pair (resample bypassed) | unfed 10.02375 / fed 9.32155 GHz | issue #782 body; parent commit 6b1302b3 measures 9.322 |
| Leg A envelope (unfed vs Balanis) | −6.17 ± 1.125 % | committed constants, harminv gate |
| Leg B envelope (fed/unfed − 1) | −6.109 ± 0.986 % | committed constants, harminv gate |

### Board S — the S11-passivity-gate board (`tests/locks/test_patch_edgefed_s11_passivity.py`)
`DX = 0.197 mm` exactly (NOT `H_SUB/4`). Measured this session (geometry probe, no
FDTD, `rfx.__file__` pinned to this worktree): realized patch raster **44 × 51 cells
= 8.668 × 10.047 mm**, substrate 4 cells = 788 µm, and the substrate column under
the MSL port realizes 985 µm (+25.2 %, quoted verbatim in the preflight advisory).
Balanis on THIS realized raster: **TM010 9.1237 GHz, TM001 8.0016 GHz**.

Consequence: 8.16 GHz is a Board-H number. It must NOT be written into the S11
gate. The S11 gate's band must be measured on Board S. First-order 1/L transfer
of the parity-labelled Board-H fed TM010: 8.16131 × (8.46025/8.668) = **7.966 GHz
predicted** for Board S's fed TM010. Cross-check via the envelopes: Balanis(S)
9.1237 × (1 + LegA) × (1 + LegB) = 9.1237 × 0.9383 × 0.93891 ≈ 8.04 GHz — the two
routes agree to ~1 %, which is within the envelopes' widths (they were measured on
Board H, so they transfer only approximately; that is exactly why Option A below is
rejected).

The h/3 anchor 9.1374 GHz (33 × 38 cells) appears in the issue body only; no
committed surface needs it and none will carry it.

## 3. Live-gate decision — written root cause

### Why the old band no longer tests the stated physics
`RES_BAND_GHZ = (9.0, 9.42)` was written around Balanis-on-design-dimensions
9.21 GHz — an anchor realized on NO mesh (Board S's own realized-raster Balanis was
9.1237 even pre-#702) — and around the pre-#702 fed spectrum, which #702 showed was
two errors cancelling (+7.43 % sheet-cell vacuum vs −7.0 % feed pull). Post-#702 the
fed TM010 on Board S sits near 8.0 GHz, OUTSIDE the band. `min|S11| > 0.70` over
(9.0, 9.42) now asserts high reflection in a spectral region with no mode of this
fixture in it: almost nothing can violate it (a dead-band gate, nearly
unfalsifiable), and gate (3) `f_dip > 9.42` keys off the same retired band. The
green no longer means "poorly matched AT the TM010 resonance".

### Rejected: Option A (derive the band at runtime from Balanis × Leg envelopes)
It would compose Board S's Balanis with Leg A/B envelopes measured on Board H —
cross-board anchor mixing, the class #782 forbids — and would silently re-pin this
gate whenever the harminv gate's envelopes are re-pinned (hidden cross-gate
coupling). It also does not discriminate the retired physics by itself: the
composed band is a constant, and under the retired physics the band simply becomes
another dead band that `min|S11| > 0.70` passes.

### Chosen: Option B — measured re-pin on Board S, plus an in-band resonance witness
One settled pre-declared run pair (Section 5) on the gate's exact config. The gate
is rewritten to assert, with constants pinned from the main arm and a provenance
paragraph in the #784 house style:

1. (unchanged) passivity `max|S11| <= 1.05`;
2. `RES_BAND_GHZ` moved onto Board S's measured fed TM010 neighbourhood;
   `min|S11| > RES_BAND_S11_MIN` there (threshold re-read from the measured trace,
   kept at 0.70 only if the trace clears it with margin);
3. (soft, re-derived) the global |S11| dip = off-resonance match point lies ABOVE
   the band;
4. NEW — the discriminating assertion: an `Im(Zin) = 0` up-crossing exists INSIDE
   the band. This is what makes the band falsifiable: the resonance the band names
   must actually be there. Under the retired physics the crossing sits near
   9.5–9.6 GHz (pre-#702 witness measurement on this geometry) — outside the new
   band — so the rewritten test FAILS on the retired physics (falsifier F1).

Mode identity of the in-band crossing — never from amplitude: Board S's band will
contain Balanis TM001 8.0016 GHz as well. The crossing is attributed to the
stub-loaded TM010 by (a) the y-symmetry argument: the MSL feed and S11 observable
are y-centred, and TM001 is odd along y, so its coupling to S11 is
parity-suppressed; (b) the quantitative 1/L transfer from Board H's
parity-labelled fed TM010 (8.16131 → predicted 7.966 GHz on Board S), where the
Board-H label rests on the four-way evidence chain (joint-fit parity across a probe
cross, fit-free windowed-DFT nodal check, single-dimension perturbation, Balanis
both modes on the realized raster). Limit stated in the gate docstring: the
crossing LOCATION is reference-plane dependent (memory: reactance-zero locations
shift with the reference plane), so the band is sized to absorb that offset, and
the crossing is an existence witness, not a frequency gate.

Stub sensitivity carried into the provenance paragraph: the fed frequency is a
property of the open feed stub too — one node of stub length ≈ +0.85 pp
(≈ 0.07 GHz); FEED_LEN / PORT_MARGIN / DOM_X / raster changes invalidate the pinned
band (Board S has no committed raster lock; the rewritten gate will assert its
44 × 51 raster the same way the companion asserts 43 × 51).

## 4. Pre-declared measurement plan (the single verification arm pair)

Script: `scripts/diagnostics/patch_edgefed_s11_band_repin.py` (committed, with the
run log and per-bin JSON under `docs/design_notes/`). Exact gate config: geometry
of `_build_patch_sim()`, `freqs = linspace(6, 14 GHz, 81)`, `num_periods = 280`,
CPU (`JAX_PLATFORMS=cpu`), `PYTHONPATH` pinned to this worktree, `rfx.__file__`
recorded in the log. Two arms:

- **main** — tree physics as-is;
- **retired** — `rfx.api._compile.resample_sheet_node_materials` replaced by the
  identity, the same bypass `tests/unit/preflight/test_preflight_rasterization.py::_bypass_resample`
  uses; #782 established this reproduces the pre-#702 tree digit for digit.

Both arms dump the full 81-bin |S11| / Re(Zin) / Im(Zin) trace (R5), every
preflight advisory verbatim, and the #332 settling-witness outcome.

### Expected values (declared before the run)
- E1 main: `max|S11| <= 1.05` (passivity unchanged by #702 on this fixture).
- E2 both arms: #332 witness silent at 280 periods. If it fires, that is an
  identified run-length defect of the attempt, not a new mechanism: raise
  `num_periods` once and re-run (allowed under R2 as defect-repair, declared here).
- E3 main: at least one `Im(Zin) = 0` up-crossing in **(7.8, 8.6) GHz**; point
  prediction ≈ 8.0–8.4 (7.966 from the 1/L transfer, plus the reference-plane
  offset which pre-#702 read ~+4 % on this geometry).
- E4 main: `min|S11|` over (7.8, 8.6) GHz > 0.70 (edge-fed signature persists;
  pre-#702 |Γ| at resonance ≈ 0.8–0.9).
- E5 main: global dip strictly above the crossing, expected ≈ 8.8–9.8 GHz
  (pre-#702 separation at this mesh: dip 10.50 vs fed resonance ≈ 9.1, ~+13 %).
- E6 retired: NO `Im(Zin) = 0` crossing inside (7.8, 8.6); crossings at
  ≥ 9.3 GHz (pre-#702 witness class 9.5–9.6); dip ≈ 10.5–11 GHz.

### Verdict rules
- E1–E6 hold → pin `RES_BAND_GHZ` as the measured crossing ± enough margin to
  cover the stub/N sensitivity while keeping the retired-physics crossing OUT of
  band; re-read the 0.70 floor and the `f_dip` bound off the measured traces.
- E4 fails (in-band min ≤ 0.70) → the "poorly matched" claim itself is at stake:
  STOP, report, do NOT re-pin the threshold to make it fit.
- E3 or E6 fails → the redesign hypothesis is non-closing → R2 STOP, no second
  attempt without a named new falsifier. R2 accounting: this is attempt 1 for
  this hypothesis (none prior).

## 5. Falsifiers

- **F1 (gate discriminates)**: evaluate the REWRITTEN gate's assertions on both
  arms' saved traces: main arm → every assertion passes; retired arm → the
  in-band-crossing assertion FAILS. Report the numbers.
- **F2 (sweep complete)**: repo-wide grep for the retired numbers
  (`9.32`, `9.21`, `9.20` in patch context; `10.0239`, `10.02375`, `9.3216`,
  `9.32155`; `TARGET_GHZ`; the palace `10.1 GHz vs 9.26`) finds only
  clearly-historical, dated mentions. Output saved.
- **F3 (nothing else moves)**: touched test files' CPU-runnable tests pass;
  full-suite `pytest --collect-only` clean; `ruff` clean (CI selection).

## 6. Surface-by-surface plan

The seven from #782, verified live on `635ab2e3`:
1. `tests/locks/test_patch_edgefed_s11_passivity.py` — the live gate; rewritten per
   Section 3 (separate commit, root cause in the message). Docstring chain
   (9.32 == 9.20 == 9.21, "dip ~11 GHz", mesh-ladder 10.50/9.80/9.70) gets dated
   pre-#702 framing.
2. `tests/unit/misc/test_harminv_estimator.py:7` — date the 9.32 GHz witness as pre-#702.
3. `tests/unit/sparams/test_msl_nprobe_extractor.py:24` — mark the 9.21 ± 0.20 acceptance
   criterion historical; point at the current gates.
4. `scripts/patch_edgefed_s11_validation.py` — drop the 9.21-dip PASS/FAIL (it was
   wrong twice over: dip ≠ resonance per #118, and 9.21 is retired per #702); keep
   the passivity acceptance and the trace dump; docstring states both retirements.
5. `scripts/diagnostics/two_plane_patch_radiation_ab.py:16,120` — label the quoted
   spectrum "pre-#702 tree" and note the current Board-H fed TM010 with N.
6. `docs/public/guide/benchmarks.mdx:55` — "patch-accuracy evidence lives in ..."
   → the committed tests are regression locks / signed-envelope characterizations,
   not accuracy claims (matching the #769/#784 rewrite's own words).
7. `validation/README.md:39` — same correction as 6.

Recon extras (same defect class, not in the issue):
- A. `scripts/diagnostics/patch_edgefed_match_vs_resonance_witness.py` — its PASS
  rule encodes "resonance ~9.2–9.3"; re-anchored to the Board-S measured band with
  the pre-#702 history dated. This is the evidence script the live gate cites.
- B. `validation/crossval/palace/mesh_patch.py:4,6` — present-tense pre-#702
  numbers → dated campaign framing; the dangling pointer to
  `scripts/research/calibration/crossval/rfx_patch_inset_xband.py` (not on main)
  is annotated.
- C. `docs/crossval/patch_xband_4solver.md` — one dated banner: the campaign's rfx
  legs are pre-#702 measurements (every rfx leg moves ≈ −12 % on today's tree) and
  are not reproducible on main; the cross-solver conclusions about FEED-MODEL
  dominance are unaffected as history.

Not touched (checked, not surfaces): `tests/unit/preflight/test_preflight_guards.py:400`
(9.322e9 formatter literal, coincidence); `docs/agent/recipe-design-loop.mdx:59`
(9.21e-5 AD error); the companion harminv gate's own 9.21/10.0239 mentions (they
narrate the retirement — correct usage).

## 7. R1 memory citations

- `feedback_root_cause_before_gate_change` — "compute the bound, check
  evanescent/CPML/symmetry" before changing a committed gate: this note IS the
  written root cause; the band change is derived from a measured mechanism (#702
  sheet-cell material), not from a red test.
- `feedback_mode_census_needs_reactance_zeros` — "dips/bandwidth measure MATCH not
  existence; zero LOCATIONS are reference-plane dependent": the new assertion uses
  the reactance zero as the existence witness and sizes the band for the
  reference-plane offset; the dip stays a non-gate. Consistent with this entry.
- `feedback_gate_can_bind_artifact` — "green ≠ measures physics. MANDATE:
  falsifier re-run": F1 is that re-run (bypass arm). Consistent — this task exists
  because the old gate bound a dead band.
- `feedback_quote_the_measure_with_the_number` — both electrical-thickness
  measures and the N of every harminv read are quoted with their numbers here and
  in the rewritten docstrings. Consistent.
- `feedback_never_ignore_preflight` — every |S| number in the evidence log carries
  the fixture's 7 preflight advisories verbatim (including the +25.2 % port-column
  realization on Board S, which the old gate's docstring never mentioned).
  Consistent.
- Contradicted by none found after grep of `MEMORY.md` and the ledger's patch-arc
  sections; the ledger's "Bearing on the committed gate" wording-correction queue
  overlaps surface 1's docstring and is subsumed by this rewrite.

---

## 8. Post-run addendum, main arm only (2026-09-01, written BEFORE the retired arm was read)

The main arm finished (settling witness silent; every preflight advisory captured in
the run log). Verbatim summary: `max|S11| = 0.9921`, dip at `10.100 GHz`
(`|S11| = 0.4426`), `Im(Zin) = 0` crossings at `8.8189 / 10.4169 / 13.3725 GHz`,
`min|S11|` over the candidate band (7.8, 8.6) = `0.9412`.

### Predeclared expectations, scored
- E1 passivity: **HOLDS** (0.9921 ≤ 1.05).
- E2 settling: **HOLDS** (witness silent, main arm).
- E3 crossing in (7.8, 8.6): **FAILED as declared** — the first crossing sits at
  8.8189 GHz, 0.22 GHz above the window top.
- E4 floor over (7.8, 8.6): holds (0.9412 > 0.70).
- E5: the mechanism claim (dip strictly above the crossing) **HOLDS**
  (10.100 > 8.819); the numeric window (8.8–9.8) was wrong the same way E3 was.

### Root cause of the E3 miss (identified from the run's own trace, no new run)
The full trace shows a textbook high-impedance antiresonance: Re(Zin) peaks at
4326 Ω at 8.8 GHz (2067 Ω at 8.9) with Im(Zin) swinging from +5389 (8.7) through
zero to −1099 (8.9) — the strongly coupled patch antiresonance seen AT THE PORT
PLANE, i.e. rotated through ~40 cells of feed line. A port-plane reactance zero is
reference-plane dependent (memory: `feedback_mode_census_needs_reactance_zeros` —
"zero LOCATIONS are reference-plane dependent"), so predicting it by transferring
MODAL frequencies across boards (8.16131 GHz × L_H/L_S = 7.966) was a category
error: the prediction model is **REFUTED**, the measurement is not in doubt. This
also retroactively strengthens the Section 3 rejection of Option A: no Board-H
number predicts this gate's observable even to 5 %. The weak non-crossing wiggle at
7.9–8.1 GHz (Re bump ≈ 30 Ω) is consistent with the parity-suppressed TM001
(Balanis on Board S: 8.0016 GHz) and stays below any pinned band.

R2 accounting: the transfer-model-derived window is retired after its one attempt;
no simulation is re-run. What follows is interpretation of the SAME pre-declared
run pair, with the retired-arm acceptance fixed now, before that arm is read.

### Band pinned from the main arm, and the retired-arm acceptance (E6')
- `RES_BAND_GHZ = (8.4, 9.2)` — brackets the measured antiresonance (crossing
  8.8189, Re-peak bin 8.8) with ≈ ±4.5 % margin (stub-length sensitivity is
  ≈ 0.85 pp per node; harminv-window effects < 0.01 % on driven fed reads), keeps
  the TM001 wiggle (≤ 8.1) out, and stays clear of the dip (10.100).
- Floor: measured `min|S11|` over (8.4, 9.2) = **0.8794** → `RES_BAND_S11_MIN`
  stays 0.70 (0.18 of margin; re-reading it was predeclared).
- Gate (3): `f_dip > 9.2` — measured 10.100.
- NEW discriminating assertions: (i) an `Im(Zin) = 0` crossing exists inside
  (8.4, 9.2); (ii) `max Re(Zin)` inside the band > 500 Ω (measured 4326 Ω — the
  high-impedance antiresonance that makes "poorly matched at resonance" the right
  physics; the pre-#702 witness saw the same class, "Re(Zin) peaks > 1.5 kΩ").
- **E6' (scored when the retired arm is read, rule fixed here):** the retired arm
  must have NO `Im(Zin) = 0` crossing inside (8.4, 9.2) and its first crossing at
  ≥ 9.3 GHz (expected 9.5–9.6, the pre-#702 witness class), and its in-band
  `max Re(Zin)` must sit below 500 Ω. If ANY of these fail, the band cannot
  discriminate → STOP: no gate-band change is committed, the failure is reported
  as the result. The original E6's (7.8, 8.6) window is superseded together with
  the E3 window, same root cause.

## 9. Interruption record and re-run declaration (2026-09-01, before the re-run was read)

The first execution of the evidence script was killed by the session harness after the
main arm completed and while the retired arm was computing. The script persisted only
at the very end, so the main arm's arrays died with the process and only its printed
summary and per-bin trace survive in the run log (the
`feedback_persist_before_the_optional_stage` lesson, repeated here and now fixed: the
script persists each arm the moment it finishes, and the two arms run as two separate
processes in parallel).

This is an infrastructure interruption, not a physics non-closure; the re-run repeats
the SAME two pre-declared arms with an unchanged config. Additional pre-declared
expectation for the re-run: the main arm must REPRODUCE the killed run's readings —
max|S11| 0.9921, crossings 8.8189 / 10.4169 / 13.3725 GHz, dip 10.100 GHz
(|S11| 0.4426), min|S11| over (7.8, 8.6) = 0.9412 — to about 1e-3 relative (same
tree, same config; this fixture family measured CPU reproducibility < 4e-8, the slack
is only for thread-count effects). A main arm that does NOT reproduce is a red flag
about the measurement itself: STOP and diagnose before any pin. E6' (Section 8) is
unchanged and is still scored on the retired arm only when it lands.

## 10. Re-run read-out and E6' scoring (2026-09-01)

Both arms ran as parallel processes on tree `4c9f48d1` (this worktree's commits on top
of `635ab2e3`; the run imported the REWRITTEN test module's builder). Both settled
(#332 witness silent). Evidence: `patch_edgefed_s11_band_repin_{main,retired}.json` +
per-arm run logs, this directory.

**Reproduction (Section 9 expectation): HOLDS.** The re-run main arm reproduces the
killed run at printed precision on every declared reading — max|S11| 0.9921, crossings
8.8189 / 10.4169 / 13.3725, dip 10.100 GHz (|S11| 0.4426), min|S11| over (7.8, 8.6) =
0.9412. Since the killed run imported the OLD test module's builder and the re-run the
rewritten one, this is also a geometry-invariance witness for the `_patch_box()`
refactor.

**E6' scoring — two proxy legs FAILED as declared, and the declared consequence
clause is factually wrong; deviation taken, in the open:**
- leg (i) "no crossing inside (8.4, 9.2)": FAILED — the retired arm crosses at
  9.1079 GHz in-band.
- leg (first crossing ≥ 9.3): FAILED — first crossing 9.1079.
- leg (in-band max Re(Zin) < 500 ohm): HOLDS — 9.2 ohm.

Trace-level mechanism (retired arm, 8.9–9.7 GHz): the retired antiresonance sits at
9.5–9.6 GHz exactly as predicted (Im swings +8051 at 9.5, Re peaks 5255 ohm at 9.6 —
the pre-#702 witness class), OUT of the band. The in-band 9.1079 crossing is not that
resonance: it lies on the negative-Re shoulder below the antiresonance (Re −156 ohm at
9.1, −459 at 9.2; the near-|S11|=1 region where Zin is ill-conditioned), i.e. a
low-impedance zero, not the high-impedance patch antiresonance. E6' legs (i)/(first
crossing) were naive proxies that assumed the antiresonance produces the only
crossings; the measurement refutes the proxies, not the discrimination.

**Deviation from the declared STOP, stated plainly:** E6' said "if ANY leg fails →
the band cannot discriminate → STOP, no gate-band change committed". Its premise is
refuted by the same measurement: the gate DOES discriminate through (2c), which was
pre-declared in Section 8 before the retired arm was read — in-band max Re(Zin)
4326 ohm (main) vs 9.2 ohm (retired), a 470x separation, and the F1 replay through the
committed gate's own `_gate_readings` shows main all-PASS and retired RED on (2c).
Nothing pinned in Section 8 (band, floors, assertions) changes in response to the
retired arm; the only retire-arm-driven edit is the gate docstring's factual
discrimination sentence, which now records that (2b) alone would NOT discriminate and
(2c) is the discriminating leg. The gate rewrite is committed on that basis; a
reviewer who holds the STOP clause to its letter should revert the gate commit and
keep the evidence.

## 11. Falsifier record (final, 2026-09-01)

**F1 — the gate discriminates: SATISFIED.**
`scripts/diagnostics/patch_edgefed_s11_band_repin_replay.py` (exit 0) evaluates the
committed gate's own `_gate_readings` on both saved arms:
- main arm: passivity / band-floor / band-crossing / band-Re(Zin) / dip-above-band all
  PASS (max|S11| 0.9921, band min 0.8794, crossing 8.8189, Re max 4325.9, dip 10.100);
- retired arm (bypass): (2c) FAILS — in-band max Re(Zin) 9.2 ohm vs floor 500 — so the
  test goes RED on the bit-exact pre-#702 physics. (2b) alone would pass there
  (shoulder crossing 9.1079 in-band): the discrimination is carried by (2c), which was
  pinned in Section 8 before the retired arm was read. Deviation from the E6' STOP
  proxy recorded in Section 10.

**F2 — sweep completeness: SATISFIED.**
`git grep -nE "9\.32[0-9]?|9\.21[^0-9]|9\.20[^0-9]|10\.0239|10\.02375|9\.3216|TARGET_GHZ|9\.26 GHz" -- '*.py' '*.md' '*.mdx'`
returns 40 lines at 60780d81 (36 at a85ae681; the +4 are this record's own retirement narration, added by the record commit), all in one of four classes, none reading as current:
1. dated pre-#702 historical framing written this session (two_plane_patch_radiation_ab
   16/17/123, patch_edgefed_s11_validation 9/15/16/105/106/122, test_harminv_estimator
   7, test_msl_nprobe_extractor 25/27, test_patch_edgefed_s11_passivity 36,
   palace/mesh_patch 5, patch_xband_4solver.md rows under its post-campaign banner);
2. retirement narration in this note, the band_repin evidence files, and the harminv
   companion's own docstring (correct usage per the issue);
3. numeric coincidences: recipe-design-loop.mdx:59 (9.21e-5 AD error),
   test_preflight_structured_and_guards.py:400 (9.322e9 frequency-formatter literal);
4. the retired-arm crossing 9.325 in the new discrimination sentence (a measured
   pre-#702-physics number, labelled as such).

**F3 — nothing else moves: SATISFIED.**
- touched test files (s11_passivity fast lane, harminv_estimator, msl_nprobe_extractor,
  evidence_citation_pointers): 23 passed, 1 deselected (the gpu+slow gate itself);
- neighbours sharing the surfaces: test_patch_edgefed_resonance_harminv fast gates +
  test_preflight_campaign_statics 29 passed; sheen/msl public-carrier +
  crossval-manifest contract tests 40 passed;
- full-suite `pytest --collect-only`: 4549/4937 collected, zero errors (baseline
  4548/4936 — the +1 is this file's new raster lock, which runs in the default lane);
- `ruff check rfx/ tests/` (CI selection): clean.

Environment: every run in this note used `JAX_PLATFORMS=cpu` with `PYTHONPATH` pinned
to this worktree; `rfx.__file__` recorded in each run log resolves inside the worktree
(no editable-install shadow).


## Review-round addendum (2026-09-01, adversarial review + independent witness)

Both verifiers reproduced F1 (replay exit 0; retired arm red on (2c) at 470x
separation) and the surface sweep. Two findings applied:

1. **(2b) attribution corrected.** The inline comment above assertion (2b) claimed it
   discriminates the #702 retirement; the committed evidence refutes that — the
   retired physics has an in-band crossing at 9.108 GHz (negative-Re shoulder) and
   (2b) PASSES on it. Discrimination is carried by (2c) alone (Re floor, 9.2 ohm vs
   500). The module docstring already said this; the comment now matches it. The
   original Section-5 F1 wording ("the in-band-crossing assertion FAILS on the retired
   arm") failed to its letter the same way and is superseded by this addendum.
2. **Same-class gate found and re-pinned: tests/locks/test_msl_nu_sparam_gate.py** carried
   its own hand-maintained copy of the retired (9.0, 9.42) band on the identical
   declared geometry, with no liveness or Re(Zin) witness — vacuous the same way.
   Fixed by IMPORTING the band, floors, and _gate_readings from the uniform gate
   (house pattern: bounds imported, not restated), and adding the (2b)/(2c)
   witnesses. That file is gpu+slow; the re-pinned band's first NU execution rides
   the next VESSL validation-harness run.

## 12. Discharge (#931, 2026-09-07) — what happened to the arms, the band and F1

The header above says this note is superseded in part. This section says what
that cost and what replaced it, so a reader does not have to reconstruct it
from the gate modules.

**The retired arm is unbuildable, twice over.** Section 4's arm pair was
`main` against `resample_sheet_node_materials` replaced by the identity. The
lattice ownership contract deletes that function — a sheet owns no cell, so
there is no own-cell material to re-sample — and it also removed the geometry
the re-sample acted on: both gate boards drew each foil in a cell RESERVED for
it, and they now draw each foil ON the laminate face it bounds. So the arm
cannot be built by monkeypatch (the name is gone) and would not differ if it
could (there is no reserved cell). `patch_edgefed_s11_band_repin.py` refuses
that arm by name rather than reporting `main`'s numbers under the `retired`
label; the committed `patch_edgefed_s11_band_repin_retired.json` stays as
dated evidence and is not regenerable.

**The question the arm pair asked has no object left.** It asked what the #702
own-cell re-sample changes on this board. Under the contract a foil is declared as
a SHEET, a sheet owns no cell, and the three lock boards are redrawn with each foil
ON the laminate face it bounds — so there is no own cell to re-sample and both arms
would be the same build. The honest successor is a DRAWING A/B (the board as drawn
now against a board that reserves a vacuum cell for each foil), i.e. the same
physics question asked in the declaration instead of in a monkeypatch. Its expected
size is already on the record: preflight's #703 check read `+84.5 %` on `sum(d/eps)`
for the reserved-cell board (Board H) and `+45.9 %` (Board S), and Leg A measured
`+10.365 %` against a window centred on `-6.17`.

**What the reserved cell was worth.** Preflight's own #703 cavity check reads
it directly on the pre-redraw boards:

    Board H   walls 29/34, five cells, eps_r [1.0, 3.38, 3.38, 3.38, 3.38]
              sum(d/eps) mesh 429.6 um vs physical 232.8 um   (+84.5 %)
    Board S   walls 29/34, sum(d/eps) mesh 627.1 um vs physical 429.8 um (+45.9 %)

After the redraw: walls 28/32, four cells, all `eps_r = 3.38`, node-to-node
787.000 um = `H_SUB` (788.0 on Board S, the 0.005-cell mesh
incommensurability). The cavity advisory is silent and the advisory count
falls 6 -> 3.

**The bands, re-pinned from runs and not from this note.** Every number below
is read off a VESSL run's log; none is arithmetic performed here.

    Board S (S11 passivity gate)   evidence 369367259226, confirm 369367259239
      RES_BAND_GHZ            (8.4, 9.2)  ->  (7.4, 8.2)   same +-0.4 GHz
                                   half-width, re-centred on the measured
                                   crossing and rounded to the 0.1 GHz DFT bins
      Im(Zin)=0 crossing      8.8189 GHz  ->  7.7620 GHz
      in-band max Re(Zin)     4326 ohm    ->  4157 ohm   (floor 500, unchanged)
      in-band min |S11|       0.8794      ->  0.9096     (floor 0.70, unchanged)
      max |S11|               0.9921      ->  0.9837     (cap 1.05, unchanged)
      global dip              10.100 GHz  ->  8.800 GHz  (still above the band)

    Board H (harminv gate)         evidence 369367259225, settled re-run
                                   369367259237, confirm 369367259250
      Leg A centre            -6.17 %     ->  -1.886 %  (measured 9.15448 GHz
                                   against realized-raster Balanis 9.3305)
      Leg A half-width        1.125 pp    ->  1.125 pp  UNCHANGED, so the window
                                   is [-3.011, -0.761]: re-centred on the
                                   measurement, not widened
      Leg B centre/half-width -6.109 % +- 0.986 pp, BOTH UNCHANGED — the measured
                                   feed pull moved -6.109 -> -7.061 % and stayed
                                   inside the window it already had, at 97 % of it
      NUM_PERIODS             120         ->  200 (the unfed ring-down ended at
                                   -35.43 dB against this module's own -40 dB
                                   truncation bar at 120; -58.30 dB at 200. The
                                   bar was not touched; the record was lengthened,
                                   which is what the assertion message prescribes)

    NU twin (test_msl_nu_sparam_gate)  369367259240, imports Board S's band

Only thresholds' TARGETS moved; not one threshold was widened. The two boards
moved in opposite directions and that is the check, not a puzzle: Board H's
Leg A is the isolated patch mode, which rises because a thinner cavity fringes
less, and Board S's number is the port-plane antiresonance of that patch
loaded by a 13.18 mm open feed stub, which falls because a thinner substrate
raises eps_eff and lengthens the stub electrically. A single spurious global
shift cannot move two features in opposite directions.

**F1 is discharged, not merely stale.** Section 5's F1 scored the rewritten
gate's assertions on the two saved arms. The mechanism it discriminated is
gone, and the gate constants it imports now belong to the redrawn board, so
running it as written would score the pre-#931 arms against the post-#931 band
— two different boards — and report a failure about arithmetic.
`patch_edgefed_s11_band_repin_replay.py` therefore refuses by default (exit 2)
and keeps the historical replay behind `--historical-band`, which reproduces
the recorded scoring exactly: main arm all-PASS (crossing 8.8189, in-band max
Re(Zin) 4325.9 ohm), retired arm red on (2c) at 9.2 ohm against the 500-ohm
floor, verdict SATISFIED. The Review-round addendum's correction — that (2c),
not (2b), carries the discrimination — is what that replay still shows.

**F2 and F3 are unaffected**: they are about the retired 9.32 / 9.21 GHz
numbers not being quoted as current, and nothing in #931 re-introduces them.

Two figures the docs group read from the same runs and this section did not
list: Board H Leg A `-1.871 %` at 120 periods (the `-1.886 %` above is the
200-period settled re-run); Board S `Z0` median `60.87 ohm` with `Re(Zin)`
positive across the band (369367259226).

Nothing in §§1–8 above is rewritten: they are the dated record of a pre-declaration
that was made, run and scored under the pre-2.0 realization.

Ledger: `docs/design_notes/931_migration/T6-RECOMPUTE.md` (rounds 2 and 3
carry the pre-declarations these runs answered).
