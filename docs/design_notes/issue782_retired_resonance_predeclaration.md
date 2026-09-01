# Issue #782 — retiring the 9.32/9.21 GHz patch numbers: predeclaration

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

### Board H — the harminv-gate board (`tests/test_patch_edgefed_resonance_harminv.py`)
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

### Board S — the S11-passivity-gate board (`tests/test_patch_edgefed_s11_passivity.py`)
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
  identity, the same bypass `tests/test_preflight_campaign_statics.py::_bypass_resample`
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
1. `tests/test_patch_edgefed_s11_passivity.py` — the live gate; rewritten per
   Section 3 (separate commit, root cause in the message). Docstring chain
   (9.32 == 9.20 == 9.21, "dip ~11 GHz", mesh-ladder 10.50/9.80/9.70) gets dated
   pre-#702 framing.
2. `tests/test_harminv_estimator.py:7` — date the 9.32 GHz witness as pre-#702.
3. `tests/test_msl_nprobe_extractor.py:24` — mark the 9.21 ± 0.20 acceptance
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

Not touched (checked, not surfaces): `tests/test_preflight_structured_and_guards.py:400`
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
