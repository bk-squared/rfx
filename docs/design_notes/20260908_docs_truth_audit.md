# Artifact-to-carrier truth audit — #931 (2026-09-08)

This audit reads the committed artifacts; it does not re-solve a crossval case,
regenerate a fixture, change a numerical tolerance, or repair physics. The source
revisions and **every changed JSON field** are in the
[field ledger](20260908_docs_truth_field_ledger.md). It compares main to the
branch's pre-edit HEAD `ba619a51`, not to an intermediate migration pass.

## Scope and method

The requested literal glob contains **8 result directories** and **10 fixture
directories**. The changed `_issue812_phase_identity/regate_evidence.json` adds a
ninth result group. Total: **28 JSON files, 6,455 changed/added fields**. Added
files are enumerated as additions, never described as if they had a main-side
numerical baseline. Arrays of records are expanded by index; numeric vectors
retain their field path, exact changed-index ranges, length and content hash.
The immutable git revisions hold their full old/new values. During this audit,
three unrelated commits landed on the shared branch (`28d2a46d`, `b5b84e39`,
`47b66f63`). They change a migration note, the normative note and a slow-lane
job definition; none changes these JSON sources. Their work is preserved.

We searched the case names, artifact filenames, JSON keys, old numeric spellings
and their displayed rounding across public docs, validation README, diagnostic
README/tool prose, case scripts, CHANGELOG, design notes and tests. A numeric hit
was checked against its quantity, window, and whether it described a current or
explicitly historical run. Old dated results were not globally substituted.
Artifact metadata, runtime changes and floating-point-last-digit differences are
included in the ledger even where no carrier quotes them. The probe/S-parameter guide
and shared spectral-estimator docstring also now identify unremeasured null-mask
counts and estimator-agreement figures as pre-#931 evidence. Script changes are
prose only, apart from the diagnostic reference-point and case05 history print strings,
which are also displayed prose. Carrier test expectations read committed JSON directly
where practical, with no changes to numerical gate tolerances.

The worktree has no repository `docs/agent-memory/`; no other worktree was read
or modified. The normative in-worktree design note §5 supplies the requested R3
reference. Its conductor-history constraint is consistent with this audit:
regenerated measurements must be read from their new artifacts, not translated
from the old digits.

## Per-case comparison and carrier disposition

The table summarizes numerical changes. The linked field ledger is the complete
field-by-field list, including metadata, arrays and additions.

| Case / artifact group | main → branch | Carriers changed | Checked and retained / reason |
|---|---|---|---|
| cv05 ringdown / `_05_patch_results`, `patch_mode_identification` | 22-mm TM110 pole 2.993459 → 3.043213 GHz (index 0 → 1); 38-mm weak pole 2.609612 → 2.702632 GHz; baseline modes 3 → 4, exit 1 → 0; new stack witnesses and edge census; new openEMS and control result records | Public benchmark live pole pointers; case05 history label; design-note live-pointer corrections; new patch/MSL carrier tests | Validation README and mode-identification gate expectations already use current poles. Old tutorial/run measurements retained as dated history. The 38-mm weak-pole problem is still an instrument falsifier, not solved here. |
| cv05 canonical farfield / `patch_canonical_farfield_e4` | New artifact `canonical_farfield_e4_measured_369367259302.json`: directivity difference 0.0659099 dB, frequency difference +3.50612%; E/H cut peaks −1/−4° (old test prose said 0/−3°) | Patch demo NTFF comment, farfield test prose, new fixture-derived carrier test | Numerical farfield locks already use the new measurement; historic values remain scoped to their run. No old JSON at this new filename. |
| Experiment patch / `experiments/patch_antenna_v2.json` | Ground lower z 0.010 → 0.012 m; patch upper z 0.016 → 0.014 m | None required | Geometry consumers read the JSON dynamically; no independent numerical-result prose carrier found. |
| cv06b results / `_06b_msl_notch_results` | Refined notch 3.625498 → 3.758601 GHz; error 1.453016 → 2.164939%; BW ratio 0.968399 → 0.999132; Z0 46.481817 → 48.192045 Ω; one-cell falsifier becomes visible; narrow-stub G1 true → false | Case06b current summary and width rationale, Sheen impedance references, historical/current test comment, design notes and CHANGELOG qualification, new fixture-derived test | Existing public and validation cv06b headlines and log-based tests were already current. Old predeclarations retained as predictions, with falsification stated. Tolerances untouched. |
| cv06b estimator / `cv06b_estimator_regate` | Six scalar changes at the floating-point-last-digit level | None required | Rounded carrier values unchanged; estimator tolerances unchanged. |
| cv07 / `_07_sheen_results`, `sheen_lpf_e4` | Passband Z0 50.3026409 → 51.9122658 Ω; in-band 52.3755760 → 54.7301865 Ω; correction 0/120 → 3/120, worst 0.01450837 → 0.65716088; rfx structure 1.5195 → 2.8668%; separate rfx argmin distance 1.5430 → 2.3756%; lower/upper interpolated zeros 6.943990/7.925928 → 7.233338/8.244069 GHz; depth and settling also move | Validation/public pages; Palace README, producer, mesh and check-sparams prose; case07 banner/comments; Palace-gate docstring; carrier expectations; estimator-falsifier history labels; design notes/CHANGELOG | OpenEMS structure 0.6644%, argmin-distance 0.6689%, interpolated argmin 7.994749 GHz and Palace raw arrays are unchanged. Dated pre-#931 evidence retained. Public max-column-power 0.9995 still rounds correctly. |
| cv15 / `_15_patch_results` | f0 2.313947 → 2.423039 GHz (+4.71453%); Q 18.8974 → 10.0605; dip −4.42984 → −19.04801 dB at 2.310000 → 2.420000 GHz; gain 7.242423 → 7.199780 dBi; settle −53.98978 → −68.69142 dB; max abs S11 0.786966 → 0.990567; new old-feed decomposition 2.377431 GHz and preserved pre-#931 leg | Case15 stale shallow-dip and 5.3% prose, decomposition-test prose and fixture-derived carrier test | Public/validation historical +6.09% negative-control figures still refer to the preserved old leg, not current production. Script's dated 2.3139-GHz history is valid. Before/after diagnostic reads JSON dynamically. |
| cv18 / `_18_wr90_iris_results`, `wr90_iris_modematch` | Fine envelope 0.0232 → 0.0106 and pooled gate 0.04 → 0.02 (already changed before this audit); Richardson envelope 0.0051 → 0.0046; per-config gates 0.006–0.016; coarse/raw/ripple summaries move; both-sign aperture detection 8/8; nearest-plane offset 0; one-cell live evidence changes | Case18 header, test header and fixture-derived summary guard, gate-policy historical label, design-note current-pointer appendix | Current numerical gates and public/validation pass-2 figures already repinned. Historical predeclarations and old/new tables retained. Modal power 1.0200/1.0012 and truncation ≤1e−5 still valid at displayed precision. |
| cv19 / `_19_iris_filter_results`, `wr90_iris_filter` | f0 residual +12.08 → +12.12 MHz; population spread 0.06 → 0.02 MHz; envelope 12.1219 MHz; current edges +17.05/+7.20, BW −9.85 MHz; settling now [−59.23,−59.23] dB | Validation population spread, public settling disclosure, both pages' +107.5-MHz bias labelled historical, gate-test prose and fixture-derived carrier guard | 19-MHz f0 gate, pass-2 envelopes and public d_f0 already current; unchanged rounded FDFD agreement 1.1/1.0 MHz. +12.08-MHz predeclared falsifier and old mutations retained as history. Protected claim_scope defect below remains. |
| cv20 / `msl_phase_referee`, `_issue812_phase_identity` | S/Z0/beta arrays; trace y 950–1550 → 900–1500 µm; probe spacing 20 → 11; volume/wall metadata added; blindness 0.2414403 → 0.0646838°; corrected residual 0.7153451 → 0.7917115°; added current-rfx/historical-OpenEMS replay 0.5308358° and beta errors 1.4121818/0.3067916% | Producer/case prose, public/validation pages, support matrix, design notes, fixture-derived carrier tests | Historical matched run-2 raw phase 0.3418° retained with its provenance. Current replay is explicitly a mixed-generation pair, not a fresh matched-board solve. |
| Waveguide chain / `waveguide_chain_battery` | New `fixture_v18_close.json`, VESSL 369367258638; 134 pass / 51 report_only / 0 fail | None required | Census matches verdicts. Support matrix explicitly scopes these measurements to pre-2.0; adding a previously absent fixture is not a fresh #931 solve. |
| Waveguide phase / `waveguide_vi_envelope` | New `s21_phase_residual_witness.json`, VESSL 369367258390; R0–R7 orders within 0.05 of 2 | None required | API/phase-test/support descriptions match the artifact. No main-side artifact at this filename. |
| cv22 / `_22_dispersive_results` | Cancellation-level numerical changes; runtimes and tail-fit reliability notes / Meep precheck metadata | None required | Public physical values round unchanged; no runtime-number carrier needed correction. This is not literally bit-identical JSON. |
| cv24 / `_24_nu_cavity_results` | Cancellation-level frequency/damping/stationarity differences; runtime changes; `metric_swap_in_scope=false` added | Dated design-note performance-table qualification | Public physical values round unchanged; historical performance timings retained with run scope. Unknown provenance commit is reported below. |

## cv07 quantities and passivity, without conflation

The current raw-bin argmins are **8.201680896 / 7.983125 GHz** (rfx/OpenEMS).
The Palace fixture's parabolic FDTD argmins are **8.244069 / 7.994749 GHz**.
The rfx **2.8668% structure distance** compares the two fitted doublet zeros;
**2.3756% argmin distance** is a different field/window. A numerical replacement
is never justified solely by proximity of those percentages.

The three correction bins are at **17.869748224, 18.033612800 and
18.197479424 GHz**. They are all above 17 GHz, none in the 5–15 GHz null band.
The public text states both **3/120, worst 0.6572** and the case's interpretation:
a high-frequency residual on the coarser realized strip at the same dx = 200 µm,
not evidence of a physics regression in the claimed null band. No new causal
experiment is claimed. Corrected column power and raw passivity correction are
also distinct quantities.

An exact re-read refines the supplied column-power estimate:
`max(s11_mag**2+s21_mag**2)` is **0.999514468068516 on main** and
**0.9995126876488003 on branch**. Both display as **0.9995**; the artifact-lock
expectation uses 0.999513 with its existing tolerance unchanged. Passband and
in-band Z0 windows contain 16 and 61 bins respectively.

## Named unresolved findings (no fixture/physics repair in this commit)

- **DT-F01 — cv07 referee narrative disagrees with its current numbers.**
  `tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json::referee.argmin_first_null.note` retains the
  claim that current FDTD solvers choose different doublet members. Both current
  FDTD argmins and Palace-mid choose the upper member; Palace-coarse chooses the
  lower. External prose is corrected. The fixture and its producer's frozen
  narrative/equality contract need a separate authorized consistency repair.
- **DT-F02 — cv07 estimator-falsifier evidence is pre-#931.**
  `tests/fixtures/cv07_estimator_regate/cv07_estimator_falsifiers.json` and the
  diagnostic's executable anchors still describe the old leg. Carriers now
  state that scope. Re-establishing isolated planted-defect responses needs a
  separate post-#931 measurement; changing anchors here would be beyond prose.
- **DT-F03 — cv07 port impedance/footprint attribution remains open.**
  HJ(2200,800)=54.22 Ω differs from the current passband median by 2.31 Ω,
  not the earlier 4 Ω. This audit corrects the rationale without choosing a new
  port width/reference or claiming a new physics diagnosis. Protected
  `_07_sheen_results/RECOMPUTE.md` still quotes the old 50.30-Ω rationale.
- **DT-F04 — cv06b electrical-width prediction was falsified.** The current
  48.192045-Ω measurement does not support the prior claim that the old 46.48-Ω
  datum settles that convention. Geometry/port attribution remains a separate
  investigation; the new prose says so explicitly.
- **DT-F05 — cv20 lacks a fresh matched-board post-#931 OpenEMS solve.** The
  current replay combines historical run-2 OpenEMS with current rfx. A new Stage
  B external run is still owed by `scripts/vessl_issue931_post_XE/RECOMPUTE.md`.
  The rfx JSON itself lacks named-run/commit metadata; VESSL 369367259200 is
  associated via that run record, not invented as fixture-local provenance.
- **DT-F06 — cv19 frozen claim_scope says settling is absent.** Both fixture
  copies say the S-matrix path has no `settling_db`, although the gated and
  coarse records contain [−59.23,−59.23] and [−50.4,−50.4] dB. The script's
  frozen claim_scope string and its equality test also preserve that mismatch.
  Public prose is corrected; neither protected fixture nor the frozen equality
  contract is loosened. The same frozen text says absorber shifts 0.0/0.1 MHz,
  while the numeric summary is now 0.00/0.01 MHz (0.0/0.0 at one decimal).
  Repair needs coordinated fixture-prose authorization.
- **DT-F07 — cv05 result claim_scope mixes old and current substrate scope.**
  `_05_patch_results/cv05_run_openems_369367259142.json` still says “two different
  substrate geometries (#325)” while also recording the exact #931 1.5-mm
  cavity. This is protected artifact prose; it was not silently corrected.
- **DT-F08 — cv11 slab comparison remains unresolved (outside changed JSON).**
  The changed run logs/RECOMPUTE record a current slab discrepancy about 0.1487
  versus 0.1468 on main, predating this branch; broad-E4 JSON was intentionally
  not refreshed pending attribution (the documented fresh comparison would
  move about 0.0707 → 0.1500 against a 0.1 gate). No gate or fixture changed.
  Current pec-short magnitude/phase carriers already reflect the corrected
  drawing. Attribution requires a separate comparator/fixture investigation.
- **DT-F09 — cv24 provenance is incomplete.** The regenerated result records
  `commit="unknown"`; numeric agreement cannot establish the missing source
  revision. Recover it from the named run logs rather than fabricating it.

These findings are recorded here for follow-up; no external issue was filed.
Protected fixture/result prose is explicitly outside the user's authorized edit
scope, even where it contains a current-looking stale statement.

## Changed-path inventory

Each path below was reviewed; the case table above states the reason and the
historical/current distinction. Test-file results are listed separately below.

- `CHANGELOG.md`
- `docs/design_notes/20260901_patch_mode_identification_predeclaration.md`
- `docs/design_notes/20260902_cv24_nu_cavity_predeclaration.md`
- `docs/design_notes/20260903_e4_all_solver_classes_plan.md`
- `docs/design_notes/chain_closure_contract.md`
- `docs/design_notes/estimator_resolution_regate.md`
- `docs/design_notes/issue812_cv17_cv18_geometry_sensitivity_predeclaration.md`
- `docs/design_notes/issue812_phase_identity_predeclaration.md`
- `docs/design_notes/v18_waveguide_s_chain_plan.md`
- `docs/guides/sparameter_support_matrix.md`
- `docs/public/guide/benchmarks.mdx`
- `docs/public/guide/probes-sparams.mdx`
- `examples/tutorials/patch_antenna_demo.py`
- `scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py`
- `scripts/diagnostics/build_sheen_lpf_palace_referee.py`
- `scripts/diagnostics/cv07_estimator_falsifiers.py`
- `scripts/diagnostics/palace_sheen_referee/README.md`
- `scripts/diagnostics/palace_sheen_referee/check_sparams.py`
- `scripts/diagnostics/palace_sheen_referee/mesh_sheen.py`
- `tests/_gate_policy.py`
- `tests/crossval/test_crossval_cv15_wall_planes.py`
- `tests/crossval/test_msl_notch_public_carriers.py`
- `tests/crossval/test_patch_canonical_farfield_e4.py`
- `tests/crossval/test_patch_msl_public_carriers.py`
- `tests/crossval/test_sheen_lpf_palace_referee_gates.py`
- `tests/crossval/test_sheen_lpf_public_carriers.py`
- `tests/crossval/test_wr90_iris_filter_gates.py`
- `tests/crossval/test_wr90_iris_modematch_gates.py`
- `validation/README.md`
- `validation/crossval/05_patch_antenna.py`
- `validation/crossval/06b_msl_notch_filter_uniform.py`
- `validation/crossval/07_sheen_lpf.py`
- `validation/crossval/15_patch_antenna_rt5880.py`
- `validation/crossval/18_wr90_iris_modematch.py`
- `validation/crossval/20_msl_phase_referee.py`
- `validation/crossval/comparators/spectral_features.py`

The audit and its field ledger are the two new report documents.

## Verification

Selected tests: **161 passed, 0 failed, 0 skipped**, in 93.63 s. The five
`slow` canonical-farfield FDTD tests were deliberately deselected; no solve
or regeneration was run. Three existing geometry/source warnings were emitted.

| Touched test file | Passed | Failed | Skipped | Deselected |
|---|---:|---:|---:|---:|
| `tests/crossval/test_crossval_cv15_wall_planes.py` | 18 | 0 | 0 | 0 |
| `tests/crossval/test_msl_notch_public_carriers.py` | 25 | 0 | 0 | 0 |
| `tests/crossval/test_patch_canonical_farfield_e4.py` | 4 | 0 | 0 | 5 |
| `tests/crossval/test_patch_msl_public_carriers.py` | 23 | 0 | 0 | 0 |
| `tests/crossval/test_sheen_lpf_palace_referee_gates.py` | 7 | 0 | 0 | 0 |
| `tests/crossval/test_sheen_lpf_public_carriers.py` | 14 | 0 | 0 | 0 |
| `tests/crossval/test_wr90_iris_filter_gates.py` | 47 | 0 | 0 | 0 |
| `tests/crossval/test_wr90_iris_modematch_gates.py` | 23 | 0 | 0 | 0 |

Command (all eight files together): `timeout 1200 python -m pytest -q -n 4
<the eight files above> --junitxml=.docs-truth-audit/tests.xml`, with
`JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and the
repository default marker selection (`not gpu and not slow and not slow_physics`).
The launcher waited until no other pytest process was active before starting.
Task invocations are serialized.

Additional evidence-reference contract first pass: **594 passed, 2 failed**.
Both failures were documentation-reference defects introduced during this sweep:
a citation resolved to a JSON object instead of a scalar, and an overly broad
text replacement copied the cv20 replay paragraph into cv21. The citation now
resolves to the scalar phase difference, and cv21 is restored verbatim.
Focused rerun: **618 passed, 0 failed, 0 skipped** in 2.93 s, comprising
`tests/contracts/test_evidence_citation_pointers.py`: **4 passed / 0 failed**,
`tests/contracts/test_evidence_numeric_provenance.py`: **591 passed / 0 failed**,
`tests/crossval/test_patch_msl_public_carriers.py`: **23 passed / 0 failed**.
The 23 patch/MSL cases repeat the first run; **756 distinct selected tests**
passed across the final evidence. No test failure remains.

`git diff --check` passed. Protected paths have no diff. AST comparison after
removing docstrings confirms production logic unchanged; the only other AST
differences are literal displayed-history/reference-point print strings in
case05 and Palace `check_sparams.py`. Numerical tolerance values were not changed.
The full JSON leaf ledger was generated from git objects, not from a solve.
