# Test-suite reorganization — tier 3b: consolidation of duplicated test files

Date: 2026-09-03. Branch `agent/reorg-tier3b-consolidate`, stacked on tier 4b
(`agent/reorg-tier4b-unit`, tip `3bafe2f` = the BEFORE reference of every
count below). PI instruction, verbatim: "3b도 진행해 누락되는거 있나 점검 하면
되잖아" — do 3b too, just check nothing gets dropped. This note is the
mechanical nothing-lost proof; it is append-only (dated sections at the end).

Rules applied (from the tier brief): a consolidated file keeps every
assertion, tolerance, fixture value, marker and parametrisation of the tests
it absorbs; no tolerance widened, no parameter case dropped, no exact check
replaced by a looser one; every source file is concatenated verbatim below a
new module docstring + merged import block (the pre-merge files are in git
history at `3bafe2f`). One commit per consolidation group.

## 1. What was merged

| group (brief) | before | after | merged file(s) | commit |
|---|---|---|---|---|
| cpml_dielectric 7→1 | 7 | 1 | `tests/unit/boundaries/test_cpml_material_aware.py` | `merge cpml_dielectric 7→1` |
| settling witness 3→1 | 3 | 1 | `tests/unit/sparams/test_settling_witness.py` | `merge settling witness 3→1` |
| sbp_sat 5→1 | 5 | 1 | `tests/unit/subgrid/test_sbp_sat.py` | `merge sbp_sat 5→1` |
| waveguide_broad_e5 5→1 | 5 | 1 | `tests/crossval/test_waveguide_broad_e5.py` | `merge waveguide_broad_e5 5→1` |
| coax 7→2 (S-matrix vs field/impedance) | 4 of 7 merged | 2 (+3 kept, see §3) | `tests/unit/sparams/test_coax_two_port_smatrix.py` (S-matrix), `tests/unit/sparams/test_coaxial_line_reflection.py` (field/impedance) | `merge coax 4→2` |
| preflight 11→3–4 (by stage) | 10 of 11 merged | 3 (+1 kept, see §3) | `tests/unit/preflight/test_preflight_absorber.py`, `test_preflight_rasterization.py`, `test_preflight_guards.py` | `merge preflight 10→3` |
| issue #677 sheet 10→3 | 5 of 10 merged | 1 (+5 kept, see §3) | `tests/unit/materials/test_sheet_impedance.py` | `merge #677 sheet materials 5→1` |
| patch 5→2 | 0 of 5 merged | — | (left unmerged, see §3) | — |

Totals: 39 files absorbed into 10 (net −29 test files; 462 → 433 on this
stack). Per-file membership:

* `test_cpml_material_aware.py` ← `nonuniform/test_nonuniform_cpml_dielectric.py`,
  `runners/test_distributed_cpml_dielectric.py`, `runners/test_distributed_nu_cpml_dielectric.py`,
  `runners/test_distributed_pmap_cpml_dielectric.py`, `runners/test_vmap_cpml_dielectric.py`,
  `sparams/test_lumped_wire_sparam_cpml_dielectric.py`, `subgrid/test_subgrid_cpml_dielectric.py`.
  Subject: one fix (#203-#205, `materials=` threaded into every CPML scan
  body); seven scan bodies. Multi-device sections keep the XLA host-device
  sentinel and `requires_multidevice` skip; module constants carry a
  section prefix (`_UNI_*`, `_NU_*`, `_PMAP_*`, `_LW_N_STEPS`), values unchanged.
* `test_settling_witness.py` ← `sparams/test_msl_settling_witness.py`,
  `sparams/test_settling_witness_enforcement.py`, `sparams/test_waveguide_settling_witness.py`
  (`FREQS`→`_MSL_FREQS`, `_thru`→`_msl_thru`, `_FREQS`→`_WG_FREQS`, `_two_port`→`_wg_two_port`; values unchanged).
* `test_sbp_sat.py` ← `subgrid/test_sbp_sat_{1d,2d,3d,alpha,jit}.py`. The 1d
  and 2d files carried a module-level `pytestmark = pytest.mark.gpu`; the
  merged file carries `@pytest.mark.gpu` on each of those 7 tests (plus their
  existing `slow` marks), so lane selection is unchanged.
* `test_waveguide_broad_e5.py` ← `crossval/test_waveguide_broad_e5_{envelope_gates,tolerance_envelope,phase_gates,phase_tolerance_envelope,live_anchor}.py`
  (`_fixture_files` disambiguated to `_mag_fixture_files` / `_phase_fixture_files`, `_build_sim`→`_live_build_sim`; the two identical `MARGIN_CEIL = ENVELOPE_GATE_MULTIPLIER` definitions became one).
* `test_coax_two_port_smatrix.py` ← `sparams/test_coax_two_port_fdtd.py`, `sparams/test_coax_two_port_solve.py`;
  `test_coaxial_line_reflection.py` ← `sparams/test_coaxial_line_calibration.py`, `sparams/test_coaxial_line_extraction.py`.
* `test_preflight_absorber.py` ← `test_preflight_absorber_frame.py` (#500 frame),
  `test_preflight_geometry_absorber_aggregation.py` (#660), `test_preflight_dispersive_pole_at_absorber.py` (#636/#808;
  its `DX`/`_sim` renamed `DP_DX`/`_dp_sim` to coexist with #660's, values unchanged).
  `test_preflight_rasterization.py` ← `test_preflight_campaign_statics.py` (#703),
  `test_preflight_graded_rasterization.py` (#562), `test_preflight_thin_metal_nu.py` (#48).
  `test_preflight_guards.py` ← `test_preflight_physics_thresholds.py` (#37),
  `test_preflight_false_positives.py`, `test_preflight_structured_and_guards.py`, `test_preflight_tfsf_lumped.py`.
  `.github/workflows/pr-tests.yml` guards-and-preflight FILES: the three merged
  entries became the one `test_preflight_guards.py` (the lane now also runs
  the 3 fast tfsf_lumped tests).
* `test_sheet_impedance.py` ← `materials/test_leontovich_sheet_identity.py` (#669 O6/O7),
  `materials/test_sheet_impedance_operator.py` (#677 G3-G9), `materials/test_sheet_lane_fences.py` (#677 G9 registry),
  `materials/test_sheet_stacked_adjacent_gap.py` (#690), `materials/test_thin_conductor_nonbox_sheet.py` (#674).
  `FENCE_REGISTRY` rows that named the absorbed modules now name
  `tests.unit.materials.test_sheet_impedance`; `test_every_registered_pinning_test_exists`
  still resolves every row.

### Identical duplicates dropped during concatenation (no test, no assertion)

| dropped text (source @ `3bafe2f`) | why | sha256[:16] of the dropped text |
|---|---|---|
| `test_preflight_thin_metal_nu.py` L13-14 | `_has` identical to `test_preflight_graded_rasterization.py::_has` | `74db639b50f549c0` |
| `test_preflight_false_positives.py` L31-36 | `_issues` / `_has` identical to `test_preflight_physics_thresholds.py` | `f5fd66dac2de95f4` |
| `test_preflight_tfsf_lumped.py` L16-17 | `_codes` identical to `test_preflight_false_positives.py::_codes` | `26ee1e3a216f7f83` |
| `test_thin_conductor_nonbox_sheet.py` L159-166 | `_sha` identical to `test_leontovich_sheet_identity.py::_sha` | `b99f8b4d6ad9f8e8` |
| `test_sheet_impedance_operator.py` L213-214 | in-function import of `_planar_sheet`, `U_Z`, `U_FOOT`, `U_HOLE` from the absorbed sibling (now in-module names) | `fbb212d5e4d9b3e8` |
| `test_preflight_graded_rasterization.py` L128-129 | mid-file `import math` / `import pytest` (in the merged header) | `5614cabadc4c0438` |

### Non-verbatim edits (found by the independent review, §7; semantics preserved)

The "concatenated verbatim" claim above holds for every test body. Four
helper-level edits in `tests/unit/boundaries/test_cpml_material_aware.py` are
not verbatim and were missing from this note:

| edit | before (`3bafe2f`) | after | check |
|---|---|---|---|
| max-abs-E helper unified | `test_distributed_cpml_dielectric.py:67-77 _max_abs_e(result)`, `test_distributed_nu_cpml_dielectric.py:110-119 _max_abs_E(state)`, `test_distributed_pmap_cpml_dielectric.py:68-75 _maxabs_e(result)` | `_max_abs_e_state(state)` plus `_max_abs_e(result)` delegating to it; the two other names deleted and their 4 call sites re-pointed | all three bodies were the same loop over `ex, ey, ez` returning `np.inf` on any non-finite value, else the max abs; the reviewer diffed the four bodies |
| `requires_multidevice` reason strings unified | three markers with the same condition `jax.device_count() < 2` and different reason strings | one marker | applied to the same 6 tests |
| `EXPECTED_BANDS` reformatted | one-line set | multi-line set | same 5 elements |
| 4 comment lines above `_MANUAL_FENCES` dropped | comments only | — | no code |

## 2. Nothing-lost proof

Method: `python -m pytest --collect-only -q -p no:cacheprovider` on the
tier-4b tip and on this tip, twice each — the default lane (addopts
`-m 'not gpu and not slow and not slow_physics'`) and the full inventory
(`-m "gpu or not gpu"`, which also counts the deselected gpu/slow/slow_physics
ids so a marker change could not hide a loss). Node ids normalised to
`basename.py::test[params]` (5478 unique of 5478, no ambiguous basename).
Mapping rule: `old_path::id → new_path::id` from the 39→10 file map, test id
unchanged; verified by script (`scripts/diagnostics/tier3b_consolidation_proof.py`)
that every BEFORE id has exactly one AFTER id present in the AFTER collection
and every AFTER id is mapped from at least one BEFORE id.

| lane | before N | after M | deleted D | moved-to-self-check K | N = M + D + K |
|---|---|---|---|---|---|
| full inventory (`-m "gpu or not gpu"`) | 5478 | 5477 | 1 | 0 | 5478 = 5477 + 1 + 0 ✓ |
| default lane | 5083 | 5082 | 1 | 0 | 5083 = 5082 + 1 + 0 ✓ |

The single D: `tests/contracts/test_gate_policy_is_shared.py::test_margin_ceil_case_imports_shared_multiplier_not_a_local_literal`
is parametrised over `_MARGIN_CEIL_FILES`; its two cases
`[test_waveguide_broad_e5_tolerance_envelope.py]` and
`[test_waveguide_broad_e5_phase_tolerance_envelope.py]` ran the SAME
function (sha256[:16] `4dcb8f89d6e51205`) on two files that are now one, so
they collapsed into the one case `[test_waveguide_broad_e5.py]` (the merged
file imports `ENVELOPE_GATE_MULTIPLIER` from `tests._gate_policy` once and
defines `MARGIN_CEIL` once, which is what the case checks). Not a test of the
merged corpus that was dropped — a contract case re-parametrised over it.
No test was deleted; no referee-header invariant moved (§3).

Full per-node table (every BEFORE id → AFTER id, 5478 rows):
`docs/design_notes/20260903_test_reorg_tier3b_node_map.tsv` (rows whose
file did not change map to themselves). The 402 rows whose node id changed
are listed in §6.

### Assertion counts (`assert`, `pytest.raises(`, `pytest.warns(`, `np.testing.assert_*`, `npt.assert_*` lines)

| merged file | assertions after | absorbed files (assertions before) | sum before |
|---|---|---|---|
| `tests/crossval/test_waveguide_broad_e5.py` | 72 | `test_waveguide_broad_e5_envelope_gates.py` (27), `test_waveguide_broad_e5_live_anchor.py` (7), `test_waveguide_broad_e5_phase_gates.py` (25), `test_waveguide_broad_e5_phase_tolerance_envelope.py` (4), `test_waveguide_broad_e5_tolerance_envelope.py` (9) | 72 |
| `tests/unit/boundaries/test_cpml_material_aware.py` | 35 | `test_nonuniform_cpml_dielectric.py` (3), `test_distributed_cpml_dielectric.py` (5), `test_distributed_nu_cpml_dielectric.py` (4), `test_distributed_pmap_cpml_dielectric.py` (6), `test_vmap_cpml_dielectric.py` (5), `test_lumped_wire_sparam_cpml_dielectric.py` (7), `test_subgrid_cpml_dielectric.py` (5) | 35 |
| `tests/unit/materials/test_sheet_impedance.py` | 142 | `test_leontovich_sheet_identity.py` (17), `test_sheet_impedance_operator.py` (31), `test_sheet_lane_fences.py` (9), `test_sheet_stacked_adjacent_gap.py` (22), `test_thin_conductor_nonbox_sheet.py` (63) | 142 |
| `tests/unit/preflight/test_preflight_absorber.py` | 91 | `test_preflight_absorber_frame.py` (40), `test_preflight_dispersive_pole_at_absorber.py` (28), `test_preflight_geometry_absorber_aggregation.py` (23) | 91 |
| `tests/unit/preflight/test_preflight_guards.py` | 88 | `test_preflight_false_positives.py` (10), `test_preflight_physics_thresholds.py` (15), `test_preflight_structured_and_guards.py` (60), `test_preflight_tfsf_lumped.py` (3) | 88 |
| `tests/unit/preflight/test_preflight_rasterization.py` | 90 | `test_preflight_campaign_statics.py` (78), `test_preflight_graded_rasterization.py` (10), `test_preflight_thin_metal_nu.py` (2) | 90 |
| `tests/unit/sparams/test_coax_two_port_smatrix.py` | 95 | `test_coax_two_port_fdtd.py` (62), `test_coax_two_port_solve.py` (33) | 95 |
| `tests/unit/sparams/test_coaxial_line_reflection.py` | 56 | `test_coaxial_line_calibration.py` (36), `test_coaxial_line_extraction.py` (20) | 56 |
| `tests/unit/sparams/test_settling_witness.py` | 51 | `test_msl_settling_witness.py` (14), `test_settling_witness_enforcement.py` (29), `test_waveguide_settling_witness.py` (8) | 51 |
| `tests/unit/subgrid/test_sbp_sat.py` | 72 | `test_sbp_sat_1d.py` (14), `test_sbp_sat_2d.py` (7), `test_sbp_sat_3d.py` (7), `test_sbp_sat_alpha.py` (15), `test_sbp_sat_jit.py` (29) | 72 |

After == before for every merged file (no documented identical-duplicate
test was merged, so equality is the expected result). Two duplicates were
found to DISAGREE on nothing: no pair of absorbed tests asserted the same
quantity with different tolerances, so no tolerance had to be chosen.

### `.test_durations`

Keys before 3215, after 3214; 260 keys rewritten to their new node id by the
file map, plus the one collapsed contract case
`[test_waveguide_broad_e5_tolerance_envelope.py]` → `[test_waveguide_broad_e5.py]`.
Its sibling key `[test_waveguide_broad_e5_phase_tolerance_envelope.py]` was
first kept as-is, which left a key matching no collected node id (review
finding 2, §7); it is now removed, and the proof script counts that removal
as the only permitted drop (`collapsed_removed=1`). Values
byte-identical. Insertion order preserved (the file is not sorted; the
rewrite touches only the renamed lines). Keys that match no collected id:
33 before (pre-existing, incl. 6 top-level `test_issue325_*` /
`test_issue80_*` keys from before tier 4b), 34 after (the kept sibling).
The five `#677` sheet files had no duration keys.

### References

Every `tests/<tier>/<dir>/test_x.py` (slash form, with or without `.py`) and
`tests.<tier>.<dir>.test_x` (dotted form) of the 39 absorbed files was
rewritten repo-wide by script (`git ls-files`, excluding `CHANGELOG.md`,
`.test_durations` — handled separately — and the frozen run-record dirs
`_*_logs/` / `_*_results/`), same policy as tiers 1-4b. Files touched:

* `.github/workflows/pr-tests.yml` (FILES list, 3 entries → `test_preflight_guards.py`, de-duplicated);
* `validation/crossval/manifest.json` (gate_paths of the waveguide broad-E5 case);
* `scripts/diagnostics/run_physics_gate.py` (`slow_sbp_sat` group, 2 entries → `test_sbp_sat.py`, de-duplicated),
  `scripts/ops/gpu_suite_shards.json` (basenames `test_sbp_sat_1d.py`/`_2d.py` → `test_sbp_sat.py`),
  `scripts/archive/vessl_phase_c_validation.yaml`,
  `scripts/diagnostics/{build_waveguide_band_broad_e5_envelope,build_waveguide_band_broad_e5_phase_envelope,check_port_external_references,waveguide_chain_battery_measure,coax_line_broad_e5_envelope,coax_msl_flux_adjudication,coax_msl_transition_settled_run,patch_edgefed_s11_band_repin}.py`;
* `tests/contracts/test_gate_policy_is_shared.py` (`_MARGIN_CEIL_FILES`: 2 path entries → 1, the collapsed case above),
  `tests/_wave_convention.py`, `tests/crossval/test_{coax_broad_e4_comparison_gates,coax_broad_e5_envelope_gates,coax_two_port_referee_header,waveguide_nu_broad_e4_comparison_gates,waveguide_nu_broad_e5_envelope_gates}.py`,
  `tests/unit/{autodiff/test_coax_two_port_ad,runners/test_runner_import_binding,runners/test_run_progress_reporting,sparams/test_coax_msl_transition,sparams/test_coax_msl_transition_wave_roles,sparams/test_coaxial_s_matrix,farfield/test_farfield_asymmetric_cpml,farfield/test_ntff_smatrix_drop_warning,preflight/test_run_preflight_parity,materials/test_thin_conductor}.py`,
  `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`, `tests/fixtures/waveguide_chain_battery/fixture.json`;
* `rfx/api/{_sparams,_spec,_preflight}.py`, `rfx/sources/coaxial_port.py`, `rfx/boundaries/pec.py`,
  `rfx/materials/thin_conductor.py`, `rfx/runners/nonuniform.py`, `rfx/topology.py` (docstring path pointers only;
  line counts of all eight files unchanged, checked against `3bafe2f`, so `file.py:LINE` evidence pointers keep holding);
* `validation/crossval/21_coax_two_port_referee.py`; `docs/design_notes/*` and `docs/guides/sparameter_support_matrix.{md,json}`.

`tests/_example_fidelity_lib.py` `CLASSIFICATION` keys are example/validation
script paths, not test paths — nothing to update (verified by grep). Bare
basenames inside prose (`docs/design_notes/20260902_test_reorg_tier4b_plan.md`
tables, `tests/_gate_policy.py` docstring) are left as history, as in tiers 1-4b.

Contract / lock gates on this tip (`python -m pytest -q -p no:cacheprovider`):
`tests/contracts/test_example_fidelity_contract.py`, `tests/contracts/test_evidence_citation_pointers.py`,
`tests/locks/test_lock_provenance_gate.py`, `tests/contracts/test_crossval_manifest_contract.py`,
`tests/contracts/test_gate_policy_is_shared.py` — 235 passed.

ruff exactly as CI (`ruff check rfx/ tests/ validation/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402`): All checks passed.

Full default suite: see §5 (appended after the run).

## 3. Left unmerged / kept, and why

* **Referee-header invariants → `--self-check`: not moved.** Neither owning
  script has a `--self-check` path: `validation/crossval/21_coax_two_port_referee.py`
  (argparse: `--output --sim-root --threads --nrts --end-criteria --use-pml --skip-stage-b --dx-scale`)
  nor `validation/crossval/20_msl_phase_referee.py`; the "self-checks" both
  scripts mention are their Stage B physics gates, which need openEMS on
  VESSL. The only `--self-check` in the tree is
  `scripts/diagnostics/probe_fed_msl_openems_referee.py`, and
  `tests/unit/sparams/test_probe_fed_msl_referee_contract.py::test_self_check_cli_exits_zero`
  already is the thin test that invokes it. `tests/crossval/test_coax_two_port_referee_header.py`
  (35 tests) and `tests/crossval/test_msl_phase_referee_header.py` (52 tests)
  therefore stay as they are; adding a `--self-check` mode to two 2,200-line
  referee scripts is a script change, out of this tier's scope (follow-up).
* **`tests/unit/misc/test_ris.py`: kept.** The API still exists and is
  exported: `rfx/ris.py`, `from rfx.ris import RISUnitCell, RISSweepResult`
  in `rfx/__init__.py:204`. The file is skip-marked as a whole
  (`pytestmark = pytest.mark.skip(reason="RIS workflow deprecated — Floquet port redesign needed")`)
  and is the only test of `rfx.ris`; deleting it would leave an exported API
  with zero tests. Kept as-is (12 skipped ids).
* **`tests/unit/sparams/test_coaxial_s_matrix.py`: kept.** `compute_coaxial_s_matrix`
  still exists (`rfx/api/_sparams.py:5342`, deprecation warning at :5413) and is
  used by `scripts/diagnostics/{coax_vi_zsweep_diag,build_coaxial_openems_calibration_fixture,coax_termination_gates}.py`
  and by 8 other tests (`test_ad_surface_contract`, `test_support_matrix_parity`,
  `test_preflight_advisory_emission_contract`, `test_sparam_passivity_guard`, …).
  Its 17 tests are the only plumbing guards on that path; its docstring
  forbids wiring physics gates onto it, which is also why it was NOT folded
  into `test_coax_two_port_smatrix.py`.
* **The visualize3d "duplicate": not a duplicate.** `tests/unit/api/test_visualize3d.py`
  tests `rfx.visualize3d` (`plot_geometry_3d`, `plot_field_3d`, `save_field_vtk`,
  `save_screenshot`; matplotlib); `tests/unit/api/test_visualize_3d.py` tests
  `rfx.visualize` (`visualize_structure`, `visualize_farfield_3d`; plotly,
  issue #38). `comm` shows only 4 trivially shared lines (`import pytest`,
  `return sim`, two closing parens); both modules exist. Both kept.
* **coax: 7 → 5, not 2.** `tests/unit/autodiff/test_coax_two_port_ad.py` stays
  in `autodiff` (gradient question → autodiff, tier-4b rule) and
  `tests/unit/ports/test_coaxial_port.py` stays in `ports` (port primitive →
  ports); both would have had to cross tier directories to reach 2. The
  deprecated file is kept per the previous bullet.
* **preflight: 11 → 4.** `test_preflight_advisory_emission_contract.py`
  (#737/#742, frozen emission-site counts + per-entry-point classification)
  is a source-structure contract, not a stage; kept alone.
* **#677 sheet: 10 → 6.** The five `tests/unit/materials` files merged. The
  two locks (`tests/locks/test_sheet_refactor_bit_identity.py`,
  `tests/locks/test_sheet_resonance_position_ab.py`) carry distinct
  `LOCK_PROVENANCE` records and `tests/locks/test_lock_provenance_gate.py`
  requires exactly one module-level dict literal per lock module — merging
  them would drop a provenance record. `tests/oracle/test_sheet_perturbation_q.py`
  (oracle), `tests/unit/autodiff/test_ad_memory_grid_and_sheet.py` (AD
  memory) and `tests/unit/sparams/test_msl_sheet_threading.py` (S-matrix
  lane) live in their own tiers by the tier-4b rules.
* **patch: 5 → 5 (unmerged).** `tests/locks/test_patch_edgefed_resonance_harminv.py`
  ("Board H", 43×51 raster, `LOCK_PROVENANCE` commit a8c3d52) and
  `tests/locks/test_patch_edgefed_s11_passivity.py` ("Board S", 44×51 raster,
  commit c7527cb) are deliberately two boards (#782 one-mesh anchor rule:
  "mixing dimensions or anchors across the two boards describes a board that
  exists on no mesh") with two provenance records the lock gate would not
  accept in one module. `tests/crossval/test_patch_canonical_farfield_e4.py`
  (crossval cv05 far-field), `tests/oracle/test_patch_cavity_eps_oracle.py`
  (Kottke ε-interface oracle) and `tests/unit/nonuniform/test_patch_uniform_fine_substrate.py`
  (#325 grading lock) are three different subjects in three tiers. Nothing
  could be merged without either losing a provenance record or crossing the
  tier layout; recorded rather than forced.

## 4. How to reproduce the proof

```
git worktree add /tmp/before 3bafe2f && cd /tmp/before
python -m pytest --collect-only -q -p no:cacheprovider -m "gpu or not gpu" > before_all.txt
python -m pytest --collect-only -q -p no:cacheprovider > before_default.txt
cd <this tip>
python -m pytest --collect-only -q -p no:cacheprovider -m "gpu or not gpu" > after_all.txt
python -m pytest --collect-only -q -p no:cacheprovider > after_default.txt
python scripts/diagnostics/tier3b_consolidation_proof.py --base 3bafe2f \
    --before-all before_all.txt --after-all after_all.txt \
    --before-default before_default.txt --after-default after_default.txt
```

## 6. Node ids that changed (402 of 5478; the other 5076 map to themselves)

Normalised `basename.py::test[params]`; produced by the proof script from the two full-inventory collections.

| BEFORE node id | AFTER node id |
|---|---|
| `test_gate_policy_is_shared.py::test_margin_ceil_case_imports_shared_multiplier_not_a_local_literal[test_waveguide_broad_e5_tolerance_envelope.py]` | `test_gate_policy_is_shared.py::test_margin_ceil_case_imports_shared_multiplier_not_a_local_literal[test_waveguide_broad_e5.py]` |
| `test_gate_policy_is_shared.py::test_margin_ceil_case_imports_shared_multiplier_not_a_local_literal[test_waveguide_broad_e5_phase_tolerance_envelope.py]` | `test_gate_policy_is_shared.py::test_margin_ceil_case_imports_shared_multiplier_not_a_local_literal[test_waveguide_broad_e5.py]` |
| `test_waveguide_broad_e5_envelope_gates.py::test_all_five_bands_present` | `test_waveguide_broad_e5.py::test_all_five_bands_present` |
| `test_waveguide_broad_e5_envelope_gates.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr10_wband_broad_e5_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr10_wband_broad_e5_envelope]` |
| `test_waveguide_broad_e5_envelope_gates.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr15_vband_broad_e5_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr15_vband_broad_e5_envelope]` |
| `test_waveguide_broad_e5_envelope_gates.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr28_kaband_broad_e5_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr28_kaband_broad_e5_envelope]` |
| `test_waveguide_broad_e5_envelope_gates.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr340_sband_broad_e5_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr340_sband_broad_e5_envelope]` |
| `test_waveguide_broad_e5_envelope_gates.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr62_kuband_broad_e5_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_envelope_passes_broad_e5[waveguide_wr62_kuband_broad_e5_envelope]` |
| `test_waveguide_broad_e5_envelope_gates.py::test_gate_passes_when_rfx_equals_airy` | `test_waveguide_broad_e5.py::test_gate_passes_when_rfx_equals_airy` |
| `test_waveguide_broad_e5_envelope_gates.py::test_gate_fails_on_magnitude_perturbation` | `test_waveguide_broad_e5.py::test_gate_fails_on_magnitude_perturbation` |
| `test_waveguide_broad_e5_envelope_gates.py::test_lossless_slab_airy_is_unitary` | `test_waveguide_broad_e5.py::test_lossless_slab_airy_is_unitary` |
| `test_waveguide_broad_e5_envelope_gates.py::test_broad_e4_comparison_committed_passes` | `test_waveguide_broad_e5.py::test_broad_e4_comparison_committed_passes` |
| `test_waveguide_broad_e5_envelope_gates.py::test_broad_e4_comparison_qualifies_for_auditor` | `test_waveguide_broad_e5.py::test_broad_e4_comparison_qualifies_for_auditor` |
| `test_waveguide_broad_e5_live_anchor.py::test_live_pec_short_s11_anchor` | `test_waveguide_broad_e5.py::test_live_pec_short_s11_anchor` |
| `test_waveguide_broad_e5_live_anchor.py::test_live_empty_guide_s21_anchor` | `test_waveguide_broad_e5.py::test_live_empty_guide_s21_anchor` |
| `test_waveguide_broad_e5_phase_gates.py::test_all_five_bands_present_phase` | `test_waveguide_broad_e5.py::test_all_five_bands_present_phase` |
| `test_waveguide_broad_e5_phase_gates.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr10_wband_broad_e5_phase_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr10_wband_broad_e5_phase_envelope]` |
| `test_waveguide_broad_e5_phase_gates.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr15_vband_broad_e5_phase_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr15_vband_broad_e5_phase_envelope]` |
| `test_waveguide_broad_e5_phase_gates.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr28_kaband_broad_e5_phase_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr28_kaband_broad_e5_phase_envelope]` |
| `test_waveguide_broad_e5_phase_gates.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr340_sband_broad_e5_phase_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr340_sband_broad_e5_phase_envelope]` |
| `test_waveguide_broad_e5_phase_gates.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr62_kuband_broad_e5_phase_envelope]` | `test_waveguide_broad_e5.py::test_committed_band_phase_envelope_passes_broad_e5[waveguide_wr62_kuband_broad_e5_phase_envelope]` |
| `test_waveguide_broad_e5_phase_gates.py::test_phase_gate_passes_when_rfx_equals_airy` | `test_waveguide_broad_e5.py::test_phase_gate_passes_when_rfx_equals_airy` |
| `test_waveguide_broad_e5_phase_gates.py::test_phase_gate_fails_on_20_degree_perturbation` | `test_waveguide_broad_e5.py::test_phase_gate_fails_on_20_degree_perturbation` |
| `test_waveguide_broad_e5_phase_gates.py::test_phase_reference_is_unitary_consistent_with_magnitude` | `test_waveguide_broad_e5.py::test_phase_reference_is_unitary_consistent_with_magnitude` |
| `test_waveguide_broad_e5_phase_gates.py::test_falsifier_json_exists_and_reds_the_gate` | `test_waveguide_broad_e5.py::test_falsifier_json_exists_and_reds_the_gate` |
| `test_waveguide_broad_e5_phase_gates.py::test_domain_invariance_witness_does_not_flip_verdict` | `test_waveguide_broad_e5.py::test_domain_invariance_witness_does_not_flip_verdict` |
| `test_waveguide_broad_e5_phase_tolerance_envelope.py::test_max_phase_tol_is_a_bounded_measured_envelope` | `test_waveguide_broad_e5.py::test_max_phase_tol_is_a_bounded_measured_envelope` |
| `test_waveguide_broad_e5_phase_tolerance_envelope.py::test_phase_residual_is_far_from_the_old_convention_masking_scale` | `test_waveguide_broad_e5.py::test_phase_residual_is_far_from_the_old_convention_masking_scale` |
| `test_waveguide_broad_e5_tolerance_envelope.py::test_max_tol_is_a_bounded_measured_envelope` | `test_waveguide_broad_e5.py::test_max_tol_is_a_bounded_measured_envelope` |
| `test_waveguide_broad_e5_tolerance_envelope.py::test_noise_floor_is_committed_and_verifiable` | `test_waveguide_broad_e5.py::test_noise_floor_is_committed_and_verifiable` |
| `test_waveguide_broad_e5_tolerance_envelope.py::test_dispersion_tolerance_model_stays_falsified` | `test_waveguide_broad_e5.py::test_dispersion_tolerance_model_stays_falsified` |
| `test_leontovich_sheet_identity.py::test_default_off_identity_and_negative_control_o6` | `test_sheet_impedance.py::test_default_off_identity_and_negative_control_o6` |
| `test_leontovich_sheet_identity.py::test_vmap_parity_o7` | `test_sheet_impedance.py::test_vmap_parity_o7` |
| `test_sheet_impedance_operator.py::test_g3a_huge_rs0_recovers_sheet_absent_fields` | `test_sheet_impedance.py::test_g3a_huge_rs0_recovers_sheet_absent_fields` |
| `test_sheet_impedance_operator.py::test_g3b_tiny_rs0_matches_pec_resonance` | `test_sheet_impedance.py::test_g3b_tiny_rs0_matches_pec_resonance` |
| `test_sheet_impedance_operator.py::test_g3c_x2_algebra_uniform_and_nu_transition` | `test_sheet_impedance.py::test_g3c_x2_algebra_uniform_and_nu_transition` |
| `test_sheet_impedance_operator.py::test_g4_footprint_identity_with_pec_mask` | `test_sheet_impedance.py::test_g4_footprint_identity_with_pec_mask` |
| `test_sheet_impedance_operator.py::test_g4_pec_owned_edges_are_excluded_from_the_ctx` | `test_sheet_impedance.py::test_g4_pec_owned_edges_are_excluded_from_the_ctx` |
| `test_sheet_impedance_operator.py::test_g5_default_off_run_byte_identity` | `test_sheet_impedance.py::test_g5_default_off_run_byte_identity` |
| `test_sheet_impedance_operator.py::test_g8_reference_strip_negative_control` | `test_sheet_impedance.py::test_g8_reference_strip_negative_control` |
| `test_sheet_impedance_operator.py::test_g9_fast_path_excludes_sheets` | `test_sheet_impedance.py::test_g9_fast_path_excludes_sheets` |
| `test_sheet_impedance_operator.py::test_g9_distributed_runners_refuse` | `test_sheet_impedance.py::test_g9_distributed_runners_refuse` |
| `test_sheet_impedance_operator.py::test_g9_refusal_helper_message_names_the_lane` | `test_sheet_impedance.py::test_g9_refusal_helper_message_names_the_lane` |
| `test_sheet_impedance_operator.py::test_g9_crossing_normals_refuse` | `test_sheet_impedance.py::test_g9_crossing_normals_refuse` |
| `test_sheet_impedance_operator.py::test_g9_dispersive_overlap_refuses_uniform_and_nu` | `test_sheet_impedance.py::test_g9_dispersive_overlap_refuses_uniform_and_nu` |
| `test_sheet_impedance_operator.py::test_g9_upml_refuses` | `test_sheet_impedance.py::test_g9_upml_refuses` |
| `test_sheet_lane_fences.py::test_fence_adi_run` | `test_sheet_impedance.py::test_fence_adi_run` |
| `test_sheet_lane_fences.py::test_fence_adi_forward` | `test_sheet_impedance.py::test_fence_adi_forward` |
| `test_sheet_lane_fences.py::test_fence_subgridded_run` | `test_sheet_impedance.py::test_fence_subgridded_run` |
| `test_sheet_lane_fences.py::test_fence_distributed_multidevice_run` | `test_sheet_impedance.py::test_fence_distributed_multidevice_run` |
| `test_sheet_lane_fences.py::test_fence_distributed_nonuniform_forward` | `test_sheet_impedance.py::test_fence_distributed_nonuniform_forward` |
| `test_sheet_lane_fences.py::test_fence_msl_junction_mixed_s_matrix` | `test_sheet_impedance.py::test_fence_msl_junction_mixed_s_matrix` |
| `test_sheet_lane_fences.py::test_fence_optimize_uniform` | `test_sheet_impedance.py::test_fence_optimize_uniform` |
| `test_sheet_lane_fences.py::test_fence_optimize_nonuniform` | `test_sheet_impedance.py::test_fence_optimize_nonuniform` |
| `test_sheet_lane_fences.py::test_fence_gradient_check` | `test_sheet_impedance.py::test_fence_gradient_check` |
| `test_sheet_lane_fences.py::test_fence_topology_optimize` | `test_sheet_impedance.py::test_fence_topology_optimize` |
| `test_sheet_lane_fences.py::test_fence_differentiable_material_fit` | `test_sheet_impedance.py::test_fence_differentiable_material_fit` |
| `test_sheet_lane_fences.py::test_fence_uniform_run_subpixel_or_conformal[kw0]` | `test_sheet_impedance.py::test_fence_uniform_run_subpixel_or_conformal[kw0]` |
| `test_sheet_lane_fences.py::test_fence_uniform_run_subpixel_or_conformal[kw1]` | `test_sheet_impedance.py::test_fence_uniform_run_subpixel_or_conformal[kw1]` |
| `test_sheet_lane_fences.py::test_fence_nonuniform_run_anisotropic_eps` | `test_sheet_impedance.py::test_fence_nonuniform_run_anisotropic_eps` |
| `test_sheet_lane_fences.py::test_fence_forward_upml` | `test_sheet_impedance.py::test_fence_forward_upml` |
| `test_sheet_lane_fences.py::test_fence_forward_dispersive_overlap` | `test_sheet_impedance.py::test_fence_forward_dispersive_overlap` |
| `test_sheet_lane_fences.py::test_fence_waveguide_s_matrix_subpixel` | `test_sheet_impedance.py::test_fence_waveguide_s_matrix_subpixel` |
| `test_sheet_lane_fences.py::test_fence_waveguide_s_matrix_multimode` | `test_sheet_impedance.py::test_fence_waveguide_s_matrix_multimode` |
| `test_sheet_lane_fences.py::test_vmap_sweep_fallback_still_applies_the_sheet` | `test_sheet_impedance.py::test_vmap_sweep_fallback_still_applies_the_sheet` |
| `test_sheet_lane_fences.py::test_every_fence_in_the_source_is_pinned` | `test_sheet_impedance.py::test_every_fence_in_the_source_is_pinned` |
| `test_sheet_lane_fences.py::test_every_registered_pinning_test_exists` | `test_sheet_impedance.py::test_every_registered_pinning_test_exists` |
| `test_sheet_stacked_adjacent_gap.py::test_adjacent_same_normal_sheets_leave_the_gap_edge_unloaded` | `test_sheet_impedance.py::test_adjacent_same_normal_sheets_leave_the_gap_edge_unloaded` |
| `test_sheet_stacked_adjacent_gap.py::test_deeper_same_normal_stack_leaves_every_gap_edge_unloaded` | `test_sheet_impedance.py::test_deeper_same_normal_stack_leaves_every_gap_edge_unloaded` |
| `test_sheet_stacked_adjacent_gap.py::test_non_adjacent_stack_is_unchanged_negative_control` | `test_sheet_impedance.py::test_non_adjacent_stack_is_unchanged_negative_control` |
| `test_sheet_stacked_adjacent_gap.py::test_single_sheet_ctx_is_unchanged_negative_control` | `test_sheet_impedance.py::test_single_sheet_ctx_is_unchanged_negative_control` |
| `test_sheet_stacked_adjacent_gap.py::test_coincident_same_normal_sheets_still_add_conductance` | `test_sheet_impedance.py::test_coincident_same_normal_sheets_still_add_conductance` |
| `test_sheet_stacked_adjacent_gap.py::test_coplanar_abutting_one_cell_strips_keep_both_in_plane_components` | `test_sheet_impedance.py::test_coplanar_abutting_one_cell_strips_keep_both_in_plane_components` |
| `test_sheet_stacked_adjacent_gap.py::test_gap_between_stacked_films_rings_instead_of_being_clamped` | `test_sheet_impedance.py::test_gap_between_stacked_films_rings_instead_of_being_clamped` |
| `test_sheet_stacked_adjacent_gap.py::test_two_d_sheet_keeps_its_out_of_plane_component` | `test_sheet_impedance.py::test_two_d_sheet_keeps_its_out_of_plane_component` |
| `test_sheet_stacked_adjacent_gap.py::test_two_d_sheet_keeps_the_g4_pec_footprint_identity` | `test_sheet_impedance.py::test_two_d_sheet_keeps_the_g4_pec_footprint_identity` |
| `test_sheet_stacked_adjacent_gap.py::test_periodic_seam_gap_edge_stays_vetoed` | `test_sheet_impedance.py::test_periodic_seam_gap_edge_stays_vetoed` |
| `test_sheet_stacked_adjacent_gap.py::test_two_d_f0_patch_shadows_like_pec_end_to_end` | `test_sheet_impedance.py::test_two_d_f0_patch_shadows_like_pec_end_to_end` |
| `test_thin_conductor_nonbox_sheet.py::test_box_and_equivalent_mask_shape_fold_bit_identically[uniform]` | `test_sheet_impedance.py::test_box_and_equivalent_mask_shape_fold_bit_identically[uniform]` |
| `test_thin_conductor_nonbox_sheet.py::test_box_and_equivalent_mask_shape_fold_bit_identically[nonuniform]` | `test_sheet_impedance.py::test_box_and_equivalent_mask_shape_fold_bit_identically[nonuniform]` |
| `test_thin_conductor_nonbox_sheet.py::test_dc_fold_also_accepts_a_mask_shape_on_the_uniform_lane` | `test_sheet_impedance.py::test_dc_fold_also_accepts_a_mask_shape_on_the_uniform_lane` |
| `test_thin_conductor_nonbox_sheet.py::test_patterned_sheet_folds_only_occupied_cells[uniform]` | `test_sheet_impedance.py::test_patterned_sheet_folds_only_occupied_cells[uniform]` |
| `test_thin_conductor_nonbox_sheet.py::test_patterned_sheet_folds_only_occupied_cells[nonuniform]` | `test_sheet_impedance.py::test_patterned_sheet_folds_only_occupied_cells[nonuniform]` |
| `test_thin_conductor_nonbox_sheet.py::test_body_with_height_refused_on_both_lanes` | `test_sheet_impedance.py::test_body_with_height_refused_on_both_lanes` |
| `test_thin_conductor_nonbox_sheet.py::test_sheet_that_rasterizes_to_nothing_is_refused_not_vaporized` | `test_sheet_impedance.py::test_sheet_that_rasterizes_to_nothing_is_refused_not_vaporized` |
| `test_thin_conductor_nonbox_sheet.py::test_shape_without_a_mask_or_bounds_is_refused_at_add_time` | `test_sheet_impedance.py::test_shape_without_a_mask_or_bounds_is_refused_at_add_time` |
| `test_thin_conductor_nonbox_sheet.py::test_nu_defensive_refusal_for_a_bounds_less_sheet` | `test_sheet_impedance.py::test_nu_defensive_refusal_for_a_bounds_less_sheet` |
| `test_thin_conductor_nonbox_sheet.py::test_graded_node_advisory_follows_a_nonbox_sheet` | `test_sheet_impedance.py::test_graded_node_advisory_follows_a_nonbox_sheet` |
| `test_thin_conductor_nonbox_sheet.py::test_design_ir_records_a_registered_nonbox_sheet_and_refuses_the_rest` | `test_sheet_impedance.py::test_design_ir_records_a_registered_nonbox_sheet_and_refuses_the_rest` |
| `test_thin_conductor_nonbox_sheet.py::test_mesh_shape_sheet_folds_bit_identically_to_its_box` | `test_sheet_impedance.py::test_mesh_shape_sheet_folds_bit_identically_to_its_box` |
| `test_thin_conductor_nonbox_sheet.py::test_mesh_shape_patterned_sheet_leaves_its_clearance_hole_alone` | `test_sheet_impedance.py::test_mesh_shape_patterned_sheet_leaves_its_clearance_hole_alone` |
| `test_thin_conductor_nonbox_sheet.py::test_alpha_invariance_transfers_to_a_nonbox_sheet[a_step]` | `test_sheet_impedance.py::test_alpha_invariance_transfers_to_a_nonbox_sheet[a_step]` |
| `test_thin_conductor_nonbox_sheet.py::test_alpha_invariance_transfers_to_a_nonbox_sheet[b_step]` | `test_sheet_impedance.py::test_alpha_invariance_transfers_to_a_nonbox_sheet[b_step]` |
| `test_thin_conductor_nonbox_sheet.py::test_occupancy_guard_does_not_break_the_traced_mesh_path` | `test_sheet_impedance.py::test_occupancy_guard_does_not_break_the_traced_mesh_path` |
| `test_thin_conductor_nonbox_sheet.py::test_vmap_batched_build_folds_a_nonbox_sheet_identically` | `test_sheet_impedance.py::test_vmap_batched_build_folds_a_nonbox_sheet_identically` |
| `test_nonuniform_cpml_dielectric.py::test_nonuniform_cpml_dielectric_stable_and_absorbing` | `test_cpml_material_aware.py::test_nonuniform_cpml_dielectric_stable_and_absorbing` |
| `test_preflight_absorber_frame.py::test_uniform_grid_pads_absorber_exterior_to_requested_domain` | `test_preflight_absorber.py::test_uniform_grid_pads_absorber_exterior_to_requested_domain` |
| `test_preflight_absorber_frame.py::test_nonuniform_grid_pads_absorber_exterior_to_requested_domain` | `test_preflight_absorber.py::test_nonuniform_grid_pads_absorber_exterior_to_requested_domain` |
| `test_preflight_absorber_frame.py::test_absorber_boundary_helper_matches_ground_truth` | `test_preflight_absorber.py::test_absorber_boundary_helper_matches_ground_truth` |
| `test_preflight_absorber_frame.py::test_last_interior_node_reads_as_overlap_not_proximity_h1_conservatism` | `test_preflight_absorber.py::test_last_interior_node_reads_as_overlap_not_proximity_h1_conservatism` |
| `test_preflight_absorber_frame.py::test_absorber_placement_silent_on_domain_centre_probe` | `test_preflight_absorber.py::test_absorber_placement_silent_on_domain_centre_probe` |
| `test_preflight_absorber_frame.py::test_absorber_placement_fires_on_probe_genuinely_in_absorber` | `test_preflight_absorber.py::test_absorber_placement_fires_on_probe_genuinely_in_absorber` |
| `test_preflight_absorber_frame.py::test_absorber_placement_proximity_advisory_fires_within_2_cells` | `test_preflight_absorber.py::test_absorber_placement_proximity_advisory_fires_within_2_cells` |
| `test_preflight_absorber_frame.py::test_absorber_placement_silent_past_the_proximity_margin` | `test_preflight_absorber.py::test_absorber_placement_silent_past_the_proximity_margin` |
| `test_preflight_absorber_frame.py::test_geometry_in_cpml_silent_when_entirely_interior` | `test_preflight_absorber.py::test_geometry_in_cpml_silent_when_entirely_interior` |
| `test_preflight_absorber_frame.py::test_geometry_in_cpml_fires_when_bbox_crosses_domain_edge` | `test_preflight_absorber.py::test_geometry_in_cpml_fires_when_bbox_crosses_domain_edge` |
| `test_preflight_absorber_frame.py::test_ntff_absorber_overlap_silent_when_box_interior` | `test_preflight_absorber.py::test_ntff_absorber_overlap_silent_when_box_interior` |
| `test_preflight_absorber_frame.py::test_ntff_absorber_overlap_fires_when_corner_crosses_domain_edge` | `test_preflight_absorber.py::test_ntff_absorber_overlap_fires_when_corner_crosses_domain_edge` |
| `test_preflight_absorber_frame.py::test_waveguide_reference_plane_silent_on_wr90_ports_in_valid_domain` | `test_preflight_absorber.py::test_waveguide_reference_plane_silent_on_wr90_ports_in_valid_domain` |
| `test_preflight_absorber_frame.py::test_waveguide_reference_plane_absorber_branch_is_dead_given_hard_check` | `test_preflight_absorber.py::test_waveguide_reference_plane_absorber_branch_is_dead_given_hard_check` |
| `test_preflight_absorber_frame.py::test_waveguide_reference_plane_silent_at_mixin_level_near_edge` | `test_preflight_absorber.py::test_waveguide_reference_plane_silent_at_mixin_level_near_edge` |
| `test_preflight_absorber_frame.py::test_msl_x_cpml_clearance_fires_on_ledger_negative_clearance_case` | `test_preflight_absorber.py::test_msl_x_cpml_clearance_fires_on_ledger_negative_clearance_case` |
| `test_preflight_absorber_frame.py::test_msl_x_cpml_clearance_silent_once_past_buffer_plus_recommended` | `test_preflight_absorber.py::test_msl_x_cpml_clearance_silent_once_past_buffer_plus_recommended` |
| `test_preflight_absorber_frame.py::test_msl_x_cpml_clearance_fires_when_genuinely_too_close` | `test_preflight_absorber.py::test_msl_x_cpml_clearance_fires_when_genuinely_too_close` |
| `test_preflight_absorber_frame.py::test_msl_y_clearance_fires_on_ledger_ly_w_plus_6dx` | `test_preflight_absorber.py::test_msl_y_clearance_fires_on_ledger_ly_w_plus_6dx` |
| `test_preflight_absorber_frame.py::test_msl_y_clearance_silent_on_ledger_calibrated_ly` | `test_preflight_absorber.py::test_msl_y_clearance_silent_on_ledger_calibrated_ly` |
| `test_preflight_absorber_frame.py::test_msl_clearance_buffer_scales_with_cpml_layers_not_hardcoded_8` | `test_preflight_absorber.py::test_msl_clearance_buffer_scales_with_cpml_layers_not_hardcoded_8` |
| `test_preflight_campaign_statics.py::TestCongruenceParity::test_off_lattice_mirror_pair_fires_once_with_basis` | `test_preflight_rasterization.py::TestCongruenceParity::test_off_lattice_mirror_pair_fires_once_with_basis` |
| `test_preflight_campaign_statics.py::TestCongruenceParity::test_on_lattice_mirror_pair_is_silent` | `test_preflight_rasterization.py::TestCongruenceParity::test_on_lattice_mirror_pair_is_silent` |
| `test_preflight_campaign_statics.py::TestCongruenceParity::test_fires_on_nonuniform_lane_too` | `test_preflight_rasterization.py::TestCongruenceParity::test_fires_on_nonuniform_lane_too` |
| `test_preflight_campaign_statics.py::TestCongruenceParity::test_gate_mutation_both_directions` | `test_preflight_rasterization.py::TestCongruenceParity::test_gate_mutation_both_directions` |
| `test_preflight_campaign_statics.py::TestCongruenceParity::test_conductors_without_bounds_are_skipped_and_said_so` | `test_preflight_rasterization.py::TestCongruenceParity::test_conductors_without_bounds_are_skipped_and_said_so` |
| `test_preflight_campaign_statics.py::TestCongruenceParity::test_patterned_sheet_mirror_pair_fires` | `test_preflight_rasterization.py::TestCongruenceParity::test_patterned_sheet_mirror_pair_fires` |
| `test_preflight_campaign_statics.py::TestCongruenceParity::test_patterned_sheet_on_lattice_mirror_pair_is_silent` | `test_preflight_rasterization.py::TestCongruenceParity::test_patterned_sheet_on_lattice_mirror_pair_is_silent` |
| `test_preflight_campaign_statics.py::TestCongruenceParity::test_sheet_pair_gate_mutation_both_directions` | `test_preflight_rasterization.py::TestCongruenceParity::test_sheet_pair_gate_mutation_both_directions` |
| `test_preflight_campaign_statics.py::TestSheetLiveEdgeMaterials::test_post_702_main_is_silent` | `test_preflight_rasterization.py::TestSheetLiveEdgeMaterials::test_post_702_main_is_silent` |
| `test_preflight_campaign_statics.py::TestSheetLiveEdgeMaterials::test_fires_when_the_resample_is_mutated_off` | `test_preflight_rasterization.py::TestSheetLiveEdgeMaterials::test_fires_when_the_resample_is_mutated_off` |
| `test_preflight_campaign_statics.py::TestSheetLiveEdgeMaterials::test_gate_mutation_both_directions` | `test_preflight_rasterization.py::TestSheetLiveEdgeMaterials::test_gate_mutation_both_directions` |
| `test_preflight_campaign_statics.py::TestSheetLiveEdgeMaterials::test_subgrid_fine_region_debt_is_named` | `test_preflight_rasterization.py::TestSheetLiveEdgeMaterials::test_subgrid_fine_region_debt_is_named` |
| `test_preflight_campaign_statics.py::TestSheetLiveEdgeMaterials::test_fires_on_nonuniform_lane_too` | `test_preflight_rasterization.py::TestSheetLiveEdgeMaterials::test_fires_on_nonuniform_lane_too` |
| `test_preflight_campaign_statics.py::TestSheetCavityThickness::test_collapsed_registration_fires_with_both_measures` | `test_preflight_rasterization.py::TestSheetCavityThickness::test_collapsed_registration_fires_with_both_measures` |
| `test_preflight_campaign_statics.py::TestSheetCavityThickness::test_node_registered_thin_sheets_are_silent` | `test_preflight_rasterization.py::TestSheetCavityThickness::test_node_registered_thin_sheets_are_silent` |
| `test_preflight_campaign_statics.py::TestSheetCavityThickness::test_gate_mutation_both_directions` | `test_preflight_rasterization.py::TestSheetCavityThickness::test_gate_mutation_both_directions` |
| `test_preflight_campaign_statics.py::TestSheetCavityThickness::test_face_registered_sheet_cell_is_a_live_edge` | `test_preflight_rasterization.py::TestSheetCavityThickness::test_face_registered_sheet_cell_is_a_live_edge` |
| `test_preflight_campaign_statics.py::TestSheetCavityThickness::test_face_registered_stack_reports_the_live_vacuum_cell` | `test_preflight_rasterization.py::TestSheetCavityThickness::test_face_registered_stack_reports_the_live_vacuum_cell` |
| `test_preflight_campaign_statics.py::TestSheetCavityThickness::test_midplane_stack_names_no_own_cell` | `test_preflight_rasterization.py::TestSheetCavityThickness::test_midplane_stack_names_no_own_cell` |
| `test_preflight_campaign_statics.py::TestSheetCavityThickness::test_upper_face_registration_attributes_nothing` | `test_preflight_rasterization.py::TestSheetCavityThickness::test_upper_face_registration_attributes_nothing` |
| `test_preflight_campaign_statics.py::TestSheetCavityThickness::test_own_cell_attribution_mutation_both_directions` | `test_preflight_rasterization.py::TestSheetCavityThickness::test_own_cell_attribution_mutation_both_directions` |
| `test_preflight_campaign_statics.py::TestOffLatticeCensus::test_off_lattice_edge_fires_with_residual_and_detune` | `test_preflight_rasterization.py::TestOffLatticeCensus::test_off_lattice_edge_fires_with_residual_and_detune` |
| `test_preflight_campaign_statics.py::TestOffLatticeCensus::test_on_lattice_edges_are_silent` | `test_preflight_rasterization.py::TestOffLatticeCensus::test_on_lattice_edges_are_silent` |
| `test_preflight_campaign_statics.py::TestOffLatticeCensus::test_gate_mutation_both_directions` | `test_preflight_rasterization.py::TestOffLatticeCensus::test_gate_mutation_both_directions` |
| `test_preflight_campaign_statics.py::TestOffLatticeCensus::test_offenders_are_aggregated_and_capped` | `test_preflight_rasterization.py::TestOffLatticeCensus::test_offenders_are_aggregated_and_capped` |
| `test_preflight_campaign_statics.py::TestWiring::test_checks_run_inside_validate_simulation_config` | `test_preflight_rasterization.py::TestWiring::test_checks_run_inside_validate_simulation_config` |
| `test_preflight_campaign_statics.py::TestWiring::test_advisory_tier_none_block` | `test_preflight_rasterization.py::TestWiring::test_advisory_tier_none_block` |
| `test_preflight_dispersive_pole_at_absorber.py::test_high_q_lorentz_touching_face_warns` | `test_preflight_absorber.py::test_high_q_lorentz_touching_face_warns` |
| `test_preflight_dispersive_pole_at_absorber.py::test_drude_touching_face_warns` | `test_preflight_absorber.py::test_drude_touching_face_warns` |
| `test_preflight_dispersive_pole_at_absorber.py::test_inset_structure_stays_quiet` | `test_preflight_absorber.py::test_inset_structure_stays_quiet` |
| `test_preflight_dispersive_pole_at_absorber.py::test_low_q_lorentz_touching_face_warns` | `test_preflight_absorber.py::test_low_q_lorentz_touching_face_warns` |
| `test_preflight_dispersive_pole_at_absorber.py::test_out_of_band_high_q_lorentz_touching_face_warns` | `test_preflight_absorber.py::test_out_of_band_high_q_lorentz_touching_face_warns` |
| `test_preflight_dispersive_pole_at_absorber.py::test_debye_touching_face_warns` | `test_preflight_absorber.py::test_debye_touching_face_warns` |
| `test_preflight_dispersive_pole_at_absorber.py::test_overdrawn_hi_face_names_the_realized_pad_truthfully` | `test_preflight_absorber.py::test_overdrawn_hi_face_names_the_realized_pad_truthfully` |
| `test_preflight_dispersive_pole_at_absorber.py::test_pec_boundary_stays_quiet` | `test_preflight_absorber.py::test_pec_boundary_stays_quiet` |
| `test_preflight_dispersive_pole_at_absorber.py::test_two_touching_entries_aggregate_into_one_finding` | `test_preflight_absorber.py::test_two_touching_entries_aggregate_into_one_finding` |
| `test_preflight_false_positives.py::test_thin_pec_strip_with_4_cell_y_silent_on_volume_warning` | `test_preflight_guards.py::test_thin_pec_strip_with_4_cell_y_silent_on_volume_warning` |
| `test_preflight_false_positives.py::test_pec_volume_partial_in_all_axes_still_warns` | `test_preflight_guards.py::test_pec_volume_partial_in_all_axes_still_warns` |
| `test_preflight_false_positives.py::test_full_domain_dielectric_silent_on_cpml_extension` | `test_preflight_guards.py::test_full_domain_dielectric_silent_on_cpml_extension` |
| `test_preflight_false_positives.py::test_inset_box_leaking_into_cpml_still_warns` | `test_preflight_guards.py::test_inset_box_leaking_into_cpml_still_warns` |
| `test_preflight_false_positives.py::test_hy_probe_at_thin_trace_pec_silent_on_inside_pec` | `test_preflight_guards.py::test_hy_probe_at_thin_trace_pec_silent_on_inside_pec` |
| `test_preflight_false_positives.py::test_ez_probe_at_thin_trace_pec_still_warns` | `test_preflight_guards.py::test_ez_probe_at_thin_trace_pec_still_warns` |
| `test_preflight_false_positives.py::test_hy_probe_inside_thick_pec_volume_still_warns` | `test_preflight_guards.py::test_hy_probe_inside_thick_pec_volume_still_warns` |
| `test_preflight_false_positives.py::test_pec_boundary_open_still_warns_when_ntff_declared` | `test_preflight_guards.py::test_pec_boundary_open_still_warns_when_ntff_declared` |
| `test_preflight_false_positives.py::test_pec_cavity_with_internal_pec_object_stays_silent` | `test_preflight_guards.py::test_pec_cavity_with_internal_pec_object_stays_silent` |
| `test_preflight_false_positives.py::test_pec_empty_cavity_with_source_stays_silent` | `test_preflight_guards.py::test_pec_empty_cavity_with_source_stays_silent` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_sixty_one_entries_collapse_to_one_warning` | `test_preflight_absorber.py::test_issue660_sixty_one_entries_collapse_to_one_warning` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_worst_offender_is_the_deepest_not_the_first` | `test_preflight_absorber.py::test_issue660_worst_offender_is_the_deepest_not_the_first` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_loc_carries_per_entry_index_face_and_overshoot` | `test_preflight_absorber.py::test_issue660_loc_carries_per_entry_index_face_and_overshoot` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_separate_axes_get_separate_warnings` | `test_preflight_absorber.py::test_issue660_separate_axes_get_separate_warnings` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_message_states_overshoot_and_crossed_boundary[0.011-11mm past the x-hi absorber boundary at 30mm]` | `test_preflight_absorber.py::test_issue660_message_states_overshoot_and_crossed_boundary[0.011-11mm past the x-hi absorber boundary at 30mm]` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_message_states_overshoot_and_crossed_boundary[0.0005-500\xb5m past the x-hi absorber boundary at 30mm]` | `test_preflight_absorber.py::test_issue660_message_states_overshoot_and_crossed_boundary[0.0005-500\xb5m past the x-hi absorber boundary at 30mm]` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_lo_side_crossing_names_the_lo_boundary` | `test_preflight_absorber.py::test_issue660_lo_side_crossing_names_the_lo_boundary` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_single_shape_overshoot_still_warns` | `test_preflight_absorber.py::test_issue660_single_shape_overshoot_still_warns` |
| `test_preflight_geometry_absorber_aggregation.py::test_issue660_geometry_fully_inside_the_domain_stays_silent` | `test_preflight_absorber.py::test_issue660_geometry_fully_inside_the_domain_stays_silent` |
| `test_preflight_graded_rasterization.py::test_shifted_box_warns_with_actual_and_implied_counts` | `test_preflight_rasterization.py::test_shifted_box_warns_with_actual_and_implied_counts` |
| `test_preflight_graded_rasterization.py::test_box_pinned_to_actual_fine_band_is_silent` | `test_preflight_rasterization.py::test_box_pinned_to_actual_fine_band_is_silent` |
| `test_preflight_graded_rasterization.py::test_uniform_dz_simulation_skips_check` | `test_preflight_rasterization.py::test_uniform_dz_simulation_skips_check` |
| `test_preflight_graded_rasterization.py::test_validator_count_matches_the_real_rasterizer[4.50-5.50mm]` | `test_preflight_rasterization.py::test_validator_count_matches_the_real_rasterizer[4.50-5.50mm]` |
| `test_preflight_graded_rasterization.py::test_validator_count_matches_the_real_rasterizer[5.00-6.00mm]` | `test_preflight_rasterization.py::test_validator_count_matches_the_real_rasterizer[5.00-6.00mm]` |
| `test_preflight_graded_rasterization.py::test_validator_count_matches_the_real_rasterizer[5.00-7.00mm]` | `test_preflight_rasterization.py::test_validator_count_matches_the_real_rasterizer[5.00-7.00mm]` |
| `test_preflight_graded_rasterization.py::test_validator_count_matches_the_real_rasterizer[4.00-5.00mm]` | `test_preflight_rasterization.py::test_validator_count_matches_the_real_rasterizer[4.00-5.00mm]` |
| `test_preflight_graded_rasterization.py::test_validator_count_matches_the_real_rasterizer[5.25-6.75mm]` | `test_preflight_rasterization.py::test_validator_count_matches_the_real_rasterizer[5.25-6.75mm]` |
| `test_preflight_graded_rasterization.py::test_validator_count_matches_the_real_rasterizer[0.00-5.00mm]` | `test_preflight_rasterization.py::test_validator_count_matches_the_real_rasterizer[0.00-5.00mm]` |
| `test_preflight_physics_thresholds.py::test_thin_pec_sheet_is_silent` | `test_preflight_guards.py::test_thin_pec_sheet_is_silent` |
| `test_preflight_physics_thresholds.py::test_partial_pec_volume_warns` | `test_preflight_guards.py::test_partial_pec_volume_warns` |
| `test_preflight_physics_thresholds.py::test_fine_dielectric_is_silent` | `test_preflight_guards.py::test_fine_dielectric_is_silent` |
| `test_preflight_physics_thresholds.py::test_coarse_dielectric_warns` | `test_preflight_guards.py::test_coarse_dielectric_warns` |
| `test_preflight_physics_thresholds.py::test_dielectric_near_old_threshold_now_warns` | `test_preflight_guards.py::test_dielectric_near_old_threshold_now_warns` |
| `test_preflight_physics_thresholds.py::test_compute_waveguide_s_matrix_rejects_unnormalized_nu` | `test_preflight_guards.py::test_compute_waveguide_s_matrix_rejects_unnormalized_nu` |
| `test_preflight_physics_thresholds.py::test_compute_waveguide_s_matrix_dispatches_nu_when_normalized` | `test_preflight_guards.py::test_compute_waveguide_s_matrix_dispatches_nu_when_normalized` |
| `test_preflight_physics_thresholds.py::test_dielectric_sparam_active_raises_threshold_to_20` | `test_preflight_guards.py::test_dielectric_sparam_active_raises_threshold_to_20` |
| `test_preflight_physics_thresholds.py::test_wg_port_evanescent_no_warning_below_threshold` | `test_preflight_guards.py::test_wg_port_evanescent_no_warning_below_threshold` |
| `test_preflight_physics_thresholds.py::test_wg_port_evanescent_warns_above_threshold` | `test_preflight_guards.py::test_wg_port_evanescent_warns_above_threshold` |
| `test_preflight_structured_and_guards.py::test_preflight_returns_back_compatible_structured_issues` | `test_preflight_guards.py::test_preflight_returns_back_compatible_structured_issues` |
| `test_preflight_structured_and_guards.py::test_preflight_issue_is_a_real_string` | `test_preflight_guards.py::test_preflight_issue_is_a_real_string` |
| `test_preflight_structured_and_guards.py::test_preflight_report_is_a_list_with_canonical_api` | `test_preflight_guards.py::test_preflight_report_is_a_list_with_canonical_api` |
| `test_preflight_structured_and_guards.py::test_codes_set_at_check_site` | `test_preflight_guards.py::test_codes_set_at_check_site` |
| `test_preflight_structured_and_guards.py::test_error_severity_mapping_end_to_end` | `test_preflight_guards.py::test_error_severity_mapping_end_to_end` |
| `test_preflight_structured_and_guards.py::test_conformal_fine_dx_warns` | `test_preflight_guards.py::test_conformal_fine_dx_warns` |
| `test_preflight_structured_and_guards.py::test_conformal_coarse_dx_silent` | `test_preflight_guards.py::test_conformal_coarse_dx_silent` |
| `test_preflight_structured_and_guards.py::test_no_conformal_silent` | `test_preflight_guards.py::test_no_conformal_silent` |
| `test_preflight_structured_and_guards.py::test_lossless_dielectric_in_cpml_warns` | `test_preflight_guards.py::test_lossless_dielectric_in_cpml_warns` |
| `test_preflight_structured_and_guards.py::test_lossy_dielectric_silent` | `test_preflight_guards.py::test_lossy_dielectric_silent` |
| `test_preflight_structured_and_guards.py::test_every_emitted_issue_carries_a_check_site_code` | `test_preflight_guards.py::test_every_emitted_issue_carries_a_check_site_code` |
| `test_preflight_structured_and_guards.py::test_error_severity_config_issue_is_coded` | `test_preflight_guards.py::test_error_severity_config_issue_is_coded` |
| `test_preflight_structured_and_guards.py::test_to_dict_and_to_json_roundtrip_carry_code_and_severity` | `test_preflight_guards.py::test_to_dict_and_to_json_roundtrip_carry_code_and_severity` |
| `test_preflight_structured_and_guards.py::test_strict_aggregates_all_issues_in_one_raise` | `test_preflight_guards.py::test_strict_aggregates_all_issues_in_one_raise` |
| `test_preflight_structured_and_guards.py::test_raise_for_failure_is_errors_only_gate` | `test_preflight_guards.py::test_raise_for_failure_is_errors_only_gate` |
| `test_preflight_structured_and_guards.py::test_validator_crash_propagates_not_swallowed` | `test_preflight_guards.py::test_validator_crash_propagates_not_swallowed` |
| `test_preflight_structured_and_guards.py::test_run_uses_ntff_advisory_tier_but_forward_gets_the_error` | `test_preflight_guards.py::test_run_uses_ntff_advisory_tier_but_forward_gets_the_error` |
| `test_preflight_structured_and_guards.py::test_run_hard_fails_on_error_severity_and_skip_bypasses` | `test_preflight_guards.py::test_run_hard_fails_on_error_severity_and_skip_bypasses` |
| `test_preflight_structured_and_guards.py::test_absorber_overlap_no_false_positive_on_2d_collapsed_z` | `test_preflight_guards.py::test_absorber_overlap_no_false_positive_on_2d_collapsed_z` |
| `test_preflight_structured_and_guards.py::test_absorber_overlap_still_fires_on_2d_xy` | `test_preflight_guards.py::test_absorber_overlap_still_fires_on_2d_xy` |
| `test_preflight_structured_and_guards.py::test_unit_adaptive_formatting_helpers` | `test_preflight_guards.py::test_unit_adaptive_formatting_helpers` |
| `test_preflight_structured_and_guards.py::test_mesh_warning_uses_adaptive_units_at_optical_scale` | `test_preflight_guards.py::test_mesh_warning_uses_adaptive_units_at_optical_scale` |
| `test_preflight_tfsf_lumped.py::test_tfsf_plus_lumped_rlc_warns` | `test_preflight_guards.py::test_tfsf_plus_lumped_rlc_warns` |
| `test_preflight_tfsf_lumped.py::test_tfsf_alone_no_warning` | `test_preflight_guards.py::test_tfsf_alone_no_warning` |
| `test_preflight_tfsf_lumped.py::test_lumped_rlc_with_port_no_warning` | `test_preflight_guards.py::test_lumped_rlc_with_port_no_warning` |
| `test_preflight_thin_metal_nu.py::test_asymmetric_metal_on_nu_triggers_warning` | `test_preflight_rasterization.py::test_asymmetric_metal_on_nu_triggers_warning` |
| `test_preflight_thin_metal_nu.py::test_symmetric_metal_on_nu_is_silent` | `test_preflight_rasterization.py::test_symmetric_metal_on_nu_is_silent` |
| `test_distributed_cpml_dielectric.py::test_distributed_cpml_dielectric_finite_and_matches_single` | `test_cpml_material_aware.py::test_distributed_cpml_dielectric_finite_and_matches_single` |
| `test_distributed_cpml_dielectric.py::test_distributed_cpml_responds_to_eps` | `test_cpml_material_aware.py::test_distributed_cpml_responds_to_eps` |
| `test_distributed_nu_cpml_dielectric.py::test_distributed_nu_cpml_dielectric_finite_and_matches_single` | `test_cpml_material_aware.py::test_distributed_nu_cpml_dielectric_finite_and_matches_single` |
| `test_distributed_nu_cpml_dielectric.py::test_distributed_nu_cpml_forward_is_ad_finite` | `test_cpml_material_aware.py::test_distributed_nu_cpml_forward_is_ad_finite` |
| `test_distributed_pmap_cpml_dielectric.py::test_pmap_distributed_cpml_dielectric_finite_and_matches_single` | `test_cpml_material_aware.py::test_pmap_distributed_cpml_dielectric_finite_and_matches_single` |
| `test_distributed_pmap_cpml_dielectric.py::test_pmap_distributed_cpml_responds_to_eps` | `test_cpml_material_aware.py::test_pmap_distributed_cpml_responds_to_eps` |
| `test_vmap_cpml_dielectric.py::test_vmap_cpml_dielectric_is_finite_and_matches_run` | `test_cpml_material_aware.py::test_vmap_cpml_dielectric_is_finite_and_matches_run` |
| `test_vmap_cpml_dielectric.py::test_vmap_cpml_distinct_eps_change_response` | `test_cpml_material_aware.py::test_vmap_cpml_distinct_eps_change_response` |
| `test_coax_two_port_fdtd.py::test_the_planted_dut_signs_are_the_fixture_geometry` | `test_coax_two_port_smatrix.py::test_the_planted_dut_signs_are_the_fixture_geometry` |
| `test_coax_two_port_fdtd.py::test_planted_fields_are_unchanged_by_the_frozen_planting_contract` | `test_coax_two_port_smatrix.py::test_planted_fields_are_unchanged_by_the_frozen_planting_contract` |
| `test_coax_two_port_fdtd.py::test_planted_voltages_recover_known_asymmetric_s_matrix` | `test_coax_two_port_smatrix.py::test_planted_voltages_recover_known_asymmetric_s_matrix` |
| `test_coax_two_port_fdtd.py::test_swapped_ab_convention_fails_the_same_asymmetric_fixture` | `test_coax_two_port_smatrix.py::test_swapped_ab_convention_fails_the_same_asymmetric_fixture` |
| `test_coax_two_port_fdtd.py::test_compute_coaxial_two_port_drive_index_matches_physical_port` | `test_coax_two_port_smatrix.py::test_compute_coaxial_two_port_drive_index_matches_physical_port` |
| `test_coax_two_port_fdtd.py::test_boundary_must_be_cpml` | `test_coax_two_port_smatrix.py::test_boundary_must_be_cpml` |
| `test_coax_two_port_fdtd.py::test_cpml_axes_must_be_z` | `test_coax_two_port_smatrix.py::test_cpml_axes_must_be_z` |
| `test_coax_two_port_fdtd.py::test_periodic_axes_rejected` | `test_coax_two_port_smatrix.py::test_periodic_axes_rejected` |
| `test_coax_two_port_fdtd.py::test_nonuniform_profiles_rejected[dx_profile]` | `test_coax_two_port_smatrix.py::test_nonuniform_profiles_rejected[dx_profile]` |
| `test_coax_two_port_fdtd.py::test_nonuniform_profiles_rejected[dz_profile]` | `test_coax_two_port_smatrix.py::test_nonuniform_profiles_rejected[dz_profile]` |
| `test_coax_two_port_fdtd.py::test_existing_tfsf_rejected` | `test_coax_two_port_smatrix.py::test_existing_tfsf_rejected` |
| `test_coax_two_port_fdtd.py::test_refinement_rejected` | `test_coax_two_port_smatrix.py::test_refinement_rejected` |
| `test_coax_two_port_fdtd.py::test_adi_solver_rejected` | `test_coax_two_port_smatrix.py::test_adi_solver_rejected` |
| `test_coax_two_port_fdtd.py::test_mixed_precision_rejected` | `test_coax_two_port_smatrix.py::test_mixed_precision_rejected` |
| `test_coax_two_port_fdtd.py::test_fourth_order_stencil_rejected` | `test_coax_two_port_smatrix.py::test_fourth_order_stencil_rejected` |
| `test_coax_two_port_fdtd.py::test_two_dimensional_mode_rejected` | `test_coax_two_port_smatrix.py::test_two_dimensional_mode_rejected` |
| `test_coax_two_port_fdtd.py::test_registered_geometry_rejected[geometry]` | `test_coax_two_port_smatrix.py::test_registered_geometry_rejected[geometry]` |
| `test_coax_two_port_fdtd.py::test_registered_geometry_rejected[thin_conductor]` | `test_coax_two_port_smatrix.py::test_registered_geometry_rejected[thin_conductor]` |
| `test_coax_two_port_fdtd.py::test_lumped_rlc_rejected` | `test_coax_two_port_smatrix.py::test_lumped_rlc_rejected` |
| `test_coax_two_port_fdtd.py::test_registered_monitor_rejected[probe]` | `test_coax_two_port_smatrix.py::test_registered_monitor_rejected[probe]` |
| `test_coax_two_port_fdtd.py::test_registered_monitor_rejected[dft]` | `test_coax_two_port_smatrix.py::test_registered_monitor_rejected[dft]` |
| `test_coax_two_port_fdtd.py::test_registered_monitor_rejected[flux]` | `test_coax_two_port_smatrix.py::test_registered_monitor_rejected[flux]` |
| `test_coax_two_port_fdtd.py::test_registered_monitor_rejected[ntff]` | `test_coax_two_port_smatrix.py::test_registered_monitor_rejected[ntff]` |
| `test_coax_two_port_fdtd.py::test_registered_coax_termination_helper_rejected[matched]` | `test_coax_two_port_smatrix.py::test_registered_coax_termination_helper_rejected[matched]` |
| `test_coax_two_port_fdtd.py::test_registered_coax_termination_helper_rejected[open]` | `test_coax_two_port_smatrix.py::test_registered_coax_termination_helper_rejected[open]` |
| `test_coax_two_port_fdtd.py::test_registered_coax_termination_helper_rejected[pec_end_cap]` | `test_coax_two_port_smatrix.py::test_registered_coax_termination_helper_rejected[pec_end_cap]` |
| `test_coax_two_port_fdtd.py::test_at_least_three_probe_planes_required[0]` | `test_coax_two_port_smatrix.py::test_at_least_three_probe_planes_required[0]` |
| `test_coax_two_port_fdtd.py::test_at_least_three_probe_planes_required[1]` | `test_coax_two_port_smatrix.py::test_at_least_three_probe_planes_required[1]` |
| `test_coax_two_port_fdtd.py::test_at_least_three_probe_planes_required[2]` | `test_coax_two_port_smatrix.py::test_at_least_three_probe_planes_required[2]` |
| `test_coax_two_port_fdtd.py::test_probe_count_must_be_an_integer[True]` | `test_coax_two_port_smatrix.py::test_probe_count_must_be_an_integer[True]` |
| `test_coax_two_port_fdtd.py::test_probe_count_must_be_an_integer[3.5]` | `test_coax_two_port_smatrix.py::test_probe_count_must_be_an_integer[3.5]` |
| `test_coax_two_port_fdtd.py::test_no_coaxial_port_rejected` | `test_coax_two_port_smatrix.py::test_no_coaxial_port_rejected` |
| `test_coax_two_port_fdtd.py::test_two_coaxial_ports_rejected` | `test_coax_two_port_smatrix.py::test_two_coaxial_ports_rejected` |
| `test_coax_two_port_fdtd.py::test_non_top_face_rejected` | `test_coax_two_port_smatrix.py::test_non_top_face_rejected` |
| `test_coax_two_port_fdtd.py::test_domain_too_short_for_two_feed_layout_is_rejected` | `test_coax_two_port_smatrix.py::test_domain_too_short_for_two_feed_layout_is_rejected` |
| `test_coax_two_port_fdtd.py::test_overlapping_probe_arrays_are_rejected` | `test_coax_two_port_smatrix.py::test_overlapping_probe_arrays_are_rejected` |
| `test_coax_two_port_fdtd.py::test_default_domain_fits_the_default_layout` | `test_coax_two_port_smatrix.py::test_default_domain_fits_the_default_layout` |
| `test_coax_two_port_fdtd.py::test_passive_result_does_not_warn_non_firing_control` | `test_coax_two_port_smatrix.py::test_passive_result_does_not_warn_non_firing_control` |
| `test_coax_two_port_fdtd.py::test_nonpassive_result_warns_and_is_not_silently_dropped` | `test_coax_two_port_smatrix.py::test_nonpassive_result_warns_and_is_not_silently_dropped` |
| `test_coax_two_port_fdtd.py::test_compute_coaxial_two_port_routes_through_finalize_sparam_result` | `test_coax_two_port_smatrix.py::test_compute_coaxial_two_port_routes_through_finalize_sparam_result` |
| `test_coax_two_port_fdtd.py::test_matched_through_line_transmits_reciprocally` | `test_coax_two_port_smatrix.py::test_matched_through_line_transmits_reciprocally` |
| `test_coax_two_port_solve.py::test_asymmetric_fixture_is_what_it_claims` | `test_coax_two_port_smatrix.py::test_asymmetric_fixture_is_what_it_claims` |
| `test_coax_two_port_solve.py::test_single_ratio_rule_has_a_terminator_floor[0.02]` | `test_coax_two_port_smatrix.py::test_single_ratio_rule_has_a_terminator_floor[0.02]` |
| `test_coax_two_port_solve.py::test_single_ratio_rule_has_a_terminator_floor[0.05]` | `test_coax_two_port_smatrix.py::test_single_ratio_rule_has_a_terminator_floor[0.05]` |
| `test_coax_two_port_solve.py::test_single_ratio_rule_has_a_terminator_floor[0.08]` | `test_coax_two_port_smatrix.py::test_single_ratio_rule_has_a_terminator_floor[0.08]` |
| `test_coax_two_port_solve.py::test_two_drive_solve_removes_the_terminator_floor` | `test_coax_two_port_smatrix.py::test_two_drive_solve_removes_the_terminator_floor` |
| `test_coax_two_port_solve.py::test_conditioning_grows_slowly_with_terminator_reflection` | `test_coax_two_port_smatrix.py::test_conditioning_grows_slowly_with_terminator_reflection` |
| `test_coax_two_port_solve.py::test_recovers_shunt_resistor_exactly_despite_terminator` | `test_coax_two_port_smatrix.py::test_recovers_shunt_resistor_exactly_despite_terminator` |
| `test_coax_two_port_solve.py::test_swapped_receive_amplitude_one_drive_is_caught` | `test_coax_two_port_smatrix.py::test_swapped_receive_amplitude_one_drive_is_caught` |
| `test_coax_two_port_solve.py::test_both_drive_swap_gap_requires_the_downstream_passivity_handle` | `test_coax_two_port_smatrix.py::test_both_drive_swap_gap_requires_the_downstream_passivity_handle` |
| `test_coax_two_port_solve.py::test_reciprocity_catches_a_per_port_amplitude_error[1.05]` | `test_coax_two_port_smatrix.py::test_reciprocity_catches_a_per_port_amplitude_error[1.05]` |
| `test_coax_two_port_solve.py::test_reciprocity_catches_a_per_port_amplitude_error[1.5]` | `test_coax_two_port_smatrix.py::test_reciprocity_catches_a_per_port_amplitude_error[1.5]` |
| `test_coax_two_port_solve.py::test_reciprocity_catches_a_per_port_amplitude_error[0.8]` | `test_coax_two_port_smatrix.py::test_reciprocity_catches_a_per_port_amplitude_error[0.8]` |
| `test_coax_two_port_solve.py::test_reciprocity_is_BLIND_to_any_unit_modulus_per_port_factor[sign]` | `test_coax_two_port_smatrix.py::test_reciprocity_is_BLIND_to_any_unit_modulus_per_port_factor[sign]` |
| `test_coax_two_port_solve.py::test_reciprocity_is_BLIND_to_any_unit_modulus_per_port_factor[phase]` | `test_coax_two_port_smatrix.py::test_reciprocity_is_BLIND_to_any_unit_modulus_per_port_factor[phase]` |
| `test_coax_two_port_solve.py::test_reciprocity_is_BLIND_to_any_unit_modulus_per_port_factor[quarter]` | `test_coax_two_port_smatrix.py::test_reciprocity_is_BLIND_to_any_unit_modulus_per_port_factor[quarter]` |
| `test_coax_two_port_solve.py::test_ill_conditioned_drives_warn_and_are_reported` | `test_coax_two_port_smatrix.py::test_ill_conditioned_drives_warn_and_are_reported` |
| `test_coax_two_port_solve.py::test_shape_and_finiteness_contract` | `test_coax_two_port_smatrix.py::test_shape_and_finiteness_contract` |
| `test_coaxial_line_calibration.py::test_short_reflects_minus_one_full_band` | `test_coaxial_line_reflection.py::test_short_reflects_minus_one_full_band` |
| `test_coaxial_line_calibration.py::test_open_reflects_unity_magnitude_full_band` | `test_coaxial_line_reflection.py::test_open_reflects_unity_magnitude_full_band` |
| `test_coaxial_line_calibration.py::test_matched_reflects_near_zero_and_recovers_z0` | `test_coaxial_line_reflection.py::test_matched_reflects_near_zero_and_recovers_z0` |
| `test_coaxial_line_calibration.py::test_resistive_load_reflection_magnitude` | `test_coaxial_line_reflection.py::test_resistive_load_reflection_magnitude` |
| `test_coaxial_line_calibration.py::test_under_resolved_annulus_is_flagged` | `test_coaxial_line_reflection.py::test_under_resolved_annulus_is_flagged` |
| `test_coaxial_line_calibration.py::test_nonuniform_profiles_are_rejected_before_coaxial_line_run[dx_profile]` | `test_coaxial_line_reflection.py::test_nonuniform_profiles_are_rejected_before_coaxial_line_run[dx_profile]` |
| `test_coaxial_line_calibration.py::test_nonuniform_profiles_are_rejected_before_coaxial_line_run[dy_profile]` | `test_coaxial_line_reflection.py::test_nonuniform_profiles_are_rejected_before_coaxial_line_run[dy_profile]` |
| `test_coaxial_line_calibration.py::test_nonuniform_profiles_are_rejected_before_coaxial_line_run[dz_profile]` | `test_coaxial_line_reflection.py::test_nonuniform_profiles_are_rejected_before_coaxial_line_run[dz_profile]` |
| `test_coaxial_line_calibration.py::test_existing_tfsf_is_rejected_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_existing_tfsf_is_rejected_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_refinement_is_rejected_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_refinement_is_rejected_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_adi_is_rejected_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_adi_is_rejected_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_nonabsorbing_boundary_is_rejected_before_coaxial_line_run[pec]` | `test_coaxial_line_reflection.py::test_nonabsorbing_boundary_is_rejected_before_coaxial_line_run[pec]` |
| `test_coaxial_line_calibration.py::test_nonabsorbing_boundary_is_rejected_before_coaxial_line_run[upml]` | `test_coaxial_line_reflection.py::test_nonabsorbing_boundary_is_rejected_before_coaxial_line_run[upml]` |
| `test_coaxial_line_calibration.py::test_nonabsorbing_boundary_is_rejected_before_coaxial_line_run[zero_cpml_layers]` | `test_coaxial_line_reflection.py::test_nonabsorbing_boundary_is_rejected_before_coaxial_line_run[zero_cpml_layers]` |
| `test_coaxial_line_calibration.py::test_two_dimensional_mode_is_rejected_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_two_dimensional_mode_is_rejected_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_fourth_order_stencil_is_rejected_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_fourth_order_stencil_is_rejected_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_mixed_precision_is_rejected_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_mixed_precision_is_rejected_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_non_axial_cpml_selection_is_rejected_before_coaxial_line_run[]` | `test_coaxial_line_reflection.py::test_non_axial_cpml_selection_is_rejected_before_coaxial_line_run[]` |
| `test_coaxial_line_calibration.py::test_non_axial_cpml_selection_is_rejected_before_coaxial_line_run[x]` | `test_coaxial_line_reflection.py::test_non_axial_cpml_selection_is_rejected_before_coaxial_line_run[x]` |
| `test_coaxial_line_calibration.py::test_non_axial_cpml_selection_is_rejected_before_coaxial_line_run[xyz]` | `test_coaxial_line_reflection.py::test_non_axial_cpml_selection_is_rejected_before_coaxial_line_run[xyz]` |
| `test_coaxial_line_calibration.py::test_periodic_axis_is_rejected_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_periodic_axis_is_rejected_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run[z_lo_pec]` | `test_coaxial_line_reflection.py::test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run[z_lo_pec]` |
| `test_coaxial_line_calibration.py::test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run[z_hi_pec]` | `test_coaxial_line_reflection.py::test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run[z_hi_pec]` |
| `test_coaxial_line_calibration.py::test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run[z_lo_zero]` | `test_coaxial_line_reflection.py::test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run[z_lo_zero]` |
| `test_coaxial_line_calibration.py::test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run[z_hi_zero]` | `test_coaxial_line_reflection.py::test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run[z_hi_zero]` |
| `test_coaxial_line_calibration.py::test_mixed_transverse_boundary_is_rejected_before_coaxial_line_run[x_pec]` | `test_coaxial_line_reflection.py::test_mixed_transverse_boundary_is_rejected_before_coaxial_line_run[x_pec]` |
| `test_coaxial_line_calibration.py::test_mixed_transverse_boundary_is_rejected_before_coaxial_line_run[x_pmc]` | `test_coaxial_line_reflection.py::test_mixed_transverse_boundary_is_rejected_before_coaxial_line_run[x_pmc]` |
| `test_coaxial_line_calibration.py::test_mixed_transverse_boundary_is_rejected_before_coaxial_line_run[y_pec]` | `test_coaxial_line_reflection.py::test_mixed_transverse_boundary_is_rejected_before_coaxial_line_run[y_pec]` |
| `test_coaxial_line_calibration.py::test_registered_geometry_is_rejected_before_coaxial_line_run[geometry]` | `test_coaxial_line_reflection.py::test_registered_geometry_is_rejected_before_coaxial_line_run[geometry]` |
| `test_coaxial_line_calibration.py::test_registered_geometry_is_rejected_before_coaxial_line_run[thin_conductor]` | `test_coaxial_line_reflection.py::test_registered_geometry_is_rejected_before_coaxial_line_run[thin_conductor]` |
| `test_coaxial_line_calibration.py::test_lumped_rlc_is_rejected_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_lumped_rlc_is_rejected_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_registered_monitor_is_rejected_before_coaxial_line_run[probe]` | `test_coaxial_line_reflection.py::test_registered_monitor_is_rejected_before_coaxial_line_run[probe]` |
| `test_coaxial_line_calibration.py::test_registered_monitor_is_rejected_before_coaxial_line_run[dft]` | `test_coaxial_line_reflection.py::test_registered_monitor_is_rejected_before_coaxial_line_run[dft]` |
| `test_coaxial_line_calibration.py::test_registered_monitor_is_rejected_before_coaxial_line_run[flux]` | `test_coaxial_line_reflection.py::test_registered_monitor_is_rejected_before_coaxial_line_run[flux]` |
| `test_coaxial_line_calibration.py::test_registered_monitor_is_rejected_before_coaxial_line_run[ntff]` | `test_coaxial_line_reflection.py::test_registered_monitor_is_rejected_before_coaxial_line_run[ntff]` |
| `test_coaxial_line_calibration.py::test_registered_coax_termination_helper_is_rejected_before_line_run[matched]` | `test_coaxial_line_reflection.py::test_registered_coax_termination_helper_is_rejected_before_line_run[matched]` |
| `test_coaxial_line_calibration.py::test_registered_coax_termination_helper_is_rejected_before_line_run[open]` | `test_coaxial_line_reflection.py::test_registered_coax_termination_helper_is_rejected_before_line_run[open]` |
| `test_coaxial_line_calibration.py::test_registered_coax_termination_helper_is_rejected_before_line_run[pec_end_cap]` | `test_coaxial_line_reflection.py::test_registered_coax_termination_helper_is_rejected_before_line_run[pec_end_cap]` |
| `test_coaxial_line_calibration.py::test_dut_impedance_is_rejected_when_termination_does_not_use_it[short]` | `test_coaxial_line_reflection.py::test_dut_impedance_is_rejected_when_termination_does_not_use_it[short]` |
| `test_coaxial_line_calibration.py::test_dut_impedance_is_rejected_when_termination_does_not_use_it[open]` | `test_coaxial_line_reflection.py::test_dut_impedance_is_rejected_when_termination_does_not_use_it[open]` |
| `test_coaxial_line_calibration.py::test_all_requested_probe_planes_must_fit_before_coaxial_line_run` | `test_coaxial_line_reflection.py::test_all_requested_probe_planes_must_fit_before_coaxial_line_run` |
| `test_coaxial_line_calibration.py::test_at_least_three_probe_planes_are_required_before_coaxial_line_run[0]` | `test_coaxial_line_reflection.py::test_at_least_three_probe_planes_are_required_before_coaxial_line_run[0]` |
| `test_coaxial_line_calibration.py::test_at_least_three_probe_planes_are_required_before_coaxial_line_run[1]` | `test_coaxial_line_reflection.py::test_at_least_three_probe_planes_are_required_before_coaxial_line_run[1]` |
| `test_coaxial_line_calibration.py::test_at_least_three_probe_planes_are_required_before_coaxial_line_run[2]` | `test_coaxial_line_reflection.py::test_at_least_three_probe_planes_are_required_before_coaxial_line_run[2]` |
| `test_coaxial_line_calibration.py::test_probe_count_must_be_an_integer_before_coaxial_line_run[True]` | `test_coaxial_line_reflection.py::test_probe_count_must_be_an_integer_before_coaxial_line_run[True]` |
| `test_coaxial_line_calibration.py::test_probe_count_must_be_an_integer_before_coaxial_line_run[3.5]` | `test_coaxial_line_reflection.py::test_probe_count_must_be_an_integer_before_coaxial_line_run[3.5]` |
| `test_coaxial_line_extraction.py::test_recovers_known_reflection_lossless[(-1+0j)]` | `test_coaxial_line_reflection.py::test_recovers_known_reflection_lossless[(-1+0j)]` |
| `test_coaxial_line_extraction.py::test_recovers_known_reflection_lossless[(1+0j)]` | `test_coaxial_line_reflection.py::test_recovers_known_reflection_lossless[(1+0j)]` |
| `test_coaxial_line_extraction.py::test_recovers_known_reflection_lossless[(0.3+0.4j)]` | `test_coaxial_line_reflection.py::test_recovers_known_reflection_lossless[(0.3+0.4j)]` |
| `test_coaxial_line_extraction.py::test_recovers_known_reflection_lossless[(-0.2-0.5j)]` | `test_coaxial_line_reflection.py::test_recovers_known_reflection_lossless[(-0.2-0.5j)]` |
| `test_coaxial_line_extraction.py::test_recovers_known_reflection_lossless[0j]` | `test_coaxial_line_reflection.py::test_recovers_known_reflection_lossless[0j]` |
| `test_coaxial_line_extraction.py::test_recovers_known_reflection_with_loss` | `test_coaxial_line_reflection.py::test_recovers_known_reflection_with_loss` |
| `test_coaxial_line_extraction.py::test_load_above_probes_branch` | `test_coaxial_line_reflection.py::test_load_above_probes_branch` |
| `test_coaxial_line_extraction.py::test_lossless_reflection_magnitude_is_unity_for_reactive_load` | `test_coaxial_line_reflection.py::test_lossless_reflection_magnitude_is_unity_for_reactive_load` |
| `test_coaxial_line_extraction.py::test_input_validation` | `test_coaxial_line_reflection.py::test_input_validation` |
| `test_coaxial_line_extraction.py::test_reflection_extractor_grad_matches_closed_form_and_fd` | `test_coaxial_line_reflection.py::test_reflection_extractor_grad_matches_closed_form_and_fd` |
| `test_coaxial_line_extraction.py::test_reflection_grad_finite_at_reactive_null` | `test_coaxial_line_reflection.py::test_reflection_grad_finite_at_reactive_null` |
| `test_lumped_wire_sparam_cpml_dielectric.py::test_lumped_port_sparam_cpml_dielectric_finite_passive` | `test_cpml_material_aware.py::test_lumped_port_sparam_cpml_dielectric_finite_passive` |
| `test_lumped_wire_sparam_cpml_dielectric.py::test_wire_port_sparam_cpml_dielectric_finite_passive` | `test_cpml_material_aware.py::test_wire_port_sparam_cpml_dielectric_finite_passive` |
| `test_msl_settling_witness.py::test_truncated_record_fails_the_witness_loudly` | `test_settling_witness.py::test_truncated_record_fails_the_witness_loudly` |
| `test_msl_settling_witness.py::test_settled_record_passes_the_witness_silently` | `test_settling_witness.py::test_settled_record_passes_the_witness_silently` |
| `test_msl_settling_witness.py::test_witness_survives_pre_existing_user_probes` | `test_settling_witness.py::test_witness_survives_pre_existing_user_probes` |
| `test_msl_settling_witness.py::test_witness_probes_do_not_leak_into_the_simulation` | `test_settling_witness.py::test_witness_probes_do_not_leak_into_the_simulation` |
| `test_msl_settling_witness.py::test_result_field_is_optional_for_backward_compatibility` | `test_settling_witness.py::test_result_field_is_optional_for_backward_compatibility` |
| `test_settling_witness_enforcement.py::test_threshold_constant_is_the_documented_bar` | `test_settling_witness.py::test_threshold_constant_is_the_documented_bar` |
| `test_settling_witness_enforcement.py::test_violating_witness_warns_and_quotes_the_measured_value` | `test_settling_witness.py::test_violating_witness_warns_and_quotes_the_measured_value` |
| `test_settling_witness_enforcement.py::test_settled_witness_stays_silent` | `test_settling_witness.py::test_settled_witness_stays_silent` |
| `test_settling_witness_enforcement.py::test_witness_exactly_at_the_bar_is_not_a_violation` | `test_settling_witness.py::test_witness_exactly_at_the_bar_is_not_a_violation` |
| `test_settling_witness_enforcement.py::test_all_nan_witness_is_silent_not_a_false_fire` | `test_settling_witness.py::test_all_nan_witness_is_silent_not_a_false_fire` |
| `test_settling_witness_enforcement.py::test_nan_beside_a_violator_does_not_mask_the_violator` | `test_settling_witness.py::test_nan_beside_a_violator_does_not_mask_the_violator` |
| `test_settling_witness_enforcement.py::test_every_violating_drive_is_named_not_only_the_worst` | `test_settling_witness.py::test_every_violating_drive_is_named_not_only_the_worst` |
| `test_settling_witness_enforcement.py::test_warning_names_the_knob_the_lane_is_actually_driven_by` | `test_settling_witness.py::test_warning_names_the_knob_the_lane_is_actually_driven_by` |
| `test_settling_witness_enforcement.py::test_every_settling_db_producer_routes_through_the_shared_warner` | `test_settling_witness.py::test_every_settling_db_producer_routes_through_the_shared_warner` |
| `test_settling_witness_enforcement.py::test_the_known_lanes_are_all_covered` | `test_settling_witness.py::test_the_known_lanes_are_all_covered` |
| `test_settling_witness_enforcement.py::test_underrun_coax_two_port_warns_instead_of_returning_it_quietly` | `test_settling_witness.py::test_underrun_coax_two_port_warns_instead_of_returning_it_quietly` |
| `test_settling_witness_enforcement.py::test_settled_coax_two_port_stays_silent` | `test_settling_witness.py::test_settled_coax_two_port_stays_silent` |
| `test_settling_witness_enforcement.py::test_differentiable_coax_path_leaves_the_witness_nan_and_silent` | `test_settling_witness.py::test_differentiable_coax_path_leaves_the_witness_nan_and_silent` |
| `test_waveguide_settling_witness.py::test_settling_populated_and_truncation_warning_fires` | `test_settling_witness.py::test_settling_populated_and_truncation_warning_fires` |
| `test_waveguide_settling_witness.py::test_longer_record_settles_deeper` | `test_settling_witness.py::test_longer_record_settles_deeper` |
| `test_waveguide_settling_witness.py::test_witness_flag_does_not_perturb_s_extractor_level` | `test_settling_witness.py::test_witness_flag_does_not_perturb_s_extractor_level` |
| `test_sbp_sat_1d.py::test_sbp_property` | `test_sbp_sat.py::test_sbp_property` |
| `test_sbp_sat_1d.py::test_stability_long_run` | `test_sbp_sat.py::test_stability_long_run` |
| `test_sbp_sat_1d.py::test_subgrid_matches_uniform` | `test_sbp_sat.py::test_subgrid_matches_uniform` |
| `test_sbp_sat_1d.py::test_energy_conservation` | `test_sbp_sat.py::test_energy_conservation` |
| `test_sbp_sat_2d.py::test_2d_stability` | `test_sbp_sat.py::test_2d_stability` |
| `test_sbp_sat_2d.py::test_2d_pulse_propagation` | `test_sbp_sat.py::test_2d_pulse_propagation` |
| `test_sbp_sat_2d.py::test_2d_small_fine_region` | `test_sbp_sat.py::test_2d_small_fine_region` |
| `test_sbp_sat_3d.py::test_3d_stability` | `test_sbp_sat.py::test_3d_stability` |
| `test_sbp_sat_3d.py::test_3d_fine_grid_receives_signal` | `test_sbp_sat.py::test_3d_fine_grid_receives_signal` |
| `test_sbp_sat_3d.py::test_3d_energy_finite` | `test_sbp_sat.py::test_3d_energy_finite` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_default_tau_is_half` | `test_sbp_sat.py::TestSBPSATAlpha::test_default_tau_is_half` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_custom_tau_stored` | `test_sbp_sat.py::TestSBPSATAlpha::test_custom_tau_stored` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_xy_margin_stored_for_research_windowed_refinement` | `test_sbp_sat.py::TestSBPSATAlpha::test_xy_margin_stored_for_research_windowed_refinement` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_custom_tau_accepted` | `test_sbp_sat.py::TestSBPSATAlpha::test_custom_tau_accepted` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_default_tau_runs` | `test_sbp_sat.py::TestSBPSATAlpha::test_default_tau_runs` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_tau_propagates_to_config` | `test_sbp_sat.py::TestSBPSATAlpha::test_tau_propagates_to_config` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_init_subgrid_3d_tau_passthrough` | `test_sbp_sat.py::TestSBPSATAlpha::test_init_subgrid_3d_tau_passthrough` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_init_subgrid_3d_default_tau` | `test_sbp_sat.py::TestSBPSATAlpha::test_init_subgrid_3d_default_tau` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_different_tau_different_results` | `test_sbp_sat.py::TestSBPSATAlpha::test_different_tau_different_results` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_alpha_values_are_dimensionless` | `test_sbp_sat.py::TestSBPSATAlpha::test_alpha_values_are_dimensionless` |
| `test_sbp_sat_alpha.py::TestSBPSATAlpha::test_energy_stable_after_fix` | `test_sbp_sat.py::TestSBPSATAlpha::test_energy_stable_after_fix` |
| `test_sbp_sat_jit.py::TestJITBasic::test_jit_pec_runs_without_error` | `test_sbp_sat.py::TestJITBasic::test_jit_pec_runs_without_error` |
| `test_sbp_sat_jit.py::TestJITBasic::test_jit_cpml_runs_without_error` | `test_sbp_sat.py::TestJITBasic::test_jit_cpml_runs_without_error` |
| `test_sbp_sat_jit.py::TestJITBasic::test_jit_produces_nonzero_fields` | `test_sbp_sat.py::TestJITBasic::test_jit_produces_nonzero_fields` |
| `test_sbp_sat_jit.py::TestJITBasic::test_jit_time_series_shape` | `test_sbp_sat.py::TestJITBasic::test_jit_time_series_shape` |
| `test_sbp_sat_jit.py::TestJITBasic::test_jit_time_series_nonzero` | `test_sbp_sat.py::TestJITBasic::test_jit_time_series_nonzero` |
| `test_sbp_sat_jit.py::TestJITEdgeCases::test_jit_no_probe` | `test_sbp_sat.py::TestJITEdgeCases::test_jit_no_probe` |
| `test_sbp_sat_jit.py::TestJITEdgeCases::test_jit_no_source` | `test_sbp_sat.py::TestJITEdgeCases::test_jit_no_source` |
| `test_sbp_sat_jit.py::TestJITEdgeCases::test_source_outside_fine_region_fails_loudly` | `test_sbp_sat.py::TestJITEdgeCases::test_source_outside_fine_region_fails_loudly` |
| `test_sbp_sat_jit.py::TestJITStability::test_jit_fields_finite_1000_steps` | `test_sbp_sat.py::TestJITStability::test_jit_fields_finite_1000_steps` |
| `test_sbp_sat_jit.py::TestJITStability::test_jit_energy_stable_1000_steps` | `test_sbp_sat.py::TestJITStability::test_jit_energy_stable_1000_steps` |
| `test_sbp_sat_jit.py::TestJITStability::test_jit_cpml_fields_finite` | `test_sbp_sat.py::TestJITStability::test_jit_cpml_fields_finite` |
| `test_sbp_sat_jit.py::TestJITRunnerDirect::test_direct_jit_runner_pec` | `test_sbp_sat.py::TestJITRunnerDirect::test_direct_jit_runner_pec` |
| `test_sbp_sat_jit.py::TestJITRunnerDirect::test_direct_jit_runner_no_probes_no_sources` | `test_sbp_sat.py::TestJITRunnerDirect::test_direct_jit_runner_no_probes_no_sources` |
| `test_sbp_sat_jit.py::TestJITRunnerHCoupling::test_jit_runner_h_coupling_energy` | `test_sbp_sat.py::TestJITRunnerHCoupling::test_jit_runner_h_coupling_energy` |
| `test_sbp_sat_jit.py::TestSubgridMaterialTransition::test_dielectric_crossing_boundary_stable` | `test_sbp_sat.py::TestSubgridMaterialTransition::test_dielectric_crossing_boundary_stable` |
| `test_sbp_sat_jit.py::TestSubgridMaterialTransition::test_dielectric_changes_field_amplitude` | `test_sbp_sat.py::TestSubgridMaterialTransition::test_dielectric_changes_field_amplitude` |
| `test_subgrid_cpml_dielectric.py::test_subgrid_cpml_dielectric_stable_and_absorbing` | `test_cpml_material_aware.py::test_subgrid_cpml_dielectric_stable_and_absorbing` |
| `test_subgrid_cpml_dielectric.py::test_subgrid_cpml_vacuum_control_still_absorbing` | `test_cpml_material_aware.py::test_subgrid_cpml_vacuum_control_still_absorbing` |

## 5. Full default suite (appended 2026-09-03)

`python -m pytest -q -p no:cacheprovider` on this tip (macOS, CPU, jax x64
off), 3164 s:

    9 failed, 5017 passed, 34 skipped, 395 deselected, 24 xfailed

Failures, all pre-existing and none in a merged file:

* the four macOS-only failures of issue #876: `tests/unit/sparams/test_msl_sparse_dft.py::test_cropped_extractor_reproduces_full_plane_s_end_to_end[uniform-x|uniform-y|nonuniform-x]`
  and `tests/unit/sparams/test_probe_fed_msl_referee_contract.py::test_plan_computes_the_post_smoothing_mesh_the_builder_hands_openems`;
* five more that fail IDENTICALLY on the untouched tier-4b tip `3bafe2f`
  (re-run there in `~/Documents/rfx-worktrees/reorg-unit`: `5 failed, 15 passed, 1 skipped`),
  i.e. environment failures of this macOS host, not of this branch:
  `tests/unit/misc/test_weekly_rss_reporter.py::{test_read_proc_vmhwm_mb_returns_a_positive_number_on_linux,test_read_proc_vmhwm_mb_agrees_with_independent_ru_maxrss_band,test_live_sampler_line_survives_sigkill_with_dash_s}`
  (read `/proc` — Linux-only), `tests/unit/boundaries/test_boundary_spec_cpml_budget.py::test_result_saturates_once_the_budget_covers_the_narrowest_axis`
  and `tests/unit/grid/test_fourth_order_stencil.py::test_update_order2_state_byte_identical`.
  Not consolidated in this tier; left for the owners (candidates for the
  #876 list).

Every test of the ten merged files passed in this run (they also passed in
isolation before each group commit: cpml 13, sbp+settling+waveguide 81,
coax 109, preflight+sheet 181 passed / 2 skipped).

## 7. Independent review (2026-09-03)

An adversarial review of the branch at `3f88600` (skip/xfail scope per node,
helper and constant shadowing by AST scan, parametrisation, assertion-line
multiset diff for all 10 merged files, conftest set, durations and shard
balance, kept-file reasons) returned SHIP with three note-level findings, all
applied here: the non-verbatim cpml helper edits (table in §1), the dead
durations key (§2; removed, proof script updated), and the visualize3d
"no shared line" overstatement (§3). Numbers the review re-derived: per-node
outcome diff between the source files at `3bafe2f` and the merged files: zero
changes (380 passed / 2 skipped on both tips, the 2 skips being the same
per-test `importorskip("trimesh")`); assertion-line multiset diff empty for all
10 files; pytest-split shard simulation before [1704.8, 1705.3, 1717.0,
1690.3] s, after [1705.1, 1705.8, 1704.7, 1702.6] s.

## 8. Post-plan moves folded in (2026-09-03)

The four cv22/cv23 tests that landed on `main` after the tier-4b plan
(`test_cv22_dispersive_eps_mapping.py`, `test_cv22_dispersive_slab_gates.py`,
`test_cv23_lossy_eps_mapping.py`, `test_cv23_lossy_slab_gates.py`) were still
at the top level and are moved to `tests/crossval/` beside
`test_cv24_nu_cavity_gates.py`. Same treatment as the cv24 move in tier 4b:
`_REPO = Path(__file__).resolve().parents[1]` → `parents[2]`, and every path
citation (manifest, validation README, benchmarks table, the two cv22/cv23
design notes, the cv22 comparator docstring, the eight cv22/cv23 VESSL YAMLs)
rewritten to the new path. No `.test_durations` keys existed for these files.
The proof was re-run with `--base origin/agent/reorg-tier4b-unit` (= `a05b584`,
tier 4b rebased onto `main` after #853/#855): full inventory 5600 → 5599 (+1
collapse), default lane 5205 → 5204, missing 0, unmapped 0, `PROOF HOLDS`.
`tests/` now has no top-level test module.
