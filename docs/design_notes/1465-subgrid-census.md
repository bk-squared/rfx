# #1465 call-site census

Base: `292e3903`. Search: `rg -n "add_refinement\(" tests examples validation`, followed by AST inspection of calls and enclosing helpers. There are 28 omitted-validation calls in existing tests, plus two helpers whose `validation` argument defaults to production (`_box`, `_lid`). `examples/` has no calls. Text in docstrings and the historical build string in `boundary_baseline.py` is not executable. Already explicit research/off calls are excluded from the default census.

The base line is the grep location before edits. Current lines link to the retained call. “New refusal” below means the #1465 admission now stops the old test before its former assertion/runner, including tests that previously expected a different refusal. It does not mean all such configurations previously stepped.

| Base call | Enclosing scope | Disposition / execution coverage |
|---|---|---|
| `tests/contracts/boundary_cases.py:22` → [22](../../tests/contracts/boundary_cases.py#L22) | `build` | All 12 already refused. Ten refusal reasons now #1465; two earlier errors unchanged. 12 tests passed; historical B1 preserved. |
| `tests/contracts/test_lattice_ownership_contract.py:1386` → [1386](../../tests/contracts/test_lattice_ownership_contract.py#L1386) | `test_validate_subgrid_refuses_the_sheet_the_runner_refuses` | Direct census selection passed; existing earlier refusal or build/validation-only behavior retained. |
| `tests/contracts/test_pec_lane_parity_numeric.py:327` → [328](../../tests/contracts/test_pec_lane_parity_numeric.py#L328) | `test_the_subgridded_lane_refuses_a_sheet_it_cannot_realize` | New refusal; retained with research opt-in and #1465 comment; relevant test group passed. |
| `tests/contracts/test_realized_geometry_record.py:302` → [303](../../tests/contracts/test_realized_geometry_record.py#L303) | `test_subgrid_record_names_unrepresented_refinement` | New refusal; retained with research opt-in and #1465 comment; relevant test group passed. |
| `tests/unit/api/test_api.py:76` → [76](../../tests/unit/api/test_api.py#L76) | `test_upml_rejects_subgridding_refinement` | Direct census selection passed; existing earlier refusal or build/validation-only behavior retained. |
| `tests/unit/api/test_rasterized_slice_viewer.py:482` → [482](../../tests/unit/api/test_rasterized_slice_viewer.py#L482) | `test_add_refinement_warns_because_the_drawn_grid_is_coarse_only` | Direct census selection passed; existing earlier refusal or build/validation-only behavior retained. |
| `tests/unit/materials/test_sheet_impedance.py:887` → [888](../../tests/unit/materials/test_sheet_impedance.py#L888) | `test_fence_subgridded_run::go` | New refusal; retained with research opt-in and #1465 comment; relevant test group passed. |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py:66` → [66](../../tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py#L66) | `_box` | Default remains production; graded/other-entry refusals unchanged; full 27-test file passed. |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py:144` → [144](../../tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py#L144) | `test_run_with_wire_port_s_parameters_refuses` | Existing graded or non-subgridded-entry refusal unchanged; full 27-test file passed. |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py:220` → [220](../../tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py#L220) | `test_graded_waveguide_s_matrix_refuses` | Existing graded or non-subgridded-entry refusal unchanged; full 27-test file passed. |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py:290` → [291](../../tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py#L291) | `_lumped_port_box` | New refusal in two sequential-sweep tests; retained with research + #1465 comment; 27-test file passed. |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py:373` → [374](../../tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py#L374) | `test_waveguide_port_reference_model_with_a_refinement_refuses::two_port` | Existing graded or non-subgridded-entry refusal unchanged; full 27-test file passed. |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py:403` → [404](../../tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py#L404) | `test_uniform_waveguide_s_matrix_refuses` | Existing graded or non-subgridded-entry refusal unchanged; full 27-test file passed. |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py:420` → [421](../../tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py#L421) | `test_differentiable_material_fit_refuses::factory` | Existing graded or non-subgridded-entry refusal unchanged; full 27-test file passed. |
| `tests/unit/ports/test_pec_filament_radius.py:101` → [101](../../tests/unit/ports/test_pec_filament_radius.py#L101) | `test_other_filament_paths_refuse_before_scan` | Direct census selection passed; existing earlier refusal or build/validation-only behavior retained. |
| `tests/unit/ports/test_wire_port_radius.py:242` → [242](../../tests/unit/ports/test_wire_port_radius.py#L242) | `test_unsupported_radius_path_refuses_before_first_step` | Direct census selection passed; existing earlier refusal or build/validation-only behavior retained. |
| `tests/unit/preflight/test_preflight_guards.py:855` → [855](../../tests/unit/preflight/test_preflight_guards.py#L855) | `_bad_sim_upml_refinement` | Build/preflight only; two tests passed; UPML error unchanged. |
| `tests/unit/preflight/test_removed_coaxial_s_matrix_lane.py:207` → [207](../../tests/unit/preflight/test_removed_coaxial_s_matrix_lane.py#L207) | `module-level lambda` | Removed-calculator refusal unchanged; sbp-sat-refinement test passed. |
| `tests/unit/runners/test_calculator_admission.py:146` → [146](../../tests/unit/runners/test_calculator_admission.py#L146) | `test_fit_migrated_inputs_refused_before_assembly_or_scan::factory` | Material-fit refusal unchanged; refinement test passed. |
| `tests/unit/runners/test_distributed_admission_refusals.py:71` → [71](../../tests/unit/runners/test_distributed_admission_refusals.py#L71) | `_build` | Distributed refusal unchanged; two refinement tests passed. |
| `tests/unit/runners/test_path_disposition_cells.py:114` → [115](../../tests/unit/runners/test_path_disposition_cells.py#L115) | `_base` | Research opt-in for experimental cells; production refusal cells remain production. 109 subgrid + 6 lid tests passed. |
| `tests/unit/runners/test_path_disposition_cells.py:215` → [217](../../tests/unit/runners/test_path_disposition_cells.py#L217) | `_board` | Research opt-in for experimental cells; production refusal cells remain production. 109 subgrid + 6 lid tests passed. |
| `tests/unit/runners/test_path_disposition_cells.py:232` → [235](../../tests/unit/runners/test_path_disposition_cells.py#L235) | `_guide` | Research opt-in for experimental cells; production refusal cells remain production. 109 subgrid + 6 lid tests passed. |
| `tests/unit/runners/test_path_disposition_cells.py:242` → [246](../../tests/unit/runners/test_path_disposition_cells.py#L246) | `_plane_wave` | Research opt-in for experimental cells; production refusal cells remain production. 109 subgrid + 6 lid tests passed. |
| `tests/unit/runners/test_path_disposition_cells.py:256` → [261](../../tests/unit/runners/test_path_disposition_cells.py#L261) | `_floquet_cell::build` | Research opt-in for experimental cells; production refusal cells remain production. 109 subgrid + 6 lid tests passed. |
| `tests/unit/runners/test_path_disposition_cells.py:312` → [321](../../tests/unit/runners/test_path_disposition_cells.py#L321) | `_mode` | Research opt-in for experimental cells; production refusal cells remain production. 109 subgrid + 6 lid tests passed. |
| `tests/unit/runners/test_path_disposition_cells.py:328` → [336](../../tests/unit/runners/test_path_disposition_cells.py#L336) | `_lid` | Research opt-in for experimental cells; production refusal cells remain production. 109 subgrid + 6 lid tests passed. |
| `tests/unit/runners/test_realized_model_cells.py:49` → [50](../../tests/unit/runners/test_realized_model_cells.py#L50) | `model` | New refusal; retained with research opt-in and #1465 comment; relevant test group passed. |
| `tests/unit/runners/test_silent_drop_warnings.py:87` → [88](../../tests/unit/runners/test_silent_drop_warnings.py#L88) | `_subgrid_sim::build` | New refusal; retained with research opt-in and #1465 comment; relevant test group passed. |
| `tests/unit/sources/test_tfsf_closed_box.py:200` → [200](../../tests/unit/sources/test_tfsf_closed_box.py#L200) | `test_new_input_is_never_dropped_on_unsupported_lanes` | Direct census selection passed; existing earlier refusal or build/validation-only behavior retained. |

## Initial failing test IDs
These are the observed failures after adding #1465 admission, before adapting callers/expectations. No test was deleted. The final group counts are in [the report](1465-subgrid-refusal.md).

### boundary_before.log

```text
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[pec--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[cpml--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[pmc-pec--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[pmc-cpml--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[pec-zlo--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[periodic-xy--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[tfsf--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[waveguide-cpml--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[waveguide-pec--subgridded]
tests/contracts/test_realized_boundary.py::test_no_unlisted_departures[floquet--subgridded]
```

### census_direct_before.log

```text
tests/contracts/test_pec_lane_parity_numeric.py::test_the_subgridded_lane_refuses_a_sheet_it_cannot_realize
tests/contracts/test_realized_geometry_record.py::test_subgrid_record_names_unrepresented_refinement
```

### test_path_disposition_cells_before.log

```text
tests/unit/runners/test_path_disposition_cells.py::test_cell[_mode--run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_materials-eps-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_materials-sigma-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_materials-mu-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_materials-debye-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_materials-lorentz-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_materials-drude-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_materials-kerr-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_geometry-pec_volume-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_geometry-pec_sheet-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_geometry-pec_wire-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_thin_conductors-lossy_sheet-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_thin_conductors-pec_sheet-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_thin_conductors-surface_impedance-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_pinned_sheets-pec_sheet-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-source-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-amplitude_kind-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-lumped_port-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-passive_port-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-wire_port-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_msl_ports-msl_port-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_waveguide_ports-waveguide_port-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_floquet_ports-floquet_port-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_floquet_ports-scan_angle-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_lumped_rlc-R-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_lumped_rlc-series_RL-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_tfsf-plane_wave-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary-cpml-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary-upml-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_pec_faces-pec_face-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary_spec-pmc_face-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary_spec-conformal-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary_spec-conformal_s_matrix-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary_spec-absorbing_lid-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_periodic_axes-periodic-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_cpml_layers-layers-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_cpml_kappa_max-kappa-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_interface_eps-dual_average-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_probes-probe-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_dft_planes-dft_plane-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_flux_monitors-flux-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_ntff-ntff_box-run_subgridded]
tests/unit/runners/test_path_disposition_cells.py::test_cell[_current_moments-block_moments-run_subgridded]
```

### test_realized_model_cells_before.log

```text
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[eps-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[eps-run_subgridded-y]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[eps-run_subgridded-z]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[sigma-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[sigma-run_subgridded-y]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[sigma-run_subgridded-z]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[mu-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[mu-run_subgridded-y]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[mu-run_subgridded-z]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[sat-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[sat-run_subgridded-y]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[soft_field-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[soft_current-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[soft_none-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[lumped_none-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_realized_cell[wire_none-run_subgridded-x]
tests/unit/runners/test_realized_model_cells.py::test_dump_is_consumed[run_subgridded-eps]
tests/unit/runners/test_realized_model_cells.py::test_dump_is_consumed[run_subgridded-sigma]
tests/unit/runners/test_realized_model_cells.py::test_dump_is_consumed[run_subgridded-mu]
```

### test_refinement_refused_on_graded_mesh_before.log

```text
tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py::test_vmap_sequential_fallback_still_solves_the_refinement
tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py::test_vmap_sequential_fallback_covers_the_duration_run_covers
```

### test_sheet_impedance_before.log

```text
tests/unit/materials/test_sheet_impedance.py::test_fence_subgridded_run
```

### test_silent_drop_warnings_before.log

```text
tests/unit/runners/test_silent_drop_warnings.py::test_lane_refuses_an_argument_it_does_not_implement[subgridded-subpixel_smoothing]
tests/unit/runners/test_silent_drop_warnings.py::test_lane_refuses_an_argument_it_does_not_implement[subgridded-checkpoint]
tests/unit/runners/test_silent_drop_warnings.py::test_lane_refuses_an_argument_it_does_not_implement[subgridded-snapshot]
tests/unit/runners/test_silent_drop_warnings.py::test_lane_refuses_an_argument_it_does_not_implement[subgridded-until_decay]
tests/unit/runners/test_silent_drop_warnings.py::test_lane_refuses_an_argument_it_does_not_implement[subgridded-conformal_pec]
tests/unit/runners/test_silent_drop_warnings.py::test_conformal_pec_with_no_pec_is_not_refused[subgridded]
tests/unit/runners/test_silent_drop_warnings.py::test_report_every_stays_a_warning[subgridded]
```
