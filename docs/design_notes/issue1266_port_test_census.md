# #1266 graded-port test census

Conclusions: 리더가 채움.

Static inventory: direct intersection of port declarations (`add_port`, `WirePort`, `LumpedPort`) and graded-mesh terms, followed by transitive imports within `tests/locks`, `tests/crossval`, and `tests/unit`. This is a conservative file candidate list; a file can contain separate uniform-port and graded-source models. Runtime instrumentation records actual graded `Simulation.add_port` declarations below. Refused declarations are included. Low-level source helpers are covered by the static inventory, not that API wrapper.

Archived seconds are sums of matching entries in the checked-in `.test_durations`, not timings from this Mac. Files with >60 archived seconds were classified as expensive for the census; required judges and selected port files were run anyway. A zero count means no stored runtime, not zero cost. Mac wall seconds include pytest startup. Default repository deselection of gpu/slow/slow_physics/docs_consistency was retained.

No existing locks/unit/crossval tests or pins were edited. The P0 table edit and the new contract are reported separately.

| Candidate file | Archived cases / seconds | Mac wall seconds | Result |
|---|---:|---:|---|
| `tests/locks/test_preflight_split_snapshot.py` | 73 / 35.35 | 63 | exit 0: 73 passed in 53.77s |
| `tests/locks/test_refplane_port_waves.py` | 33 / 51.33 | 75 | exit 0: 33 passed, 23 warnings in 69.55s (0:01:09) |
| `tests/unit/api/test_api.py` | 49 / 82.34 | — | not run; expensive file |
| `tests/unit/api/test_artifacts.py` | 13 / 0.71 | 9 | exit 0: 13 passed in 2.73s |
| `tests/unit/autodiff/test_design_box_holds_port.py` | 44 / 347.89 | — | not run; expensive file |
| `tests/unit/autodiff/test_design_box_port_fence_edges.py` | 6 / 9.21 | 13 | exit 0: 6 passed in 9.19s |
| `tests/unit/autodiff/test_design_box_tape.py` | 27 / 78.39 | — | not run; expensive file |
| `tests/unit/autodiff/test_design_box_tape_nu_occupancy.py` | 25 / 98.50 | — | not run; expensive file |
| `tests/unit/autodiff/test_forward_jit_compile_once.py` | 19 / 114.56 | — | not run; expensive file |
| `tests/unit/autodiff/test_forward_jit_contract.py` | 0 / 0.00 | 58 | exit 0: 34 passed, 14 warnings in 54.78s |
| `tests/unit/autodiff/test_forward_settling_ad.py` | 7 / 15.16 | 22 | exit 0: 7 passed in 17.08s |
| `tests/unit/autodiff/test_nonuniform_grad_sparams.py` | 2 / 6.20 | 13 | exit 0: 2 passed, 4 warnings in 8.35s |
| `tests/unit/autodiff/test_reciprocity_adjoint.py` | 0 / 0.00 | — | not run; runtime unrecorded |
| `tests/unit/autodiff/test_scan_segmented_checkpoint.py` | 6 / 6.80 | 14 | exit 0: 6 passed, 2 warnings in 10.44s |
| `tests/unit/autodiff/test_sparam_ad_end_to_end.py` | 5 / 192.79 | — | not run; expensive file |
| `tests/unit/autodiff/test_stage1_tier2_tracer_fixes.py` | 6 / 3.56 | 9 | exit 0: 6 passed in 3.64s |
| `tests/unit/autodiff/test_wire_port_on_traced_mesh.py` | 27 / 135.22 | 69 | exit 0: 27 passed, 49 warnings in 64.08s (0:01:04) |
| `tests/unit/boundaries/test_cpml_material_aware.py` | 11 / 81.89 | — | not run; expensive file |
| `tests/unit/boundaries/test_magnetic_wall_faces_not_shorted.py` | 15 / 5.87 | 13 | exit 0: 15 passed, 6 warnings in 8.45s |
| `tests/unit/farfield/test_current_moment_monitor.py` | 117 / 260.43 | 16 | exit 0: 4 passed, 113 deselected, 10 warnings in 13.67s |
| `tests/unit/geometry/test_conductor_continuation_consumers.py` | 33 / 11.47 | 19 | exit 0: 33 passed, 14 warnings in 12.98s |
| `tests/unit/geometry/test_conductor_geometry_continuation.py` | 104 / 3.34 | 5 | exit 0: 104 passed, 29 warnings in 2.79s |
| `tests/unit/materials/test_conductor_mask_accessor.py` | 10 / 27.07 | 17 | exit 0: 10 passed, 4 warnings in 14.38s |
| `tests/unit/materials/test_sheet_continuation.py` | 33 / 101.47 | — | not run; expensive file |
| `tests/unit/materials/test_sheet_continuation_guide.py` | 2 / 21.37 | 17 | exit 0: 2 passed, 4 warnings in 14.67s |
| `tests/unit/materials/test_sheet_impedance.py` | 67 / 87.66 | — | not run; expensive file |
| `tests/unit/nonuniform/test_dispersive_port_load_1257.py` | 13 / 36.83 | 19 | exit 1: 2 failed, 11 passed, 57 warnings in 16.99s |
| `tests/unit/nonuniform/test_dz_only_dispatch_contract.py` | 14 / 42.28 | 24 | exit 0: 14 passed, 94 warnings in 21.02s |
| `tests/unit/nonuniform/test_nonuniform_source_port_dual_spacing.py` | 29 / 14.81 | 10 | exit 0: 29 passed, 10 warnings in 7.78s |
| `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py` | 20 / 124.13 | 86 | exit 1: 9 failed, 11 passed, 11 warnings in 81.82s (0:01:21) |
| `tests/unit/nonuniform/test_nu_forward_port_freqs.py` | 9 / 15.42 | 11 | exit 0: 7 passed, 2 xfailed, 10 warnings in 8.53s |
| `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py` | 26 / 4.16 | 5 | exit 0: 26 passed in 2.83s |
| `tests/unit/nonuniform/test_nu_rlc_under_traced_override.py` | 0 / 0.00 | 35 | exit 0: 2 passed, 4 warnings in 32.12s |
| `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py` | 24 / 79.88 | 31 | exit 0: 24 passed, 70 warnings in 29.47s |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py` | 27 / 24.99 | 13 | exit 0: 27 passed in 10.24s |
| `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py` | 15 / 32.49 | 28 | exit 1: 11 failed, 4 passed, 9 warnings in 25.34s |
| `tests/unit/ports/test_lumped_capacitor_smoothing.py` | 81 / 56.51 | 25 | exit 0: 81 passed, 73 warnings in 22.45s |
| `tests/unit/ports/test_msl_clearance_diagnostic.py` | 57 / 13.89 | 10 | exit 0: 57 passed, 22 warnings in 7.65s |
| `tests/unit/ports/test_msl_port_preflight.py` | 40 / 45.26 | 32 | exit 0: 40 passed in 29.75s |
| `tests/unit/ports/test_msl_realized_port_contract.py` | 53 / 19.86 | 17 | exit 0: 53 passed, 34 warnings in 14.24s |
| `tests/unit/ports/test_pec_filament_radius.py` | 70 / 16.42 | 9 | exit 0: 70 passed, 12 warnings in 6.04s |
| `tests/unit/ports/test_port_current_boundary_convention.py` | 21 / 11.62 | 7 | exit 0: 21 passed, 1 warning in 4.97s |
| `tests/unit/ports/test_wire_port_live_mid_764.py` | 3 / 2.87 | 3 | exit 0: 3 passed, 2 warnings in 1.65s |
| `tests/unit/ports/test_wire_port_radius.py` | 36 / 40.65 | 26 | exit 0: 36 passed, 11 warnings in 22.44s |
| `tests/unit/preflight/test_half_node_split.py` | 17 / 10.81 | 9 | exit 0: 19 passed, 4 warnings in 6.87s |
| `tests/unit/preflight/test_nonuniform_wire_port_centers.py` | 3 / 0.04 | 3 | exit 0: 3 passed, 3 warnings in 0.22s |
| `tests/unit/preflight/test_removed_coaxial_s_matrix_lane.py` | 18 / 0.08 | 2 | exit 0: 18 passed, 3 warnings in 0.08s |
| `tests/unit/runners/test_calculator_admission.py` | 76 / 7.98 | 8 | exit 0: 76 passed, 4 warnings in 5.77s |
| `tests/unit/runners/test_decay_chunking.py` | 30 / 146.28 | — | not run; expensive file |
| `tests/unit/runners/test_distributed_lumped_port_s.py` | 57 / 373.10 | — | not run; expensive file |
| `tests/unit/runners/test_distributed_ntff.py` | 19 / 48.11 | 33 | exit 0: 17 passed, 2 skipped, 38 warnings in 28.37s |
| `tests/unit/runners/test_distributed_run_result_finishing.py` | 5 / 7.84 | 11 | exit 0: 5 passed, 5 warnings in 7.89s |
| `tests/unit/runners/test_jitted_loop_takes_arrays_as_arguments.py` | 16 / 62.43 | — | not run; expensive file |
| `tests/unit/runners/test_path_disposition_cells.py` | 1318 / 448.49 | 347 | exit 1: 4 failed, 1311 passed, 3 xfailed, 18 warnings in 340.14s (0:05:40) |
| `tests/unit/runners/test_realized_model_cells.py` | 281 / 439.85 | 348 | exit 0: 264 passed, 17 xfailed, 12 warnings in 343.71s (0:05:43) |
| `tests/unit/runners/test_silent_drop_warnings.py` | 82 / 70.98 | — | not run; expensive file |
| `tests/unit/runners/test_silent_routes.py` | 38 / 64.39 | — | not run; expensive file |
| `tests/unit/sparams/test_lumped_twoport_vi_validation_battery.py` | 11 / 104.45 | — | not run; expensive file |
| `tests/unit/sparams/test_mixed_port_sparam.py` | 32 / 116.83 | — | not run; expensive file |
| `tests/unit/sparams/test_plain_source_refusal.py` | 0 / 0.00 | 9 | exit 0: 54 passed, 13 warnings in 6.63s |
| `tests/unit/sparams/test_ringdown_early_stop.py` | 36 / 94.97 | — | not run; expensive file |
| `tests/unit/sparams/test_ringdown_forward.py` | 55 / 356.56 | — | not run; expensive file |
| `tests/unit/sparams/test_ringdown_identification.py` | 0 / 0.00 | 37 | exit 0: 18 passed, 26 warnings in 32.82s |
| `tests/unit/sparams/test_ringdown_run.py` | 39 / 52.16 | 99 | exit 0: 39 passed, 82 warnings in 94.05s (0:01:34) |
| `tests/unit/sparams/test_sparameter_support_contract.py` | 22 / 2.04 | 6 | exit 0: 22 passed, 4 warnings in 1.87s |
| `tests/unit/sparams/test_thru_singular_value_dx_ladder_replay.py` | 13 / 0.04 | 4 | exit 0: 13 passed, 1 deselected in 0.09s |
| `tests/unit/sparams/test_twoport_wire_port.py` | 4 / 23.33 | 29 | exit 0: 4 passed, 13 warnings in 24.90s |
| `tests/unit/sparams/test_waveguide_calculator_admission.py` | 7 / 0.55 | 3 | exit 0: 7 passed, 5 warnings in 0.49s |
| `tests/unit/sparams/test_waveguide_nu_grading_zone.py` | 7 / 5.98 | 8 | exit 0: 7 passed in 4.90s |

## Additional executed files

| File | Mac wall seconds | Result |
|---|---:|---|
| `tests/contracts/test_path_disposition.py` | 12 | exit 0: 9 passed in 6.65s |
| `tests/unit/nonuniform/test_nu_wire_port_index_zero_stencil.py` | 3 | exit 0: 9 passed in 1.43s |
| `tests/unit/nonuniform/test_waveguide_port_local_metric.py` | 10 | exit 0: 11 passed in 6.04s |
| `tests/unit/runners/test_subgridded_port_load_reaches_its_edge.py` | 4 | exit 0: 4 passed in 0.27s |
| `tests/unit/ports/test_port_metric_dual_face_nu.py` | 12 | exit 0: 77 passed in 7.91s |
| `tests/unit/ports/test_wire_port.py` | 26 | exit 0: 5 passed, 1 warning in 23.87s |
| `tests/unit/ports/test_lumped_port_known_load_line.py` | 5 | exit 0: 8 passed, 6 xfailed, 12 warnings in 2.94s |

## Runtime declaration witnesses

Instrumented by wrapping `Simulation.add_port` without changing its arguments or result, testing whether any of `_dx_profile`, `_dy_profile`, `_dz_profile` is present. This enumerates declarations in executed tests, not every test in an unexecuted expensive file.

- `tests/locks/test_preflight_split_snapshot.py::test_preflight_report_matches_the_committed_snapshot[nu_wire_port_on_graded_node]` — lumped
- `tests/locks/test_preflight_split_snapshot.py::test_preflight_report_matches_the_committed_snapshot[wire_port_dead_cell_nu]` — wire
- `tests/locks/test_refplane_port_waves.py::test_nonuniform_lane_end_to_end_raises` — wire
- `tests/unit/api/test_artifacts.py::test_nonuniform_mesh_report_included_when_profiles_present` — lumped
- `tests/unit/autodiff/test_forward_jit_contract.py::test_jit_of_the_gradient_equals_the_plain_call[lumped port/graded]` — lumped
- `tests/unit/autodiff/test_forward_jit_contract.py::test_jit_of_the_gradient_equals_the_plain_call[wire port/graded]` — wire
- `tests/unit/autodiff/test_forward_jit_contract.py::test_jit_with_no_traced_input_equals_the_plain_call[lumped port, cpml/graded]` — lumped
- `tests/unit/autodiff/test_nonuniform_grad_sparams.py::test_nu_wireport_concrete_path_s_params_still_numpy_compatible` — wire
- `tests/unit/autodiff/test_nonuniform_grad_sparams.py::test_nu_wireport_grad_does_not_crash` — wire
- `tests/unit/autodiff/test_scan_segmented_checkpoint.py::test_segmented_rejects_nonuniform_path` — lumped
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_a_deformation_that_keeps_its_outer_nodes_is_not_refused` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_a_passive_wire_port_also_sizes_itself_on_the_traced_mesh` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_a_pinned_step_is_the_step_the_run_uses` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_a_thinning_substrate_moves_the_same_way_on_both_routes` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_concrete_s11_is_byte_identical_with_the_tracer_branch_forced_off[0.000125]` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_concrete_s11_is_byte_identical_with_the_tracer_branch_forced_off[0.0]` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_mutation_a_the_tracer_branch_off_refuses_the_traced_run` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_mutation_b_a_port_sized_on_the_nominal_cell_goes_red` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_mutation_b_on_the_z_axis_goes_red` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_reverse_mode_agrees_with_forward_mode` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_s11_gradient_matches_a_central_difference_and_is_second_order` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_substrate_thickness_gradient_matches_a_central_difference` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_the_step_the_run_used_is_in_the_derivative_too` — wire
- `tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_traced_and_concrete_solve_the_same_board` — wire
- `tests/unit/farfield/test_current_moment_monitor.py::test_ringdown_completion_returns_the_same_moments[graded]` — wire
- `tests/unit/farfield/test_current_moment_monitor.py::test_the_public_run_matches_the_plane_route_with_a_port[graded]` — wire
- `tests/unit/geometry/test_conductor_continuation_consumers.py::test_termination_holds_only_the_named_entry_when_shapes_are_identical[geometry-0-True]` — wire
- `tests/unit/geometry/test_conductor_continuation_consumers.py::test_termination_holds_only_the_named_entry_when_shapes_are_identical[geometry-1-True]` — wire
- `tests/unit/geometry/test_conductor_continuation_consumers.py::test_termination_holds_only_the_named_entry_when_shapes_are_identical[thin-0-True]` — wire
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_lane_on_uniform_cells_matches_the_uniform_lane[lumped-debye]` — lumped
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_lane_on_uniform_cells_matches_the_uniform_lane[lumped-lorentz]` — lumped
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_port_load_reaches_the_dispersive_update[lumped-both]` — lumped
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_port_load_reaches_the_dispersive_update[lumped-debye]` — lumped
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_port_load_reaches_the_dispersive_update[lumped-lorentz]` — lumped
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_port_load_reaches_the_dispersive_update[wire-debye]` — wire
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_port_load_reaches_the_dispersive_update[wire-lorentz]` — wire
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_mutation_both_builds_before_the_stamps_sends_everything_red` — lumped
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_mutation_both_builds_before_the_stamps_sends_everything_red` — wire
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_mutation_one_build_before_the_stamps_sends_its_boards_red[debye]` — lumped
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_mutation_one_build_before_the_stamps_sends_its_boards_red[debye]` — wire
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_mutation_one_build_before_the_stamps_sends_its_boards_red[lorentz]` — lumped
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_mutation_one_build_before_the_stamps_sends_its_boards_red[lorentz]` — wire
- `tests/unit/nonuniform/test_dz_only_dispatch_contract.py::test_mixed_dz_only_raises` — wire
- `tests/unit/nonuniform/test_nonuniform_source_port_dual_spacing.py::test_preflight_advises_on_a_wire_port_on_a_graded_node` — wire
- `tests/unit/nonuniform/test_nonuniform_source_port_dual_spacing.py::test_wire_port_metrics_reach_the_ampere_loop_on_the_right_axes` — wire
- `tests/unit/nonuniform/test_nonuniform_source_port_dual_spacing.py::test_wp_meta_live_cell_slots_carry_primal_widths` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_a_concrete_override_keeps_the_msl_launch_fixture_on_the_drawn_eps` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_a_traced_override_drives_the_msl_feed_and_keeps_its_fixture` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_every_graded_port_drive_under_an_override_uses_the_steppers_cb[lumped+C]` — lumped
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_every_graded_port_drive_under_an_override_uses_the_steppers_cb[lumped]` — lumped
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_every_graded_port_drive_under_an_override_uses_the_steppers_cb[wire+C]` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_every_graded_port_drive_under_an_override_uses_the_steppers_cb[wire]` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_mutation_the_drive_from_the_drawn_arrays_sends_1_to_3_red` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_raising_eps_r_by_override_changes_the_board_not_the_drive` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_build_check_sees_a_drive_from_the_drawn_arrays[lumped+C]` — lumped
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_build_check_sees_a_drive_from_the_drawn_arrays[lumped]` — lumped
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_build_check_sees_a_drive_from_the_drawn_arrays[wire+C]` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_build_check_sees_a_drive_from_the_drawn_arrays[wire]` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_gradient_through_the_override_is_the_boards_gradient[upper]` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_gradient_through_the_override_is_the_boards_gradient[whole]` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_layers_declared_by_override_and_by_materials_are_one_board` — wire
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_without_an_override_the_run_is_byte_identical` — wire
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_lumped_before_passive_wire_preserves_wire_index` — lumped
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_lumped_before_passive_wire_preserves_wire_index` — wire
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_lumped_default_stays_without_sparams` — lumped
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_lumped_known_load_graded[1.0-0.0]` — lumped
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_lumped_known_load_graded[2.0-0.3333333333333333]` — lumped
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_lumped_uniform_profile_lane_parity` — lumped
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_narrow_port_band_gradient` — wire
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_port_bins_subset_matches_default` — wire
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_requested_port_bins[lumped]` — lumped
- `tests/unit/nonuniform/test_nu_forward_port_freqs.py::test_requested_port_bins[wire]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_a_transverse_axis_swap_would_be_caught[0.002-wire-ex]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_a_transverse_axis_swap_would_be_caught[0.002-wire-ey]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_a_transverse_axis_swap_would_be_caught[0.002-wire-ez]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_a_transverse_axis_swap_would_be_caught[None-lumped-ex]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_a_transverse_axis_swap_would_be_caught[None-lumped-ey]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_a_transverse_axis_swap_would_be_caught[None-lumped-ez]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_passive_port_quasi_static_s11_matches_the_closed_form[ex-0.0015]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_passive_port_quasi_static_s11_matches_the_closed_form[ey-0.002]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_passive_port_quasi_static_s11_matches_the_closed_form[ez-0.0025]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_passive_port_quasi_static_s11_matches_the_closed_form[ez-0.0045000000000000005]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_preflight_flags_a_lumped_port_on_a_graded_node` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_preflight_is_silent_for_a_lumped_port_on_a_uniform_node` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_stamped_sigma_realizes_z0_on_a_doubly_graded_node[0.002-wire-ex]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_stamped_sigma_realizes_z0_on_a_doubly_graded_node[0.002-wire-ey]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_stamped_sigma_realizes_z0_on_a_doubly_graded_node[0.002-wire-ez]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_stamped_sigma_realizes_z0_on_a_doubly_graded_node[None-lumped-ex]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_stamped_sigma_realizes_z0_on_a_doubly_graded_node[None-lumped-ey]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_stamped_sigma_realizes_z0_on_a_doubly_graded_node[None-lumped-ez]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_the_fixture_axes_are_pairwise_distinct` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_the_primal_spelling_is_blind_on_this_fixture[0.002-wire-ex]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_the_primal_spelling_is_blind_on_this_fixture[0.002-wire-ey]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_the_primal_spelling_is_blind_on_this_fixture[0.002-wire-ez]` — wire
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_the_primal_spelling_is_blind_on_this_fixture[None-lumped-ex]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_the_primal_spelling_is_blind_on_this_fixture[None-lumped-ey]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_the_primal_spelling_is_blind_on_this_fixture[None-lumped-ez]` — lumped
- `tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py::test_uniform_transverse_mesh_is_bit_identical_to_the_primal_spelling` — lumped
- `tests/unit/nonuniform/test_nu_rlc_under_traced_override.py::test_a_graded_board_with_a_lumped_element_differentiates_through_the_override[parallel_L]` — wire
- `tests/unit/nonuniform/test_nu_rlc_under_traced_override.py::test_a_graded_board_with_a_lumped_element_differentiates_through_the_override[series_RL]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_both_lanes_hand_back_the_raw_accumulators_as_meta_acc_pairs` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_excited_port_lane_ordering_disagreement_is_open_683[pec_plates]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_excited_port_lane_ordering_disagreement_is_open_683[vacuum]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_nu_wave_split_is_direction_free` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_lane_parity_across_the_domain[0.004]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_lane_parity_across_the_domain[0.012]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_lane_parity_and_passivity[0.001-vacuum]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_lane_parity_and_passivity[0.003-dielectric]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_lane_parity_and_passivity[0.003-vacuum]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_lane_parity_and_passivity[0.005-pec_plates]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_s11_regression_lock_on_the_known_wrong_normalization[0.001]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_s11_regression_lock_on_the_known_wrong_normalization[0.003]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_passive_s11_regression_lock_on_the_known_wrong_normalization[0.005]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_the_analytic_oracle_reads_the_runners_own_span[0.001-True]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_the_analytic_oracle_reads_the_runners_own_span[0.003-True]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_the_analytic_oracle_reads_the_runners_own_span[0.005-True]` — wire
- `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py::test_the_pec_plates_load_realizes_two_solid_plates[True]` — wire
- `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py::test_run_with_wire_port_s_parameters_refuses` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_every_graded_port_drive_uses_the_steppers_cb[lumped+C]` — lumped
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_every_graded_port_drive_uses_the_steppers_cb[lumped]` — lumped
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_every_graded_port_drive_uses_the_steppers_cb[wire+C]` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_every_graded_port_drive_uses_the_steppers_cb[wire]` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_field_per_unit_drive_does_not_depend_on_the_time_step` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_graded_and_uniform_lanes_give_the_same_field_per_unit_drive` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_lanes_agree_on_s11_with_a_capacitor_across_the_port_edge` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_mutation_a_drive_without_the_capacitor_fold_breaks_s11_parity` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_mutation_the_drive_without_the_ports_load_sends_both_red` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_the_build_check_sees_a_drive_without_the_load[lumped+C]` — lumped
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_the_build_check_sees_a_drive_without_the_load[lumped]` — lumped
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_the_build_check_sees_a_drive_without_the_load[wire+C]` — wire
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_the_build_check_sees_a_drive_without_the_load[wire]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.05-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.05-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.1-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.1-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.2-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.2-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.2001-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.2001-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.3-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.3-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.499999-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[0.499999-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[1e-09-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_every_positive_subcell_radius_refuses_before_scan[1e-09-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_filament_without_radius_port_refuses_on_dispersive_board[debye-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_filament_without_radius_port_refuses_on_dispersive_board[lorentz-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_legacy_zero_radius_filament_runs_with_either_port[5e-05-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_legacy_zero_radius_filament_runs_with_either_port[5e-05-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_legacy_zero_radius_filament_runs_with_either_port[None-forward-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_legacy_zero_radius_filament_runs_with_either_port[None-run-nonuniform]` — wire
- `tests/unit/ports/test_pec_filament_radius.py::test_outer_jit_cannot_silently_drop_a_declared_filament_radius[nonuniform]` — wire
- `tests/unit/ports/test_wire_port_live_mid_764.py::test_nu_spec_mid_is_live_run_midpoint` — wire
- `tests/unit/ports/test_wire_port_radius.py::test_fixed_radius_parallel_plate_oracle[nonuniform]` — wire
- `tests/unit/ports/test_wire_port_radius.py::test_radius_forward_material_gradient[nonuniform]` — wire
- `tests/unit/ports/test_wire_port_radius.py::test_radius_none_full_field_history_is_bit_identical[nonuniform]` — wire
- `tests/unit/ports/test_wire_port_radius.py::test_radius_refuses_axial_grading_before_step` — wire
- `tests/unit/ports/test_wire_port_radius.py::test_radius_refuses_dispersive_board_before_step[debye-nonuniform]` — wire
- `tests/unit/ports/test_wire_port_radius.py::test_radius_refuses_dispersive_board_before_step[lorentz-nonuniform]` — wire
- `tests/unit/ports/test_wire_port_radius.py::test_radius_refuses_local_transverse_grading` — wire
- `tests/unit/ports/test_wire_port_radius.py::test_unsupported_radius_path_refuses_before_first_step[distributed_nu]` — wire
- `tests/unit/preflight/test_half_node_split.py::test_a_float32_port_end_straddling_the_half_node_of_a_sheet_is_refused[nu-0.0035]` — wire
- `tests/unit/preflight/test_half_node_split.py::test_a_float32_port_end_straddling_the_half_node_of_a_sheet_is_refused[nu-0.0505]` — wire
- `tests/unit/preflight/test_half_node_split.py::test_on_the_nu_lane_a_feature_at_a_half_node_lands_on_its_conductor[dipole]` — wire
- `tests/unit/preflight/test_half_node_split.py::test_on_the_nu_lane_a_feature_at_a_half_node_lands_on_its_conductor[patch]` — wire
- `tests/unit/preflight/test_half_node_split.py::test_the_message_names_both_features_and_both_nodes[nu]` — wire
- `tests/unit/preflight/test_half_node_split.py::test_the_same_model_on_a_node_runs[dipole-nu]` — wire
- `tests/unit/preflight/test_half_node_split.py::test_the_same_model_on_a_node_runs[patch-nu]` — wire
- `tests/unit/preflight/test_nonuniform_wire_port_centers.py::test_graded_wire_port_centers_and_unavailable_classification[x]` — wire
- `tests/unit/preflight/test_nonuniform_wire_port_centers.py::test_graded_wire_port_centers_and_unavailable_classification[y]` — wire
- `tests/unit/preflight/test_nonuniform_wire_port_centers.py::test_graded_wire_port_centers_and_unavailable_classification[z]` — wire
- `tests/unit/runners/test_distributed_run_result_finishing.py::test_graded_lumped_port_default_is_no_s_request` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary_spec-conformal_s_matrix-fwd_distributed_nu]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary_spec-conformal_s_matrix-fwd_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_boundary_spec-conformal_s_matrix-run_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-lumped_port-fwd_distributed_nu]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-lumped_port-fwd_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-lumped_port-run_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-passive_port-fwd_distributed_nu]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-passive_port-fwd_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-passive_port-run_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-wire_port-fwd_distributed_nu]` — wire
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-wire_port-fwd_nonuniform]` — wire
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-wire_port-run_nonuniform]` — wire
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_boundary_spec-conformal_s_matrix-fwd_distributed_nu]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_boundary_spec-conformal_s_matrix-fwd_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_boundary_spec-conformal_s_matrix-run_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-lumped_port-fwd_distributed_nu]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-lumped_port-fwd_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-lumped_port-run_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-passive_port-fwd_distributed_nu]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-passive_port-fwd_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-passive_port-run_nonuniform]` — lumped
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-wire_port-fwd_distributed_nu]` — wire
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-wire_port-fwd_nonuniform]` — wire
- `tests/unit/runners/test_path_disposition_cells.py::test_the_rows_detector_fires_on_the_cells_model[_ports-wire_port-run_nonuniform]` — wire
- `tests/unit/runners/test_realized_model_cells.py::test_realized_cell[lumped_none-fwd_distributed_nu-x]` — lumped
- `tests/unit/runners/test_realized_model_cells.py::test_realized_cell[lumped_none-fwd_nonuniform-x]` — lumped
- `tests/unit/runners/test_realized_model_cells.py::test_realized_cell[lumped_none-run_nonuniform-x]` — lumped
- `tests/unit/runners/test_realized_model_cells.py::test_realized_cell[wire_none-fwd_distributed_nu-x]` — wire
- `tests/unit/runners/test_realized_model_cells.py::test_realized_cell[wire_none-fwd_nonuniform-x]` — wire
- `tests/unit/runners/test_realized_model_cells.py::test_realized_cell[wire_none-run_nonuniform-x]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_direct_runner_s_request_refused[None-True]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_direct_runner_s_request_refused[True-True]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_forward_port_s11_with_plain_source_refused[True-True]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_forward_port_s11_without_plain_source_runs[True-True]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_plain_source_s_request_refused[cpml-None-run_nonuniform-True-False]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_plain_source_s_request_refused[cpml-True-run_nonuniform-True-False]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_plain_source_s_request_refused[pec-None-run_nonuniform-True-False]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_plain_source_s_request_refused[pec-True-run_nonuniform-True-False]` — wire
- `tests/unit/sparams/test_plain_source_refusal.py::test_plain_source_without_s_request_runs[True-True-False]` — wire
- `tests/unit/sparams/test_ringdown_identification.py::test_distributed_lane_refuses_identification_probes[forward]` — wire
- `tests/unit/sparams/test_ringdown_identification.py::test_distributed_lane_refuses_identification_probes[run]` — wire
- `tests/unit/sparams/test_ringdown_identification.py::test_early_stop_identification_uses_extra_channels[graded]` — wire
- `tests/unit/sparams/test_ringdown_identification.py::test_identification_channels_reach_both_pencils_and_preserve_ports[graded]` — wire
- `tests/unit/sparams/test_ringdown_identification.py::test_identification_probe_outside_interior_refuses[position0-graded]` — wire
- `tests/unit/sparams/test_ringdown_identification.py::test_identification_probe_outside_interior_refuses[position1-graded]` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_a_graded_port_on_an_absorbing_floor_is_probed_inside_the_grid` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_a_short_record_completed_is_the_long_record[graded]` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_a_three_cell_port_completes_its_whole_gap_voltage[graded]` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_the_driven_column_of_a_two_port_graded_box` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_the_error_witness_fails_a_graded_record_cut_at_a_fifth_of_its_decay_time` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_the_passivity_witness_reads_the_plain_record_too` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_w0_and_byte_identity_hold_under_scoped_x64[graded]` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_w0_holds_and_the_completion_has_the_shape_of_the_run[graded]` — wire
- `tests/unit/sparams/test_ringdown_run.py::test_what_the_completion_does_not_cover_is_refused[_case_graded_loop_in_the_pad-absorber pad]` — wire
- `tests/unit/sparams/test_twoport_wire_port.py::test_direction_does_not_change_the_s_matrix` — wire
- `tests/unit/sparams/test_twoport_wire_port.py::test_realized_conductor_planes_equal_the_declaration` — wire
- `tests/unit/sparams/test_twoport_wire_port.py::test_two_port_s_envelope_on_matched_line` — wire
- `tests/unit/sparams/test_twoport_wire_port.py::test_two_port_s_matrix_has_nonzero_s21` — wire

## Failing test IDs

- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-lumped_port-run_nonuniform]`
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-lumped_port-fwd_nonuniform]`
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-wire_port-run_nonuniform]`
- `tests/unit/runners/test_path_disposition_cells.py::test_cell[_ports-wire_port-fwd_nonuniform]`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_mutation_the_drive_from_the_drawn_arrays_sends_1_to_3_red`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_every_graded_port_drive_under_an_override_uses_the_steppers_cb[wire]`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_every_graded_port_drive_under_an_override_uses_the_steppers_cb[lumped]`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_every_graded_port_drive_under_an_override_uses_the_steppers_cb[wire+C]`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_every_graded_port_drive_under_an_override_uses_the_steppers_cb[lumped+C]`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_build_check_sees_a_drive_from_the_drawn_arrays[wire]`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_build_check_sees_a_drive_from_the_drawn_arrays[lumped]`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_build_check_sees_a_drive_from_the_drawn_arrays[wire+C]`
- `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py::test_the_build_check_sees_a_drive_from_the_drawn_arrays[lumped+C]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_graded_and_uniform_lanes_give_the_same_field_per_unit_drive`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_mutation_the_drive_without_the_ports_load_sends_both_red`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_every_graded_port_drive_uses_the_steppers_cb[wire]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_every_graded_port_drive_uses_the_steppers_cb[lumped]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_every_graded_port_drive_uses_the_steppers_cb[wire+C]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_every_graded_port_drive_uses_the_steppers_cb[lumped+C]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_the_build_check_sees_a_drive_without_the_load[wire]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_the_build_check_sees_a_drive_without_the_load[lumped]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_the_build_check_sees_a_drive_without_the_load[wire+C]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_the_build_check_sees_a_drive_without_the_load[lumped+C]`
- `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py::test_mutation_a_drive_without_the_capacitor_fold_breaks_s11_parity`
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_lane_on_uniform_cells_matches_the_uniform_lane[lumped-debye]`
- `tests/unit/nonuniform/test_dispersive_port_load_1257.py::test_graded_lane_on_uniform_cells_matches_the_uniform_lane[lumped-lorentz]`
