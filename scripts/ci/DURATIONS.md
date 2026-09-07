# .test_durations — provenance

Generated 2026-09-07 for pull request 939. Every value is the measured pytest-split duration
plus a 0.3 s per-test overhead floor (pytest-split stores no setup/teardown below its
threshold; on the runners the real cost of a sub-millisecond test is 0.2–0.4 s).

Sources, by nodeid count (7594 entries, exactly the union of the fast, slow and highmem
selections at main 0b7e4294):

- 7484 measured on the GitHub runners by `regen-durations` run 34090225242 (fast shards 1–6,
  slow shards 1, 3, 4; slow shard 2 was killed by the runner at a 14.9 GB peak). The 8
  highmem tests are among these counts' complement below.
- 8 highmem tests measured on the a6000 VESSL lane (run 369367259054).
- 7 measured on lab CPU pods (same commands; the pods crashed natively on four shards).
- 79 carried from the previous file where the nodeid still collects (9 of them
  `tests/unit/geometry/test_mesh_import.py`, which the measurement lane skipped for lack of
  the `cad` extra — fixed in the workflow).
- 24 estimated (no measurement reached them: slow shard 2's tail): the file's previous mean
  where the file had entries, else 60 s. Listed below so a reader can tell them apart; a
  dispatch of `regen-durations` from main replaces them.

Regenerate: dispatch `.github/workflows/regen-durations.yml`, download the artifacts, then
`python scripts/ci/merge_test_durations.py .test_durations <shard json files...>` and re-add the
0.3 s floor (or fold it into the script when the next regeneration lands).

Estimated entries (24), before the +0.3 s floor:

- `tests/crossval/test_patch_canonical_farfield_e4.py::test_mode_pair_present_with_aspect_ratio` — 0.003 s (mean of the file's previous entries)
- `tests/locks/test_patch_edgefed_resonance_harminv.py::test_leg_a_isolated_patch_discretization_bias` — 60 s (no previous entry for the file)
- `tests/locks/test_patch_edgefed_resonance_harminv.py::test_leg_b_edge_feed_pull` — 60 s (no previous entry for the file)
- `tests/locks/test_patch_edgefed_resonance_harminv.py::test_patch_mode_is_in_the_patch_band_and_dominates_the_feed_band` — 60 s (no previous entry for the file)
- `tests/locks/test_patch_edgefed_resonance_harminv.py::test_ringdowns_are_settled` — 60 s (no previous entry for the file)
- `tests/locks/test_sheet_resonance_position_ab.py::test_g1_resonance_position_ab` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_alpha_envelope_regression_lock` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_alpha_oracle_o3` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_o3_model_fits_measured_field` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_o4a_guide_ratio_regression_lock` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_pec_control_o4c` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_sheet_transmission_matches_closed_form` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_sqrt_sigma_discriminator_o4a` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_sqrt_sigma_discriminator_o4a_guide_leg` — 60 s (no previous entry for the file)
- `tests/oracle/test_leontovich_alpha_oracle.py::test_thickness_invariance_o4b` — 60 s (no previous entry for the file)
- `tests/oracle/test_sheet_perturbation_q.py::test_g2_loss_is_real` — 60 s (no previous entry for the file)
- `tests/oracle/test_sheet_perturbation_q.py::test_g2_perturbation_q_band` — 60 s (no previous entry for the file)
- `tests/oracle/test_waveguide_chain_battery.py::test_live_ad_vs_fd_slab_s21_mag2_coarse_rung` — 0.005 s (mean of the file's previous entries)
- `tests/oracle/test_waveguide_chain_battery_v18_close.py::test_live_cells_reproduce_the_fixture_cpu[coarse]` — 60 s (no previous entry for the file)
- `tests/oracle/test_waveguide_chain_battery_v18_close.py::test_live_cells_reproduce_the_fixture_cpu[mid]` — 60 s (no previous entry for the file)
- `tests/oracle/test_waveguide_chain_battery_v18_close.py::test_live_plane_shift_rotation_coarse_rung[false]` — 60 s (no previous entry for the file)
- `tests/oracle/test_waveguide_chain_battery_v18_close.py::test_live_plane_shift_rotation_coarse_rung[flux]` — 60 s (no previous entry for the file)
- `tests/unit/geometry/test_mesh_import.py::test_mid_plane_registered_sheet_deterministic_and_correct_both_sides` — 0.378 s (mean of the file's previous entries)
- `tests/unit/geometry/test_mesh_import.py::test_surface_on_nodes_matches_box_convention_both_sides` — 0.378 s (mean of the file's previous entries)
