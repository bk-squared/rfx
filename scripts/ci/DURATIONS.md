# .test_durations — provenance

Generated 2026-09-07 for pull request 939. Every value is the measured pytest-split duration
plus a 0.3 s per-test overhead floor (pytest-split stores no setup/teardown below its
threshold; on the runners the real cost of a sub-millisecond test is 0.2–0.4 s).

Sources, by nodeid count (7594 entries, exactly the union of the fast, slow and highmem
selections at main 0b7e4294):

- 7495 measured on the GitHub runners by `regen-durations` runs 34090225242 and 34110728423
  (the second pass wins where both measured a test; in both passes slow shard 2 was killed by the
  runner before it finished, see issue 940).
- 8 highmem tests measured on the a6000 VESSL lane (run 369367259054) are among the entries above
  where the runner also measured them; 7 entries come only from lab pods.
- 70 carried from the previous file where the nodeid still collects.
- 22 estimated (no measurement reached them): the file's previous mean where the file had
  entries, else 60 s. Listed below; a dispatch of `regen-durations` after the runner-killing
  tests are marked `highmem` replaces them.

Regenerate: dispatch `.github/workflows/regen-durations.yml`, download the artifacts, then
`python scripts/ci/merge_test_durations.py .test_durations <shard json files...>` and re-add the
0.3 s floor (or fold it into the script when the next regeneration lands).

Estimated entries (22), before the +0.3 s floor:

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
