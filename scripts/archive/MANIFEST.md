# scripts/archive/ — closed-investigation one-off manifest

These scripts are **archived diagnostics**, not live machinery. They were
one-off investigation tooling tied to closed or superseded RF/EM
investigations (WR-90 waveguide-port extractor, V/I-split port camp, TFSF
leakage probes, slab S11 phase, issue #31/#40/#44/#48 nonuniform-mesh +
patch FFRP, Y2 Kottke-dilation gradient, and their VESSL job specs). They
were swept here by roadmap task **W6.7** to keep `scripts/` (the
grep-first feature-discovery surface mandated by
`.claude/rules/rfx-feature-discovery.md`) uncluttered.

Nothing here is referenced by CI, tests, the public-docs export pipeline,
or `scripts/diagnostics/port_external_reference_requirements.json`. They
are kept (not deleted) for git-history provenance and in case a closed
investigation is reopened. Do **not** treat anything here as a supported
entry point.

One line per file: `filename — investigation/topic — approx date (last
commit, %as) — status`.

| File | Investigation / topic | Date | Status |
|------|-----------------------|------|--------|

| `isolate_extractor_vs_engine.py` | WR-90 waveguide-port extractor vs FDTD-engine isolation | 2026-04-22 | closed |
| `waveguide_port_canonical_diagnostics.py` | WR-90 waveguide-port canonical diagnostics | 2026-04-22 | closed |
| `s11_pec_short_resolution_sweep.py` | WR-90 PEC-short |S11| resolution sweep | 2026-04-24 | closed |
| `diagnose_pec_short_8pct_residual.py` | WR-90 PEC-short 8% residual (diagnostic-comparator artefact) | 2026-04-25 | closed |
| `diagnose_pec_wall_thickness.py` | WR-90 PEC-short wall-thickness diagnostic | 2026-04-25 | closed |
| `diagnose_pec_boundary_vs_mask.py` | WR-90 PEC boundary-vs-mask diagnostic | 2026-04-26 | closed |
| `diagnose_pec_mask_variants.py` | WR-90 PEC mask-variant diagnostic | 2026-04-26 | closed |
| `diagnose_pec_sigma_sweep.py` | WR-90 PEC sigma sweep | 2026-04-26 | closed |
| `diagnose_pec_sigma_test.py` | WR-90 PEC sigma test | 2026-04-26 | closed |
| `_pec_edge_zeroing_test.py` | WR-90 PEC edge-zeroing test | 2026-04-26 | closed |
| `_pec_face_style_runner.py` | WR-90 PEC face-style runner | 2026-04-26 | closed |
| `_pec_sigma_runner_inner.py` | WR-90 PEC sigma runner (inner) | 2026-04-26 | closed |
| `_pec_variant_runner.py` | WR-90 PEC variant runner | 2026-04-26 | closed |
| `_aperture_fractional_test.py` | WR-90 port aperture fractional-weight test | 2026-04-26 | closed |
| `_aperture_trapezoidal_test.py` | WR-90 port aperture trapezoidal-weight test | 2026-04-26 | closed |
| `_aperture_trim_test.py` | WR-90 port aperture DROP-weight trim test | 2026-04-26 | closed |
| `diagnose_mask_missing_cells.py` | WR-90 port mask missing-cells diagnostic | 2026-04-26 | closed |
| `diagnose_mode_profile_swap.py` | WR-90 port mode-profile swap diagnostic | 2026-04-26 | closed |
| `diagnose_gated_DFT.py` | WR-90 port gated-DFT diagnostic | 2026-04-26 | closed |
| `diagnose_probe_offset_sweep.py` | WR-90 port probe-offset sweep | 2026-04-26 | closed |
| `diagnose_VI_vs_Z.py` | WR-90 port V/I-vs-Z extractor diagnostic | 2026-04-25 | closed |
| `diagnose_VI_at_wall.py` | WR-90 port V/I-at-wall diagnostic | 2026-04-26 | closed |
| `diagnose_VI_compare_A_vs_B2.py` | WR-90 port V/I A-vs-B2 comparison | 2026-04-26 | closed |
| `diagnose_VI_compare_visualize.py` | WR-90 port V/I comparison visualization | 2026-04-26 | closed |
| `diagnose_VI_phase_compare.py` | WR-90 port V/I phase comparison | 2026-04-26 | closed |
| `diagnose_VI_timeseries_dump.py` | WR-90 port V/I time-series dump | 2026-04-26 | closed |
| `diagnose_VI_timeseries_visualize.py` | WR-90 port V/I time-series visualization | 2026-04-26 | closed |
| `diagnose_field_past_wall_check.py` | WR-90 port field-past-wall check | 2026-04-26 | closed |
| `diagnose_field_trace_at_wall.py` | WR-90 port field-trace-at-wall diagnostic | 2026-04-26 | closed |
| `diagnose_field_trace_at_wall_v2.py` | WR-90 port field-trace-at-wall diagnostic (v2) | 2026-04-26 | closed |
| `diagnose_field_trace_at_wall_v3.py` | WR-90 port field-trace-at-wall diagnostic (v3) | 2026-04-26 | closed |
| `diagnose_tfsf_directional_leakage.py` | TFSF directional-leakage diagnostic | 2026-04-25 | closed |
| `diagnose_tfsf_upstream_probe.py` | TFSF upstream-probe diagnostic | 2026-04-25 | closed |
| `soft_e_vs_tfsf_phase_offset.py` | soft-E vs TFSF phase-offset diagnostic | 2026-04-24 | closed |
| `diagnose_setupE_no_cpml_no_box.py` | WR-90 port setup-E (no CPML / no box) diagnostic | 2026-04-26 | closed |
| `diagnose_setup_I_1cell.py` | WR-90 port setup-I 1-cell diagnostic | 2026-04-26 | closed |
| `diagnose_setup_combinations.py` | WR-90 port setup-combination sweep | 2026-04-26 | closed |
| `rfx_vs_analytic_slab_phase.py` | slab S11 rfx-vs-analytic phase comparison | 2026-04-24 | superseded |
| `slab_physical_diagnostics.py` | slab physical diagnostics | 2026-04-22 | closed |
| `slab_cpml_sweep_magnitude.py` | slab CPML magnitude sweep | 2026-04-24 | closed |
| `slab_resolution_sweep_magnitude.py` | slab resolution magnitude sweep | 2026-04-24 | closed |
| `subpixel_phase_test.py` | slab subpixel-phase test | 2026-04-22 | closed |
| `phase_offset_per_geometry.py` | port phase-offset per-geometry diagnostic | 2026-04-22 | closed |
| `meep_alpha_reference_plane.py` | Meep alpha reference-plane comparison | 2026-04-24 | closed |
| `cpml_reflectivity_sweep.py` | CPML reflectivity sweep | 2026-04-07 | closed |
| `gate_b2_check.py` | WR-90 port B2-status gate check | 2026-05-16 | superseded |
| `issue31_ad_memory_sweep.py` | issue #31 segmented-checkpoint AD-memory sweep | 2026-04-15 | closed |
| `issue31_big_smoke.py` | issue #31 big-smoke run | 2026-04-15 | closed |
| `issue31_nu_physics_pin.py` | issue #31 nonuniform-physics pin | 2026-04-15 | closed |
| `issue31_ffrp_deepdive.py` | issue #31 far-field/FFRP deep-dive | 2026-04-16 | closed |
| `issue31_patch_validation.py` | issue #31 patch-antenna validation | 2026-04-16 | closed |
| `convergence_grid.py` | nonuniform convergence-grid harness | 2026-04-16 | closed |
| `issue40_nwarmup_physics.py` | issue #40 n_warmup physics sweep | 2026-04-16 | closed |
| `issue48_pec_mask_diagnostic.py` | issue #48 PEC-mask diagnostic | 2026-04-16 | closed |
| `issue48_uniform_patch_ffrp.py` | issue #48 uniform-patch FFRP | 2026-04-16 | closed |
| `validate_codex_fixes.py` | Codex-fix validation one-off | 2026-04-08 | closed |
| `gpu_verify_issues.py` | GPU issue-verification one-off | 2026-04-06 | closed |
| `test_patch_nonuniform.py` | nonuniform patch ad-hoc test (not pytest-collected) | 2026-04-06 | closed |
| `y2_dx_diagnostic.py` | Y2 nonuniform dx diagnostic | 2026-05-07 | closed |
| `y2_stub_mode_diagnostic.py` | Y2 stub-mode diagnostic | 2026-05-07 | closed |
| `y2_dx_alignment_sweep.py` | Y2 dx-alignment sweep | 2026-05-08 | closed |
| `y2_extractor_agreement.py` | Y2 extractor-agreement check | 2026-05-08 | closed |
| `y2_grad_fd_delta_sweep.py` | Y2 gradient FD-delta sweep | 2026-05-08 | closed |
| `y2_grad_fd_vs_ad.py` | Y2 gradient FD-vs-AD comparison | 2026-05-08 | closed |
| `y2_grad_localize.py` | Y2 gradient localization | 2026-05-08 | closed |
| `y2_grad_option3.py` | Y2 gradient option-3 diagnostic | 2026-05-08 | closed |
| `y2_kottke_debug.py` | Y2 Kottke-dilation debug | 2026-05-09 | closed |
| `vessl_nu_diagnostic.yaml` | VESSL: nonuniform diagnostic job | 2026-04-07 | superseded |
| `vessl_v150_validation.yaml` | VESSL: v1.5.0 validation job | 2026-04-12 | superseded |
| `vessl_phase_c_validation.yaml` | VESSL: Phase-C validation job | 2026-04-13 | superseded |
| `vessl_issue31_big_smoke.yaml` | VESSL: issue #31 big-smoke job | 2026-04-15 | superseded |
| `vessl_issue31_nu_physics.yaml` | VESSL: issue #31 nonuniform-physics job | 2026-04-15 | superseded |
| `vessl_issue31_physics.yaml` | VESSL: issue #31 physics job | 2026-04-15 | superseded |
| `vessl_issue31_segmented.yaml` | VESSL: issue #31 segmented-checkpoint job | 2026-04-15 | superseded |
| `vessl_issue31_smoke_sweep.yaml` | VESSL: issue #31 smoke-sweep job | 2026-04-15 | superseded |
| `vessl_convergence_grid.yaml` | VESSL: convergence-grid job | 2026-04-16 | superseded |
| `vessl_convergence_proper.yaml` | VESSL: convergence (proper) job | 2026-04-16 | superseded |
| `vessl_issue31_patch_viz.yaml` | VESSL: issue #31 patch-viz job | 2026-04-16 | superseded |
| `vessl_issue40_nwarmup_physics.yaml` | VESSL: issue #40 n_warmup-physics job | 2026-04-16 | superseded |
| `vessl_issue48_deepdive.yaml` | VESSL: issue #48 deep-dive job | 2026-04-16 | superseded |
| `vessl_issue48_fine_mesh.yaml` | VESSL: issue #48 fine-mesh job | 2026-04-16 | superseded |
| `vessl_issue48_large_ground.yaml` | VESSL: issue #48 large-ground job | 2026-04-16 | superseded |
| `vessl_issue48_uniform.yaml` | VESSL: issue #48 uniform job | 2026-04-16 | superseded |
| `vessl_issue44_job0.yaml` | VESSL: issue #44 job-0 | 2026-04-17 | superseded |
| `vessl_issue44_job_a.yaml` | VESSL: issue #44 job-A | 2026-04-17 | superseded |
| `vessl_issue44_job_a_v2.yaml` | VESSL: issue #44 job-A (v2) | 2026-04-17 | superseded |
| `vessl_issue44_job_a_v3.yaml` | VESSL: issue #44 job-A (v3) | 2026-04-17 | superseded |
| `vessl_issue31_crossval_regression.yaml` | VESSL: issue #31 crossval-regression job | 2026-04-20 | superseded |
| `vessl_v150_gpu_validation.yaml` | VESSL: v1.5.0 GPU-validation job | 2026-04-20 | superseded |
