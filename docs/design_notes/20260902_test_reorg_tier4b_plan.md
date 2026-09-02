# Test-suite reorganization — tier 4b plan (oracle + unit)

Date: 2026-09-02. Branch `agent/reorg-tier4b-unit`, stacked on tier 4a
(`agent/reorg-tier4a-studio`). Approved by the PI on 2026-09-02 as the last
MOVE tier: everything still at top level `tests/test_*.py` after tiers 1-4a
(390 files) moves into `tests/oracle/` (flat) or `tests/unit/<module>/`.
MOVES ONLY — `git mv`, basenames unchanged, no content rewrites beyond the
path / import fixes listed under "Mechanical fixes". One file per physics
law inside `tests/oracle/` is tier 3b's job, not this one.

## Classification rules

* `tests/oracle/` — the subject is a physical law or closed form (Mie,
  Fresnel/TMM, Pozar cavity, Hammerstad-Jensen, skin depth, numerical
  dispersion, Hertzian dipole, conservation laws, PML reflectivity, Leontovich,
  Kerr SPM, ...). Decided from the module docstring first, then the name
  tokens `oracle|analytic|validation|physics|fresnel|mie|cavity|dispersion|`
  `conservation|reflectivity|skin_depth|groundtruth|battery|convergence`.
  A name token alone was NOT enough: e.g. `test_lumped_twoport_vi_validation_battery`
  (Z0-normalization algebra) and `test_coaxial_line_calibration` (extractor
  calibration) stay in `unit/sparams`; `test_cpml` (mechanism + reference
  reflection gate) stays in `unit/boundaries`.
* `tests/unit/<module>/` — decided from the dominant `rfx.<module>` import
  and the name. Tie-break conventions used consistently:
  - name/subject is a gradient, forward(), optimize, objective, checkpoint-
    segments or AD-memory question → `autodiff` (even when the lane is NU,
    S-parameter or waveguide);
  - port *primitive*, mode solver, port preflight → `ports`; S-matrix
    extraction / assembly / normalization / passivity / settling / replay →
    `sparams`;
  - `nonuniform_*` / `nu_*` / mesh-planner / grading → `nonuniform`, unless
    the subject is an analytic cavity (→ oracle) or a gradient (→ autodiff);
  - `distributed*`, `vmap*`, batch/sweep, until_decay, progress, GPU, ADI →
    `runners`; `ntff_*`/`rcs*`/`farfield_*` mechanisms → `farfield`;
  - core Yee kernel options with no module dir of their own: stencil order →
    `grid` (discretization), aniso inv-eps update → `geometry` (Stage-2
    smoothing), Bloch complex path → `boundaries` (periodic/bloch);
  - `rfx.probes` has no approved directory: its three pure probe tests go to
    `misc` rather than inventing `unit/probes`.

## Counts

| destination | files |
|---|---|
| `tests/oracle/` | 40 |
| `tests/unit/grid/` | 9 |
| `tests/unit/geometry/` | 19 |
| `tests/unit/materials/` | 17 |
| `tests/unit/boundaries/` | 20 |
| `tests/unit/sources/` | 6 |
| `tests/unit/ports/` | 26 |
| `tests/unit/sparams/` | 62 |
| `tests/unit/farfield/` | 12 |
| `tests/unit/nonuniform/` | 25 |
| `tests/unit/subgrid/` | 22 |
| `tests/unit/runners/` | 30 |
| `tests/unit/autodiff/` | 57 |
| `tests/unit/preflight/` | 18 |
| `tests/unit/api/` | 16 |
| `tests/unit/misc/` | 11 |
| total | 390 |

`misc` members: `test_dft_probes`, `test_flux_monitor_finite_size`, `test_flux_silent_zero_guard`, `test_harminv_decimation`, `test_harminv_estimator`, `test_jax_checks`, `test_ladder_guard`, `test_review_tier1_validation_battery`, `test_ris`, `test_v173a_physics_equivalence_slow`, `test_weekly_rss_reporter`.

## Mechanical fixes applied inside moved files

Top level → `tests/oracle/` is ONE extra level; top level → `tests/unit/<m>/`
is TWO. Line counts are never changed (evidence-citation `file.py:LINE`
pointers must keep holding).

* `Path(__file__).resolve().parents[1]` → `parents[2]` (oracle) / `parents[3]` (unit);
  `.parent.parent` → the same `parents[N]`.
* `Path(__file__).parent / "fixtures"` → `Path(__file__).parents[2] / "fixtures"`;
  `os.path.join(os.path.dirname(__file__), "fixtures", ...)` →
  `os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", ...)`;
  `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` gains the
  extra `os.path.dirname(...)` wrappers.
* `from tests.test_x import` / `tests.test_x` nodeids → the new dotted or
  slash path (`tests.unit.<m>.test_x`, `tests.oracle.test_x`). Bare sibling
  imports (`from test_coax_msl_transition import`, `import test_coax_msl_transition`)
  are kept where the sibling moved to the same directory (sparams); the one
  cross-directory bare import (`test_waveguide_settling_witness` →
  `test_simulation`) becomes `from tests.unit.runners.test_simulation import`.
* `"tests/test_x.py"` string literals (subprocess pytest args, nodeid
  strings, assertion messages, docstrings) → the new path.
* `tests/_*.py` helper imports were already in package form
  (`from tests._x64_compat import ...`) and need no change.

## References updated outside the moved files

Every `tests/test_<name>` (slash form, with or without `.py`) and
`tests.test_<name>` (dotted form) whose `<name>` is in the table below is
rewritten repo-wide: `.github/workflows/pr-tests.yml` (the guards-and-
preflight FILES list, 11 files, and the conformal tripwire nodeid),
`validation.yml` (comment), `.test_durations` (keys rewritten, none dropped),
`validation/crossval/manifest.json` gate_paths, `validation/README.md`,
`docs/**`, `rfx/**` docstrings, `scripts/**` (incl. `run_physics_gate.py` and
`scripts/archive/*.yaml`), `validation/**`, `examples/**`, `conftest.py`,
`tests/_*.py`, `tests/{locks,contracts,crossval,studio}/**`, `tests/fixtures/**`.
`CHANGELOG.md` is left as a historical record (same as tiers 1-4a). References
to files that no longer exist (e.g. `tests/test_msl_plane_extractor_jax.py`)
are left untouched. `pyproject.toml` `testpaths = ["tests"]` is unchanged.

## File → destination

### `tests/oracle/` (40)

| file | reason |
|---|---|
| `test_boundary_cpml_oracle.py` | asymmetric-CPML sigma_max analytic oracle |
| `test_boundary_pmc_oracle.py` | PMC lambda/4 cavity mode-ladder vs analytic |
| `test_cavity.py` | PEC rectangular cavity TM_mnp resonance vs closed form |
| `test_conformal_convergence.py` | conformal-PEC convergence order vs TM010 cylindrical cavity analytic |
| `test_conservation_laws.py` | passivity/unitarity/reciprocity/causality invariants |
| `test_dispersive_fresnel_validation.py` | dispersive slab R(f) vs rigorous Fresnel/TMM |
| `test_farfield.py` | Hertzian dipole sin(theta) pattern and 1.76 dBi directivity |
| `test_forward_tfsf_fresnel_groundtruth.py` | complex Fresnel Gamma(f0) ground-truth gate |
| `test_fresnel_investigation.py` | oblique TFSF Fresnel tolerance diagnostic |
| `test_kerr_spm_absolute_oracle.py` | Kerr SPM absolute-magnitude oracle |
| `test_kerr_spm_fingerprint.py` | Kerr SPM sign / chi3 A^2 scaling fingerprint |
| `test_leontovich_alpha_oracle.py` | parallel-plate conductor attenuation vs Leontovich |
| `test_lossy_material_validation.py` | plane-wave alpha(f) and cavity Q vs exact lossy-medium analytics |
| `test_lossy_reflection_oracle.py` | lossy half-space Gamma vs complex-eps Fresnel |
| `test_magnetic.py` | mu_r slab Fresnel |R|=1/3 and phase velocity c/sqrt(mu eps) |
| `test_microstrip.py` | Hammerstad-Jensen closed-form microstrip (Pozar 3.8) |
| `test_nonuniform_cavity_accuracy.py` | NU grid vs analytic PEC cavity TM111 |
| `test_nonuniform_convergence.py` | NU dz refinement converges to analytic resonance |
| `test_nonuniform_xy_cavity_accuracy.py` | in-plane NU grid vs analytic PEC cavity TM110 |
| `test_numerical_dispersion_oracle.py` | measured phase velocity vs Yee numerical-dispersion relation |
| `test_oblique_fresnel_magnitude.py` | oblique |Gamma|(theta) vs Fresnel r_TE |
| `test_oblique_fresnel_phase.py` | oblique complex Gamma(theta) vs Fresnel r_TE |
| `test_oblique_rcs_specular.py` | finite PEC plate specular-peak / normal-reduction physics |
| `test_patch_cavity_eps_oracle.py` | dielectric-interface subpixel cavity oracle |
| `test_physics.py` | Fresnel, effective-eps, six Codex physics scenarios |
| `test_physics_integrity.py` | discrete Maxwell equations satisfied (curl/divergence identities) |
| `test_pml_reflectivity.py` | CPML broadband reflectivity < -40 dB vs reference |
| `test_pml_reflectivity_upml.py` | UPML reflectivity physics oracle |
| `test_ram_magnetic_mu_r_design.py` | mu_r RAM design vs analytic magnetic TMM |
| `test_ram_multilayer_inverse_design.py` | multilayer RAM vs analytic TMM oracle |
| `test_rcs.py` | PEC plate vs physical optics, sphere vs analytic RCS |
| `test_rcs_mie_fixture.py` | PEC sphere monostatic RCS vs exact Mie series |
| `test_series_rlc_current.py` | series RLC f0=1/(2pi sqrt(LC)), Q closed form |
| `test_sheet_film_rta_analytic.py` | resistive film R/T/A vs exact conductive-slab solution |
| `test_sheet_perturbation_q.py` | conductor-loss perturbation-Q band on patch cavity |
| `test_skin_depth_oracle.py` | good-conductor skin depth vs analytic |
| `test_stage1_nu_physics_gate.py` | NU cavity physics gate (scripts/stage1_nu_cavity_physics_gate) |
| `test_waveguide_group_delay_near_cutoff.py` | group delay vs L_eff/v_g(f) near cutoff |
| `test_waveguide_group_delay_tolerance_envelope.py` | sibling of the group-delay oracle: its tolerance envelope |
| `test_waveguide_port_validation_battery.py` | falsifiable physical properties of the waveguide port (cutoff, reciprocity...) |

### `tests/unit/grid/` (9)

| file | reason |
|---|---|
| `test_auto_config.py` | auto_configure dx/dt/memory estimation |
| `test_dft_accumulator_dtype.py` | x64 dtype contract of DFT accumulators |
| `test_fourth_order_api.py` | stencil_order=(2,4) discretization option on Simulation/run/forward |
| `test_fourth_order_stencil.py` | (2,4) core Yee stencil kernel contract |
| `test_grid.py` | Grid class |
| `test_mixed_precision.py` | float16 fields / float32 accumulation precision |
| `test_precision_lane_guard.py` | precision= lane guard (fail loud on non-threading runners) |
| `test_stage5.py` | 2D grid mode, snapshots, HDF5 checkpoint |
| `test_x64_scan_carry_dtypes.py` | scan carries follow ambient precision |

### `tests/unit/geometry/` (19)

| file | reason |
|---|---|
| `test_conformal.py` | Dey-Mittra conformal PEC |
| `test_curved_patch.py` | geometry.curved.CurvedPatch |
| `test_fidelity_report.py` | rfx.fidelity declared-vs-realized report |
| `test_fidelity_topology_findings.py` | rfx.fidelity topology findings |
| `test_geometry.py` | CSG primitives / rasterize |
| `test_kottke_inv_eps_from_occupancy.py` | geometry.smoothing Kottke inv-eps |
| `test_kottke_pec_limit.py` | geometry.smoothing PEC limit |
| `test_mesh_import.py` | geometry.mesh_import MeshShape |
| `test_pcb.py` | rfx.pcb stackup builder (Box geometry) |
| `test_rasterization_coordinate_exactness.py` | rasterization node coordinates exact/lane-independent |
| `test_stage2_acceptance.py` | kottke_pec on curved PEC acceptance |
| `test_stage2_acceptance_ladder.py` | kottke_pec acceptance ladder |
| `test_stage2_dual_path.py` | aniso inv-eps (kottke_pec) dual-path wiring |
| `test_subpixel.py` | anisotropic subpixel smoothing |
| `test_subpixel_nonuniform.py` | subpixel smoothing on NU mesh |
| `test_subpixel_pec.py` | Stage 1 conformal PEC face-shift |
| `test_thin_wire.py` | Holland thin-wire subcell model |
| `test_update_e_aniso_inv.py` | update_e_aniso_inv kernel for Stage 2 smoothing |
| `test_via.py` | geometry.via |

### `tests/unit/materials/` (17)

| file | reason |
|---|---|
| `test_conductor_mask_accessor.py` | thin_conductor.conductor_footprint accessor |
| `test_debye.py` | Debye ADE |
| `test_dispersion_pole_keying.py` | dispersion pole mask keying |
| `test_kerr_api_paths.py` | Kerr chi3 on run()/forward() paths |
| `test_leontovich_sheet_identity.py` | surface_impedance_f0 default-off identity |
| `test_lorentz.py` | Lorentz/Drude ADE |
| `test_lossy.py` | finite-conductivity Ca/Cb path |
| `test_material_fit.py` | rfx.material_fit Debye/Lorentz fitting |
| `test_nonlinear.py` | Kerr nonlinear ADE |
| `test_sheet_impedance_operator.py` | node-thin surface-impedance sheet operator |
| `test_sheet_lane_fences.py` | surface-impedance sheet lane fences |
| `test_sheet_node_permittivity.py` | node-thin conductor live-edge permittivity |
| `test_sheet_stacked_adjacent_gap.py` | adjacent f0 sheets must not load the gap |
| `test_thin_conductor.py` | thin conductor subcell model |
| `test_thin_conductor_honesty.py` | add_thin_conductor PEC routing honesty |
| `test_thin_conductor_nonbox_sheet.py` | Leontovich sheets on non-Box shapes |
| `test_thin_conductor_nu_dual_spacing.py` | thin-conductor sheet resistance on graded node |

### `tests/unit/boundaries/` (20)

| file | reason |
|---|---|
| `test_boundary_cpml_asymmetric.py` | asymmetric per-face CPML mechanism |
| `test_boundary_pmc_composition.py` | TFSF+PMC / NTFF+PMC composition |
| `test_boundary_pmc_distributed.py` | PMC on sharded runners |
| `test_boundary_pmc_guard.py` | PMC phase-2 guard removal |
| `test_boundary_pmc_hi_faces.py` | PMC enforcement on _hi faces |
| `test_boundary_pmc_runtime.py` | apply_pmc_faces mechanism |
| `test_boundary_spec.py` | boundaries.spec type surface |
| `test_boundary_spec_cpml_budget.py` | cpml_layers allocation budget |
| `test_boundary_spec_legacy.py` | legacy API -> BoundarySpec shim |
| `test_boundary_spec_preflight.py` | preflight compatibility with BoundarySpec |
| `test_boundary_spec_thickness.py` | lo/hi_thickness spec + runtime |
| `test_cfs_cpml.py` | CFS-CPML kappa_max mechanism |
| `test_cpml.py` | CPML mechanism + reflection gate vs reference |
| `test_cpml_pad_face_notch.py` | one-cell vacuum notch at CPML face |
| `test_cpml_pad_material_extension.py` | CPML pad material extension |
| `test_crossval_migration_smoke.py` | BoundarySpec migration smoke of crossval scripts |
| `test_pec_mask_boundary_convention.py` | tangential-edge rule at domain face |
| `test_periodic_cpml.py` | periodic xy + CPML z |
| `test_pmc_plane_convention.py` | PMC-plane convention stated in every PMC script |
| `test_yee_bloch_complex_path.py` | Bloch-periodic complex yee path |

### `tests/unit/sources/` (6)

| file | reason |
|---|---|
| `test_custom_waveforms.py` | CW / CustomWaveform sources |
| `test_source_amplitude_kind.py` | add_source(amplitude_kind=) |
| `test_source_dc_floor_guards.py` | soft-source deposited-DC guards |
| `test_tfsf_oblique.py` | oblique TFSF 2D aux grid |
| `test_tfsf_oblique_coverage.py` | oblique TFSF ey polarization |
| `test_tfsf_run_oblique_integration.py` | run() drives oblique-periodic Bloch TFSF path |

### `tests/unit/ports/` (26)

| file | reason |
|---|---|
| `test_coaxial_port.py` | CoaxialPort abstraction |
| `test_diagnostic_harness_synthetic.py` | wr90 port comparator V/I projection harness |
| `test_eigenmode.py` | numerical eigenmode solver |
| `test_eigenmode_port.py` | eigenmode solver with waveguide ports |
| `test_floquet.py` | Floquet port + periodic BC |
| `test_floquet_s_params_contract.py` | compute_floquet_s_params contract + AD classification |
| `test_lumped_rlc.py` | lumped RLC element |
| `test_msl_eigenmode_solver.py` | MSL vectorial eigenmode solver |
| `test_msl_nu_abscissa.py` | MSL probe abscissa on graded mesh |
| `test_msl_port.py` | MSLPort primitive |
| `test_msl_port_axis_generality.py` | add_msl_port direction generality |
| `test_msl_port_preflight.py` | MSL port geometry preflight (pr-tests guard) |
| `test_msl_reflector_scan_conductors.py` | MSL downstream-reflector preflight scan |
| `test_msl_source_fixture_static.py` | MSL launch fixture from registered materials |
| `test_multimode_waveguide.py` | multi-mode waveguide port |
| `test_port_aperture_rasterization.py` | waveguide-port preflight reads rasterized geometry |
| `test_port_current_boundary_convention.py` | port_current out-of-domain convention |
| `test_port_metric_dual_face_nu.py` | lumped folds on NU lane realize nominal value |
| `test_port_preflight.py` | single-cell port in dielectric preflight (pr-tests guard) |
| `test_waveguide_geometry_hygiene.py` | waveguide S-param setup absorber/geometry hygiene |
| `test_waveguide_modes_extraction_contract.py` | sources._waveguide_modes import paths |
| `test_waveguide_port.py` | rectangular waveguide port TE/TM profiles |
| `test_waveguide_port_spectrum_guard.py` | waveguide source-spectrum defaults + guards |
| `test_wire_port.py` | multi-cell wire port |
| `test_wire_port_live_mid_764.py` | wire-port live-mid reference cell |
| `test_wr90_port_oracles.py` | scripts._wr90_port_oracle_matrix unit tests |

### `tests/unit/sparams/` (62)

| file | reason |
|---|---|
| `test_coax_msl_ladder_witnesses.py` | coax-MSL ladder witnesses post-processor |
| `test_coax_msl_transition.py` | compute_coax_msl_transition assembly + FDTD |
| `test_coax_msl_transition_ladder_dump.py` | return_ladder_voltages byte identity |
| `test_coax_msl_transition_wave_roles.py` | coax<->MSL assembler wave-role convention |
| `test_coax_two_port_fdtd.py` | compute_coaxial_two_port FDTD path |
| `test_coax_two_port_solve.py` | two-drive 2x2 coax solve |
| `test_coaxial_line_calibration.py` | coax TL reflection extractor calibration (short/open/match) |
| `test_coaxial_line_extraction.py` | coax TL reflection extractor on synthetic voltages |
| `test_coaxial_s_matrix.py` | deprecated compute_coaxial_s_matrix |
| `test_deembed.py` | S-parameter de-embedding |
| `test_extract_s_matrix_pec_mask.py` | extract_s_matrix honours PEC mask |
| `test_forward_run_s11_passivity_warn.py` | lumped/wire S11 passivity self-check |
| `test_lumped_port_sparams_jit.py` | JIT lumped-port S-parameter path |
| `test_lumped_twoport_vi_validation_battery.py` | lumped/wire V-I extraction + Z0 normalization post-processing |
| `test_lumped_wire_sparam_cpml_dielectric.py` | lumped/wire S extraction CPML-material-aware |
| `test_mixed_port_sparam.py` | mixed-family S-matrix power-wave lane |
| `test_mixed_refplane_artifact_adapter.py` | refplane artifact -> referee adapter |
| `test_mixed_refplane_measurement.py` | mixed-lane reference-plane measurement |
| `test_msl_beta_scan_rail.py` | N-probe beta-scan rail warning |
| `test_msl_internal_probe_advisories.py` | compute_msl_s_matrix internal witness probes |
| `test_msl_ladder_standoff.py` | MSL ladder standoff + self-consistency witness |
| `test_msl_modal_voltage_and_wave_solve.py` | msl_modal_voltage / msl_solve_s_from_waves |
| `test_msl_nprobe_extractor.py` | N-probe least-squares MSL extractor |
| `test_msl_null_reliability_mask.py` | MSL standing-wave-null reliability mask |
| `test_msl_passive_port_reflection_fit.py` | passive-port two-wave reflection fit |
| `test_msl_passivity_enforcement.py` | compute_msl_s_matrix passivity projection |
| `test_msl_plane_primitives_parity.py` | MSL plane-primitive V/I parity |
| `test_msl_plane_primitives_smoke.py` | MSL plane-probe primitives smoke |
| `test_msl_port_integration.py` | MSL thru-line S-parameter passivity gate |
| `test_msl_probe_offset_interval.py` | _resolve_msl_auto_offsets |
| `test_msl_settling_witness.py` | MSL ring-down settling witness |
| `test_msl_sheet_threading.py` | surface-impedance sheet through MSL S-param lane |
| `test_msl_sparse_dft.py` | sparse DFT-region contracts for MSL extraction |
| `test_msl_wave_decomp_jvp.py` | custom_jvp on _solve_q wave decomposition |
| `test_msl_z0_bias_floor_sweep_realized_anchor.py` | msl_z0_bias_floor_sweep realized-board anchor |
| `test_normalization.py` | two-run waveguide S normalization |
| `test_normalize_flux.py` | normalize='flux' S extraction |
| `test_overlap_extraction.py` | overlap-integral modal extraction |
| `test_passivity_guard_wiring.py` | passivity guard wiring (#337/#342) |
| `test_periodic_lumped_sparam_guard.py` | lumped/wire S extractor rejects periodic axes |
| `test_port_dump_replay.py` | V/I dump replay of port S-matrices |
| `test_port_observable_validation.py` | validate_port_smatrix helpers |
| `test_probe_fed_msl_referee_contract.py` | #498 openEMS referee lane stage contract |
| `test_rf_audit_fixes.py` | RF-core extractor audit fixes |
| `test_run_forward_s11_contract.py` | run() vs forward() S11 agreement |
| `test_settling_witness_enforcement.py` | settling witness enforces -40 dB |
| `test_sparam.py` | lumped-port S extraction limiting cases |
| `test_sparam_driver_dump_parity.py` | scan driver V/I dump == eager dump |
| `test_sparam_driver_matches_eager.py` | scan S-matrix driver matches eager extractor |
| `test_sparam_passive_port_drive.py` | scan driver vs passive ports |
| `test_sparam_passivity_guard.py` | waveguide/coax S-matrix passivity self-flag (pr-tests guard) |
| `test_sparameter_support_contract.py` | port-family S-parameter support contract |
| `test_twoport_wire_port.py` | multi-port wire-port S-matrix |
| `test_waveguide_nu_flux.py` | NU normalize='flux' branch |
| `test_waveguide_nu_nontrivial.py` | NU WR-90 non-trivial S-matrix |
| `test_waveguide_nu_sparam.py` | NU _compute_waveguide_s_matrix_nu assembly |
| `test_waveguide_phase_gate.py` | convention-independent waveguide phase witness |
| `test_waveguide_port_reference_sims.py` | port_reference_sims plumbing |
| `test_waveguide_settling_witness.py` | waveguide S-matrix settling witness |
| `test_waveguide_twoport_contract_v1.py` | probe-aware normalized two-port contract |
| `test_wire_port_sparams_forward.py` | forward-path WirePort S-parameter wiring |
| `test_wire_sparam.py` | WirePort S-parameter extraction |

### `tests/unit/farfield/` (12)

| file | reason |
|---|---|
| `test_antenna.py` | antenna metric extraction |
| `test_farfield_asymmetric_cpml.py` | NTFFBox.from_grid per-face CPML |
| `test_farfield_chunking.py` | compute_far_field_jax direction chunking |
| `test_farfield_inplane_nonuniform.py` | far-field on in-plane graded mesh |
| `test_farfield_nonuniform.py` | NU NTFF capability |
| `test_ntff_box_nu_pads.py` | NU-lane NTFF box CPML depths |
| `test_ntff_lateral_clearance.py` | NTFF lateral PEC-sheet clearance preflight |
| `test_ntff_small_gp_advisory.py` | sub-wavelength ground plane NTFF advisory |
| `test_ntff_smatrix_drop_warning.py` | NTFF box on S-matrix path not dropped silently |
| `test_ntff_sweep.py` | streaming NTFF multi-frequency sweep |
| `test_oblique_rcs_absolute_sigma.py` | absolute oblique sigma normalization + boundary guards |
| `test_rcs280_reference_subtraction.py` | two-run incident-reference subtraction for RCS |

### `tests/unit/nonuniform/` (25)

| file | reason |
|---|---|
| `test_auto_dz_profile_preserve.py` | _make_dz_profile realizes declared mesh |
| `test_dz_only_dispatch_contract.py` | dz-only dispatch across S-matrix entry points |
| `test_inplane_grading_guards.py` | in-plane grading guards |
| `test_mesh_intelligence_report.py` | MeshIntelligenceReport |
| `test_mesh_planner.py` | plan_mesh / plan_simulation_mesh |
| `test_multiband_nu_envelope.py` | multi-band graded-mesh envelope |
| `test_nonuniform_api.py` | NU mesh with Simulation API |
| `test_nonuniform_checkpoint.py` | jax.checkpoint plumbing on NU scan |
| `test_nonuniform_cpml_dielectric.py` | NU CPML material-aware |
| `test_nonuniform_emit_ts.py` | emit_time_series=False on NU forward |
| `test_nonuniform_grid_extent_contract.py` | NU grid realizes requested extent |
| `test_nonuniform_pec_scatterer_limit.py` | volumetric PEC scatterers on NU waveguide path |
| `test_nonuniform_segmented.py` | segmented scan on NU path |
| `test_nonuniform_source_port_dual_spacing.py` | source/wire-port dual spacing on NU |
| `test_nonuniform_uniform_end_to_end_reduction.py` | NU solve reduces to uniform solve |
| `test_nonuniform_until_decay.py` | until_decay on NU lane |
| `test_nonuniform_upml_guard.py` | NU lane refuses boundary='upml' |
| `test_nonuniform_xy.py` | in-plane NU mesh support |
| `test_nu_port_sigma_dual_spacing.py` | NU port termination sigma dual widths |
| `test_nu_progress_chunking.py` | NU chunked progress re-entry identity |
| `test_nu_wire_port_index_zero_stencil.py` | NU wire-port Ampere loop at index 0 |
| `test_nu_wire_port_lane_parity.py` | NU vs uniform wire-port lane parity |
| `test_patch_uniform_fine_substrate.py` | smooth_grading uniform-fine band lock |
| `test_smooth_grading_preserve.py` | smooth_grading(preserve_regions=) |
| `test_state_purity_dz_profile.py` | dz-profile state purity |

### `tests/unit/subgrid/` (22)

| file | reason |
|---|---|
| `test_amr_surrogate.py` | rfx.amr refinement indicator + surrogate export |
| `test_disjoint_subgrid_3d.py` | disjoint-domain 3D subgrid |
| `test_material_weighted_sat.py` | material-weighted SAT algebra |
| `test_sbp_sat_1d.py` | 1D SBP-SAT |
| `test_sbp_sat_2d.py` | 2D SBP-SAT |
| `test_sbp_sat_3d.py` | 3D SBP-SAT |
| `test_sbp_sat_alpha.py` | SBP-SAT penalty coefficient |
| `test_sbp_sat_jit.py` | SBP-SAT JIT runner |
| `test_subgrid_cpml_dielectric.py` | subgrid CPML material-aware |
| `test_subgrid_crossval.py` | subgrid vs uniform fine reference |
| `test_subgrid_disjoint_runner_contract.py` | disjoint runner contract |
| `test_subgrid_fine_shape_parity.py` | fine-grid shape conventions |
| `test_subgrid_jit_coarse_probes.py` | jit runner coarse probes |
| `test_subgrid_jit_modal_trace_filter.py` | jit runner modal trace filter |
| `test_subgrid_jit_step_captures.py` | step_fn capture contract |
| `test_subgrid_material_zhi_eps_blend.py` | z-hi material eps blend |
| `test_subgrid_n_minus_1_guard.py` | subgrid topology guard |
| `test_subgrid_pml_overlap_warning.py` | subgrid PML-overlap warning frame |
| `test_subgrid_port_research.py` | subgrid impedance ports |
| `test_subgrid_public_api_regression.py` | validate_subgrid public API |
| `test_subgrid_source_injection_dtype.py` | subgrid source injection dtype |
| `test_subgrid_validation.py` | production-envelope subgrid validation |

### `tests/unit/runners/` (30)

| file | reason |
|---|---|
| `test_adi.py` | ADI-FDTD 2D/3D solver |
| `test_adi_gradient.py` | ADI 3D scheme AD regression |
| `test_batch.py` | batch simulation / parameter sweep |
| `test_batch_provenance.py` | manifest-backed batch provenance |
| `test_convergence.py` | rfx.convergence study tooling |
| `test_decay_convergence.py` | run_until_decay stopping criterion |
| `test_decay_flux_convergence.py` | run_until_decay stop quality harness |
| `test_decay_rlc.py` | run_until_decay with lumped RLC |
| `test_device_count_sentinel.py` | 2-device sentinel |
| `test_distributed.py` | multi-device distributed runner |
| `test_distributed_cpml_dielectric.py` | distributed CPML material-aware |
| `test_distributed_nu_composition.py` | distributed + NU composition |
| `test_distributed_nu_cpml_dielectric.py` | distributed NU CPML material-aware |
| `test_distributed_nu_kernel.py` | distributed_nu kernels |
| `test_distributed_nu_pec_mask_lane_parity.py` | distributed NU PEC mask parity |
| `test_distributed_nu_smoke.py` | distributed NU smoke |
| `test_distributed_pmap_cpml_dielectric.py` | legacy pmap CPML material-aware |
| `test_distributed_v2_gather_traceable.py` | distributed_v2 gather traceability |
| `test_flux_stop_criterion.py` | radiated-flux stop criterion |
| `test_gpu.py` | rfx.gpu utilities |
| `test_profiling.py` | profile_forward |
| `test_run_progress_reporting.py` | progress reporting on long solves |
| `test_runner_import_binding.py` | runners/uniform import-time binding |
| `test_silent_drop_warnings.py` | dispatch paths must not drop run kwargs |
| `test_simulation.py` | compiled simulation runner + multiport S |
| `test_sweep.py` | parametric sweep API |
| `test_vmap_cpml_dielectric.py` | vmap_sweep CPML material-aware |
| `test_vmap_sweep.py` | vmap batched material sweep |
| `test_vmap_sweep_dft_planes.py` | DFT planes on vmap fast path |
| `test_vmap_sweep_eligibility.py` | vmap fast-path eligibility guards |

### `tests/unit/autodiff/` (57)

| file | reason |
|---|---|
| `test_ad_diagnostics.py` | rfx.ad_diagnostics |
| `test_ad_memory_grid_and_sheet.py` | AD memory estimate vs solve grid |
| `test_ad_surface_contract.py` | AD classification of every S-parameter entry point |
| `test_benchmark_jacobian_fwd.py` | scripts/benchmark_jacobian_fwd relations gate |
| `test_coax_end_to_end_ad.py` | coax reflection end-to-end AD gate |
| `test_coax_two_port_ad.py` | compute_coaxial_two_port AD gate |
| `test_design_mask_removed.py` | design_mask removed from AD surface |
| `test_differentiable.py` | differentiable FDTD (Stage 3) |
| `test_differentiable_material_fit.py` | jax.grad through material fitting |
| `test_differentiable_material_fit_normalization.py` | reference_probe loss mode |
| `test_directivity_gradient.py` | maximize_directivity gradient |
| `test_estimate_ad_memory.py` | estimate_ad_memory model |
| `test_forward_dft_planes_carry.py` | ForwardResult carries DFT plane accumulators |
| `test_forward_outer_jit_traceable.py` | forward()/optimize() under outer jit |
| `test_forward_tfsf_differentiable.py` | differentiable TFSF forward |
| `test_forward_tfsf_gradient_doctrine.py` | TFSF forward gradient doctrine |
| `test_forward_tfsf_inverse_design_smoke.py` | TFSF inverse-design smoke |
| `test_gradient_coverage.py` | gradient coverage of physics paths |
| `test_gradient_dx_ladder_gates.py` | AD-vs-FD mesh ladder evidence gate |
| `test_gradient_simple.py` | simple gradient sign/nonzero |
| `test_jacobian_fwd.py` | observables.jacobian_fwd |
| `test_lumped_rlc_ad.py` | lumped component value AD |
| `test_memory_reduction_planning_artifact.py` | memory reduction planning artifact script |
| `test_minimize_s11_at_freq_physical.py` | minimize_s11_at_freq objective physics |
| `test_msl_ad_fd_converged.py` | MSL converged AD-vs-FD |
| `test_msl_multistart.py` | multi-start Adam helper (MSL example) |
| `test_msl_sparam_ad.py` | JAX-native MSL S-matrix assembly AD |
| `test_n_warmup.py` | n_warmup gradient-free warmup phase |
| `test_nonuniform_forward_grad.py` | forward() gradient on NU path |
| `test_nonuniform_grad_sparams.py` | NU s_params tracer-safe under grad |
| `test_nonuniform_gradient.py` | run_nonuniform gradient regression |
| `test_objective_library.py` | objective function library |
| `test_observables_dft_field.py` | rfx.observables AD-vs-FD legs |
| `test_optimize.py` | inverse design optimizer |
| `test_optimize_convergence.py` | optimization convergence end-to-end |
| `test_optimize_multistart.py` | optimize multi-start knobs |
| `test_optimize_nonuniform.py` | optimize() on NU meshes |
| `test_optimize_proxy_objectives.py` | time-domain proxy objectives |
| `test_optimize_s11_wave_decomp.py` | optimize plumbs port_s11_freqs |
| `test_optimizer_bakeoff_gates.py` | optimizer bake-off evidence gate |
| `test_pareto.py` | rfx.pareto multi-objective front |
| `test_per_cell_snr_gates.py` | per-cell gradient SNR ladder gate |
| `test_progressive_optimize.py` | progressive_optimize orchestrator |
| `test_rcs_jax_differentiable.py` | compute_rcs_jax gradient |
| `test_rcs_reduction_inverse_design.py` | RCS-reduction inverse design |
| `test_s11_at_freq.py` | minimize_s11_at_freq objective |
| `test_scan_segmented_checkpoint.py` | segmented-checkpoint scan on forward() |
| `test_sparam_ad_end_to_end.py` | end-to-end AD through S-parameter extractors |
| `test_stage1_tier2_tracer_fixes.py` | differentiable-path tracer-break fixes |
| `test_topology.py` | density-based topology optimization |
| `test_verification.py` | AD through TFSF / DFT plane + oblique Fresnel |
| `test_waveguide_flux_ad.py` | normalize='flux' waveguide S on AD tape |
| `test_waveguide_forward.py` | waveguide port + forward() regression |
| `test_waveguide_nu_checkpoint.py` | sqrt(N) checkpointing on NU waveguide flux S |
| `test_waveguide_nu_flux_ad.py` | NU normalize='flux' S on AD tape |
| `test_waveguide_smatrix_checkpoint.py` | compute_waveguide_s_matrix(checkpoint_segments=) |
| `test_waveguide_sparam_ad.py` | waveguide S-matrix assembly AD-traceable |

### `tests/unit/preflight/` (18)

| file | reason |
|---|---|
| `test_adi_preflight.py` | ADI 3D accuracy advisory |
| `test_auto_guard.py` | empty-geometry guard |
| `test_auto_preflight.py` | forward()/optimize() auto-run preflight (pr-tests guard) |
| `test_inverse_design_preflight.py` | inverse-design preflight checks (pr-tests guard) |
| `test_postrun_energy_witness.py` | post-run energy advisories |
| `test_preflight_absorber_frame.py` | absorber-overlap validator frame |
| `test_preflight_advisory_emission_contract.py` | advisories have an emission site |
| `test_preflight_campaign_statics.py` | campaign statics advisory checks |
| `test_preflight_dispersive_pole_at_absorber.py` | dispersive pole at CPML advisory |
| `test_preflight_false_positives.py` | preflight false-positive refinements (pr-tests guard) |
| `test_preflight_geometry_absorber_aggregation.py` | geometry-in-CPML reporting contract |
| `test_preflight_graded_rasterization.py` | Box displaced from fine band advisory |
| `test_preflight_physics_thresholds.py` | physics-based thresholds (pr-tests guard) |
| `test_preflight_structured_and_guards.py` | structured preflight records (pr-tests guard) |
| `test_preflight_tfsf_lumped.py` | TFSF + lumped RLC guard |
| `test_preflight_thin_metal_nu.py` | thin PEC on NU axis warning |
| `test_run_preflight_parity.py` | run() preflight parity with forward() (pr-tests guard) |
| `test_stage2_tier3_guards.py` | silent-wrong-answer guards |

### `tests/unit/api/` (16)

| file | reason |
|---|---|
| `test_animation.py` | field animation export |
| `test_api.py` | high-level Simulation API (pr-tests guard) |
| `test_artifacts.py` | runtime artifact/report/bundle export |
| `test_config_loader.py` | config-driven loader / CLI |
| `test_dashboard.py` | rfx.dashboard |
| `test_dashboard_cli.py` | rfx-dashboard entry point |
| `test_diagnostics.py` | rfx-diagnose + hello-world spine |
| `test_io.py` | Touchstone I/O |
| `test_rasterized_slice_viewer.py` | visualize.plot_rasterized_slice |
| `test_result_accessors.py` | Result accessors + plotting |
| `test_result_finite_guard.py` | Result.assert_finite (pr-tests guard) |
| `test_smith.py` | Smith chart plotting |
| `test_touchstone_interop.py` | Touchstone metadata interop |
| `test_visualize.py` | field visualization |
| `test_visualize3d.py` | rfx.visualize3d |
| `test_visualize_3d.py` | 3D structure + far-field visualisation API |

### `tests/unit/misc/` (11)

| file | reason |
|---|---|
| `test_dft_probes.py` | rfx.probes DFT point/plane probes (no probes dir in the approved list) |
| `test_flux_monitor_finite_size.py` | rfx.probes FluxMonitor finite-size (no probes dir) |
| `test_flux_silent_zero_guard.py` | rfx.probes flux_spectrum underflow guard (no probes dir) |
| `test_harminv_decimation.py` | rfx.harminv auto-decimation |
| `test_harminv_estimator.py` | rfx.harminv Matrix Pencil estimator |
| `test_jax_checks.py` | rfx.jax_checks checkify invariants |
| `test_ladder_guard.py` | validation/research convergence_floor ladder_guard self-checks |
| `test_review_tier1_validation_battery.py` | 2026-05-16 code-review reproduction battery (mixed modules) |
| `test_ris.py` | rfx.ris (deprecated, skipped) |
| `test_v173a_physics_equivalence_slow.py` | V173-A bit-identity release gate over scripts |
| `test_weekly_rss_reporter.py` | root conftest RSS reporter |

