# Committed artifact field ledger (2026-09-08)

Comparison: `main` **b59e1d991dd62868bdf8689a1f642eeb8f7c5b89** → branch **b5b84e3991226b318e7001e3319cb4bd02cc300f** (before docs edits).

This is a read-only JSON comparison, with no simulation or artifact regeneration. Every changed scalar/object leaf is listed. Numeric arrays are one JSON field: lengths, canonical-JSON SHA256 prefixes and every changed zero-based index are listed instead of copying the raw traces. Arrays of records are expanded by index and field. Full old/new array values remain at the two git revisions above. `absent` means an addition/removal, not a numerical move. No tolerance or rounding is applied to the comparison.

Scope: **8** literal `_..._results` JSON directories, **10** fixture directories, and the additional `_issue812_phase_identity` result directory (**9** result groups total). `_11_wr90_port_results`, `_16_ka_sweep_results`, `_17_dielectric_results`, `_20_msl_phase_referee_logs` and `_23_lossy_results` have changed recompute prose but no changed JSON under the requested glob.

| Artifact | Status | Changed fields |
|---|---|---:|
| `tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json` | modified | 6 |
| `tests/fixtures/experiments/patch_antenna_v2.json` | modified | 2 |
| `tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json` | modified | 15 |
| `tests/fixtures/patch_canonical_farfield_e4/canonical_farfield_e4_measured_369367259302.json` | added | 56 |
| `tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json` | modified | 188 |
| `tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json` | modified | 5 |
| `tests/fixtures/waveguide_chain_battery/fixture_v18_close.json` | added | 3520 |
| `tests/fixtures/waveguide_vi_envelope/s21_phase_residual_witness.json` | added | 55 |
| `tests/fixtures/wr90_iris_filter/fixture.json` | modified | 260 |
| `tests/fixtures/wr90_iris_modematch/fixture.json` | modified | 492 |
| `validation/crossval/_05_patch_results/control_witness_2b/control_baseline_d990e18c.json` | added | 62 |
| `validation/crossval/_05_patch_results/control_witness_2b/control_merged_b4961b56.json` | added | 62 |
| `validation/crossval/_05_patch_results/cv05_run_openems_369367259142.json` | added | 81 |
| `validation/crossval/_06b_msl_notch_results/cv06b_build_falsifiers_summary.json` | modified | 24 |
| `validation/crossval/_06b_msl_notch_results/cv06b_falsifier_baseline.json` | modified | 20 |
| `validation/crossval/_06b_msl_notch_results/cv06b_falsifier_stub_1cell.json` | modified | 18 |
| `validation/crossval/_06b_msl_notch_results/cv06b_falsifier_stub_narrow.json` | modified | 19 |
| `validation/crossval/_07_sheen_results/rfx.json` | modified | 11 |
| `validation/crossval/_15_patch_results/rfx.json` | modified | 19 |
| `validation/crossval/_15_patch_results/rfx_decomposition_feed_pre931.json` | added | 31 |
| `validation/crossval/_15_patch_results/rfx_pre931_two_plane_ground_1f005d0d.json` | added | 26 |
| `validation/crossval/_18_wr90_iris_results/aperture_resolution.json` | modified | 129 |
| `validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json` | modified | 27 |
| `validation/crossval/_18_wr90_iris_results/rfx.json` | modified | 492 |
| `validation/crossval/_19_iris_filter_results/rfx.json` | modified | 260 |
| `validation/crossval/_22_dispersive_results/rfx.json` | modified | 19 |
| `validation/crossval/_24_nu_cavity_results/rfx.json` | modified | 533 |
| `validation/crossval/_issue812_phase_identity/regate_evidence.json` | modified | 23 |

## tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/case_C_shallow_notch_from_geometry/independence/model_notch_depth_db` | `-32.38987697287823` | `-32.389876972878234` | — |
| `/case_C_shallow_notch_from_geometry/rows/0/notch_depth_db` | `-32.38987697287823` | `-32.389876972878234` | — |
| `/case_C_shallow_notch_from_geometry/rows/1/notch_depth_db` | `-33.16358417945059` | `-33.163584179450595` | — |
| `/case_C_shallow_notch_from_geometry/rows/2/bw_frac` | `0.1652072784694682` | `0.1652072784694683` | — |
| `/case_C_shallow_notch_from_geometry/rows/2/bw_ratio` | `0.7856762056624604` | `0.785676205662461` | — |
| `/case_C_shallow_notch_from_geometry/rows/3/notch_depth_db` | `-28.971523961467383` | `-28.971523961467387` | — |

## tests/fixtures/experiments/patch_antenna_v2.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/geometry/0/bounds_m` | `[[0.011, 0.01, 0.01], [0.029, 0.026, 0.012]]` | `[[0.011, 0.01, 0.012], [0.029, 0.026, 0.012]]` | 0 |
| `/geometry/2/bounds_m` | `[[0.015, 0.014, 0.014], [0.025, 0.022, 0.016]]` | `[[0.015, 0.014, 0.014], [0.025, 0.022, 0.014]]` | 1 |

## tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/beta_first_port` | array[30], SHA256 `382fd74f6920` | array[30], SHA256 `995340b25d9c` | 0–29 |
| `/meta/elapsed_s` | `1543.7` | `584.4` | — |
| `/meta/t_metal_realized_m` | *absent* | `5e-05` | — |
| `/meta/trace_realization_kind` | *absent* | `"volume"` | — |
| `/meta/trace_wall_planes_realized` | *absent* | `[5, 6]` | 0–1 |
| `/meta/trace_wall_planes_realized_z_m` | *absent* | `[0.00025, 0.00030000000000000003]` | 0–1 |
| `/meta/trace_y_hi_realized_m` | `0.0015499999999999984` | `0.0015` | — |
| `/meta/trace_y_lo_realized_m` | `0.0009499999999999995` | `0.0009000000000000001` | — |
| `/meta/w_trace_realized_m` | `0.0005999999999999989` | `0.0006` | — |
| `/reference_plane_geometry/msl_0/n_probe_spacing` | `20` | `11` | — |
| `/reference_plane_geometry/msl_1/n_probe_spacing` | `20` | `11` | — |
| `/s11` | array[30], SHA256 `1b8ed3a1a0f8` | array[30], SHA256 `102a2c3766f4` | 0–29 |
| `/s21` | array[30], SHA256 `97e512a2439c` | array[30], SHA256 `0a8fdb46922e` | 0–29 |
| `/z0_port0` | array[30], SHA256 `18b7d289888f` | array[30], SHA256 `92e5d24cf82e` | 0–29 |
| `/z0_port1` | array[30], SHA256 `1aad2e71502b` | array[30], SHA256 `898c9ec3d852` | 0–29 |

## tests/fixtures/patch_canonical_farfield_e4/canonical_farfield_e4_measured_369367259302.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/_what` | *absent* | `"Re-derivation of tests/crossval/test_patch_canonical_farfield_e4.py's gated envelope constants on the #931 sheet-declared canonical patch. Measured by calling that file's own rfx_run fixture function, so the build and the reduction are the gate's, not a copy."` | — |
| `/measured/beam_peak_theta_deg` | *absent* | array[7], SHA256 `d460a174f062` | 0–6 |
| `/measured/cuts_deg/E_peak_deg` | *absent* | `-1.0` | — |
| `/measured/cuts_deg/H_peak_deg` | *absent* | `-4.0` | — |
| `/measured/d_abs_diff_db` | *absent* | `0.06590990066528324` | — |
| `/measured/d_dbi_per_bin` | *absent* | array[7], SHA256 `766a62eb19a0` | 0–6 |
| `/measured/directivity_dbi` | *absent* | `6.724090099334717` | — |
| `/measured/f_radiating_hz` | *absent* | `2507021731.087216` | — |
| `/measured/f_rel_vs_reference` | *absent* | `0.03506119940845378` | — |
| `/measured/k_star` | *absent* | `5` | — |
| `/measured/mode_pair_ghz` | *absent* | `[2.03457486756225, 2.507021731087216]` | 0–1 |
| `/measured/mode_pair_ratio` | *absent* | `1.2322091317735746` | — |
| `/measured/modes/0/Q` | *absent* | `63.333344091855004` | — |
| `/measured/modes/0/amplitude` | *absent* | `41133.112439126955` | — |
| `/measured/modes/0/freq_hz` | *absent* | `2034574867.5622501` | — |
| `/measured/modes/1/Q` | *absent* | `43.691042952041265` | — |
| `/measured/modes/1/amplitude` | *absent* | `36909.13163062917` | — |
| `/measured/modes/1/freq_hz` | *absent* | `2507021731.087216` | — |
| `/measured/modes/2/Q` | *absent* | `74.4658957609839` | — |
| `/measured/modes/2/amplitude` | *absent* | `8078.826508119678` | — |
| `/measured/modes/2/freq_hz` | *absent* | `3316827225.299326` | — |
| `/measured/p_rel_db` | *absent* | array[7], SHA256 `f71de50febc0` | 0–6 |
| `/measured/preflight_advisories` | *absent* | `["2 PEC sheet(s) realized (lattice ownership contract #931 §1.3: one node plane each, closed footprint, normal E through the plane live), 0 of them off their declared mid-plane; worst first: thin_conductor[0] 'thin_conductor[0]' normal z: declared mid-plane 35.29mm, realized node plane 29 at 35.29mm (offset +0.000 cell = 1.388e-08nm); thin_conductor[1] 'thin_conductor[1]' normal z: declared mid-plane 36.81mm, realized node plane 33 at 36.81mm (offset +0.000 cell = 1.388e-08nm). A sheet snaps to the node plane nearest its declared mid-plane (an exact half-cell tie resolves LOWER); an offset means the declared plane — a laminate face, a ground level — is not on this mesh's node line, and the conductor sits that far from where it was drawn. REMEDY when the offset matters: put a mesh node on the declared plane (dx = h/N for an interface at height h, or a preserved region on the non-uniform lane). COVERAGE: every sheet declaration on the nonuniform lane (zero-thickness PEC Boxes via add() and PEC add_thin_conductor entries). STALE IF: the named sheet's SheetSpec.plane is not the printed node.", "4 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 4 listed): thin_conductor[1] 'thin_conductor[1]' (sheet) x: extent 32mm, worst face residual 1mm (3.13% of the extent, df/f ~ 3.13%); thin_conductor[1] 'thin_conductor[1]' (sheet) y: extent 40mm, worst face residual 1mm (2.50% of the extent, df/f ~ 2.50%); thin_conductor[0] 'thin_conductor[0]' (sheet) x: extent 60mm, worst face residual 1mm (1.67% of the extent, df/f ~ 1.67%); thin_conductor[0] 'thin_conductor[0]' (sheet) y: extent 60mm, worst face residual 1mm (1.67% of the extent, df/f ~ 1.67%). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3), so the realized extent can differ from the design by up to the printed residual, and a resonant dimension realized dL off detunes df/f ~ dL/L. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: choose dx commensurate with the patterned dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on the design edges. COVERAGE: examined 4 axis extent(s) on 2 conductor Box declaration(s) on the nonuniform lane; 2 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: &#124;face - nearest node&#124; on the run's node coordinates does not reproduce the printed residuals.", "NTFF face z_lo is 6.00mm from geometry 'sub' — below λ/4 = 26.77mm at f_max=2.80GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).", "Far-field pattern advisory: the PEC sheet backing a source (bbox (85.0, 85.0, 35.3)–(145.0, 145.0, 35.3) mm) spans 60.0mm × 60.0mm = 0.56λ × 0.56λ at f_max=2.80GHz — a ground plane under ~1λ across. Expect the radiation pattern to be shaped by ground-plane edge diffraction (broadside dip, off-axis side peaks). This is expected physics, not a solver defect, and the fixture stays fine for resonance / impedance work. For a clean broadside pattern enlarge the ground plane to at least ~1.4λ; if the small ground plane is intentional, interpret the pattern accordingly."]` | 0–3 |
| `/measured/q_radiating` | *absent* | `43.691042952041265` | — |
| `/measured/settling_bar_db` | *absent* | `-40.0` | — |
| `/measured/settling_clears_bar` | *absent* | `true` | — |
| `/measured/settling_end_db` | *absent* | `-42.90464101187098` | — |
| `/measured/wall_s` | *absent* | `466.9681308269501` | — |
| `/proposed_constants/D_ABS_TOL_DB/arithmetic` | *absent* | `"&#124;6.7241 - 6.7900&#124; = 0.0659 dB; ceil(0.0659 x 1.5, .1) = 0.1; max(0.1, 1.0) = 1.0"` | — |
| `/proposed_constants/D_ABS_TOL_DB/new` | *absent* | `1.0` | — |
| `/proposed_constants/D_ABS_TOL_DB/old` | *absent* | `1.0` | — |
| `/proposed_constants/D_ABS_TOL_DB/rule` | *absent* | `"round-UP(measured x 1.5, 1 decimal), floored at the pre-#931 pin so a rerun can never narrow the gate"` | — |
| `/proposed_constants/F_RES_REL_HI/arithmetic` | *absent* | `"measured +3.51%; ceil(+3.51 + 5) = +9%"` | — |
| `/proposed_constants/F_RES_REL_HI/new` | *absent* | `0.09` | — |
| `/proposed_constants/F_RES_REL_HI/old` | *absent* | `0.16` | — |
| `/proposed_constants/F_RES_REL_HI/rule` | *absent* | `"measured + 5 percentage points, rounded outward to whole percent"` | — |
| `/proposed_constants/F_RES_REL_LO/arithmetic` | *absent* | `"measured +3.51%; floor(+3.51 - 5) = -2%"` | — |
| `/proposed_constants/F_RES_REL_LO/new` | *absent* | `-0.02` | — |
| `/proposed_constants/F_RES_REL_LO/old` | *absent* | `0.06` | — |
| `/proposed_constants/F_RES_REL_LO/rule` | *absent* | `"measured - 5 percentage points, rounded outward to whole percent"` | — |
| `/proposed_constants/mode_pair_ratio_band/arithmetic` | *absent* | `"measured 1.2322; [1.16, 1.31]"` | — |
| `/proposed_constants/mode_pair_ratio_band/new` | *absent* | `[1.16, 1.31]` | 0–1 |
| `/proposed_constants/mode_pair_ratio_band/old` | *absent* | `[1.15, 1.3]` | 0–1 |
| `/proposed_constants/mode_pair_ratio_band/rule` | *absent* | `"measured +- 0.07, rounded outward to two decimals"` | — |
| `/provenance/frame/dx_m` | *absent* | `0.002` | — |
| `/provenance/frame/n_sub` | *absent* | `4` | — |
| `/provenance/frame/ntff_freqs_hz` | *absent* | array[7], SHA256 `15e96dc5bd13` | 0–6 |
| `/provenance/frame/num_periods` | *absent* | `110` | — |
| `/provenance/gate_file` | *absent* | `"tests/crossval/test_patch_canonical_farfield_e4.py"` | — |
| `/provenance/producer` | *absent* | `"scripts/diagnostics/measure_patch_canonical_farfield_e4.py"` | — |
| `/provenance/recorded_utc` | *absent* | `"2026-09-07T21:02:29Z"` | — |
| `/provenance/repo_commit` | *absent* | `"09aab9d98da883d2c0cf3321c6d9430cda0be950"` | — |
| `/provenance/repo_dirty` | *absent* | `false` | — |
| `/reference/directivity_dbi` | *absent* | `6.79` | — |
| `/reference/f_res_hz` | *absent* | `2422100000.0` | — |
| `/reference/source` | *absent* | `"tests/fixtures/patch_canonical_farfield_e4/patch_farfield_openems.json"` | — |

## tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/_provenance/host` | `"VESSL 369367257743, remilab-c0 CPU, image ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8, jax[cpu]==0.6.2 (CI's jax); the rfx leg of the same job's shipped-script run reproduces runs.baseline.modes[0].freq exactly"` | *absent* | — |
| `/_provenance/recorded_utc` | `"2026-09-02T05:52:09Z"` | `"2026-09-07T10:01:07Z"` | — |
| `/_provenance/repo_commit` | `"f247575 (agent/regate-patch-selector; the staged tree carries no .git, so the harness could not read it)"` | `"a66f4a27ab3e5df839a26002cc30e3f5fe157e1d"` | — |
| `/_provenance/repo_dirty` | `false` | `true` | — |
| `/_provenance/supersedes` | `"the round-1 fixture built on macOS (jax 0.8.x) at commit 5b6db32: same mode count at four of five lengths, TM100 frequencies within 2e-4 relative, Q within 2 %, and one extra low-amplitude pole at 38.0 mm on this host -- recorded in the design note 6.11"` | *absent* | — |
| `/_realized_x_cell_census/20.5` | `21` | `20` | — |
| `/_realized_x_cell_census/21.0` | `21` | `20` | — |
| `/_realized_x_cell_census/21.5` | `21` | `20` | — |
| `/_realized_x_cell_census/21.65` | `21` | `20` | — |
| `/_realized_x_cell_census/22.5` | `23` | `22` | — |
| `/_realized_x_cell_census/23.0` | `23` | `22` | — |
| `/_realized_x_cell_census/29.5` | `29` | `28` | — |
| `/_realized_x_cell_census/_unit` | *absent* | `"edges (= mm at dx = 1 mm)"` | — |
| `/_realized_x_cell_census/_what` | `"patch x-extent rasterization measured from the geometry mask on cv05's own grid (dx = 1 mm), no FDTD: declared length -> realized x cells. Establishes which design-mode errors are reachable."` | `"patch x-extent measured from the REALIZED PEC edge set on cv05's own grid (dx = 1 mm), no FDTD: declared length (mm) -> realized conductor length in edges. Since #931 the patch is a sheet and this counts tangential E edges, not masked nodes; the pre-#931 census counted nodes and every row read one higher. Establishes which design-mode errors are reachable."` | — |
| `/runs/baseline/modes/0/Q` | `41.966108772711124` | `90.49033429630059` | — |
| `/runs/baseline/modes/0/amplitude` | `315963.09986257134` | `62949.052569474894` | — |
| `/runs/baseline/modes/0/freq` | `2331854551.1670175` | `1900346670.4241853` | — |
| `/runs/baseline/modes/1/Q` | `72.46593821862187` | `50.00444385698477` | — |
| `/runs/baseline/modes/1/amplitude` | `38991.32422854564` | `447161.0223403334` | — |
| `/runs/baseline/modes/1/freq` | `3027300356.287401` | `2446496652.4277554` | — |
| `/runs/baseline/modes/2/Q` | `104.47228436822941` | `67.85645072951348` | — |
| `/runs/baseline/modes/2/amplitude` | `550918.9274581942` | `42274.13506518711` | — |
| `/runs/baseline/modes/2/freq` | `3605107274.604681` | `3186911011.360049` | — |
| `/runs/baseline/modes/3/Q` | *absent* | `109.52499024938646` | — |
| `/runs/baseline/modes/3/amplitude` | *absent* | `602958.9256829545` | — |
| `/runs/baseline/modes/3/freq` | *absent* | `3758955223.270803` | — |
| `/runs/baseline/realized_stack/cavity_eps_r` | *absent* | array[7], SHA256 `c6fd6f01b1f4` | 0–6 |
| `/runs/baseline/realized_stack/cavity_node_to_node_mm` | *absent* | `1.5000000000000013` | — |
| `/runs/baseline/realized_stack/cavity_physical_mm` | *absent* | `1.5` | — |
| `/runs/baseline/realized_stack/ground/declared_x_mm` | *absent* | `60.0` | — |
| `/runs/baseline/realized_stack/ground/declared_y_mm` | *absent* | `55.0` | — |
| `/runs/baseline/realized_stack/ground/declared_z_mm` | *absent* | `11.999999999999993` | — |
| `/runs/baseline/realized_stack/ground/k` | *absent* | `22` | — |
| `/runs/baseline/realized_stack/ground/realized_x_mm` | *absent* | `60.0` | — |
| `/runs/baseline/realized_stack/ground/realized_y_mm` | *absent* | `55.0` | — |
| `/runs/baseline/realized_stack/ground/x_edge_cells` | *absent* | `60` | — |
| `/runs/baseline/realized_stack/ground/x_index_range` | *absent* | `[18, 77]` | 0–1 |
| `/runs/baseline/realized_stack/ground/y_edge_cells` | *absent* | `55` | — |
| `/runs/baseline/realized_stack/ground/y_index_range` | *absent* | `[18, 72]` | 0–1 |
| `/runs/baseline/realized_stack/patch/declared_x_mm` | *absent* | `29.500000000000004` | — |
| `/runs/baseline/realized_stack/patch/declared_y_mm` | *absent* | `38.0` | — |
| `/runs/baseline/realized_stack/patch/declared_z_mm` | *absent* | `13.499999999999995` | — |
| `/runs/baseline/realized_stack/patch/k` | *absent* | `28` | — |
| `/runs/baseline/realized_stack/patch/realized_x_mm` | *absent* | `28.0` | — |
| `/runs/baseline/realized_stack/patch/realized_y_mm` | *absent* | `37.0` | — |
| `/runs/baseline/realized_stack/patch/x_edge_cells` | *absent* | `28` | — |
| `/runs/baseline/realized_stack/patch/x_index_range` | *absent* | `[34, 61]` | 0–1 |
| `/runs/baseline/realized_stack/patch/y_edge_cells` | *absent* | `37` | — |
| `/runs/baseline/realized_stack/patch/y_index_range` | *absent* | `[27, 63]` | 0–1 |
| `/runs/baseline/realized_stack/sheet_plane_delta` | *absent* | `0` | — |
| `/runs/baseline/realized_stack/substrate_cells_between` | *absent* | `6` | — |
| `/runs/baseline/script_exit` | `1` | `0` | — |
| `/runs/patch_len_21p0mm/modes/0/Q` | `32.64658674109165` | `127.68905554554874` | — |
| `/runs/patch_len_21p0mm/modes/0/amplitude` | `251555.04960569437` | `51416.73949736543` | — |
| `/runs/patch_len_21p0mm/modes/0/freq` | `3111893977.1247616` | `1928514417.7936974` | — |
| `/runs/patch_len_21p0mm/modes/1/Q` | `128.9919521920353` | `41.112303099220696` | — |
| `/runs/patch_len_21p0mm/modes/1/amplitude` | `807430.1686991099` | `237060.88947030727` | — |
| `/runs/patch_len_21p0mm/modes/1/freq` | `3637220114.939245` | `3313171253.619663` | — |
| `/runs/patch_len_21p0mm/modes/2/Q` | *absent* | `132.4205372308718` | — |
| `/runs/patch_len_21p0mm/modes/2/amplitude` | *absent* | `840520.3448008626` | — |
| `/runs/patch_len_21p0mm/modes/2/freq` | *absent* | `3805765597.1334476` | — |
| `/runs/patch_len_21p0mm/realized_stack/cavity_eps_r` | *absent* | array[7], SHA256 `c6fd6f01b1f4` | 0–6 |
| `/runs/patch_len_21p0mm/realized_stack/cavity_node_to_node_mm` | *absent* | `1.5000000000000013` | — |
| `/runs/patch_len_21p0mm/realized_stack/cavity_physical_mm` | *absent* | `1.5` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/declared_x_mm` | *absent* | `60.0` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/declared_y_mm` | *absent* | `55.0` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/declared_z_mm` | *absent* | `11.999999999999993` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/k` | *absent* | `22` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/realized_x_mm` | *absent* | `60.0` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/realized_y_mm` | *absent* | `55.0` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/x_edge_cells` | *absent* | `60` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/x_index_range` | *absent* | `[18, 77]` | 0–1 |
| `/runs/patch_len_21p0mm/realized_stack/ground/y_edge_cells` | *absent* | `55` | — |
| `/runs/patch_len_21p0mm/realized_stack/ground/y_index_range` | *absent* | `[18, 72]` | 0–1 |
| `/runs/patch_len_21p0mm/realized_stack/patch/declared_x_mm` | *absent* | `21.0` | — |
| `/runs/patch_len_21p0mm/realized_stack/patch/declared_y_mm` | *absent* | `38.0` | — |
| `/runs/patch_len_21p0mm/realized_stack/patch/declared_z_mm` | *absent* | `13.499999999999995` | — |
| `/runs/patch_len_21p0mm/realized_stack/patch/k` | *absent* | `28` | — |
| `/runs/patch_len_21p0mm/realized_stack/patch/realized_x_mm` | *absent* | `20.0` | — |
| `/runs/patch_len_21p0mm/realized_stack/patch/realized_y_mm` | *absent* | `37.0` | — |
| `/runs/patch_len_21p0mm/realized_stack/patch/x_edge_cells` | *absent* | `20` | — |
| `/runs/patch_len_21p0mm/realized_stack/patch/x_index_range` | *absent* | `[38, 57]` | 0–1 |
| `/runs/patch_len_21p0mm/realized_stack/patch/y_edge_cells` | *absent* | `37` | — |
| `/runs/patch_len_21p0mm/realized_stack/patch/y_index_range` | *absent* | `[27, 63]` | 0–1 |
| `/runs/patch_len_21p0mm/realized_stack/sheet_plane_delta` | *absent* | `0` | — |
| `/runs/patch_len_21p0mm/realized_stack/substrate_cells_between` | *absent* | `6` | — |
| `/runs/patch_len_22p0mm/modes/0/Q` | `34.26642911560174` | `115.8603065384501` | — |
| `/runs/patch_len_22p0mm/modes/0/amplitude` | `93462.47989933677` | `46826.888029416084` | — |
| `/runs/patch_len_22p0mm/modes/0/freq` | `2993459153.4531217` | `1921606083.2691016` | — |
| `/runs/patch_len_22p0mm/modes/1/Q` | `116.84629527235155` | `43.835259020007854` | — |
| `/runs/patch_len_22p0mm/modes/1/amplitude` | `734526.1662209275` | `291209.94365677866` | — |
| `/runs/patch_len_22p0mm/modes/1/freq` | `3633339337.891849` | `3043213212.2984095` | — |
| `/runs/patch_len_22p0mm/modes/2/Q` | *absent* | `122.26492023889415` | — |
| `/runs/patch_len_22p0mm/modes/2/amplitude` | *absent* | `787248.7277494344` | — |
| `/runs/patch_len_22p0mm/modes/2/freq` | *absent* | `3791640239.835446` | — |
| `/runs/patch_len_22p0mm/realized_stack/cavity_eps_r` | *absent* | array[7], SHA256 `c6fd6f01b1f4` | 0–6 |
| `/runs/patch_len_22p0mm/realized_stack/cavity_node_to_node_mm` | *absent* | `1.5000000000000013` | — |
| `/runs/patch_len_22p0mm/realized_stack/cavity_physical_mm` | *absent* | `1.5` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/declared_x_mm` | *absent* | `60.0` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/declared_y_mm` | *absent* | `55.0` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/declared_z_mm` | *absent* | `11.999999999999993` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/k` | *absent* | `22` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/realized_x_mm` | *absent* | `60.0` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/realized_y_mm` | *absent* | `55.0` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/x_edge_cells` | *absent* | `60` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/x_index_range` | *absent* | `[18, 77]` | 0–1 |
| `/runs/patch_len_22p0mm/realized_stack/ground/y_edge_cells` | *absent* | `55` | — |
| `/runs/patch_len_22p0mm/realized_stack/ground/y_index_range` | *absent* | `[18, 72]` | 0–1 |
| `/runs/patch_len_22p0mm/realized_stack/patch/declared_x_mm` | *absent* | `22.0` | — |
| `/runs/patch_len_22p0mm/realized_stack/patch/declared_y_mm` | *absent* | `38.0` | — |
| `/runs/patch_len_22p0mm/realized_stack/patch/declared_z_mm` | *absent* | `13.499999999999995` | — |
| `/runs/patch_len_22p0mm/realized_stack/patch/k` | *absent* | `28` | — |
| `/runs/patch_len_22p0mm/realized_stack/patch/realized_x_mm` | *absent* | `22.0` | — |
| `/runs/patch_len_22p0mm/realized_stack/patch/realized_y_mm` | *absent* | `37.0` | — |
| `/runs/patch_len_22p0mm/realized_stack/patch/x_edge_cells` | *absent* | `22` | — |
| `/runs/patch_len_22p0mm/realized_stack/patch/x_index_range` | *absent* | `[37, 58]` | 0–1 |
| `/runs/patch_len_22p0mm/realized_stack/patch/y_edge_cells` | *absent* | `37` | — |
| `/runs/patch_len_22p0mm/realized_stack/patch/y_index_range` | *absent* | `[27, 63]` | 0–1 |
| `/runs/patch_len_22p0mm/realized_stack/sheet_plane_delta` | *absent* | `0` | — |
| `/runs/patch_len_22p0mm/realized_stack/substrate_cells_between` | *absent* | `6` | — |
| `/runs/patch_len_22p5mm/modes/0/Q` | `36.096441345382374` | `115.8603065384501` | — |
| `/runs/patch_len_22p5mm/modes/0/amplitude` | `232357.2505219953` | `46826.888029416084` | — |
| `/runs/patch_len_22p5mm/modes/0/freq` | `2872715922.661824` | `1921606083.2691016` | — |
| `/runs/patch_len_22p5mm/modes/1/Q` | `21.035098741800503` | `43.835259020007854` | — |
| `/runs/patch_len_22p5mm/modes/1/amplitude` | `859027.2089938433` | `291209.94365677866` | — |
| `/runs/patch_len_22p5mm/modes/1/freq` | `3520324859.0426426` | `3043213212.2984095` | — |
| `/runs/patch_len_22p5mm/modes/2/Q` | `118.23176861613322` | `122.26492023889415` | — |
| `/runs/patch_len_22p5mm/modes/2/amplitude` | `719519.4871881948` | `787248.7277494344` | — |
| `/runs/patch_len_22p5mm/modes/2/freq` | `3627990688.422372` | `3791640239.835446` | — |
| `/runs/patch_len_22p5mm/realized_stack/cavity_eps_r` | *absent* | array[7], SHA256 `c6fd6f01b1f4` | 0–6 |
| `/runs/patch_len_22p5mm/realized_stack/cavity_node_to_node_mm` | *absent* | `1.5000000000000013` | — |
| `/runs/patch_len_22p5mm/realized_stack/cavity_physical_mm` | *absent* | `1.5` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/declared_x_mm` | *absent* | `60.0` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/declared_y_mm` | *absent* | `55.0` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/declared_z_mm` | *absent* | `11.999999999999993` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/k` | *absent* | `22` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/realized_x_mm` | *absent* | `60.0` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/realized_y_mm` | *absent* | `55.0` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/x_edge_cells` | *absent* | `60` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/x_index_range` | *absent* | `[18, 77]` | 0–1 |
| `/runs/patch_len_22p5mm/realized_stack/ground/y_edge_cells` | *absent* | `55` | — |
| `/runs/patch_len_22p5mm/realized_stack/ground/y_index_range` | *absent* | `[18, 72]` | 0–1 |
| `/runs/patch_len_22p5mm/realized_stack/patch/declared_x_mm` | *absent* | `22.5` | — |
| `/runs/patch_len_22p5mm/realized_stack/patch/declared_y_mm` | *absent* | `38.0` | — |
| `/runs/patch_len_22p5mm/realized_stack/patch/declared_z_mm` | *absent* | `13.499999999999995` | — |
| `/runs/patch_len_22p5mm/realized_stack/patch/k` | *absent* | `28` | — |
| `/runs/patch_len_22p5mm/realized_stack/patch/realized_x_mm` | *absent* | `22.0` | — |
| `/runs/patch_len_22p5mm/realized_stack/patch/realized_y_mm` | *absent* | `37.0` | — |
| `/runs/patch_len_22p5mm/realized_stack/patch/x_edge_cells` | *absent* | `22` | — |
| `/runs/patch_len_22p5mm/realized_stack/patch/x_index_range` | *absent* | `[37, 58]` | 0–1 |
| `/runs/patch_len_22p5mm/realized_stack/patch/y_edge_cells` | *absent* | `37` | — |
| `/runs/patch_len_22p5mm/realized_stack/patch/y_index_range` | *absent* | `[27, 63]` | 0–1 |
| `/runs/patch_len_22p5mm/realized_stack/sheet_plane_delta` | *absent* | `0` | — |
| `/runs/patch_len_22p5mm/realized_stack/substrate_cells_between` | *absent* | `6` | — |
| `/runs/patch_len_38p0mm/modes/0/Q` | `49.769244013414685` | `65.70277841528525` | — |
| `/runs/patch_len_38p0mm/modes/0/amplitude` | `280403.0612220936` | `356583.39808769344` | — |
| `/runs/patch_len_38p0mm/modes/0/freq` | `1813339498.9205768` | `1840047744.6319866` | — |
| `/runs/patch_len_38p0mm/modes/1/Q` | `57.67583424221865` | `81.82986269025866` | — |
| `/runs/patch_len_38p0mm/modes/1/amplitude` | `25317.84702256452` | `31034.5743866984` | — |
| `/runs/patch_len_38p0mm/modes/1/freq` | `2609611690.1454163` | `2702631824.789781` | — |
| `/runs/patch_len_38p0mm/modes/2/Q` | `77.46108490925334` | `92.2417698684421` | — |
| `/runs/patch_len_38p0mm/modes/2/amplitude` | `339847.5167262771` | `366874.48300793517` | — |
| `/runs/patch_len_38p0mm/modes/2/freq` | `3576915699.684167` | `3725142223.4133167` | — |
| `/runs/patch_len_38p0mm/realized_stack/cavity_eps_r` | *absent* | array[7], SHA256 `c6fd6f01b1f4` | 0–6 |
| `/runs/patch_len_38p0mm/realized_stack/cavity_node_to_node_mm` | *absent* | `1.5000000000000013` | — |
| `/runs/patch_len_38p0mm/realized_stack/cavity_physical_mm` | *absent* | `1.5` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/declared_x_mm` | *absent* | `60.0` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/declared_y_mm` | *absent* | `55.0` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/declared_z_mm` | *absent* | `11.999999999999993` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/k` | *absent* | `22` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/realized_x_mm` | *absent* | `60.0` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/realized_y_mm` | *absent* | `55.0` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/x_edge_cells` | *absent* | `60` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/x_index_range` | *absent* | `[18, 77]` | 0–1 |
| `/runs/patch_len_38p0mm/realized_stack/ground/y_edge_cells` | *absent* | `55` | — |
| `/runs/patch_len_38p0mm/realized_stack/ground/y_index_range` | *absent* | `[18, 72]` | 0–1 |
| `/runs/patch_len_38p0mm/realized_stack/patch/declared_x_mm` | *absent* | `38.0` | — |
| `/runs/patch_len_38p0mm/realized_stack/patch/declared_y_mm` | *absent* | `38.0` | — |
| `/runs/patch_len_38p0mm/realized_stack/patch/declared_z_mm` | *absent* | `13.499999999999995` | — |
| `/runs/patch_len_38p0mm/realized_stack/patch/k` | *absent* | `28` | — |
| `/runs/patch_len_38p0mm/realized_stack/patch/realized_x_mm` | *absent* | `38.0` | — |
| `/runs/patch_len_38p0mm/realized_stack/patch/realized_y_mm` | *absent* | `37.0` | — |
| `/runs/patch_len_38p0mm/realized_stack/patch/x_edge_cells` | *absent* | `38` | — |
| `/runs/patch_len_38p0mm/realized_stack/patch/x_index_range` | *absent* | `[29, 66]` | 0–1 |
| `/runs/patch_len_38p0mm/realized_stack/patch/y_edge_cells` | *absent* | `37` | — |
| `/runs/patch_len_38p0mm/realized_stack/patch/y_index_range` | *absent* | `[27, 63]` | 0–1 |
| `/runs/patch_len_38p0mm/realized_stack/sheet_plane_delta` | *absent* | `0` | — |
| `/runs/patch_len_38p0mm/realized_stack/substrate_cells_between` | *absent* | `6` | — |

## tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/referee/argmin_first_null/distances_pct/rfx` | `1.543` | `2.3756` | — |
| `/referee/fdtd_doublet_ghz/rfx/argmin_first_null_ghz` | `7.925928` | `8.244069` | — |
| `/referee/fdtd_doublet_ghz/rfx/lower_ghz` | `6.94399` | `7.233338` | — |
| `/referee/fdtd_doublet_ghz/rfx/upper_ghz` | `7.925928` | `8.244069` | — |
| `/referee/structure_distance_pct/rfx` | `1.5195` | `2.8668` | — |

## tests/fixtures/waveguide_chain_battery/fixture_v18_close.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/ad_vs_fd/0/ad_vs_fd_float32/f_minus` | *absent* | `1.0000498383360008` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/f_plus` | *absent* | `1.0001262714391654` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/fd_ulp_span` | *absent* | `344224094931.0` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/g_ad` | *absent* | `0.0007709434721618891` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/g_fd` | *absent* | `0.0007643310316463037` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/rel` | *absent* | `0.008651278362129002` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/0/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/0/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/0/dut` | *absent* | `"pec_short"` | — |
| `/ad_vs_fd/0/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/0/expected_ulp_floor_skip` | *absent* | `true` | — |
| `/ad_vs_fd/0/f_minus` | *absent* | `1.0000498383360008` | — |
| `/ad_vs_fd/0/f_plus` | *absent* | `1.0001262714391654` | — |
| `/ad_vs_fd/0/fd_ulp_span` | *absent* | `344224094931.0` | — |
| `/ad_vs_fd/0/forward_identity/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/0/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/0/forward_identity/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/forward_identity/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/0/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/0/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/0/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/0/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/0/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/0/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/0/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/0/forward_identity_float32/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/0/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/0/forward_identity_float32/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/forward_identity_float32/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/forward_identity_float32/pass` | *absent* | `true` | — |
| `/ad_vs_fd/0/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/0/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/0/g_ad` | *absent* | `0.0007709434721618891` | — |
| `/ad_vs_fd/0/g_fd` | *absent* | `0.0007643310316463037` | — |
| `/ad_vs_fd/0/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/0/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/0/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/0/lane` | *absent* | `"false"` | — |
| `/ad_vs_fd/0/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/0/objective` | *absent* | `"s11_mag2"` | — |
| `/ad_vs_fd/0/primary_precision` | *absent* | `"float32"` | — |
| `/ad_vs_fd/0/rel` | *absent* | `0.008651278362129002` | — |
| `/ad_vs_fd/0/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/0/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/0/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/0/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/0/value_at_theta0` | *absent* | `1.0000901222229004` | — |
| `/ad_vs_fd/0/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/0/wall_time_s/ad` | *absent* | `14.610287427902222` | — |
| `/ad_vs_fd/0/wall_time_s/fd_pair` | *absent* | `13.598352909088135` | — |
| `/ad_vs_fd/0/wall_time_s/x64_witness` | *absent* | `22.179444074630737` | — |
| `/ad_vs_fd/0/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/0/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `0.9998754500267755` | — |
| `/ad_vs_fd/0/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/0/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/0/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/0/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/0/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/0/x64_witness/g_ad_x64` | *absent* | `0.000770589685998857` | — |
| `/ad_vs_fd/0/x64_witness/value_x64` | *absent* | `1.0000900843872071` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/f_minus` | *absent* | `-0.7916232426538564` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/f_plus` | *absent* | `-0.9392309639530954` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/fd_ulp_span` | *absent* | `1329532157280521.0` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/g_ad` | *absent* | `-1.4919252395629883` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/g_fd` | *absent* | `-1.4760772129923894` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/rel` | *absent* | `0.01073658371737267` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/1/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/1/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/1/dut` | *absent* | `"pec_short"` | — |
| `/ad_vs_fd/1/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/1/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/1/f_minus` | *absent* | `-0.7916232426538564` | — |
| `/ad_vs_fd/1/f_plus` | *absent* | `-0.9392309639530954` | — |
| `/ad_vs_fd/1/fd_ulp_span` | *absent* | `1329532157280521.0` | — |
| `/ad_vs_fd/1/forward_identity/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/1/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/1/forward_identity/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/1/forward_identity/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/1/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/1/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/1/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/1/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/1/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/1/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/1/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/1/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/1/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/1/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/1/forward_identity_float32/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/1/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/1/forward_identity_float32/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/1/forward_identity_float32/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/1/forward_identity_float32/pass` | *absent* | `true` | — |
| `/ad_vs_fd/1/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/1/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/1/g_ad` | *absent* | `-1.4919252395629883` | — |
| `/ad_vs_fd/1/g_fd` | *absent* | `-1.4760772129923894` | — |
| `/ad_vs_fd/1/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/1/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/1/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/1/lane` | *absent* | `"false"` | — |
| `/ad_vs_fd/1/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/1/objective` | *absent* | `"re_s11"` | — |
| `/ad_vs_fd/1/primary_precision` | *absent* | `"float32"` | — |
| `/ad_vs_fd/1/rel` | *absent* | `0.01073658371737267` | — |
| `/ad_vs_fd/1/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/1/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/1/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/1/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/1/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/1/value_at_theta0` | *absent* | `-0.8742772340774536` | — |
| `/ad_vs_fd/1/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/1/wall_time_s/ad` | *absent* | `13.673903703689575` | — |
| `/ad_vs_fd/1/wall_time_s/fd_pair` | *absent* | `13.598352909088135` | — |
| `/ad_vs_fd/1/wall_time_s/x64_witness` | *absent* | `22.179444074630737` | — |
| `/ad_vs_fd/1/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/1/x64_witness` | *absent* | `null` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/f_minus` | *absent* | `-0.6110503089158815` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/f_plus` | *absent* | `-0.3434697479995941` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/fd_ulp_span` | *absent* | `4820302857736721.0` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/g_ad` | *absent* | `2.685720920562744` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/g_fd` | *absent* | `2.675805609162874` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/rel` | *absent* | `0.0037055424975255347` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/2/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/2/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/2/dut` | *absent* | `"pec_short"` | — |
| `/ad_vs_fd/2/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/2/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/2/f_minus` | *absent* | `-0.6110503089158815` | — |
| `/ad_vs_fd/2/f_plus` | *absent* | `-0.3434697479995941` | — |
| `/ad_vs_fd/2/fd_ulp_span` | *absent* | `4820302857736721.0` | — |
| `/ad_vs_fd/2/forward_identity/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/2/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/2/forward_identity/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/2/forward_identity/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/2/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/2/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/2/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/2/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/2/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/2/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/2/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/2/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/2/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/2/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/2/forward_identity_float32/abs_s_at_worst` | *absent* | `0.9998767071529435` | — |
| `/ad_vs_fd/2/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/2/forward_identity_float32/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/2/forward_identity_float32/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/2/forward_identity_float32/pass` | *absent* | `true` | — |
| `/ad_vs_fd/2/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/2/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/2/g_ad` | *absent* | `2.685720920562744` | — |
| `/ad_vs_fd/2/g_fd` | *absent* | `2.675805609162874` | — |
| `/ad_vs_fd/2/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/2/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/2/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/2/lane` | *absent* | `"false"` | — |
| `/ad_vs_fd/2/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/2/objective` | *absent* | `"im_s11"` | — |
| `/ad_vs_fd/2/primary_precision` | *absent* | `"float32"` | — |
| `/ad_vs_fd/2/rel` | *absent* | `0.0037055424975255347` | — |
| `/ad_vs_fd/2/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/2/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/2/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/2/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/2/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/2/value_at_theta0` | *absent* | `-0.48551979660987854` | — |
| `/ad_vs_fd/2/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/2/wall_time_s/ad` | *absent* | `13.69206714630127` | — |
| `/ad_vs_fd/2/wall_time_s/fd_pair` | *absent* | `13.598352909088135` | — |
| `/ad_vs_fd/2/wall_time_s/x64_witness` | *absent* | `22.179444074630737` | — |
| `/ad_vs_fd/2/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/2/x64_witness` | *absent* | `null` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/f_minus` | *absent* | `0.6074386241163148` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/f_plus` | *absent* | `0.5431240309372117` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/fd_ulp_span` | *absent* | `579294355751787.0` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/g_ad` | *absent* | `-6.428282737731934` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/g_fd` | *absent* | `-6.431459317910304` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/rel` | *absent* | `0.00049391281532699` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/3/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/3/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/3/dut` | *absent* | `"pec_short"` | — |
| `/ad_vs_fd/3/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/3/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/3/f_minus` | *absent* | `0.6074386241163148` | — |
| `/ad_vs_fd/3/f_plus` | *absent* | `0.5431240309372117` | — |
| `/ad_vs_fd/3/fd_ulp_span` | *absent* | `579294355751787.0` | — |
| `/ad_vs_fd/3/forward_identity/abs_s_at_worst` | *absent* | `0.80779330690592` | — |
| `/ad_vs_fd/3/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/3/forward_identity/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/3/forward_identity/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/3/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/3/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/3/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/3/forward_identity_concrete_override_vs_plain` | *absent* | `null` | — |
| `/ad_vs_fd/3/forward_identity_float32/abs_s_at_worst` | *absent* | `0.80779330690592` | — |
| `/ad_vs_fd/3/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/3/forward_identity_float32/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/3/forward_identity_float32/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/3/forward_identity_float32/pass` | *absent* | `true` | — |
| `/ad_vs_fd/3/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/3/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/3/g_ad` | *absent* | `-6.428282737731934` | — |
| `/ad_vs_fd/3/g_fd` | *absent* | `-6.431459317910304` | — |
| `/ad_vs_fd/3/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/3/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/3/h` | *absent* | `0.005` | — |
| `/ad_vs_fd/3/lane` | *absent* | `"false"` | — |
| `/ad_vs_fd/3/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/3/objective` | *absent* | `"s11_mag2"` | — |
| `/ad_vs_fd/3/primary_precision` | *absent* | `"float32"` | — |
| `/ad_vs_fd/3/rel` | *absent* | `0.00049391281532699` | — |
| `/ad_vs_fd/3/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/3/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/3/theta0` | *absent* | `0.05` | — |
| `/ad_vs_fd/3/theta_kind` | *absent* | `"sigma"` | — |
| `/ad_vs_fd/3/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/3/value_at_theta0` | *absent* | `0.574424684047699` | — |
| `/ad_vs_fd/3/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/3/wall_time_s/ad` | *absent* | `14.991808891296387` | — |
| `/ad_vs_fd/3/wall_time_s/fd_pair` | *absent* | `14.55851697921753` | — |
| `/ad_vs_fd/3/wall_time_s/x64_witness` | *absent* | `21.833844423294067` | — |
| `/ad_vs_fd/3/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/3/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `0.8077930110398442` | — |
| `/ad_vs_fd/3/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/3/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/3/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/3/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/3/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/3/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/3/x64_witness/g_ad_x64` | *absent* | `-6.428294658660889` | — |
| `/ad_vs_fd/3/x64_witness/value_x64` | *absent* | `0.5744257761484107` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/f_minus` | *absent* | `0.9999992054235289` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/f_plus` | *absent* | `0.9999992002693727` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/fd_ulp_span` | *absent* | `46424512.0` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/g_ad` | *absent* | `2.786023287626449e-05` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/g_fd` | *absent* | `-5.1541562129386875e-08` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/rel` | *absent* | `541.5391634488263` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/4/ad_vs_fd_float32/verdict` | *absent* | `"fail"` | — |
| `/ad_vs_fd/4/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/4/dut` | *absent* | `"pec_short"` | — |
| `/ad_vs_fd/4/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/4/expected_ulp_floor_skip` | *absent* | `true` | — |
| `/ad_vs_fd/4/f_minus` | *absent* | `0.9999992054235289` | — |
| `/ad_vs_fd/4/f_plus` | *absent* | `0.9999992002693727` | — |
| `/ad_vs_fd/4/fd_ulp_span` | *absent* | `46424512.0` | — |
| `/ad_vs_fd/4/forward_identity/abs_s_at_worst` | *absent* | `1.0000045821841472` | — |
| `/ad_vs_fd/4/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/4/forward_identity/max_abs_diff` | *absent* | `1.7554167342883505e-15` | — |
| `/ad_vs_fd/4/forward_identity/max_scaled_diff` | *absent* | `1.7380284854671055e-10` | — |
| `/ad_vs_fd/4/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/4/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/4/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/4/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.9999989626787793` | — |
| `/ad_vs_fd/4/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/4/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/4/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/4/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/4/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/4/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/4/forward_identity_float32/abs_s_at_worst` | *absent* | `1.0000064767307797` | — |
| `/ad_vs_fd/4/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/4/forward_identity_float32/max_abs_diff` | *absent* | `1.0940537680154818e-05` | — |
| `/ad_vs_fd/4/forward_identity_float32/max_scaled_diff` | *absent* | `1.0832146062634656` | — |
| `/ad_vs_fd/4/forward_identity_float32/pass` | *absent* | `false` | — |
| `/ad_vs_fd/4/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/4/forward_identity_float32/worst_entry` | *absent* | `[1, 1, 16]` | 0–2 |
| `/ad_vs_fd/4/g_ad` | *absent* | `-2.9425831371554523e-07` | — |
| `/ad_vs_fd/4/g_fd` | *absent* | `-5.1541562129386875e-08` | — |
| `/ad_vs_fd/4/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/4/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/4/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/4/lane` | *absent* | `"flux"` | — |
| `/ad_vs_fd/4/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/4/objective` | *absent* | `"s11_mag2"` | — |
| `/ad_vs_fd/4/primary_precision` | *absent* | `"x64"` | — |
| `/ad_vs_fd/4/rel` | *absent* | `4.709146202764608` | — |
| `/ad_vs_fd/4/report_only_reason` | *absent* | `"pre-declared zero-derivative objective; AD and FD are O(1e-7) discretization residuals of a physically zero derivative (closing pre-declaration section 2)"` | — |
| `/ad_vs_fd/4/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/4/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/4/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/4/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/4/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/4/value_at_theta0` | *absent* | `1.0000007152557373` | — |
| `/ad_vs_fd/4/verdict` | *absent* | `"report_only"` | — |
| `/ad_vs_fd/4/wall_time_s/ad` | *absent* | `23.63206195831299` | — |
| `/ad_vs_fd/4/wall_time_s/fd_pair` | *absent* | `28.612026929855347` | — |
| `/ad_vs_fd/4/wall_time_s/x64_witness` | *absent* | `94.6474142074585` | — |
| `/ad_vs_fd/4/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/4/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `1.0000045821841472` | — |
| `/ad_vs_fd/4/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/4/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `1.7554167342883505e-15` | — |
| `/ad_vs_fd/4/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `1.7380284854671055e-10` | — |
| `/ad_vs_fd/4/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/4/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/4/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/4/x64_witness/g_ad_x64` | *absent* | `-2.9425831371554523e-07` | — |
| `/ad_vs_fd/4/x64_witness/value_x64` | *absent* | `0.9999992140242192` | — |
| `/ad_vs_fd/4/zero_derivative/ratio` | *absent* | `5.709146202764608` | — |
| `/ad_vs_fd/4/zero_derivative/ratio_max` | *absent* | `3.0` | — |
| `/ad_vs_fd/4/zero_derivative/same_sign` | *absent* | `true` | — |
| `/ad_vs_fd/4/zero_derivative/verdict` | *absent* | `"fail"` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/f_minus` | *absent* | `-0.7941080242518279` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/f_plus` | *absent* | `-0.9404369311543546` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/fd_ulp_span` | *absent* | `1318013621199502.0` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/g_ad` | *absent* | `-1.478940725326538` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/g_fd` | *absent* | `-1.4632890690252665` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/rel` | *absent* | `0.010696216238188394` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/5/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/5/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/5/dut` | *absent* | `"pec_short"` | — |
| `/ad_vs_fd/5/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/5/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/5/f_minus` | *absent* | `-0.7941080242518279` | — |
| `/ad_vs_fd/5/f_plus` | *absent* | `-0.9404369311543546` | — |
| `/ad_vs_fd/5/fd_ulp_span` | *absent* | `1318013621199502.0` | — |
| `/ad_vs_fd/5/forward_identity/abs_s_at_worst` | *absent* | `1.0000045821841472` | — |
| `/ad_vs_fd/5/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/5/forward_identity/max_abs_diff` | *absent* | `1.7554167342883505e-15` | — |
| `/ad_vs_fd/5/forward_identity/max_scaled_diff` | *absent* | `1.7380284854671055e-10` | — |
| `/ad_vs_fd/5/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/5/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/5/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/5/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.9999989626787793` | — |
| `/ad_vs_fd/5/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/5/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/5/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/5/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/5/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/5/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/5/forward_identity_float32/abs_s_at_worst` | *absent* | `1.0000064767307797` | — |
| `/ad_vs_fd/5/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/5/forward_identity_float32/max_abs_diff` | *absent* | `1.0940537680154818e-05` | — |
| `/ad_vs_fd/5/forward_identity_float32/max_scaled_diff` | *absent* | `1.0832146062634656` | — |
| `/ad_vs_fd/5/forward_identity_float32/pass` | *absent* | `false` | — |
| `/ad_vs_fd/5/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/5/forward_identity_float32/worst_entry` | *absent* | `[1, 1, 16]` | 0–2 |
| `/ad_vs_fd/5/g_ad` | *absent* | `-1.4789305925369263` | — |
| `/ad_vs_fd/5/g_fd` | *absent* | `-1.4632890690252665` | — |
| `/ad_vs_fd/5/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/5/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/5/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/5/lane` | *absent* | `"flux"` | — |
| `/ad_vs_fd/5/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/5/objective` | *absent* | `"re_s11"` | — |
| `/ad_vs_fd/5/primary_precision` | *absent* | `"x64"` | — |
| `/ad_vs_fd/5/rel` | *absent* | `0.010689291571130936` | — |
| `/ad_vs_fd/5/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/5/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/5/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/5/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/5/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/5/value_at_theta0` | *absent* | `-0.8761526346206665` | — |
| `/ad_vs_fd/5/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/5/wall_time_s/ad` | *absent* | `22.044326543807983` | — |
| `/ad_vs_fd/5/wall_time_s/fd_pair` | *absent* | `28.612026929855347` | — |
| `/ad_vs_fd/5/wall_time_s/x64_witness` | *absent* | `94.6474142074585` | — |
| `/ad_vs_fd/5/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/5/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `1.0000045821841472` | — |
| `/ad_vs_fd/5/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/5/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `1.7554167342883505e-15` | — |
| `/ad_vs_fd/5/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `1.7380284854671055e-10` | — |
| `/ad_vs_fd/5/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/5/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/5/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/5/x64_witness/g_ad_x64` | *absent* | `-1.4789305925369263` | — |
| `/ad_vs_fd/5/x64_witness/value_x64` | *absent* | `-0.8761514117225249` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/f_minus` | *absent* | `-0.6077759877145421` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/f_plus` | *absent* | `-0.3399670260339267` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/fd_ulp_span` | *absent* | `4824417360125196.0` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/g_ad` | *absent* | `2.6881139278411865` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/g_fd` | *absent* | `2.678089616806154` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/rel` | *absent* | `0.0037430827452993374` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/6/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/6/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/6/dut` | *absent* | `"pec_short"` | — |
| `/ad_vs_fd/6/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/6/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/6/f_minus` | *absent* | `-0.6077759877145421` | — |
| `/ad_vs_fd/6/f_plus` | *absent* | `-0.3399670260339267` | — |
| `/ad_vs_fd/6/fd_ulp_span` | *absent* | `4824417360125196.0` | — |
| `/ad_vs_fd/6/forward_identity/abs_s_at_worst` | *absent* | `1.0000045821841472` | — |
| `/ad_vs_fd/6/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/6/forward_identity/max_abs_diff` | *absent* | `1.7554167342883505e-15` | — |
| `/ad_vs_fd/6/forward_identity/max_scaled_diff` | *absent* | `1.7380284854671055e-10` | — |
| `/ad_vs_fd/6/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/6/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/6/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/6/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.9999989626787793` | — |
| `/ad_vs_fd/6/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/6/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/6/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/6/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/6/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/6/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/6/forward_identity_float32/abs_s_at_worst` | *absent* | `1.0000064767307797` | — |
| `/ad_vs_fd/6/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/6/forward_identity_float32/max_abs_diff` | *absent* | `1.0940537680154818e-05` | — |
| `/ad_vs_fd/6/forward_identity_float32/max_scaled_diff` | *absent* | `1.0832146062634656` | — |
| `/ad_vs_fd/6/forward_identity_float32/pass` | *absent* | `false` | — |
| `/ad_vs_fd/6/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/6/forward_identity_float32/worst_entry` | *absent* | `[1, 1, 16]` | 0–2 |
| `/ad_vs_fd/6/g_ad` | *absent* | `2.688117504119873` | — |
| `/ad_vs_fd/6/g_fd` | *absent* | `2.678089616806154` | — |
| `/ad_vs_fd/6/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/6/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/6/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/6/lane` | *absent* | `"flux"` | — |
| `/ad_vs_fd/6/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/6/objective` | *absent* | `"im_s11"` | — |
| `/ad_vs_fd/6/primary_precision` | *absent* | `"x64"` | — |
| `/ad_vs_fd/6/rel` | *absent* | `0.003744418129546407` | — |
| `/ad_vs_fd/6/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/6/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/6/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/6/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/6/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/6/value_at_theta0` | *absent* | `-0.4820345342159271` | — |
| `/ad_vs_fd/6/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/6/wall_time_s/ad` | *absent* | `21.776337385177612` | — |
| `/ad_vs_fd/6/wall_time_s/fd_pair` | *absent* | `28.612026929855347` | — |
| `/ad_vs_fd/6/wall_time_s/x64_witness` | *absent* | `94.6474142074585` | — |
| `/ad_vs_fd/6/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/6/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `1.0000045821841472` | — |
| `/ad_vs_fd/6/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/6/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `1.7554167342883505e-15` | — |
| `/ad_vs_fd/6/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `1.7380284854671055e-10` | — |
| `/ad_vs_fd/6/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/6/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/6/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/6/x64_witness/g_ad_x64` | *absent* | `2.688117504119873` | — |
| `/ad_vs_fd/6/x64_witness/value_x64` | *absent* | `-0.48203518311513927` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/f_minus` | *absent* | `0.6081239566961222` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/f_plus` | *absent* | `0.5438781008070093` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/fd_ulp_span` | *absent* | `578675225284615.0` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/g_ad` | *absent* | `-6.4214372634887695` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/g_fd` | *absent* | `-6.424585588911291` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/rel` | *absent* | `0.0004900433466020363` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/7/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/7/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/7/dut` | *absent* | `"pec_short"` | — |
| `/ad_vs_fd/7/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/7/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/7/f_minus` | *absent* | `0.6081239566961222` | — |
| `/ad_vs_fd/7/f_plus` | *absent* | `0.5438781008070093` | — |
| `/ad_vs_fd/7/fd_ulp_span` | *absent* | `578675225284615.0` | — |
| `/ad_vs_fd/7/forward_identity/abs_s_at_worst` | *absent* | `0.8060908199945326` | — |
| `/ad_vs_fd/7/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/7/forward_identity/max_abs_diff` | *absent* | `2.3135566778103808e-15` | — |
| `/ad_vs_fd/7/forward_identity/max_scaled_diff` | *absent* | `2.8349255023183324e-10` | — |
| `/ad_vs_fd/7/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/7/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/7/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/7/forward_identity_concrete_override_vs_plain` | *absent* | `null` | — |
| `/ad_vs_fd/7/forward_identity_float32/abs_s_at_worst` | *absent* | `0.7474536620891785` | — |
| `/ad_vs_fd/7/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/7/forward_identity_float32/max_abs_diff` | *absent* | `1.1474474669297353e-05` | — |
| `/ad_vs_fd/7/forward_identity_float32/max_scaled_diff` | *absent* | `1.5148748027237617` | — |
| `/ad_vs_fd/7/forward_identity_float32/pass` | *absent* | `false` | — |
| `/ad_vs_fd/7/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/7/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 16]` | 0–2 |
| `/ad_vs_fd/7/g_ad` | *absent* | `-6.421442031860352` | — |
| `/ad_vs_fd/7/g_fd` | *absent* | `-6.424585588911291` | — |
| `/ad_vs_fd/7/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/7/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/7/h` | *absent* | `0.005` | — |
| `/ad_vs_fd/7/lane` | *absent* | `"flux"` | — |
| `/ad_vs_fd/7/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/7/objective` | *absent* | `"s11_mag2"` | — |
| `/ad_vs_fd/7/primary_precision` | *absent* | `"x64"` | — |
| `/ad_vs_fd/7/rel` | *absent* | `0.0004893011397287616` | — |
| `/ad_vs_fd/7/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/7/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/7/theta0` | *absent* | `0.05` | — |
| `/ad_vs_fd/7/theta_kind` | *absent* | `"sigma"` | — |
| `/ad_vs_fd/7/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/7/value_at_theta0` | *absent* | `0.5751491785049438` | — |
| `/ad_vs_fd/7/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/7/wall_time_s/ad` | *absent* | `24.24846363067627` | — |
| `/ad_vs_fd/7/wall_time_s/fd_pair` | *absent* | `28.816654443740845` | — |
| `/ad_vs_fd/7/wall_time_s/x64_witness` | *absent* | `41.21760630607605` | — |
| `/ad_vs_fd/7/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/7/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `0.8060908199945326` | — |
| `/ad_vs_fd/7/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/7/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `2.3135566778103808e-15` | — |
| `/ad_vs_fd/7/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `2.8349255023183324e-10` | — |
| `/ad_vs_fd/7/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/7/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/7/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/7/x64_witness/g_ad_x64` | *absent* | `-6.421442031860352` | — |
| `/ad_vs_fd/7/x64_witness/value_x64` | *absent* | `0.5751489992408281` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/f_minus` | *absent* | `0.3917346730025593` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/f_plus` | *absent* | `0.42310869464333195` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/fd_ulp_span` | *absent* | `565184128681990.0` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/g_ad` | *absent* | `0.313778817653656` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/g_fd` | *absent* | `0.3137402164077263` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/rel` | *absent* | `0.00012303569612997145` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/8/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/8/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/8/dut` | *absent* | `"slab"` | — |
| `/ad_vs_fd/8/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/8/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/8/f_minus` | *absent* | `0.3917346730025593` | — |
| `/ad_vs_fd/8/f_plus` | *absent* | `0.42310869464333195` | — |
| `/ad_vs_fd/8/fd_ulp_span` | *absent* | `565184128681990.0` | — |
| `/ad_vs_fd/8/forward_identity/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/8/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/8/forward_identity/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/forward_identity/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/8/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/8/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/8/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/8/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/8/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/8/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/8/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/8/forward_identity_float32/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/8/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/8/forward_identity_float32/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/forward_identity_float32/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/forward_identity_float32/pass` | *absent* | `true` | — |
| `/ad_vs_fd/8/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/8/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/8/g_ad` | *absent* | `0.313778817653656` | — |
| `/ad_vs_fd/8/g_fd` | *absent* | `0.3137402164077263` | — |
| `/ad_vs_fd/8/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/8/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/8/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/8/lane` | *absent* | `"false"` | — |
| `/ad_vs_fd/8/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/8/objective` | *absent* | `"s11_mag2"` | — |
| `/ad_vs_fd/8/primary_precision` | *absent* | `"float32"` | — |
| `/ad_vs_fd/8/rel` | *absent* | `0.00012303569612997145` | — |
| `/ad_vs_fd/8/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/8/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/8/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/8/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/8/value_at_theta0` | *absent* | `0.40772706270217896` | — |
| `/ad_vs_fd/8/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/8/wall_time_s/ad` | *absent* | `16.212509393692017` | — |
| `/ad_vs_fd/8/wall_time_s/fd_pair` | *absent* | `15.666167974472046` | — |
| `/ad_vs_fd/8/wall_time_s/x64_witness` | *absent* | `23.301864624023438` | — |
| `/ad_vs_fd/8/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/8/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `0.20411501936617793` | — |
| `/ad_vs_fd/8/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/8/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/8/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/8/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/8/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/8/x64_witness/g_ad_x64` | *absent* | `0.31377890706062317` | — |
| `/ad_vs_fd/8/x64_witness/value_x64` | *absent* | `0.40772705883597776` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/f_minus` | *absent* | `0.6079900693090815` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/f_plus` | *absent* | `0.5766197435802966` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/fd_ulp_span` | *absent* | `282558774525294.0` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/g_ad` | *absent* | `-0.31374168395996094` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/g_fd` | *absent* | `-0.31370325728784954` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/rel` | *absent* | `0.00012249369816438543` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/9/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/9/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/9/dut` | *absent* | `"slab"` | — |
| `/ad_vs_fd/9/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/9/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/9/f_minus` | *absent* | `0.6079900693090815` | — |
| `/ad_vs_fd/9/f_plus` | *absent* | `0.5766197435802966` | — |
| `/ad_vs_fd/9/fd_ulp_span` | *absent* | `282558774525294.0` | — |
| `/ad_vs_fd/9/forward_identity/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/9/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/9/forward_identity/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/9/forward_identity/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/9/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/9/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/9/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/9/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/9/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/9/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/9/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/9/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/9/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/9/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/9/forward_identity_float32/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/9/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/9/forward_identity_float32/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/9/forward_identity_float32/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/9/forward_identity_float32/pass` | *absent* | `true` | — |
| `/ad_vs_fd/9/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/9/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/9/g_ad` | *absent* | `-0.31374168395996094` | — |
| `/ad_vs_fd/9/g_fd` | *absent* | `-0.31370325728784954` | — |
| `/ad_vs_fd/9/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/9/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/9/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/9/lane` | *absent* | `"false"` | — |
| `/ad_vs_fd/9/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/9/objective` | *absent* | `"s21_mag2"` | — |
| `/ad_vs_fd/9/primary_precision` | *absent* | `"float32"` | — |
| `/ad_vs_fd/9/rel` | *absent* | `0.00012249369816438543` | — |
| `/ad_vs_fd/9/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/9/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/9/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/9/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/9/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/9/value_at_theta0` | *absent* | `0.5919994115829468` | — |
| `/ad_vs_fd/9/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/9/wall_time_s/ad` | *absent* | `13.969720602035522` | — |
| `/ad_vs_fd/9/wall_time_s/fd_pair` | *absent* | `15.666167974472046` | — |
| `/ad_vs_fd/9/wall_time_s/x64_witness` | *absent* | `23.301864624023438` | — |
| `/ad_vs_fd/9/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/9/x64_witness` | *absent* | `null` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/f_minus` | *absent* | `-0.7507499357536281` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/f_plus` | *absent* | `-0.7410395731771008` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/fd_ulp_span` | *absent* | `87463170562562.0` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/g_ad` | *absent* | `0.09743057191371918` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/g_fd` | *absent* | `0.09710362576527354` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/rel` | *absent* | `0.0033669818801201003` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/10/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/10/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/10/dut` | *absent* | `"slab"` | — |
| `/ad_vs_fd/10/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/10/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/10/f_minus` | *absent* | `-0.7507499357536281` | — |
| `/ad_vs_fd/10/f_plus` | *absent* | `-0.7410395731771008` | — |
| `/ad_vs_fd/10/fd_ulp_span` | *absent* | `87463170562562.0` | — |
| `/ad_vs_fd/10/forward_identity/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/10/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/10/forward_identity/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/10/forward_identity/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/10/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/10/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/10/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/10/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/10/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/10/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/10/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/10/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/10/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/10/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/10/forward_identity_float32/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/10/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/10/forward_identity_float32/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/10/forward_identity_float32/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/10/forward_identity_float32/pass` | *absent* | `true` | — |
| `/ad_vs_fd/10/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/10/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/10/g_ad` | *absent* | `0.09743057191371918` | — |
| `/ad_vs_fd/10/g_fd` | *absent* | `0.09710362576527354` | — |
| `/ad_vs_fd/10/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/10/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/10/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/10/lane` | *absent* | `"false"` | — |
| `/ad_vs_fd/10/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/10/objective` | *absent* | `"re_s21"` | — |
| `/ad_vs_fd/10/primary_precision` | *absent* | `"float32"` | — |
| `/ad_vs_fd/10/rel` | *absent* | `0.0033669818801201003` | — |
| `/ad_vs_fd/10/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/10/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/10/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/10/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/10/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/10/value_at_theta0` | *absent* | `-0.7461746335029602` | — |
| `/ad_vs_fd/10/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/10/wall_time_s/ad` | *absent* | `14.02440094947815` | — |
| `/ad_vs_fd/10/wall_time_s/fd_pair` | *absent* | `15.666167974472046` | — |
| `/ad_vs_fd/10/wall_time_s/x64_witness` | *absent* | `23.301864624023438` | — |
| `/ad_vs_fd/10/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/10/x64_witness` | *absent* | `null` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/f_minus` | *absent* | `-0.21062906559875574` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/f_plus` | *absent* | `-0.16577121151091562` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/fd_ulp_span` | *absent* | `1616174519637095.0` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/g_ad` | *absent* | `0.44848573207855225` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/g_fd` | *absent* | `0.44857854087840127` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/rel` | *absent* | `0.00020689531796880658` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/11/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/11/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/11/dut` | *absent* | `"slab"` | — |
| `/ad_vs_fd/11/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/11/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/11/f_minus` | *absent* | `-0.21062906559875574` | — |
| `/ad_vs_fd/11/f_plus` | *absent* | `-0.16577121151091562` | — |
| `/ad_vs_fd/11/fd_ulp_span` | *absent* | `1616174519637095.0` | — |
| `/ad_vs_fd/11/forward_identity/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/11/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/11/forward_identity/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/11/forward_identity/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/11/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/11/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/11/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/11/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/11/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/11/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/11/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/11/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/11/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/11/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/11/forward_identity_float32/abs_s_at_worst` | *absent* | `0.20411509704882477` | — |
| `/ad_vs_fd/11/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/11/forward_identity_float32/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/11/forward_identity_float32/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/11/forward_identity_float32/pass` | *absent* | `true` | — |
| `/ad_vs_fd/11/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/11/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/11/g_ad` | *absent* | `0.44848573207855225` | — |
| `/ad_vs_fd/11/g_fd` | *absent* | `0.44857854087840127` | — |
| `/ad_vs_fd/11/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/11/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/11/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/11/lane` | *absent* | `"false"` | — |
| `/ad_vs_fd/11/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/11/objective` | *absent* | `"im_s21"` | — |
| `/ad_vs_fd/11/primary_precision` | *absent* | `"float32"` | — |
| `/ad_vs_fd/11/rel` | *absent* | `0.00020689531796880658` | — |
| `/ad_vs_fd/11/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/11/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/11/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/11/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/11/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/11/value_at_theta0` | *absent* | `-0.18767736852169037` | — |
| `/ad_vs_fd/11/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/11/wall_time_s/ad` | *absent* | `13.830885171890259` | — |
| `/ad_vs_fd/11/wall_time_s/fd_pair` | *absent* | `15.666167974472046` | — |
| `/ad_vs_fd/11/wall_time_s/x64_witness` | *absent* | `23.301864624023438` | — |
| `/ad_vs_fd/11/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/11/x64_witness` | *absent* | `null` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/f_minus` | *absent* | `0.3890478751791018` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/f_plus` | *absent* | `0.42054142024003516` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/fd_ulp_span` | *absent* | `567337271203982.0` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/g_ad` | *absent* | `0.3149757981300354` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/g_fd` | *absent* | `0.3149354506093338` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/rel` | *absent* | `0.00012811362018319986` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/12/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/12/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/12/dut` | *absent* | `"slab"` | — |
| `/ad_vs_fd/12/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/12/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/12/f_minus` | *absent* | `0.3890478751791018` | — |
| `/ad_vs_fd/12/f_plus` | *absent* | `0.42054142024003516` | — |
| `/ad_vs_fd/12/fd_ulp_span` | *absent* | `567337271203982.0` | — |
| `/ad_vs_fd/12/forward_identity/abs_s_at_worst` | *absent* | `0.20266831557969123` | — |
| `/ad_vs_fd/12/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/12/forward_identity/max_abs_diff` | *absent* | `2.2353207304865925e-14` | — |
| `/ad_vs_fd/12/forward_identity/max_scaled_diff` | *absent* | `1.0510831029970571e-08` | — |
| `/ad_vs_fd/12/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/12/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/12/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/12/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.2026725196948864` | — |
| `/ad_vs_fd/12/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/12/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/12/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/12/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/12/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/12/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/12/forward_identity_float32/abs_s_at_worst` | *absent* | `0.2026725196948864` | — |
| `/ad_vs_fd/12/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/12/forward_identity_float32/max_abs_diff` | *absent* | `9.150247281516294e-06` | — |
| `/ad_vs_fd/12/forward_identity_float32/max_scaled_diff` | *absent* | `1.7607200487636583` | — |
| `/ad_vs_fd/12/forward_identity_float32/pass` | *absent* | `false` | — |
| `/ad_vs_fd/12/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/12/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/12/g_ad` | *absent* | `0.3149772882461548` | — |
| `/ad_vs_fd/12/g_fd` | *absent* | `0.3149354506093338` | — |
| `/ad_vs_fd/12/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/12/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/12/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/12/lane` | *absent* | `"flux"` | — |
| `/ad_vs_fd/12/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/12/objective` | *absent* | `"s11_mag2"` | — |
| `/ad_vs_fd/12/primary_precision` | *absent* | `"x64"` | — |
| `/ad_vs_fd/12/rel` | *absent* | `0.00013284511711852964` | — |
| `/ad_vs_fd/12/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/12/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/12/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/12/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/12/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/12/value_at_theta0` | *absent* | `0.4050973355770111` | — |
| `/ad_vs_fd/12/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/12/wall_time_s/ad` | *absent* | `25.397393703460693` | — |
| `/ad_vs_fd/12/wall_time_s/fd_pair` | *absent* | `28.023831367492676` | — |
| `/ad_vs_fd/12/wall_time_s/x64_witness` | *absent* | `122.00688409805298` | — |
| `/ad_vs_fd/12/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/12/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `0.20266831557969123` | — |
| `/ad_vs_fd/12/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/12/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `2.2353207304865925e-14` | — |
| `/ad_vs_fd/12/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `1.0510831029970571e-08` | — |
| `/ad_vs_fd/12/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/12/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/12/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/12/x64_witness/g_ad_x64` | *absent* | `0.3149772882461548` | — |
| `/ad_vs_fd/12/x64_witness/value_x64` | *absent* | `0.40509848356585926` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/f_minus` | *absent* | `0.6109511571698815` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/f_plus` | *absent* | `0.5794575562012311` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/fd_ulp_span` | *absent* | `283669139173938.0` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/g_ad` | *absent* | `-0.3149767816066742` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/g_fd` | *absent* | `-0.3149360096865039` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/rel` | *absent* | `0.0001294609664066482` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/13/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/13/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/13/dut` | *absent* | `"slab"` | — |
| `/ad_vs_fd/13/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/13/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/13/f_minus` | *absent* | `0.6109511571698815` | — |
| `/ad_vs_fd/13/f_plus` | *absent* | `0.5794575562012311` | — |
| `/ad_vs_fd/13/fd_ulp_span` | *absent* | `283669139173938.0` | — |
| `/ad_vs_fd/13/forward_identity/abs_s_at_worst` | *absent* | `0.20266831557969123` | — |
| `/ad_vs_fd/13/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/13/forward_identity/max_abs_diff` | *absent* | `2.2353207304865925e-14` | — |
| `/ad_vs_fd/13/forward_identity/max_scaled_diff` | *absent* | `1.0510831029970571e-08` | — |
| `/ad_vs_fd/13/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/13/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/13/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/13/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.2026725196948864` | — |
| `/ad_vs_fd/13/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/13/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/13/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/13/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/13/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/13/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/13/forward_identity_float32/abs_s_at_worst` | *absent* | `0.2026725196948864` | — |
| `/ad_vs_fd/13/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/13/forward_identity_float32/max_abs_diff` | *absent* | `9.150247281516294e-06` | — |
| `/ad_vs_fd/13/forward_identity_float32/max_scaled_diff` | *absent* | `1.7607200487636583` | — |
| `/ad_vs_fd/13/forward_identity_float32/pass` | *absent* | `false` | — |
| `/ad_vs_fd/13/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/13/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/13/g_ad` | *absent* | `-0.3149776756763458` | — |
| `/ad_vs_fd/13/g_fd` | *absent* | `-0.3149360096865039` | — |
| `/ad_vs_fd/13/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/13/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/13/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/13/lane` | *absent* | `"flux"` | — |
| `/ad_vs_fd/13/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/13/objective` | *absent* | `"s21_mag2"` | — |
| `/ad_vs_fd/13/primary_precision` | *absent* | `"x64"` | — |
| `/ad_vs_fd/13/rel` | *absent* | `0.0001322998595282091` | — |
| `/ad_vs_fd/13/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/13/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/13/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/13/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/13/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/13/value_at_theta0` | *absent* | `0.5948984622955322` | — |
| `/ad_vs_fd/13/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/13/wall_time_s/ad` | *absent* | `21.661340951919556` | — |
| `/ad_vs_fd/13/wall_time_s/fd_pair` | *absent* | `28.023831367492676` | — |
| `/ad_vs_fd/13/wall_time_s/x64_witness` | *absent* | `122.00688409805298` | — |
| `/ad_vs_fd/13/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/13/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `0.20266831557969123` | — |
| `/ad_vs_fd/13/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/13/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `2.2353207304865925e-14` | — |
| `/ad_vs_fd/13/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `1.0510831029970571e-08` | — |
| `/ad_vs_fd/13/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/13/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/13/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/13/x64_witness/g_ad_x64` | *absent* | `-0.3149776756763458` | — |
| `/ad_vs_fd/13/x64_witness/value_x64` | *absent* | `0.5949005137509278` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/f_minus` | *absent* | `-0.7528147820912482` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/f_plus` | *absent* | `-0.7430776215403447` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/fd_ulp_span` | *absent* | `87704545257391.0` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/g_ad` | *absent* | `0.09770140051841736` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/g_fd` | *absent* | `0.09737160550903456` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/rel` | *absent* | `0.003386973108420135` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/14/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/14/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/14/dut` | *absent* | `"slab"` | — |
| `/ad_vs_fd/14/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/14/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/14/f_minus` | *absent* | `-0.7528147820912482` | — |
| `/ad_vs_fd/14/f_plus` | *absent* | `-0.7430776215403447` | — |
| `/ad_vs_fd/14/fd_ulp_span` | *absent* | `87704545257391.0` | — |
| `/ad_vs_fd/14/forward_identity/abs_s_at_worst` | *absent* | `0.20266831557969123` | — |
| `/ad_vs_fd/14/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/14/forward_identity/max_abs_diff` | *absent* | `2.2353207304865925e-14` | — |
| `/ad_vs_fd/14/forward_identity/max_scaled_diff` | *absent* | `1.0510831029970571e-08` | — |
| `/ad_vs_fd/14/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/14/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/14/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/14/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.2026725196948864` | — |
| `/ad_vs_fd/14/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/14/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/14/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/14/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/14/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/14/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/14/forward_identity_float32/abs_s_at_worst` | *absent* | `0.2026725196948864` | — |
| `/ad_vs_fd/14/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/14/forward_identity_float32/max_abs_diff` | *absent* | `9.150247281516294e-06` | — |
| `/ad_vs_fd/14/forward_identity_float32/max_scaled_diff` | *absent* | `1.7607200487636583` | — |
| `/ad_vs_fd/14/forward_identity_float32/pass` | *absent* | `false` | — |
| `/ad_vs_fd/14/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/14/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/14/g_ad` | *absent* | `0.09770097583532333` | — |
| `/ad_vs_fd/14/g_fd` | *absent* | `0.09737160550903456` | — |
| `/ad_vs_fd/14/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/14/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/14/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/14/lane` | *absent* | `"flux"` | — |
| `/ad_vs_fd/14/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/14/objective` | *absent* | `"re_s21"` | — |
| `/ad_vs_fd/14/primary_precision` | *absent* | `"x64"` | — |
| `/ad_vs_fd/14/rel` | *absent* | `0.003382611640908076` | — |
| `/ad_vs_fd/14/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/14/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/14/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/14/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/14/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/14/value_at_theta0` | *absent* | `-0.7482287883758545` | — |
| `/ad_vs_fd/14/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/14/wall_time_s/ad` | *absent* | `21.6990008354187` | — |
| `/ad_vs_fd/14/wall_time_s/fd_pair` | *absent* | `28.023831367492676` | — |
| `/ad_vs_fd/14/wall_time_s/x64_witness` | *absent* | `122.00688409805298` | — |
| `/ad_vs_fd/14/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/14/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `0.20266831557969123` | — |
| `/ad_vs_fd/14/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/14/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `2.2353207304865925e-14` | — |
| `/ad_vs_fd/14/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `1.0510831029970571e-08` | — |
| `/ad_vs_fd/14/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/14/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/14/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/14/x64_witness/g_ad_x64` | *absent* | `0.09770097583532333` | — |
| `/ad_vs_fd/14/x64_witness/value_x64` | *absent* | `-0.7482298705603626` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/f_minus` | *absent* | `-0.21028804301430898` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/f_plus` | *absent* | `-0.16520655122353775` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/fd_ulp_span` | *absent* | `1624231917041787.0` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/g_ad` | *absent* | `0.4507231116294861` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/g_fd` | *absent* | `0.4508149179077123` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/rel` | *absent* | `0.00020364516474360742` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/15/ad_vs_fd_float32/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/15/checkpoint_segments` | *absent* | `37` | — |
| `/ad_vs_fd/15/dut` | *absent* | `"slab"` | — |
| `/ad_vs_fd/15/dx_m` | *absent* | `0.000635` | — |
| `/ad_vs_fd/15/expected_ulp_floor_skip` | *absent* | `false` | — |
| `/ad_vs_fd/15/f_minus` | *absent* | `-0.21028804301430898` | — |
| `/ad_vs_fd/15/f_plus` | *absent* | `-0.16520655122353775` | — |
| `/ad_vs_fd/15/fd_ulp_span` | *absent* | `1624231917041787.0` | — |
| `/ad_vs_fd/15/forward_identity/abs_s_at_worst` | *absent* | `0.20266831557969123` | — |
| `/ad_vs_fd/15/forward_identity/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/15/forward_identity/max_abs_diff` | *absent* | `2.2353207304865925e-14` | — |
| `/ad_vs_fd/15/forward_identity/max_scaled_diff` | *absent* | `1.0510831029970571e-08` | — |
| `/ad_vs_fd/15/forward_identity/pass` | *absent* | `true` | — |
| `/ad_vs_fd/15/forward_identity/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/15/forward_identity/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/15/forward_identity_concrete_override_vs_plain/abs_s_at_worst` | *absent* | `0.2026725196948864` | — |
| `/ad_vs_fd/15/forward_identity_concrete_override_vs_plain/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/15/forward_identity_concrete_override_vs_plain/max_abs_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/15/forward_identity_concrete_override_vs_plain/max_scaled_diff` | *absent* | `0.0` | — |
| `/ad_vs_fd/15/forward_identity_concrete_override_vs_plain/pass` | *absent* | `true` | — |
| `/ad_vs_fd/15/forward_identity_concrete_override_vs_plain/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/15/forward_identity_concrete_override_vs_plain/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/15/forward_identity_float32/abs_s_at_worst` | *absent* | `0.2026725196948864` | — |
| `/ad_vs_fd/15/forward_identity_float32/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/15/forward_identity_float32/max_abs_diff` | *absent* | `9.150247281516294e-06` | — |
| `/ad_vs_fd/15/forward_identity_float32/max_scaled_diff` | *absent* | `1.7607200487636583` | — |
| `/ad_vs_fd/15/forward_identity_float32/pass` | *absent* | `false` | — |
| `/ad_vs_fd/15/forward_identity_float32/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/15/forward_identity_float32/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/15/g_ad` | *absent* | `0.45072394609451294` | — |
| `/ad_vs_fd/15/g_fd` | *absent* | `0.4508149179077123` | — |
| `/ad_vs_fd/15/gate` | *absent* | `0.05` | — |
| `/ad_vs_fd/15/grad_dtype` | *absent* | `"float32"` | — |
| `/ad_vs_fd/15/h` | *absent* | `0.05` | — |
| `/ad_vs_fd/15/lane` | *absent* | `"flux"` | — |
| `/ad_vs_fd/15/loss_dtype` | *absent* | `"float64"` | — |
| `/ad_vs_fd/15/objective` | *absent* | `"im_s21"` | — |
| `/ad_vs_fd/15/primary_precision` | *absent* | `"x64"` | — |
| `/ad_vs_fd/15/rel` | *absent* | `0.0002017941500728234` | — |
| `/ad_vs_fd/15/rung` | *absent* | `"fine"` | — |
| `/ad_vs_fd/15/s_dtype_fd` | *absent* | `"complex128"` | — |
| `/ad_vs_fd/15/theta0` | *absent* | `0.0` | — |
| `/ad_vs_fd/15/theta_kind` | *absent* | `"eps"` | — |
| `/ad_vs_fd/15/ulp_floor` | *absent* | `10000.0` | — |
| `/ad_vs_fd/15/value_at_theta0` | *absent* | `-0.18722222745418549` | — |
| `/ad_vs_fd/15/verdict` | *absent* | `"pass"` | — |
| `/ad_vs_fd/15/wall_time_s/ad` | *absent* | `21.84258222579956` | — |
| `/ad_vs_fd/15/wall_time_s/fd_pair` | *absent* | `28.023831367492676` | — |
| `/ad_vs_fd/15/wall_time_s/x64_witness` | *absent* | `122.00688409805298` | — |
| `/ad_vs_fd/15/x64_context` | *absent* | `true` | — |
| `/ad_vs_fd/15/x64_witness/forward_identity_x64/abs_s_at_worst` | *absent* | `0.20266831557969123` | — |
| `/ad_vs_fd/15/x64_witness/forward_identity_x64/atol` | *absent* | `1e-07` | — |
| `/ad_vs_fd/15/x64_witness/forward_identity_x64/max_abs_diff` | *absent* | `2.2353207304865925e-14` | — |
| `/ad_vs_fd/15/x64_witness/forward_identity_x64/max_scaled_diff` | *absent* | `1.0510831029970571e-08` | — |
| `/ad_vs_fd/15/x64_witness/forward_identity_x64/pass` | *absent* | `true` | — |
| `/ad_vs_fd/15/x64_witness/forward_identity_x64/rtol` | *absent* | `1e-05` | — |
| `/ad_vs_fd/15/x64_witness/forward_identity_x64/worst_entry` | *absent* | `[0, 0, 0]` | 0–2 |
| `/ad_vs_fd/15/x64_witness/g_ad_x64` | *absent* | `0.45072394609451294` | — |
| `/ad_vs_fd/15/x64_witness/value_x64` | *absent* | `-0.1872233280126994` | — |
| `/cells/0/column_power_max` | *absent* | `1.0047155530599599` | — |
| `/cells/0/column_power_per_bin` | *absent* | `[[0.9973186577165408, 0.995566568406492, 0.9944072520571304, 0.994050417470522, 0.9945839059230541, 0.9959931462246061, 0.997985004689288, 1.0002590175664072, 1.0023775943946882, 1.0039543840119267, 1.0047155530599599, 1.0045297494418752, 1.0035304406785572, 1.0019485288855106, 1.0002044664004826, 0.998713574201318, 0.9979474407098303], [0.9961847308824365, 0.9948681788252146, 0.9945725320530022, 0.995031941046812, 0.9962968899403931, 0.998109218548539, 1.0000127968129462, 1.0016445444113755, 1.0025523738448252, 1.002618392858395, 1.0018739891006065, 1.0006101324508614, 0.9992463335812048, 0.9981814612706686, 0.9978633702573597, 0.9985771697646231, 1.0006459162294874]]` | 0–1 |
| `/cells/0/cpml_layers` | *absent* | `17` | — |
| `/cells/0/dt_s` | *absent* | `4.842700168608755e-12` | — |
| `/cells/0/dut` | *absent* | `"pec_short"` | — |
| `/cells/0/dut_cells` | *absent* | `72` | — |
| `/cells/0/dut_runs_xyz` | *absent* | `[2, 9, 4]` | 0–2 |
| `/cells/0/dx_m` | *absent* | `0.00254` | — |
| `/cells/0/fc_discrete_guide_hz` | *absent* | `6523900723.790886` | — |
| `/cells/0/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/0/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/0/grid_shape` | *absent* | `[83, 10, 5]` | 0–2 |
| `/cells/0/guide_cells_yz` | *absent* | `[9, 4]` | 0–1 |
| `/cells/0/lane` | *absent* | `"false"` | — |
| `/cells/0/n_steps` | *absent* | `713` | — |
| `/cells/0/non_vacuity_max_s11` | *absent* | `1.0023550035092157` | — |
| `/cells/0/num_periods` | *absent* | `40.0` | — |
| `/cells/0/port_f_cutoff_hz` | *absent* | `[6523900723.7908745, 6523900723.7908745]` | 0–1 |
| `/cells/0/power_closure_max` | *absent* | `0.00594958252947797` | — |
| `/cells/0/preflight` | *absent* | `[]` | — |
| `/cells/0/reciprocity_complex_max` | *absent* | `1.468931504282033e-21` | — |
| `/cells/0/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `f3b610f1108a` | 0–16 |
| `/cells/0/reciprocity_mag_mean` | *absent* | `5.856469615409605e-10` | — |
| `/cells/0/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/0/rung` | *absent* | `"coarse"` | — |
| `/cells/0/s_params/S11` | *absent* | array[17], SHA256 `d0f188529a01` | 0–16 |
| `/cells/0/s_params/S12` | *absent* | array[17], SHA256 `77803469c3e9` | 0–16 |
| `/cells/0/s_params/S21` | *absent* | array[17], SHA256 `47c60c343ea9` | 0–16 |
| `/cells/0/s_params/S22` | *absent* | array[17], SHA256 `c35b5c719efb` | 0–16 |
| `/cells/0/settling_db/left` | *absent* | `-85.55851995120653` | — |
| `/cells/0/settling_db/right` | *absent* | `-81.82784975937672` | — |
| `/cells/0/settling_db_over_normal_records` | *absent* | `-81.82784975937672` | — |
| `/cells/0/settling_degenerate_records` | *absent* | array[8], SHA256 `479e776c8a0e` | 0–7 |
| `/cells/0/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.2577648240185795, "end": 7.167565258694274e-10, "n_nonzero": 702, "n_steps": 713, "db": -85.55851995120653, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.668860185716543, "end": 8.482791169575122e-10, "n_nonzero": 709, "n_steps": 713, "db": -88.9679656947883, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 1.0823295082812922e-05, "end": 9.82804465375299e-16, "n_nonzero": 702, "n_steps": 713, "db": -100.4189237804148, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 6.425294725427447e-06, "end": 3.673730091074864e-15, "n_nonzero": 709, "n_steps": 713, "db": -92.42785807826857, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 8.93299502404371e-40, "end": 2.8192270727888778e-49, "n_nonzero": 684, "n_steps": 713, "db": -95.00867034829751, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 8.715224501676256e-40, "end": 4.1065383157994424e-49, "n_nonzero": 677, "n_steps": 713, "db": -93.26802699551243, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.1764350322884696e-45, "end": 4.941674443190604e-55, "n_nonzero": 684, "n_steps": 713, "db": -98.08065846375567, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.2671324639996316e-45, "end": 9.855744740854169e-55, "n_nonzero": 676, "n_steps": 713, "db": -95.20477296107936, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 9.164508701810065e-40, "end": 1.873820946253308e-49, "n_nonzero": 684, "n_steps": 713, "db": -96.89381098579196, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 9.305938165145415e-40, "end": 4.175748835802808e-49, "n_nonzero": 677, "n_steps": 713, "db": -93.48025793760263, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.3718765864157466e-45, "end": 2.589176481154959e-54, "n_nonzero": 684, "n_steps": 713, "db": -91.14710017252145, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.342230946809503e-45, "end": 1.1703822425444902e-54, "n_nonzero": 676, "n_steps": 713, "db": -94.55708732270715, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.6992021284552266, "end": 1.7198631625551836e-09, "n_nonzero": 702, "n_steps": 713, "db": -86.09108847155625, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.7051307975936005, "end": 4.62897363908177e-09, "n_nonzero": 709, "n_steps": 713, "db": -81.82784975937672, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 8.221049155890609e-06, "end": 5.478092416617927e-15, "n_nonzero": 702, "n_steps": 713, "db": -91.76297890355133, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 5.998855615573786e-06, "end": 1.0100605983689869e-14, "n_nonzero": 709, "n_steps": 713, "db": -87.73720979207724, "peak_is_zero": false}]]` | 0–1 |
| `/cells/0/settling_rerun` | *absent* | `null` | — |
| `/cells/0/wall_time_s` | *absent* | `4.207638502120972` | — |
| `/cells/0/warnings/0/count` | *absent* | `1` | — |
| `/cells/0/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/1/column_power_max` | *absent* | `1.0000876917090205` | — |
| `/cells/1/column_power_per_bin` | *absent* | `[[1.000065354107267, 0.9999788582152932, 1.0000204619510296, 1.000001192557254, 1.0000024092508288, 1.0000062709161137, 0.9999966216531853, 1.0000057851724797, 0.9999935415638139, 1.0000033319743, 0.9999922671242167, 0.9999998143192277, 0.9999916030121909, 0.9999946208573036, 0.9999918153615858, 0.999984426407199, 1.0000067020614696], [1.0000876917090205, 0.9999641022894061, 1.0000235683472314, 0.9999965524160722, 0.9999989735931982, 1.0000028482222456, 0.9999916526351244, 1.0000033722301769, 0.9999886470565437, 1.0000003026231672, 0.9999901592071618, 0.9999974752462681, 0.9999924655659242, 0.9999923055984039, 0.9999951591336541, 0.9999832541996198, 1.0000208674967157]]` | 0–1 |
| `/cells/1/cpml_layers` | *absent* | `17` | — |
| `/cells/1/dt_s` | *absent* | `4.842700168608755e-12` | — |
| `/cells/1/dut` | *absent* | `"pec_short"` | — |
| `/cells/1/dut_cells` | *absent* | `72` | — |
| `/cells/1/dut_runs_xyz` | *absent* | `[2, 9, 4]` | 0–2 |
| `/cells/1/dx_m` | *absent* | `0.00254` | — |
| `/cells/1/fc_discrete_guide_hz` | *absent* | `6523900723.790886` | — |
| `/cells/1/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/1/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/1/grid_shape` | *absent* | `[83, 10, 5]` | 0–2 |
| `/cells/1/guide_cells_yz` | *absent* | `[9, 4]` | 0–1 |
| `/cells/1/lane` | *absent* | `"flux"` | — |
| `/cells/1/n_steps` | *absent* | `713` | — |
| `/cells/1/non_vacuity_max_s11` | *absent* | `1.000032676519756` | — |
| `/cells/1/num_periods` | *absent* | `40.0` | — |
| `/cells/1/port_f_cutoff_hz` | *absent* | `[6523900723.7908745, 6523900723.7908745]` | 0–1 |
| `/cells/1/power_closure_max` | *absent* | `8.769170902045431e-05` | — |
| `/cells/1/preflight` | *absent* | `[]` | — |
| `/cells/1/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/cells/1/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/cells/1/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/cells/1/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/1/rung` | *absent* | `"coarse"` | — |
| `/cells/1/s_params/S11` | *absent* | array[17], SHA256 `45bf702103b7` | 0–16 |
| `/cells/1/s_params/S12` | *absent* | array[17], SHA256 `b99231a2753b` | 0–16 |
| `/cells/1/s_params/S21` | *absent* | array[17], SHA256 `b99231a2753b` | 0–16 |
| `/cells/1/s_params/S22` | *absent* | array[17], SHA256 `9977f69b2fc3` | 0–16 |
| `/cells/1/settling_db/left` | *absent* | `-84.89087147965036` | — |
| `/cells/1/settling_db/right` | *absent* | `-81.82784975937672` | — |
| `/cells/1/settling_db_over_normal_records` | *absent* | `-81.82784975937672` | — |
| `/cells/1/settling_degenerate_records` | *absent* | array[8], SHA256 `5a8a585c6b9d` | 0–7 |
| `/cells/1/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.2577648240185795, "end": 7.167565258694274e-10, "n_nonzero": 702, "n_steps": 713, "db": -85.55851995120653, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.668860185716543, "end": 8.482791169575122e-10, "n_nonzero": 709, "n_steps": 713, "db": -88.9679656947883, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 1.0823295082812922e-05, "end": 9.82804465375299e-16, "n_nonzero": 702, "n_steps": 713, "db": -100.4189237804148, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 6.425294725427447e-06, "end": 3.673730091074864e-15, "n_nonzero": 709, "n_steps": 713, "db": -92.42785807826857, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 8.93299502404371e-40, "end": 2.8192270727888778e-49, "n_nonzero": 684, "n_steps": 713, "db": -95.00867034829751, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 8.715224501676256e-40, "end": 4.1065383157994424e-49, "n_nonzero": 677, "n_steps": 713, "db": -93.26802699551243, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.1764350322884696e-45, "end": 4.941674443190604e-55, "n_nonzero": 684, "n_steps": 713, "db": -98.08065846375567, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.2671324639996316e-45, "end": 9.855744740854169e-55, "n_nonzero": 676, "n_steps": 713, "db": -95.20477296107936, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9400219121162081, "end": 1.574705427662944e-10, "n_nonzero": 702, "n_steps": 713, "db": -97.75938652809123, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9126197906760503, "end": 1.3338837049163657e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.35171915334394, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.344678748221921e-06, "end": 1.317588554617071e-16, "n_nonzero": 702, "n_steps": 713, "db": -104.04574597091673, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.4152605662281064e-06, "end": 7.493914466072573e-17, "n_nonzero": 709, "n_steps": 713, "db": -106.58715111799884, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9169043789893863, "end": 1.5905884254842815e-09, "n_nonzero": 684, "n_steps": 713, "db": -87.60766229034988, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9144998107090352, "end": 2.9654900551484287e-09, "n_nonzero": 677, "n_steps": 713, "db": -84.89087147965036, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.3944336881157355e-06, "end": 3.0907051577765066e-15, "n_nonzero": 684, "n_steps": 713, "db": -90.40709752044778, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.337795644381731e-06, "end": 8.649387559171213e-15, "n_nonzero": 677, "n_steps": 713, "db": -85.86474386316092, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 9.164508701810065e-40, "end": 1.873820946253308e-49, "n_nonzero": 684, "n_steps": 713, "db": -96.89381098579196, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 9.305938165145415e-40, "end": 4.175748835802808e-49, "n_nonzero": 677, "n_steps": 713, "db": -93.48025793760263, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.3718765864157466e-45, "end": 2.589176481154959e-54, "n_nonzero": 684, "n_steps": 713, "db": -91.14710017252145, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.342230946809503e-45, "end": 1.1703822425444902e-54, "n_nonzero": 676, "n_steps": 713, "db": -94.55708732270715, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.6992021284552266, "end": 1.7198631625551836e-09, "n_nonzero": 702, "n_steps": 713, "db": -86.09108847155625, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.7051307975936005, "end": 4.62897363908177e-09, "n_nonzero": 709, "n_steps": 713, "db": -81.82784975937672, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 8.221049155890609e-06, "end": 5.478092416617927e-15, "n_nonzero": 702, "n_steps": 713, "db": -91.76297890355133, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 5.998855615573786e-06, "end": 1.0100605983689869e-14, "n_nonzero": 709, "n_steps": 713, "db": -87.73720979207724, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9714454174723528, "end": 1.4750886192055612e-09, "n_nonzero": 684, "n_steps": 713, "db": -88.18600291653425, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9523866158439809, "end": 2.79419012295999e-09, "n_nonzero": 677, "n_steps": 713, "db": -85.32557330359965, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.4862024442234267e-06, "end": 3.995148781550744e-15, "n_nonzero": 684, "n_steps": 713, "db": -89.40819645724636, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.470284873978999e-06, "end": 1.0150291162093852e-14, "n_nonzero": 677, "n_steps": 713, "db": -85.33886627067159, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9693575972582416, "end": 9.735966978419422e-11, "n_nonzero": 702, "n_steps": 713, "db": -99.98104926176009, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9670672115243271, "end": 1.4141394761706615e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.3496441284352, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.5456732884719527e-06, "end": 1.8119224989871332e-16, "n_nonzero": 702, "n_steps": 713, "db": -102.91559097855527, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.508374309928316e-06, "end": 1.056021782366216e-16, "n_nonzero": 709, "n_steps": 713, "db": -105.2143304582434, "peak_is_zero": false}]]` | 0–3 |
| `/cells/1/settling_rerun` | *absent* | `null` | — |
| `/cells/1/wall_time_s` | *absent* | `8.472352743148804` | — |
| `/cells/1/warnings/0/count` | *absent* | `8` | — |
| `/cells/1/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/1/warnings/1/count` | *absent* | `56` | — |
| `/cells/1/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/1/warnings/2/count` | *absent* | `8` | — |
| `/cells/1/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/1/warnings/3/count` | *absent* | `16` | — |
| `/cells/1/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/1/warnings/4/count` | *absent* | `1` | — |
| `/cells/1/warnings/4/message` | *absent* | `"UserWarning: flux_spectrum returned exactly 0.0 at all 17 frequencies, but the DFT accumulators are healthy and a float64 recompute of the same sum gives nonzero flux (peak &#124;flux&#124; = 1.189e-61). The per-cell E x H* products underflowed the float32 minimum normal (~1.18e-38) and were flushed to zero (issue #304). Remedies: enable x64 in a scoped context for the flux computation, increase the source amplitude, or recompute from the (healthy) accumulators in float64 as done for this check."` | — |
| `/cells/1/warnings/5/count` | *absent* | `1` | — |
| `/cells/1/warnings/5/message` | *absent* | `"UserWarning: flux_spectrum returned exactly 0.0 at all 17 frequencies, but the DFT accumulators are healthy and a float64 recompute of the same sum gives nonzero flux (peak &#124;flux&#124; = 1.237e-61). The per-cell E x H* products underflowed the float32 minimum normal (~1.18e-38) and were flushed to zero (issue #304). Remedies: enable x64 in a scoped context for the flux computation, increase the source amplitude, or recompute from the (healthy) accumulators in float64 as done for this check."` | — |
| `/cells/2/column_power_max` | *absent* | `1.0002866706244224` | — |
| `/cells/2/column_power_per_bin` | *absent* | `[[0.9997534295070132, 0.9995900874711233, 0.9994714303494652, 0.9994298339177093, 0.999467236870295, 0.9995773933792409, 0.9997398277276037, 0.9999174330090631, 1.0000901549262207, 1.0002157006393877, 1.0002866706244224, 1.000283699144422, 1.0002187617250347, 1.000101871620721, 0.9999585614582827, 0.9998266332122796, 0.9997386432463669], [0.999717837892902, 0.9995696299012342, 0.999470315710163, 0.9994489972343171, 0.9995094872125776, 0.9996315537514086, 0.9997990258045665, 0.9999690571926756, 1.0001210384991808, 1.0002186681953253, 1.0002538566554124, 1.000219523059901, 1.0001274068181807, 1.0000011805074058, 0.9998660622943945, 0.9997630116124866, 0.9997281552999077]]` | 0–1 |
| `/cells/2/cpml_layers` | *absent* | `68` | — |
| `/cells/2/dt_s` | *absent* | `1.2106750421521888e-12` | — |
| `/cells/2/dut` | *absent* | `"pec_short"` | — |
| `/cells/2/dut_cells` | *absent* | `4608` | — |
| `/cells/2/dut_runs_xyz` | *absent* | `[8, 36, 16]` | 0–2 |
| `/cells/2/dx_m` | *absent* | `0.000635` | — |
| `/cells/2/fc_discrete_guide_hz` | *absent* | `6555059929.275007` | — |
| `/cells/2/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/2/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/2/grid_shape` | *absent* | `[329, 37, 17]` | 0–2 |
| `/cells/2/guide_cells_yz` | *absent* | `[36, 16]` | 0–1 |
| `/cells/2/lane` | *absent* | `"false"` | — |
| `/cells/2/n_steps` | *absent* | `2849` | — |
| `/cells/2/non_vacuity_max_s11` | *absent* | `1.0001433250411775` | — |
| `/cells/2/num_periods` | *absent* | `40.0` | — |
| `/cells/2/port_f_cutoff_hz` | *absent* | `[6555059929.275057, 6555059929.275057]` | 0–1 |
| `/cells/2/power_closure_max` | *absent* | `0.0005701660822906574` | — |
| `/cells/2/preflight` | *absent* | `[]` | — |
| `/cells/2/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/cells/2/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/cells/2/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/cells/2/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/2/rung` | *absent* | `"fine"` | — |
| `/cells/2/s_params/S11` | *absent* | array[17], SHA256 `1997ffe060c9` | 0–16 |
| `/cells/2/s_params/S12` | *absent* | array[17], SHA256 `0b268722bb83` | 0–16 |
| `/cells/2/s_params/S21` | *absent* | array[17], SHA256 `0b268722bb83` | 0–16 |
| `/cells/2/s_params/S22` | *absent* | array[17], SHA256 `5c530fd30883` | 0–16 |
| `/cells/2/settling_db/left` | *absent* | `-102.27939898162612` | — |
| `/cells/2/settling_db/right` | *absent* | `-101.96950180120103` | — |
| `/cells/2/settling_db_over_normal_records` | *absent* | `-101.96950180120103` | — |
| `/cells/2/settling_degenerate_records` | *absent* | array[8], SHA256 `479e776c8a0e` | 0–7 |
| `/cells/2/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.2679795502314306, "end": 1.585483605857326e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -102.27939898162612, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.7077522112490584, "end": 3.020186096809856e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -103.6984753092429, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 1.2617253415839885e-05, "end": 6.209407420887833e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -123.07914669536173, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 7.4206644743593855e-06, "end": 9.467359569928123e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -118.94213923473272, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 1, "record": "v_ref_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 1, "record": "i_probe_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 1, "record": "i_ref_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 0, "record": "v_ref_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 0, "record": "i_probe_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 0, "record": "i_ref_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 1, "record": "v_probe_t", "peak": 0.3206070294274923, "end": 7.952242891081954e-12, "n_nonzero": 2808, "n_steps": 2849, "db": -106.05483403628389, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.6883820993455316, "end": 4.3740061396772056e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -101.96950180120103, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 1.2140047059690457e-05, "end": 9.633053035518998e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -121.00456418907434, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 7.509789125491892e-06, "end": 3.7658506759044617e-17, "n_nonzero": 2836, "n_steps": 2849, "db": -112.99764646897911, "peak_is_zero": false}]]` | 0–1 |
| `/cells/2/settling_rerun` | *absent* | `null` | — |
| `/cells/2/wall_time_s` | *absent* | `6.329253673553467` | — |
| `/cells/2/warnings/0/count` | *absent* | `1` | — |
| `/cells/2/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/3/column_power_max` | *absent* | `1.0000131118209492` | — |
| `/cells/3/column_power_per_bin` | *absent* | `[[0.9999979253586346, 0.9999896106688659, 0.9999925498753742, 0.9999963509621708, 0.9999957015159032, 0.9999981851731724, 0.9999986114153003, 1.0000007932952035, 1.0000005694209808, 1.0000014211816834, 1.000002879389794, 1.0000054848150983, 1.0000031539778904, 1.0000062077105605, 1.0000091086775893, 1.0000110098765873, 1.0000131118209492], [0.9999920911957928, 0.9999925056479464, 0.9999921863994301, 0.9999977090350604, 0.9999944452996637, 0.9999990105665248, 0.999998793052512, 1.0000015167719063, 1.0000018361792096, 1.0000006975774198, 1.0000053389074175, 1.0000049874527757, 1.00000645882524, 1.0000064450884814, 1.0000095991187121, 1.0000104177716092, 1.0000129535035074]]` | 0–1 |
| `/cells/3/cpml_layers` | *absent* | `68` | — |
| `/cells/3/dt_s` | *absent* | `1.2106750421521888e-12` | — |
| `/cells/3/dut` | *absent* | `"pec_short"` | — |
| `/cells/3/dut_cells` | *absent* | `4608` | — |
| `/cells/3/dut_runs_xyz` | *absent* | `[8, 36, 16]` | 0–2 |
| `/cells/3/dx_m` | *absent* | `0.000635` | — |
| `/cells/3/fc_discrete_guide_hz` | *absent* | `6555059929.275007` | — |
| `/cells/3/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/3/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/3/grid_shape` | *absent* | `[329, 37, 17]` | 0–2 |
| `/cells/3/guide_cells_yz` | *absent* | `[36, 16]` | 0–1 |
| `/cells/3/lane` | *absent* | `"flux"` | — |
| `/cells/3/n_steps` | *absent* | `2849` | — |
| `/cells/3/non_vacuity_max_s11` | *absent* | `1.0000065558889848` | — |
| `/cells/3/num_periods` | *absent* | `40.0` | — |
| `/cells/3/port_f_cutoff_hz` | *absent* | `[6555059929.275057, 6555059929.275057]` | 0–1 |
| `/cells/3/power_closure_max` | *absent* | `1.3111820949207598e-05` | — |
| `/cells/3/preflight` | *absent* | `[]` | — |
| `/cells/3/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/cells/3/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/cells/3/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/cells/3/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/3/rung` | *absent* | `"fine"` | — |
| `/cells/3/s_params/S11` | *absent* | array[17], SHA256 `4e1367260634` | 0–16 |
| `/cells/3/s_params/S12` | *absent* | array[17], SHA256 `6616093ac548` | 0–16 |
| `/cells/3/s_params/S21` | *absent* | array[17], SHA256 `6616093ac548` | 0–16 |
| `/cells/3/s_params/S22` | *absent* | array[17], SHA256 `9e5ab556a853` | 0–16 |
| `/cells/3/settling_db/left` | *absent* | `-100.89879388166203` | — |
| `/cells/3/settling_db/right` | *absent* | `-101.1073500105272` | — |
| `/cells/3/settling_db_over_normal_records` | *absent* | `-100.89879388166203` | — |
| `/cells/3/settling_degenerate_records` | *absent* | array[8], SHA256 `5a8a585c6b9d` | 0–7 |
| `/cells/3/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.2679795502314306, "end": 1.585483605857326e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -102.27939898162612, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.7077522112490584, "end": 3.020186096809856e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -103.6984753092429, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 1.2617253415839885e-05, "end": 6.209407420887833e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -123.07914669536173, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 7.4206644743593855e-06, "end": 9.467359569928123e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -118.94213923473272, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 1, "record": "v_ref_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 1, "record": "i_probe_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 1, "record": "i_ref_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9952865483487727, "end": 1.1326987547930805e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.43833711385913, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9901537090407864, "end": 4.063429080539113e-12, "n_nonzero": 2836, "n_steps": 2849, "db": -113.86809934277436, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.894456923516818e-06, "end": 3.47745836310291e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -120.49184965756021, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.87815157222674e-06, "end": 4.993003318337167e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -118.90262924191421, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9876574828956706, "end": 7.026260629168506e-11, "n_nonzero": 2724, "n_steps": 2849, "db": -101.4788210318288, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9710577227363046, "end": 7.895245855392312e-11, "n_nonzero": 2684, "n_steps": 2849, "db": -100.89879388166203, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.863768512605984e-06, "end": 1.5251477431339775e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -114.0369918250916, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.819790315780442e-06, "end": 1.875604550381926e-17, "n_nonzero": 2681, "n_steps": 2849, "db": -113.08898245626067, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 0, "record": "v_ref_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 0, "record": "i_probe_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 0, "record": "i_ref_t", "peak": 0.0, "end": 0.0, "n_nonzero": 0, "n_steps": 2849, "db": null, "peak_is_zero": true}, {"port_index": 1, "record": "v_probe_t", "peak": 0.3206070294274923, "end": 7.952242891081954e-12, "n_nonzero": 2808, "n_steps": 2849, "db": -106.05483403628389, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.6883820993455316, "end": 4.3740061396772056e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -101.96950180120103, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 1.2140047059690457e-05, "end": 9.633053035518998e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -121.00456418907434, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 7.509789125491892e-06, "end": 3.7658506759044617e-17, "n_nonzero": 2836, "n_steps": 2849, "db": -112.99764646897911, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9896803494050346, "end": 5.6752706840378657e-11, "n_nonzero": 2723, "n_steps": 2849, "db": -102.41508367030224, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9751309640854657, "end": 7.55662631100277e-11, "n_nonzero": 2683, "n_steps": 2849, "db": -101.1073500105272, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.877838872034557e-06, "end": 2.0105031689624555e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -112.852849977847, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.819990540764604e-06, "end": 1.9712635296494987e-17, "n_nonzero": 2680, "n_steps": 2849, "db": -112.87317600391744, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9966298174796897, "end": 1.165745793462025e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.31930019387812, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9908542896482686, "end": 1.7895967871154716e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -107.43254602490282, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.907805115333819e-06, "end": 3.14251345486049e-19, "n_nonzero": 2808, "n_steps": 2849, "db": -130.94655751231517, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.889752383737103e-06, "end": 3.0493122322745653e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -121.05720059685383, "peak_is_zero": false}]]` | 0–3 |
| `/cells/3/settling_rerun` | *absent* | `null` | — |
| `/cells/3/wall_time_s` | *absent* | `12.627668380737305` | — |
| `/cells/3/warnings/0/count` | *absent* | `8` | — |
| `/cells/3/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/3/warnings/1/count` | *absent* | `56` | — |
| `/cells/3/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/3/warnings/2/count` | *absent* | `8` | — |
| `/cells/3/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/3/warnings/3/count` | *absent* | `16` | — |
| `/cells/3/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/4/column_power_max` | *absent* | `1.0011618775172906` | — |
| `/cells/4/column_power_per_bin` | *absent* | `[[0.9990884773335759, 0.998510762547358, 0.9980996273366841, 0.9979658002867496, 0.998117933250459, 0.998543462388135, 0.999148859884034, 0.9998222985662615, 1.000453780073678, 1.0009190971023734, 1.0011618775172906, 1.001134342265292, 1.0008694742323385, 1.0004236615359539, 0.9998952175910009, 0.9994169118241454, 0.9991116064561608], [0.998864140577837, 0.9983800227954484, 0.9981047611511862, 0.9981247497134476, 0.9984176178988412, 0.9989254442118289, 0.9995425795968063, 1.0001385715763789, 1.0006070698158902, 1.000850228820656, 1.000843302721428, 1.0005969648270827, 1.0001868084341985, 0.9997182996676911, 0.999313438195859, 0.9991085419258047, 0.9992215960605008]]` | 0–1 |
| `/cells/4/cpml_layers` | *absent* | `34` | — |
| `/cells/4/dt_s` | *absent* | `2.4213500843043776e-12` | — |
| `/cells/4/dut` | *absent* | `"pec_short"` | — |
| `/cells/4/dut_cells` | *absent* | `576` | — |
| `/cells/4/dut_runs_xyz` | *absent* | `[4, 18, 8]` | 0–2 |
| `/cells/4/dx_m` | *absent* | `0.00127` | — |
| `/cells/4/fc_discrete_guide_hz` | *absent* | `6548820964.704695` | — |
| `/cells/4/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/4/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/4/grid_shape` | *absent* | `[165, 19, 9]` | 0–2 |
| `/cells/4/guide_cells_yz` | *absent* | `[18, 8]` | 0–1 |
| `/cells/4/lane` | *absent* | `"false"` | — |
| `/cells/4/n_steps` | *absent* | `1425` | — |
| `/cells/4/non_vacuity_max_s11` | *absent* | `1.000580770111684` | — |
| `/cells/4/num_periods` | *absent* | `40.0` | — |
| `/cells/4/port_f_cutoff_hz` | *absent* | `[6548820964.704762, 6548820964.704762]` | 0–1 |
| `/cells/4/power_closure_max` | *absent* | `0.002034199713250362` | — |
| `/cells/4/preflight/0/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/4/preflight/0/message` | *absent* | `"PEC 'pec_like' x-extent 5.08mm = 4.0 cells — volume under-resolved (PEC volume needs ≥5 cells; thin sheets <3 cells are fine)."` | — |
| `/cells/4/preflight/0/severity` | *absent* | `"warning"` | — |
| `/cells/4/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/cells/4/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/cells/4/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/cells/4/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/4/rung` | *absent* | `"mid"` | — |
| `/cells/4/s_params/S11` | *absent* | array[17], SHA256 `46c9a8a3a8f0` | 0–16 |
| `/cells/4/s_params/S12` | *absent* | array[17], SHA256 `b55b6608d771` | 0–16 |
| `/cells/4/s_params/S21` | *absent* | array[17], SHA256 `bdd4118ee0b3` | 0–16 |
| `/cells/4/s_params/S22` | *absent* | array[17], SHA256 `3ceea8782b9d` | 0–16 |
| `/cells/4/settling_db/left` | *absent* | `-96.79995087291942` | — |
| `/cells/4/settling_db/right` | *absent* | `-94.48311984474861` | — |
| `/cells/4/settling_db_over_normal_records` | *absent* | `-94.48311984474861` | — |
| `/cells/4/settling_degenerate_records` | *absent* | array[8], SHA256 `479e776c8a0e` | 0–7 |
| `/cells/4/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.2654292783775567, "end": 5.5456663751219106e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -96.79995087291942, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.6999361709837473, "end": 3.477169820748534e-11, "n_nonzero": 1418, "n_steps": 1425, "db": -103.03832535750499, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 1.2248736052782603e-05, "end": 2.963995848514074e-17, "n_nonzero": 1404, "n_steps": 1425, "db": -116.16213685141659, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 7.223331295824753e-06, "end": 1.346641703121042e-16, "n_nonzero": 1418, "n_steps": 1425, "db": -107.29485474422164, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 2.37458794269478e-74, "end": 1.2133686895835078e-83, "n_nonzero": 1323, "n_steps": 1425, "db": -92.91595474296123, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 2.364437390046544e-74, "end": 4.434071549319804e-84, "n_nonzero": 1295, "n_steps": 1425, "db": -97.26925121727304, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 8.979231811562552e-80, "end": 1.9709387983717025e-87, "n_nonzero": 1181, "n_steps": 1425, "db": -76.58566044862093, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 8.981247464917575e-80, "end": 0.0, "n_nonzero": 820, "n_steps": 1425, "db": null, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 2.3989412952222705e-74, "end": 1.5510925865680293e-81, "n_nonzero": 1323, "n_steps": 1425, "db": -71.89381898282568, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 2.4001099827020213e-74, "end": 1.7046963849098635e-83, "n_nonzero": 1296, "n_steps": 1425, "db": -91.4858410312776, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 9.029691008746177e-80, "end": 3.963694216554209e-84, "n_nonzero": 1206, "n_steps": 1425, "db": -43.57572746248684, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 9.031712317376872e-80, "end": 2.732763727710886e-86, "n_nonzero": 1107, "n_steps": 1425, "db": -65.19168011165657, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.40010555398967895, "end": 2.634207697954762e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -101.81524565438906, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.6787353631827955, "end": 2.417622521637839e-10, "n_nonzero": 1418, "n_steps": 1425, "db": -94.48311984474861, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 1.1133229138940437e-05, "end": 1.2588763159488609e-16, "n_nonzero": 1404, "n_steps": 1425, "db": -109.46638084647657, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 7.2452244851174685e-06, "end": 4.7038506389891055e-16, "n_nonzero": 1418, "n_steps": 1425, "db": -101.87598322973435, "peak_is_zero": false}]]` | 0–1 |
| `/cells/4/settling_rerun` | *absent* | `null` | — |
| `/cells/4/wall_time_s` | *absent* | `5.973793268203735` | — |
| `/cells/4/warnings/0/count` | *absent* | `1` | — |
| `/cells/4/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/5/column_power_max` | *absent* | `1.0000204614627888` | — |
| `/cells/5/column_power_per_bin` | *absent* | `[[1.0000204614627888, 0.9999942575222545, 1.0000026408122848, 1.0000035729905972, 0.9999995818146947, 1.0000025535873789, 0.9999992981772615, 1.0000020139786512, 0.9999976919286342, 1.000000291427622, 0.9999976074279076, 0.9999980777895026, 0.999997810067481, 0.9999976615250532, 0.999999491035087, 0.9999943627166274, 1.000001148795051], [1.0000083747842228, 1.0000031001697975, 0.9999986369760246, 1.0000048782431212, 0.9999985789739484, 1.0000021939038188, 0.9999985002492765, 1.0000004239738942, 0.999998022091173, 0.9999980261305909, 0.9999977499259849, 0.9999970705819692, 0.9999977284704761, 0.9999974907230902, 1.0000017415439297, 0.9999963563302083, 0.9999987741772502]]` | 0–1 |
| `/cells/5/cpml_layers` | *absent* | `34` | — |
| `/cells/5/dt_s` | *absent* | `2.4213500843043776e-12` | — |
| `/cells/5/dut` | *absent* | `"pec_short"` | — |
| `/cells/5/dut_cells` | *absent* | `576` | — |
| `/cells/5/dut_runs_xyz` | *absent* | `[4, 18, 8]` | 0–2 |
| `/cells/5/dx_m` | *absent* | `0.00127` | — |
| `/cells/5/fc_discrete_guide_hz` | *absent* | `6548820964.704695` | — |
| `/cells/5/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/5/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/5/grid_shape` | *absent* | `[165, 19, 9]` | 0–2 |
| `/cells/5/guide_cells_yz` | *absent* | `[18, 8]` | 0–1 |
| `/cells/5/lane` | *absent* | `"flux"` | — |
| `/cells/5/n_steps` | *absent* | `1425` | — |
| `/cells/5/non_vacuity_max_s11` | *absent* | `1.000010230679061` | — |
| `/cells/5/num_periods` | *absent* | `40.0` | — |
| `/cells/5/port_f_cutoff_hz` | *absent* | `[6548820964.704762, 6548820964.704762]` | 0–1 |
| `/cells/5/power_closure_max` | *absent* | `2.0461462788778917e-05` | — |
| `/cells/5/preflight/0/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/5/preflight/0/message` | *absent* | `"PEC 'pec_like' x-extent 5.08mm = 4.0 cells — volume under-resolved (PEC volume needs ≥5 cells; thin sheets <3 cells are fine)."` | — |
| `/cells/5/preflight/0/severity` | *absent* | `"warning"` | — |
| `/cells/5/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/cells/5/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/cells/5/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/cells/5/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/5/rung` | *absent* | `"mid"` | — |
| `/cells/5/s_params/S11` | *absent* | array[17], SHA256 `1241b319b12c` | 0–16 |
| `/cells/5/s_params/S12` | *absent* | array[17], SHA256 `6616093ac548` | 0–16 |
| `/cells/5/s_params/S21` | *absent* | array[17], SHA256 `6616093ac548` | 0–16 |
| `/cells/5/s_params/S22` | *absent* | array[17], SHA256 `0c2434e97873` | 0–16 |
| `/cells/5/settling_db/left` | *absent* | `-96.79995087291942` | — |
| `/cells/5/settling_db/right` | *absent* | `-94.48311984474861` | — |
| `/cells/5/settling_db_over_normal_records` | *absent* | `-94.48311984474861` | — |
| `/cells/5/settling_degenerate_records` | *absent* | array[8], SHA256 `5a8a585c6b9d` | 0–7 |
| `/cells/5/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.2654292783775567, "end": 5.5456663751219106e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -96.79995087291942, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.6999361709837473, "end": 3.477169820748534e-11, "n_nonzero": 1418, "n_steps": 1425, "db": -103.03832535750499, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 1.2248736052782603e-05, "end": 2.963995848514074e-17, "n_nonzero": 1404, "n_steps": 1425, "db": -116.16213685141659, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 7.223331295824753e-06, "end": 1.346641703121042e-16, "n_nonzero": 1418, "n_steps": 1425, "db": -107.29485474422164, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 2.37458794269478e-74, "end": 1.2133686895835078e-83, "n_nonzero": 1323, "n_steps": 1425, "db": -92.91595474296123, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 2.364437390046544e-74, "end": 4.434071549319804e-84, "n_nonzero": 1295, "n_steps": 1425, "db": -97.26925121727304, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 8.979231811562552e-80, "end": 1.9709387983717025e-87, "n_nonzero": 1181, "n_steps": 1425, "db": -76.58566044862093, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 8.981247464917575e-80, "end": 0.0, "n_nonzero": 820, "n_steps": 1425, "db": null, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9840568725104646, "end": 1.6246217449680167e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -107.82267936929927, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9781914306959152, "end": 9.87697700895852e-12, "n_nonzero": 1418, "n_steps": 1425, "db": -109.95799811083303, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.770321407869787e-06, "end": 9.299535888939829e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.07917099201708, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.7602532100404506e-06, "end": 1.5409660321771818e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -113.8742402509415, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.973896732193694, "end": 1.653486340667483e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -97.70112297018578, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9579650783061879, "end": 1.4844254594097538e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.09791283246584, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.760061780475109e-06, "end": 1.477807940725195e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -104.05576984922492, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.7104500469976377e-06, "end": 3.722243600521151e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.98621797222226, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 2.3989412952222705e-74, "end": 1.5510925865680293e-81, "n_nonzero": 1323, "n_steps": 1425, "db": -71.89381898282568, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 2.4001099827020213e-74, "end": 1.7046963849098635e-83, "n_nonzero": 1296, "n_steps": 1425, "db": -91.4858410312776, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 9.029691008746177e-80, "end": 3.963694216554209e-84, "n_nonzero": 1206, "n_steps": 1425, "db": -43.57572746248684, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 9.031712317376872e-80, "end": 2.732763727710886e-86, "n_nonzero": 1107, "n_steps": 1425, "db": -65.19168011165657, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.40010555398967895, "end": 2.634207697954762e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -101.81524565438906, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.6787353631827955, "end": 2.417622521637839e-10, "n_nonzero": 1418, "n_steps": 1425, "db": -94.48311984474861, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 1.1133229138940437e-05, "end": 1.2588763159488609e-16, "n_nonzero": 1404, "n_steps": 1425, "db": -109.46638084647657, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 7.2452244851174685e-06, "end": 4.7038506389891055e-16, "n_nonzero": 1418, "n_steps": 1425, "db": -101.87598322973435, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9842692702859637, "end": 1.2781294628933416e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -98.86539080145886, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9710144936946108, "end": 1.436051083479876e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.30055823398186, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.7995909916816676e-06, "end": 1.735459768409189e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -103.40322299080681, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.734578927705405e-06, "end": 4.008070987695212e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.6930623758277, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9905771124758758, "end": 1.1545021515743483e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -109.3349354266141, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9832422629269786, "end": 3.948578715644504e-11, "n_nonzero": 1418, "n_steps": 1425, "db": -103.9621973748268, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.826783093803738e-06, "end": 8.897830616173685e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.33549712611666, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.8117678910391204e-06, "end": 1.3470216057300002e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -114.51751885691651, "peak_is_zero": false}]]` | 0–3 |
| `/cells/5/settling_rerun` | *absent* | `null` | — |
| `/cells/5/wall_time_s` | *absent* | `11.83987045288086` | — |
| `/cells/5/warnings/0/count` | *absent* | `8` | — |
| `/cells/5/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/5/warnings/1/count` | *absent* | `56` | — |
| `/cells/5/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/5/warnings/2/count` | *absent* | `8` | — |
| `/cells/5/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/5/warnings/3/count` | *absent* | `16` | — |
| `/cells/5/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/6/column_power_max` | *absent* | `1.0071137369445617` | — |
| `/cells/6/column_power_per_bin` | *absent* | `[[0.999405679471109, 0.9997970141332199, 0.9999494921849996, 0.9989400306734166, 0.9979060269617097, 0.99674776399677, 0.9961418060212291, 0.9963749925679795, 0.9973785722953161, 0.999101971984887, 1.001216408642005, 1.0034500416589862, 1.0054431884041373, 1.006667284527735, 1.0071137369445617, 1.0064512149214777, 1.0053506239290448], [0.9986640911063761, 0.9982402160324765, 0.9981890259195542, 0.997155196955946, 0.99661219126743, 0.996383154168621, 0.996887964105583, 0.9980705760669812, 0.9996157609437475, 1.001435396652333, 1.0029681549753526, 1.003945402601182, 1.0040994028973387, 1.0034215554468364, 1.0023994491288337, 1.0011634184210947, 1.00058625221404]]` | 0–1 |
| `/cells/6/cpml_layers` | *absent* | `17` | — |
| `/cells/6/dt_s` | *absent* | `4.842700168608755e-12` | — |
| `/cells/6/dut` | *absent* | `"slab"` | — |
| `/cells/6/dut_cells` | *absent* | `144` | — |
| `/cells/6/dut_runs_xyz` | *absent* | `[4, 9, 4]` | 0–2 |
| `/cells/6/dx_m` | *absent* | `0.00254` | — |
| `/cells/6/fc_discrete_guide_hz` | *absent* | `6523900723.790886` | — |
| `/cells/6/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/6/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/6/grid_shape` | *absent* | `[83, 10, 5]` | 0–2 |
| `/cells/6/guide_cells_yz` | *absent* | `[9, 4]` | 0–1 |
| `/cells/6/lane` | *absent* | `"false"` | — |
| `/cells/6/n_steps` | *absent* | `713` | — |
| `/cells/6/non_vacuity_max_s11` | *absent* | `0.7681297878421356` | — |
| `/cells/6/num_periods` | *absent* | `40.0` | — |
| `/cells/6/port_f_cutoff_hz` | *absent* | `[6523900723.7908745, 6523900723.7908745]` | 0–1 |
| `/cells/6/power_closure_max` | *absent* | `0.007113736944561744` | — |
| `/cells/6/preflight/0/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/6/preflight/0/message` | *absent* | `"dielectric 'diel' on x: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/6/preflight/0/severity` | *absent* | `"warning"` | — |
| `/cells/6/preflight/1/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/6/preflight/1/message` | *absent* | `"dielectric 'diel' on y: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/6/preflight/1/severity` | *absent* | `"warning"` | — |
| `/cells/6/preflight/2/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/6/preflight/2/message` | *absent* | `"dielectric 'diel' on z: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/6/preflight/2/severity` | *absent* | `"warning"` | — |
| `/cells/6/preflight/3/code` | *absent* | `"lossless_q"` | — |
| `/cells/6/preflight/3/message` | *absent* | `"all dielectric(s) ['diel'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)"` | — |
| `/cells/6/preflight/3/severity` | *absent* | `"warning"` | — |
| `/cells/6/reciprocity_complex_max` | *absent* | `0.030861195594530717` | — |
| `/cells/6/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `ba5426dce1af` | 0–16 |
| `/cells/6/reciprocity_mag_mean` | *absent* | `0.013225390084792519` | — |
| `/cells/6/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/6/rung` | *absent* | `"coarse"` | — |
| `/cells/6/s_params/S11` | *absent* | array[17], SHA256 `70e2557d502c` | 0–16 |
| `/cells/6/s_params/S12` | *absent* | array[17], SHA256 `20d10ca48798` | 0–16 |
| `/cells/6/s_params/S21` | *absent* | array[17], SHA256 `1870d3d14bf0` | 0–16 |
| `/cells/6/s_params/S22` | *absent* | array[17], SHA256 `90c69c86a82b` | 0–16 |
| `/cells/6/settling_db/left` | *absent* | `-81.76875582425814` | — |
| `/cells/6/settling_db/right` | *absent* | `-79.58523748320167` | — |
| `/cells/6/settling_db_over_normal_records` | *absent* | `-79.58523748320167` | — |
| `/cells/6/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/6/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.8790966971557737, "end": 1.2815811338525837e-09, "n_nonzero": 702, "n_steps": 713, "db": -88.36290542833596, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 1.0156869962813744, "end": 1.785685855489823e-09, "n_nonzero": 709, "n_steps": 713, "db": -87.54954833698915, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 5.242268031076041e-06, "end": 5.9801885312804444e-15, "n_nonzero": 702, "n_steps": 713, "db": -89.42804346389055, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.3188115156244317e-06, "end": 6.792953253288299e-15, "n_nonzero": 709, "n_steps": 713, "db": -86.88923962399156, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.46070695457844124, "end": 3.065837879292888e-09, "n_nonzero": 684, "n_steps": 713, "db": -81.76875582425814, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.45666555559984445, "end": 2.1702880156825124e-09, "n_nonzero": 677, "n_steps": 713, "db": -83.2308088355451, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 1.6403096100978316e-06, "end": 4.8062426550382764e-15, "n_nonzero": 684, "n_steps": 713, "db": -85.33120135856036, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 1.6733493400306401e-06, "end": 7.530663764426457e-15, "n_nonzero": 677, "n_steps": 713, "db": -83.46753359480795, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.4751438584563097, "end": 1.9777041720689314e-09, "n_nonzero": 684, "n_steps": 713, "db": -83.80663790393263, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.4782379627477127, "end": 2.893140072917709e-09, "n_nonzero": 677, "n_steps": 713, "db": -82.182745868866, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 1.7141895597400878e-06, "end": 8.254411260656995e-15, "n_nonzero": 684, "n_steps": 713, "db": -83.17372742713239, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 1.7263212608034043e-06, "end": 6.082664789730401e-15, "n_nonzero": 677, "n_steps": 713, "db": -84.53027735657523, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.47762992202569876, "end": 5.2549401954856135e-09, "n_nonzero": 702, "n_steps": 713, "db": -79.58523748320167, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.821763073482348, "end": 4.182500172916848e-09, "n_nonzero": 709, "n_steps": 713, "db": -82.93310654438426, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 7.359752157241406e-06, "end": 3.773073111588347e-15, "n_nonzero": 702, "n_steps": 713, "db": -92.90167968840751, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 4.357726015206259e-06, "end": 1.339283064711491e-14, "n_nonzero": 709, "n_steps": 713, "db": -85.12387544024838, "peak_is_zero": false}]]` | 0–1 |
| `/cells/6/settling_rerun` | *absent* | `null` | — |
| `/cells/6/wall_time_s` | *absent* | `4.128620147705078` | — |
| `/cells/6/warnings/0/count` | *absent* | `1` | — |
| `/cells/6/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/6/warnings/1/count` | *absent* | `1` | — |
| `/cells/6/warnings/1/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix: reciprocity ADVISORY (warn-only -- nothing failed, the S-parameters are returned unchanged): worst complex reciprocity deviation max&#124;S_ij - S_ji&#124; / max&#124;S&#124; = 0.03086 exceeds 0.011, at ports (left, right), frequency index 5 (9.4000 GHz). The tolerance is derived from the WR-90 chain-battery envelope -- see WAVEGUIDE_RECIPROCITY_ADVISORY_TOL in rfx/api/_sparams.py for the cells it comes from. rfx models eps, sigma and mu_r as isotropic scalars, and a structure built only from those is reciprocal by Lorentz reciprocity, so on this lane a deviation this large is normally a discretization or extraction artifact: refine dx (the measured ladder runs 6.8e-2 coarse -> 7.0e-3 fine on normalize=False) or use normalize='flux', which measured ~21x tighter at the same mesh. If your structure genuinely IS non-reciprocal (magnetised ferrite, an active device), this advisory is expected -- it is informational and changes nothing."` | — |
| `/cells/7/column_power_max` | *absent* | `1.0001005294479495` | — |
| `/cells/7/column_power_per_bin` | *absent* | `[[1.000045648937981, 0.9999092553079381, 1.0000740531322723, 0.9999540192034058, 1.0000312602299992, 0.9999903785717529, 1.0000070376487187, 1.0000068641367266, 0.9999960984849292, 1.0000131161232098, 0.999991358190255, 1.0000144941054092, 0.9999888650799553, 1.0000104643489822, 0.9999902111765566, 1.000000198903486, 1.0000077666037885], [1.0000581497726597, 0.9998898285675293, 1.0001005294479495, 0.9999368524716707, 1.0000416325599575, 0.9999876744861046, 1.0000062624110324, 1.0000092839351762, 0.9999883557991347, 1.0000192391178082, 0.9999800516326669, 1.0000193677788831, 0.9999787884421514, 1.0000101872184004, 0.9999866407140977, 0.9999834148502529, 1.0000330658375534]]` | 0–1 |
| `/cells/7/cpml_layers` | *absent* | `17` | — |
| `/cells/7/dt_s` | *absent* | `4.842700168608755e-12` | — |
| `/cells/7/dut` | *absent* | `"slab"` | — |
| `/cells/7/dut_cells` | *absent* | `144` | — |
| `/cells/7/dut_runs_xyz` | *absent* | `[4, 9, 4]` | 0–2 |
| `/cells/7/dx_m` | *absent* | `0.00254` | — |
| `/cells/7/fc_discrete_guide_hz` | *absent* | `6523900723.790886` | — |
| `/cells/7/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/7/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/7/grid_shape` | *absent* | `[83, 10, 5]` | 0–2 |
| `/cells/7/guide_cells_yz` | *absent* | `[9, 4]` | 0–1 |
| `/cells/7/lane` | *absent* | `"flux"` | — |
| `/cells/7/n_steps` | *absent* | `713` | — |
| `/cells/7/non_vacuity_max_s11` | *absent* | `0.7748136768726505` | — |
| `/cells/7/num_periods` | *absent* | `40.0` | — |
| `/cells/7/port_f_cutoff_hz` | *absent* | `[6523900723.7908745, 6523900723.7908745]` | 0–1 |
| `/cells/7/power_closure_max` | *absent* | `0.00011017143247071814` | — |
| `/cells/7/preflight/0/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/7/preflight/0/message` | *absent* | `"dielectric 'diel' on x: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/7/preflight/0/severity` | *absent* | `"warning"` | — |
| `/cells/7/preflight/1/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/7/preflight/1/message` | *absent* | `"dielectric 'diel' on y: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/7/preflight/1/severity` | *absent* | `"warning"` | — |
| `/cells/7/preflight/2/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/7/preflight/2/message` | *absent* | `"dielectric 'diel' on z: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/7/preflight/2/severity` | *absent* | `"warning"` | — |
| `/cells/7/preflight/3/code` | *absent* | `"lossless_q"` | — |
| `/cells/7/preflight/3/message` | *absent* | `"all dielectric(s) ['diel'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)"` | — |
| `/cells/7/preflight/3/severity` | *absent* | `"warning"` | — |
| `/cells/7/reciprocity_complex_max` | *absent* | `0.0014405302999060377` | — |
| `/cells/7/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `346f9210ad8e` | 0–16 |
| `/cells/7/reciprocity_mag_mean` | *absent* | `1.458768989854963e-06` | — |
| `/cells/7/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/7/rung` | *absent* | `"coarse"` | — |
| `/cells/7/s_params/S11` | *absent* | array[17], SHA256 `38bad7e04729` | 0–16 |
| `/cells/7/s_params/S12` | *absent* | array[17], SHA256 `b8d3c5563c35` | 0–16 |
| `/cells/7/s_params/S21` | *absent* | array[17], SHA256 `3281369bbf24` | 0–16 |
| `/cells/7/s_params/S22` | *absent* | array[17], SHA256 `c07e49e489fe` | 0–16 |
| `/cells/7/settling_db/left` | *absent* | `-81.76875582425814` | — |
| `/cells/7/settling_db/right` | *absent* | `-79.58523748320167` | — |
| `/cells/7/settling_db_over_normal_records` | *absent* | `-79.58523748320167` | — |
| `/cells/7/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/7/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.8790966971557737, "end": 1.2815811338525837e-09, "n_nonzero": 702, "n_steps": 713, "db": -88.36290542833596, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 1.0156869962813744, "end": 1.785685855489823e-09, "n_nonzero": 709, "n_steps": 713, "db": -87.54954833698915, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 5.242268031076041e-06, "end": 5.9801885312804444e-15, "n_nonzero": 702, "n_steps": 713, "db": -89.42804346389055, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.3188115156244317e-06, "end": 6.792953253288299e-15, "n_nonzero": 709, "n_steps": 713, "db": -86.88923962399156, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.46070695457844124, "end": 3.065837879292888e-09, "n_nonzero": 684, "n_steps": 713, "db": -81.76875582425814, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.45666555559984445, "end": 2.1702880156825124e-09, "n_nonzero": 677, "n_steps": 713, "db": -83.2308088355451, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 1.6403096100978316e-06, "end": 4.8062426550382764e-15, "n_nonzero": 684, "n_steps": 713, "db": -85.33120135856036, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 1.6733493400306401e-06, "end": 7.530663764426457e-15, "n_nonzero": 677, "n_steps": 713, "db": -83.46753359480795, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9400219121162081, "end": 1.574705427662944e-10, "n_nonzero": 702, "n_steps": 713, "db": -97.75938652809123, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9126197906760503, "end": 1.3338837049163657e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.35171915334394, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.344678748221921e-06, "end": 1.317588554617071e-16, "n_nonzero": 702, "n_steps": 713, "db": -104.04574597091673, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.4152605662281064e-06, "end": 7.493914466072573e-17, "n_nonzero": 709, "n_steps": 713, "db": -106.58715111799884, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9169043789893863, "end": 1.5905884254842815e-09, "n_nonzero": 684, "n_steps": 713, "db": -87.60766229034988, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9144998107090352, "end": 2.9654900551484287e-09, "n_nonzero": 677, "n_steps": 713, "db": -84.89087147965036, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.3944336881157355e-06, "end": 3.0907051577765066e-15, "n_nonzero": 684, "n_steps": 713, "db": -90.40709752044778, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.337795644381731e-06, "end": 8.649387559171213e-15, "n_nonzero": 677, "n_steps": 713, "db": -85.86474386316092, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.4751438584563097, "end": 1.9777041720689314e-09, "n_nonzero": 684, "n_steps": 713, "db": -83.80663790393263, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.4782379627477127, "end": 2.893140072917709e-09, "n_nonzero": 677, "n_steps": 713, "db": -82.182745868866, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 1.7141895597400878e-06, "end": 8.254411260656995e-15, "n_nonzero": 684, "n_steps": 713, "db": -83.17372742713239, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 1.7263212608034043e-06, "end": 6.082664789730401e-15, "n_nonzero": 677, "n_steps": 713, "db": -84.53027735657523, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.47762992202569876, "end": 5.2549401954856135e-09, "n_nonzero": 702, "n_steps": 713, "db": -79.58523748320167, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.821763073482348, "end": 4.182500172916848e-09, "n_nonzero": 709, "n_steps": 713, "db": -82.93310654438426, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 7.359752157241406e-06, "end": 3.773073111588347e-15, "n_nonzero": 702, "n_steps": 713, "db": -92.90167968840751, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 4.357726015206259e-06, "end": 1.339283064711491e-14, "n_nonzero": 709, "n_steps": 713, "db": -85.12387544024838, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9714454174723528, "end": 1.4750886192055612e-09, "n_nonzero": 684, "n_steps": 713, "db": -88.18600291653425, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9523866158439809, "end": 2.79419012295999e-09, "n_nonzero": 677, "n_steps": 713, "db": -85.32557330359965, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.4862024442234267e-06, "end": 3.995148781550744e-15, "n_nonzero": 684, "n_steps": 713, "db": -89.40819645724636, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.470284873978999e-06, "end": 1.0150291162093852e-14, "n_nonzero": 677, "n_steps": 713, "db": -85.33886627067159, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9693575972582416, "end": 9.735966978419422e-11, "n_nonzero": 702, "n_steps": 713, "db": -99.98104926176009, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9670672115243271, "end": 1.4141394761706615e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.3496441284352, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.5456732884719527e-06, "end": 1.8119224989871332e-16, "n_nonzero": 702, "n_steps": 713, "db": -102.91559097855527, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.508374309928316e-06, "end": 1.056021782366216e-16, "n_nonzero": 709, "n_steps": 713, "db": -105.2143304582434, "peak_is_zero": false}]]` | 0–3 |
| `/cells/7/settling_rerun` | *absent* | `null` | — |
| `/cells/7/wall_time_s` | *absent* | `8.142054080963135` | — |
| `/cells/7/warnings/0/count` | *absent* | `8` | — |
| `/cells/7/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/7/warnings/1/count` | *absent* | `56` | — |
| `/cells/7/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/7/warnings/2/count` | *absent* | `8` | — |
| `/cells/7/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/7/warnings/3/count` | *absent* | `16` | — |
| `/cells/7/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/8/column_power_max` | *absent* | `1.0003528195646192` | — |
| `/cells/8/column_power_per_bin` | *absent* | `[[0.9998186764219097, 0.9998329225707896, 0.9998595861610522, 0.9998037812280319, 0.9997361438237871, 0.9996667672539508, 0.999629074743494, 0.999648399295318, 0.9997264647265979, 0.9998505823781558, 1.0000029753172477, 1.000146455947367, 1.000270468568908, 1.0003364227058613, 1.0003528195646192, 1.0003047901503186, 1.000228433926643], [0.9998191864268888, 0.9998123223258459, 0.9998286234972413, 0.9997714866983056, 0.999706450199529, 0.9996521836602459, 0.9996385168775721, 0.9996808125605359, 0.9997757558111833, 0.9999058400183036, 1.0000528022989466, 1.0001794833039617, 1.000272285022345, 1.000303316124361, 1.000281625652744, 1.0002118023670714, 1.0001249954071176]]` | 0–1 |
| `/cells/8/cpml_layers` | *absent* | `68` | — |
| `/cells/8/dt_s` | *absent* | `1.2106750421521888e-12` | — |
| `/cells/8/dut` | *absent* | `"slab"` | — |
| `/cells/8/dut_cells` | *absent* | `9216` | — |
| `/cells/8/dut_runs_xyz` | *absent* | `[16, 36, 16]` | 0–2 |
| `/cells/8/dx_m` | *absent* | `0.000635` | — |
| `/cells/8/fc_discrete_guide_hz` | *absent* | `6555059929.275007` | — |
| `/cells/8/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/8/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/8/grid_shape` | *absent* | `[329, 37, 17]` | 0–2 |
| `/cells/8/guide_cells_yz` | *absent* | `[36, 16]` | 0–1 |
| `/cells/8/lane` | *absent* | `"false"` | — |
| `/cells/8/n_steps` | *absent* | `2849` | — |
| `/cells/8/non_vacuity_max_s11` | *absent* | `0.6964204948388697` | — |
| `/cells/8/num_periods` | *absent* | `40.0` | — |
| `/cells/8/port_f_cutoff_hz` | *absent* | `[6555059929.275057, 6555059929.275057]` | 0–1 |
| `/cells/8/power_closure_max` | *absent* | `0.00037092525650594954` | — |
| `/cells/8/preflight/0/code` | *absent* | `"lossless_q"` | — |
| `/cells/8/preflight/0/message` | *absent* | `"all dielectric(s) ['diel'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)"` | — |
| `/cells/8/preflight/0/severity` | *absent* | `"warning"` | — |
| `/cells/8/reciprocity_complex_max` | *absent* | `0.004806417344254802` | — |
| `/cells/8/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `c3e0582463fb` | 0–16 |
| `/cells/8/reciprocity_mag_mean` | *absent* | `0.0022506870363776755` | — |
| `/cells/8/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/8/rung` | *absent* | `"fine"` | — |
| `/cells/8/s_params/S11` | *absent* | array[17], SHA256 `0d4dadac4c3c` | 0–16 |
| `/cells/8/s_params/S12` | *absent* | array[17], SHA256 `38ee8bfb7a8a` | 0–16 |
| `/cells/8/s_params/S21` | *absent* | array[17], SHA256 `7937e5051283` | 0–16 |
| `/cells/8/s_params/S22` | *absent* | array[17], SHA256 `5c6f8b5692b3` | 0–16 |
| `/cells/8/settling_db/left` | *absent* | `-102.64564907998336` | — |
| `/cells/8/settling_db/right` | *absent* | `-101.23601747187766` | — |
| `/cells/8/settling_db_over_normal_records` | *absent* | `-101.23601747187766` | — |
| `/cells/8/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/8/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.7895694809217844, "end": 2.173436371682012e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -105.60243422735873, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 1.0130604991914538, "end": 5.508970843669776e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -102.64564907998336, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 6.144104391727712e-06, "end": 2.088010966590132e-17, "n_nonzero": 2808, "n_steps": 2849, "db": -114.6872581066431, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 4.032622869077664e-06, "end": 1.0820898986725686e-17, "n_nonzero": 2836, "n_steps": 2849, "db": -115.71324265772671, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.6081972687834423, "end": 1.7623005655856395e-11, "n_nonzero": 2714, "n_steps": 2849, "db": -105.37964484932405, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.6108305425021996, "end": 1.7817954419315166e-11, "n_nonzero": 2673, "n_steps": 2849, "db": -105.35062900726697, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 2.3651201504029277e-06, "end": 2.4813450543188346e-17, "n_nonzero": 2711, "n_steps": 2849, "db": -109.79166047020607, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 2.373371156992306e-06, "end": 3.602389538702095e-17, "n_nonzero": 2669, "n_steps": 2849, "db": -108.18774987414442, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.6086130916230559, "end": 1.4023972727879608e-11, "n_nonzero": 2714, "n_steps": 2849, "db": -106.37470231527224, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.6115018216484636, "end": 1.8133701019961167e-11, "n_nonzero": 2672, "n_steps": 2849, "db": -105.27911304119246, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 2.372051350080977e-06, "end": 2.9261376474311784e-17, "n_nonzero": 2711, "n_steps": 2849, "db": -109.08829334618417, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 2.3791758446619233e-06, "end": 3.4459352239305404e-17, "n_nonzero": 2668, "n_steps": 2849, "db": -108.3911943248834, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.664618826001572, "end": 1.1524763059536577e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -107.60940633562657, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9564301087970684, "end": 7.195342847069857e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -101.23601747187766, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 6.678873669745686e-06, "end": 1.265932511811174e-17, "n_nonzero": 2808, "n_steps": 2849, "db": -117.22292675299471, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 4.287916340979232e-06, "end": 6.909922302760649e-17, "n_nonzero": 2836, "n_steps": 2849, "db": -107.92773139438248, "peak_is_zero": false}]]` | 0–1 |
| `/cells/8/settling_rerun` | *absent* | `null` | — |
| `/cells/8/wall_time_s` | *absent* | `6.252399921417236` | — |
| `/cells/8/warnings/0/count` | *absent* | `1` | — |
| `/cells/8/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/9/column_power_max` | *absent* | `1.0000092485237313` | — |
| `/cells/9/column_power_per_bin` | *absent* | `[[1.0000091812823373, 0.9999945324783336, 1.000000340981666, 0.9999967099041849, 0.9999954827880382, 0.9999985847809456, 0.9999963882460224, 0.9999954035077412, 0.9999958497914936, 1.0000002165074453, 0.9999958438661709, 0.9999998875615519, 0.9999991212535702, 1.0000011158558144, 1.0000021525659166, 1.000003828221055, 1.0000070317141354], [1.000005871113985, 0.9999960005392848, 0.999999145762539, 0.9999977836169992, 0.9999956152660835, 0.9999982517879383, 0.99999766761389, 0.9999947573295476, 0.9999946637732635, 1.0000023694090825, 0.9999962080149105, 0.9999991637861996, 1.0000010549854244, 1.0000002500931524, 1.0000051657657654, 1.0000041255607688, 1.0000092485237313]]` | 0–1 |
| `/cells/9/cpml_layers` | *absent* | `68` | — |
| `/cells/9/dt_s` | *absent* | `1.2106750421521888e-12` | — |
| `/cells/9/dut` | *absent* | `"slab"` | — |
| `/cells/9/dut_cells` | *absent* | `9216` | — |
| `/cells/9/dut_runs_xyz` | *absent* | `[16, 36, 16]` | 0–2 |
| `/cells/9/dx_m` | *absent* | `0.000635` | — |
| `/cells/9/fc_discrete_guide_hz` | *absent* | `6555059929.275007` | — |
| `/cells/9/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/9/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/9/grid_shape` | *absent* | `[329, 37, 17]` | 0–2 |
| `/cells/9/guide_cells_yz` | *absent* | `[36, 16]` | 0–1 |
| `/cells/9/lane` | *absent* | `"flux"` | — |
| `/cells/9/n_steps` | *absent* | `2849` | — |
| `/cells/9/non_vacuity_max_s11` | *absent* | `0.6965226408629608` | — |
| `/cells/9/num_periods` | *absent* | `40.0` | — |
| `/cells/9/port_f_cutoff_hz` | *absent* | `[6555059929.275057, 6555059929.275057]` | 0–1 |
| `/cells/9/power_closure_max` | *absent* | `9.248523731297809e-06` | — |
| `/cells/9/preflight/0/code` | *absent* | `"lossless_q"` | — |
| `/cells/9/preflight/0/message` | *absent* | `"all dielectric(s) ['diel'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)"` | — |
| `/cells/9/preflight/0/severity` | *absent* | `"warning"` | — |
| `/cells/9/reciprocity_complex_max` | *absent* | `2.730281774495818e-05` | — |
| `/cells/9/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `4481bac61704` | 0–16 |
| `/cells/9/reciprocity_mag_mean` | *absent* | `6.022391006592541e-07` | — |
| `/cells/9/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/9/rung` | *absent* | `"fine"` | — |
| `/cells/9/s_params/S11` | *absent* | array[17], SHA256 `78f5c3f8842a` | 0–16 |
| `/cells/9/s_params/S12` | *absent* | array[17], SHA256 `e610d908e227` | 0–16 |
| `/cells/9/s_params/S21` | *absent* | array[17], SHA256 `50d40ed310a0` | 0–16 |
| `/cells/9/s_params/S22` | *absent* | array[17], SHA256 `2899c4656e0b` | 0–16 |
| `/cells/9/settling_db/left` | *absent* | `-100.89879388166203` | — |
| `/cells/9/settling_db/right` | *absent* | `-101.1073500105272` | — |
| `/cells/9/settling_db_over_normal_records` | *absent* | `-100.89879388166203` | — |
| `/cells/9/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/9/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.7895694809217844, "end": 2.173436371682012e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -105.60243422735873, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 1.0130604991914538, "end": 5.508970843669776e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -102.64564907998336, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 6.144104391727712e-06, "end": 2.088010966590132e-17, "n_nonzero": 2808, "n_steps": 2849, "db": -114.6872581066431, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 4.032622869077664e-06, "end": 1.0820898986725686e-17, "n_nonzero": 2836, "n_steps": 2849, "db": -115.71324265772671, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.6081972687834423, "end": 1.7623005655856395e-11, "n_nonzero": 2714, "n_steps": 2849, "db": -105.37964484932405, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.6108305425021996, "end": 1.7817954419315166e-11, "n_nonzero": 2673, "n_steps": 2849, "db": -105.35062900726697, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 2.3651201504029277e-06, "end": 2.4813450543188346e-17, "n_nonzero": 2711, "n_steps": 2849, "db": -109.79166047020607, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 2.373371156992306e-06, "end": 3.602389538702095e-17, "n_nonzero": 2669, "n_steps": 2849, "db": -108.18774987414442, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9952865483487727, "end": 1.1326987547930805e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.43833711385913, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9901537090407864, "end": 4.063429080539113e-12, "n_nonzero": 2836, "n_steps": 2849, "db": -113.86809934277436, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.894456923516818e-06, "end": 3.47745836310291e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -120.49184965756021, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.87815157222674e-06, "end": 4.993003318337167e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -118.90262924191421, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9876574828956706, "end": 7.026260629168506e-11, "n_nonzero": 2724, "n_steps": 2849, "db": -101.4788210318288, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9710577227363046, "end": 7.895245855392312e-11, "n_nonzero": 2684, "n_steps": 2849, "db": -100.89879388166203, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.863768512605984e-06, "end": 1.5251477431339775e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -114.0369918250916, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.819790315780442e-06, "end": 1.875604550381926e-17, "n_nonzero": 2681, "n_steps": 2849, "db": -113.08898245626067, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.6086130916230559, "end": 1.4023972727879608e-11, "n_nonzero": 2714, "n_steps": 2849, "db": -106.37470231527224, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.6115018216484636, "end": 1.8133701019961167e-11, "n_nonzero": 2672, "n_steps": 2849, "db": -105.27911304119246, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 2.372051350080977e-06, "end": 2.9261376474311784e-17, "n_nonzero": 2711, "n_steps": 2849, "db": -109.08829334618417, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 2.3791758446619233e-06, "end": 3.4459352239305404e-17, "n_nonzero": 2668, "n_steps": 2849, "db": -108.3911943248834, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.664618826001572, "end": 1.1524763059536577e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -107.60940633562657, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9564301087970684, "end": 7.195342847069857e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -101.23601747187766, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 6.678873669745686e-06, "end": 1.265932511811174e-17, "n_nonzero": 2808, "n_steps": 2849, "db": -117.22292675299471, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 4.287916340979232e-06, "end": 6.909922302760649e-17, "n_nonzero": 2836, "n_steps": 2849, "db": -107.92773139438248, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9896803494050346, "end": 5.6752706840378657e-11, "n_nonzero": 2723, "n_steps": 2849, "db": -102.41508367030224, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9751309640854657, "end": 7.55662631100277e-11, "n_nonzero": 2683, "n_steps": 2849, "db": -101.1073500105272, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.877838872034557e-06, "end": 2.0105031689624555e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -112.852849977847, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.819990540764604e-06, "end": 1.9712635296494987e-17, "n_nonzero": 2680, "n_steps": 2849, "db": -112.87317600391744, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9966298174796897, "end": 1.165745793462025e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.31930019387812, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9908542896482686, "end": 1.7895967871154716e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -107.43254602490282, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.907805115333819e-06, "end": 3.14251345486049e-19, "n_nonzero": 2808, "n_steps": 2849, "db": -130.94655751231517, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.889752383737103e-06, "end": 3.0493122322745653e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -121.05720059685383, "peak_is_zero": false}]]` | 0–3 |
| `/cells/9/settling_rerun` | *absent* | `null` | — |
| `/cells/9/wall_time_s` | *absent* | `12.46511697769165` | — |
| `/cells/9/warnings/0/count` | *absent* | `8` | — |
| `/cells/9/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/9/warnings/1/count` | *absent* | `56` | — |
| `/cells/9/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/9/warnings/2/count` | *absent* | `8` | — |
| `/cells/9/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/9/warnings/3/count` | *absent* | `16` | — |
| `/cells/9/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/10/column_power_max` | *absent* | `1.0015357753132264` | — |
| `/cells/10/column_power_per_bin` | *absent* | `[[0.9994416413828243, 0.9995388216947492, 0.9996166088810983, 0.9994057932967702, 0.9991368871603195, 0.9988603725447016, 0.9986958404097808, 0.9987574619370019, 0.9990398790668886, 0.9995119169617009, 1.000087775473979, 1.0006580444237123, 1.0011533741178806, 1.0014468359766013, 1.0015357753132264, 1.0013791193461765, 1.001077175255559], [0.9994056000553116, 0.9993556695499848, 0.9993767664196863, 0.9991540506019121, 0.9989182844898628, 0.9987667886737239, 0.9987772195448095, 0.9990049378592236, 0.9993943771505951, 0.9999009566527364, 1.0004260628724708, 1.0008498280452969, 1.0011081717358281, 1.0011248073051116, 1.000958467098621, 1.000646900121925, 1.0003195321424005]]` | 0–1 |
| `/cells/10/cpml_layers` | *absent* | `34` | — |
| `/cells/10/dt_s` | *absent* | `2.4213500843043776e-12` | — |
| `/cells/10/dut` | *absent* | `"slab"` | — |
| `/cells/10/dut_cells` | *absent* | `1152` | — |
| `/cells/10/dut_runs_xyz` | *absent* | `[8, 18, 8]` | 0–2 |
| `/cells/10/dx_m` | *absent* | `0.00127` | — |
| `/cells/10/fc_discrete_guide_hz` | *absent* | `6548820964.704695` | — |
| `/cells/10/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/10/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/10/grid_shape` | *absent* | `[165, 19, 9]` | 0–2 |
| `/cells/10/guide_cells_yz` | *absent* | `[18, 8]` | 0–1 |
| `/cells/10/lane` | *absent* | `"false"` | — |
| `/cells/10/n_steps` | *absent* | `1425` | — |
| `/cells/10/non_vacuity_max_s11` | *absent* | `0.7112169757211388` | — |
| `/cells/10/num_periods` | *absent* | `40.0` | — |
| `/cells/10/port_f_cutoff_hz` | *absent* | `[6548820964.704762, 6548820964.704762]` | 0–1 |
| `/cells/10/power_closure_max` | *absent* | `0.0015357753132263507` | — |
| `/cells/10/preflight/0/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/10/preflight/0/message` | *absent* | `"dielectric 'diel' on x: 10.2 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=1.27mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/10/preflight/0/severity` | *absent* | `"warning"` | — |
| `/cells/10/preflight/1/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/10/preflight/1/message` | *absent* | `"dielectric 'diel' on y: 10.2 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=1.27mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/10/preflight/1/severity` | *absent* | `"warning"` | — |
| `/cells/10/preflight/2/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/10/preflight/2/message` | *absent* | `"dielectric 'diel' on z: 10.2 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=1.27mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/10/preflight/2/severity` | *absent* | `"warning"` | — |
| `/cells/10/preflight/3/code` | *absent* | `"lossless_q"` | — |
| `/cells/10/preflight/3/message` | *absent* | `"all dielectric(s) ['diel'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)"` | — |
| `/cells/10/preflight/3/severity` | *absent* | `"warning"` | — |
| `/cells/10/reciprocity_complex_max` | *absent* | `0.010866879557759017` | — |
| `/cells/10/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `5055a615d4ea` | 0–16 |
| `/cells/10/reciprocity_mag_mean` | *absent* | `0.004589836602120118` | — |
| `/cells/10/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/10/rung` | *absent* | `"mid"` | — |
| `/cells/10/s_params/S11` | *absent* | array[17], SHA256 `ec641ec0f209` | 0–16 |
| `/cells/10/s_params/S12` | *absent* | array[17], SHA256 `cd42130c0795` | 0–16 |
| `/cells/10/s_params/S21` | *absent* | array[17], SHA256 `83ef0a7c658a` | 0–16 |
| `/cells/10/s_params/S22` | *absent* | array[17], SHA256 `782b42c973b9` | 0–16 |
| `/cells/10/settling_db/left` | *absent* | `-94.36620451017994` | — |
| `/cells/10/settling_db/right` | *absent* | `-93.97150511228007` | — |
| `/cells/10/settling_db_over_normal_records` | *absent* | `-93.97150511228007` | — |
| `/cells/10/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/10/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.8305678613140834, "end": 6.481881098265266e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -101.07674061947833, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 1.0320157844540176, "end": 1.2403550380733672e-10, "n_nonzero": 1418, "n_steps": 1425, "db": -99.20140324774394, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 5.847490898162194e-06, "end": 1.9424468182557024e-16, "n_nonzero": 1404, "n_steps": 1425, "db": -104.7862041728566, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.841755251033025e-06, "end": 1.4518653551234505e-16, "n_nonzero": 1418, "n_steps": 1425, "db": -104.22603351457064, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.5789113349496944, "end": 1.939918678938821e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -94.74828528274335, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.5803542108335478, "end": 2.1235998639661825e-10, "n_nonzero": 1353, "n_steps": 1425, "db": -94.36620451017994, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 2.20890920333075e-06, "end": 3.9429047538573874e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -97.48361578435166, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 2.2210372823622607e-06, "end": 6.480801438181232e-16, "n_nonzero": 1352, "n_steps": 1425, "db": -95.34927133142801, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.5828277301188791, "end": 1.5508847497030394e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -95.74960681251285, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.5868905633762758, "end": 2.3518337331724916e-10, "n_nonzero": 1353, "n_steps": 1425, "db": -93.97150511228007, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 2.2292894366346752e-06, "end": 5.10905854220078e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -96.3982557907089, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 2.2336070648068204e-06, "end": 6.138894230070854e-16, "n_nonzero": 1351, "n_steps": 1425, "db": -95.60916623822212, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.595054242840888, "end": 8.435384464128791e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -98.48451674683288, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9188463958562352, "end": 2.06898192989057e-10, "n_nonzero": 1418, "n_steps": 1425, "db": -96.47486218531802, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 6.926389709184392e-06, "end": 6.719068242531108e-17, "n_nonzero": 1404, "n_steps": 1425, "db": -110.13197871176423, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 4.360199326678554e-06, "end": 5.462138507368064e-16, "n_nonzero": 1418, "n_steps": 1425, "db": -99.0214363485086, "peak_is_zero": false}]]` | 0–1 |
| `/cells/10/settling_rerun` | *absent* | `null` | — |
| `/cells/10/wall_time_s` | *absent* | `5.846665620803833` | — |
| `/cells/10/warnings/0/count` | *absent* | `1` | — |
| `/cells/10/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/11/column_power_max` | *absent* | `1.0000368078695865` | — |
| `/cells/11/column_power_per_bin` | *absent* | `[[1.0000368078695865, 0.9999764859943777, 1.0000125711337926, 0.9999944964276885, 1.000003116203847, 1.0000020678981598, 0.9999998309844466, 1.0000043077464342, 0.9999990932065804, 1.0000044619957618, 0.9999992933832491, 1.0000035637745146, 0.9999983127640428, 1.0000001937551604, 0.9999975136186565, 0.9999961522906755, 1.0000017167003294], [1.0000346375150926, 0.9999761612375446, 1.0000116007749411, 0.9999959892306115, 1.0000007910403428, 1.0000037647459636, 0.9999975567319632, 1.0000053651717247, 0.9999981459481393, 1.0000038672107803, 0.9999991612227372, 1.0000012650890826, 0.9999999543273035, 0.9999961116228453, 1.0000007407812461, 0.9999899839415243, 1.000006810967459]]` | 0–1 |
| `/cells/11/cpml_layers` | *absent* | `34` | — |
| `/cells/11/dt_s` | *absent* | `2.4213500843043776e-12` | — |
| `/cells/11/dut` | *absent* | `"slab"` | — |
| `/cells/11/dut_cells` | *absent* | `1152` | — |
| `/cells/11/dut_runs_xyz` | *absent* | `[8, 18, 8]` | 0–2 |
| `/cells/11/dx_m` | *absent* | `0.00127` | — |
| `/cells/11/fc_discrete_guide_hz` | *absent* | `6548820964.704695` | — |
| `/cells/11/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/11/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/11/grid_shape` | *absent* | `[165, 19, 9]` | 0–2 |
| `/cells/11/guide_cells_yz` | *absent* | `[18, 8]` | 0–1 |
| `/cells/11/lane` | *absent* | `"flux"` | — |
| `/cells/11/n_steps` | *absent* | `1425` | — |
| `/cells/11/non_vacuity_max_s11` | *absent* | `0.7122135495891054` | — |
| `/cells/11/num_periods` | *absent* | `40.0` | — |
| `/cells/11/port_f_cutoff_hz` | *absent* | `[6548820964.704762, 6548820964.704762]` | 0–1 |
| `/cells/11/power_closure_max` | *absent* | `3.6807869586485964e-05` | — |
| `/cells/11/preflight/0/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/11/preflight/0/message` | *absent* | `"dielectric 'diel' on x: 10.2 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=1.27mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/11/preflight/0/severity` | *absent* | `"warning"` | — |
| `/cells/11/preflight/1/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/11/preflight/1/message` | *absent* | `"dielectric 'diel' on y: 10.2 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=1.27mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/11/preflight/1/severity` | *absent* | `"warning"` | — |
| `/cells/11/preflight/2/code` | *absent* | `"mesh_resolution"` | — |
| `/cells/11/preflight/2/message` | *absent* | `"dielectric 'diel' on z: 10.2 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=1.27mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into &#124;S&#124; magnitude error; ~5% &#124;S21&#124; deficit expected at 17 cells/λ_eff."` | — |
| `/cells/11/preflight/2/severity` | *absent* | `"warning"` | — |
| `/cells/11/preflight/3/code` | *absent* | `"lossless_q"` | — |
| `/cells/11/preflight/3/message` | *absent* | `"all dielectric(s) ['diel'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)"` | — |
| `/cells/11/preflight/3/severity` | *absent* | `"warning"` | — |
| `/cells/11/reciprocity_complex_max` | *absent* | `0.00021136326370809695` | — |
| `/cells/11/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `59d6b571a59e` | 0–16 |
| `/cells/11/reciprocity_mag_mean` | *absent* | `5.676502369878185e-07` | — |
| `/cells/11/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/11/rung` | *absent* | `"mid"` | — |
| `/cells/11/s_params/S11` | *absent* | array[17], SHA256 `5da6e130f3c2` | 0–16 |
| `/cells/11/s_params/S12` | *absent* | array[17], SHA256 `89aafe205012` | 0–16 |
| `/cells/11/s_params/S21` | *absent* | array[17], SHA256 `307e43f7ae88` | 0–16 |
| `/cells/11/s_params/S22` | *absent* | array[17], SHA256 `b64b44d8ad85` | 0–16 |
| `/cells/11/settling_db/left` | *absent* | `-94.36620451017994` | — |
| `/cells/11/settling_db/right` | *absent* | `-93.97150511228007` | — |
| `/cells/11/settling_db_over_normal_records` | *absent* | `-93.97150511228007` | — |
| `/cells/11/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/11/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.8305678613140834, "end": 6.481881098265266e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -101.07674061947833, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 1.0320157844540176, "end": 1.2403550380733672e-10, "n_nonzero": 1418, "n_steps": 1425, "db": -99.20140324774394, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 5.847490898162194e-06, "end": 1.9424468182557024e-16, "n_nonzero": 1404, "n_steps": 1425, "db": -104.7862041728566, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.841755251033025e-06, "end": 1.4518653551234505e-16, "n_nonzero": 1418, "n_steps": 1425, "db": -104.22603351457064, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.5789113349496944, "end": 1.939918678938821e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -94.74828528274335, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.5803542108335478, "end": 2.1235998639661825e-10, "n_nonzero": 1353, "n_steps": 1425, "db": -94.36620451017994, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 2.20890920333075e-06, "end": 3.9429047538573874e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -97.48361578435166, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 2.2210372823622607e-06, "end": 6.480801438181232e-16, "n_nonzero": 1352, "n_steps": 1425, "db": -95.34927133142801, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9840568725104646, "end": 1.6246217449680167e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -107.82267936929927, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9781914306959152, "end": 9.87697700895852e-12, "n_nonzero": 1418, "n_steps": 1425, "db": -109.95799811083303, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.770321407869787e-06, "end": 9.299535888939829e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.07917099201708, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.7602532100404506e-06, "end": 1.5409660321771818e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -113.8742402509415, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.973896732193694, "end": 1.653486340667483e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -97.70112297018578, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9579650783061879, "end": 1.4844254594097538e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.09791283246584, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.760061780475109e-06, "end": 1.477807940725195e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -104.05576984922492, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.7104500469976377e-06, "end": 3.722243600521151e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.98621797222226, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.5828277301188791, "end": 1.5508847497030394e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -95.74960681251285, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.5868905633762758, "end": 2.3518337331724916e-10, "n_nonzero": 1353, "n_steps": 1425, "db": -93.97150511228007, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 2.2292894366346752e-06, "end": 5.10905854220078e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -96.3982557907089, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 2.2336070648068204e-06, "end": 6.138894230070854e-16, "n_nonzero": 1351, "n_steps": 1425, "db": -95.60916623822212, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.595054242840888, "end": 8.435384464128791e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -98.48451674683288, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9188463958562352, "end": 2.06898192989057e-10, "n_nonzero": 1418, "n_steps": 1425, "db": -96.47486218531802, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 6.926389709184392e-06, "end": 6.719068242531108e-17, "n_nonzero": 1404, "n_steps": 1425, "db": -110.13197871176423, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 4.360199326678554e-06, "end": 5.462138507368064e-16, "n_nonzero": 1418, "n_steps": 1425, "db": -99.0214363485086, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9842692702859637, "end": 1.2781294628933416e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -98.86539080145886, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9710144936946108, "end": 1.436051083479876e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.30055823398186, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.7995909916816676e-06, "end": 1.735459768409189e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -103.40322299080681, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.734578927705405e-06, "end": 4.008070987695212e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.6930623758277, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9905771124758758, "end": 1.1545021515743483e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -109.3349354266141, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9832422629269786, "end": 3.948578715644504e-11, "n_nonzero": 1418, "n_steps": 1425, "db": -103.9621973748268, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.826783093803738e-06, "end": 8.897830616173685e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.33549712611666, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.8117678910391204e-06, "end": 1.3470216057300002e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -114.51751885691651, "peak_is_zero": false}]]` | 0–3 |
| `/cells/11/settling_rerun` | *absent* | `null` | — |
| `/cells/11/wall_time_s` | *absent* | `11.294960260391235` | — |
| `/cells/11/warnings/0/count` | *absent* | `8` | — |
| `/cells/11/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/11/warnings/1/count` | *absent* | `56` | — |
| `/cells/11/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/11/warnings/2/count` | *absent* | `8` | — |
| `/cells/11/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/11/warnings/3/count` | *absent* | `16` | — |
| `/cells/11/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/12/column_power_max` | *absent* | `1.0061350915129816` | — |
| `/cells/12/column_power_per_bin` | *absent* | `[[0.9984410013100264, 0.998754822524623, 0.9988458314304183, 0.999002358998443, 0.9989666341281697, 0.9988067278954174, 0.9988136347065547, 0.9990065195933193, 0.9993342114345213, 0.9997164824825199, 1.0002371139002277, 1.0009794612843506, 1.0018423780524013, 1.00278142457781, 1.0037196758710514, 1.0047908579584364, 1.0061350915129816], [0.997441149887275, 0.9975410650015472, 0.9975330614801491, 0.9976797358710777, 0.9980265235209318, 0.9985679644232203, 0.9991789730242138, 0.9996746022720213, 1.000148633839889, 1.0006425616376253, 1.0011311735462975, 1.001568224893956, 1.0019420545699147, 1.0024199021328521, 1.003036325196102, 1.0038104500186502, 1.0048326480260106]]` | 0–1 |
| `/cells/12/cpml_layers` | *absent* | `17` | — |
| `/cells/12/dt_s` | *absent* | `4.842700168608755e-12` | — |
| `/cells/12/dut` | *absent* | `"thru"` | — |
| `/cells/12/dut_cells` | *absent* | `null` | — |
| `/cells/12/dut_runs_xyz` | *absent* | `null` | — |
| `/cells/12/dx_m` | *absent* | `0.00254` | — |
| `/cells/12/fc_discrete_guide_hz` | *absent* | `6523900723.790886` | — |
| `/cells/12/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/12/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/12/grid_shape` | *absent* | `[83, 10, 5]` | 0–2 |
| `/cells/12/guide_cells_yz` | *absent* | `[9, 4]` | 0–1 |
| `/cells/12/lane` | *absent* | `"false"` | — |
| `/cells/12/n_steps` | *absent* | `713` | — |
| `/cells/12/non_vacuity_max_s11` | *absent* | `0.04104806371179083` | — |
| `/cells/12/num_periods` | *absent* | `40.0` | — |
| `/cells/12/port_f_cutoff_hz` | *absent* | `[6523900723.7908745, 6523900723.7908745]` | 0–1 |
| `/cells/12/power_closure_max` | *absent* | `0.006135091512981639` | — |
| `/cells/12/preflight` | *absent* | `[]` | — |
| `/cells/12/reciprocity_complex_max` | *absent* | `0.0016440615127438186` | — |
| `/cells/12/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `1fa130a9bc86` | 0–16 |
| `/cells/12/reciprocity_mag_mean` | *absent* | `0.0004833618509940362` | — |
| `/cells/12/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/12/rung` | *absent* | `"coarse"` | — |
| `/cells/12/s_params/S11` | *absent* | array[17], SHA256 `6d0da0bf5f2a` | 0–16 |
| `/cells/12/s_params/S12` | *absent* | array[17], SHA256 `aaaa32f94599` | 0–16 |
| `/cells/12/s_params/S21` | *absent* | array[17], SHA256 `d05d5bd297e1` | 0–16 |
| `/cells/12/s_params/S22` | *absent* | array[17], SHA256 `9f2ffcc6c4a7` | 0–16 |
| `/cells/12/settling_db/left` | *absent* | `-84.89087147965036` | — |
| `/cells/12/settling_db/right` | *absent* | `-85.32557330359965` | — |
| `/cells/12/settling_db_over_normal_records` | *absent* | `-84.89087147965036` | — |
| `/cells/12/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/12/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.9400219121162081, "end": 1.574705427662944e-10, "n_nonzero": 702, "n_steps": 713, "db": -97.75938652809123, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9126197906760503, "end": 1.3338837049163657e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.35171915334394, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.344678748221921e-06, "end": 1.317588554617071e-16, "n_nonzero": 702, "n_steps": 713, "db": -104.04574597091673, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.4152605662281064e-06, "end": 7.493914466072573e-17, "n_nonzero": 709, "n_steps": 713, "db": -106.58715111799884, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9169043789893863, "end": 1.5905884254842815e-09, "n_nonzero": 684, "n_steps": 713, "db": -87.60766229034988, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9144998107090352, "end": 2.9654900551484287e-09, "n_nonzero": 677, "n_steps": 713, "db": -84.89087147965036, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.3944336881157355e-06, "end": 3.0907051577765066e-15, "n_nonzero": 684, "n_steps": 713, "db": -90.40709752044778, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.337795644381731e-06, "end": 8.649387559171213e-15, "n_nonzero": 677, "n_steps": 713, "db": -85.86474386316092, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9714454174723528, "end": 1.4750886192055612e-09, "n_nonzero": 684, "n_steps": 713, "db": -88.18600291653425, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9523866158439809, "end": 2.79419012295999e-09, "n_nonzero": 677, "n_steps": 713, "db": -85.32557330359965, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.4862024442234267e-06, "end": 3.995148781550744e-15, "n_nonzero": 684, "n_steps": 713, "db": -89.40819645724636, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.470284873978999e-06, "end": 1.0150291162093852e-14, "n_nonzero": 677, "n_steps": 713, "db": -85.33886627067159, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9693575972582416, "end": 9.735966978419422e-11, "n_nonzero": 702, "n_steps": 713, "db": -99.98104926176009, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9670672115243271, "end": 1.4141394761706615e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.3496441284352, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.5456732884719527e-06, "end": 1.8119224989871332e-16, "n_nonzero": 702, "n_steps": 713, "db": -102.91559097855527, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.508374309928316e-06, "end": 1.056021782366216e-16, "n_nonzero": 709, "n_steps": 713, "db": -105.2143304582434, "peak_is_zero": false}]]` | 0–1 |
| `/cells/12/settling_rerun` | *absent* | `null` | — |
| `/cells/12/wall_time_s` | *absent* | `5.345367431640625` | — |
| `/cells/12/warnings/0/count` | *absent* | `1` | — |
| `/cells/12/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/13/column_power_max` | *absent* | `1.000022659840888` | — |
| `/cells/13/column_power_per_bin` | *absent* | `[[0.9999516931017539, 1.000022659840888, 0.9999885307767362, 1.0000057409710195, 0.9999989542541615, 0.9999991652142138, 1.0000019036588412, 0.9999976661292499, 1.0000029276857347, 0.9999973444986133, 1.000001889745523, 0.99999708308687, 1.0000004373067701, 0.999999024538912, 0.9999945293317954, 1.0000050996129666, 0.9999946555700279], [0.9999556274241073, 1.0000193086207128, 0.9999904428605682, 1.0000041296791882, 1.000000995762857, 0.9999982649878695, 1.0000017973678335, 0.9999964914592533, 1.0000026474083692, 0.999997368880379, 1.0000019118959762, 0.9999989326985205, 0.9999980683612117, 1.000001739782249, 0.9999917240474123, 1.0000104840131643, 0.9999872470232559]]` | 0–1 |
| `/cells/13/cpml_layers` | *absent* | `17` | — |
| `/cells/13/dt_s` | *absent* | `4.842700168608755e-12` | — |
| `/cells/13/dut` | *absent* | `"thru"` | — |
| `/cells/13/dut_cells` | *absent* | `null` | — |
| `/cells/13/dut_runs_xyz` | *absent* | `null` | — |
| `/cells/13/dx_m` | *absent* | `0.00254` | — |
| `/cells/13/fc_discrete_guide_hz` | *absent* | `6523900723.790886` | — |
| `/cells/13/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/13/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/13/grid_shape` | *absent* | `[83, 10, 5]` | 0–2 |
| `/cells/13/guide_cells_yz` | *absent* | `[9, 4]` | 0–1 |
| `/cells/13/lane` | *absent* | `"flux"` | — |
| `/cells/13/n_steps` | *absent* | `713` | — |
| `/cells/13/non_vacuity_max_s11` | *absent* | `0.0` | — |
| `/cells/13/num_periods` | *absent* | `40.0` | — |
| `/cells/13/port_f_cutoff_hz` | *absent* | `[6523900723.7908745, 6523900723.7908745]` | 0–1 |
| `/cells/13/power_closure_max` | *absent* | `4.830689824608658e-05` | — |
| `/cells/13/preflight` | *absent* | `[]` | — |
| `/cells/13/reciprocity_complex_max` | *absent* | `0.001440761837417787` | — |
| `/cells/13/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `58fa9af1d1ff` | 0–16 |
| `/cells/13/reciprocity_mag_mean` | *absent* | `1.11442927786361e-06` | — |
| `/cells/13/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/13/rung` | *absent* | `"coarse"` | — |
| `/cells/13/s_params/S11` | *absent* | array[17], SHA256 `aaba09f402c6` | 0–16 |
| `/cells/13/s_params/S12` | *absent* | array[17], SHA256 `34dced4634cb` | 0–16 |
| `/cells/13/s_params/S21` | *absent* | array[17], SHA256 `eb2a4c4110a0` | 0–16 |
| `/cells/13/s_params/S22` | *absent* | array[17], SHA256 `41cee68c13c0` | 0–16 |
| `/cells/13/settling_db/left` | *absent* | `-84.89087147965036` | — |
| `/cells/13/settling_db/right` | *absent* | `-85.32557330359965` | — |
| `/cells/13/settling_db_over_normal_records` | *absent* | `-84.89087147965036` | — |
| `/cells/13/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/13/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.9400219121162081, "end": 1.574705427662944e-10, "n_nonzero": 702, "n_steps": 713, "db": -97.75938652809123, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9126197906760503, "end": 1.3338837049163657e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.35171915334394, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.344678748221921e-06, "end": 1.317588554617071e-16, "n_nonzero": 702, "n_steps": 713, "db": -104.04574597091673, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.4152605662281064e-06, "end": 7.493914466072573e-17, "n_nonzero": 709, "n_steps": 713, "db": -106.58715111799884, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9169043789893863, "end": 1.5905884254842815e-09, "n_nonzero": 684, "n_steps": 713, "db": -87.60766229034988, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9144998107090352, "end": 2.9654900551484287e-09, "n_nonzero": 677, "n_steps": 713, "db": -84.89087147965036, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.3944336881157355e-06, "end": 3.0907051577765066e-15, "n_nonzero": 684, "n_steps": 713, "db": -90.40709752044778, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.337795644381731e-06, "end": 8.649387559171213e-15, "n_nonzero": 677, "n_steps": 713, "db": -85.86474386316092, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9400219121162081, "end": 1.574705427662944e-10, "n_nonzero": 702, "n_steps": 713, "db": -97.75938652809123, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9126197906760503, "end": 1.3338837049163657e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.35171915334394, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.344678748221921e-06, "end": 1.317588554617071e-16, "n_nonzero": 702, "n_steps": 713, "db": -104.04574597091673, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.4152605662281064e-06, "end": 7.493914466072573e-17, "n_nonzero": 709, "n_steps": 713, "db": -106.58715111799884, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9169043789893863, "end": 1.5905884254842815e-09, "n_nonzero": 684, "n_steps": 713, "db": -87.60766229034988, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9144998107090352, "end": 2.9654900551484287e-09, "n_nonzero": 677, "n_steps": 713, "db": -84.89087147965036, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.3944336881157355e-06, "end": 3.0907051577765066e-15, "n_nonzero": 684, "n_steps": 713, "db": -90.40709752044778, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.337795644381731e-06, "end": 8.649387559171213e-15, "n_nonzero": 677, "n_steps": 713, "db": -85.86474386316092, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9714454174723528, "end": 1.4750886192055612e-09, "n_nonzero": 684, "n_steps": 713, "db": -88.18600291653425, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9523866158439809, "end": 2.79419012295999e-09, "n_nonzero": 677, "n_steps": 713, "db": -85.32557330359965, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.4862024442234267e-06, "end": 3.995148781550744e-15, "n_nonzero": 684, "n_steps": 713, "db": -89.40819645724636, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.470284873978999e-06, "end": 1.0150291162093852e-14, "n_nonzero": 677, "n_steps": 713, "db": -85.33886627067159, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9693575972582416, "end": 9.735966978419422e-11, "n_nonzero": 702, "n_steps": 713, "db": -99.98104926176009, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9670672115243271, "end": 1.4141394761706615e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.3496441284352, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.5456732884719527e-06, "end": 1.8119224989871332e-16, "n_nonzero": 702, "n_steps": 713, "db": -102.91559097855527, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.508374309928316e-06, "end": 1.056021782366216e-16, "n_nonzero": 709, "n_steps": 713, "db": -105.2143304582434, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9714454174723528, "end": 1.4750886192055612e-09, "n_nonzero": 684, "n_steps": 713, "db": -88.18600291653425, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9523866158439809, "end": 2.79419012295999e-09, "n_nonzero": 677, "n_steps": 713, "db": -85.32557330359965, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.4862024442234267e-06, "end": 3.995148781550744e-15, "n_nonzero": 684, "n_steps": 713, "db": -89.40819645724636, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.470284873978999e-06, "end": 1.0150291162093852e-14, "n_nonzero": 677, "n_steps": 713, "db": -85.33886627067159, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9693575972582416, "end": 9.735966978419422e-11, "n_nonzero": 702, "n_steps": 713, "db": -99.98104926176009, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9670672115243271, "end": 1.4141394761706615e-10, "n_nonzero": 709, "n_steps": 713, "db": -98.3496441284352, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.5456732884719527e-06, "end": 1.8119224989871332e-16, "n_nonzero": 702, "n_steps": 713, "db": -102.91559097855527, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.508374309928316e-06, "end": 1.056021782366216e-16, "n_nonzero": 709, "n_steps": 713, "db": -105.2143304582434, "peak_is_zero": false}]]` | 0–3 |
| `/cells/13/settling_rerun` | *absent* | `null` | — |
| `/cells/13/wall_time_s` | *absent* | `8.545785903930664` | — |
| `/cells/13/warnings/0/count` | *absent* | `8` | — |
| `/cells/13/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/13/warnings/1/count` | *absent* | `56` | — |
| `/cells/13/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/13/warnings/2/count` | *absent* | `8` | — |
| `/cells/13/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/13/warnings/3/count` | *absent* | `16` | — |
| `/cells/13/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/14/column_power_max` | *absent* | `1.0002546142243787` | — |
| `/cells/14/column_power_per_bin` | *absent* | `[[0.9997275853169826, 0.9997674838384105, 0.9997640496230102, 0.9997839849898255, 0.999798056205093, 0.9998082910105757, 0.9998237838384465, 0.9998397123291957, 0.9998641779827162, 0.9998896235890978, 0.999918025009238, 0.9999558322448847, 0.9999960247403177, 1.000049144868709, 1.0001018396677706, 1.0001676631537957, 1.0002546142243787], [0.9997197300754955, 0.9997543240801801, 0.9997472955230738, 0.9997638165682656, 0.999778887066119, 0.9997989618522343, 0.9998258971961428, 0.9998491148726785, 0.9998766005009281, 0.999906347665159, 0.9999381793890925, 0.9999752703931876, 1.0000103360315065, 1.000056034054761, 1.000103902305274, 1.0001650150432009, 1.0002491988238105]]` | 0–1 |
| `/cells/14/cpml_layers` | *absent* | `68` | — |
| `/cells/14/dt_s` | *absent* | `1.2106750421521888e-12` | — |
| `/cells/14/dut` | *absent* | `"thru"` | — |
| `/cells/14/dut_cells` | *absent* | `null` | — |
| `/cells/14/dut_runs_xyz` | *absent* | `null` | — |
| `/cells/14/dx_m` | *absent* | `0.000635` | — |
| `/cells/14/fc_discrete_guide_hz` | *absent* | `6555059929.275007` | — |
| `/cells/14/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/14/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/14/grid_shape` | *absent* | `[329, 37, 17]` | 0–2 |
| `/cells/14/guide_cells_yz` | *absent* | `[36, 16]` | 0–1 |
| `/cells/14/lane` | *absent* | `"false"` | — |
| `/cells/14/n_steps` | *absent* | `2849` | — |
| `/cells/14/non_vacuity_max_s11` | *absent* | `0.007028131986406278` | — |
| `/cells/14/num_periods` | *absent* | `40.0` | — |
| `/cells/14/port_f_cutoff_hz` | *absent* | `[6555059929.275057, 6555059929.275057]` | 0–1 |
| `/cells/14/power_closure_max` | *absent* | `0.00028026992450447263` | — |
| `/cells/14/preflight` | *absent* | `[]` | — |
| `/cells/14/reciprocity_complex_max` | *absent* | `2.732828764712657e-05` | — |
| `/cells/14/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `ce404d7aae46` | 0–16 |
| `/cells/14/reciprocity_mag_mean` | *absent* | `7.081816602137183e-06` | — |
| `/cells/14/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/14/rung` | *absent* | `"fine"` | — |
| `/cells/14/s_params/S11` | *absent* | array[17], SHA256 `caa665ae5810` | 0–16 |
| `/cells/14/s_params/S12` | *absent* | array[17], SHA256 `512609136421` | 0–16 |
| `/cells/14/s_params/S21` | *absent* | array[17], SHA256 `ca33ffd8c5c7` | 0–16 |
| `/cells/14/s_params/S22` | *absent* | array[17], SHA256 `2a4a0fb34f9d` | 0–16 |
| `/cells/14/settling_db/left` | *absent* | `-100.89879388166203` | — |
| `/cells/14/settling_db/right` | *absent* | `-101.1073500105272` | — |
| `/cells/14/settling_db_over_normal_records` | *absent* | `-100.89879388166203` | — |
| `/cells/14/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/14/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.9952865483487727, "end": 1.1326987547930805e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.43833711385913, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9901537090407864, "end": 4.063429080539113e-12, "n_nonzero": 2836, "n_steps": 2849, "db": -113.86809934277436, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.894456923516818e-06, "end": 3.47745836310291e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -120.49184965756021, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.87815157222674e-06, "end": 4.993003318337167e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -118.90262924191421, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9876574828956706, "end": 7.026260629168506e-11, "n_nonzero": 2724, "n_steps": 2849, "db": -101.4788210318288, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9710577227363046, "end": 7.895245855392312e-11, "n_nonzero": 2684, "n_steps": 2849, "db": -100.89879388166203, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.863768512605984e-06, "end": 1.5251477431339775e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -114.0369918250916, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.819790315780442e-06, "end": 1.875604550381926e-17, "n_nonzero": 2681, "n_steps": 2849, "db": -113.08898245626067, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9896803494050346, "end": 5.6752706840378657e-11, "n_nonzero": 2723, "n_steps": 2849, "db": -102.41508367030224, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9751309640854657, "end": 7.55662631100277e-11, "n_nonzero": 2683, "n_steps": 2849, "db": -101.1073500105272, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.877838872034557e-06, "end": 2.0105031689624555e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -112.852849977847, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.819990540764604e-06, "end": 1.9712635296494987e-17, "n_nonzero": 2680, "n_steps": 2849, "db": -112.87317600391744, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9966298174796897, "end": 1.165745793462025e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.31930019387812, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9908542896482686, "end": 1.7895967871154716e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -107.43254602490282, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.907805115333819e-06, "end": 3.14251345486049e-19, "n_nonzero": 2808, "n_steps": 2849, "db": -130.94655751231517, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.889752383737103e-06, "end": 3.0493122322745653e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -121.05720059685383, "peak_is_zero": false}]]` | 0–1 |
| `/cells/14/settling_rerun` | *absent* | `null` | — |
| `/cells/14/wall_time_s` | *absent* | `6.835801839828491` | — |
| `/cells/14/warnings/0/count` | *absent* | `1` | — |
| `/cells/14/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/15/column_power_max` | *absent* | `1.0000052143659535` | — |
| `/cells/15/column_power_per_bin` | *absent* | `[[1.0000050254255486, 0.9999953011865473, 0.9999998172691703, 1.000001483764098, 0.9999964241096824, 1.0000003097305887, 0.9999993088704073, 1.0000011563890054, 1.0000014172415208, 1.0000000748062376, 0.9999986937256189, 0.9999988703130107, 1.0000009156810918, 1.0000024265069134, 1.000003297286231, 1.000001892679624, 1.0000021515605972], [1.0000052143659535, 0.9999979450306243, 0.9999974410183121, 0.9999999939741191, 0.9999966621453193, 1.0000012749426088, 0.9999985922647914, 0.9999982615634769, 1.000001261003626, 1.0000019700672915, 0.9999994825381647, 0.9999996161490634, 1.0000022065135004, 1.0000014024295325, 1.0000023662811035, 1.00000396009616, 1.0000010184946326]]` | 0–1 |
| `/cells/15/cpml_layers` | *absent* | `68` | — |
| `/cells/15/dt_s` | *absent* | `1.2106750421521888e-12` | — |
| `/cells/15/dut` | *absent* | `"thru"` | — |
| `/cells/15/dut_cells` | *absent* | `null` | — |
| `/cells/15/dut_runs_xyz` | *absent* | `null` | — |
| `/cells/15/dx_m` | *absent* | `0.000635` | — |
| `/cells/15/fc_discrete_guide_hz` | *absent* | `6555059929.275007` | — |
| `/cells/15/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/15/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/15/grid_shape` | *absent* | `[329, 37, 17]` | 0–2 |
| `/cells/15/guide_cells_yz` | *absent* | `[36, 16]` | 0–1 |
| `/cells/15/lane` | *absent* | `"flux"` | — |
| `/cells/15/n_steps` | *absent* | `2849` | — |
| `/cells/15/non_vacuity_max_s11` | *absent* | `0.0` | — |
| `/cells/15/num_periods` | *absent* | `40.0` | — |
| `/cells/15/port_f_cutoff_hz` | *absent* | `[6555059929.275057, 6555059929.275057]` | 0–1 |
| `/cells/15/power_closure_max` | *absent* | `5.214365953465361e-06` | — |
| `/cells/15/preflight` | *absent* | `[]` | — |
| `/cells/15/reciprocity_complex_max` | *absent* | `2.7312132288204973e-05` | — |
| `/cells/15/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `52377e4e2276` | 0–16 |
| `/cells/15/reciprocity_mag_mean` | *absent* | `6.337069369487438e-07` | — |
| `/cells/15/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/15/rung` | *absent* | `"fine"` | — |
| `/cells/15/s_params/S11` | *absent* | array[17], SHA256 `8edbf849950e` | 0–16 |
| `/cells/15/s_params/S12` | *absent* | array[17], SHA256 `766a6dce994f` | 0–16 |
| `/cells/15/s_params/S21` | *absent* | array[17], SHA256 `1ab8df83d82a` | 0–16 |
| `/cells/15/s_params/S22` | *absent* | array[17], SHA256 `40e3b9ee6303` | 0–16 |
| `/cells/15/settling_db/left` | *absent* | `-100.89879388166203` | — |
| `/cells/15/settling_db/right` | *absent* | `-101.1073500105272` | — |
| `/cells/15/settling_db_over_normal_records` | *absent* | `-100.89879388166203` | — |
| `/cells/15/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/15/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.9952865483487727, "end": 1.1326987547930805e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.43833711385913, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9901537090407864, "end": 4.063429080539113e-12, "n_nonzero": 2836, "n_steps": 2849, "db": -113.86809934277436, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.894456923516818e-06, "end": 3.47745836310291e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -120.49184965756021, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.87815157222674e-06, "end": 4.993003318337167e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -118.90262924191421, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9876574828956706, "end": 7.026260629168506e-11, "n_nonzero": 2724, "n_steps": 2849, "db": -101.4788210318288, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9710577227363046, "end": 7.895245855392312e-11, "n_nonzero": 2684, "n_steps": 2849, "db": -100.89879388166203, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.863768512605984e-06, "end": 1.5251477431339775e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -114.0369918250916, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.819790315780442e-06, "end": 1.875604550381926e-17, "n_nonzero": 2681, "n_steps": 2849, "db": -113.08898245626067, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9952865483487727, "end": 1.1326987547930805e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.43833711385913, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9901537090407864, "end": 4.063429080539113e-12, "n_nonzero": 2836, "n_steps": 2849, "db": -113.86809934277436, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.894456923516818e-06, "end": 3.47745836310291e-18, "n_nonzero": 2808, "n_steps": 2849, "db": -120.49184965756021, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.87815157222674e-06, "end": 4.993003318337167e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -118.90262924191421, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9876574828956706, "end": 7.026260629168506e-11, "n_nonzero": 2724, "n_steps": 2849, "db": -101.4788210318288, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9710577227363046, "end": 7.895245855392312e-11, "n_nonzero": 2684, "n_steps": 2849, "db": -100.89879388166203, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.863768512605984e-06, "end": 1.5251477431339775e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -114.0369918250916, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.819790315780442e-06, "end": 1.875604550381926e-17, "n_nonzero": 2681, "n_steps": 2849, "db": -113.08898245626067, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9896803494050346, "end": 5.6752706840378657e-11, "n_nonzero": 2723, "n_steps": 2849, "db": -102.41508367030224, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9751309640854657, "end": 7.55662631100277e-11, "n_nonzero": 2683, "n_steps": 2849, "db": -101.1073500105272, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.877838872034557e-06, "end": 2.0105031689624555e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -112.852849977847, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.819990540764604e-06, "end": 1.9712635296494987e-17, "n_nonzero": 2680, "n_steps": 2849, "db": -112.87317600391744, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9966298174796897, "end": 1.165745793462025e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.31930019387812, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9908542896482686, "end": 1.7895967871154716e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -107.43254602490282, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.907805115333819e-06, "end": 3.14251345486049e-19, "n_nonzero": 2808, "n_steps": 2849, "db": -130.94655751231517, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.889752383737103e-06, "end": 3.0493122322745653e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -121.05720059685383, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9896803494050346, "end": 5.6752706840378657e-11, "n_nonzero": 2723, "n_steps": 2849, "db": -102.41508367030224, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9751309640854657, "end": 7.55662631100277e-11, "n_nonzero": 2683, "n_steps": 2849, "db": -101.1073500105272, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.877838872034557e-06, "end": 2.0105031689624555e-17, "n_nonzero": 2721, "n_steps": 2849, "db": -112.852849977847, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.819990540764604e-06, "end": 1.9712635296494987e-17, "n_nonzero": 2680, "n_steps": 2849, "db": -112.87317600391744, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9966298174796897, "end": 1.165745793462025e-11, "n_nonzero": 2808, "n_steps": 2849, "db": -109.31930019387812, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9908542896482686, "end": 1.7895967871154716e-11, "n_nonzero": 2836, "n_steps": 2849, "db": -107.43254602490282, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.907805115333819e-06, "end": 3.14251345486049e-19, "n_nonzero": 2808, "n_steps": 2849, "db": -130.94655751231517, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.889752383737103e-06, "end": 3.0493122322745653e-18, "n_nonzero": 2836, "n_steps": 2849, "db": -121.05720059685383, "peak_is_zero": false}]]` | 0–3 |
| `/cells/15/settling_rerun` | *absent* | `null` | — |
| `/cells/15/wall_time_s` | *absent* | `12.431143045425415` | — |
| `/cells/15/warnings/0/count` | *absent* | `8` | — |
| `/cells/15/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/15/warnings/1/count` | *absent* | `56` | — |
| `/cells/15/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/15/warnings/2/count` | *absent* | `8` | — |
| `/cells/15/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/15/warnings/3/count` | *absent* | `16` | — |
| `/cells/15/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/16/column_power_max` | *absent* | `1.001161080318815` | — |
| `/cells/16/column_power_per_bin` | *absent* | `[[0.9991541054666406, 0.9992430747360409, 0.9992768598679259, 0.9993376362672526, 0.9993812323138636, 0.9993978674496997, 0.9994327990830867, 0.999491411886901, 0.9995772759103393, 0.9996745925464157, 0.9997888211036469, 0.9999431362453642, 1.0001251293830868, 1.0003413578078966, 1.0005695943282762, 1.0008309878063517, 1.001161080318815], [0.9990694366620767, 0.9991186202974709, 0.9991279954111261, 0.9991655711580452, 0.9992352058775746, 0.9993355682953471, 0.9994584004737525, 0.9995676612083046, 0.9996806432176022, 0.9998036938878732, 0.9999358807007228, 1.0000725219223, 1.0002044813411521, 1.0003616475102997, 1.0005474038992068, 1.0007727424378072, 1.0010674095167578]]` | 0–1 |
| `/cells/16/cpml_layers` | *absent* | `34` | — |
| `/cells/16/dt_s` | *absent* | `2.4213500843043776e-12` | — |
| `/cells/16/dut` | *absent* | `"thru"` | — |
| `/cells/16/dut_cells` | *absent* | `null` | — |
| `/cells/16/dut_runs_xyz` | *absent* | `null` | — |
| `/cells/16/dx_m` | *absent* | `0.00127` | — |
| `/cells/16/fc_discrete_guide_hz` | *absent* | `6548820964.704695` | — |
| `/cells/16/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/16/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/16/grid_shape` | *absent* | `[165, 19, 9]` | 0–2 |
| `/cells/16/guide_cells_yz` | *absent* | `[18, 8]` | 0–1 |
| `/cells/16/lane` | *absent* | `"false"` | — |
| `/cells/16/n_steps` | *absent* | `1425` | — |
| `/cells/16/non_vacuity_max_s11` | *absent* | `0.01640495359122128` | — |
| `/cells/16/num_periods` | *absent* | `40.0` | — |
| `/cells/16/port_f_cutoff_hz` | *absent* | `[6548820964.704762, 6548820964.704762]` | 0–1 |
| `/cells/16/power_closure_max` | *absent* | `0.0011610803188149` | — |
| `/cells/16/preflight` | *absent* | `[]` | — |
| `/cells/16/reciprocity_complex_max` | *absent* | `0.00021321126994232987` | — |
| `/cells/16/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `f37842ee0924` | 0–16 |
| `/cells/16/reciprocity_mag_mean` | *absent* | `5.6895471025336136e-05` | — |
| `/cells/16/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/16/rung` | *absent* | `"mid"` | — |
| `/cells/16/s_params/S11` | *absent* | array[17], SHA256 `e7c10176bced` | 0–16 |
| `/cells/16/s_params/S12` | *absent* | array[17], SHA256 `e433800f258c` | 0–16 |
| `/cells/16/s_params/S21` | *absent* | array[17], SHA256 `ec56d8383bd6` | 0–16 |
| `/cells/16/s_params/S22` | *absent* | array[17], SHA256 `6804b163c89b` | 0–16 |
| `/cells/16/settling_db/left` | *absent* | `-97.70112297018578` | — |
| `/cells/16/settling_db/right` | *absent* | `-98.30055823398186` | — |
| `/cells/16/settling_db_over_normal_records` | *absent* | `-97.70112297018578` | — |
| `/cells/16/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/16/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.9840568725104646, "end": 1.6246217449680167e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -107.82267936929927, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9781914306959152, "end": 9.87697700895852e-12, "n_nonzero": 1418, "n_steps": 1425, "db": -109.95799811083303, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.770321407869787e-06, "end": 9.299535888939829e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.07917099201708, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.7602532100404506e-06, "end": 1.5409660321771818e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -113.8742402509415, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.973896732193694, "end": 1.653486340667483e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -97.70112297018578, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9579650783061879, "end": 1.4844254594097538e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.09791283246584, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.760061780475109e-06, "end": 1.477807940725195e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -104.05576984922492, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.7104500469976377e-06, "end": 3.722243600521151e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.98621797222226, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9842692702859637, "end": 1.2781294628933416e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -98.86539080145886, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9710144936946108, "end": 1.436051083479876e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.30055823398186, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.7995909916816676e-06, "end": 1.735459768409189e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -103.40322299080681, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.734578927705405e-06, "end": 4.008070987695212e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.6930623758277, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9905771124758758, "end": 1.1545021515743483e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -109.3349354266141, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9832422629269786, "end": 3.948578715644504e-11, "n_nonzero": 1418, "n_steps": 1425, "db": -103.9621973748268, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.826783093803738e-06, "end": 8.897830616173685e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.33549712611666, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.8117678910391204e-06, "end": 1.3470216057300002e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -114.51751885691651, "peak_is_zero": false}]]` | 0–1 |
| `/cells/16/settling_rerun` | *absent* | `null` | — |
| `/cells/16/wall_time_s` | *absent* | `6.539209842681885` | — |
| `/cells/16/warnings/0/count` | *absent* | `1` | — |
| `/cells/16/warnings/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/cells/17/column_power_max` | *absent* | `1.0000024479673224` | — |
| `/cells/17/column_power_per_bin` | *absent* | `[[1.0000002954794356, 1.0000002599067828, 0.9999984961004035, 1.0000007135469076, 0.9999992979156678, 1.0000007066437462, 1.0000011929153345, 0.9999991759533241, 0.9999999965186142, 0.9999991203606555, 1.0000018813397642, 0.9999993797482647, 1.000001891294975, 0.9999996911111176, 1.0000014595229665, 0.9999999530607177, 0.9999963118706676], [1.0000024479673224, 0.9999991806716222, 0.9999993044192373, 1.0000000141023202, 0.9999994990784044, 0.9999995076832777, 1.0000003144149476, 0.9999989346343701, 0.9999999505154245, 0.9999994117680729, 1.0000009324454495, 0.9999998156246456, 1.000001306330508, 1.0000005059801431, 0.9999993520759397, 1.0000024060182997, 0.9999933540905914]]` | 0–1 |
| `/cells/17/cpml_layers` | *absent* | `34` | — |
| `/cells/17/dt_s` | *absent* | `2.4213500843043776e-12` | — |
| `/cells/17/dut` | *absent* | `"thru"` | — |
| `/cells/17/dut_cells` | *absent* | `null` | — |
| `/cells/17/dut_runs_xyz` | *absent* | `null` | — |
| `/cells/17/dx_m` | *absent* | `0.00127` | — |
| `/cells/17/fc_discrete_guide_hz` | *absent* | `6548820964.704695` | — |
| `/cells/17/fc_te10_numerical_hz` | *absent* | `6557140376.202973` | — |
| `/cells/17/float32_normal_min` | *absent* | `1.1754943508222875e-38` | — |
| `/cells/17/grid_shape` | *absent* | `[165, 19, 9]` | 0–2 |
| `/cells/17/guide_cells_yz` | *absent* | `[18, 8]` | 0–1 |
| `/cells/17/lane` | *absent* | `"flux"` | — |
| `/cells/17/n_steps` | *absent* | `1425` | — |
| `/cells/17/non_vacuity_max_s11` | *absent* | `0.0` | — |
| `/cells/17/num_periods` | *absent* | `40.0` | — |
| `/cells/17/port_f_cutoff_hz` | *absent* | `[6548820964.704762, 6548820964.704762]` | 0–1 |
| `/cells/17/power_closure_max` | *absent* | `6.64590940857579e-06` | — |
| `/cells/17/preflight` | *absent* | `[]` | — |
| `/cells/17/reciprocity_complex_max` | *absent* | `0.0002113002826070289` | — |
| `/cells/17/reciprocity_complex_per_bin` | *absent* | array[17], SHA256 `61705bf3bc9a` | 0–16 |
| `/cells/17/reciprocity_mag_mean` | *absent* | `5.264596456391561e-07` | — |
| `/cells/17/reference_planes_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/cells/17/rung` | *absent* | `"mid"` | — |
| `/cells/17/s_params/S11` | *absent* | array[17], SHA256 `606f3d161ec4` | 0–16 |
| `/cells/17/s_params/S12` | *absent* | array[17], SHA256 `9425602e59f1` | 0–16 |
| `/cells/17/s_params/S21` | *absent* | array[17], SHA256 `f75aeddaece8` | 0–16 |
| `/cells/17/s_params/S22` | *absent* | array[17], SHA256 `04df75a69874` | 0–16 |
| `/cells/17/settling_db/left` | *absent* | `-97.70112297018578` | — |
| `/cells/17/settling_db/right` | *absent* | `-98.30055823398186` | — |
| `/cells/17/settling_db_over_normal_records` | *absent* | `-97.70112297018578` | — |
| `/cells/17/settling_degenerate_records` | *absent* | `[]` | — |
| `/cells/17/settling_records` | *absent* | `[[{"port_index": 0, "record": "v_probe_t", "peak": 0.9840568725104646, "end": 1.6246217449680167e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -107.82267936929927, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9781914306959152, "end": 9.87697700895852e-12, "n_nonzero": 1418, "n_steps": 1425, "db": -109.95799811083303, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.770321407869787e-06, "end": 9.299535888939829e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.07917099201708, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.7602532100404506e-06, "end": 1.5409660321771818e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -113.8742402509415, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.973896732193694, "end": 1.653486340667483e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -97.70112297018578, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9579650783061879, "end": 1.4844254594097538e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.09791283246584, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.760061780475109e-06, "end": 1.477807940725195e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -104.05576984922492, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.7104500469976377e-06, "end": 3.722243600521151e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.98621797222226, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9840568725104646, "end": 1.6246217449680167e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -107.82267936929927, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9781914306959152, "end": 9.87697700895852e-12, "n_nonzero": 1418, "n_steps": 1425, "db": -109.95799811083303, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.770321407869787e-06, "end": 9.299535888939829e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.07917099201708, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.7602532100404506e-06, "end": 1.5409660321771818e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -113.8742402509415, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.973896732193694, "end": 1.653486340667483e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -97.70112297018578, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9579650783061879, "end": 1.4844254594097538e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.09791283246584, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.760061780475109e-06, "end": 1.477807940725195e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -104.05576984922492, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.7104500469976377e-06, "end": 3.722243600521151e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.98621797222226, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9842692702859637, "end": 1.2781294628933416e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -98.86539080145886, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9710144936946108, "end": 1.436051083479876e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.30055823398186, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.7995909916816676e-06, "end": 1.735459768409189e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -103.40322299080681, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.734578927705405e-06, "end": 4.008070987695212e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.6930623758277, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9905771124758758, "end": 1.1545021515743483e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -109.3349354266141, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9832422629269786, "end": 3.948578715644504e-11, "n_nonzero": 1418, "n_steps": 1425, "db": -103.9621973748268, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.826783093803738e-06, "end": 8.897830616173685e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.33549712611666, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.8117678910391204e-06, "end": 1.3470216057300002e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -114.51751885691651, "peak_is_zero": false}], [{"port_index": 0, "record": "v_probe_t", "peak": 0.9842692702859637, "end": 1.2781294628933416e-10, "n_nonzero": 1368, "n_steps": 1425, "db": -98.86539080145886, "peak_is_zero": false}, {"port_index": 0, "record": "v_ref_t", "peak": 0.9710144936946108, "end": 1.436051083479876e-10, "n_nonzero": 1354, "n_steps": 1425, "db": -98.30055823398186, "peak_is_zero": false}, {"port_index": 0, "record": "i_probe_t", "peak": 3.7995909916816676e-06, "end": 1.735459768409189e-16, "n_nonzero": 1368, "n_steps": 1425, "db": -103.40322299080681, "peak_is_zero": false}, {"port_index": 0, "record": "i_ref_t", "peak": 3.734578927705405e-06, "end": 4.008070987695212e-16, "n_nonzero": 1354, "n_steps": 1425, "db": -99.6930623758277, "peak_is_zero": false}, {"port_index": 1, "record": "v_probe_t", "peak": 0.9905771124758758, "end": 1.1545021515743483e-11, "n_nonzero": 1404, "n_steps": 1425, "db": -109.3349354266141, "peak_is_zero": false}, {"port_index": 1, "record": "v_ref_t", "peak": 0.9832422629269786, "end": 3.948578715644504e-11, "n_nonzero": 1418, "n_steps": 1425, "db": -103.9621973748268, "peak_is_zero": false}, {"port_index": 1, "record": "i_probe_t", "peak": 3.826783093803738e-06, "end": 8.897830616173685e-18, "n_nonzero": 1404, "n_steps": 1425, "db": -116.33549712611666, "peak_is_zero": false}, {"port_index": 1, "record": "i_ref_t", "peak": 3.8117678910391204e-06, "end": 1.3470216057300002e-17, "n_nonzero": 1418, "n_steps": 1425, "db": -114.51751885691651, "peak_is_zero": false}]]` | 0–3 |
| `/cells/17/settling_rerun` | *absent* | `null` | — |
| `/cells/17/wall_time_s` | *absent* | `11.919247150421143` | — |
| `/cells/17/warnings/0/count` | *absent* | `8` | — |
| `/cells/17/warnings/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/17/warnings/1/count` | *absent* | `56` | — |
| `/cells/17/warnings/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/17/warnings/2/count` | *absent* | `8` | — |
| `/cells/17/warnings/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/cells/17/warnings/3/count` | *absent* | `16` | — |
| `/cells/17/warnings/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/fixture/a_m` | *absent* | `0.02286` | — |
| `/fixture/b_m` | *absent* | `0.01016` | — |
| `/fixture/band_centre_bin` | *absent* | `8` | — |
| `/fixture/bandwidth` | *absent* | `0.5` | — |
| `/fixture/boundary` | *absent* | `"cpml-x, pec-y, pec-z"` | — |
| `/fixture/domain_x_m` | *absent* | `0.12192` | — |
| `/fixture/dx_ladder_m` | *absent* | `[0.00254, 0.00127, 0.000635]` | 0–2 |
| `/fixture/f0_hz` | *absent* | `10000000000.0` | — |
| `/fixture/fd_step_eps` | *absent* | `0.05` | — |
| `/fixture/fd_step_sigma_s_per_m` | *absent* | `0.005` | — |
| `/fixture/freqs_hz` | *absent* | array[17], SHA256 `c0ed1e088b59` | 0–16 |
| `/fixture/lanes` | *absent* | `["false", "flux"]` | 0–1 |
| `/fixture/n_ladder` | *absent* | `[9, 18, 36]` | 0–2 |
| `/fixture/num_periods` | *absent* | `40.0` | — |
| `/fixture/pec_short_window_x_m` | *absent* | `[0.048260000000000004, 0.05842000000000001]` | 0–1 |
| `/fixture/pec_short_x_m` | *absent* | `[0.05842000000000001, 0.0635]` | 0–1 |
| `/fixture/port_planes_m` | *absent* | `[0.012700000000000001, 0.10922000000000001]` | 0–1 |
| `/fixture/probe_planes_m` | *absent* | `[0.0381, 0.08382]` | 0–1 |
| `/fixture/reference_planes_default_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/fixture/reference_planes_shifted_m` | *absent* | `[0.025400000000000002, 0.09906000000000001]` | 0–1 |
| `/fixture/slab_eps_r` | *absent* | `4.0` | — |
| `/fixture/slab_x_m` | *absent* | `[0.055880000000000006, 0.06604]` | 0–1 |
| `/fixture/theta0_eps` | *absent* | `0.0` | — |
| `/fixture/theta0_sigma_s_per_m` | *absent* | `0.05` | — |
| `/generated_at` | *absent* | `"2026-09-05T15:15:48.745227+00:00"` | — |
| `/ladder/pec_short_s11_mag|false/coarse_delta_per_bin` | *absent* | array[17], SHA256 `89c3d5b8fbfb` | 0–16 |
| `/ladder/pec_short_s11_mag|false/coarse_delta_worst` | *absent* | `0.0019616115385114874` | — |
| `/ladder/pec_short_s11_mag|false/dut` | *absent* | `"pec_short"` | — |
| `/ladder/pec_short_s11_mag|false/excess_worst` | *absent* | `-0.00012294746046903615` | — |
| `/ladder/pec_short_s11_mag|false/fine_delta_per_bin` | *absent* | array[17], SHA256 `d5e2017e1b2e` | 0–16 |
| `/ladder/pec_short_s11_mag|false/fine_delta_worst` | *absent* | `0.0007324939405407571` | — |
| `/ladder/pec_short_s11_mag|false/floor` | *absent* | `0.005` | — |
| `/ladder/pec_short_s11_mag|false/gate_pass` | *absent* | `true` | — |
| `/ladder/pec_short_s11_mag|false/interpretable` | *absent* | `true` | — |
| `/ladder/pec_short_s11_mag|false/kind` | *absent* | `"mag"` | — |
| `/ladder/pec_short_s11_mag|false/lane` | *absent* | `"false"` | — |
| `/ladder/pec_short_s11_mag|false/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/pec_short_s11_mag|false/n_conditioned_bins` | *absent* | `0` | — |
| `/ladder/pec_short_s11_mag|false/observable` | *absent* | `"pec_short_s11_mag"` | — |
| `/ladder/pec_short_s11_mag|false/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/pec_short_s11_mag|false/pinned_richardson_gate` | *absent* | `null` | — |
| `/ladder/pec_short_s11_mag|false/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/pec_short_s11_mag|false/successive_ratio_per_bin` | *absent* | array[17], SHA256 `d5c26fe4843e` | 0–16 |
| `/ladder/pec_short_s11_mag|false/successive_ratio_worst` | *absent* | `null` | — |
| `/ladder/pec_short_s11_mag|false/successive_ratio_worst_bin_hz` | *absent* | `null` | — |
| `/ladder/pec_short_s11_mag|false/values_by_rung/coarse` | *absent* | array[17], SHA256 `35e259256e52` | 0–16 |
| `/ladder/pec_short_s11_mag|false/values_by_rung/fine` | *absent* | array[17], SHA256 `1f8286d83b32` | 0–16 |
| `/ladder/pec_short_s11_mag|false/values_by_rung/mid` | *absent* | array[17], SHA256 `78184944e334` | 0–16 |
| `/ladder/pec_short_s11_mag|false/verdict` | *absent* | `"pass"` | — |
| `/ladder/pec_short_s11_mag|false/worst_bin_hz` | *absent* | `11200000000.0` | — |
| `/ladder/pec_short_s11_mag|flux/coarse_delta_per_bin` | *absent* | array[17], SHA256 `dccbfdfb9365` | 0–16 |
| `/ladder/pec_short_s11_mag|flux/coarse_delta_worst` | *absent* | `2.2445840694995667e-05` | — |
| `/ladder/pec_short_s11_mag|flux/dut` | *absent* | `"pec_short"` | — |
| `/ladder/pec_short_s11_mag|flux/excess_worst` | *absent* | `3.3553877409886468e-06` | — |
| `/ladder/pec_short_s11_mag|flux/fine_delta_per_bin` | *absent* | array[17], SHA256 `133d5fc04678` | 0–16 |
| `/ladder/pec_short_s11_mag|flux/fine_delta_worst` | *absent* | `1.1268000281727808e-05` | — |
| `/ladder/pec_short_s11_mag|flux/floor` | *absent* | `0.005` | — |
| `/ladder/pec_short_s11_mag|flux/gate_pass` | *absent* | `true` | — |
| `/ladder/pec_short_s11_mag|flux/interpretable` | *absent* | `true` | — |
| `/ladder/pec_short_s11_mag|flux/kind` | *absent* | `"mag"` | — |
| `/ladder/pec_short_s11_mag|flux/lane` | *absent* | `"flux"` | — |
| `/ladder/pec_short_s11_mag|flux/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/pec_short_s11_mag|flux/n_conditioned_bins` | *absent* | `0` | — |
| `/ladder/pec_short_s11_mag|flux/observable` | *absent* | `"pec_short_s11_mag"` | — |
| `/ladder/pec_short_s11_mag|flux/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/pec_short_s11_mag|flux/pinned_richardson_gate` | *absent* | `null` | — |
| `/ladder/pec_short_s11_mag|flux/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/pec_short_s11_mag|flux/successive_ratio_per_bin` | *absent* | array[17], SHA256 `d5c26fe4843e` | 0–16 |
| `/ladder/pec_short_s11_mag|flux/successive_ratio_worst` | *absent* | `null` | — |
| `/ladder/pec_short_s11_mag|flux/successive_ratio_worst_bin_hz` | *absent* | `null` | — |
| `/ladder/pec_short_s11_mag|flux/values_by_rung/coarse` | *absent* | array[17], SHA256 `74086e855b01` | 0–16 |
| `/ladder/pec_short_s11_mag|flux/values_by_rung/fine` | *absent* | array[17], SHA256 `d1f2a4b16231` | 0–16 |
| `/ladder/pec_short_s11_mag|flux/values_by_rung/mid` | *absent* | array[17], SHA256 `29f10d233b48` | 0–16 |
| `/ladder/pec_short_s11_mag|flux/verdict` | *absent* | `"pass"` | — |
| `/ladder/pec_short_s11_mag|flux/worst_bin_hz` | *absent* | `11400000000.0` | — |
| `/ladder/pec_short_s11_phase_deg|false/coarse_delta_per_bin` | *absent* | array[17], SHA256 `ad4ae2ce9172` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/coarse_delta_worst` | *absent* | `5.069479200940265` | — |
| `/ladder/pec_short_s11_phase_deg|false/dut` | *absent* | `"pec_short"` | — |
| `/ladder/pec_short_s11_phase_deg|false/excess_worst` | *absent* | `-0.29006723445999677` | — |
| `/ladder/pec_short_s11_phase_deg|false/fine_delta_per_bin` | *absent* | array[17], SHA256 `91be4e82bc30` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/fine_delta_worst` | *absent* | `1.3234968959137063` | — |
| `/ladder/pec_short_s11_phase_deg|false/floor` | *absent* | `1.0` | — |
| `/ladder/pec_short_s11_phase_deg|false/gate_pass` | *absent* | `true` | — |
| `/ladder/pec_short_s11_phase_deg|false/interpretable` | *absent* | `true` | — |
| `/ladder/pec_short_s11_phase_deg|false/kind` | *absent* | `"phase"` | — |
| `/ladder/pec_short_s11_phase_deg|false/lane` | *absent* | `"false"` | — |
| `/ladder/pec_short_s11_phase_deg|false/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/pec_short_s11_phase_deg|false/n_conditioned_bins` | *absent* | `12` | — |
| `/ladder/pec_short_s11_phase_deg|false/observable` | *absent* | `"pec_short_s11_phase_deg"` | — |
| `/ladder/pec_short_s11_phase_deg|false/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/pec_short_s11_phase_deg|false/pinned_richardson_gate` | *absent* | `1.5` | — |
| `/ladder/pec_short_s11_phase_deg|false/pinned_richardson_pair` | *absent* | `"mid-fine"` | — |
| `/ladder/pec_short_s11_phase_deg|false/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/abs_diff_per_bin` | *absent* | array[17], SHA256 `b87e8531fa67` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/estimate_per_bin` | *absent* | array[17], SHA256 `8895f00a38de` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/finer_rung_abs_diff_max` | *absent* | `1.362209900091159` | — |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/max_abs_diff` | *absent* | `3.78862445853268` | — |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/max_abs_diff_bin_hz` | *absent* | `11200000000.0` | — |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/max_abs_diff_continuous` | *absent* | `3.29372942992228` | — |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/oracle_continuous_per_bin` | *absent* | array[17], SHA256 `f67372dd2625` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/oracle_per_bin` | *absent* | array[17], SHA256 `9b43fcab72e4` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/coarse-mid/pair` | *absent* | `[0.00254, 0.00127]` | 0–1 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/abs_diff_per_bin` | *absent* | array[17], SHA256 `ecbb7795708b` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/estimate_per_bin` | *absent* | array[17], SHA256 `41147d60aa1f` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/finer_rung_abs_diff_max` | *absent* | `0.3474765381461325` | — |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/max_abs_diff` | *absent* | `0.9945386406513278` | — |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/max_abs_diff_bin_hz` | *absent* | `11200000000.0` | — |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/max_abs_diff_continuous` | *absent* | `0.8712440208094279` | — |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/oracle_continuous_per_bin` | *absent* | array[17], SHA256 `f67372dd2625` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/oracle_per_bin` | *absent* | array[17], SHA256 `49d3cca3c377` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/richardson/mid-fine/pair` | *absent* | `[0.00127, 0.000635]` | 0–1 |
| `/ladder/pec_short_s11_phase_deg|false/richardson_max_abs_diff` | *absent* | `3.78862445853268` | — |
| `/ladder/pec_short_s11_phase_deg|false/successive_ratio_per_bin` | *absent* | array[17], SHA256 `8a844a1a4ad9` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/successive_ratio_worst` | *absent* | `0.21949557744780837` | — |
| `/ladder/pec_short_s11_phase_deg|false/successive_ratio_worst_bin_hz` | *absent* | `10000000000.0` | — |
| `/ladder/pec_short_s11_phase_deg|false/values_by_rung/coarse` | *absent* | array[17], SHA256 `0ae8c65e151c` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/values_by_rung/fine` | *absent* | array[17], SHA256 `eacbc4477cfd` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/values_by_rung/mid` | *absent* | array[17], SHA256 `8b5fde8a5157` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|false/verdict` | *absent* | `"pass"` | — |
| `/ladder/pec_short_s11_phase_deg|false/worst_bin_hz` | *absent* | `9400000000.0` | — |
| `/ladder/pec_short_s11_phase_deg|flux/coarse_delta_per_bin` | *absent* | array[17], SHA256 `a651ba7b9313` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/coarse_delta_worst` | *absent* | `4.421065304735301` | — |
| `/ladder/pec_short_s11_phase_deg|flux/dut` | *absent* | `"pec_short"` | — |
| `/ladder/pec_short_s11_phase_deg|flux/excess_worst` | *absent* | `-0.41744500951472924` | — |
| `/ladder/pec_short_s11_phase_deg|flux/fine_delta_per_bin` | *absent* | array[17], SHA256 `4a4faace1457` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/fine_delta_worst` | *absent* | `1.1514661537713167` | — |
| `/ladder/pec_short_s11_phase_deg|flux/floor` | *absent* | `1.0` | — |
| `/ladder/pec_short_s11_phase_deg|flux/gate_pass` | *absent* | `true` | — |
| `/ladder/pec_short_s11_phase_deg|flux/interpretable` | *absent* | `true` | — |
| `/ladder/pec_short_s11_phase_deg|flux/kind` | *absent* | `"phase"` | — |
| `/ladder/pec_short_s11_phase_deg|flux/lane` | *absent* | `"flux"` | — |
| `/ladder/pec_short_s11_phase_deg|flux/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/pec_short_s11_phase_deg|flux/n_conditioned_bins` | *absent* | `14` | — |
| `/ladder/pec_short_s11_phase_deg|flux/observable` | *absent* | `"pec_short_s11_phase_deg"` | — |
| `/ladder/pec_short_s11_phase_deg|flux/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/pec_short_s11_phase_deg|flux/pinned_richardson_gate` | *absent* | `1.4` | — |
| `/ladder/pec_short_s11_phase_deg|flux/pinned_richardson_pair` | *absent* | `"mid-fine"` | — |
| `/ladder/pec_short_s11_phase_deg|flux/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/abs_diff_per_bin` | *absent* | array[17], SHA256 `617cb98dc236` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/estimate_per_bin` | *absent* | array[17], SHA256 `42df50c6ecc5` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/finer_rung_abs_diff_max` | *absent* | `1.7581662062081236` | — |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/max_abs_diff` | *absent* | `3.4629616565582917` | — |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/max_abs_diff_bin_hz` | *absent* | `11600000000.0` | — |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/max_abs_diff_continuous` | *absent* | `2.8615382964335514` | — |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/oracle_continuous_per_bin` | *absent* | array[17], SHA256 `f67372dd2625` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/oracle_per_bin` | *absent* | array[17], SHA256 `9b43fcab72e4` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/coarse-mid/pair` | *absent* | `[0.00254, 0.00127]` | 0–1 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/abs_diff_per_bin` | *absent* | array[17], SHA256 `a469d849ec6c` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/estimate_per_bin` | *absent* | array[17], SHA256 `909be9e49c88` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/finer_rung_abs_diff_max` | *absent* | `0.5943116823983629` | — |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/max_abs_diff` | *absent* | `0.8841535398564099` | — |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/max_abs_diff_bin_hz` | *absent* | `11600000000.0` | — |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/max_abs_diff_continuous` | *absent* | `0.7289492022641966` | — |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/oracle_continuous_per_bin` | *absent* | array[17], SHA256 `f67372dd2625` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/oracle_per_bin` | *absent* | array[17], SHA256 `49d3cca3c377` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson/mid-fine/pair` | *absent* | `[0.00127, 0.000635]` | 0–1 |
| `/ladder/pec_short_s11_phase_deg|flux/richardson_max_abs_diff` | *absent* | `3.4629616565582917` | — |
| `/ladder/pec_short_s11_phase_deg|flux/successive_ratio_per_bin` | *absent* | array[17], SHA256 `9c505c7a7beb` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/successive_ratio_worst` | *absent* | `0.2078313572514365` | — |
| `/ladder/pec_short_s11_phase_deg|flux/successive_ratio_worst_bin_hz` | *absent* | `9600000000.0` | — |
| `/ladder/pec_short_s11_phase_deg|flux/values_by_rung/coarse` | *absent* | array[17], SHA256 `507b563e9a85` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/values_by_rung/fine` | *absent* | array[17], SHA256 `85fff9662550` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/values_by_rung/mid` | *absent* | array[17], SHA256 `5c63ff479919` | 0–16 |
| `/ladder/pec_short_s11_phase_deg|flux/verdict` | *absent* | `"pass"` | — |
| `/ladder/pec_short_s11_phase_deg|flux/worst_bin_hz` | *absent* | `9200000000.0` | — |
| `/ladder/slab_s11_mag|false/coarse_delta_per_bin` | *absent* | array[17], SHA256 `e5d73923726e` | 0–16 |
| `/ladder/slab_s11_mag|false/coarse_delta_worst` | *absent* | `0.11117482145668231` | — |
| `/ladder/slab_s11_mag|false/dut` | *absent* | `"slab"` | — |
| `/ladder/slab_s11_mag|false/excess_worst` | *absent* | `-0.03369626203063458` | — |
| `/ladder/slab_s11_mag|false/fine_delta_per_bin` | *absent* | array[17], SHA256 `3e9972a4ea6d` | 0–16 |
| `/ladder/slab_s11_mag|false/fine_delta_worst` | *absent* | `0.02705757203268294` | — |
| `/ladder/slab_s11_mag|false/floor` | *absent* | `0.005` | — |
| `/ladder/slab_s11_mag|false/gate_pass` | *absent* | `true` | — |
| `/ladder/slab_s11_mag|false/interpretable` | *absent* | `true` | — |
| `/ladder/slab_s11_mag|false/kind` | *absent* | `"mag"` | — |
| `/ladder/slab_s11_mag|false/lane` | *absent* | `"false"` | — |
| `/ladder/slab_s11_mag|false/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/slab_s11_mag|false/n_conditioned_bins` | *absent* | `17` | — |
| `/ladder/slab_s11_mag|false/observable` | *absent* | `"slab_s11_mag"` | — |
| `/ladder/slab_s11_mag|false/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/slab_s11_mag|false/pinned_richardson_gate` | *absent* | `0.03` | — |
| `/ladder/slab_s11_mag|false/pinned_richardson_pair` | *absent* | `"mid-fine"` | — |
| `/ladder/slab_s11_mag|false/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/slab_s11_mag|false/richardson/coarse-mid/abs_diff_per_bin` | *absent* | array[17], SHA256 `7ce12cb320b3` | 0–16 |
| `/ladder/slab_s11_mag|false/richardson/coarse-mid/estimate_per_bin` | *absent* | array[17], SHA256 `3157d6c6da34` | 0–16 |
| `/ladder/slab_s11_mag|false/richardson/coarse-mid/finer_rung_abs_diff_max` | *absent* | `0.0396771010605817` | — |
| `/ladder/slab_s11_mag|false/richardson/coarse-mid/max_abs_diff` | *absent* | `0.07368675165660973` | — |
| `/ladder/slab_s11_mag|false/richardson/coarse-mid/max_abs_diff_bin_hz` | *absent* | `8400000000.0` | — |
| `/ladder/slab_s11_mag|false/richardson/coarse-mid/oracle_per_bin` | *absent* | array[17], SHA256 `9b0876e835b5` | 0–16 |
| `/ladder/slab_s11_mag|false/richardson/coarse-mid/pair` | *absent* | `[0.00254, 0.00127]` | 0–1 |
| `/ladder/slab_s11_mag|false/richardson/mid-fine/abs_diff_per_bin` | *absent* | array[17], SHA256 `7abb274ee633` | 0–16 |
| `/ladder/slab_s11_mag|false/richardson/mid-fine/estimate_per_bin` | *absent* | array[17], SHA256 `18fe703b1256` | 0–16 |
| `/ladder/slab_s11_mag|false/richardson/mid-fine/finer_rung_abs_diff_max` | *absent* | `0.012619529027898757` | — |
| `/ladder/slab_s11_mag|false/richardson/mid-fine/max_abs_diff` | *absent* | `0.017589593070640785` | — |
| `/ladder/slab_s11_mag|false/richardson/mid-fine/max_abs_diff_bin_hz` | *absent* | `9400000000.0` | — |
| `/ladder/slab_s11_mag|false/richardson/mid-fine/oracle_per_bin` | *absent* | array[17], SHA256 `9b0876e835b5` | 0–16 |
| `/ladder/slab_s11_mag|false/richardson/mid-fine/pair` | *absent* | `[0.00127, 0.000635]` | 0–1 |
| `/ladder/slab_s11_mag|false/richardson_max_abs_diff` | *absent* | `0.07368675165660973` | — |
| `/ladder/slab_s11_mag|false/successive_ratio_per_bin` | *absent* | array[17], SHA256 `800395eb3ca4` | 0–16 |
| `/ladder/slab_s11_mag|false/successive_ratio_worst` | *absent* | `0.21562371343952016` | — |
| `/ladder/slab_s11_mag|false/successive_ratio_worst_bin_hz` | *absent* | `9200000000.0` | — |
| `/ladder/slab_s11_mag|false/values_by_rung/coarse` | *absent* | array[17], SHA256 `9c9b5671187a` | 0–16 |
| `/ladder/slab_s11_mag|false/values_by_rung/fine` | *absent* | array[17], SHA256 `7369df01c4d9` | 0–16 |
| `/ladder/slab_s11_mag|false/values_by_rung/mid` | *absent* | array[17], SHA256 `03d2f483d245` | 0–16 |
| `/ladder/slab_s11_mag|false/verdict` | *absent* | `"pass"` | — |
| `/ladder/slab_s11_mag|false/worst_bin_hz` | *absent* | `11600000000.0` | — |
| `/ladder/slab_s11_mag|flux/coarse_delta_per_bin` | *absent* | array[17], SHA256 `95d4565f5e86` | 0–16 |
| `/ladder/slab_s11_mag|flux/coarse_delta_worst` | *absent* | `0.11044346967116675` | — |
| `/ladder/slab_s11_mag|flux/dut` | *absent* | `"slab"` | — |
| `/ladder/slab_s11_mag|flux/excess_worst` | *absent* | `-0.032858265057023606` | — |
| `/ladder/slab_s11_mag|flux/fine_delta_per_bin` | *absent* | array[17], SHA256 `22c6a7c513c5` | 0–16 |
| `/ladder/slab_s11_mag|flux/fine_delta_worst` | *absent* | `0.027138452156494713` | — |
| `/ladder/slab_s11_mag|flux/floor` | *absent* | `0.005` | — |
| `/ladder/slab_s11_mag|flux/gate_pass` | *absent* | `true` | — |
| `/ladder/slab_s11_mag|flux/interpretable` | *absent* | `true` | — |
| `/ladder/slab_s11_mag|flux/kind` | *absent* | `"mag"` | — |
| `/ladder/slab_s11_mag|flux/lane` | *absent* | `"flux"` | — |
| `/ladder/slab_s11_mag|flux/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/slab_s11_mag|flux/n_conditioned_bins` | *absent* | `17` | — |
| `/ladder/slab_s11_mag|flux/observable` | *absent* | `"slab_s11_mag"` | — |
| `/ladder/slab_s11_mag|flux/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/slab_s11_mag|flux/pinned_richardson_gate` | *absent* | `0.03` | — |
| `/ladder/slab_s11_mag|flux/pinned_richardson_pair` | *absent* | `"mid-fine"` | — |
| `/ladder/slab_s11_mag|flux/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/slab_s11_mag|flux/richardson/coarse-mid/abs_diff_per_bin` | *absent* | array[17], SHA256 `f772ad0a88b2` | 0–16 |
| `/ladder/slab_s11_mag|flux/richardson/coarse-mid/estimate_per_bin` | *absent* | array[17], SHA256 `4bc0716fabaf` | 0–16 |
| `/ladder/slab_s11_mag|flux/richardson/coarse-mid/finer_rung_abs_diff_max` | *absent* | `0.03616716445747356` | — |
| `/ladder/slab_s11_mag|flux/richardson/coarse-mid/max_abs_diff` | *absent* | `0.07427630521369319` | — |
| `/ladder/slab_s11_mag|flux/richardson/coarse-mid/max_abs_diff_bin_hz` | *absent* | `8400000000.0` | — |
| `/ladder/slab_s11_mag|flux/richardson/coarse-mid/oracle_per_bin` | *absent* | array[17], SHA256 `9b0876e835b5` | 0–16 |
| `/ladder/slab_s11_mag|flux/richardson/coarse-mid/pair` | *absent* | `[0.00254, 0.00127]` | 0–1 |
| `/ladder/slab_s11_mag|flux/richardson/mid-fine/abs_diff_per_bin` | *absent* | array[17], SHA256 `d415fb58095f` | 0–16 |
| `/ladder/slab_s11_mag|flux/richardson/mid-fine/estimate_per_bin` | *absent* | array[17], SHA256 `929ef1b3788a` | 0–16 |
| `/ladder/slab_s11_mag|flux/richardson/mid-fine/finer_rung_abs_diff_max` | *absent* | `0.009028712300978847` | — |
| `/ladder/slab_s11_mag|flux/richardson/mid-fine/max_abs_diff` | *absent* | `0.018109739855515866` | — |
| `/ladder/slab_s11_mag|flux/richardson/mid-fine/max_abs_diff_bin_hz` | *absent* | `8400000000.0` | — |
| `/ladder/slab_s11_mag|flux/richardson/mid-fine/oracle_per_bin` | *absent* | array[17], SHA256 `9b0876e835b5` | 0–16 |
| `/ladder/slab_s11_mag|flux/richardson/mid-fine/pair` | *absent* | `[0.00127, 0.000635]` | 0–1 |
| `/ladder/slab_s11_mag|flux/richardson_max_abs_diff` | *absent* | `0.07427630521369319` | — |
| `/ladder/slab_s11_mag|flux/successive_ratio_per_bin` | *absent* | array[17], SHA256 `f4d271ceac01` | 0–16 |
| `/ladder/slab_s11_mag|flux/successive_ratio_worst` | *absent* | `0.22582930324922043` | — |
| `/ladder/slab_s11_mag|flux/successive_ratio_worst_bin_hz` | *absent* | `9000000000.0` | — |
| `/ladder/slab_s11_mag|flux/values_by_rung/coarse` | *absent* | array[17], SHA256 `a3cceb3640b1` | 0–16 |
| `/ladder/slab_s11_mag|flux/values_by_rung/fine` | *absent* | array[17], SHA256 `d9a4aa1d2393` | 0–16 |
| `/ladder/slab_s11_mag|flux/values_by_rung/mid` | *absent* | array[17], SHA256 `d575486ad5ef` | 0–16 |
| `/ladder/slab_s11_mag|flux/verdict` | *absent* | `"pass"` | — |
| `/ladder/slab_s11_mag|flux/worst_bin_hz` | *absent* | `11600000000.0` | — |
| `/ladder/slab_s21_mag|false/coarse_delta_per_bin` | *absent* | array[17], SHA256 `ec70ab9f3c72` | 0–16 |
| `/ladder/slab_s21_mag|false/coarse_delta_worst` | *absent* | `0.08563210102973062` | — |
| `/ladder/slab_s21_mag|false/dut` | *absent* | `"slab"` | — |
| `/ladder/slab_s21_mag|false/excess_worst` | *absent* | `-0.02714464246255155` | — |
| `/ladder/slab_s21_mag|false/fine_delta_per_bin` | *absent* | array[17], SHA256 `c172410faffb` | 0–16 |
| `/ladder/slab_s21_mag|false/fine_delta_worst` | *absent* | `0.01989948501715688` | — |
| `/ladder/slab_s21_mag|false/floor` | *absent* | `0.005` | — |
| `/ladder/slab_s21_mag|false/gate_pass` | *absent* | `true` | — |
| `/ladder/slab_s21_mag|false/interpretable` | *absent* | `true` | — |
| `/ladder/slab_s21_mag|false/kind` | *absent* | `"mag"` | — |
| `/ladder/slab_s21_mag|false/lane` | *absent* | `"false"` | — |
| `/ladder/slab_s21_mag|false/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/slab_s21_mag|false/n_conditioned_bins` | *absent* | `17` | — |
| `/ladder/slab_s21_mag|false/observable` | *absent* | `"slab_s21_mag"` | — |
| `/ladder/slab_s21_mag|false/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/slab_s21_mag|false/pinned_richardson_gate` | *absent* | `0.02` | — |
| `/ladder/slab_s21_mag|false/pinned_richardson_pair` | *absent* | `"mid-fine"` | — |
| `/ladder/slab_s21_mag|false/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/slab_s21_mag|false/richardson/coarse-mid/abs_diff_per_bin` | *absent* | array[17], SHA256 `617cf33331ef` | 0–16 |
| `/ladder/slab_s21_mag|false/richardson/coarse-mid/estimate_per_bin` | *absent* | array[17], SHA256 `e07a6f0364af` | 0–16 |
| `/ladder/slab_s21_mag|false/richardson/coarse-mid/finer_rung_abs_diff_max` | *absent* | `0.02725030493367575` | — |
| `/ladder/slab_s21_mag|false/richardson/coarse-mid/max_abs_diff` | *absent* | `0.06042994679365454` | — |
| `/ladder/slab_s21_mag|false/richardson/coarse-mid/max_abs_diff_bin_hz` | *absent* | `9400000000.0` | — |
| `/ladder/slab_s21_mag|false/richardson/coarse-mid/oracle_per_bin` | *absent* | array[17], SHA256 `c60cd112d2d7` | 0–16 |
| `/ladder/slab_s21_mag|false/richardson/coarse-mid/pair` | *absent* | `[0.00254, 0.00127]` | 0–1 |
| `/ladder/slab_s21_mag|false/richardson/mid-fine/abs_diff_per_bin` | *absent* | array[17], SHA256 `72fff1d54503` | 0–16 |
| `/ladder/slab_s21_mag|false/richardson/mid-fine/estimate_per_bin` | *absent* | array[17], SHA256 `ffdf14bfb480` | 0–16 |
| `/ladder/slab_s21_mag|false/richardson/mid-fine/finer_rung_abs_diff_max` | *absent* | `0.007350819916518869` | — |
| `/ladder/slab_s21_mag|false/richardson/mid-fine/max_abs_diff` | *absent* | `0.01311100865645587` | — |
| `/ladder/slab_s21_mag|false/richardson/mid-fine/max_abs_diff_bin_hz` | *absent* | `9600000000.0` | — |
| `/ladder/slab_s21_mag|false/richardson/mid-fine/oracle_per_bin` | *absent* | array[17], SHA256 `c60cd112d2d7` | 0–16 |
| `/ladder/slab_s21_mag|false/richardson/mid-fine/pair` | *absent* | `[0.00127, 0.000635]` | 0–1 |
| `/ladder/slab_s21_mag|false/richardson_max_abs_diff` | *absent* | `0.06042994679365454` | — |
| `/ladder/slab_s21_mag|false/successive_ratio_per_bin` | *absent* | array[17], SHA256 `0bf9a449a2be` | 0–16 |
| `/ladder/slab_s21_mag|false/successive_ratio_worst` | *absent* | `0.186396665640836` | — |
| `/ladder/slab_s21_mag|false/successive_ratio_worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/ladder/slab_s21_mag|false/values_by_rung/coarse` | *absent* | array[17], SHA256 `11068a535b8e` | 0–16 |
| `/ladder/slab_s21_mag|false/values_by_rung/fine` | *absent* | array[17], SHA256 `733442c41ffa` | 0–16 |
| `/ladder/slab_s21_mag|false/values_by_rung/mid` | *absent* | array[17], SHA256 `d045283ff07f` | 0–16 |
| `/ladder/slab_s21_mag|false/verdict` | *absent* | `"pass"` | — |
| `/ladder/slab_s21_mag|false/worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/ladder/slab_s21_mag|flux/coarse_delta_per_bin` | *absent* | array[17], SHA256 `0e419491f56f` | 0–16 |
| `/ladder/slab_s21_mag|flux/coarse_delta_worst` | *absent* | `0.07743036729310071` | — |
| `/ladder/slab_s21_mag|flux/dut` | *absent* | `"slab"` | — |
| `/ladder/slab_s21_mag|flux/excess_worst` | *absent* | `-0.026899447756018158` | — |
| `/ladder/slab_s21_mag|flux/fine_delta_per_bin` | *absent* | array[17], SHA256 `1ab92b4a9613` | 0–16 |
| `/ladder/slab_s21_mag|flux/fine_delta_worst` | *absent* | `0.0172018513654989` | — |
| `/ladder/slab_s21_mag|flux/floor` | *absent* | `0.005` | — |
| `/ladder/slab_s21_mag|flux/gate_pass` | *absent* | `true` | — |
| `/ladder/slab_s21_mag|flux/interpretable` | *absent* | `true` | — |
| `/ladder/slab_s21_mag|flux/kind` | *absent* | `"mag"` | — |
| `/ladder/slab_s21_mag|flux/lane` | *absent* | `"flux"` | — |
| `/ladder/slab_s21_mag|flux/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/slab_s21_mag|flux/n_conditioned_bins` | *absent* | `17` | — |
| `/ladder/slab_s21_mag|flux/observable` | *absent* | `"slab_s21_mag"` | — |
| `/ladder/slab_s21_mag|flux/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/slab_s21_mag|flux/pinned_richardson_gate` | *absent* | `0.02` | — |
| `/ladder/slab_s21_mag|flux/pinned_richardson_pair` | *absent* | `"mid-fine"` | — |
| `/ladder/slab_s21_mag|flux/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/slab_s21_mag|flux/richardson/coarse-mid/abs_diff_per_bin` | *absent* | array[17], SHA256 `ecf794264129` | 0–16 |
| `/ladder/slab_s21_mag|flux/richardson/coarse-mid/estimate_per_bin` | *absent* | array[17], SHA256 `b4ba97874442` | 0–16 |
| `/ladder/slab_s21_mag|flux/richardson/coarse-mid/finer_rung_abs_diff_max` | *absent* | `0.02271557305158267` | — |
| `/ladder/slab_s21_mag|flux/richardson/coarse-mid/max_abs_diff` | *absent* | `0.05618207067531711` | — |
| `/ladder/slab_s21_mag|flux/richardson/coarse-mid/max_abs_diff_bin_hz` | *absent* | `9400000000.0` | — |
| `/ladder/slab_s21_mag|flux/richardson/coarse-mid/oracle_per_bin` | *absent* | array[17], SHA256 `c60cd112d2d7` | 0–16 |
| `/ladder/slab_s21_mag|flux/richardson/coarse-mid/pair` | *absent* | `[0.00254, 0.00127]` | 0–1 |
| `/ladder/slab_s21_mag|flux/richardson/mid-fine/abs_diff_per_bin` | *absent* | array[17], SHA256 `ac652d0f4373` | 0–16 |
| `/ladder/slab_s21_mag|flux/richardson/mid-fine/estimate_per_bin` | *absent* | array[17], SHA256 `da400575c9aa` | 0–16 |
| `/ladder/slab_s21_mag|flux/richardson/mid-fine/finer_rung_abs_diff_max` | *absent* | `0.005513721686083772` | — |
| `/ladder/slab_s21_mag|flux/richardson/mid-fine/max_abs_diff` | *absent* | `0.011688129679415127` | — |
| `/ladder/slab_s21_mag|flux/richardson/mid-fine/max_abs_diff_bin_hz` | *absent* | `9600000000.0` | — |
| `/ladder/slab_s21_mag|flux/richardson/mid-fine/oracle_per_bin` | *absent* | array[17], SHA256 `c60cd112d2d7` | 0–16 |
| `/ladder/slab_s21_mag|flux/richardson/mid-fine/pair` | *absent* | `[0.00127, 0.000635]` | 0–1 |
| `/ladder/slab_s21_mag|flux/richardson_max_abs_diff` | *absent* | `0.05618207067531711` | — |
| `/ladder/slab_s21_mag|flux/successive_ratio_per_bin` | *absent* | array[17], SHA256 `7d16455600a5` | 0–16 |
| `/ladder/slab_s21_mag|flux/successive_ratio_worst` | *absent* | `0.18230098282124177` | — |
| `/ladder/slab_s21_mag|flux/successive_ratio_worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/ladder/slab_s21_mag|flux/values_by_rung/coarse` | *absent* | array[17], SHA256 `c5fcf56af69f` | 0–16 |
| `/ladder/slab_s21_mag|flux/values_by_rung/fine` | *absent* | array[17], SHA256 `026320bbf1e0` | 0–16 |
| `/ladder/slab_s21_mag|flux/values_by_rung/mid` | *absent* | array[17], SHA256 `92e66b321de2` | 0–16 |
| `/ladder/slab_s21_mag|flux/verdict` | *absent* | `"pass"` | — |
| `/ladder/slab_s21_mag|flux/worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/ladder/slab_s21_phase_deg|false/coarse_delta_per_bin` | *absent* | array[17], SHA256 `591cd51ad09e` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/coarse_delta_worst` | *absent* | `12.917567822450877` | — |
| `/ladder/slab_s21_phase_deg|false/dut` | *absent* | `"slab"` | — |
| `/ladder/slab_s21_phase_deg|false/excess_worst` | *absent* | `-6.538156015817776` | — |
| `/ladder/slab_s21_phase_deg|false/fine_delta_per_bin` | *absent* | array[17], SHA256 `0efae909193f` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/fine_delta_worst` | *absent* | `3.0378610690196997` | — |
| `/ladder/slab_s21_phase_deg|false/floor` | *absent* | `1.0` | — |
| `/ladder/slab_s21_phase_deg|false/gate_pass` | *absent* | `true` | — |
| `/ladder/slab_s21_phase_deg|false/interpretable` | *absent* | `true` | — |
| `/ladder/slab_s21_phase_deg|false/kind` | *absent* | `"phase"` | — |
| `/ladder/slab_s21_phase_deg|false/lane` | *absent* | `"false"` | — |
| `/ladder/slab_s21_phase_deg|false/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/slab_s21_phase_deg|false/n_conditioned_bins` | *absent* | `17` | — |
| `/ladder/slab_s21_phase_deg|false/observable` | *absent* | `"slab_s21_phase_deg"` | — |
| `/ladder/slab_s21_phase_deg|false/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/slab_s21_phase_deg|false/pinned_richardson_gate` | *absent* | `3.1` | — |
| `/ladder/slab_s21_phase_deg|false/pinned_richardson_pair` | *absent* | `"mid-fine"` | — |
| `/ladder/slab_s21_phase_deg|false/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/slab_s21_phase_deg|false/richardson/coarse-mid/abs_diff_per_bin` | *absent* | array[17], SHA256 `28fd35f54a4f` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/richardson/coarse-mid/estimate_per_bin` | *absent* | array[17], SHA256 `1cdfb0da69c2` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/richardson/coarse-mid/finer_rung_abs_diff_max` | *absent* | `4.024594628056193` | — |
| `/ladder/slab_s21_phase_deg|false/richardson/coarse-mid/max_abs_diff` | *absent* | `8.892973194394685` | — |
| `/ladder/slab_s21_phase_deg|false/richardson/coarse-mid/max_abs_diff_bin_hz` | *absent* | `11600000000.0` | — |
| `/ladder/slab_s21_phase_deg|false/richardson/coarse-mid/oracle_per_bin` | *absent* | array[17], SHA256 `77896242b465` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/richardson/coarse-mid/pair` | *absent* | `[0.00254, 0.00127]` | 0–1 |
| `/ladder/slab_s21_phase_deg|false/richardson/mid-fine/abs_diff_per_bin` | *absent* | array[17], SHA256 `449bdd91da94` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/richardson/mid-fine/estimate_per_bin` | *absent* | array[17], SHA256 `279e4e7023b6` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/richardson/mid-fine/finer_rung_abs_diff_max` | *absent* | `0.9867335590364928` | — |
| `/ladder/slab_s21_phase_deg|false/richardson/mid-fine/max_abs_diff` | *absent* | `2.051127509983207` | — |
| `/ladder/slab_s21_phase_deg|false/richardson/mid-fine/max_abs_diff_bin_hz` | *absent* | `11600000000.0` | — |
| `/ladder/slab_s21_phase_deg|false/richardson/mid-fine/oracle_per_bin` | *absent* | array[17], SHA256 `77896242b465` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/richardson/mid-fine/pair` | *absent* | `[0.00127, 0.000635]` | 0–1 |
| `/ladder/slab_s21_phase_deg|false/richardson_max_abs_diff` | *absent* | `8.892973194394685` | — |
| `/ladder/slab_s21_phase_deg|false/successive_ratio_per_bin` | *absent* | array[17], SHA256 `4208e2709aeb` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/successive_ratio_worst` | *absent* | `0.2241973991793538` | — |
| `/ladder/slab_s21_phase_deg|false/successive_ratio_worst_bin_hz` | *absent* | `9000000000.0` | — |
| `/ladder/slab_s21_phase_deg|false/values_by_rung/coarse` | *absent* | array[17], SHA256 `d61ba8863255` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/values_by_rung/fine` | *absent* | array[17], SHA256 `6566af7239c6` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/values_by_rung/mid` | *absent* | array[17], SHA256 `4003910ea428` | 0–16 |
| `/ladder/slab_s21_phase_deg|false/verdict` | *absent* | `"pass"` | — |
| `/ladder/slab_s21_phase_deg|false/worst_bin_hz` | *absent* | `10000000000.0` | — |
| `/ladder/slab_s21_phase_deg|flux/coarse_delta_per_bin` | *absent* | array[17], SHA256 `b8a361f61543` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/coarse_delta_worst` | *absent* | `12.15982157802349` | — |
| `/ladder/slab_s21_phase_deg|flux/dut` | *absent* | `"slab"` | — |
| `/ladder/slab_s21_phase_deg|flux/excess_worst` | *absent* | `-6.4209519927740075` | — |
| `/ladder/slab_s21_phase_deg|flux/fine_delta_per_bin` | *absent* | array[17], SHA256 `151f02c57bc7` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/fine_delta_worst` | *absent* | `2.8482988769637636` | — |
| `/ladder/slab_s21_phase_deg|flux/floor` | *absent* | `1.0` | — |
| `/ladder/slab_s21_phase_deg|flux/gate_pass` | *absent* | `true` | — |
| `/ladder/slab_s21_phase_deg|flux/interpretable` | *absent* | `true` | — |
| `/ladder/slab_s21_phase_deg|flux/kind` | *absent* | `"phase"` | — |
| `/ladder/slab_s21_phase_deg|flux/lane` | *absent* | `"flux"` | — |
| `/ladder/slab_s21_phase_deg|flux/monotone_fraction_of_bins` | *absent* | `1.0` | — |
| `/ladder/slab_s21_phase_deg|flux/n_conditioned_bins` | *absent* | `17` | — |
| `/ladder/slab_s21_phase_deg|flux/observable` | *absent* | `"slab_s21_phase_deg"` | — |
| `/ladder/slab_s21_phase_deg|flux/pinned_monotone_fraction_min` | *absent* | `0.66` | — |
| `/ladder/slab_s21_phase_deg|flux/pinned_richardson_gate` | *absent* | `2.9` | — |
| `/ladder/slab_s21_phase_deg|flux/pinned_richardson_pair` | *absent* | `"mid-fine"` | — |
| `/ladder/slab_s21_phase_deg|flux/ratio_window` | *absent* | `[0.15, 0.7]` | 0–1 |
| `/ladder/slab_s21_phase_deg|flux/richardson/coarse-mid/abs_diff_per_bin` | *absent* | array[17], SHA256 `90591b6a151d` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/richardson/coarse-mid/estimate_per_bin` | *absent* | array[17], SHA256 `304f014bcb37` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/richardson/coarse-mid/finer_rung_abs_diff_max` | *absent* | `3.7839370223476547` | — |
| `/ladder/slab_s21_phase_deg|flux/richardson/coarse-mid/max_abs_diff` | *absent* | `8.375884555675835` | — |
| `/ladder/slab_s21_phase_deg|flux/richardson/coarse-mid/max_abs_diff_bin_hz` | *absent* | `11600000000.0` | — |
| `/ladder/slab_s21_phase_deg|flux/richardson/coarse-mid/oracle_per_bin` | *absent* | array[17], SHA256 `77896242b465` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/richardson/coarse-mid/pair` | *absent* | `[0.00254, 0.00127]` | 0–1 |
| `/ladder/slab_s21_phase_deg|flux/richardson/mid-fine/abs_diff_per_bin` | *absent* | array[17], SHA256 `8e495cf45be7` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/richardson/mid-fine/estimate_per_bin` | *absent* | array[17], SHA256 `4cabd442d7a3` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/richardson/mid-fine/finer_rung_abs_diff_max` | *absent* | `0.9356381453838908` | — |
| `/ladder/slab_s21_phase_deg|flux/richardson/mid-fine/max_abs_diff` | *absent* | `1.912660731579873` | — |
| `/ladder/slab_s21_phase_deg|flux/richardson/mid-fine/max_abs_diff_bin_hz` | *absent* | `11600000000.0` | — |
| `/ladder/slab_s21_phase_deg|flux/richardson/mid-fine/oracle_per_bin` | *absent* | array[17], SHA256 `77896242b465` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/richardson/mid-fine/pair` | *absent* | `[0.00127, 0.000635]` | 0–1 |
| `/ladder/slab_s21_phase_deg|flux/richardson_max_abs_diff` | *absent* | `8.375884555675835` | — |
| `/ladder/slab_s21_phase_deg|flux/successive_ratio_per_bin` | *absent* | array[17], SHA256 `1a8a5aacd38c` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/successive_ratio_worst` | *absent* | `0.22922564757092323` | — |
| `/ladder/slab_s21_phase_deg|flux/successive_ratio_worst_bin_hz` | *absent* | `8800000000.0` | — |
| `/ladder/slab_s21_phase_deg|flux/values_by_rung/coarse` | *absent* | array[17], SHA256 `40e1681d036a` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/values_by_rung/fine` | *absent* | array[17], SHA256 `d520ec7e90d7` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/values_by_rung/mid` | *absent* | array[17], SHA256 `7872b1ea4526` | 0–16 |
| `/ladder/slab_s21_phase_deg|flux/verdict` | *absent* | `"pass"` | — |
| `/ladder/slab_s21_phase_deg|flux/worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/legs_rung` | *absent* | `"fine"` | — |
| `/physics_gates/claims_rung` | *absent* | `"fine"` | — |
| `/physics_gates/pec_short|coarse|false/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/pec_short|coarse|false/column_power_max` | *absent* | `1.0047155530599599` | — |
| `/physics_gates/pec_short|coarse|false/gated` | *absent* | `false` | — |
| `/physics_gates/pec_short|coarse|false/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/pec_short|coarse|false/power_closure_max` | *absent* | `0.00594958252947797` | — |
| `/physics_gates/pec_short|coarse|false/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|coarse|false/reciprocity_complex_max` | *absent* | `1.468931504282033e-21` | — |
| `/physics_gates/pec_short|coarse|false/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|coarse|false/reciprocity_mag_mean` | *absent* | `5.856469615409605e-10` | — |
| `/physics_gates/pec_short|coarse|flux/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/pec_short|coarse|flux/column_power_max` | *absent* | `1.0000876917090205` | — |
| `/physics_gates/pec_short|coarse|flux/gated` | *absent* | `false` | — |
| `/physics_gates/pec_short|coarse|flux/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/pec_short|coarse|flux/power_closure_max` | *absent* | `8.769170902045431e-05` | — |
| `/physics_gates/pec_short|coarse|flux/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|coarse|flux/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|coarse|flux/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|coarse|flux/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|fine|false/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/pec_short|fine|false/column_power_max` | *absent* | `1.0002866706244224` | — |
| `/physics_gates/pec_short|fine|false/gated` | *absent* | `true` | — |
| `/physics_gates/pec_short|fine|false/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/pec_short|fine|false/power_closure_max` | *absent* | `0.0005701660822906574` | — |
| `/physics_gates/pec_short|fine|false/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|fine|false/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|fine|false/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|fine|false/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|fine|flux/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/pec_short|fine|flux/column_power_max` | *absent* | `1.0000131118209492` | — |
| `/physics_gates/pec_short|fine|flux/gated` | *absent* | `true` | — |
| `/physics_gates/pec_short|fine|flux/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/pec_short|fine|flux/power_closure_max` | *absent* | `1.3111820949207598e-05` | — |
| `/physics_gates/pec_short|fine|flux/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|fine|flux/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|fine|flux/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|fine|flux/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|mid|false/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/pec_short|mid|false/column_power_max` | *absent* | `1.0011618775172906` | — |
| `/physics_gates/pec_short|mid|false/gated` | *absent* | `false` | — |
| `/physics_gates/pec_short|mid|false/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/pec_short|mid|false/power_closure_max` | *absent* | `0.002034199713250362` | — |
| `/physics_gates/pec_short|mid|false/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|mid|false/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|mid|false/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|mid|false/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|mid|flux/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/pec_short|mid|flux/column_power_max` | *absent* | `1.0000204614627888` | — |
| `/physics_gates/pec_short|mid|flux/gated` | *absent* | `false` | — |
| `/physics_gates/pec_short|mid|flux/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/pec_short|mid|flux/power_closure_max` | *absent* | `2.0461462788778917e-05` | — |
| `/physics_gates/pec_short|mid|flux/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|mid|flux/reciprocity_complex_max` | *absent* | `0.0` | — |
| `/physics_gates/pec_short|mid|flux/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/pec_short|mid|flux/reciprocity_mag_mean` | *absent* | `0.0` | — |
| `/physics_gates/settling_all_below_minus_40_db` | *absent* | `true` | — |
| `/physics_gates/slab|coarse|false/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/slab|coarse|false/column_power_max` | *absent* | `1.0071137369445617` | — |
| `/physics_gates/slab|coarse|false/gated` | *absent* | `false` | — |
| `/physics_gates/slab|coarse|false/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/slab|coarse|false/power_closure_max` | *absent* | `0.007113736944561744` | — |
| `/physics_gates/slab|coarse|false/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|coarse|false/reciprocity_complex_max` | *absent* | `0.030861195594530717` | — |
| `/physics_gates/slab|coarse|false/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|coarse|false/reciprocity_mag_mean` | *absent* | `0.013225390084792519` | — |
| `/physics_gates/slab|coarse|flux/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/slab|coarse|flux/column_power_max` | *absent* | `1.0001005294479495` | — |
| `/physics_gates/slab|coarse|flux/gated` | *absent* | `false` | — |
| `/physics_gates/slab|coarse|flux/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/slab|coarse|flux/power_closure_max` | *absent* | `0.00011017143247071814` | — |
| `/physics_gates/slab|coarse|flux/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|coarse|flux/reciprocity_complex_max` | *absent* | `0.0014405302999060377` | — |
| `/physics_gates/slab|coarse|flux/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|coarse|flux/reciprocity_mag_mean` | *absent* | `1.458768989854963e-06` | — |
| `/physics_gates/slab|fine|false/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/slab|fine|false/column_power_max` | *absent* | `1.0003528195646192` | — |
| `/physics_gates/slab|fine|false/gated` | *absent* | `true` | — |
| `/physics_gates/slab|fine|false/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/slab|fine|false/power_closure_max` | *absent* | `0.00037092525650594954` | — |
| `/physics_gates/slab|fine|false/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|fine|false/reciprocity_complex_max` | *absent* | `0.004806417344254802` | — |
| `/physics_gates/slab|fine|false/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|fine|false/reciprocity_mag_mean` | *absent* | `0.0022506870363776755` | — |
| `/physics_gates/slab|fine|flux/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/slab|fine|flux/column_power_max` | *absent* | `1.0000092485237313` | — |
| `/physics_gates/slab|fine|flux/gated` | *absent* | `true` | — |
| `/physics_gates/slab|fine|flux/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/slab|fine|flux/power_closure_max` | *absent* | `9.248523731297809e-06` | — |
| `/physics_gates/slab|fine|flux/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|fine|flux/reciprocity_complex_max` | *absent* | `2.730281774495818e-05` | — |
| `/physics_gates/slab|fine|flux/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|fine|flux/reciprocity_mag_mean` | *absent* | `6.022391006592541e-07` | — |
| `/physics_gates/slab|mid|false/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/slab|mid|false/column_power_max` | *absent* | `1.0015357753132264` | — |
| `/physics_gates/slab|mid|false/gated` | *absent* | `false` | — |
| `/physics_gates/slab|mid|false/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/slab|mid|false/power_closure_max` | *absent* | `0.0015357753132263507` | — |
| `/physics_gates/slab|mid|false/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|mid|false/reciprocity_complex_max` | *absent* | `0.010866879557759017` | — |
| `/physics_gates/slab|mid|false/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|mid|false/reciprocity_mag_mean` | *absent* | `0.004589836602120118` | — |
| `/physics_gates/slab|mid|flux/column_power_gate` | *absent* | `1.02` | — |
| `/physics_gates/slab|mid|flux/column_power_max` | *absent* | `1.0000368078695865` | — |
| `/physics_gates/slab|mid|flux/gated` | *absent* | `false` | — |
| `/physics_gates/slab|mid|flux/power_closure_gate` | *absent* | `"report-only (WP3)"` | — |
| `/physics_gates/slab|mid|flux/power_closure_max` | *absent* | `3.6807869586485964e-05` | — |
| `/physics_gates/slab|mid|flux/reciprocity_complex_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|mid|flux/reciprocity_complex_max` | *absent* | `0.00021136326370809695` | — |
| `/physics_gates/slab|mid|flux/reciprocity_mag_gate` | *absent* | `0.01` | — |
| `/physics_gates/slab|mid|flux/reciprocity_mag_mean` | *absent* | `5.676502369878185e-07` | — |
| `/pins/gradient_invariance_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/pins/gradient_invariance_gate` | *absent* | `0.001` | — |
| `/pins/gradient_quantum` | *absent* | `1000` | — |
| `/pins/monotone_quantum` | *absent* | `100` | — |
| `/pins/policy` | *absent* | `"tests/_gate_policy.py gate_from_envelope (x ENVELOPE_GATE_MULTIPLIER, rounded up); lower bounds rounded down by the same multiplier"` | — |
| `/pins/richardson_quantum/mag` | *absent* | `100` | — |
| `/pins/richardson_quantum/phase` | *absent* | `10` | — |
| `/plane_shift/cheap_refute/abs_s_still_invariant` | *absent* | `true` | — |
| `/plane_shift/cheap_refute/per_case/0/dut` | *absent* | `"pec_short"` | — |
| `/plane_shift/cheap_refute/per_case/0/entries_measurable` | *absent* | `["S11", "S22"]` | 0–1 |
| `/plane_shift/cheap_refute/per_case/0/lane` | *absent* | `"false"` | — |
| `/plane_shift/cheap_refute/per_case/0/provenance/commit` | *absent* | `"f914a7caf1ff8c63cac6f5f8c975b7f9f420a0c7"` | — |
| `/plane_shift/cheap_refute/per_case/0/provenance/jax_default_backend` | *absent* | `"gpu"` | — |
| `/plane_shift/cheap_refute/per_case/0/provenance/run_id` | *absent* | `"369367258638"` | — |
| `/plane_shift/cheap_refute/per_case/0/provenance/run_lane` | *absent* | `"vessl"` | — |
| `/plane_shift/cheap_refute/per_case/0/resid_yee_per_entry/S11` | *absent* | `178.0043869385972` | — |
| `/plane_shift/cheap_refute/per_case/0/resid_yee_per_entry/S12` | *absent* | `null` | — |
| `/plane_shift/cheap_refute/per_case/0/resid_yee_per_entry/S21` | *absent* | `null` | — |
| `/plane_shift/cheap_refute/per_case/0/resid_yee_per_entry/S22` | *absent* | `117.26918474585037` | — |
| `/plane_shift/cheap_refute/per_case/1/dut` | *absent* | `"pec_short"` | — |
| `/plane_shift/cheap_refute/per_case/1/entries_measurable` | *absent* | `["S11", "S22"]` | 0–1 |
| `/plane_shift/cheap_refute/per_case/1/lane` | *absent* | `"flux"` | — |
| `/plane_shift/cheap_refute/per_case/1/provenance/commit` | *absent* | `"f914a7caf1ff8c63cac6f5f8c975b7f9f420a0c7"` | — |
| `/plane_shift/cheap_refute/per_case/1/provenance/jax_default_backend` | *absent* | `"gpu"` | — |
| `/plane_shift/cheap_refute/per_case/1/provenance/run_id` | *absent* | `"369367258638"` | — |
| `/plane_shift/cheap_refute/per_case/1/provenance/run_lane` | *absent* | `"vessl"` | — |
| `/plane_shift/cheap_refute/per_case/1/resid_yee_per_entry/S11` | *absent* | `178.00438631363247` | — |
| `/plane_shift/cheap_refute/per_case/1/resid_yee_per_entry/S12` | *absent* | `null` | — |
| `/plane_shift/cheap_refute/per_case/1/resid_yee_per_entry/S21` | *absent* | `null` | — |
| `/plane_shift/cheap_refute/per_case/1/resid_yee_per_entry/S22` | *absent* | `117.26918777895827` | — |
| `/plane_shift/cheap_refute/per_case/2/dut` | *absent* | `"slab"` | — |
| `/plane_shift/cheap_refute/per_case/2/entries_measurable` | *absent* | `["S11", "S22", "S21", "S12"]` | 0–3 |
| `/plane_shift/cheap_refute/per_case/2/lane` | *absent* | `"false"` | — |
| `/plane_shift/cheap_refute/per_case/2/provenance/commit` | *absent* | `"f914a7caf1ff8c63cac6f5f8c975b7f9f420a0c7"` | — |
| `/plane_shift/cheap_refute/per_case/2/provenance/jax_default_backend` | *absent* | `"gpu"` | — |
| `/plane_shift/cheap_refute/per_case/2/provenance/run_id` | *absent* | `"369367258638"` | — |
| `/plane_shift/cheap_refute/per_case/2/provenance/run_lane` | *absent* | `"vessl"` | — |
| `/plane_shift/cheap_refute/per_case/2/resid_yee_per_entry/S11` | *absent* | `178.00438403371078` | — |
| `/plane_shift/cheap_refute/per_case/2/resid_yee_per_entry/S12` | *absent* | `175.90377527745972` | — |
| `/plane_shift/cheap_refute/per_case/2/resid_yee_per_entry/S21` | *absent* | `175.90377526493708` | — |
| `/plane_shift/cheap_refute/per_case/2/resid_yee_per_entry/S22` | *absent* | `117.26918622923573` | — |
| `/plane_shift/cheap_refute/per_case/3/dut` | *absent* | `"slab"` | — |
| `/plane_shift/cheap_refute/per_case/3/entries_measurable` | *absent* | `["S11", "S22", "S21", "S12"]` | 0–3 |
| `/plane_shift/cheap_refute/per_case/3/lane` | *absent* | `"flux"` | — |
| `/plane_shift/cheap_refute/per_case/3/provenance/commit` | *absent* | `"f914a7caf1ff8c63cac6f5f8c975b7f9f420a0c7"` | — |
| `/plane_shift/cheap_refute/per_case/3/provenance/jax_default_backend` | *absent* | `"gpu"` | — |
| `/plane_shift/cheap_refute/per_case/3/provenance/run_id` | *absent* | `"369367258638"` | — |
| `/plane_shift/cheap_refute/per_case/3/provenance/run_lane` | *absent* | `"vessl"` | — |
| `/plane_shift/cheap_refute/per_case/3/resid_yee_per_entry/S11` | *absent* | `178.00439470553806` | — |
| `/plane_shift/cheap_refute/per_case/3/resid_yee_per_entry/S12` | *absent* | `175.90377908689524` | — |
| `/plane_shift/cheap_refute/per_case/3/resid_yee_per_entry/S21` | *absent* | `175.90377578383527` | — |
| `/plane_shift/cheap_refute/per_case/3/resid_yee_per_entry/S22` | *absent* | `117.26918201984941` | — |
| `/plane_shift/cheap_refute/refute` | *absent* | `"local copy of _shift_modal_waves with the shift sign flipped"` | — |
| `/plane_shift/cheap_refute/resid_yee_max_over_entries` | *absent* | `178.00439470553806` | — |
| `/plane_shift/cheap_refute/resid_yee_min_over_entries` | *absent* | `117.26918201984941` | — |
| `/plane_shift/cheap_refute/rotation_gate_would_pass` | *absent* | `false` | — |
| `/plane_shift/cheap_refute/rung` | *absent* | `"coarse"` | — |
| `/plane_shift/pec_short|false/abs_s_allclose` | *absent* | `true` | — |
| `/plane_shift/pec_short|false/abs_s_max_diff` | *absent* | `1.760519436899699e-07` | — |
| `/plane_shift/pec_short|false/base_source` | *absent* | `"cell__pec_short__fine__false.json"` | — |
| `/plane_shift/pec_short|false/dut` | *absent* | `"pec_short"` | — |
| `/plane_shift/pec_short|false/dx_m` | *absent* | `0.000635` | — |
| `/plane_shift/pec_short|false/entries_measurable` | *absent* | `["S11", "S22"]` | 0–1 |
| `/plane_shift/pec_short|false/fc_port_hz` | *absent* | `6555059929.275057` | — |
| `/plane_shift/pec_short|false/fc_predeclared_hz` | *absent* | `6557140376.203117` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/from_objectives` | *absent* | `["re_s11", "im_s11"]` | 0–1 |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/kind` | *absent* | `"complex"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/phi_measured_deg` | *absent* | `92.13631373833611` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/phi_predeclared_deg` | *absent* | `92.11422972729224` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/rel_change` | *absent* | `1.1295647224760565e-08` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/rel_change_predeclared_phi` | *absent* | `0.00038544936645878183` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/rotated_base` | *absent* | `[-2.628239658403131, -1.591004111116771]` | 0–1 |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/value_base` | *absent* | `[-1.4919252395629883, 2.685720920562744]` | 0–1 |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_complex/value_shifted` | *absent* | `[-2.628239631652832, -1.5910041332244873]` | 0–1 |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/excluded_from_envelope` | *absent* | `"pre-declared ULP-floor skip leg (§5(a)): a physically zero derivative, reported only"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/pinned_gate` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/rel_change` | *absent* | `0.00026644604018450236` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/value_base` | *absent* | `0.0007709434721618891` | — |
| `/plane_shift/pec_short|false/gradient_invariance/eps:s11_mag2/value_shifted` | *absent* | `0.0007711488869972527` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/rel_change` | *absent* | `7.417800019968717e-08` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/value_base` | *absent* | `-6.428282737731934` | — |
| `/plane_shift/pec_short|false/gradient_invariance/sigma:s11_mag2/value_shifted` | *absent* | `-6.428282260894775` | — |
| `/plane_shift/pec_short|false/lane` | *absent* | `"false"` | — |
| `/plane_shift/pec_short|false/reference_planes_base_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/plane_shift/pec_short|false/reference_planes_shifted_m` | *absent* | `[0.025400000000000002, 0.09906000000000001]` | 0–1 |
| `/plane_shift/pec_short|false/resid_cont_max` | *absent* | `0.040690885298857886` | — |
| `/plane_shift/pec_short|false/resid_port_beta_max` | *absent* | `1.1155632392956251e-05` | — |
| `/plane_shift/pec_short|false/resid_yee_max` | *absent* | `0.03171225232247821` | — |
| `/plane_shift/pec_short|false/rotation_deg/S11/abs_s_base_peak` | *absent* | `1.0001433250411775` | — |
| `/plane_shift/pec_short|false/rotation_deg/S11/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/pec_short|false/rotation_deg/S11/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/pec_short|false/rotation_deg/S11/measurable` | *absent* | `true` | — |
| `/plane_shift/pec_short|false/rotation_deg/S11/measured` | *absent* | array[17], SHA256 `1f7654332b4a` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S11/predicted_continuous` | *absent* | array[17], SHA256 `2215e9e7a80e` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S11/predicted_port_beta` | *absent* | array[17], SHA256 `350997473cb4` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S11/predicted_yee` | *absent* | array[17], SHA256 `720e95ebed38` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S11/resid_cont_max` | *absent* | `0.040690885298857886` | — |
| `/plane_shift/pec_short|false/rotation_deg/S11/resid_port_beta_max` | *absent* | `1.1155632392956251e-05` | — |
| `/plane_shift/pec_short|false/rotation_deg/S11/resid_yee_max` | *absent* | `0.03171225232247821` | — |
| `/plane_shift/pec_short|false/rotation_deg/S11/wrong_sign_resid_min` | *absent* | `126.44637281047184` | — |
| `/plane_shift/pec_short|false/rotation_deg/S12/abs_s_base_peak` | *absent* | `0.0` | — |
| `/plane_shift/pec_short|false/rotation_deg/S12/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/pec_short|false/rotation_deg/S12/masked_bins_hz` | *absent* | array[17], SHA256 `c0ed1e088b59` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S12/measurable` | *absent* | `false` | — |
| `/plane_shift/pec_short|false/rotation_deg/S12/measured` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S12/predicted_continuous` | *absent* | array[17], SHA256 `0c00681ea182` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S12/predicted_port_beta` | *absent* | array[17], SHA256 `117c6c953b89` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S12/predicted_yee` | *absent* | array[17], SHA256 `60e7b6af4746` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S12/resid_cont_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/rotation_deg/S12/resid_port_beta_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/rotation_deg/S12/resid_yee_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/rotation_deg/S12/wrong_sign_resid_min` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/rotation_deg/S21/abs_s_base_peak` | *absent* | `0.0` | — |
| `/plane_shift/pec_short|false/rotation_deg/S21/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/pec_short|false/rotation_deg/S21/masked_bins_hz` | *absent* | array[17], SHA256 `c0ed1e088b59` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S21/measurable` | *absent* | `false` | — |
| `/plane_shift/pec_short|false/rotation_deg/S21/measured` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S21/predicted_continuous` | *absent* | array[17], SHA256 `0c00681ea182` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S21/predicted_port_beta` | *absent* | array[17], SHA256 `117c6c953b89` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S21/predicted_yee` | *absent* | array[17], SHA256 `60e7b6af4746` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S21/resid_cont_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/rotation_deg/S21/resid_port_beta_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/rotation_deg/S21/resid_yee_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/rotation_deg/S21/wrong_sign_resid_min` | *absent* | `null` | — |
| `/plane_shift/pec_short|false/rotation_deg/S22/abs_s_base_peak` | *absent* | `1.0001269202733283` | — |
| `/plane_shift/pec_short|false/rotation_deg/S22/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/pec_short|false/rotation_deg/S22/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/pec_short|false/rotation_deg/S22/measurable` | *absent* | `true` | — |
| `/plane_shift/pec_short|false/rotation_deg/S22/measured` | *absent* | array[17], SHA256 `2ba9ea66e597` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S22/predicted_continuous` | *absent* | array[17], SHA256 `29ae3cb10cd0` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S22/predicted_port_beta` | *absent* | array[17], SHA256 `68dd10ef20d2` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S22/predicted_yee` | *absent* | array[17], SHA256 `71d654dbf351` | 0–16 |
| `/plane_shift/pec_short|false/rotation_deg/S22/resid_cont_max` | *absent* | `0.02034519124136125` | — |
| `/plane_shift/pec_short|false/rotation_deg/S22/resid_port_beta_max` | *absent* | `5.1307962181113e-06` | — |
| `/plane_shift/pec_short|false/rotation_deg/S22/resid_yee_max` | *absent* | `0.01585904025828455` | — |
| `/plane_shift/pec_short|false/rotation_deg/S22/wrong_sign_resid_min` | *absent* | `64.05492509115297` | — |
| `/plane_shift/pec_short|false/rung` | *absent* | `"fine"` | — |
| `/plane_shift/pec_short|false/s_params_shifted/S11` | *absent* | array[17], SHA256 `07497b2434af` | 0–16 |
| `/plane_shift/pec_short|false/s_params_shifted/S12` | *absent* | array[17], SHA256 `e973cd6bc3bd` | 0–16 |
| `/plane_shift/pec_short|false/s_params_shifted/S21` | *absent* | array[17], SHA256 `3f5df82286b3` | 0–16 |
| `/plane_shift/pec_short|false/s_params_shifted/S22` | *absent* | array[17], SHA256 `9c0f4b575f71` | 0–16 |
| `/plane_shift/pec_short|false/settling_db_shifted/left` | *absent* | `-102.27939898162612` | — |
| `/plane_shift/pec_short|false/settling_db_shifted/right` | *absent* | `-101.96950180120103` | — |
| `/plane_shift/pec_short|false/shift_m` | *absent* | `[0.005080000000000001, -0.0025400000000000006]` | 0–1 |
| `/plane_shift/pec_short|false/wall_time_s/shifted_forward` | *absent* | `8.157795667648315` | — |
| `/plane_shift/pec_short|false/wall_time_s/total` | *absent* | `66.94146919250488` | — |
| `/plane_shift/pec_short|false/warnings_shifted/0/count` | *absent* | `1` | — |
| `/plane_shift/pec_short|false/warnings_shifted/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/plane_shift/pec_short|false/wrong_sign_resid_min` | *absent* | `64.05492509115297` | — |
| `/plane_shift/pec_short|flux/abs_s_allclose` | *absent* | `true` | — |
| `/plane_shift/pec_short|flux/abs_s_max_diff` | *absent* | `8.225788072913076e-08` | — |
| `/plane_shift/pec_short|flux/base_source` | *absent* | `"cell__pec_short__fine__flux.json"` | — |
| `/plane_shift/pec_short|flux/dut` | *absent* | `"pec_short"` | — |
| `/plane_shift/pec_short|flux/dx_m` | *absent* | `0.000635` | — |
| `/plane_shift/pec_short|flux/entries_measurable` | *absent* | `["S11", "S22"]` | 0–1 |
| `/plane_shift/pec_short|flux/fc_port_hz` | *absent* | `6555059929.275057` | — |
| `/plane_shift/pec_short|flux/fc_predeclared_hz` | *absent* | `6557140376.203117` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/from_objectives` | *absent* | `["re_s11", "im_s11"]` | 0–1 |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/kind` | *absent* | `"complex"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/phi_measured_deg` | *absent* | `92.136315598176` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/phi_predeclared_deg` | *absent* | `92.11422972729225` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/rel_change` | *absent* | `1.61561377395616e-07` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/rel_change_predeclared_phi` | *absent* | `0.0003853428125953509` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/rotated_base` | *absent* | `[-2.6311149758852475, -1.5781179112333568]` | 0–1 |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/value_base` | *absent* | `[-1.478940725326538, 2.6881139278411865]` | 0–1 |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_complex/value_shifted` | *absent* | `[-2.631115436553955, -1.5781177282333374]` | 0–1 |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/excluded_from_envelope` | *absent* | `"pre-declared ULP-floor skip leg (§5(a)): a physically zero derivative, reported only"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/pinned_gate` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/rel_change` | *absent* | `0.00649894801788288` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/value_base` | *absent* | `2.786023287626449e-05` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/eps:s11_mag2/value_shifted` | *absent* | `2.8041295081493445e-05` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/rel_change` | *absent* | `1.4851415302749813e-07` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/value_base` | *absent* | `-6.4214372634887695` | — |
| `/plane_shift/pec_short|flux/gradient_invariance/sigma:s11_mag2/value_shifted` | *absent* | `-6.421438217163086` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/base_precision` | *absent* | `"x64"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/from_objectives` | *absent* | `["re_s11", "im_s11"]` | 0–1 |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/kind` | *absent* | `"complex"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/note` | *absent* | `"the closing run's stored reading: x64 base against a float32 shifted gradient — reports the float32 gradient error on this lane, not the plane invariance"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/phi_measured_deg` | *absent* | `92.136315598176` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/phi_predeclared_deg` | *absent* | `92.11422972729224` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/pinned_gate_envelope` | *absent* | `4.730868462554797e-06` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/rel_change` | *absent* | `3.3936554670803806e-06` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/rel_change_predeclared_phi` | *absent* | `0.0003887989679576951` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/rotated_base` | *absent* | `[-2.631118927399339, -1.578107918799672]` | 0–1 |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/value_base` | *absent* | `[-1.4789305925369263, 2.688117504119873]` | 0–1 |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_complex/value_shifted` | *absent* | `[-2.631115436553955, -1.5781177282333374]` | 0–1 |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/base_precision` | *absent* | `"x64"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/excluded_from_envelope` | *absent* | `"pre-declared ULP-floor skip leg (§5(a)): a physically zero derivative, reported only"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/note` | *absent* | `"the closing run's stored reading: x64 base against a float32 shifted gradient — reports the float32 gradient error on this lane, not the plane invariance"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/pinned_gate` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/rel_change` | *absent* | `96.29482694106822` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/value_base` | *absent* | `-2.9425831371554523e-07` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/eps:s11_mag2/value_shifted` | *absent* | `2.8041295081493445e-05` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/base_precision` | *absent* | `"x64"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/note` | *absent* | `"the closing run's stored reading: x64 base against a float32 shifted gradient — reports the float32 gradient error on this lane, not the plane invariance"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/pinned_gate_envelope` | *absent* | `4.730868462554797e-06` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/rel_change` | *absent* | `5.940561709812471e-07` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/value_base` | *absent* | `-6.421442031860352` | — |
| `/plane_shift/pec_short|flux/gradient_invariance_x64_base/sigma:s11_mag2/value_shifted` | *absent* | `-6.421438217163086` | — |
| `/plane_shift/pec_short|flux/lane` | *absent* | `"flux"` | — |
| `/plane_shift/pec_short|flux/reference_planes_base_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/plane_shift/pec_short|flux/reference_planes_shifted_m` | *absent* | `[0.025400000000000002, 0.09906000000000001]` | 0–1 |
| `/plane_shift/pec_short|flux/resid_cont_max` | *absent* | `0.04070093353668369` | — |
| `/plane_shift/pec_short|flux/resid_port_beta_max` | *absent* | `2.3404682487182527e-05` | — |
| `/plane_shift/pec_short|flux/resid_yee_max` | *absent* | `0.03171997853307573` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S11/abs_s_base_peak` | *absent* | `1.0000065558889848` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S11/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S11/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S11/measurable` | *absent* | `true` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S11/measured` | *absent* | array[17], SHA256 `d8d72e5499e6` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S11/predicted_continuous` | *absent* | array[17], SHA256 `2215e9e7a80e` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S11/predicted_port_beta` | *absent* | array[17], SHA256 `350997473cb4` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S11/predicted_yee` | *absent* | array[17], SHA256 `720e95ebed38` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S11/resid_cont_max` | *absent* | `0.04070093353668369` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S11/resid_port_beta_max` | *absent* | `1.522389035812921e-05` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S11/resid_yee_max` | *absent* | `0.03171997853307573` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S11/wrong_sign_resid_min` | *absent* | `126.44636276223399` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S12/abs_s_base_peak` | *absent* | `0.0` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S12/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S12/masked_bins_hz` | *absent* | array[17], SHA256 `c0ed1e088b59` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S12/measurable` | *absent* | `false` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S12/measured` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S12/predicted_continuous` | *absent* | array[17], SHA256 `0c00681ea182` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S12/predicted_port_beta` | *absent* | array[17], SHA256 `117c6c953b89` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S12/predicted_yee` | *absent* | array[17], SHA256 `60e7b6af4746` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S12/resid_cont_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S12/resid_port_beta_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S12/resid_yee_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S12/wrong_sign_resid_min` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S21/abs_s_base_peak` | *absent* | `0.0` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S21/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S21/masked_bins_hz` | *absent* | array[17], SHA256 `c0ed1e088b59` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S21/measurable` | *absent* | `false` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S21/measured` | *absent* | array[17], SHA256 `615abadcd565` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S21/predicted_continuous` | *absent* | array[17], SHA256 `0c00681ea182` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S21/predicted_port_beta` | *absent* | array[17], SHA256 `117c6c953b89` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S21/predicted_yee` | *absent* | array[17], SHA256 `60e7b6af4746` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S21/resid_cont_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S21/resid_port_beta_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S21/resid_yee_max` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S21/wrong_sign_resid_min` | *absent* | `null` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S22/abs_s_base_peak` | *absent* | `1.0000064767307797` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S22/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S22/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S22/measurable` | *absent* | `true` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S22/measured` | *absent* | array[17], SHA256 `3a1451a80371` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S22/predicted_continuous` | *absent* | array[17], SHA256 `29ae3cb10cd0` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S22/predicted_port_beta` | *absent* | array[17], SHA256 `68dd10ef20d2` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S22/predicted_yee` | *absent* | array[17], SHA256 `71d654dbf351` | 0–16 |
| `/plane_shift/pec_short|flux/rotation_deg/S22/resid_cont_max` | *absent* | `0.02035355338062317` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S22/resid_port_beta_max` | *absent* | `2.3404682487182527e-05` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S22/resid_yee_max` | *absent* | `0.01585519086012255` | — |
| `/plane_shift/pec_short|flux/rotation_deg/S22/wrong_sign_resid_min` | *absent* | `64.0549212417548` | — |
| `/plane_shift/pec_short|flux/rung` | *absent* | `"fine"` | — |
| `/plane_shift/pec_short|flux/s_params_shifted/S11` | *absent* | array[17], SHA256 `8739a6e4ae1d` | 0–16 |
| `/plane_shift/pec_short|flux/s_params_shifted/S12` | *absent* | array[17], SHA256 `6616093ac548` | 0–16 |
| `/plane_shift/pec_short|flux/s_params_shifted/S21` | *absent* | array[17], SHA256 `6616093ac548` | 0–16 |
| `/plane_shift/pec_short|flux/s_params_shifted/S22` | *absent* | array[17], SHA256 `f58e39459423` | 0–16 |
| `/plane_shift/pec_short|flux/settling_db_shifted/left` | *absent* | `-100.89879388166203` | — |
| `/plane_shift/pec_short|flux/settling_db_shifted/right` | *absent* | `-101.1073500105272` | — |
| `/plane_shift/pec_short|flux/shift_m` | *absent* | `[0.005080000000000001, -0.0025400000000000006]` | 0–1 |
| `/plane_shift/pec_short|flux/wall_time_s/shifted_forward` | *absent* | `12.732009649276733` | — |
| `/plane_shift/pec_short|flux/wall_time_s/total` | *absent* | `105.74105405807495` | — |
| `/plane_shift/pec_short|flux/warnings_shifted/0/count` | *absent* | `8` | — |
| `/plane_shift/pec_short|flux/warnings_shifted/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/plane_shift/pec_short|flux/warnings_shifted/1/count` | *absent* | `56` | — |
| `/plane_shift/pec_short|flux/warnings_shifted/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/plane_shift/pec_short|flux/warnings_shifted/2/count` | *absent* | `8` | — |
| `/plane_shift/pec_short|flux/warnings_shifted/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/plane_shift/pec_short|flux/warnings_shifted/3/count` | *absent* | `16` | — |
| `/plane_shift/pec_short|flux/warnings_shifted/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/plane_shift/pec_short|flux/wrong_sign_resid_min` | *absent* | `64.0549212417548` | — |
| `/plane_shift/slab|false/abs_s_allclose` | *absent* | `true` | — |
| `/plane_shift/slab|false/abs_s_max_diff` | *absent* | `1.58374969694286e-07` | — |
| `/plane_shift/slab|false/base_source` | *absent* | `"cell__slab__fine__false.json"` | — |
| `/plane_shift/slab|false/dut` | *absent* | `"slab"` | — |
| `/plane_shift/slab|false/dx_m` | *absent* | `0.000635` | — |
| `/plane_shift/slab|false/entries_measurable` | *absent* | `["S11", "S22", "S21", "S12"]` | 0–3 |
| `/plane_shift/slab|false/fc_port_hz` | *absent* | `6555059929.275057` | — |
| `/plane_shift/slab|false/fc_predeclared_hz` | *absent* | `6557140376.203117` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/rel_change` | *absent* | `1.8995751600154626e-07` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/value_base` | *absent* | `0.313778817653656` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s11_mag2/value_shifted` | *absent* | `0.3137788772583008` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/from_objectives` | *absent* | `["re_s21", "im_s21"]` | 0–1 |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/kind` | *absent* | `"complex"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/phi_measured_deg` | *absent* | `69.10223323772178` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/phi_predeclared_deg` | *absent* | `69.08567229546918` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/rel_change` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/rel_change_predeclared_phi` | *absent* | `0.00028921349381511523` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/rotated_base` | *absent* | `[-0.3842299737700634, 0.2509970029265802]` | 0–1 |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/value_base` | *absent* | `[0.09743057191371918, 0.44848573207855225]` | 0–1 |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_complex/value_shifted` | *absent* | `[-0.3842300772666931, 0.25099697709083557]` | 0–1 |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/rel_change` | *absent* | `0.0` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/value_base` | *absent* | `-0.31374168395996094` | — |
| `/plane_shift/slab|false/gradient_invariance/eps:s21_mag2/value_shifted` | *absent* | `-0.31374168395996094` | — |
| `/plane_shift/slab|false/lane` | *absent* | `"false"` | — |
| `/plane_shift/slab|false/reference_planes_base_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/plane_shift/slab|false/reference_planes_shifted_m` | *absent* | `[0.025400000000000002, 0.09906000000000001]` | 0–1 |
| `/plane_shift/slab|false/resid_cont_max` | *absent* | `0.040696455747138316` | — |
| `/plane_shift/slab|false/resid_port_beta_max` | *absent* | `8.964618444906591e-06` | — |
| `/plane_shift/slab|false/resid_yee_max` | *absent* | `0.03171547203277214` | — |
| `/plane_shift/slab|false/rotation_deg/S11/abs_s_base_peak` | *absent* | `0.6964204948388697` | — |
| `/plane_shift/slab|false/rotation_deg/S11/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/slab|false/rotation_deg/S11/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/slab|false/rotation_deg/S11/measurable` | *absent* | `true` | — |
| `/plane_shift/slab|false/rotation_deg/S11/measured` | *absent* | array[17], SHA256 `2938d59c9f9e` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S11/predicted_continuous` | *absent* | array[17], SHA256 `2215e9e7a80e` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S11/predicted_port_beta` | *absent* | array[17], SHA256 `350997473cb4` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S11/predicted_yee` | *absent* | array[17], SHA256 `720e95ebed38` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S11/resid_cont_max` | *absent* | `0.040696455747138316` | — |
| `/plane_shift/slab|false/rotation_deg/S11/resid_port_beta_max` | *absent* | `8.964618444906591e-06` | — |
| `/plane_shift/slab|false/rotation_deg/S11/resid_yee_max` | *absent* | `0.03171547203277214` | — |
| `/plane_shift/slab|false/rotation_deg/S11/wrong_sign_resid_min` | *absent* | `126.44636724002355` | — |
| `/plane_shift/slab|false/rotation_deg/S12/abs_s_base_peak` | *absent* | `0.9801680572336376` | — |
| `/plane_shift/slab|false/rotation_deg/S12/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/slab|false/rotation_deg/S12/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/slab|false/rotation_deg/S12/measurable` | *absent* | `true` | — |
| `/plane_shift/slab|false/rotation_deg/S12/measured` | *absent* | array[17], SHA256 `2e973069da97` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S12/predicted_continuous` | *absent* | array[17], SHA256 `0c00681ea182` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S12/predicted_port_beta` | *absent* | array[17], SHA256 `117c6c953b89` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S12/predicted_yee` | *absent* | array[17], SHA256 `60e7b6af4746` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S12/resid_cont_max` | *absent* | `0.030522336918068053` | — |
| `/plane_shift/slab|false/rotation_deg/S12/resid_port_beta_max` | *absent* | `7.596767524375992e-06` | — |
| `/plane_shift/slab|false/rotation_deg/S12/resid_yee_max` | *absent* | `0.023791771889342783` | — |
| `/plane_shift/slab|false/rotation_deg/S12/wrong_sign_resid_min` | *absent* | `96.08239084823138` | — |
| `/plane_shift/slab|false/rotation_deg/S21/abs_s_base_peak` | *absent* | `0.9788542810748996` | — |
| `/plane_shift/slab|false/rotation_deg/S21/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/slab|false/rotation_deg/S21/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/slab|false/rotation_deg/S21/measurable` | *absent* | `true` | — |
| `/plane_shift/slab|false/rotation_deg/S21/measured` | *absent* | array[17], SHA256 `e03a9da95dc5` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S21/predicted_continuous` | *absent* | array[17], SHA256 `0c00681ea182` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S21/predicted_port_beta` | *absent* | array[17], SHA256 `117c6c953b89` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S21/predicted_yee` | *absent* | array[17], SHA256 `60e7b6af4746` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S21/resid_cont_max` | *absent* | `0.0305196622484516` | — |
| `/plane_shift/slab|false/rotation_deg/S21/resid_port_beta_max` | *absent* | `4.721462659063036e-06` | — |
| `/plane_shift/slab|false/rotation_deg/S21/resid_yee_max` | *absent* | `0.02378887341473046` | — |
| `/plane_shift/slab|false/rotation_deg/S21/wrong_sign_resid_min` | *absent* | `96.08238794975676` | — |
| `/plane_shift/slab|false/rotation_deg/S22/abs_s_base_peak` | *absent* | `0.6955785421916003` | — |
| `/plane_shift/slab|false/rotation_deg/S22/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/slab|false/rotation_deg/S22/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/slab|false/rotation_deg/S22/measurable` | *absent* | `true` | — |
| `/plane_shift/slab|false/rotation_deg/S22/measured` | *absent* | array[17], SHA256 `09277ddee11f` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S22/predicted_continuous` | *absent* | array[17], SHA256 `29ae3cb10cd0` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S22/predicted_port_beta` | *absent* | array[17], SHA256 `68dd10ef20d2` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S22/predicted_yee` | *absent* | array[17], SHA256 `71d654dbf351` | 0–16 |
| `/plane_shift/slab|false/rotation_deg/S22/resid_cont_max` | *absent* | `0.02035163600152856` | — |
| `/plane_shift/slab|false/rotation_deg/S22/resid_port_beta_max` | *absent* | `7.879465051985335e-06` | — |
| `/plane_shift/slab|false/rotation_deg/S22/resid_yee_max` | *absent* | `0.015862873107586495` | — |
| `/plane_shift/slab|false/rotation_deg/S22/wrong_sign_resid_min` | *absent* | `64.05492892400227` | — |
| `/plane_shift/slab|false/rung` | *absent* | `"fine"` | — |
| `/plane_shift/slab|false/s_params_shifted/S11` | *absent* | array[17], SHA256 `c4d5e0b8a730` | 0–16 |
| `/plane_shift/slab|false/s_params_shifted/S12` | *absent* | array[17], SHA256 `491e91f3595c` | 0–16 |
| `/plane_shift/slab|false/s_params_shifted/S21` | *absent* | array[17], SHA256 `fb5affce676f` | 0–16 |
| `/plane_shift/slab|false/s_params_shifted/S22` | *absent* | array[17], SHA256 `437de31a0f04` | 0–16 |
| `/plane_shift/slab|false/settling_db_shifted/left` | *absent* | `-102.64564907998336` | — |
| `/plane_shift/slab|false/settling_db_shifted/right` | *absent* | `-101.23601747187766` | — |
| `/plane_shift/slab|false/shift_m` | *absent* | `[0.005080000000000001, -0.0025400000000000006]` | 0–1 |
| `/plane_shift/slab|false/wall_time_s/shifted_forward` | *absent* | `6.429701566696167` | — |
| `/plane_shift/slab|false/wall_time_s/total` | *absent* | `64.47826981544495` | — |
| `/plane_shift/slab|false/warnings_shifted/0/count` | *absent* | `1` | — |
| `/plane_shift/slab|false/warnings_shifted/0/message` | *absent* | `"UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. For &#124;S11&#124; of strong reflectors (PEC short, resonators) normalize=False is more accurate — see the normalize parameter docstring."` | — |
| `/plane_shift/slab|false/wrong_sign_resid_min` | *absent* | `64.05492892400227` | — |
| `/plane_shift/slab|flux/abs_s_allclose` | *absent* | `true` | — |
| `/plane_shift/slab|flux/abs_s_max_diff` | *absent* | `6.802115548598664e-08` | — |
| `/plane_shift/slab|flux/base_source` | *absent* | `"cell__slab__fine__flux.json"` | — |
| `/plane_shift/slab|flux/dut` | *absent* | `"slab"` | — |
| `/plane_shift/slab|flux/dx_m` | *absent* | `0.000635` | — |
| `/plane_shift/slab|flux/entries_measurable` | *absent* | `["S11", "S22", "S21", "S12"]` | 0–3 |
| `/plane_shift/slab|flux/fc_port_hz` | *absent* | `6555059929.275057` | — |
| `/plane_shift/slab|flux/fc_predeclared_hz` | *absent* | `6557140376.203117` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/rel_change` | *absent* | `0.0` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/value_base` | *absent* | `0.3149757981300354` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s11_mag2/value_shifted` | *absent* | `0.3149757981300354` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/from_objectives` | *absent* | `["re_s21", "im_s21"]` | 0–1 |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/kind` | *absent* | `"complex"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/phi_measured_deg` | *absent* | `69.10222627212148` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/phi_predeclared_deg` | *absent* | `69.08567229546918` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/rel_change` | *absent* | `1.0997232214955108e-07` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/rel_change_predeclared_phi` | *absent* | `0.00028882389847919676` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/rotated_base` | *absent* | `[-0.38622353923075625, 0.25204813977550966]` | 0–1 |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/value_base` | *absent* | `[0.09770140051841736, 0.4507231116294861]` | 0–1 |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_complex/value_shifted` | *absent* | `[-0.3862234950065613, 0.25204816460609436]` | 0–1 |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/base_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/pinned_gate_envelope` | *absent* | `2.3242906004440017e-07` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/rel_change` | *absent* | `1.8923504288586468e-07` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/value_base` | *absent* | `-0.3149767816066742` | — |
| `/plane_shift/slab|flux/gradient_invariance/eps:s21_mag2/value_shifted` | *absent* | `-0.31497684121131897` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/base_precision` | *absent* | `"x64"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/note` | *absent* | `"the closing run's stored reading: x64 base against a float32 shifted gradient — reports the float32 gradient error on this lane, not the plane invariance"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/pinned_gate_envelope` | *absent* | `4.730868462554797e-06` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/rel_change` | *absent* | `4.730868462554797e-06` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/value_base` | *absent* | `0.3149772882461548` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s11_mag2/value_shifted` | *absent* | `0.3149757981300354` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/base_precision` | *absent* | `"x64"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/from_objectives` | *absent* | `["re_s21", "im_s21"]` | 0–1 |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/kind` | *absent* | `"complex"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/note` | *absent* | `"the closing run's stored reading: x64 base against a float32 shifted gradient — reports the float32 gradient error on this lane, not the plane invariance"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/phi_measured_deg` | *absent* | `69.10222627212148` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/phi_predeclared_deg` | *absent* | `69.08567229546918` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/pinned_gate_envelope` | *absent* | `4.730868462554797e-06` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/rel_change` | *absent* | `2.1317033052611427e-06` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/rel_change_predeclared_phi` | *absent* | `0.00028754500837931097` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/rotated_base` | *absent* | `[-0.3862244702884657, 0.2520480406838703]` | 0–1 |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/value_base` | *absent* | `[0.09770097583532333, 0.45072394609451294]` | 0–1 |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_complex/value_shifted` | *absent* | `[-0.3862234950065613, 0.25204816460609436]` | 0–1 |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/base_precision` | *absent* | `"x64"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/kind` | *absent* | `"magnitude"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/note` | *absent* | `"the closing run's stored reading: x64 base against a float32 shifted gradient — reports the float32 gradient error on this lane, not the plane invariance"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/pinned_gate` | *absent* | `0.001` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/pinned_gate_envelope` | *absent* | `4.730868462554797e-06` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/rel_change` | *absent* | `2.6492830803441455e-06` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/report_bar` | *absent* | `0.01` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/shift_precision` | *absent* | `"float32"` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/value_base` | *absent* | `-0.3149776756763458` | — |
| `/plane_shift/slab|flux/gradient_invariance_x64_base/eps:s21_mag2/value_shifted` | *absent* | `-0.31497684121131897` | — |
| `/plane_shift/slab|flux/lane` | *absent* | `"flux"` | — |
| `/plane_shift/slab|flux/reference_planes_base_m` | *absent* | `[0.02032, 0.10160000000000001]` | 0–1 |
| `/plane_shift/slab|flux/reference_planes_shifted_m` | *absent* | `[0.025400000000000002, 0.09906000000000001]` | 0–1 |
| `/plane_shift/slab|flux/resid_cont_max` | *absent* | `0.04069406716313039` | — |
| `/plane_shift/slab|flux/resid_port_beta_max` | *absent* | `1.615927966724939e-05` | — |
| `/plane_shift/slab|flux/resid_yee_max` | *absent* | `0.03170333539493697` | — |
| `/plane_shift/slab|flux/rotation_deg/S11/abs_s_base_peak` | *absent* | `0.6965226408629608` | — |
| `/plane_shift/slab|flux/rotation_deg/S11/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/slab|flux/rotation_deg/S11/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/slab|flux/rotation_deg/S11/measurable` | *absent* | `true` | — |
| `/plane_shift/slab|flux/rotation_deg/S11/measured` | *absent* | array[17], SHA256 `227721507812` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S11/predicted_continuous` | *absent* | array[17], SHA256 `2215e9e7a80e` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S11/predicted_port_beta` | *absent* | array[17], SHA256 `350997473cb4` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S11/predicted_yee` | *absent* | array[17], SHA256 `720e95ebed38` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S11/resid_cont_max` | *absent* | `0.04069406716313039` | — |
| `/plane_shift/slab|flux/rotation_deg/S11/resid_port_beta_max` | *absent* | `1.615927966724939e-05` | — |
| `/plane_shift/slab|flux/rotation_deg/S11/resid_yee_max` | *absent* | `0.03170333539493697` | — |
| `/plane_shift/slab|flux/rotation_deg/S11/wrong_sign_resid_min` | *absent* | `126.44636962860754` | — |
| `/plane_shift/slab|flux/rotation_deg/S12/abs_s_base_peak` | *absent* | `0.9792523301131798` | — |
| `/plane_shift/slab|flux/rotation_deg/S12/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/slab|flux/rotation_deg/S12/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/slab|flux/rotation_deg/S12/measurable` | *absent* | `true` | — |
| `/plane_shift/slab|flux/rotation_deg/S12/measured` | *absent* | array[17], SHA256 `74a8dbe545f4` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S12/predicted_continuous` | *absent* | array[17], SHA256 `0c00681ea182` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S12/predicted_port_beta` | *absent* | array[17], SHA256 `117c6c953b89` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S12/predicted_yee` | *absent* | array[17], SHA256 `60e7b6af4746` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S12/resid_cont_max` | *absent* | `0.03052677773649749` | — |
| `/plane_shift/slab|flux/rotation_deg/S12/resid_port_beta_max` | *absent* | `1.3435089499580501e-05` | — |
| `/plane_shift/slab|flux/rotation_deg/S12/resid_yee_max` | *absent* | `0.023788240757738777` | — |
| `/plane_shift/slab|flux/rotation_deg/S12/wrong_sign_resid_min` | *absent* | `96.08238731709976` | — |
| `/plane_shift/slab|flux/rotation_deg/S21/abs_s_base_peak` | *absent* | `0.9792512604244444` | — |
| `/plane_shift/slab|flux/rotation_deg/S21/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/slab|flux/rotation_deg/S21/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/slab|flux/rotation_deg/S21/measurable` | *absent* | `true` | — |
| `/plane_shift/slab|flux/rotation_deg/S21/measured` | *absent* | array[17], SHA256 `8c086cb988f7` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S21/predicted_continuous` | *absent* | array[17], SHA256 `0c00681ea182` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S21/predicted_port_beta` | *absent* | array[17], SHA256 `117c6c953b89` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S21/predicted_yee` | *absent* | array[17], SHA256 `60e7b6af4746` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S21/resid_cont_max` | *absent* | `0.03052007633380072` | — |
| `/plane_shift/slab|flux/rotation_deg/S21/resid_port_beta_max` | *absent* | `9.727579907803374e-06` | — |
| `/plane_shift/slab|flux/rotation_deg/S21/resid_yee_max` | *absent* | `0.02378844802825597` | — |
| `/plane_shift/slab|flux/rotation_deg/S21/wrong_sign_resid_min` | *absent* | `96.08238752437028` | — |
| `/plane_shift/slab|flux/rotation_deg/S22/abs_s_base_peak` | *absent* | `0.6965252169681946` | — |
| `/plane_shift/slab|flux/rotation_deg/S22/mask_frac` | *absent* | `0.05` | — |
| `/plane_shift/slab|flux/rotation_deg/S22/masked_bins_hz` | *absent* | `[]` | — |
| `/plane_shift/slab|flux/rotation_deg/S22/measurable` | *absent* | `true` | — |
| `/plane_shift/slab|flux/rotation_deg/S22/measured` | *absent* | array[17], SHA256 `8bdc9397840f` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S22/predicted_continuous` | *absent* | array[17], SHA256 `29ae3cb10cd0` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S22/predicted_port_beta` | *absent* | array[17], SHA256 `68dd10ef20d2` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S22/predicted_yee` | *absent* | array[17], SHA256 `71d654dbf351` | 0–16 |
| `/plane_shift/slab|flux/rotation_deg/S22/resid_cont_max` | *absent* | `0.020358208987673262` | — |
| `/plane_shift/slab|flux/rotation_deg/S22/resid_port_beta_max` | *absent* | `1.4452451196689253e-05` | — |
| `/plane_shift/slab|flux/rotation_deg/S22/resid_yee_max` | *absent* | `0.015846649812338853` | — |
| `/plane_shift/slab|flux/rotation_deg/S22/wrong_sign_resid_min` | *absent* | `64.054912700707` | — |
| `/plane_shift/slab|flux/rung` | *absent* | `"fine"` | — |
| `/plane_shift/slab|flux/s_params_shifted/S11` | *absent* | array[17], SHA256 `9c5debd3a29a` | 0–16 |
| `/plane_shift/slab|flux/s_params_shifted/S12` | *absent* | array[17], SHA256 `c7f6e9f390d7` | 0–16 |
| `/plane_shift/slab|flux/s_params_shifted/S21` | *absent* | array[17], SHA256 `454edb43d46c` | 0–16 |
| `/plane_shift/slab|flux/s_params_shifted/S22` | *absent* | array[17], SHA256 `a68ecbf22a1e` | 0–16 |
| `/plane_shift/slab|flux/settling_db_shifted/left` | *absent* | `-100.89879388166203` | — |
| `/plane_shift/slab|flux/settling_db_shifted/right` | *absent* | `-101.1073500105272` | — |
| `/plane_shift/slab|flux/shift_m` | *absent* | `[0.005080000000000001, -0.0025400000000000006]` | 0–1 |
| `/plane_shift/slab|flux/wall_time_s/shifted_forward` | *absent* | `12.146903991699219` | — |
| `/plane_shift/slab|flux/wall_time_s/total` | *absent* | `101.45535063743591` | — |
| `/plane_shift/slab|flux/warnings_shifted/0/count` | *absent* | `8` | — |
| `/plane_shift/slab|flux/warnings_shifted/0/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in zeros is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/plane_shift/slab|flux/warnings_shifted/1/count` | *absent* | `56` | — |
| `/plane_shift/slab|flux/warnings_shifted/1/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/plane_shift/slab|flux/warnings_shifted/2/count` | *absent* | `8` | — |
| `/plane_shift/slab|flux/warnings_shifted/2/message` | *absent* | `"UserWarning: Explicitly requested dtype float64 requested in asarray is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/plane_shift/slab|flux/warnings_shifted/3/count` | *absent* | `16` | — |
| `/plane_shift/slab|flux/warnings_shifted/3/message` | *absent* | `"UserWarning: Explicitly requested dtype <class 'jax.numpy.complex128'> requested in astype is not available, and will be truncated to dtype complex64. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotchas for more."` | — |
| `/plane_shift/slab|flux/wrong_sign_resid_min` | *absent* | `64.054912700707` | — |
| `/port_cutoff/length_between_declared_planes_m` | *absent* | `0.08128` | — |
| `/port_cutoff/per_rung/coarse|false/const_deg_at_fit` | *absent* | `359.7003680787279` | — |
| `/port_cutoff/per_rung/coarse|false/fc_c_over_2a_hz` | *absent* | `6557140376.203117` | — |
| `/port_cutoff/per_rung/coarse|false/fc_discrete_guide_hz` | *absent* | `6523900723.790886` | — |
| `/port_cutoff/per_rung/coarse|false/fc_fit_hz` | *absent* | `6526000000.0` | — |
| `/port_cutoff/per_rung/coarse|false/fc_port_hz` | *absent* | `6523900723.7908745` | — |
| `/port_cutoff/per_rung/coarse|false/length_m` | *absent* | `0.08128` | — |
| `/port_cutoff/per_rung/coarse|false/port_cutoff_effective_width_cells` | *absent* | `9.045855521776712` | — |
| `/port_cutoff/per_rung/coarse|false/rms_deg_at_c_over_2a` | *absent* | `0.5012095326403567` | — |
| `/port_cutoff/per_rung/coarse|false/rms_deg_at_discrete_guide` | *absent* | `0.03361977347555447` | — |
| `/port_cutoff/per_rung/coarse|false/rms_deg_at_fit` | *absent* | `0.014319078228672856` | — |
| `/port_cutoff/per_rung/coarse|false/rms_deg_at_port_cutoff` | *absent* | `0.03361977347555447` | — |
| `/port_cutoff/per_rung/coarse|flux/const_deg_at_fit` | *absent* | `359.70036753242294` | — |
| `/port_cutoff/per_rung/coarse|flux/fc_c_over_2a_hz` | *absent* | `6557140376.203117` | — |
| `/port_cutoff/per_rung/coarse|flux/fc_discrete_guide_hz` | *absent* | `6523900723.790886` | — |
| `/port_cutoff/per_rung/coarse|flux/fc_fit_hz` | *absent* | `6526000000.0` | — |
| `/port_cutoff/per_rung/coarse|flux/fc_port_hz` | *absent* | `6523900723.7908745` | — |
| `/port_cutoff/per_rung/coarse|flux/length_m` | *absent* | `0.08128` | — |
| `/port_cutoff/per_rung/coarse|flux/port_cutoff_effective_width_cells` | *absent* | `9.045855521776712` | — |
| `/port_cutoff/per_rung/coarse|flux/rms_deg_at_c_over_2a` | *absent* | `0.5012090801624824` | — |
| `/port_cutoff/per_rung/coarse|flux/rms_deg_at_discrete_guide` | *absent* | `0.03361977069424417` | — |
| `/port_cutoff/per_rung/coarse|flux/rms_deg_at_fit` | *absent* | `0.01431807688170905` | — |
| `/port_cutoff/per_rung/coarse|flux/rms_deg_at_port_cutoff` | *absent* | `0.03361977069424417` | — |
| `/port_cutoff/per_rung/fine|false/const_deg_at_fit` | *absent* | `359.99653078422637` | — |
| `/port_cutoff/per_rung/fine|false/fc_c_over_2a_hz` | *absent* | `6557140376.203117` | — |
| `/port_cutoff/per_rung/fine|false/fc_discrete_guide_hz` | *absent* | `6555059929.275007` | — |
| `/port_cutoff/per_rung/fine|false/fc_fit_hz` | *absent* | `6555000000.0` | — |
| `/port_cutoff/per_rung/fine|false/fc_port_hz` | *absent* | `6555059929.275057` | — |
| `/port_cutoff/per_rung/fine|false/length_m` | *absent* | `0.08128` | — |
| `/port_cutoff/per_rung/fine|false/port_cutoff_effective_width_cells` | *absent* | `36.01142569102542` | — |
| `/port_cutoff/per_rung/fine|false/rms_deg_at_c_over_2a` | *absent* | `0.030451242381750992` | — |
| `/port_cutoff/per_rung/fine|false/rms_deg_at_discrete_guide` | *absent* | `0.003592142923323918` | — |
| `/port_cutoff/per_rung/fine|false/rms_deg_at_fit` | *absent* | `0.0045503131154870195` | — |
| `/port_cutoff/per_rung/fine|false/rms_deg_at_port_cutoff` | *absent* | `0.003592142923323918` | — |
| `/port_cutoff/per_rung/fine|flux/const_deg_at_fit` | *absent* | `359.9965307829527` | — |
| `/port_cutoff/per_rung/fine|flux/fc_c_over_2a_hz` | *absent* | `6557140376.203117` | — |
| `/port_cutoff/per_rung/fine|flux/fc_discrete_guide_hz` | *absent* | `6555059929.275007` | — |
| `/port_cutoff/per_rung/fine|flux/fc_fit_hz` | *absent* | `6555000000.0` | — |
| `/port_cutoff/per_rung/fine|flux/fc_port_hz` | *absent* | `6555059929.275057` | — |
| `/port_cutoff/per_rung/fine|flux/length_m` | *absent* | `0.08128` | — |
| `/port_cutoff/per_rung/fine|flux/port_cutoff_effective_width_cells` | *absent* | `36.01142569102542` | — |
| `/port_cutoff/per_rung/fine|flux/rms_deg_at_c_over_2a` | *absent* | `0.030449764731814346` | — |
| `/port_cutoff/per_rung/fine|flux/rms_deg_at_discrete_guide` | *absent* | `0.003593771840399084` | — |
| `/port_cutoff/per_rung/fine|flux/rms_deg_at_fit` | *absent* | `0.004551893264039493` | — |
| `/port_cutoff/per_rung/fine|flux/rms_deg_at_port_cutoff` | *absent* | `0.003593771840399084` | — |
| `/port_cutoff/per_rung/mid|false/const_deg_at_fit` | *absent* | `359.8628376495432` | — |
| `/port_cutoff/per_rung/mid|false/fc_c_over_2a_hz` | *absent* | `6557140376.203117` | — |
| `/port_cutoff/per_rung/mid|false/fc_discrete_guide_hz` | *absent* | `6548820964.704695` | — |
| `/port_cutoff/per_rung/mid|false/fc_fit_hz` | *absent* | `6550000000.0` | — |
| `/port_cutoff/per_rung/mid|false/fc_port_hz` | *absent* | `6548820964.704762` | — |
| `/port_cutoff/per_rung/mid|false/length_m` | *absent* | `0.08128` | — |
| `/port_cutoff/per_rung/mid|false/port_cutoff_effective_width_cells` | *absent* | `18.022866620995583` | — |
| `/port_cutoff/per_rung/mid|false/rms_deg_at_c_over_2a` | *absent* | `0.12322747628489032` | — |
| `/port_cutoff/per_rung/mid|false/rms_deg_at_discrete_guide` | *absent* | `0.012386986900272994` | — |
| `/port_cutoff/per_rung/mid|false/rms_deg_at_fit` | *absent* | `0.007158684415303936` | — |
| `/port_cutoff/per_rung/mid|false/rms_deg_at_port_cutoff` | *absent* | `0.012386986900272994` | — |
| `/port_cutoff/per_rung/mid|flux/const_deg_at_fit` | *absent* | `359.8628380680426` | — |
| `/port_cutoff/per_rung/mid|flux/fc_c_over_2a_hz` | *absent* | `6557140376.203117` | — |
| `/port_cutoff/per_rung/mid|flux/fc_discrete_guide_hz` | *absent* | `6548820964.704695` | — |
| `/port_cutoff/per_rung/mid|flux/fc_fit_hz` | *absent* | `6550000000.0` | — |
| `/port_cutoff/per_rung/mid|flux/fc_port_hz` | *absent* | `6548820964.704762` | — |
| `/port_cutoff/per_rung/mid|flux/length_m` | *absent* | `0.08128` | — |
| `/port_cutoff/per_rung/mid|flux/port_cutoff_effective_width_cells` | *absent* | `18.022866620995583` | — |
| `/port_cutoff/per_rung/mid|flux/rms_deg_at_c_over_2a` | *absent* | `0.12322634121292021` | — |
| `/port_cutoff/per_rung/mid|flux/rms_deg_at_discrete_guide` | *absent* | `0.01238813220058421` | — |
| `/port_cutoff/per_rung/mid|flux/rms_deg_at_fit` | *absent* | `0.007157618857849404` | — |
| `/port_cutoff/per_rung/mid|flux/rms_deg_at_port_cutoff` | *absent* | `0.01238813220058421` | — |
| `/predeclaration` | *absent* | `"docs/design_notes/20260905_v18_close_predeclaration.md"` | — |
| `/predeclaration_sha` | *absent* | `"f914a7ca"` | — |
| `/provenance/commit` | *absent* | `"f914a7caf1ff8c63cac6f5f8c975b7f9f420a0c7"` | — |
| `/provenance/hostname` | *absent* | `"run-execution-arne70zoxcvz-0"` | — |
| `/provenance/jax_default_backend` | *absent* | `"gpu"` | — |
| `/provenance/jax_devices` | *absent* | `["cuda:0"]` | 0 |
| `/provenance/jax_enable_x64` | *absent* | `false` | — |
| `/provenance/jax_version` | *absent* | `"0.4.33.dev20241023+e3c6d6430"` | — |
| `/provenance/numpy_version` | *absent* | `"1.26.4"` | — |
| `/provenance/post_run_edits` | *absent* | `"three strings written after the run, before the pin step: provenance.run_id (from the backed-up log filename), supersedes and supersedes_reason (the pod ran the script with run 2's constants; the constants in scripts/diagnostics/waveguide_chain_battery_measure.py now say what this artifact says); predeclaration_sha 10b39787 -> f914a7ca (independent review of PR #908): the pod fetched f914a7ca and the note at that commit is the binding version — 10b39787 was the note's first commit, whose section 2 still declared the zero-derivative branch as a gate; report_only entered at 04c42a57 (14:24 UTC), the CPU-smoke section at f914a7ca (14:38), the run started 14:39; the reader commit ba463005 (14:41, after the start, before the first stage finished at 14:42) edited section 3 row 2 and section 3.1 — every revision is listed in the note's section 6; recapture_command --fixture-out corrected from run 2's filename to this artifact's (the pod ran with run 2's ARTIFACT constant); plane_shift.cheap_refute.per_case[*].provenance.run_id (4 entries) set from UNSET-see-log-filename to 369367258638 like the top level; section_4_falsifier attached by the pin step from the pod's falsifier_float32/ stage files (independent review of PR #908, findings 7 and minor)"` | — |
| `/provenance/precision` | *absent* | `"float32"` | — |
| `/provenance/python` | *absent* | `"3.10.12"` | — |
| `/provenance/recapture_command` | *absent* | `"PYTHONPATH=. python scripts/diagnostics/waveguide_chain_battery_measure.py --out-dir <run-dir> --run-id <id> --run-lane <lane>; then --stages assemble --fixture-out tests/fixtures/waveguide_chain_battery/fixture_v18_close.json"` | — |
| `/provenance/recapture_entry_point` | *absent* | `"scripts/diagnostics/waveguide_chain_battery_measure.py"` | — |
| `/provenance/recapture_vessl_yaml` | *absent* | `"scripts/vessl_waveguide_chain_battery_v18_close.yaml"` | — |
| `/provenance/rfx_version` | *absent* | `"1.7.0"` | — |
| `/provenance/run_id` | *absent* | `"369367258638"` | — |
| `/provenance/run_id_note` | *absent* | `"VESSL_RUN_ID was unset in the pod (same as run 2, 369367258205); the id is written in from the backed-up log filename docs/vessl-logs/waveguide_chain_battery_v18_close_369367258638_completed.log"` | — |
| `/provenance/run_lane` | *absent* | `"vessl"` | — |
| `/provenance/wall_time_note` | *absent* | `"sum of the per-case solve wall times (cells incl. settling reruns, AD legs, FD pairs, shifted planes); JIT compile included"` | — |
| `/provenance/wall_time_s` | *absent* | `1350.4885160923004` | — |
| `/readme` | *absent* | `"tests/fixtures/waveguide_chain_battery/README.md"` | — |
| `/referee/broad_e5_replay/fixtures` | *absent* | `["tests/fixtures/waveguide_broad_e5/waveguide_wr10_wband_broad_e5_envelope.json", "tests/fixtures/waveguide_broad_e5/waveguide_wr15_vband_broad_e5_envelope.json", "tests/fixtures/waveguide_broad_e5/waveguide_wr28_kaband_broad_e5_envelope.json", "tests/fixtures/waveguide_broad_e5/waveguide_wr340_sband_broad_e5_envelope.json", "tests/fixtures/waveguide_broad_e5/waveguide_wr62_kuband_broad_e5_envelope.json"]` | 0–4 |
| `/referee/broad_e5_replay/gate_test` | *absent* | `"tests/crossval/test_waveguide_broad_e5.py"` | — |
| `/referee/broad_e5_replay/note` | *absent* | `"criterion 3(d) support set; replayed by its own gate test, not re-run here"` | — |
| `/referee/conventions/dft` | *absent* | `"rectangular full-record DFT with kernel exp(-j omega t) (rfx.sources.waveguide_port._rect_dft)"` | — |
| `/referee/conventions/external_phase_data` | *absent* | `"none enters; if it ever does it is conjugated first (rfx-known-issues.md, time-convention conjugation)"` | — |
| `/referee/conventions/time` | *absent* | `"exp(+j omega t); forward wave exp(-j beta x)"` | — |
| `/referee/conventions/yee_half_step` | *absent* | `"beta for plane shifts and the PEC-short oracle from _compute_beta(dt, dx) (Yee-discrete dispersion); the Airy oracle uses the continuous vacuum beta"` | — |
| `/referee/pec_short/coarse|false/bins_above_1_03` | *absent* | `[]` | — |
| `/referee/pec_short/coarse|false/bins_below_0_99` | *absent* | `[]` | — |
| `/referee/pec_short/coarse|false/gate_max` | *absent* | `1.03` | — |
| `/referee/pec_short/coarse|false/gate_mean_tol` | *absent* | `0.02` | — |
| `/referee/pec_short/coarse|false/gate_min` | *absent* | `0.99` | — |
| `/referee/pec_short/coarse|false/max_s11` | *absent* | `1.0023550035092157` | — |
| `/referee/pec_short/coarse|false/mean_s11` | *absent* | `0.9996478922628529` | — |
| `/referee/pec_short/coarse|false/min_s11` | *absent* | `0.9970207708320434` | — |
| `/referee/pec_short/coarse|false/s22_max` | *absent* | `1.0013083405516978` | — |
| `/referee/pec_short/coarse|false/s22_mean` | *absent* | `0.9993780623127253` | — |
| `/referee/pec_short/coarse|false/s22_min` | *absent* | `0.9972825738239901` | — |
| `/referee/pec_short/coarse|flux/bins_above_1_03` | *absent* | `[]` | — |
| `/referee/pec_short/coarse|flux/bins_below_0_99` | *absent* | `[]` | — |
| `/referee/pec_short/coarse|flux/gate_max` | *absent* | `1.03` | — |
| `/referee/pec_short/coarse|flux/gate_mean_tol` | *absent* | `0.02` | — |
| `/referee/pec_short/coarse|flux/gate_min` | *absent* | `0.99` | — |
| `/referee/pec_short/coarse|flux/max_s11` | *absent* | `1.000032676519756` | — |
| `/referee/pec_short/coarse|flux/mean_s11` | *absent* | `1.0000010316192969` | — |
| `/referee/pec_short/coarse|flux/min_s11` | *absent* | `0.9999894290517741` | — |
| `/referee/pec_short/coarse|flux/s22_max` | *absent* | `1.0000438448933229` | — |
| `/referee/pec_short/coarse|flux/s22_mean` | *absent* | `1.0000008645555627` | — |
| `/referee/pec_short/coarse|flux/s22_min` | *absent* | `0.9999820509836195` | — |
| `/referee/pec_short/fine|false/bins_above_1_03` | *absent* | `[]` | — |
| `/referee/pec_short/fine|false/bins_below_0_99` | *absent* | `[]` | — |
| `/referee/pec_short/fine|false/gate_max` | *absent* | `1.03` | — |
| `/referee/pec_short/fine|false/gate_mean_tol` | *absent* | `0.02` | — |
| `/referee/pec_short/fine|false/gate_min` | *absent* | `0.99` | — |
| `/referee/pec_short/fine|false/max_s11` | *absent* | `1.0001433250411775` | — |
| `/referee/pec_short/fine|false/mean_s11` | *absent* | `0.9999313803612577` | — |
| `/referee/pec_short/fine|false/min_s11` | *absent* | `0.9997148763110957` | — |
| `/referee/pec_short/fine|false/s22_max` | *absent* | `1.0001269202733283` | — |
| `/referee/pec_short/fine|false/s22_mean` | *absent* | `0.9999239531193446` | — |
| `/referee/pec_short/fine|false/s22_min` | *absent* | `0.9997244606561936` | — |
| `/referee/pec_short/fine|flux/bins_above_1_03` | *absent* | `[]` | — |
| `/referee/pec_short/fine|flux/bins_below_0_99` | *absent* | `[]` | — |
| `/referee/pec_short/fine|flux/gate_max` | *absent* | `1.03` | — |
| `/referee/pec_short/fine|flux/gate_mean_tol` | *absent* | `0.02` | — |
| `/referee/pec_short/fine|flux/gate_min` | *absent* | `0.99` | — |
| `/referee/pec_short/fine|flux/max_s11` | *absent* | `1.0000065558889848` | — |
| `/referee/pec_short/fine|flux/mean_s11` | *absent* | `1.0000006669108261` | — |
| `/referee/pec_short/fine|flux/min_s11` | *absent* | `0.9999948053209406` | — |
| `/referee/pec_short/fine|flux/s22_max` | *absent* | `1.0000064767307797` | — |
| `/referee/pec_short/fine|flux/s22_mean` | *absent* | `1.0000007938885533` | — |
| `/referee/pec_short/fine|flux/s22_min` | *absent* | `0.9999960455900777` | — |
| `/referee/pec_short/mid|false/bins_above_1_03` | *absent* | `[]` | — |
| `/referee/pec_short/mid|false/bins_below_0_99` | *absent* | `[]` | — |
| `/referee/pec_short/mid|false/gate_max` | *absent* | `1.03` | — |
| `/referee/pec_short/mid|false/gate_mean_tol` | *absent* | `0.02` | — |
| `/referee/pec_short/mid|false/gate_min` | *absent* | `0.99` | — |
| `/referee/pec_short/mid|false/max_s11` | *absent* | `1.000580770111684` | — |
| `/referee/pec_short/mid|false/mean_s11` | *absent* | `0.9997846306392343` | — |
| `/referee/pec_short/mid|false/min_s11` | *absent* | `0.9989823823705549` | — |
| `/referee/pec_short/mid|false/s22_max` | *absent* | `1.0004250240875905` | — |
| `/referee/pec_short/mid|false/s22_mean` | *absent* | `0.9997335109161286` | — |
| `/referee/pec_short/mid|false/s22_min` | *absent* | `0.9990519311583288` | — |
| `/referee/pec_short/mid|flux/bins_above_1_03` | *absent* | `[]` | — |
| `/referee/pec_short/mid|flux/bins_below_0_99` | *absent* | `[]` | — |
| `/referee/pec_short/mid|flux/gate_max` | *absent* | `1.03` | — |
| `/referee/pec_short/mid|flux/gate_mean_tol` | *absent* | `0.02` | — |
| `/referee/pec_short/mid|flux/gate_min` | *absent* | `0.99` | — |
| `/referee/pec_short/mid|flux/max_s11` | *absent* | `1.000010230679061` | — |
| `/referee/pec_short/mid|flux/mean_s11` | *absent* | `1.0000002506742238` | — |
| `/referee/pec_short/mid|flux/min_s11` | *absent* | `0.9999971287570052` | — |
| `/referee/pec_short/mid|flux/s22_max` | *absent* | `1.0000041873833443` | — |
| `/referee/pec_short/mid|flux/s22_mean` | *absent* | `0.9999999308002185` | — |
| `/referee/pec_short/mid|flux/s22_min` | *absent* | `0.9999981781634446` | — |
| `/referee/slab_airy/coarse|false/d_left_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/coarse|false/d_right_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/coarse|false/gate_mag` | *absent* | `0.05` | — |
| `/referee/slab_airy/coarse|false/gate_phase_deg` | *absent* | `15.0` | — |
| `/referee/slab_airy/coarse|false/max_mag_abs_diff` | *absent* | `0.1486628912567549` | — |
| `/referee/slab_airy/coarse|false/max_phase_diff_deg` | *absent* | `16.942162450507066` | — |
| `/referee/slab_airy/coarse|false/max_phase_diff_deg_unmasked` | *absent* | `16.942162450507066` | — |
| `/referee/slab_airy/coarse|false/oracle_s11` | *absent* | array[17], SHA256 `161336c88d1c` | 0–16 |
| `/referee/slab_airy/coarse|false/oracle_s21` | *absent* | array[17], SHA256 `0e4c6e9bb45e` | 0–16 |
| `/referee/slab_airy/coarse|false/oracle_shift_convention` | *absent* | `"exp(-2j beta_v d_L) / exp(-1j beta_v (d_L+d_R))"` | — |
| `/referee/slab_airy/coarse|false/phase_bins_masked_s11` | *absent* | `[8400000000.0, 8600000000.0]` | 0–1 |
| `/referee/slab_airy/coarse|false/phase_bins_masked_s21` | *absent* | `[]` | — |
| `/referee/slab_airy/coarse|false/phase_mask_floor` | *absent* | `0.3` | — |
| `/referee/slab_airy/coarse|false/s11_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `0e89a873fb67` | 0–16 |
| `/referee/slab_airy/coarse|false/s11_max_mag_abs_diff` | *absent* | `0.1486628912567549` | — |
| `/referee/slab_airy/coarse|false/s11_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `6c1f944d60e0` | 0–16 |
| `/referee/slab_airy/coarse|false/s21_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `5800c95119ac` | 0–16 |
| `/referee/slab_airy/coarse|false/s21_max_mag_abs_diff` | *absent* | `0.11113065170076009` | — |
| `/referee/slab_airy/coarse|false/s21_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `29f492730862` | 0–16 |
| `/referee/slab_airy/coarse|false/worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/referee/slab_airy/coarse|flux/d_left_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/coarse|flux/d_right_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/coarse|flux/gate_mag` | *absent* | `0.05` | — |
| `/referee/slab_airy/coarse|flux/gate_phase_deg` | *absent* | `15.0` | — |
| `/referee/slab_airy/coarse|flux/max_mag_abs_diff` | *absent* | `0.1466106341286403` | — |
| `/referee/slab_airy/coarse|flux/max_phase_diff_deg` | *absent* | `15.943758600371142` | — |
| `/referee/slab_airy/coarse|flux/max_phase_diff_deg_unmasked` | *absent* | `15.943758600371142` | — |
| `/referee/slab_airy/coarse|flux/oracle_s11` | *absent* | array[17], SHA256 `161336c88d1c` | 0–16 |
| `/referee/slab_airy/coarse|flux/oracle_s21` | *absent* | array[17], SHA256 `0e4c6e9bb45e` | 0–16 |
| `/referee/slab_airy/coarse|flux/oracle_shift_convention` | *absent* | `"exp(-2j beta_v d_L) / exp(-1j beta_v (d_L+d_R))"` | — |
| `/referee/slab_airy/coarse|flux/phase_bins_masked_s11` | *absent* | `[8400000000.0, 8600000000.0]` | 0–1 |
| `/referee/slab_airy/coarse|flux/phase_bins_masked_s21` | *absent* | `[]` | — |
| `/referee/slab_airy/coarse|flux/phase_mask_floor` | *absent* | `0.3` | — |
| `/referee/slab_airy/coarse|flux/s11_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `1db2d65c259e` | 0–16 |
| `/referee/slab_airy/coarse|flux/s11_max_mag_abs_diff` | *absent* | `0.1466106341286403` | — |
| `/referee/slab_airy/coarse|flux/s11_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `dbc2f047c5e5` | 0–16 |
| `/referee/slab_airy/coarse|flux/s21_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `514555cd0053` | 0–16 |
| `/referee/slab_airy/coarse|flux/s21_max_mag_abs_diff` | *absent* | `0.10004171161450481` | — |
| `/referee/slab_airy/coarse|flux/s21_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `acc400138f23` | 0–16 |
| `/referee/slab_airy/coarse|flux/worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/referee/slab_airy/fine|false/d_left_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/fine|false/d_right_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/fine|false/gate_mag` | *absent* | `0.05` | — |
| `/referee/slab_airy/fine|false/gate_phase_deg` | *absent* | `15.0` | — |
| `/referee/slab_airy/fine|false/max_mag_abs_diff` | *absent* | `0.012619529027898757` | — |
| `/referee/slab_airy/fine|false/max_phase_diff_deg` | *absent* | `6.098643058445276` | — |
| `/referee/slab_airy/fine|false/max_phase_diff_deg_unmasked` | *absent* | `6.098643058445276` | — |
| `/referee/slab_airy/fine|false/oracle_s11` | *absent* | array[17], SHA256 `161336c88d1c` | 0–16 |
| `/referee/slab_airy/fine|false/oracle_s21` | *absent* | array[17], SHA256 `0e4c6e9bb45e` | 0–16 |
| `/referee/slab_airy/fine|false/oracle_shift_convention` | *absent* | `"exp(-2j beta_v d_L) / exp(-1j beta_v (d_L+d_R))"` | — |
| `/referee/slab_airy/fine|false/phase_bins_masked_s11` | *absent* | `[8400000000.0, 8600000000.0]` | 0–1 |
| `/referee/slab_airy/fine|false/phase_bins_masked_s21` | *absent* | `[]` | — |
| `/referee/slab_airy/fine|false/phase_mask_floor` | *absent* | `0.3` | — |
| `/referee/slab_airy/fine|false/s11_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `b534e5ecd9a1` | 0–16 |
| `/referee/slab_airy/fine|false/s11_max_mag_abs_diff` | *absent* | `0.012619529027898757` | — |
| `/referee/slab_airy/fine|false/s11_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `f912dd9bd348` | 0–16 |
| `/referee/slab_airy/fine|false/s21_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `905e0d9747a1` | 0–16 |
| `/referee/slab_airy/fine|false/s21_max_mag_abs_diff` | *absent* | `0.007350819916518869` | — |
| `/referee/slab_airy/fine|false/s21_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `92486f7a1790` | 0–16 |
| `/referee/slab_airy/fine|false/worst_bin_hz` | *absent* | `8600000000.0` | — |
| `/referee/slab_airy/fine|flux/d_left_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/fine|flux/d_right_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/fine|flux/gate_mag` | *absent* | `0.05` | — |
| `/referee/slab_airy/fine|flux/gate_phase_deg` | *absent* | `15.0` | — |
| `/referee/slab_airy/fine|flux/max_mag_abs_diff` | *absent* | `0.009028712300978847` | — |
| `/referee/slab_airy/fine|flux/max_phase_diff_deg` | *absent* | `6.149738805854002` | — |
| `/referee/slab_airy/fine|flux/max_phase_diff_deg_unmasked` | *absent* | `6.149738805854002` | — |
| `/referee/slab_airy/fine|flux/oracle_s11` | *absent* | array[17], SHA256 `161336c88d1c` | 0–16 |
| `/referee/slab_airy/fine|flux/oracle_s21` | *absent* | array[17], SHA256 `0e4c6e9bb45e` | 0–16 |
| `/referee/slab_airy/fine|flux/oracle_shift_convention` | *absent* | `"exp(-2j beta_v d_L) / exp(-1j beta_v (d_L+d_R))"` | — |
| `/referee/slab_airy/fine|flux/phase_bins_masked_s11` | *absent* | `[8400000000.0, 8600000000.0]` | 0–1 |
| `/referee/slab_airy/fine|flux/phase_bins_masked_s21` | *absent* | `[]` | — |
| `/referee/slab_airy/fine|flux/phase_mask_floor` | *absent* | `0.3` | — |
| `/referee/slab_airy/fine|flux/s11_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `b9cd8f5ef576` | 0–16 |
| `/referee/slab_airy/fine|flux/s11_max_mag_abs_diff` | *absent* | `0.009028712300978847` | — |
| `/referee/slab_airy/fine|flux/s11_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `180b4e429a9b` | 0–16 |
| `/referee/slab_airy/fine|flux/s21_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `eb54a782103f` | 0–16 |
| `/referee/slab_airy/fine|flux/s21_max_mag_abs_diff` | *absent* | `0.005513721686083772` | — |
| `/referee/slab_airy/fine|flux/s21_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `ffd702ba6a59` | 0–16 |
| `/referee/slab_airy/fine|flux/worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/referee/slab_airy/mid|false/d_left_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/mid|false/d_right_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/mid|false/gate_mag` | *absent* | `0.05` | — |
| `/referee/slab_airy/mid|false/gate_phase_deg` | *absent* | `15.0` | — |
| `/referee/slab_airy/mid|false/max_mag_abs_diff` | *absent* | `0.0396771010605817` | — |
| `/referee/slab_airy/mid|false/max_phase_diff_deg` | *absent* | `10.053906726298727` | — |
| `/referee/slab_airy/mid|false/max_phase_diff_deg_unmasked` | *absent* | `10.053906726298727` | — |
| `/referee/slab_airy/mid|false/oracle_s11` | *absent* | array[17], SHA256 `161336c88d1c` | 0–16 |
| `/referee/slab_airy/mid|false/oracle_s21` | *absent* | array[17], SHA256 `0e4c6e9bb45e` | 0–16 |
| `/referee/slab_airy/mid|false/oracle_shift_convention` | *absent* | `"exp(-2j beta_v d_L) / exp(-1j beta_v (d_L+d_R))"` | — |
| `/referee/slab_airy/mid|false/phase_bins_masked_s11` | *absent* | `[8400000000.0, 8600000000.0]` | 0–1 |
| `/referee/slab_airy/mid|false/phase_bins_masked_s21` | *absent* | `[]` | — |
| `/referee/slab_airy/mid|false/phase_mask_floor` | *absent* | `0.3` | — |
| `/referee/slab_airy/mid|false/s11_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `4b76e5370b79` | 0–16 |
| `/referee/slab_airy/mid|false/s11_max_mag_abs_diff` | *absent* | `0.0396771010605817` | — |
| `/referee/slab_airy/mid|false/s11_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `ef94d3394979` | 0–16 |
| `/referee/slab_airy/mid|false/s21_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `09c287d00ba7` | 0–16 |
| `/referee/slab_airy/mid|false/s21_max_mag_abs_diff` | *absent* | `0.02725030493367575` | — |
| `/referee/slab_airy/mid|false/s21_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `2031380511ad` | 0–16 |
| `/referee/slab_airy/mid|false/worst_bin_hz` | *absent* | `8600000000.0` | — |
| `/referee/slab_airy/mid|flux/d_left_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/mid|flux/d_right_m` | *absent* | `0.03556000000000001` | — |
| `/referee/slab_airy/mid|flux/gate_mag` | *absent* | `0.05` | — |
| `/referee/slab_airy/mid|flux/gate_phase_deg` | *absent* | `15.0` | — |
| `/referee/slab_airy/mid|flux/max_mag_abs_diff` | *absent* | `0.03616716445747356` | — |
| `/referee/slab_airy/mid|flux/max_phase_diff_deg` | *absent* | `10.225986904095997` | — |
| `/referee/slab_airy/mid|flux/max_phase_diff_deg_unmasked` | *absent* | `10.225986904095997` | — |
| `/referee/slab_airy/mid|flux/oracle_s11` | *absent* | array[17], SHA256 `161336c88d1c` | 0–16 |
| `/referee/slab_airy/mid|flux/oracle_s21` | *absent* | array[17], SHA256 `0e4c6e9bb45e` | 0–16 |
| `/referee/slab_airy/mid|flux/oracle_shift_convention` | *absent* | `"exp(-2j beta_v d_L) / exp(-1j beta_v (d_L+d_R))"` | — |
| `/referee/slab_airy/mid|flux/phase_bins_masked_s11` | *absent* | `[8400000000.0, 8600000000.0]` | 0–1 |
| `/referee/slab_airy/mid|flux/phase_bins_masked_s21` | *absent* | `[]` | — |
| `/referee/slab_airy/mid|flux/phase_mask_floor` | *absent* | `0.3` | — |
| `/referee/slab_airy/mid|flux/s11_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `41850c53f6e9` | 0–16 |
| `/referee/slab_airy/mid|flux/s11_max_mag_abs_diff` | *absent* | `0.03616716445747356` | — |
| `/referee/slab_airy/mid|flux/s11_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `982650ee19c8` | 0–16 |
| `/referee/slab_airy/mid|flux/s21_mag_abs_diff_per_bin` | *absent* | array[17], SHA256 `e46ee560fbf1` | 0–16 |
| `/referee/slab_airy/mid|flux/s21_max_mag_abs_diff` | *absent* | `0.02271557305158267` | — |
| `/referee/slab_airy/mid|flux/s21_phase_diff_deg_per_bin` | *absent* | array[17], SHA256 `6c1e62e9791d` | 0–16 |
| `/referee/slab_airy/mid|flux/worst_bin_hz` | *absent* | `8400000000.0` | — |
| `/schema` | *absent* | `"rfx.waveguide_chain_battery"` | — |
| `/schema_version` | *absent* | `3` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|im_s11/forward_identity_max_scaled_diff` | *absent* | `0.0` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|im_s11/forward_identity_pass` | *absent* | `true` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|im_s11/g_ad` | *absent* | `2.685720920562744` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|im_s11/g_fd` | *absent* | `2.675805609162874` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|im_s11/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|im_s11/rel` | *absent* | `0.0037055424975255347` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|im_s11/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|re_s11/forward_identity_max_scaled_diff` | *absent* | `0.0` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|re_s11/forward_identity_pass` | *absent* | `true` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|re_s11/g_ad` | *absent* | `-1.4919252395629883` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|re_s11/g_fd` | *absent* | `-1.4760772129923894` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|re_s11/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|re_s11/rel` | *absent* | `0.01073658371737267` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|re_s11/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|s11_mag2/forward_identity_max_scaled_diff` | *absent* | `0.0` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|s11_mag2/forward_identity_pass` | *absent* | `true` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|s11_mag2/g_ad` | *absent* | `0.0007709434721618891` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|s11_mag2/g_fd` | *absent* | `0.0007643310316463037` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|s11_mag2/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|s11_mag2/rel` | *absent* | `0.008651278362129002` | — |
| `/section_4_falsifier/legs/pec_short|false|eps|s11_mag2/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/pec_short|false|sigma|s11_mag2/forward_identity_max_scaled_diff` | *absent* | `0.0` | — |
| `/section_4_falsifier/legs/pec_short|false|sigma|s11_mag2/forward_identity_pass` | *absent* | `true` | — |
| `/section_4_falsifier/legs/pec_short|false|sigma|s11_mag2/g_ad` | *absent* | `-6.428282737731934` | — |
| `/section_4_falsifier/legs/pec_short|false|sigma|s11_mag2/g_fd` | *absent* | `-6.431459317910304` | — |
| `/section_4_falsifier/legs/pec_short|false|sigma|s11_mag2/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/pec_short|false|sigma|s11_mag2/rel` | *absent* | `0.00049391281532699` | — |
| `/section_4_falsifier/legs/pec_short|false|sigma|s11_mag2/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|im_s11/forward_identity_max_scaled_diff` | *absent* | `1.0832146062634656` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|im_s11/forward_identity_pass` | *absent* | `false` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|im_s11/g_ad` | *absent* | `2.6881139278411865` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|im_s11/g_fd` | *absent* | `2.678089616806154` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|im_s11/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|im_s11/rel` | *absent* | `0.0037430827452993374` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|im_s11/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|re_s11/forward_identity_max_scaled_diff` | *absent* | `1.0832146062634656` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|re_s11/forward_identity_pass` | *absent* | `false` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|re_s11/g_ad` | *absent* | `-1.478940725326538` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|re_s11/g_fd` | *absent* | `-1.4632890690252665` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|re_s11/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|re_s11/rel` | *absent* | `0.010696216238188394` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|re_s11/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|s11_mag2/forward_identity_max_scaled_diff` | *absent* | `1.0832146062634656` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|s11_mag2/forward_identity_pass` | *absent* | `false` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|s11_mag2/g_ad` | *absent* | `2.786023287626449e-05` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|s11_mag2/g_fd` | *absent* | `-5.1541562129386875e-08` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|s11_mag2/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|s11_mag2/rel` | *absent* | `541.5391634488263` | — |
| `/section_4_falsifier/legs/pec_short|flux|eps|s11_mag2/verdict` | *absent* | `"fail"` | — |
| `/section_4_falsifier/legs/pec_short|flux|sigma|s11_mag2/forward_identity_max_scaled_diff` | *absent* | `1.5148748027237617` | — |
| `/section_4_falsifier/legs/pec_short|flux|sigma|s11_mag2/forward_identity_pass` | *absent* | `false` | — |
| `/section_4_falsifier/legs/pec_short|flux|sigma|s11_mag2/g_ad` | *absent* | `-6.4214372634887695` | — |
| `/section_4_falsifier/legs/pec_short|flux|sigma|s11_mag2/g_fd` | *absent* | `-6.424585588911291` | — |
| `/section_4_falsifier/legs/pec_short|flux|sigma|s11_mag2/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/pec_short|flux|sigma|s11_mag2/rel` | *absent* | `0.0004900433466020363` | — |
| `/section_4_falsifier/legs/pec_short|flux|sigma|s11_mag2/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/slab|false|eps|im_s21/forward_identity_max_scaled_diff` | *absent* | `0.0` | — |
| `/section_4_falsifier/legs/slab|false|eps|im_s21/forward_identity_pass` | *absent* | `true` | — |
| `/section_4_falsifier/legs/slab|false|eps|im_s21/g_ad` | *absent* | `0.44848573207855225` | — |
| `/section_4_falsifier/legs/slab|false|eps|im_s21/g_fd` | *absent* | `0.44857854087840127` | — |
| `/section_4_falsifier/legs/slab|false|eps|im_s21/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/slab|false|eps|im_s21/rel` | *absent* | `0.00020689531796880658` | — |
| `/section_4_falsifier/legs/slab|false|eps|im_s21/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/slab|false|eps|re_s21/forward_identity_max_scaled_diff` | *absent* | `0.0` | — |
| `/section_4_falsifier/legs/slab|false|eps|re_s21/forward_identity_pass` | *absent* | `true` | — |
| `/section_4_falsifier/legs/slab|false|eps|re_s21/g_ad` | *absent* | `0.09743057191371918` | — |
| `/section_4_falsifier/legs/slab|false|eps|re_s21/g_fd` | *absent* | `0.09710362576527354` | — |
| `/section_4_falsifier/legs/slab|false|eps|re_s21/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/slab|false|eps|re_s21/rel` | *absent* | `0.0033669818801201003` | — |
| `/section_4_falsifier/legs/slab|false|eps|re_s21/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/slab|false|eps|s11_mag2/forward_identity_max_scaled_diff` | *absent* | `0.0` | — |
| `/section_4_falsifier/legs/slab|false|eps|s11_mag2/forward_identity_pass` | *absent* | `true` | — |
| `/section_4_falsifier/legs/slab|false|eps|s11_mag2/g_ad` | *absent* | `0.313778817653656` | — |
| `/section_4_falsifier/legs/slab|false|eps|s11_mag2/g_fd` | *absent* | `0.3137402164077263` | — |
| `/section_4_falsifier/legs/slab|false|eps|s11_mag2/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/slab|false|eps|s11_mag2/rel` | *absent* | `0.00012303569612997145` | — |
| `/section_4_falsifier/legs/slab|false|eps|s11_mag2/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/slab|false|eps|s21_mag2/forward_identity_max_scaled_diff` | *absent* | `0.0` | — |
| `/section_4_falsifier/legs/slab|false|eps|s21_mag2/forward_identity_pass` | *absent* | `true` | — |
| `/section_4_falsifier/legs/slab|false|eps|s21_mag2/g_ad` | *absent* | `-0.31374168395996094` | — |
| `/section_4_falsifier/legs/slab|false|eps|s21_mag2/g_fd` | *absent* | `-0.31370325728784954` | — |
| `/section_4_falsifier/legs/slab|false|eps|s21_mag2/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/slab|false|eps|s21_mag2/rel` | *absent* | `0.00012249369816438543` | — |
| `/section_4_falsifier/legs/slab|false|eps|s21_mag2/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/slab|flux|eps|im_s21/forward_identity_max_scaled_diff` | *absent* | `1.7607200487636583` | — |
| `/section_4_falsifier/legs/slab|flux|eps|im_s21/forward_identity_pass` | *absent* | `false` | — |
| `/section_4_falsifier/legs/slab|flux|eps|im_s21/g_ad` | *absent* | `0.4507231116294861` | — |
| `/section_4_falsifier/legs/slab|flux|eps|im_s21/g_fd` | *absent* | `0.4508149179077123` | — |
| `/section_4_falsifier/legs/slab|flux|eps|im_s21/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/slab|flux|eps|im_s21/rel` | *absent* | `0.00020364516474360742` | — |
| `/section_4_falsifier/legs/slab|flux|eps|im_s21/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/slab|flux|eps|re_s21/forward_identity_max_scaled_diff` | *absent* | `1.7607200487636583` | — |
| `/section_4_falsifier/legs/slab|flux|eps|re_s21/forward_identity_pass` | *absent* | `false` | — |
| `/section_4_falsifier/legs/slab|flux|eps|re_s21/g_ad` | *absent* | `0.09770140051841736` | — |
| `/section_4_falsifier/legs/slab|flux|eps|re_s21/g_fd` | *absent* | `0.09737160550903456` | — |
| `/section_4_falsifier/legs/slab|flux|eps|re_s21/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/slab|flux|eps|re_s21/rel` | *absent* | `0.003386973108420135` | — |
| `/section_4_falsifier/legs/slab|flux|eps|re_s21/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s11_mag2/forward_identity_max_scaled_diff` | *absent* | `1.7607200487636583` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s11_mag2/forward_identity_pass` | *absent* | `false` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s11_mag2/g_ad` | *absent* | `0.3149757981300354` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s11_mag2/g_fd` | *absent* | `0.3149354506093338` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s11_mag2/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s11_mag2/rel` | *absent* | `0.00012811362018319986` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s11_mag2/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s21_mag2/forward_identity_max_scaled_diff` | *absent* | `1.7607200487636583` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s21_mag2/forward_identity_pass` | *absent* | `false` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s21_mag2/g_ad` | *absent* | `-0.3149767816066742` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s21_mag2/g_fd` | *absent* | `-0.3149360096865039` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s21_mag2/primary_precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s21_mag2/rel` | *absent* | `0.0001294609664066482` | — |
| `/section_4_falsifier/legs/slab|flux|eps|s21_mag2/verdict` | *absent* | `"pass"` | — |
| `/section_4_falsifier/n_legs` | *absent* | `16` | — |
| `/section_4_falsifier/n_red` | *absent* | `9` | — |
| `/section_4_falsifier/provenance/commit` | *absent* | `"f914a7caf1ff8c63cac6f5f8c975b7f9f420a0c7"` | — |
| `/section_4_falsifier/provenance/jax_devices` | *absent* | `["cuda:0"]` | 0 |
| `/section_4_falsifier/provenance/jax_enable_x64` | *absent* | `false` | — |
| `/section_4_falsifier/provenance/jax_version` | *absent* | `"0.4.33.dev20241023+e3c6d6430"` | — |
| `/section_4_falsifier/provenance/precision` | *absent* | `"float32"` | — |
| `/section_4_falsifier/red_keys` | *absent* | array[9], SHA256 `acc5f2cad8cc` | 0–8 |
| `/section_4_falsifier/stage_dir` | *absent* | `"falsifier_float32"` | — |
| `/section_4_falsifier/what` | *absent* | `"the ad_fd stage re-run in the same pod with RFX_CHAIN_PRIMARY=float32 (closing pre-declaration section 4); must reproduce run 2's 9 red"` | — |
| `/shift_pair_name` | *absent* | `"sign_discriminating_pair"` | — |
| `/supersedes` | *absent* | `"tests/fixtures/waveguide_chain_battery/fixture_guide_cell_aperture.json"` | — |
| `/supersedes_reason` | *absent* | `"same port, same battery: this artifact reads contract criterion 1 (forward identity) and 3(a) (AD-vs-FD) under x64 on the flux lane per the v1.8 closing declaration, stores the float32 reading beside it, and carries the pre-declared zero-derivative leg as report_only"` | — |
| `/verdicts/ad_vs_fd|pec_short|false|eps|im_s11` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|pec_short|false|eps|re_s11` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|pec_short|false|eps|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|pec_short|false|sigma|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|pec_short|flux|eps|im_s11` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|pec_short|flux|eps|re_s11` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|pec_short|flux|eps|s11_mag2` | *absent* | `"report_only"` | — |
| `/verdicts/ad_vs_fd|pec_short|flux|sigma|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|slab|false|eps|im_s21` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|slab|false|eps|re_s21` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|slab|false|eps|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|slab|false|eps|s21_mag2` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|slab|flux|eps|im_s21` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|slab|flux|eps|re_s21` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|slab|flux|eps|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/ad_vs_fd|slab|flux|eps|s21_mag2` | *absent* | `"pass"` | — |
| `/verdicts/cheap_refute_flip_shift_sign` | *absent* | `"pass"` | — |
| `/verdicts/column_power|pec_short|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/column_power|pec_short|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/column_power|pec_short|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/column_power|pec_short|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/column_power|pec_short|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/column_power|pec_short|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/column_power|slab|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/column_power|slab|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/column_power|slab|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/column_power|slab|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/column_power|slab|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/column_power|slab|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/forward_identity|pec_short|false|eps|im_s11` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|pec_short|false|eps|re_s11` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|pec_short|false|eps|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|pec_short|false|sigma|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|pec_short|flux|eps|im_s11` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|pec_short|flux|eps|re_s11` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|pec_short|flux|eps|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|pec_short|flux|sigma|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|slab|false|eps|im_s21` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|slab|false|eps|re_s21` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|slab|false|eps|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|slab|false|eps|s21_mag2` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|slab|flux|eps|im_s21` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|slab|flux|eps|re_s21` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|slab|flux|eps|s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/forward_identity|slab|flux|eps|s21_mag2` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|pec_short|false|eps:s11_complex` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|pec_short|false|eps:s11_mag2` | *absent* | `"report_only"` | — |
| `/verdicts/gradient_invariance|pec_short|false|sigma:s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|pec_short|flux|eps:s11_complex` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|pec_short|flux|eps:s11_mag2` | *absent* | `"report_only"` | — |
| `/verdicts/gradient_invariance|pec_short|flux|sigma:s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|slab|false|eps:s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|slab|false|eps:s21_complex` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|slab|false|eps:s21_mag2` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|slab|flux|eps:s11_mag2` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|slab|flux|eps:s21_complex` | *absent* | `"pass"` | — |
| `/verdicts/gradient_invariance|slab|flux|eps:s21_mag2` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|pec_short_s11_mag|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|pec_short_s11_mag|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|pec_short_s11_phase_deg|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|pec_short_s11_phase_deg|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|slab_s11_mag|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|slab_s11_mag|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|slab_s21_mag|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|slab_s21_mag|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|slab_s21_phase_deg|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_monotone|slab_s21_phase_deg|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder_richardson|pec_short_s11_phase_deg|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_richardson|pec_short_s11_phase_deg|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder_richardson|slab_s11_mag|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_richardson|slab_s11_mag|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder_richardson|slab_s21_mag|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_richardson|slab_s21_mag|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder_richardson|slab_s21_phase_deg|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder_richardson|slab_s21_phase_deg|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder|pec_short_s11_mag|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder|pec_short_s11_mag|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder|pec_short_s11_phase_deg|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder|pec_short_s11_phase_deg|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder|slab_s11_mag|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder|slab_s11_mag|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder|slab_s21_mag|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder|slab_s21_mag|flux` | *absent* | `"pass"` | — |
| `/verdicts/ladder|slab_s21_phase_deg|false` | *absent* | `"pass"` | — |
| `/verdicts/ladder|slab_s21_phase_deg|flux` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|pec_short|coarse|false` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|pec_short|coarse|flux` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|pec_short|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|pec_short|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|pec_short|mid|false` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|pec_short|mid|flux` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|slab|coarse|false` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|slab|coarse|flux` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|slab|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|slab|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|slab|mid|false` | *absent* | `"pass"` | — |
| `/verdicts/non_vacuity|slab|mid|flux` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_abs_s|pec_short|false` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_abs_s|pec_short|flux` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_abs_s|slab|false` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_abs_s|slab|flux` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_rotation_continuous|pec_short|false` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_rotation_continuous|pec_short|flux` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_rotation_continuous|slab|false` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_rotation_continuous|slab|flux` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_rotation_yee|pec_short|false` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_rotation_yee|pec_short|flux` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_rotation_yee|slab|false` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_rotation_yee|slab|flux` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_wrong_sign|pec_short|false` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_wrong_sign|pec_short|flux` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_wrong_sign|slab|false` | *absent* | `"pass"` | — |
| `/verdicts/plane_shift_wrong_sign|slab|flux` | *absent* | `"pass"` | — |
| `/verdicts/power_closure|pec_short|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|pec_short|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|pec_short|fine|false` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|pec_short|fine|flux` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|pec_short|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|pec_short|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|slab|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|slab|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|slab|fine|false` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|slab|fine|flux` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|slab|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/power_closure|slab|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_complex|pec_short|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_complex|pec_short|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_complex|pec_short|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/reciprocity_complex|pec_short|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/reciprocity_complex|pec_short|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_complex|pec_short|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_complex|slab|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_complex|slab|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_complex|slab|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/reciprocity_complex|slab|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/reciprocity_complex|slab|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_complex|slab|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_mag|pec_short|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_mag|pec_short|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_mag|pec_short|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/reciprocity_mag|pec_short|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/reciprocity_mag|pec_short|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_mag|pec_short|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_mag|slab|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_mag|slab|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_mag|slab|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/reciprocity_mag|slab|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/reciprocity_mag|slab|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/reciprocity_mag|slab|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/referee_pec_short|pec_short|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/referee_pec_short|pec_short|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/referee_pec_short|pec_short|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/referee_pec_short|pec_short|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/referee_pec_short|pec_short|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/referee_pec_short|pec_short|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/referee_slab_airy_mag|slab|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/referee_slab_airy_mag|slab|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/referee_slab_airy_mag|slab|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/referee_slab_airy_mag|slab|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/referee_slab_airy_mag|slab|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/referee_slab_airy_mag|slab|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/referee_slab_airy_phase|slab|coarse|false` | *absent* | `"report_only"` | — |
| `/verdicts/referee_slab_airy_phase|slab|coarse|flux` | *absent* | `"report_only"` | — |
| `/verdicts/referee_slab_airy_phase|slab|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/referee_slab_airy_phase|slab|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/referee_slab_airy_phase|slab|mid|false` | *absent* | `"report_only"` | — |
| `/verdicts/referee_slab_airy_phase|slab|mid|flux` | *absent* | `"report_only"` | — |
| `/verdicts/settling|pec_short|coarse|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|pec_short|coarse|flux` | *absent* | `"pass"` | — |
| `/verdicts/settling|pec_short|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|pec_short|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/settling|pec_short|mid|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|pec_short|mid|flux` | *absent* | `"pass"` | — |
| `/verdicts/settling|slab|coarse|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|slab|coarse|flux` | *absent* | `"pass"` | — |
| `/verdicts/settling|slab|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|slab|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/settling|slab|mid|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|slab|mid|flux` | *absent* | `"pass"` | — |
| `/verdicts/settling|thru|coarse|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|thru|coarse|flux` | *absent* | `"pass"` | — |
| `/verdicts/settling|thru|fine|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|thru|fine|flux` | *absent* | `"pass"` | — |
| `/verdicts/settling|thru|mid|false` | *absent* | `"pass"` | — |
| `/verdicts/settling|thru|mid|flux` | *absent* | `"pass"` | — |

## tests/fixtures/waveguide_vi_envelope/s21_phase_residual_witness.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/absorber_K` | *absent* | `3.0` | — |
| `/bands/R0/cases` | *absent* | `["R0_N9_K3p0", "R0_N18_K3p0", "R0_N36_K3p0"]` | 0–2 |
| `/bands/R0/pairwise_orders` | *absent* | `[2.0056790647093963, 2.0054730515955637]` | 0–1 |
| `/bands/R0/r_hi` | *absent* | `1.03` | — |
| `/bands/R0/r_lo` | *absent* | `1.01` | — |
| `/bands/R0/rms_deg` | *absent* | `[30.377597134811392, 7.564563242298626, 1.8839801126352582]` | 0–2 |
| `/bands/R0/rungs` | *absent* | `[9, 18, 36]` | 0–2 |
| `/bands/R1/cases` | *absent* | `["R1_N9_K3p0", "R1_N18_K3p0", "R1_N36_K3p0", "R1_N72_K3p0"]` | 0–3 |
| `/bands/R1/pairwise_orders` | *absent* | `[2.000977682614266, 2.00644651044904, 2.026789924033475]` | 0–2 |
| `/bands/R1/r_hi` | *absent* | `1.045` | — |
| `/bands/R1/r_lo` | *absent* | `1.017` | — |
| `/bands/R1/rms_deg` | *absent* | `[19.648107421694668, 4.908699210808957, 1.2217035516113623, 0.2998066587555095]` | 0–3 |
| `/bands/R1/rungs` | *absent* | `[9, 18, 36, 72]` | 0–3 |
| `/bands/R2/cases` | *absent* | `["R2_N9_K3p0", "R2_N18_K3p0", "S0b_R2_N36_K3p0", "S0c_R2_N72_K3p0"]` | 0–3 |
| `/bands/R2/pairwise_orders` | *absent* | `[1.9970853290410409, 2.0016526922408215, 2.018812764388554]` | 0–2 |
| `/bands/R2/r_hi` | *absent* | `1.06` | — |
| `/bands/R2/r_lo` | *absent* | `1.023` | — |
| `/bands/R2/rms_deg` | *absent* | `[15.172081536742073, 3.80069115393774, 0.9490849327467457, 0.2341972984778569]` | 0–3 |
| `/bands/R2/rungs` | *absent* | `[9, 18, 36, 72]` | 0–3 |
| `/bands/R3/cases` | *absent* | `["R3_N9_K3p0", "R3_N18_K3p0", "R3_N36_K3p0", "R3_N72_K3p0"]` | 0–3 |
| `/bands/R3/pairwise_orders` | *absent* | `[1.9913563284528664, 1.9910656847243757, 1.9692486662239044]` | 0–2 |
| `/bands/R3/r_hi` | *absent* | `1.08` | — |
| `/bands/R3/r_lo` | *absent* | `1.03` | — |
| `/bands/R3/rms_deg` | *absent* | `[12.179242360075058, 3.063107832769331, 0.7705339726556358, 0.19678358757552533]` | 0–3 |
| `/bands/R3/rungs` | *absent* | `[9, 18, 36, 72]` | 0–3 |
| `/bands/R4/cases` | *absent* | `["R4_N9_K3p0", "R4_N18_K3p0", "R4_N36_K3p0", "R4_N72_K3p0"]` | 0–3 |
| `/bands/R4/pairwise_orders` | *absent* | `[1.9733703368128, 1.9944726098812193, 2.00351498259742]` | 0–2 |
| `/bands/R4/r_hi` | *absent* | `1.16` | — |
| `/bands/R4/r_lo` | *absent* | `1.08` | — |
| `/bands/R4/rms_deg` | *absent* | `[5.5371309298680895, 1.4100714808648416, 0.35387106070693813, 0.08825248465040178]` | 0–3 |
| `/bands/R4/rungs` | *absent* | `[9, 18, 36, 72]` | 0–3 |
| `/bands/R5/cases` | *absent* | `["R5_N9_K3p0", "R5_N18_K3p0", "R5_N36_K3p0", "S0a_R5_N72_K3p0"]` | 0–3 |
| `/bands/R5/pairwise_orders` | *absent* | `[2.017810602723836, 2.000812059025736, 1.9948136285569893]` | 0–2 |
| `/bands/R5/r_hi` | *absent* | `1.769063850165303` | — |
| `/bands/R5/r_lo` | *absent* | `1.2810462363265989` | — |
| `/bands/R5/rms_deg` | *absent* | `[1.6071297235989668, 0.3968527703568656, 0.0991573635360475, 0.024878616924092684]` | 0–3 |
| `/bands/R5/rungs` | *absent* | `[9, 18, 36, 72]` | 0–3 |
| `/bands/R7/cases` | *absent* | `["R7_N18_K3p0", "R7_N36_K3p0", "R7_N72_K3p0"]` | 0–2 |
| `/bands/R7/pairwise_orders` | *absent* | `[2.0387192027460146, 2.0114277456028247]` | 0–1 |
| `/bands/R7/r_hi` | *absent* | `2.18` | — |
| `/bands/R7/r_lo` | *absent* | `2.05` | — |
| `/bands/R7/rms_deg` | *absent* | `[1.1927952527335304, 0.2903021613322468, 0.07200293235979813]` | 0–2 |
| `/bands/R7/rungs` | *absent* | `[18, 36, 72]` | 0–2 |
| `/invariance_deg/F2_R5_N72_f64` | *absent* | `0.024851256963135893` | — |
| `/invariance_deg/R2_N18_K3p0` | *absent* | `3.80069115393774` | — |
| `/invariance_deg/R2_N18_K3p0_t10` | *absent* | `3.8034928148272624` | — |
| `/invariance_deg/R2_N36_K3.0` | *absent* | `0.9490849327467457` | — |
| `/invariance_deg/R2_N36_K4.5` | *absent* | `0.9494079300907505` | — |
| `/invariance_deg/R2_N36_K6.0` | *absent* | `0.9492887687821487` | — |
| `/invariance_deg/R2_N36_K9.0` | *absent* | `0.9492896182389042` | — |
| `/invariance_deg/S0a_R5_N72_K3p0` | *absent* | `0.024878616924092684` | — |
| `/record_rule` | *absent* | `"interior n_trav=4 (the phase residual is record-invariant; see invariance)"` | — |
| `/rfx_sha` | *absent* | `"b59e1d991dd62868bdf8689a1f642eeb8f7c5b89"` | — |
| `/run` | *absent* | `"369367258390"` | — |
| `/what` | *absent* | `"&#124;S21&#124; phase residual against analytic -beta*L, RMS over bins, deg. beta from the DISCRETE TE10 cutoff (port_f_cutoff_hz), L = reference-plane separation. The one observable in the sweep that does not pass through the absorber: invariant to absorber thickness, record length and precision. Converges at second order at every band down to f/f_c=1.010, where the reflection headline reads ~1.1. This is the discretization witness for #894."` | — |

## tests/fixtures/wr90_iris_filter/fixture.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/absorber_depth_witness/d_hi_mhz` | `0.06` | `0.01` | — |
| `/absorber_depth_witness/d_lo_mhz` | `0.02` | `0.0` | — |
| `/absorber_depth_witness/deep/d_hi_mhz` | `0.063` | `0.014` | — |
| `/absorber_depth_witness/deep/d_lo_mhz` | `0.015` | `0.004` | — |
| `/absorber_depth_witness/deep/hi` | `11140753280.781218` | `11140910475.46354` | — |
| `/absorber_depth_witness/deep/lo` | `10800389034.893412` | `10800344622.687653` | — |
| `/absorber_depth_witness/deep/s11` | array[131], SHA256 `80701c043a3b` | array[131], SHA256 `835ba4548bb8` | 0–8, 10–15, 17–21, 24–99, 102–108, 110–114, 116–119, 121–122, 125–130 |
| `/absorber_depth_witness/deep/s21` | array[131], SHA256 `f98624952c50` | array[131], SHA256 `cf2a439b0afb` | 0–11, 13–114, 116–130 |
| `/absorber_depth_witness/deep/wall_s` | `1483.5` | `1033.7` | — |
| `/absorber_depth_witness/gated/hi` | `11140816346.186275` | `11140924560.619234` | — |
| `/absorber_depth_witness/gated/lo` | `10800373669.774708` | `10800340317.655672` | — |
| `/absorber_depth_witness/mid/d_hi_mhz` | `0.051` | `0.009` | — |
| `/absorber_depth_witness/mid/d_lo_mhz` | `0.011` | `0.003` | — |
| `/absorber_depth_witness/mid/hi` | `11140765734.608944` | `11140915351.637445` | — |
| `/absorber_depth_witness/mid/lo` | `10800385111.490433` | `10800343078.352245` | — |
| `/absorber_depth_witness/mid/s11` | array[131], SHA256 `ec4862e79b34` | array[131], SHA256 `e803a62b9b4c` | 1–19, 21, 23, 25, 27–96, 100–105, 107–112, 114–116, 118–120, 123, 125–128, 130 |
| `/absorber_depth_witness/mid/s21` | array[131], SHA256 `90f014a0e3ff` | array[131], SHA256 `a96bbee72b56` | 0–10, 12–14, 16–111, 113–114, 116–130 |
| `/absorber_depth_witness/mid/wall_s` | `1324.4` | `940.9` | — |
| `/b_invariance_witness/0/bw` | `340442676.4115677` | `340584242.963562` | — |
| `/b_invariance_witness/0/f0` | `10970595007.980492` | `10970632439.137453` | — |
| `/b_invariance_witness/0/hi` | *absent* | `11140924560.619234` | — |
| `/b_invariance_witness/0/lo` | *absent* | `10800340317.655672` | — |
| `/b_invariance_witness/0/s11` | array[131], SHA256 `19dd17dae901` | array[131], SHA256 `a8d2053b1cec` | 0–18, 20–26, 28–99, 101–106, 108–113, 115–122, 124–130 |
| `/b_invariance_witness/0/s21` | array[131], SHA256 `74e9a501b715` | array[131], SHA256 `b14514c8010c` | 0–130 |
| `/b_invariance_witness/0/wall_s` | `1160.3` | `850.0` | — |
| `/b_invariance_witness/1/bw` | `340442419.3242817` | `340584420.32989883` | — |
| `/b_invariance_witness/1/f0` | `10970594978.610989` | `10970632624.384037` | — |
| `/b_invariance_witness/1/hi` | *absent* | `11140924834.548986` | — |
| `/b_invariance_witness/1/lo` | *absent* | `10800340414.219088` | — |
| `/b_invariance_witness/1/max_dev_vs_b4` | `29.369503021240234` | `185.24658393859863` | — |
| `/b_invariance_witness/1/s11` | array[131], SHA256 `c1cc2ac94cf2` | array[131], SHA256 `9319cb5d9744` | 0–16, 18, 20–23, 25–26, 28–99, 101–107, 109, 111–130 |
| `/b_invariance_witness/1/s21` | array[131], SHA256 `1f66898b2053` | array[131], SHA256 `368ba5c1b004` | 0–130 |
| `/b_invariance_witness/1/wall_s` | `1526.9` | `1037.1` | — |
| `/b_invariance_witness/2/bw` | `340442195.4324913` | `340583617.81036377` | — |
| `/b_invariance_witness/2/f0` | `10970594855.77844` | `10970632206.62834` | — |
| `/b_invariance_witness/2/hi` | *absent* | `11140924015.533522` | — |
| `/b_invariance_witness/2/lo` | *absent* | `10800340397.723158` | — |
| `/b_invariance_witness/2/max_dev_vs_b4` | `152.20205116271973` | `232.50911331176758` | — |
| `/b_invariance_witness/2/s11` | array[131], SHA256 `e9e31388046c` | array[131], SHA256 `a9e1bfc92400` | 0–7, 9–16, 18–26, 28–99, 102–104, 106–109, 111–122, 124–130 |
| `/b_invariance_witness/2/s21` | array[131], SHA256 `b3770b632d22` | array[131], SHA256 `27a9f50e765a` | 0–130 |
| `/b_invariance_witness/2/wall_s` | `1938.9` | `1374.6` | — |
| `/claim_scope` | `"A published 4th-order WR-90 inductive-iris bandpass filter (Aghanim et al., E3S Web of Conferences 351, 01059 (2022), CC BY 4.0, Table 6 optimized: five irises t = 2.00 mm, apertures 10.27/6.65/6.18/6.65/10.27 mm, cavities 14.29/15.73/15.73/14.29 mm) built at dx = a/90 and compared against a TEn0 mode-matching cascade oracle over 10.40-11.70 GHz on 131 points at 10 MHz. Stage S3 of the waveguide-obstacle campaign and the first RESONANT multi-obstacle case in the lane: unlike the single iris of S1, a per-face geometry error here is a passband shift rather than a magnitude tolerance. GATED: centre frequency f0 within 19 MHz = round-up(measured envelope 12.1230 x 1.5); the structural reflection-zero COUNT (an integer, depth-independent); and passband CONTIGUITY as a regression lock (span_holes <= 1, the committed envelope) -- added after a post-merge joint review showed f0 is computed from the OUTERMOST -10 dB crossings, so a future regeneration whose passband split into separated resonances could have shipped green with its bridged midpoint inside the f0 gate. All against the oracle evaluated on the AS-REALIZED geometry. Measured d_f0 = +12.08 MHz, zeros 3 vs 3, one interior hole bin. The zero-count gate is additionally witnessed ROBUST to the unsettled iris-thickness convention: an oracle-side sweep of t_elec across 8.00-8.50 cells (covering both candidate conventions; committed as iris_thickness_zero_count_sweep) holds the count at 3 throughout while bandwidth moves 20 MHz across the same band, so the gated integer does not depend on which convention the comparator picks. The envelope is a population of NINE configurations over four setup axes, not a single run, and each axis carries an INTERIOR sample as well as an endpoint: guide height b = 4/6/8 cells, run length num_periods 400/600/800, port standoff 3.05/7.62/15.24 mm, absorber depth 0.75/1.00/1.25 lambda_g. The interior samples are the point rather than decoration: a one-alternative-per-axis envelope cannot detect NON-MONOTONIC sensitivity, which is exactly the failure of PR #475, where three sampled clearances passed while 9 of 13 exceeded the gate and the passing three were the sampled ones. Every population member carries its own committed &#124;S11&#124; trace, so each residual is recomputable rather than a free-floating scalar whose integrity is borrowed from asserts living in other tests. WHAT THAT GATE IS AND IS NOT, stated because the phrasing invites more than it delivers: the population makes the envelope ROBUST rather than resting on one datum, but it does not make the gate independent of the datum. The spread is 0.06 MHz while every member's &#124;d_f0&#124; is about 12.08 MHz, so the envelope is dominated by the RESIDUAL and not by the scatter, and gate = round-up(env x 1.5) is therefore 1.5x the measured agreement. This is a REGRESSION LOCK with 50 percent headroom, not an independent accuracy bound, exactly as the merged case 18's gate is; what gives the measured agreement meaning is not the gate but the comparison of that agreement against an external scale, namely the reference's own 21.9 MHz f0 spread between two independent commercial codes. That tightness is the substance of the result: the residual is a reproducible systematic difference rather than a setup artifact, and at the measured cavity sensitivity of -105 MHz/cell it corresponds to about 0.12 cell of cavity length. The num_periods = 200 run is EXCLUDED from the envelope rather than folded in, because it fails the settling criterion at column power 1.207; it stays committed as the evidence that the settling gate can fire. WHY f0 AND NOT BANDWIDTH, which is the correction this case exists to record: the oracle must be fed the geometry that was BUILT, not the geometry that was DRAWN, and the three legs of that convention are not equally settled. The transverse aperture leg d_c*dx is confirmed to better than 0.05 cell by an independent refit of 16 committed case-18 configurations during the #499 review -- a session measurement; the committed corroboration is the per-run raster assert on the open-node count and the exact-mask FDFD agreement. The cavity leg (L_c + 1)*dx - the distance between the bounding zeroed node planes - is confirmed to 0.04-0.17 cell and carries about 105 of the 107.5 MHz that separates a drawn-count oracle from a realized-geometry one. But the IRIS-THICKNESS leg is NOT (t_c - 1)*dx: four independent FDTD runs at drawn t_c = 2/4/6/8 give a flat offset of -0.66/-0.68/-0.68/-0.70 cell, i.e. t_elec is about (t_c - 0.68)*dx, matching neither this case's earlier rule nor the merged case 18's t_c*dx, with a residual 10-33x below both. That leaves an irreducible comparator-input uncertainty of order half a cell, and the gated observable is therefore chosen by SENSITIVITY to it: per cell of convention error, f0 moves about 2.4 MHz, bandwidth about 40 MHz, and individual band edges 22-30 MHz. So f0 and the zero count are gated; band edges and bandwidth are REPORTED, because a +/-20 MHz input uncertainty cannot honestly sit under a 15 MHz gate. Adopting the fitted -0.68 cell would absorb the disagreement into a free parameter, after which the residual would measure nothing - the tautological-validation failure this campaign has hit repeatedly - so the offset is recorded as an uncertainty and NOT adopted. Handing the oracle the drawn cell counts instead of the realized ones biases f0 by +107.5 MHz, five times the reference's own 21.9 MHz CST-vs-HFSS spread, and the envelope-times-1.5 rule does NOT catch that class because the rule bounds SCATTER and this is BIAS. The realized lengths are read back off the rasterized metal and re-derived again from the committed node indices in the frozen gate test. Total electrical length is a face-continuity CHECK across region types, NOT a uniqueness argument: putting the metal/open interface at sigma*dx beyond the outermost metal node gives total = span - 1 + 2*sigma = the outer extent measured at the same sigma, conserved for EVERY sigma, and sigma = 0.5 is the drawn pairing itself. An earlier revision claimed this was \"the only pairing that conserves total electrical length\"; that is false and is withdrawn. Drawn counts are COMPENSATED (t_c = round(t/dx) + 1, L_c = round(L/dx) - 1) so the stated electrical dimensions land on nominal, which is nearest-representable rounding with zero free parameters and no reference number entering - that, not any mesh comparison, is why it is a statement of intent rather than a fit. It is NOT a monotone improvement: at a/60 it moves f0 from -35.8 to +120.2 MHz, because it converts a uniformly-signed set of per-cavity errors into a mixed-sign one. REPORTED, NEVER GATED: individual band edges (+17.08 / +7.09 MHz) and bandwidth (-9.99 MHz), which are ONE fact and not two - d_bw is identically d_hi - d_lo - so the earlier framing of an \"unexplained asymmetric edge residual\" separate from a bandwidth deficit was an algebraic error; worst in-band return loss; individual ripple levels; every reflection-zero DEPTH (four nominally identical equiripple zeros bottom out across a wide spread in the published figure, so the paper's frequency step and not physics sets those depths - zero FREQUENCIES are meaningful, depths are not values); passband contiguity beyond the span_holes <= 1 lock (hole depth, hole position, longest contiguous run); the coarse a/60 rung; and phase. PASSBAND CONTIGUITY IS RECORDED, NOT ASSUMED: lo and hi are the OUTERMOST interpolated -10 dB crossings, so the span between them is not necessarily a passband. The built filter has one 10 MHz bin at 9.80 dB inside its 340 MHz span (longest contiguous -10 dB run 270 MHz), while the oracle on the same geometry is contiguous over 35 bins - a real difference that an earlier revision hid behind a clamped statistic, because worst-RL had been computed as the minimum over samples already filtered to >= 10 dB and therefore could not report a violation at all. The coarse a/60 rung has no meaningful passband: 16 of 24 bins in its nominal span are above -10 dB with an interior trough at 2.73 dB, i.e. two separated resonances. That, and not any gate comparison, is the evidence that the gated mesh had to be a/90. The a/60 rung's own numbers do not disqualify it cleanly: its zero count matches its oracle (2 vs 2) and its f0 residual (+19.85 MHz) is the same ~0.12-cell offset seen at a/90; against the committed 19 MHz constant it happens to fail by 0.85 MHz, but a self-derived envelope-times-1.5 gate would pass it. The broken passband is the disqualifier. SETUP IS GATED SEPARATELY FROM PHYSICS, because a resonant band read off an unsettled or absorber-limited run is not a measurement. The repo's preferred ENERGY-BASED ring-down witness (terminal energy in dB below the post-source peak, rule < -40 dB) is NOT AVAILABLE on this path: it is implemented for the lumped/MSL S-matrix extractor, but compute_waveguide_s_matrix returns no settling_db, and the null fields are committed in every row rather than papered over (filed as an rfx capability gap). The ENFORCED settling criterion is therefore the independent axis: the num_periods scan 400/600/800 holds f0 and BW to under 0.1 MHz (the gate is the 400 -> 800 doubling within one 10 MHz bin), the gated run is passivity-clean at column power 1.0065, and the criterion demonstrably fires -- the np=200 run is excluded non-passive at 1.207. The feed-clearance and absorber-depth scans each hold the edges to one 10 MHz bin across their interior and outer samples (standoff 3.05 -> 7.62 -> 15.24 mm: 0.0/0.1 MHz; absorber 0.75 -> 1.00 -> 1.25 lambda_g: 0.0/0.1 MHz). Absorber depth is scanned because in S1 it was the envelope-limiting term at 0.5 lambda_g; here 0.75 lambda_g is measurably sufficient, which is a negative result worth recording rather than a rule inherited. Extractor warnings and the passivity footprint (the bins where column power exceeds 1.02, not merely the scalar maximum) are committed per row. Guide height is reduced to 4 cells on a MEASURED b-invariance witness: b = 4 and b = 8 agree to 152 Hz in f0 on THIS resonant five-iris filter, not merely on the single iris where it was first measured, which is the 8x saving that makes the case affordable to generate. THE ORACLE HAS HAD AN ADVERSARIAL PASS, and it did not find the residual. Its own witnesses are unitarity 2.2e-15, reciprocity and mirror symmetry exact, and an L -> 0 collapse closing two thin irises onto one thick one; its N=1 centred limit reproduces the merged case-18 oracle to 1.05e-04 in an independent odd-mode formulation, and THAT object is what PR #480 confirmed against a formulation-independent 2-D H-plane FDFD at 5.8e-4 - a comparison the frozen gate test now EXECUTES rather than asserting in prose. Those witnesses have known limits, stated because an earlier revision leaned on them too hard: unitarity constrains only the propagating sub-block, mirror symmetry holds by construction for a symmetric geometry, and injected overlap-integral errors leave the reduction and collapse axes silent because they share the overlap routine, so the gate test's re-typed cascade agreeing to 0.0e+00 is a REGRESSION LOCK and not a second opinion. An independent review then closed the real gaps. What is COMMITTED: the overlap integral is validated against direct numerical quadrature IN CI (216 combinations across three apertures including the as-realized 6.096 mm -- where n*pi/a equals m*pi/d exactly for six mode pairs -- centred and 0.19 mm off-centre, worst deviation bounded at 1e-12 with the small-denominator guard exercised), and the oracle's truncation is witnessed on the GATED observable at generation time: f0 by bisection moves 0.33 MHz for n_a 90 -> 130 and 1.85 MHz for an aperture-mode-count doubling -- the aperture axis is the sensitive one, and both are an order under the 19 MHz gate. Session measurements recorded in the research notes but NOT in this record put the gauge invariance of the sqrt(Y) normalisation at machine precision, truncation saturation near 1 MHz in bandwidth, and inter-cavity evanescent transport near 1.6 MHz; they are corroborating colour rather than load-bearing, because the formulation-level check below subsumes their role. THE FORMULATION-LEVEL CHECK IS NOW DONE, and it lands the residual on the rfx side. A 2-D H-plane FDFD -- scalar Helmholtz on a finite-difference grid with an exact discrete transparent port condition, sharing only numpy and scipy with the cascade and no rfx code path at all (validation/crossval/comparators/fdfd_hplane.py) -- was run on the same electrical geometry, grid-exact at every refinement level. It is FIRST-order convergent, so no single level is meaningful; the record carries THREE levels (r = 2, 3, 4, whose bandwidth deviations from the extrapolate shrink as 1/r: measured ratios 1.55 and 1.33 against the first-order 1.50 and 1.33) and BOTH Richardson estimates, which agree to 0.37 MHz in centre frequency and 0.36 MHz in bandwidth -- the two-estimate consistency protocol the porting handoff mandates before either estimate is trusted. The finer pair gives f0 = 10.95742 GHz and BW = 351.42 MHz against the cascade's 10.95851 and 350.43: agreement to 1.09 MHz in centre frequency and 0.98 MHz in bandwidth between two formulations that share nothing but their numerical libraries, with the extrapolation's own order-assumption uncertainty, not the two-estimate consistency, bounding how much of that gap the FDFD itself owns: the consistency (0.37/0.36 MHz) is agreement between two fits of the SAME first-order model, not an error bound -- fitting the convergence order to the three committed levels edge-by-edge gives p = 1.26 (lower edge) / 1.24 (upper edge) rather than the assumed 1, and re-extrapolating the finer pair at the fitted order moves f0 by +0.68 MHz and bandwidth by -0.65 MHz, so the FDFD owns roughly 0.7-1.0 MHz of the 1.09/0.98 MHz gap; f0 and bandwidth are also algebraic combinations of the SAME two band edges, one confirmation and not two. rfx differs from the FDFD by +13.17 MHz in f0 and -10.97 MHz in bandwidth, essentially the same as it differs from the cascade (+12.08 / -9.99), so the 12 MHz residual is not an oracle error. The FDFD's gates: lossless unitarity is enforced on EVERY evaluation (worst 4.6e-07 across all levels), and the empty-guide transparency gate -- &#124;S11&#124; = 5.0e-14 with &#124;S21&#124; = 1.000000000000, the test that originally caught a missing 1/h in the discrete propagation constant -- runs once per generation and once per CI pass. One defect in the comparator itself was found by an independent port review and fixed before this record was generated: its aperture mask realized every aperture two fine cells wide of the stated convention, a first-order bias that Richardson cancelled -- making the extrapolated numbers right for the wrong per-level geometry -- and that produced a spurious FOURTH reflection zero at the coarser levels. With the mask exact, every level shows THREE zeros, matching the cascade and rfx. Independently, the cascade's zero count was checked against its own aperture-mode truncation, which nobody had done for the COUNT: it is 3 at nb_scale 1.0, 1.5, 2.0 and 3.0, with f0 moving 1.7 MHz over that 3x range. What remains genuinely unexplained is the ~12 MHz rfx residual itself: it is mesh-invariant when expressed in cells (-0.1169 cell at a/90 against -0.1241 at a/60, where dispersion would have given 0.083), so it behaves like a fixed geometric offset rather than a frequency-dependent solver error, but attributing it to a specific convention leg has FAILED: propagating the independently measured iris thickness (t_c - 0.68)*dx through node-plane length conservation overshoots and flips the sign, taking a/90 from +12.08 to -30.62 MHz. That attribution is recorded as falsified, not as pending. _gamma at exact cutoff (k equal to n*pi/w, where the sqrt argument vanishes) is unreachable at these band edges and untested. FENCED: nothing here promotes the lane beyond S1. Multi-iris filters, posts and septa remain EXPERIMENTAL; this measures one published design on one mesh with one gated observable, and certifies neither arbitrary filters nor the a/60 rung. Says nothing about phase, group delay, loss, higher-order-mode ports, or fabrication tolerance. The reference is an ANCHOR, not a solver run: the CST and HFSS scalars are digitized from the paper's Fig. 5, no external solver is invoked here, and the case does not compare rfx against either commercial code on any geometry - so the 12.1 MHz f0 residual against this case's own analytic oracle must not be read as an accuracy claim relative to CST or HFSS. Reported for context and not as a yardstick beaten: the oracle on nominal dimensions sits -6.2 MHz from CST and -28.2 MHz from HFSS in f0, +14.7 MHz from both in bandwidth (against a published inter-solver bandwidth spread of only 0.4 MHz), -0.4 and -1.1 dB in worst return loss, and up to 25.4 and 61.9 MHz in individual reflection-zero frequencies. The bandwidth and zero-frequency misses are larger than the f0 miss and are stated here because quoting only f0 would be selective. The built structure is a SNAPPED Aghanim filter: its centre frequency is within the reference's own solver scatter of CST (though not of HFSS), one of the four structural reflection zeros is lost (4 -> 3, confirmed grid-robust by refining the oracle to 1 MHz, with the loss occurring in the upper band), and worst in-band return loss degrades from 13.82 dB to 10.65 dB by rasterization alone, oracle to oracle, with rfx at 9.80 dB. Say snapped, not equivalent. OBSERVABLE PRIORITY for a resonant structure, as this case measures it: the structural reflection-zero COUNT first (an integer, depth-independent, and shown grid-robust), then centre frequency (least sensitive of the continuous quantities to the unsettled convention, ~2.4 MHz per cell), then band edges and bandwidth (~22-40 MHz per cell, hence reported), then worst return loss, and last individual ripple levels and null depths, which are not values at all. TOPOLOGY FIRST, AND f0 IS NOT EXONERATED: the zero count is the most robust observable, but f0 is not thereby safe -- it carries the +12.08 MHz residual this case gates, and at -105 MHz per cell of cavity length it is the quantity a geometry error moves first. A cell snap is inherently non-uniform, since each cavity rounds independently, so every snap figure quoted here was MEASURED on the as-snapped geometry and none may be re-derived by multiplying a sensitivity coefficient by a half cell."` | `"A published 4th-order WR-90 inductive-iris bandpass filter (Aghanim et al., E3S Web of Conferences 351, 01059 (2022), CC BY 4.0, Table 6 optimized: five irises t = 2.00 mm, apertures 10.27/6.65/6.18/6.65/10.27 mm, cavities 14.29/15.73/15.73/14.29 mm) built at dx = a/90 and compared against a TEn0 mode-matching cascade oracle over 10.40-11.70 GHz on 131 points at 10 MHz. Stage S3 of the waveguide-obstacle campaign and the first RESONANT multi-obstacle case in the lane: unlike the single iris of S1, a per-face geometry error here is a passband shift rather than a magnitude tolerance. GATED: centre frequency f0 within 19 MHz = round-up(measured envelope 12.1219 x 1.5); the structural reflection-zero COUNT (an integer, depth-independent); and passband CONTIGUITY as a regression lock (span_holes <= 1, the committed envelope) -- added after a post-merge joint review showed f0 is computed from the OUTERMOST -10 dB crossings, so a future regeneration whose passband split into separated resonances could have shipped green with its bridged midpoint inside the f0 gate. All against the oracle evaluated on the AS-REALIZED geometry. Measured d_f0 = +12.12 MHz, zeros 3 vs 3, one interior hole bin. The zero-count gate is additionally witnessed ROBUST to a perturbation of the comparator's most length-sensitive input: an oracle-side sweep of t_elec across one full cell CENTRED on the realized iris thickness (committed as iris_thickness_zero_count_sweep) holds the count throughout while bandwidth moves across the same band, so the gated integer does not sit on one exact value. Before #931 that window was one-sided, 8.00-8.50 cells, spanning a genuine disagreement about what the iris thickness realized as; the lattice ownership contract removed the disagreement, so the sweep now measures sensitivity rather than ambiguity. The envelope is a population of NINE configurations over four setup axes, not a single run, and each axis carries an INTERIOR sample as well as an endpoint: guide height b = 4/6/8 cells, run length num_periods 400/600/800, port standoff 3.05/7.62/15.24 mm, absorber depth 0.75/1.00/1.25 lambda_g. The interior samples are the point rather than decoration: a one-alternative-per-axis envelope cannot detect NON-MONOTONIC sensitivity, which is exactly the failure of PR #475, where three sampled clearances passed while 9 of 13 exceeded the gate and the passing three were the sampled ones. Every population member carries its own committed &#124;S11&#124; trace, so each residual is recomputable rather than a free-floating scalar whose integrity is borrowed from asserts living in other tests. WHAT THAT GATE IS AND IS NOT, stated because the phrasing invites more than it delivers: the population makes the envelope ROBUST rather than resting on one datum, but it does not make the gate independent of the datum. The spread is 0.02 MHz while every member's &#124;d_f0&#124; is about 12.12 MHz, so the envelope is dominated by the RESIDUAL and not by the scatter, and gate = round-up(env x 1.5) is therefore 1.5x the measured agreement. This is a REGRESSION LOCK with 50 percent headroom, not an independent accuracy bound, exactly as the merged case 18's gate is; what gives the measured agreement meaning is not the gate but the comparison of that agreement against an external scale, namely the reference's own 21.9 MHz f0 spread between two independent commercial codes. That tightness is the substance of the result: the residual is a reproducible systematic difference rather than a setup artifact, and at the measured cavity sensitivity of -105 MHz/cell it corresponds to about 0.12 cell of cavity length. The num_periods = 200 run is EXCLUDED from the envelope rather than folded in, because it fails the settling criterion at column power 1.207; it stays committed as the evidence that the settling gate can fire. WHY f0 AND NOT BANDWIDTH, which is the correction this case exists to record: the oracle must be fed the geometry that was BUILT, not the geometry that was DRAWN. Under the #931 lattice ownership contract those are the same geometry -- a PEC volume drawn on node planes realizes tangential walls at BOTH faces and shorts every normal edge between them, so realized == drawn on all three legs -- and this case reads all three off realized_pec_edge_masks instead of asserting a locally written rule. HISTORY, kept because the posture was built on it and the numbers are still in the record: until #931 a body's far face was never a wall, so the three legs disagreed with the drawing and with each other. The transverse aperture leg d_c*dx was confirmed to better than 0.05 cell by an independent refit of 16 committed case-18 configurations during the #499 review. The cavity leg was (L_c + 1)*dx -- the distance between the bounding zeroed node planes -- confirmed to 0.04-0.17 cell, and carried about 105 of the 107.5 MHz that separated a drawn-count oracle from a realized-geometry one. The IRIS-THICKNESS leg fitted neither rule: four independent FDTD runs at drawn t_c = 2/4/6/8 gave a flat offset of -0.66/-0.68/-0.68/-0.70 cell, t_elec about (t_c - 0.68)*dx, matching neither this case's (t_c - 1)*dx nor the merged case 18's t_c*dx, with a residual 10-33x below both. That was recorded as an irreducible half-cell comparator-input uncertainty and never adopted as a fitted parameter, which was the right call: it was not a physical property but the signature of a missing far face plus a corner recipe that put every face half a cell off the node planes. Both are gone. The gated observable is still chosen by SENSITIVITY -- per cell of iris thickness, f0 moves about 2.4 MHz, bandwidth about 40 MHz, individual band edges 22-30 MHz -- so f0 and the zero count carry the gates. Band edges and bandwidth stay REPORTED, and #931 replaces the reason rather than the posture: the input uncertainty that made a gate on them dishonest is removed by the contract and the envelope a gate needs is now measured (17.0553 and 9.9024 MHz over the nine-configuration population), but that population is single-mesh while lattice rounding dominates these two, so the standing objection is sensitivity, not ambiguity, and it wants its own pre-declaration. Handing the oracle drawn counts under the old realization biased f0 by +107.5 MHz, five times the reference's own 21.9 MHz CST-vs-HFSS spread, and the envelope-times-1.5 rule does NOT catch that class because the rule bounds SCATTER and this is BIAS. The realized lengths are read off the realized edge set at build time and re-derived again from the committed node indices in the frozen gate test. Total realized length is now plain addition -- five irises plus four cavities equals the outer extent between the first and last wall plane -- where the retired rule needed a span - 1 and a face-continuity argument to close. An earlier revision claimed that pairing was \"the only pairing that conserves total electrical length\"; that was false, is withdrawn, and the contract makes the question moot. Drawn counts are the plain roundings t_c = round(t/dx), L_c = round(L/dx), d_c = round(d/dx), with NO compensation: the +1 / -1 this case carried until #931 existed only to cancel the missing far face, and deleting it together with the realization it cancelled leaves the built structure unchanged -- same wall planes, same cavities, same apertures. Snapping is still nearest-representable rounding with zero free parameters and no reference number entering, and it is still NOT a monotone improvement across meshes: a/60 and a/90 land on opposite sides of the nominal design. REPORTED, NEVER GATED: individual band edges (+17.05 / +7.20 MHz) and bandwidth (-9.85 MHz), which are ONE fact and not two - d_bw is identically d_hi - d_lo - so the earlier framing of an \"unexplained asymmetric edge residual\" separate from a bandwidth deficit was an algebraic error. #931 discharges BOTH stated blockers on gating them and still does not gate them: the half-cell comparator-input uncertainty is removed by the contract, and the envelope a gate needs is now measured over the nine-configuration population (17.0553 and 9.9024 MHz, so a gate would be 26.0 and 15.0 MHz, committed as gates.edge_bw_gate_would_be_mhz with applied=false). What is left is lattice rounding, which is dominant for these two at 22-40 MHz per cell against f0's 2.4, and the population is single-mesh - every member is a/90 - so a 1.5x lock over it would pin the mesh choice rather than bound the solver. Re-gating them needs its own pre-declaration and a cross-mesh sensitivity measurement, which is separate work; the envelope committed here is what that pre-declaration starts from. Also reported: worst in-band return loss; individual ripple levels; every reflection-zero DEPTH (four nominally identical equiripple zeros bottom out across a wide spread in the published figure, so the paper's frequency step and not physics sets those depths - zero FREQUENCIES are meaningful, depths are not values); passband contiguity beyond the span_holes <= 1 lock (hole depth, hole position, longest contiguous run); the coarse a/60 rung; and phase. PASSBAND CONTIGUITY IS RECORDED, NOT ASSUMED: lo and hi are the OUTERMOST interpolated -10 dB crossings, so the span between them is not necessarily a passband. The built filter has one 10 MHz bin at 9.84 dB inside its 341 MHz span (longest contiguous -10 dB run 270 MHz), while the oracle on the same geometry is contiguous over 35 bins - a real difference that an earlier revision hid behind a clamped statistic, because worst-RL had been computed as the minimum over samples already filtered to >= 10 dB and therefore could not report a violation at all. The coarse a/60 rung has no meaningful passband: 16 of 24 bins in its nominal span are above -10 dB with an interior trough at 2.73 dB, i.e. two separated resonances. That, and not any gate comparison, is the evidence that the gated mesh had to be a/90. The a/60 rung's own numbers do not disqualify it cleanly: its zero count matches its oracle (2 vs 2) and its f0 residual (+19.87 MHz) is the same ~0.12-cell offset seen at a/90; against the committed 19 MHz constant it happens to fail by 0.87 MHz, but a self-derived envelope-times-1.5 gate would pass it. The broken passband is the disqualifier. SETUP IS GATED SEPARATELY FROM PHYSICS, because a resonant band read off an unsettled or absorber-limited run is not a measurement. The repo's preferred ENERGY-BASED ring-down witness (terminal energy in dB below the post-source peak, rule < -40 dB) is NOT AVAILABLE on this path: it is implemented for the lumped/MSL S-matrix extractor, but compute_waveguide_s_matrix returns no settling_db, and the null fields are committed in every row rather than papered over (filed as an rfx capability gap). The ENFORCED settling criterion is therefore the independent axis: the num_periods scan 400/600/800 holds f0 and BW to under 0.1 MHz (the gate is the 400 -> 800 doubling within one 10 MHz bin), the gated run is passivity-clean at column power 1.0065, and the criterion demonstrably fires -- the np=200 run is excluded non-passive at 1.207. The feed-clearance and absorber-depth scans each hold the edges to one 10 MHz bin across their interior and outer samples (standoff 3.05 -> 7.62 -> 15.24 mm: 0.0/0.1 MHz; absorber 0.75 -> 1.00 -> 1.25 lambda_g: 0.0/0.1 MHz). Absorber depth is scanned because in S1 it was the envelope-limiting term at 0.5 lambda_g; here 0.75 lambda_g is measurably sufficient, which is a negative result worth recording rather than a rule inherited. Extractor warnings and the passivity footprint (the bins where column power exceeds 1.02, not merely the scalar maximum) are committed per row. Guide height is reduced to 4 cells on a MEASURED b-invariance witness: b = 4 and b = 8 agree to 233 Hz in f0 on THIS resonant five-iris filter, not merely on the single iris where it was first measured, which is the 8x saving that makes the case affordable to generate. THE ORACLE HAS HAD AN ADVERSARIAL PASS, and it did not find the residual. Its own witnesses are unitarity 2.2e-15, reciprocity and mirror symmetry exact, and an L -> 0 collapse closing two thin irises onto one thick one; its N=1 centred limit reproduces the merged case-18 oracle to 1.05e-04 in an independent odd-mode formulation, and THAT object is what PR #480 confirmed against a formulation-independent 2-D H-plane FDFD at 5.8e-4 - a comparison the frozen gate test now EXECUTES rather than asserting in prose. Those witnesses have known limits, stated because an earlier revision leaned on them too hard: unitarity constrains only the propagating sub-block, mirror symmetry holds by construction for a symmetric geometry, and injected overlap-integral errors leave the reduction and collapse axes silent because they share the overlap routine, so the gate test's re-typed cascade agreeing to 0.0e+00 is a REGRESSION LOCK and not a second opinion. An independent review then closed the real gaps. What is COMMITTED: the overlap integral is validated against direct numerical quadrature IN CI (216 combinations across three apertures including the as-realized 6.096 mm -- where n*pi/a equals m*pi/d exactly for six mode pairs -- centred and 0.19 mm off-centre, worst deviation bounded at 1e-12 with the small-denominator guard exercised), and the oracle's truncation is witnessed on the GATED observable at generation time: f0 by bisection moves 0.33 MHz for n_a 90 -> 130 and 1.85 MHz for an aperture-mode-count doubling -- the aperture axis is the sensitive one, and both are an order under the 19 MHz gate. Session measurements recorded in the research notes but NOT in this record put the gauge invariance of the sqrt(Y) normalisation at machine precision, truncation saturation near 1 MHz in bandwidth, and inter-cavity evanescent transport near 1.6 MHz; they are corroborating colour rather than load-bearing, because the formulation-level check below subsumes their role. THE FORMULATION-LEVEL CHECK IS NOW DONE, and it lands the residual on the rfx side. A 2-D H-plane FDFD -- scalar Helmholtz on a finite-difference grid with an exact discrete transparent port condition, sharing only numpy and scipy with the cascade and no rfx code path at all (validation/crossval/comparators/fdfd_hplane.py) -- was run on the same electrical geometry, grid-exact at every refinement level. It is FIRST-order convergent, so no single level is meaningful; the record carries THREE levels (r = 2, 3, 4, whose bandwidth deviations from the extrapolate shrink as 1/r: measured ratios 1.55 and 1.33 against the first-order 1.50 and 1.33) and BOTH Richardson estimates, which agree to 0.37 MHz in centre frequency and 0.36 MHz in bandwidth -- the two-estimate consistency protocol the porting handoff mandates before either estimate is trusted. The finer pair gives f0 = 10.95742 GHz and BW = 351.42 MHz against the cascade's 10.95851 and 350.43: agreement to 1.09 MHz in centre frequency and 0.98 MHz in bandwidth between two formulations that share nothing but their numerical libraries, with the extrapolation's own order-assumption uncertainty, not the two-estimate consistency, bounding how much of that gap the FDFD itself owns: the consistency (0.37/0.36 MHz) is agreement between two fits of the SAME first-order model, not an error bound -- fitting the convergence order to the three committed levels edge-by-edge gives p = 1.26 (lower edge) / 1.24 (upper edge) rather than the assumed 1, and re-extrapolating the finer pair at the fitted order moves f0 by +0.68 MHz and bandwidth by -0.65 MHz, so the FDFD owns roughly 0.7-1.0 MHz of the 1.09/0.98 MHz gap; f0 and bandwidth are also algebraic combinations of the SAME two band edges, one confirmation and not two. rfx differs from the FDFD by +13.21 MHz in f0 and -10.83 MHz in bandwidth, essentially the same as it differs from the cascade (+12.12 / -9.85), so the 12 MHz residual is not an oracle error. The FDFD's gates: lossless unitarity is enforced on EVERY evaluation (worst 5.0e-07 across all levels), and the empty-guide transparency gate -- &#124;S11&#124; = 5.0e-14 with &#124;S21&#124; = 1.000000000000, the test that originally caught a missing 1/h in the discrete propagation constant -- runs once per generation and once per CI pass. One defect in the comparator itself was found by an independent port review and fixed before this record was generated: its aperture mask realized every aperture two fine cells wide of the stated convention, a first-order bias that Richardson cancelled -- making the extrapolated numbers right for the wrong per-level geometry -- and that produced a spurious FOURTH reflection zero at the coarser levels. With the mask exact, every level shows THREE zeros, matching the cascade and rfx. Independently, the cascade's zero count was checked against its own aperture-mode truncation, which nobody had done for the COUNT: it is 3 at nb_scale 1.0, 1.5, 2.0 and 3.0, with f0 moving 1.7 MHz over that 3x range. What remains genuinely unexplained is the ~12 MHz rfx residual itself: it is mesh-invariant when expressed in cells (-0.117 cell at a/90 against -0.124 at a/60 on the same measured per-mesh cavity sensitivity, where dispersion would have given 0.083), so it behaves like a fixed geometric offset rather than a frequency-dependent solver error, but attributing it to a specific convention leg had FAILED under the old realization: propagating the independently measured iris thickness (t_c - 0.68)*dx through node-plane length conservation overshot and flipped the sign, taking a/90 from +12.08 to -30.62 MHz, and that attribution was recorded as falsified rather than pending. Under the #931 contract the convention legs are no longer free at all -- realized == drawn on every one -- so a residual that survives the regeneration is not a convention artifact. Whether it survived is stated with the regenerated numbers. _gamma at exact cutoff (k equal to n*pi/w, where the sqrt argument vanishes) is unreachable at these band edges and untested. FENCED: nothing here promotes the lane beyond S1. Multi-iris filters, posts and septa remain EXPERIMENTAL; this measures one published design on one mesh with one gated observable, and certifies neither arbitrary filters nor the a/60 rung. Says nothing about phase, group delay, loss, higher-order-mode ports, or fabrication tolerance. The reference is an ANCHOR, not a solver run: the CST and HFSS scalars are digitized from the paper's Fig. 5, no external solver is invoked here, and the case does not compare rfx against either commercial code on any geometry - so the 12.1 MHz f0 residual against this case's own analytic oracle must not be read as an accuracy claim relative to CST or HFSS. Reported for context and not as a yardstick beaten: the oracle on nominal dimensions sits -6.2 MHz from CST and -28.2 MHz from HFSS in f0, +14.7 MHz from both in bandwidth (against a published inter-solver bandwidth spread of only 0.4 MHz), -0.4 and -1.1 dB in worst return loss, and up to 25.4 and 61.9 MHz in individual reflection-zero frequencies. The bandwidth and zero-frequency misses are larger than the f0 miss and are stated here because quoting only f0 would be selective. The built structure is a SNAPPED Aghanim filter: its centre frequency is within the reference's own solver scatter of CST (though not of HFSS), one of the four structural reflection zeros is lost (4 -> 3, confirmed grid-robust by refining the oracle to 1 MHz, with the loss occurring in the upper band), and worst in-band return loss degrades from 13.82 dB to 10.65 dB by rasterization alone, oracle to oracle, with rfx at 9.84 dB. Say snapped, not equivalent. OBSERVABLE PRIORITY for a resonant structure, as this case measures it: the structural reflection-zero COUNT first (an integer, depth-independent, and shown grid-robust), then centre frequency (least sensitive of the continuous quantities to a length error, ~2.4 MHz per cell of iris thickness), then band edges and bandwidth (~22-40 MHz per cell, hence reported), then worst return loss, and last individual ripple levels and null depths, which are not values at all. TOPOLOGY FIRST, AND f0 IS NOT EXONERATED: the zero count is the most robust observable, but f0 is not thereby safe -- it carries the +12.12 MHz residual this case gates, and at -105 MHz per cell of cavity length it is the quantity a geometry error moves first. A cell snap is inherently non-uniform, since each cavity rounds independently, so every snap figure quoted here was MEASURED on the as-snapped geometry and none may be re-derived by multiplying a sensitivity coefficient by a half cell."` | — |
| `/coarse_diagnostic/aperture_nodes` | `[[17, 42], [22, 37], [23, 37], [22, 37], [17, 42]]` | *absent* | 0–4 |
| `/coarse_diagnostic/aperture_wall_nodes` | *absent* | `[[16, 43], [21, 38], [22, 38], [21, 38], [16, 43]]` | 0–4 |
| `/coarse_diagnostic/band/bw` | `236253485.27890015` | `236295820.43458366` | — |
| `/coarse_diagnostic/band/f0` | `11095241431.043194` | `11095267252.543339` | — |
| `/coarse_diagnostic/band/hi` | `11213368173.682644` | `11213415162.760632` | — |
| `/coarse_diagnostic/band/lo` | `10977114688.403744` | `10977119342.326048` | — |
| `/coarse_diagnostic/band/worst_rl_db` | `2.7303308022352253` | `2.7314608133164775` | — |
| `/coarse_diagnostic/d_bw_mhz` | `-9.3` | `-9.26` | — |
| `/coarse_diagnostic/d_hi_mhz` | `15.2` | `15.24` | — |
| `/coarse_diagnostic/d_lo_mhz` | `24.5` | `24.51` | — |
| `/coarse_diagnostic/extractor_warnings` | array[88], SHA256 `8269bb2c8308` | array[89], SHA256 `2425b81e95ce` | 0, 2, 4–6, 8, 14–16, 18, 22, 24, 26–28, 30, 36–38, 40, 44, 46, 48–50, 52, 58–60, 62, 66, 68, 70–72, 74, 80–82, 84, 88 |
| `/coarse_diagnostic/glen_cells` | `264` | `263` | — |
| `/coarse_diagnostic/grid` | `[413, 61, 5]` | `[412, 61, 5]` | 0 |
| `/coarse_diagnostic/iris_wall_nodes` | *absent* | `[[114, 119], [157, 162], [203, 208], [249, 254], [292, 297]]` | 0–4 |
| `/coarse_diagnostic/iris_x_nodes` | `[[114, 119], [157, 162], [203, 208], [249, 254], [292, 297]]` | *absent* | 0–4 |
| `/coarse_diagnostic/oracle_s11` | array[131], SHA256 `c89658b611e5` | array[131], SHA256 `7eb1d7ec3983` | 3, 10, 12–13, 21, 23, 26, 34, 42–43, 47, 52–53, 56, 66, 73, 79, 87–89, 92, 105–106, 109, 121 |
| `/coarse_diagnostic/s11` | array[131], SHA256 `56fb3869ff2d` | array[131], SHA256 `dde625f5cdfb` | 0–130 |
| `/coarse_diagnostic/s21` | array[131], SHA256 `a52cc3e02f69` | array[131], SHA256 `ce7f9f11c2f5` | 0–113, 115–122, 124–130 |
| `/coarse_diagnostic/settling_db` | `null` | `[-50.4, -50.4]` | 0–1 |
| `/coarse_diagnostic/wall_s` | `445.5` | `255.7` | — |
| `/electrical_geometry/aperture_cells` | *absent* | `[40, 26, 24, 26, 40]` | 0–4 |
| `/electrical_geometry/aperture_wall_nodes` | *absent* | `[[25, 65], [32, 58], [33, 57], [32, 58], [25, 65]]` | 0–4 |
| `/electrical_geometry/compensation` | `"drawn counts are chosen so the ELECTRICAL dimensions land on nominal: t_c = round(t/dx) + 1, L_c = round(L/dx) - 1. At a/90 that puts f0 +3.3 MHz from the paper's exact design (inside its own 21.9 MHz CST-vs-HFSS spread) against -101.4 MHz uncompensated. Compensation is NOT a monotone improvement -- it only picks which side of the sub-cell rounding you land on, and at a/60 it moves f0 from -35.8 to +120.2 MHz."` | `"none. Drawn counts are the plain roundings t_c = round(t/dx), L_c = round(L/dx), d_c = round(d/dx), and under the contract they are also the realized ones. Until #931 this case drew t_c = round(t/dx) + 1 and L_c = round(L/dx) - 1 so the ELECTRICAL dimensions would land on nominal against a realization that lost one plane per body; the compensation and that realization are deleted together and the built structure is unchanged. Snapping remains a rounding and is NOT a monotone improvement across meshes: a/90 and a/60 land on opposite sides of the nominal design."` | — |
| `/electrical_geometry/cost_note` | `"measured 2026-07-29: feeding the oracle the DRAWN cell counts puts its band at 10.9054-11.2267 GHz against 10.7833-11.1337 GHz for the realized geometry, a +107.5 MHz f0 error -- five times the paper's own 21.9 MHz CST-vs-HFSS spread, and it would have had to be absorbed by a ~162 MHz gate (46% of the passband) that pins nothing. First found as +90.0 MHz on the uncompensated counts; compensation changes which pair is confused, not the class."` | `"HISTORICAL, measured 2026-07-29 under the pre-#931 realization: feeding the oracle the DRAWN cell counts put its band at 10.9054-11.2267 GHz against 10.7833-11.1337 GHz for the realized geometry, a +107.5 MHz f0 error -- five times the paper's own 21.9 MHz CST-vs-HFSS spread, and it would have had to be absorbed by a ~162 MHz gate (46% of the passband) that pins nothing. First found as +90.0 MHz on the uncompensated counts; compensation changed which pair was confused, not the class. The contract removes the pair: there is one geometry, drawn and realized, and this number is kept as the record of what the defect was worth."` | — |
| `/electrical_geometry/drawn_aperture_cells` | *absent* | `[40, 26, 24, 26, 40]` | 0–4 |
| `/electrical_geometry/drawn_cavity_cells` | `[55, 61, 61, 55]` | `[56, 62, 62, 56]` | 0–3 |
| `/electrical_geometry/drawn_iris_thickness_cells` | `9` | `8` | — |
| `/electrical_geometry/iris_wall_nodes` | *absent* | `[[150, 158], [214, 222], [284, 292], [354, 362], [418, 426]]` | 0–4 |
| `/electrical_geometry/rule` | `"oracle inputs are READ BACK off the rasterized metal. The CAVITY leg is (L_c + 1)*dx -- the distance between the bounding zeroed node planes -- and is confirmed to 0.04-0.17 cell by the committed residual against the measured cavity sensitivity; the transverse APERTURE leg d_c*dx is confirmed to better than 0.05 cell by an independent refit of 16 committed case-18 configurations. The IRIS-THICKNESS leg is NOT (t_c - 1)*dx: four FDTD runs at drawn t_c = 2/4/6/8 give a flat offset of -0.66/-0.68/-0.68/-0.70 cell, i.e. t_elec ~ (t_c - 0.68)*dx, matching neither this rule nor case 18's t_c*dx. That ~1/3-cell ambiguity is an irreducible comparator-input uncertainty here and it is why bandwidth and individual band edges are REPORTED rather than gated (they move ~40 and ~22-30 MHz per cell of it) while f0 and the zero count are gated (~2.4 MHz per cell, and an integer). Total electrical length is a face-continuity CHECK across region types, NOT a uniqueness argument: putting the interface at sigma*dx beyond the outermost metal node gives total = span - 1 + 2*sigma = the outer extent measured at the same sigma, conserved for EVERY sigma, and sigma = 0.5 is the drawn pairing itself."` | `"oracle inputs are READ BACK off the REALIZED PEC edge set (rfx.boundaries.pec.realized_pec_edge_masks through validation/crossval/_wr90_iris_realized.py), never off the drawn counts. Under the #931 lattice ownership contract a PEC volume drawn on node planes realizes tangential walls at BOTH faces and shorts every normal edge between them, so all three legs are the drawn ones: iris thickness t_c*dx, cavity L_c*dx, aperture d_c*dx. raster_assert asserts that identity per iris, per cavity and per aperture at build time, with no solve. HISTORY: before #931 a body's far face was never a wall, so the realized cavity was (L_c + 1)*dx and the realized iris (t_c - 1)*dx against the drawing, this case carried a +1/-1 compensation in the drawn counts to land the electrical dimensions on nominal, and the iris-thickness leg fitted neither rule (four FDTD runs at drawn t_c = 2/4/6/8 gave t_elec ~ (t_c - 0.68)*dx, matching neither this rule nor case 18's t_c*dx). That ~1/3-cell offset was read as an irreducible comparator-input uncertainty; it is better read as the missing far face plus a corner recipe that placed every face half a cell off the node planes. Both are gone, and with them the compensation and the face-continuity argument that closed the old total-length identity through a span - 1. Total realized length is now plain addition: five irises plus four cavities equals the outer extent between the first and last wall plane."` | — |
| `/fdfd_formulation_independent/d_bw_vs_cascade_mhz` | `0.985` | `0.984` | — |
| `/fdfd_formulation_independent/d_f0_rfx_vs_fdfd_mhz` | `13.171` | `13.209` | — |
| `/fdfd_formulation_independent/levels/2/band/bw` | `346043170.02007294` | `346043218.8118229` | — |
| `/fdfd_formulation_independent/levels/2/band/f0` | `10963397722.378645` | `10963397746.77452` | — |
| `/fdfd_formulation_independent/levels/2/band/hi` | `11136419307.388681` | `11136419356.180431` | — |
| `/fdfd_formulation_independent/levels/2/s11` | array[131], SHA256 `e086d8743a0e` | array[131], SHA256 `394d7de7f64c` | 41, 71, 73, 109 |
| `/fdfd_formulation_independent/levels/2/wall_s` | `385.4` | `121.1` | — |
| `/fdfd_formulation_independent/levels/2/worst_unitarity` | `2.910083445328837e-07` | `2.5891862298621504e-07` | — |
| `/fdfd_formulation_independent/levels/3/band/bw` | `347953193.3703842` | `347953184.3007183` | — |
| `/fdfd_formulation_independent/levels/3/band/f0` | `10961284753.373653` | `10961284757.908485` | — |
| `/fdfd_formulation_independent/levels/3/band/lo` | `10787308156.688461` | `10787308165.758127` | — |
| `/fdfd_formulation_independent/levels/3/s11` | array[131], SHA256 `886aa6f2b8e4` | array[131], SHA256 `dd15808dcadb` | 8, 38, 40, 44, 46–47, 49, 54, 57–58, 64, 70, 72, 75, 77 |
| `/fdfd_formulation_independent/levels/3/wall_s` | `1186.6` | `308.1` | — |
| `/fdfd_formulation_independent/levels/3/worst_unitarity` | `4.2312295167601377e-07` | `3.9601045442871907e-07` | — |
| `/fdfd_formulation_independent/levels/4/band/bw` | `348818859.72759247` | `348818754.1513157` | — |
| `/fdfd_formulation_independent/levels/4/band/f0` | `10960319536.995258` | `10960319521.61694` | — |
| `/fdfd_formulation_independent/levels/4/band/hi` | `11134728966.859055` | `11134728898.692596` | — |
| `/fdfd_formulation_independent/levels/4/band/lo` | `10785910107.131462` | `10785910144.54128` | — |
| `/fdfd_formulation_independent/levels/4/s11` | array[131], SHA256 `15d6d2735350` | array[131], SHA256 `c6d33eb5ed88` | 39–40, 51, 63–64, 68–69, 71, 73, 78, 90 |
| `/fdfd_formulation_independent/levels/4/wall_s` | `2540.5` | `601.9` | — |
| `/fdfd_formulation_independent/levels/4/worst_unitarity` | `4.5658172764806437e-07` | `5.024902829386946e-07` | — |
| `/fdfd_formulation_independent/richardson_23/bw` | `351773240.0710068` | `351773115.27850914` | — |
| `/fdfd_formulation_independent/richardson_23/f0` | `10957058815.36367` | `10957058780.176414` | — |
| `/fdfd_formulation_independent/richardson_23/hi` | `11132945435.399174` | `11132945337.815674` | — |
| `/fdfd_formulation_independent/richardson_23/lo` | `10781172195.328167` | `10781172222.537167` | — |
| `/fdfd_formulation_independent/richardson_34/bw` | `351415858.7992172` | `351415463.70310783` | — |
| `/fdfd_formulation_independent/richardson_34/f0` | `10957423887.860073` | `10957423812.742302` | — |
| `/fdfd_formulation_independent/richardson_34/hi` | `11133131817.259682` | `11133131544.59385` | — |
| `/fdfd_formulation_independent/richardson_34/lo` | `10781715958.460464` | `10781716080.89074` | — |
| `/fdfd_formulation_independent/richardson_consistency_mhz/bw` | `0.357` | `0.358` | — |
| `/fdfd_formulation_independent/self_test/empty_s11` | `4.998689747642886e-14` | `4.9977732337688505e-14` | — |
| `/fdfd_formulation_independent/self_test/empty_s21` | `1.0000000000000018` | `1.0000000000000016` | — |
| `/fdfd_formulation_independent/self_test/unitarity` | `1.4655321400880439e-09` | `2.3153723383018132e-09` | — |
| `/feed_clearance_witness/d_hi_mhz` | `0.11` | `0.05` | — |
| `/feed_clearance_witness/d_lo_mhz` | `0.03` | `0.01` | — |
| `/feed_clearance_witness/gated/hi` | `11140816346.186275` | `11140924560.619234` | — |
| `/feed_clearance_witness/gated/lo` | `10800373669.774708` | `10800340317.655672` | — |
| `/feed_clearance_witness/generous/d_hi_mhz` | `0.109` | `0.001` | — |
| `/feed_clearance_witness/generous/d_lo_mhz` | `0.032` | `0.003` | — |
| `/feed_clearance_witness/generous/hi` | `11140925564.344723` | `11140923080.683405` | — |
| `/feed_clearance_witness/generous/lo` | `10800341884.471735` | `10800337332.294498` | — |
| `/feed_clearance_witness/generous/s11` | array[131], SHA256 `b2a8527e031c` | array[131], SHA256 `29d2f24c0cd1` | 0–13, 15–99, 102, 104–108, 110–117, 119–126, 128–130 |
| `/feed_clearance_witness/generous/s21` | array[131], SHA256 `6c47bbc339ee` | array[131], SHA256 `a3b7e36e7283` | 0–7, 10–118, 120–122, 124–126, 128–130 |
| `/feed_clearance_witness/generous/wall_s` | `1308.9` | `977.5` | — |
| `/feed_clearance_witness/mid/d_hi_mhz` | `0.016` | `0.045` | — |
| `/feed_clearance_witness/mid/d_lo_mhz` | `0.003` | `0.01` | — |
| `/feed_clearance_witness/mid/hi` | `11140800129.524334` | `11140879350.558094` | — |
| `/feed_clearance_witness/mid/lo` | `10800371060.821806` | `10800350360.551697` | — |
| `/feed_clearance_witness/mid/s11` | array[131], SHA256 `295d3b127ebf` | array[131], SHA256 `a2915cf7ab01` | 0–26, 28–96, 98–99, 101–109, 111–113, 115–124, 126–130 |
| `/feed_clearance_witness/mid/s21` | array[131], SHA256 `01ffc558141f` | array[131], SHA256 `9a7f42f34830` | 0–5, 7–11, 13–113, 115–123, 125–130 |
| `/feed_clearance_witness/mid/wall_s` | `1235.5` | `907.8` | — |
| `/gated_rfx/aperture_nodes` | `[[26, 64], [33, 57], [34, 56], [33, 57], [26, 64]]` | *absent* | 0–4 |
| `/gated_rfx/aperture_wall_nodes` | *absent* | `[[25, 65], [32, 58], [33, 57], [32, 58], [25, 65]]` | 0–4 |
| `/gated_rfx/band/bw` | `340442676.4115677` | `340584242.963562` | — |
| `/gated_rfx/band/f0` | `10970595007.980492` | `10970632439.137453` | — |
| `/gated_rfx/band/hi` | `11140816346.186275` | `11140924560.619234` | — |
| `/gated_rfx/band/lo` | `10800373669.774708` | `10800340317.655672` | — |
| `/gated_rfx/band/worst_rl_db` | `9.802406901625071` | `9.83633015873974` | — |
| `/gated_rfx/d_bw_mhz` | `-9.99` | `-9.85` | — |
| `/gated_rfx/d_f0_mhz` | `12.08` | `12.12` | — |
| `/gated_rfx/d_hi_mhz` | `7.09` | `7.2` | — |
| `/gated_rfx/d_lo_mhz` | `17.08` | `17.05` | — |
| `/gated_rfx/extractor_warnings` | array[88], SHA256 `8269bb2c8308` | array[89], SHA256 `a623f30a2587` | 0, 2, 4–6, 8, 14–16, 18, 22, 24, 26–28, 30, 36–38, 40, 44, 46, 48–50, 52, 58–60, 62, 66, 68, 70–72, 74, 80–82, 84, 88 |
| `/gated_rfx/glen_cells` | `357` | `356` | — |
| `/gated_rfx/grid` | `[578, 91, 5]` | `[577, 91, 5]` | 0 |
| `/gated_rfx/iris_wall_nodes` | *absent* | `[[150, 158], [214, 222], [284, 292], [354, 362], [418, 426]]` | 0–4 |
| `/gated_rfx/iris_x_nodes` | `[[150, 158], [214, 222], [284, 292], [354, 362], [418, 426]]` | *absent* | 0–4 |
| `/gated_rfx/oracle_s11` | array[131], SHA256 `8742d7e4cd9e` | array[131], SHA256 `bcb27e2f197d` | 6, 28, 31, 33, 35, 55, 76, 82, 87, 89, 95, 98, 100, 102, 105–106, 117, 130 |
| `/gated_rfx/s11` | array[131], SHA256 `19dd17dae901` | array[131], SHA256 `a8d2053b1cec` | 0–18, 20–26, 28–99, 101–106, 108–113, 115–122, 124–130 |
| `/gated_rfx/s21` | array[131], SHA256 `74e9a501b715` | array[131], SHA256 `b14514c8010c` | 0–130 |
| `/gated_rfx/settling_db` | `null` | `[-59.23, -59.23]` | 0–1 |
| `/gated_rfx/wall_s` | `1160.3` | `850.0` | — |
| `/gates/bw_measured_envelope_mhz` | *absent* | `9.9024` | — |
| `/gates/bw_reported_residual_mhz` | `9.99` | *absent* | — |
| `/gates/edge_bw_envelope_population/0/config` | *absent* | `"gated a/90 np400 b4"` | — |
| `/gates/edge_bw_envelope_population/0/d_bw_mhz` | *absent* | `-9.8471` | — |
| `/gates/edge_bw_envelope_population/0/d_hi_mhz` | *absent* | `7.1982` | — |
| `/gates/edge_bw_envelope_population/0/d_lo_mhz` | *absent* | `17.0453` | — |
| `/gates/edge_bw_envelope_population/1/config` | *absent* | `"ring np600"` | — |
| `/gates/edge_bw_envelope_population/1/d_bw_mhz` | *absent* | `-9.8641` | — |
| `/gates/edge_bw_envelope_population/1/d_hi_mhz` | *absent* | `7.1698` | — |
| `/gates/edge_bw_envelope_population/1/d_lo_mhz` | *absent* | `17.034` | — |
| `/gates/edge_bw_envelope_population/2/config` | *absent* | `"ring np800"` | — |
| `/gates/edge_bw_envelope_population/2/d_bw_mhz` | *absent* | `-9.864` | — |
| `/gates/edge_bw_envelope_population/2/d_hi_mhz` | *absent* | `7.1699` | — |
| `/gates/edge_bw_envelope_population/2/d_lo_mhz` | *absent* | `17.0339` | — |
| `/gates/edge_bw_envelope_population/3/config` | *absent* | `"b=6 cells"` | — |
| `/gates/edge_bw_envelope_population/3/d_bw_mhz` | *absent* | `-9.8469` | — |
| `/gates/edge_bw_envelope_population/3/d_hi_mhz` | *absent* | `7.1985` | — |
| `/gates/edge_bw_envelope_population/3/d_lo_mhz` | *absent* | `17.0454` | — |
| `/gates/edge_bw_envelope_population/4/config` | *absent* | `"b=8 cells"` | — |
| `/gates/edge_bw_envelope_population/4/d_bw_mhz` | *absent* | `-9.8477` | — |
| `/gates/edge_bw_envelope_population/4/d_hi_mhz` | *absent* | `7.1976` | — |
| `/gates/edge_bw_envelope_population/4/d_lo_mhz` | *absent* | `17.0454` | — |
| `/gates/edge_bw_envelope_population/5/config` | *absent* | `"mid feed"` | — |
| `/gates/edge_bw_envelope_population/5/d_bw_mhz` | *absent* | `-9.9024` | — |
| `/gates/edge_bw_envelope_population/5/d_hi_mhz` | *absent* | `7.153` | — |
| `/gates/edge_bw_envelope_population/5/d_lo_mhz` | *absent* | `17.0553` | — |
| `/gates/edge_bw_envelope_population/6/config` | *absent* | `"generous feed"` | — |
| `/gates/edge_bw_envelope_population/6/d_bw_mhz` | *absent* | `-9.8456` | — |
| `/gates/edge_bw_envelope_population/6/d_hi_mhz` | *absent* | `7.1967` | — |
| `/gates/edge_bw_envelope_population/6/d_lo_mhz` | *absent* | `17.0423` | — |
| `/gates/edge_bw_envelope_population/7/config` | *absent* | `"mid absorber"` | — |
| `/gates/edge_bw_envelope_population/7/d_bw_mhz` | *absent* | `-9.8591` | — |
| `/gates/edge_bw_envelope_population/7/d_hi_mhz` | *absent* | `7.189` | — |
| `/gates/edge_bw_envelope_population/7/d_lo_mhz` | *absent* | `17.0481` | — |
| `/gates/edge_bw_envelope_population/8/config` | *absent* | `"deep absorber"` | — |
| `/gates/edge_bw_envelope_population/8/d_bw_mhz` | *absent* | `-9.8655` | — |
| `/gates/edge_bw_envelope_population/8/d_hi_mhz` | *absent* | `7.1841` | — |
| `/gates/edge_bw_envelope_population/8/d_lo_mhz` | *absent* | `17.0496` | — |
| `/gates/edge_bw_gate_would_be_mhz/applied` | *absent* | `false` | — |
| `/gates/edge_bw_gate_would_be_mhz/bw` | *absent* | `15.0` | — |
| `/gates/edge_bw_gate_would_be_mhz/edges` | *absent* | `26.0` | — |
| `/gates/edge_bw_gate_would_be_mhz/why_not` | *absent* | `"the population is single-mesh (every member a/90) while lattice rounding is the dominant term for these two observables at 22-40 MHz per cell, so a 1.5x lock over it would pin the mesh choice rather than bound the solver; re-gating needs its own pre-declaration and a cross-mesh sensitivity measurement"` | — |
| `/gates/edge_measured_envelope_mhz` | *absent* | `17.0553` | — |
| `/gates/edge_reported_residual_mhz` | `17.08` | *absent* | — |
| `/gates/f0_envelope_population/0/d_f0_mhz` | `12.0843` | `12.1217` | — |
| `/gates/f0_envelope_population/1/d_f0_mhz` | `12.064` | `12.1019` | — |
| `/gates/f0_envelope_population/2/d_f0_mhz` | `12.0641` | `12.1019` | — |
| `/gates/f0_envelope_population/3/d_f0_mhz` | `12.0843` | `12.1219` | — |
| `/gates/f0_envelope_population/4/d_f0_mhz` | `12.0842` | `12.1215` | — |
| `/gates/f0_envelope_population/5/d_f0_mhz` | `12.0749` | `12.1042` | — |
| `/gates/f0_envelope_population/6/d_f0_mhz` | `12.123` | `12.1195` | — |
| `/gates/f0_envelope_population/7/d_f0_mhz` | `12.0647` | `12.1185` | — |
| `/gates/f0_envelope_population/8/d_f0_mhz` | `12.0605` | `12.1169` | — |
| `/gates/f0_measured_envelope_mhz` | `12.123` | `12.1219` | — |
| `/gates/f0_population_excluded/0/max_colpow` | `1.207` | `1.2071` | — |
| `/gates/posture` | `"gate = round-UP(measured envelope x 1.5) over a MULTI-CONFIGURATION population, enforced as EXACT equality by the write-fixture self-check. That makes the envelope robust rather than resting on one datum, but it does NOT make the gate independent of the datum: the population spread is 0.06 MHz while every member is about 12.08 MHz from the oracle, so the envelope is dominated by the residual and the gate is 1.5x the measured agreement. It is a REGRESSION LOCK with 50% headroom, not an independent accuracy bound. What gives the agreement meaning is its comparison against an external scale (the reference's own 21.9 MHz f0 spread between two independent commercial codes), not the gate. GATED: centre frequency f0; the structural zero COUNT (witnessed invariant across the 8.00-8.50-cell iris-thickness ambiguity band, see iris_thickness_zero_count_sweep); and passband contiguity as a regression lock (span_holes <= 1, the committed envelope -- the f0 gate alone cannot see a split passband because band edges are the outermost crossings), against the oracle on as-realized geometry. REPORTED, never gated: individual band edges and bandwidth (their comparator-input uncertainty from the unsettled iris-thickness convention, ~20 MHz, exceeds any defensible gate on them, and d_bw is identically d_hi - d_lo so they are one fact), worst-case RL, ripple levels, zero depths, contiguity detail beyond the span_holes lock, the coarse rung and phase"` | `"gate = round-UP(measured envelope x 1.5) over a MULTI-CONFIGURATION population, enforced as EXACT equality by the write-fixture self-check. That makes the envelope robust rather than resting on one datum, but it does NOT make the gate independent of the datum: the population spread is 0.02 MHz while every member is about 12.12 MHz from the oracle, so the envelope is dominated by the residual and the gate is 1.5x the measured agreement. It is a REGRESSION LOCK with 50% headroom, not an independent accuracy bound. What gives the agreement meaning is its comparison against an external scale (the reference's own 21.9 MHz f0 spread between two independent commercial codes), not the gate. GATED: centre frequency f0; the structural zero COUNT (witnessed invariant across a one-cell iris-thickness band centred on the realized thickness, see iris_thickness_zero_count_sweep); and passband contiguity as a regression lock (span_holes <= 1, the committed envelope -- the f0 gate alone cannot see a split passband because band edges are the outermost crossings), against the oracle on as-realized geometry. REPORTED, never gated: individual band edges and bandwidth -- #931 removes the half-cell input uncertainty that was the original reason, and the regenerated record now MEASURES their envelope (17.0553 and 9.9024 MHz over the nine-configuration population, a gate would be 26.0 and 15.0, committed as edge_bw_gate_would_be_mhz with applied=false), but the population is single-mesh while lattice rounding dominates these two at 22-40 MHz per cell, so a lock over it would pin the mesh rather than bound the solver; re-gating needs its own pre-declaration and a cross-mesh sensitivity measurement. d_bw is identically d_hi - d_lo so they are one fact. Also reported: worst-case RL, ripple levels, zero depths, contiguity detail beyond the span_holes lock, the coarse rung and phase"` | — |
| `/iris_thickness_zero_count_sweep/note` | `"oracle-side robustness witness for the ZERO-COUNT gate (post-merge joint review, N3): the iris-thickness electrical leg is the one unsettled convention input, so the gated integer must not depend on which convention the comparator picks. t_elec swept 8.00-8.50 cells -- the built (t_c - 1)*dx rule at 8.00, the measured (t_c - 0.68)*dx offset at 8.32 -- on the committed frequency grid; oracle evaluations only, no FDTD."` | `"oracle-side robustness witness for the ZERO-COUNT gate (post-merge joint review, N3; re-centred for #931): the gated integer must survive a perturbation of the comparator's most length-sensitive input. t_elec swept over one full cell CENTRED on the realized iris thickness (realized -0.5 .. +0.5 cell, eleven points) on the committed frequency grid; oracle evaluations only, no FDTD. Before #931 the window was one-sided, 8.00-8.50 cells, spanning the disagreement between this case's built (t_c - 1)*dx rule and the measured (t_c - 0.68)*dx offset; the lattice ownership contract makes the realized thickness exact, so there is no ambiguity band left to span and the remaining question is symmetric sensitivity."` | — |
| `/iris_thickness_zero_count_sweep/rows/0/bw_hz` | `350431353` | `371605947` | — |
| `/iris_thickness_zero_count_sweep/rows/0/f0_hz` | `10958510693` | `10957110862` | — |
| `/iris_thickness_zero_count_sweep/rows/0/t_elec_cells` | `8.0` | `7.5` | — |
| `/iris_thickness_zero_count_sweep/rows/1/bw_hz` | `348350618` | `367450743` | — |
| `/iris_thickness_zero_count_sweep/rows/1/f0_hz` | `10958634569` | `10957252317` | — |
| `/iris_thickness_zero_count_sweep/rows/1/t_elec_cells` | `8.05` | `7.6` | — |
| `/iris_thickness_zero_count_sweep/rows/2/bw_hz` | `346177876` | `363001206` | — |
| `/iris_thickness_zero_count_sweep/rows/2/f0_hz` | `10958798136` | `10957613410` | — |
| `/iris_thickness_zero_count_sweep/rows/2/t_elec_cells` | `8.1` | `7.7` | — |
| `/iris_thickness_zero_count_sweep/rows/3/bw_hz` | `343911491` | `358268053` | — |
| `/iris_thickness_zero_count_sweep/rows/3/f0_hz` | `10959001732` | `10958203195` | — |
| `/iris_thickness_zero_count_sweep/rows/3/t_elec_cells` | `8.15` | `7.8` | — |
| `/iris_thickness_zero_count_sweep/rows/4/bw_hz` | `341547717` | `354311209` | — |
| `/iris_thickness_zero_count_sweep/rows/4/f0_hz` | `10959246710` | `10958386496` | — |
| `/iris_thickness_zero_count_sweep/rows/4/t_elec_cells` | `8.2` | `7.9` | — |
| `/iris_thickness_zero_count_sweep/rows/5/bw_hz` | `339161490` | `350431353` | — |
| `/iris_thickness_zero_count_sweep/rows/5/f0_hz` | `10959575668` | `10958510693` | — |
| `/iris_thickness_zero_count_sweep/rows/5/t_elec_cells` | `8.25` | `8.0` | — |
| `/iris_thickness_zero_count_sweep/rows/6/bw_hz` | `337525422` | `346177876` | — |
| `/iris_thickness_zero_count_sweep/rows/6/f0_hz` | `10959529952` | `10958798136` | — |
| `/iris_thickness_zero_count_sweep/rows/6/t_elec_cells` | `8.3` | `8.1` | — |
| `/iris_thickness_zero_count_sweep/rows/7/bw_hz` | `335790673` | `341547717` | — |
| `/iris_thickness_zero_count_sweep/rows/7/f0_hz` | `10959532008` | `10959246710` | — |
| `/iris_thickness_zero_count_sweep/rows/7/t_elec_cells` | `8.35` | `8.2` | — |
| `/iris_thickness_zero_count_sweep/rows/8/bw_hz` | `333963749` | `337525422` | — |
| `/iris_thickness_zero_count_sweep/rows/8/f0_hz` | `10959578303` | `10959529952` | — |
| `/iris_thickness_zero_count_sweep/rows/8/t_elec_cells` | `8.4` | `8.3` | — |
| `/iris_thickness_zero_count_sweep/rows/9/bw_hz` | `332050107` | `333963749` | — |
| `/iris_thickness_zero_count_sweep/rows/9/f0_hz` | `10959665799` | `10959578303` | — |
| `/iris_thickness_zero_count_sweep/rows/9/t_elec_cells` | `8.45` | `8.4` | — |
| `/ring_down_witness/0/bw` | `339091610.2254982` | `339279358.9056721` | — |
| `/ring_down_witness/0/f0` | `10969497751.822296` | `10969517916.62199` | — |
| `/ring_down_witness/0/hi` | *absent* | `11139157596.074827` | — |
| `/ring_down_witness/0/lo` | *absent* | `10799878237.169155` | — |
| `/ring_down_witness/0/max_colpow` | `1.207` | `1.2071` | — |
| `/ring_down_witness/0/s11` | array[131], SHA256 `4fa6437fb50f` | array[131], SHA256 `7e96c261ee82` | 0–11, 13–33, 35–130 |
| `/ring_down_witness/0/s21` | array[131], SHA256 `c6fbb213fe4b` | array[131], SHA256 `083a4f61cbaf` | 0–130 |
| `/ring_down_witness/0/wall_s` | `586.3` | `427.7` | — |
| `/ring_down_witness/1/bw` | `340442676.4115677` | `340584242.963562` | — |
| `/ring_down_witness/1/f0` | `10970595007.980492` | `10970632439.137453` | — |
| `/ring_down_witness/1/hi` | *absent* | `11140924560.619234` | — |
| `/ring_down_witness/1/lo` | *absent* | `10800340317.655672` | — |
| `/ring_down_witness/1/s11` | array[131], SHA256 `19dd17dae901` | array[131], SHA256 `a8d2053b1cec` | 0–18, 20–26, 28–99, 101–106, 108–113, 115–122, 124–130 |
| `/ring_down_witness/1/s21` | array[131], SHA256 `74e9a501b715` | array[131], SHA256 `b14514c8010c` | 0–130 |
| `/ring_down_witness/1/wall_s` | `1160.3` | `850.0` | — |
| `/ring_down_witness/2/bw` | `340426072.2811279` | `340567212.6795101` | — |
| `/ring_down_witness/2/f0` | `10970574707.84292` | `10970612585.738922` | — |
| `/ring_down_witness/2/hi` | *absent* | `11140896192.078676` | — |
| `/ring_down_witness/2/lo` | *absent* | `10800328979.399166` | — |
| `/ring_down_witness/2/s11` | array[131], SHA256 `45d75d62dde3` | array[131], SHA256 `99756a759dc8` | 0–3, 5, 7–9, 11, 15, 17, 19, 21–97, 99–101, 104, 106–107, 109–115, 118–124, 126–128, 130 |
| `/ring_down_witness/2/s21` | array[131], SHA256 `cac2961e684f` | array[131], SHA256 `fde1f8f94b2b` | 0–130 |
| `/ring_down_witness/2/wall_s` | `1728.3` | `1262.2` | — |
| `/ring_down_witness/3/bw` | `340426165.5448818` | `340567333.74594116` | — |
| `/ring_down_witness/3/f0` | `10970574763.497368` | `10970612623.710629` | — |
| `/ring_down_witness/3/hi` | *absent* | `11140896290.5836` | — |
| `/ring_down_witness/3/lo` | *absent* | `10800328956.837658` | — |
| `/ring_down_witness/3/s11` | array[131], SHA256 `e4c759b2c17e` | array[131], SHA256 `f45ff7f82b0b` | 0–4, 7–10, 15–19, 21–22, 24–101, 103–107, 109–110, 112, 114, 117, 119–122, 124, 126–130 |
| `/ring_down_witness/3/s21` | array[131], SHA256 `673831a61700` | array[131], SHA256 `5e8be8492798` | 0–130 |
| `/ring_down_witness/3/wall_s` | `2302.6` | `1698.9` | — |
| `/schema_version` | `1` | `2` | — |

## tests/fixtures/wr90_iris_modematch/fixture.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/claim_scope` | `"One symmetric inductive PEC iris (t = 1.524 mm = exactly 2 coarse / 4 fine cells; apertures 18.288/12.192/7.62 mm, grid-exact) in WR-90 over 8.2-12.4 GHz on 29 frequency points, flux-normalized &#124;S11&#124; vs a twice-implemented TEn0 mode-matching cascade oracle (self-witnesses: unitarity 1.1e-16, mode convergence 4.3e-5, Marcuvitz cot^2 thin-limit anchor 10.8% with the inductive sign, d->a and deep-constriction limits; the PR #480 review reproduced the oracle with a formulation-independent 2-D H-plane FDFD to 6e-4 and measured rfx's same-geometry agreement at <= 0.02 — attributed, not imported). GATED: fine rung dx = a/60 within 0.04 abs = round-up(measured envelope 0.0232 x 1.5) over 8 configs (3 apertures x {centred, iris off-centre at 0.42 of the guide} + 2 extra guide lengths; every config lands within 0.023, so no single configuration sets the envelope), and the Richardson extrapolation 2*S(a/60) - S(a/30) on the oracle within 0.01 abs (envelope 0.0051) at EVERY one of those 8 pairs, which cross-confirms the oracle and the first-order attribution (gap ratios 0.527-0.604 = textbook first order). REPORTED, NOT GATED: the coarse rung dx = a/30 (0.018-0.043 abs); the raw normalize=False record, which is WORSE than flux (gaps 0.021-0.054) with a pointwise &#124;raw - flux&#124; difference up to 0.033 at the wide aperture; residual detrended ripple, i.e. a quadratic detrend of &#124;S11&#124; MINUS the oracle so the oracle's own curvature is not counted (fine <= 0.0077, coarse <= 0.0158, both at the wide aperture with the iris off-centre, down from the 0.0706 the PR #480 review measured on the same basis before the absorber fix); and phase (magnitude-only lane posture). FENCED, never gated: everything beyond ONE symmetric inductive iris — multi-iris filters, posts, septa and off-centre apertures stay EXPERIMENTAL per docs/guides/support_matrix.md. THREE SETUP DEFECTS were found during this campaign, each having corrupted an earlier revision's numbers and each now fenced by an assert or a derived setting: (1) a parasitic wall-slot (fins drawn to the NOMINAL guide width leave a 1-cell gap at the actual grid wall); (2) node-plane box corners are half-ulp fragile because the volume mask is half-open over NODE coordinates — one fine config rasterized 3 thickness nodes instead of 4, and an apparent +/-0.07 'domain sensitivity' was that ulp lottery; (3) the fin footprint made the ELECTRICAL aperture d + 2*dx instead of d, which alone inflated the envelope 4-6x, and a 0.5*lambda_g absorber left the envelope set by CPML reflection rather than discretization (PR #480 review B2/B3; CPML is now 0.75*lambda_g at the band edge = 60 coarse / 120 fine). RETRACTED: an earlier revision fenced normalize=True modal extraction on the strength of a measured column power 1.112-1.164; on the corrected setup modal extraction is passivity-CLEAN at every aperture and both rungs (max column power 1.0207 at d = 7.62 mm / a-30, 1.0013 at d = 18.288 mm, ZERO extractor warnings), so that non-passivity was a symptom of defects (1)-(3) and not a reflector-inflation property of the extractor — the fence is withdrawn and the measurement is committed as modal_extraction_witness, which also records modal ACCURACY so the retraction does not rest on passivity alone: modal &#124;S11&#124; gaps come out comparable to flux and consistently a little worse, which is why flux still carries the gate. Palace WavePort corroboration (stage S2) and a published multi-iris filter (stage S3) are follow-on stages, not claimed here. APERTURE RESOLUTION (issue #812 re-gate, 2026-09-01): the fine gate is now per-CONFIGURATION -- gate = round-up(that configuration's own committed envelope x 1.5) at quantum 1000, giving 0.019/0.034/0.015/0.022/0.035/0.015/0.034/0.034 for the eight configs -- because the pooled 0.04 was set by the worst configuration and then spent at all eight. Measured against those gates, a one-cell aperture error at each rung (the smallest the grid-snapped geometry can express, and the campaign's own setup defect (3) at half its size) is detected as an OVER-aperture at every configuration and as an UNDER-aperture at only two of the eight, both at the weak aperture and both below the repo's own 1.5x margin. Those counts, the margins and the per-configuration oracle distances are COMMITTED rather than restated here: validation/crossval/_18_wr90_iris_results/aperture_resolution.json, keys summary.over_aperture_detected, summary.over_aperture_min_margin_x, summary.under_aperture_detected, summary.under_aperture_detected_configs and summary.under_aperture_max_margin_x, with the per-configuration rows under pairs[*] (pairs[2] is d = 7.620 mm centred); each one is re-derived from the committed traces by an INDEPENDENT oracle in tests/crossval/test_wr90_iris_modematch_gates.py. A one-cell under-aperture is therefore NOT resolved with margin at any configuration, and at d = 12.192 and d = 7.620 mm it is not resolved at all: at both d = 7.620 configurations the modelled under-aperture defect scores BETTER than the undefected committed row (summary.under_aperture_scores_better_configs; pairs[*].one_cell_defect.under.scores_better_than_undefected), because the fine rung's own staircase error is an effective aperture WIDER than nominal rather than narrower -- over the declared offset grid the committed fine trace's NEAREST oracle sits at d PLUS half a fine cell at all eight configurations (summary.nearest_offset_fine_cells_values; pairs[*].oracle_distance_abs) -- so narrowing the geometry by one cell moves it TOWARD the trace instead of away. CORRECTION (issue #812 round 2): an earlier revision of this paragraph asserted the opposite sign, that the committed fine trace sat closer to the oracle one fine cell NARROW, quoting the under-aperture DEFECT metric as if it were that distance; that claim was mis-sourced and sign-inverted, and aperture_resolution.json is now the only source for this class. The Richardson witness is blind to this whole class in both signs at all eight configurations BY CONSTRUCTION: an aperture error of one cell at each rung is proportional to dx, which is exactly what 2*S(a/60) - S(a/30) is built to remove, so no tightening of its 0.01 gate can catch it and none is attempted. The calibration this case supplies to any downstream multi-iris filter is aperture-resolved to +1 fine cell, NOT to -1. The three declared apertures are now pinned as claims (G18-C): each must be an exact and EVEN integer cell count at BOTH rungs, a geometric condition no one-fine-cell relabel can satisfy."` | `"One symmetric inductive PEC iris (t = 1.524 mm = exactly 2 coarse / 4 fine cells; apertures 18.288/12.192/7.62 mm, grid-exact) in WR-90 over 8.2-12.4 GHz on 29 frequency points, flux-normalized &#124;S11&#124; vs a twice-implemented TEn0 mode-matching cascade oracle (self-witnesses: unitarity 1.1e-16, mode convergence 4.3e-5, Marcuvitz cot^2 thin-limit anchor 10.8% with the inductive sign, d->a and deep-constriction limits; the PR #480 review reproduced the oracle with a formulation-independent 2-D H-plane FDFD to 6e-4 and measured rfx's same-geometry agreement at <= 0.02 — attributed, not imported). GATED: fine rung dx = a/60 within 0.02 abs = round-up(measured envelope 0.0106 x 1.5) over 8 configs (3 apertures x {centred, iris off-centre at 0.42 of the guide} + 2 extra guide lengths; every config lands within 0.011, so no single configuration sets the envelope), and the Richardson extrapolation 2*S(a/60) - S(a/30) on the oracle within 0.01 abs (envelope 0.0046) at EVERY one of those 8 pairs, which cross-confirms the oracle and the first-order attribution (gap ratios 0.407-0.440 = textbook first order). REPORTED, NOT GATED: the coarse rung dx = a/30 (0.008-0.025 abs); the raw normalize=False record, which is WORSE than flux (gaps 0.009-0.025) with a pointwise &#124;raw - flux&#124; difference up to 0.0068 at the wide aperture; residual detrended ripple, i.e. a quadratic detrend of &#124;S11&#124; MINUS the oracle so the oracle's own curvature is not counted (fine <= 0.0076, coarse <= 0.0152, both at the wide aperture with the iris off-centre, down from the 0.0706 the PR #480 review measured on the same basis before the absorber fix); and phase (magnitude-only lane posture). FENCED, never gated: everything beyond ONE symmetric inductive iris — multi-iris filters, posts, septa and off-centre apertures stay EXPERIMENTAL per docs/guides/support_matrix.md. THREE SETUP DEFECTS were found during this campaign, each having corrupted an earlier revision's numbers and each now fenced by an assert or a derived setting: (1) a parasitic wall-slot (fins drawn to the NOMINAL guide width leave a 1-cell gap at the actual grid wall); (2) node-plane box corners were half-ulp fragile under the pre-#931 half-open NODE mask — one fine config rasterized 3 thickness nodes instead of 4, and an apparent +/-0.07 'domain sensitivity' was that ulp lottery — so every corner was moved half a cell OFF the node planes. The #931 lattice ownership contract INVERTS that: a PEC volume is sampled at cell CENTRES, so a node-plane corner selects whole cells and is the well-defined position while a half-cell offset lands exactly on a centre. The corners are back on the node planes and the footprint asserts read the realized edge set rather than a sigma mask; (3) the fin footprint made the ELECTRICAL aperture d + 2*dx instead of d, which alone inflated the envelope 4-6x, and a 0.5*lambda_g absorber left the envelope set by CPML reflection rather than discretization (PR #480 review B2/B3; CPML is now 0.75*lambda_g at the band edge = 60 coarse / 120 fine). THE #931 THICKNESS CORRECTION: until the contract landed, this case fed its oracle the drawn t = 1.524 mm while the lattice realized (t_c - 1)*dx — 0.762 mm at a/30 and 1.143 mm at a/60, a 50% / 25% thickness deficit — because a body's far face was never a wall. Nothing in this case measured that: every assert counted MASKED PLANES, a quantity that agreed with the drawing by construction. Under the contract the realized thickness is the drawn thickness and the oracle input is correct for the first time; the whole record was regenerated on the corrected geometry and every gate re-derived from the new envelopes. The case also gains the contract's one-cell volume witness (one_cell_volume_witness): an iris-thickness sweep t = 1..8 cells against the lattice-blind mode-matching oracle, so that a one-cell PEC body standing two walls has an independent check rather than a thin-limit anchor that only speaks about t -> 0. Each rung is asked to IDENTIFY its own thickness -- the oracle at t-1, t and t+1 cells, argmin on t -- and all seven do; at t = 1 the pre-#931 one-wall alternative (a zero-thickness screen) is 4.32x worse than the two-wall one, so the contract's rule at one cell is measured rather than assumed. The witness's first-stated criterion (t = 1 inside the t = 2..8 range) is RETIRED as vacuous and its verdict kept: the residual is monotone in t, so t = 1 is the extremum whatever the physics does, and a perfect 0.0000 would fail it too. RETRACTED: an earlier revision fenced normalize=True modal extraction on the strength of a measured column power 1.112-1.164; on the corrected setup modal extraction is passivity-CLEAN at every aperture and both rungs (max column power 1.0200 at d = 7.62 mm / a-30, 1.0012 at d = 18.288 mm, ZERO extractor warnings), so that non-passivity was a symptom of defects (1)-(3) and not a reflector-inflation property of the extractor — the fence is withdrawn and the measurement is committed as modal_extraction_witness, which also records modal ACCURACY so the retraction does not rest on passivity alone: modal &#124;S11&#124; gaps come out comparable to flux and consistently a little worse, which is why flux still carries the gate. Palace WavePort corroboration (stage S2) and a published multi-iris filter (stage S3) are follow-on stages, not claimed here. APERTURE RESOLUTION (issue #812 re-gate 2026-09-01, RE-MEASURED under #931 2026-09-07): the fine gate is per-CONFIGURATION -- gate = round-up(that configuration's own committed envelope x 1.5) at quantum 1000, giving 0.012/0.016/0.006/0.016/0.016/0.006/0.015/0.015 for the eight configs -- because the pooled gate is set by the worst configuration and then spent at all eight. All eight moved DOWN when the thickness deficit closed (0.019/0.034/0.015/0.022/0.035/0.015/0.034/0.034 before it), which is a re-derivation of the same rule on a better geometry, not a re-tuning. Measured against those gates, a one-cell aperture error at each rung (the smallest the grid-snapped geometry can express, and the campaign's own setup defect (3) at half its size) is now detected in BOTH signs at every one of the eight configurations, at worst 1.623x the gate for an over-aperture and 2.608x for an under-aperture. Those counts, the margins and the per-configuration oracle distances are COMMITTED rather than restated here: validation/crossval/_18_wr90_iris_results/aperture_resolution.json, keys summary.over_aperture_detected, summary.over_aperture_min_margin_x, summary.under_aperture_detected, summary.under_aperture_detected_configs, summary.under_aperture_min_margin_x and summary.under_aperture_max_margin_x, with the per-configuration rows under pairs[*] (pairs[2] is d = 7.620 mm centred); each one is re-derived from the committed traces by an INDEPENDENT oracle in tests/crossval/test_wr90_iris_modematch_gates.py. WHAT #931 CHANGED HERE, and it is the whole paragraph: before the contract, the committed fine trace's NEAREST oracle over the declared offset grid sat at d PLUS half a fine cell at all eight configurations, an apparent effective aperture WIDER than nominal; a one-cell under-aperture therefore moved the geometry TOWARD the trace, scored BETTER than the undefected row at both d = 7.620 configurations, and was detected at only two of the eight. That half-cell offset was not an aperture property at all -- it was the thickness deficit ((t_c - 1)*dx instead of t) reading out on the aperture axis, the two being the only free dimensions of a symmetric iris. With the realized thickness equal to the drawn one, the nearest oracle sits at the DECLARED d at all eight configurations (summary.nearest_offset_fine_cells_values == [0.0]), no defect scores better than the undefected row (summary.under_aperture_scores_better_configs == []), and the asymmetry between the two signs is gone. CORRECTION HISTORY (issue #812 round 2): an earlier revision of this paragraph asserted that the committed fine trace sat closer to the oracle one fine cell NARROW, quoting the under-aperture DEFECT metric as if it were that distance; that claim was mis-sourced and sign-inverted, and aperture_resolution.json is the only source for this class. The Richardson witness is blind to this whole class in both signs at all eight configurations BY CONSTRUCTION: an aperture error of one cell at each rung is proportional to dx, which is exactly what 2*S(a/60) - S(a/30) is built to remove, so no tightening of its 0.01 gate can catch it and none is attempted. The calibration this case supplies to any downstream multi-iris filter is aperture-resolved to one fine cell in both signs. The three declared apertures are pinned as claims (G18-C): each must be an exact and EVEN integer cell count at BOTH rungs, a geometric condition no one-fine-cell relabel can satisfy."` | — |
| `/coarse_diagnostic/0/aperture_cells` | `23` | *absent* | — |
| `/coarse_diagnostic/0/aperture_wall_nodes` | *absent* | `[3, 27]` | 0–1 |
| `/coarse_diagnostic/0/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/coarse_diagnostic/0/max_gap_abs` | `0.0202` | `0.0193` | — |
| `/coarse_diagnostic/0/realized_aperture_cells` | *absent* | `24` | — |
| `/coarse_diagnostic/0/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/0/richardson_dev_abs` | `0.0042` | `0.0034` | — |
| `/coarse_diagnostic/0/s11` | array[29], SHA256 `5fd05eebb8c1` | array[29], SHA256 `bda599c3f8c3` | 0–28 |
| `/coarse_diagnostic/0/s21` | array[29], SHA256 `37f78dfc0e6e` | array[29], SHA256 `3d5cbf3112ed` | 0–28 |
| `/coarse_diagnostic/0/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/0/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/0/wall_s` | `66.0` | `25.2` | — |
| `/coarse_diagnostic/1/aperture_cells` | `15` | *absent* | — |
| `/coarse_diagnostic/1/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/coarse_diagnostic/1/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/coarse_diagnostic/1/max_gap_abs` | `0.0405` | `0.0246` | — |
| `/coarse_diagnostic/1/realized_aperture_cells` | *absent* | `16` | — |
| `/coarse_diagnostic/1/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/1/richardson_dev_abs` | `0.005` | `0.0045` | — |
| `/coarse_diagnostic/1/s11` | array[29], SHA256 `cf29bc66aa1e` | array[29], SHA256 `40f1fb558306` | 0–28 |
| `/coarse_diagnostic/1/s21` | array[29], SHA256 `f3bfa41f708a` | array[29], SHA256 `399c1f6476a8` | 0–28 |
| `/coarse_diagnostic/1/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/1/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/1/wall_s` | `65.8` | `27.7` | — |
| `/coarse_diagnostic/2/aperture_cells` | `9` | *absent* | — |
| `/coarse_diagnostic/2/aperture_wall_nodes` | *absent* | `[10, 20]` | 0–1 |
| `/coarse_diagnostic/2/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/coarse_diagnostic/2/max_gap_abs` | `0.0184` | `0.0081` | — |
| `/coarse_diagnostic/2/realized_aperture_cells` | *absent* | `10` | — |
| `/coarse_diagnostic/2/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/2/richardson_dev_abs` | `0.001` | `0.0012` | — |
| `/coarse_diagnostic/2/s11` | array[29], SHA256 `5a94f9871f44` | array[29], SHA256 `bd791aada6ca` | 0–28 |
| `/coarse_diagnostic/2/s21` | array[29], SHA256 `a9b6858dcc87` | array[29], SHA256 `5940957773af` | 0–28 |
| `/coarse_diagnostic/2/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/2/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/2/wall_s` | `65.6` | `27.6` | — |
| `/coarse_diagnostic/3/aperture_cells` | `23` | *absent* | — |
| `/coarse_diagnostic/3/aperture_wall_nodes` | *absent* | `[3, 27]` | 0–1 |
| `/coarse_diagnostic/3/iris_wall_nodes` | *absent* | `[169, 171]` | 0–1 |
| `/coarse_diagnostic/3/max_gap_abs` | `0.0256` | `0.0232` | — |
| `/coarse_diagnostic/3/realized_aperture_cells` | *absent* | `24` | — |
| `/coarse_diagnostic/3/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/3/richardson_dev_abs` | `0.0039` | `0.0036` | — |
| `/coarse_diagnostic/3/s11` | array[29], SHA256 `5f2b042f96e4` | array[29], SHA256 `9f851d1128f2` | 0–28 |
| `/coarse_diagnostic/3/s21` | array[29], SHA256 `682abde4f3cf` | array[29], SHA256 `695c9f6244fb` | 0–28 |
| `/coarse_diagnostic/3/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/3/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/3/wall_s` | `67.4` | `25.7` | — |
| `/coarse_diagnostic/4/aperture_cells` | `15` | *absent* | — |
| `/coarse_diagnostic/4/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/coarse_diagnostic/4/iris_wall_nodes` | *absent* | `[169, 171]` | 0–1 |
| `/coarse_diagnostic/4/max_gap_abs` | `0.043` | `0.0252` | — |
| `/coarse_diagnostic/4/realized_aperture_cells` | *absent* | `16` | — |
| `/coarse_diagnostic/4/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/4/richardson_dev_abs` | `0.005` | `0.0045` | — |
| `/coarse_diagnostic/4/s11` | array[29], SHA256 `6eecae5815e2` | array[29], SHA256 `b7ecfa16b9ea` | 0–28 |
| `/coarse_diagnostic/4/s21` | array[29], SHA256 `4fe7c4d61ef5` | array[29], SHA256 `2e8f99f77179` | 0–28 |
| `/coarse_diagnostic/4/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/4/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/4/wall_s` | `64.7` | `26.2` | — |
| `/coarse_diagnostic/5/aperture_cells` | `9` | *absent* | — |
| `/coarse_diagnostic/5/aperture_wall_nodes` | *absent* | `[10, 20]` | 0–1 |
| `/coarse_diagnostic/5/iris_wall_nodes` | *absent* | `[169, 171]` | 0–1 |
| `/coarse_diagnostic/5/max_gap_abs` | `0.0183` | `0.0082` | — |
| `/coarse_diagnostic/5/realized_aperture_cells` | *absent* | `10` | — |
| `/coarse_diagnostic/5/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/5/richardson_dev_abs` | `0.001` | `0.0013` | — |
| `/coarse_diagnostic/5/s11` | array[29], SHA256 `6aaa7ef5849a` | array[29], SHA256 `c313a64c04db` | 0–28 |
| `/coarse_diagnostic/5/s21` | array[29], SHA256 `b70219bb9423` | array[29], SHA256 `85c2523747f3` | 0–28 |
| `/coarse_diagnostic/5/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/5/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/5/wall_s` | `63.8` | `24.2` | — |
| `/coarse_diagnostic/6/aperture_cells` | `15` | *absent* | — |
| `/coarse_diagnostic/6/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/coarse_diagnostic/6/iris_wall_nodes` | *absent* | `[164, 166]` | 0–1 |
| `/coarse_diagnostic/6/max_gap_abs` | `0.0405` | `0.0246` | — |
| `/coarse_diagnostic/6/realized_aperture_cells` | *absent* | `16` | — |
| `/coarse_diagnostic/6/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/6/richardson_dev_abs` | `0.0051` | `0.0045` | — |
| `/coarse_diagnostic/6/s11` | array[29], SHA256 `aa9fb71568aa` | array[29], SHA256 `77ba4951bc5a` | 0–28 |
| `/coarse_diagnostic/6/s21` | array[29], SHA256 `d693a294d6f1` | array[29], SHA256 `f6a127703d6d` | 0–28 |
| `/coarse_diagnostic/6/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/6/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/6/wall_s` | `60.5` | `17.5` | — |
| `/coarse_diagnostic/7/aperture_cells` | `15` | *absent* | — |
| `/coarse_diagnostic/7/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/coarse_diagnostic/7/iris_wall_nodes` | *absent* | `[217, 219]` | 0–1 |
| `/coarse_diagnostic/7/max_gap_abs` | `0.0401` | `0.0245` | — |
| `/coarse_diagnostic/7/realized_aperture_cells` | *absent* | `16` | — |
| `/coarse_diagnostic/7/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/7/richardson_dev_abs` | `0.005` | `0.0046` | — |
| `/coarse_diagnostic/7/s11` | array[29], SHA256 `510999ea23ac` | array[29], SHA256 `74c027fbec56` | 0–28 |
| `/coarse_diagnostic/7/s21` | array[29], SHA256 `ce37344265a9` | array[29], SHA256 `1ce987a21f76` | 0–28 |
| `/coarse_diagnostic/7/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/7/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/7/wall_s` | `71.1` | `28.4` | — |
| `/gated_fine/0/aperture_cells` | `47` | *absent* | — |
| `/gated_fine/0/aperture_wall_nodes` | *absent* | `[6, 54]` | 0–1 |
| `/gated_fine/0/fine_gate_abs` | *absent* | `0.012` | — |
| `/gated_fine/0/iris_wall_nodes` | *absent* | `[380, 384]` | 0–1 |
| `/gated_fine/0/max_gap_abs` | `0.0122` | `0.0079` | — |
| `/gated_fine/0/realized_aperture_cells` | *absent* | `48` | — |
| `/gated_fine/0/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/0/s11` | array[29], SHA256 `305f4b231b58` | array[29], SHA256 `54c1a47f3087` | 0–28 |
| `/gated_fine/0/s21` | array[29], SHA256 `8cd9fc99a011` | array[29], SHA256 `a7a0f4c88505` | 0–28 |
| `/gated_fine/0/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/0/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/0/wall_s` | `823.4` | `424.3` | — |
| `/gated_fine/1/aperture_cells` | `31` | *absent* | — |
| `/gated_fine/1/aperture_wall_nodes` | *absent* | `[14, 46]` | 0–1 |
| `/gated_fine/1/fine_gate_abs` | *absent* | `0.016` | — |
| `/gated_fine/1/iris_wall_nodes` | *absent* | `[380, 384]` | 0–1 |
| `/gated_fine/1/max_gap_abs` | `0.0223` | `0.0101` | — |
| `/gated_fine/1/realized_aperture_cells` | *absent* | `32` | — |
| `/gated_fine/1/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/1/s11` | array[29], SHA256 `594f86e4ae94` | array[29], SHA256 `2dd8dd57effe` | 0–28 |
| `/gated_fine/1/s21` | array[29], SHA256 `f9a9b7e47dd7` | array[29], SHA256 `878ceed4b42c` | 0–28 |
| `/gated_fine/1/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/1/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/1/wall_s` | `766.5` | `418.9` | — |
| `/gated_fine/2/aperture_cells` | `19` | *absent* | — |
| `/gated_fine/2/aperture_wall_nodes` | *absent* | `[20, 40]` | 0–1 |
| `/gated_fine/2/fine_gate_abs` | *absent* | `0.006` | — |
| `/gated_fine/2/iris_wall_nodes` | *absent* | `[380, 384]` | 0–1 |
| `/gated_fine/2/max_gap_abs` | `0.0097` | `0.0034` | — |
| `/gated_fine/2/realized_aperture_cells` | *absent* | `20` | — |
| `/gated_fine/2/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/2/s11` | array[29], SHA256 `5ac0f4d88e80` | array[29], SHA256 `e4635c6aa431` | 0–28 |
| `/gated_fine/2/s21` | array[29], SHA256 `0dc81ce656cb` | array[29], SHA256 `066b30fd703b` | 0–28 |
| `/gated_fine/2/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/2/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/2/wall_s` | `756.3` | `415.9` | — |
| `/gated_fine/3/aperture_cells` | `47` | *absent* | — |
| `/gated_fine/3/aperture_wall_nodes` | *absent* | `[6, 54]` | 0–1 |
| `/gated_fine/3/fine_gate_abs` | *absent* | `0.016` | — |
| `/gated_fine/3/iris_wall_nodes` | *absent* | `[338, 342]` | 0–1 |
| `/gated_fine/3/max_gap_abs` | `0.0145` | `0.0102` | — |
| `/gated_fine/3/realized_aperture_cells` | *absent* | `48` | — |
| `/gated_fine/3/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/3/s11` | array[29], SHA256 `b5fc974d7452` | array[29], SHA256 `a56501b6921f` | 0–28 |
| `/gated_fine/3/s21` | array[29], SHA256 `69dc06b62af9` | array[29], SHA256 `acb5ce12a592` | 0–28 |
| `/gated_fine/3/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/3/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/3/wall_s` | `761.4` | `413.7` | — |
| `/gated_fine/4/aperture_cells` | `31` | *absent* | — |
| `/gated_fine/4/aperture_wall_nodes` | *absent* | `[14, 46]` | 0–1 |
| `/gated_fine/4/fine_gate_abs` | *absent* | `0.016` | — |
| `/gated_fine/4/iris_wall_nodes` | *absent* | `[338, 342]` | 0–1 |
| `/gated_fine/4/max_gap_abs` | `0.0232` | `0.0106` | — |
| `/gated_fine/4/realized_aperture_cells` | *absent* | `32` | — |
| `/gated_fine/4/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/4/s11` | array[29], SHA256 `2c6df57b63d9` | array[29], SHA256 `09176314a5e0` | 0–28 |
| `/gated_fine/4/s21` | array[29], SHA256 `07c6295d6743` | array[29], SHA256 `5b3d74897c05` | 0–28 |
| `/gated_fine/4/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/4/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/4/wall_s` | `758.2` | `419.7` | — |
| `/gated_fine/5/aperture_cells` | `19` | *absent* | — |
| `/gated_fine/5/aperture_wall_nodes` | *absent* | `[20, 40]` | 0–1 |
| `/gated_fine/5/fine_gate_abs` | *absent* | `0.006` | — |
| `/gated_fine/5/iris_wall_nodes` | *absent* | `[338, 342]` | 0–1 |
| `/gated_fine/5/max_gap_abs` | `0.0097` | `0.0035` | — |
| `/gated_fine/5/realized_aperture_cells` | *absent* | `20` | — |
| `/gated_fine/5/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/5/s11` | array[29], SHA256 `9ae33598b24f` | array[29], SHA256 `df07ed9053e2` | 0–28 |
| `/gated_fine/5/s21` | array[29], SHA256 `04331e70b6f6` | array[29], SHA256 `9c341ca9ec7f` | 0–28 |
| `/gated_fine/5/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/5/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/5/wall_s` | `757.7` | `427.0` | — |
| `/gated_fine/6/aperture_cells` | `31` | *absent* | — |
| `/gated_fine/6/aperture_wall_nodes` | *absent* | `[14, 46]` | 0–1 |
| `/gated_fine/6/fine_gate_abs` | *absent* | `0.015` | — |
| `/gated_fine/6/iris_wall_nodes` | *absent* | `[328, 332]` | 0–1 |
| `/gated_fine/6/max_gap_abs` | `0.0222` | `0.01` | — |
| `/gated_fine/6/realized_aperture_cells` | *absent* | `32` | — |
| `/gated_fine/6/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/6/s11` | array[29], SHA256 `d9e5237bd0de` | array[29], SHA256 `d797aca93eb9` | 0–28 |
| `/gated_fine/6/s21` | array[29], SHA256 `2af47ec230c3` | array[29], SHA256 `b2cc66f58bab` | 0–28 |
| `/gated_fine/6/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/6/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/6/wall_s` | `696.4` | `352.1` | — |
| `/gated_fine/7/aperture_cells` | `31` | *absent* | — |
| `/gated_fine/7/aperture_wall_nodes` | *absent* | `[14, 46]` | 0–1 |
| `/gated_fine/7/fine_gate_abs` | *absent* | `0.015` | — |
| `/gated_fine/7/iris_wall_nodes` | *absent* | `[433, 437]` | 0–1 |
| `/gated_fine/7/max_gap_abs` | `0.0222` | `0.01` | — |
| `/gated_fine/7/realized_aperture_cells` | *absent* | `32` | — |
| `/gated_fine/7/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/7/s11` | array[29], SHA256 `96c7aa45fd71` | array[29], SHA256 `e52f4b46c7c6` | 0–28 |
| `/gated_fine/7/s21` | array[29], SHA256 `0a0dea4e5cdd` | array[29], SHA256 `fe9a2f7fc9ed` | 0–28 |
| `/gated_fine/7/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/7/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/7/wall_s` | `840.7` | `496.3` | — |
| `/gates/fine_gate_abs` | `0.04` | `0.02` | — |
| `/gates/fine_gate_abs_per_config/12.192|0.16|0.50` | `0.034` | `0.015` | — |
| `/gates/fine_gate_abs_per_config/12.192|0.20|0.42` | `0.035` | `0.016` | — |
| `/gates/fine_gate_abs_per_config/12.192|0.20|0.50` | `0.034` | `0.016` | — |
| `/gates/fine_gate_abs_per_config/12.192|0.24|0.50` | `0.034` | `0.015` | — |
| `/gates/fine_gate_abs_per_config/18.288|0.20|0.42` | `0.022` | `0.016` | — |
| `/gates/fine_gate_abs_per_config/18.288|0.20|0.50` | `0.019` | `0.012` | — |
| `/gates/fine_gate_abs_per_config/7.620|0.20|0.42` | `0.015` | `0.006` | — |
| `/gates/fine_gate_abs_per_config/7.620|0.20|0.50` | `0.015` | `0.006` | — |
| `/gates/fine_measured_envelope_abs` | `0.0232` | `0.0106` | — |
| `/gates/first_order_ratios` | array[8], SHA256 `bf6e482da00f` | array[8], SHA256 `3ea7b0725f7d` | 0–7 |
| `/gates/posture` | `"gate = round-UP(measured envelope x 1.5), enforced as EXACT equality by the write-fixture self-check (PR #475 convention, PR #480 tightening); coarse rung, raw extraction, ripple and phase are reported, never gated; modal extraction is no longer fenced (retracted, see provenance) but structures beyond one symmetric inductive iris remain fenced, never gated; issue #812 re-gate: the BINDING fine gate is now per-configuration, gate = round-UP(that configuration's own envelope x 1.5) at quantum 1000, with the pooled 0.04 retained unchanged as a ceiling and the one-cell aperture detection table gated as its own claim"` | `"gate = round-UP(measured envelope x 1.5), enforced as EXACT equality by the write-fixture self-check (PR #475 convention, PR #480 tightening); coarse rung, raw extraction, ripple and phase are reported, never gated; modal extraction is no longer fenced (retracted, see provenance) but structures beyond one symmetric inductive iris remain fenced, never gated; issue #812 re-gate: the BINDING fine gate is now per-configuration, gate = round-UP(that configuration's own envelope x 1.5) at quantum 1000, with the pooled 0.02 kept as a ceiling and the one-cell aperture detection table gated as its own claim; #931: pooled 0.04 -> 0.02 and all eight per-config gates re-derived DOWN from the corrected-thickness envelopes (VESSL 369367259159), never widened"` | — |
| `/gates/richardson_measured_envelope_abs` | `0.0051` | `0.0046` | — |
| `/modal_extraction_witness/rows/0/max_colpow` | `1.0207` | `1.02` | — |
| `/modal_extraction_witness/rows/0/max_gap_abs` | `0.0189` | `0.0119` | — |
| `/modal_extraction_witness/rows/0/s11` | array[29], SHA256 `bff29b1b2dfe` | array[29], SHA256 `d1a5d8f68e37` | 0–28 |
| `/modal_extraction_witness/rows/1/max_colpow` | `1.0101` | `1.0099` | — |
| `/modal_extraction_witness/rows/1/max_gap_abs` | `0.0099` | `0.0059` | — |
| `/modal_extraction_witness/rows/1/s11` | array[29], SHA256 `63f5e974bfc1` | array[29], SHA256 `db46109e24b7` | 0–28 |
| `/modal_extraction_witness/rows/2/max_colpow` | `1.0144` | `1.015` | — |
| `/modal_extraction_witness/rows/2/max_gap_abs` | `0.046` | `0.0269` | — |
| `/modal_extraction_witness/rows/2/s11` | array[29], SHA256 `721985c5b000` | array[29], SHA256 `210aa5aae51e` | 0–28 |
| `/modal_extraction_witness/rows/3/max_colpow` | `1.0013` | `1.0012` | — |
| `/modal_extraction_witness/rows/3/max_gap_abs` | `0.0234` | `0.019` | — |
| `/modal_extraction_witness/rows/3/s11` | array[29], SHA256 `77bc9ae97010` | array[29], SHA256 `fb2f88671f91` | 0–28 |
| `/one_cell_aperture_detection_witness/0/config` | *absent* | `"18.288&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/0/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/0/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/0/fine_gap_abs` | *absent* | `0.0245` | — |
| `/one_cell_aperture_detection_witness/0/fine_gate_abs` | *absent* | `0.012` | — |
| `/one_cell_aperture_detection_witness/0/richardson_dev_abs` | *absent* | `0.0061` | — |
| `/one_cell_aperture_detection_witness/0/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/1/config` | *absent* | `"18.288&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/1/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/1/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/1/fine_gap_abs` | *absent* | `0.0426` | — |
| `/one_cell_aperture_detection_witness/1/fine_gate_abs` | *absent* | `0.012` | — |
| `/one_cell_aperture_detection_witness/1/richardson_dev_abs` | *absent* | `0.0054` | — |
| `/one_cell_aperture_detection_witness/1/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/2/config` | *absent* | `"12.192&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/2/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/2/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/2/fine_gap_abs` | *absent* | `0.0336` | — |
| `/one_cell_aperture_detection_witness/2/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/2/richardson_dev_abs` | *absent* | `0.0062` | — |
| `/one_cell_aperture_detection_witness/2/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/3/config` | *absent* | `"12.192&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/3/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/3/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/3/fine_gap_abs` | *absent* | `0.0538` | — |
| `/one_cell_aperture_detection_witness/3/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/3/richardson_dev_abs` | *absent* | `0.0056` | — |
| `/one_cell_aperture_detection_witness/3/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/4/config` | *absent* | `"7.620&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/4/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/4/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/4/fine_gap_abs` | *absent* | `0.0134` | — |
| `/one_cell_aperture_detection_witness/4/fine_gate_abs` | *absent* | `0.006` | — |
| `/one_cell_aperture_detection_witness/4/richardson_dev_abs` | *absent* | `0.0027` | — |
| `/one_cell_aperture_detection_witness/4/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/5/config` | *absent* | `"7.620&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/5/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/5/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/5/fine_gap_abs` | *absent* | `0.0167` | — |
| `/one_cell_aperture_detection_witness/5/fine_gate_abs` | *absent* | `0.006` | — |
| `/one_cell_aperture_detection_witness/5/richardson_dev_abs` | *absent* | `0.0016` | — |
| `/one_cell_aperture_detection_witness/5/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/6/config` | *absent* | `"18.288&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/6/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/6/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/6/fine_gap_abs` | *absent* | `0.026` | — |
| `/one_cell_aperture_detection_witness/6/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/6/richardson_dev_abs` | *absent* | `0.0063` | — |
| `/one_cell_aperture_detection_witness/6/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/7/config` | *absent* | `"18.288&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/7/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/7/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/7/fine_gap_abs` | *absent* | `0.0417` | — |
| `/one_cell_aperture_detection_witness/7/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/7/richardson_dev_abs` | *absent* | `0.0056` | — |
| `/one_cell_aperture_detection_witness/7/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/8/config` | *absent* | `"12.192&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/8/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/8/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/8/fine_gap_abs` | *absent* | `0.0337` | — |
| `/one_cell_aperture_detection_witness/8/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/8/richardson_dev_abs` | *absent* | `0.0061` | — |
| `/one_cell_aperture_detection_witness/8/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/9/config` | *absent* | `"12.192&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/9/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/9/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/9/fine_gap_abs` | *absent* | `0.054` | — |
| `/one_cell_aperture_detection_witness/9/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/9/richardson_dev_abs` | *absent* | `0.0056` | — |
| `/one_cell_aperture_detection_witness/9/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/10/config` | *absent* | `"7.620&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/10/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/10/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/10/fine_gap_abs` | *absent* | `0.0134` | — |
| `/one_cell_aperture_detection_witness/10/fine_gate_abs` | *absent* | `0.006` | — |
| `/one_cell_aperture_detection_witness/10/richardson_dev_abs` | *absent* | `0.0027` | — |
| `/one_cell_aperture_detection_witness/10/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/11/config` | *absent* | `"7.620&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/11/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/11/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/11/fine_gap_abs` | *absent* | `0.0167` | — |
| `/one_cell_aperture_detection_witness/11/fine_gate_abs` | *absent* | `0.006` | — |
| `/one_cell_aperture_detection_witness/11/richardson_dev_abs` | *absent* | `0.0016` | — |
| `/one_cell_aperture_detection_witness/11/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/12/config` | *absent* | `"12.192&#124;0.16&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/12/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/12/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/12/fine_gap_abs` | *absent* | `0.0335` | — |
| `/one_cell_aperture_detection_witness/12/fine_gate_abs` | *absent* | `0.015` | — |
| `/one_cell_aperture_detection_witness/12/richardson_dev_abs` | *absent* | `0.0062` | — |
| `/one_cell_aperture_detection_witness/12/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/13/config` | *absent* | `"12.192&#124;0.16&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/13/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/13/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/13/fine_gap_abs` | *absent* | `0.0538` | — |
| `/one_cell_aperture_detection_witness/13/fine_gate_abs` | *absent* | `0.015` | — |
| `/one_cell_aperture_detection_witness/13/richardson_dev_abs` | *absent* | `0.0057` | — |
| `/one_cell_aperture_detection_witness/13/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/14/config` | *absent* | `"12.192&#124;0.24&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/14/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/14/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/14/fine_gap_abs` | *absent* | `0.0336` | — |
| `/one_cell_aperture_detection_witness/14/fine_gate_abs` | *absent* | `0.015` | — |
| `/one_cell_aperture_detection_witness/14/richardson_dev_abs` | *absent* | `0.0063` | — |
| `/one_cell_aperture_detection_witness/14/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/15/config` | *absent* | `"12.192&#124;0.24&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/15/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/15/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/15/fine_gap_abs` | *absent* | `0.0538` | — |
| `/one_cell_aperture_detection_witness/15/fine_gate_abs` | *absent* | `0.015` | — |
| `/one_cell_aperture_detection_witness/15/richardson_dev_abs` | *absent* | `0.0058` | — |
| `/one_cell_aperture_detection_witness/15/sign` | *absent* | `-1` | — |
| `/one_cell_volume_witness/aperture_mm` | *absent* | `12.192` | — |
| `/one_cell_volume_witness/cells_per_a` | *absent* | `30` | — |
| `/one_cell_volume_witness/identified_every_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/monotone_range_criterion/status` | *absent* | `"RETIRED — vacuous for a monotone residual"` | — |
| `/one_cell_volume_witness/monotone_range_criterion/verdict` | *absent* | `false` | — |
| `/one_cell_volume_witness/multi_cell_gap_range_abs` | *absent* | `[0.0109, 0.0246]` | 0–1 |
| `/one_cell_volume_witness/note` | *absent* | `"#931 lattice ownership contract, design note 20260906 section 5: a PEC volume one cell thick is a filled slab with a tangential wall at BOTH faces, at every thickness, with no flag. Before #931 the far face was never a wall and a two_plane flag put it back for t = 1 only; nothing independent said which was right AT ONE CELL, because the thin-limit anchor is a t -> 0 statement rather than a t = dx one. Here the mode-matching oracle — which takes the physical t and knows nothing about the lattice — is run against rfx at t = 1..8 cells on the coarse rung at the worst-gap aperture. GATE: every swept rung must IDENTIFY its own drawn thickness — the oracle is evaluated at t-1, t and t+1 cells and the residual argmin must land on t. At t = 1 the t-1 alternative is precisely the pre-#931 realization (one wall, a zero-thickness screen), so the rule in dispute is decided by a measurement rather than by a convention. RETIRED, and recorded rather than deleted (monotone_range_criterion): the original statement — the t = 1 residual lies inside the range t = 2..8 spans — turned out to be vacuous once the residual was measured to be monotone decreasing in t, because then t = 1 is the extremum for every possible outcome, a perfect 0.0000 included. It was retired for having no power in either direction, not for its verdict."` | — |
| `/one_cell_volume_witness/one_cell_gap_abs` | *absent* | `0.0312` | — |
| `/one_cell_volume_witness/one_cell_two_wall_vs_one_wall_x` | *absent* | `4.317` | — |
| `/one_cell_volume_witness/passed` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/0/identification/argmin_t_cells` | *absent* | `1` | — |
| `/one_cell_volume_witness/rows/0/identification/gap_at_t` | *absent* | `0.0312` | — |
| `/one_cell_volume_witness/rows/0/identification/gap_at_t_minus_1` | *absent* | `0.1347` | — |
| `/one_cell_volume_witness/rows/0/identification/gap_at_t_plus_1` | *absent* | `0.04` | — |
| `/one_cell_volume_witness/rows/0/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/0/identification/margin_vs_runner_up_x` | *absent* | `1.282` | — |
| `/one_cell_volume_witness/rows/0/iris_wall_nodes` | *absent* | `[191, 192]` | 0–1 |
| `/one_cell_volume_witness/rows/0/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/0/max_gap_abs` | *absent* | `0.0312` | — |
| `/one_cell_volume_witness/rows/0/oracle_s11` | *absent* | array[29], SHA256 `bd0c9992cfbe` | 0–28 |
| `/one_cell_volume_witness/rows/0/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/0/realized_thickness_cells` | *absent* | `1` | — |
| `/one_cell_volume_witness/rows/0/s11` | *absent* | array[29], SHA256 `775b78889ddd` | 0–28 |
| `/one_cell_volume_witness/rows/0/t_cells` | *absent* | `1` | — |
| `/one_cell_volume_witness/rows/0/t_mm` | *absent* | `0.762` | — |
| `/one_cell_volume_witness/rows/0/wall_s` | *absent* | `26.1` | — |
| `/one_cell_volume_witness/rows/1/identification/argmin_t_cells` | *absent* | `2` | — |
| `/one_cell_volume_witness/rows/1/identification/gap_at_t` | *absent* | `0.0246` | — |
| `/one_cell_volume_witness/rows/1/identification/gap_at_t_minus_1` | *absent* | `0.0942` | — |
| `/one_cell_volume_witness/rows/1/identification/gap_at_t_plus_1` | *absent* | `0.0329` | — |
| `/one_cell_volume_witness/rows/1/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/1/identification/margin_vs_runner_up_x` | *absent* | `1.337` | — |
| `/one_cell_volume_witness/rows/1/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/one_cell_volume_witness/rows/1/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/1/max_gap_abs` | *absent* | `0.0246` | — |
| `/one_cell_volume_witness/rows/1/oracle_s11` | *absent* | array[29], SHA256 `bed1e9c4e3a2` | 0–28 |
| `/one_cell_volume_witness/rows/1/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/1/realized_thickness_cells` | *absent* | `2` | — |
| `/one_cell_volume_witness/rows/1/s11` | *absent* | array[29], SHA256 `40f1fb558306` | 0–28 |
| `/one_cell_volume_witness/rows/1/t_cells` | *absent* | `2` | — |
| `/one_cell_volume_witness/rows/1/t_mm` | *absent* | `1.524` | — |
| `/one_cell_volume_witness/rows/1/wall_s` | *absent* | `27.0` | — |
| `/one_cell_volume_witness/rows/2/identification/argmin_t_cells` | *absent* | `3` | — |
| `/one_cell_volume_witness/rows/2/identification/gap_at_t` | *absent* | `0.0207` | — |
| `/one_cell_volume_witness/rows/2/identification/gap_at_t_minus_1` | *absent* | `0.0767` | — |
| `/one_cell_volume_witness/rows/2/identification/gap_at_t_plus_1` | *absent* | `0.0277` | — |
| `/one_cell_volume_witness/rows/2/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/2/identification/margin_vs_runner_up_x` | *absent* | `1.338` | — |
| `/one_cell_volume_witness/rows/2/iris_wall_nodes` | *absent* | `[190, 193]` | 0–1 |
| `/one_cell_volume_witness/rows/2/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/2/max_gap_abs` | *absent* | `0.0207` | — |
| `/one_cell_volume_witness/rows/2/oracle_s11` | *absent* | array[29], SHA256 `57665830614d` | 0–28 |
| `/one_cell_volume_witness/rows/2/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/2/realized_thickness_cells` | *absent* | `3` | — |
| `/one_cell_volume_witness/rows/2/s11` | *absent* | array[29], SHA256 `396803b8a62e` | 0–28 |
| `/one_cell_volume_witness/rows/2/t_cells` | *absent* | `3` | — |
| `/one_cell_volume_witness/rows/2/t_mm` | *absent* | `2.286` | — |
| `/one_cell_volume_witness/rows/2/wall_s` | *absent* | `22.5` | — |
| `/one_cell_volume_witness/rows/3/identification/argmin_t_cells` | *absent* | `4` | — |
| `/one_cell_volume_witness/rows/3/identification/gap_at_t` | *absent* | `0.0179` | — |
| `/one_cell_volume_witness/rows/3/identification/gap_at_t_minus_1` | *absent* | `0.0647` | — |
| `/one_cell_volume_witness/rows/3/identification/gap_at_t_plus_1` | *absent* | `0.0236` | — |
| `/one_cell_volume_witness/rows/3/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/3/identification/margin_vs_runner_up_x` | *absent* | `1.318` | — |
| `/one_cell_volume_witness/rows/3/iris_wall_nodes` | *absent* | `[189, 193]` | 0–1 |
| `/one_cell_volume_witness/rows/3/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/3/max_gap_abs` | *absent* | `0.0179` | — |
| `/one_cell_volume_witness/rows/3/oracle_s11` | *absent* | array[29], SHA256 `226f8c0145f4` | 0–28 |
| `/one_cell_volume_witness/rows/3/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/3/realized_thickness_cells` | *absent* | `4` | — |
| `/one_cell_volume_witness/rows/3/s11` | *absent* | array[29], SHA256 `43cc528859fa` | 0–28 |
| `/one_cell_volume_witness/rows/3/t_cells` | *absent* | `4` | — |
| `/one_cell_volume_witness/rows/3/t_mm` | *absent* | `3.048` | — |
| `/one_cell_volume_witness/rows/3/wall_s` | *absent* | `24.6` | — |
| `/one_cell_volume_witness/rows/4/identification/argmin_t_cells` | *absent* | `5` | — |
| `/one_cell_volume_witness/rows/4/identification/gap_at_t` | *absent* | `0.0157` | — |
| `/one_cell_volume_witness/rows/4/identification/gap_at_t_minus_1` | *absent* | `0.0555` | — |
| `/one_cell_volume_witness/rows/4/identification/gap_at_t_plus_1` | *absent* | `0.0202` | — |
| `/one_cell_volume_witness/rows/4/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/4/identification/margin_vs_runner_up_x` | *absent* | `1.287` | — |
| `/one_cell_volume_witness/rows/4/iris_wall_nodes` | *absent* | `[189, 194]` | 0–1 |
| `/one_cell_volume_witness/rows/4/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/4/max_gap_abs` | *absent* | `0.0157` | — |
| `/one_cell_volume_witness/rows/4/oracle_s11` | *absent* | array[29], SHA256 `c75b34c3a600` | 0–28 |
| `/one_cell_volume_witness/rows/4/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/4/realized_thickness_cells` | *absent* | `5` | — |
| `/one_cell_volume_witness/rows/4/s11` | *absent* | array[29], SHA256 `27599ab95c1f` | 0–28 |
| `/one_cell_volume_witness/rows/4/t_cells` | *absent* | `5` | — |
| `/one_cell_volume_witness/rows/4/t_mm` | *absent* | `3.81` | — |
| `/one_cell_volume_witness/rows/4/wall_s` | *absent* | `25.2` | — |
| `/one_cell_volume_witness/rows/5/identification/argmin_t_cells` | *absent* | `6` | — |
| `/one_cell_volume_witness/rows/5/identification/gap_at_t` | *absent* | `0.0139` | — |
| `/one_cell_volume_witness/rows/5/identification/gap_at_t_minus_1` | *absent* | `0.0481` | — |
| `/one_cell_volume_witness/rows/5/identification/gap_at_t_plus_1` | *absent* | `0.0173` | — |
| `/one_cell_volume_witness/rows/5/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/5/identification/margin_vs_runner_up_x` | *absent* | `1.245` | — |
| `/one_cell_volume_witness/rows/5/iris_wall_nodes` | *absent* | `[188, 194]` | 0–1 |
| `/one_cell_volume_witness/rows/5/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/5/max_gap_abs` | *absent* | `0.0139` | — |
| `/one_cell_volume_witness/rows/5/oracle_s11` | *absent* | array[29], SHA256 `0edc599036e2` | 0–28 |
| `/one_cell_volume_witness/rows/5/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/5/realized_thickness_cells` | *absent* | `6` | — |
| `/one_cell_volume_witness/rows/5/s11` | *absent* | array[29], SHA256 `b121a5cbbd7c` | 0–28 |
| `/one_cell_volume_witness/rows/5/t_cells` | *absent* | `6` | — |
| `/one_cell_volume_witness/rows/5/t_mm` | *absent* | `4.572` | — |
| `/one_cell_volume_witness/rows/5/wall_s` | *absent* | `27.0` | — |
| `/one_cell_volume_witness/rows/6/identification/argmin_t_cells` | *absent* | `8` | — |
| `/one_cell_volume_witness/rows/6/identification/gap_at_t` | *absent* | `0.0109` | — |
| `/one_cell_volume_witness/rows/6/identification/gap_at_t_minus_1` | *absent* | `0.0367` | — |
| `/one_cell_volume_witness/rows/6/identification/gap_at_t_plus_1` | *absent* | `0.0131` | — |
| `/one_cell_volume_witness/rows/6/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/6/identification/margin_vs_runner_up_x` | *absent* | `1.202` | — |
| `/one_cell_volume_witness/rows/6/iris_wall_nodes` | *absent* | `[187, 195]` | 0–1 |
| `/one_cell_volume_witness/rows/6/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/6/max_gap_abs` | *absent* | `0.0109` | — |
| `/one_cell_volume_witness/rows/6/oracle_s11` | *absent* | array[29], SHA256 `278c32a67858` | 0–28 |
| `/one_cell_volume_witness/rows/6/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/6/realized_thickness_cells` | *absent* | `8` | — |
| `/one_cell_volume_witness/rows/6/s11` | *absent* | array[29], SHA256 `6f7db1192fff` | 0–28 |
| `/one_cell_volume_witness/rows/6/t_cells` | *absent* | `8` | — |
| `/one_cell_volume_witness/rows/6/t_mm` | *absent* | `6.096` | — |
| `/one_cell_volume_witness/rows/6/wall_s` | *absent* | `25.8` | — |
| `/provenance/modal_fence_retraction_2026_07_28` | `"An earlier revision of this case FENCED normalize=True modal extraction, citing measured max column power 1.112 (later 1.15374 at driven port 0 coarse / 1.16407 at driven port 1 fine, with a second per-frequency advisory referencing issue #337). Those runs carried the d + 2*dx electrical aperture and a 0.5*lambda_g absorber. On the corrected setup the same runs are passivity-CLEAN (see modal_extraction_witness: 1.0207 / 1.0101 / 1.0144 / 1.0013, zero extractor warnings), so the fence is RETRACTED: the non-passivity was a setup symptom, not an extractor property. Recorded so the withdrawn claim stays auditable."` | `"An earlier revision of this case FENCED normalize=True modal extraction, citing measured max column power 1.112 (later 1.15374 at driven port 0 coarse / 1.16407 at driven port 1 fine, with a second per-frequency advisory referencing issue #337). Those runs carried the d + 2*dx electrical aperture and a 0.5*lambda_g absorber. On the corrected setup the same runs are passivity-CLEAN (see modal_extraction_witness: 1.0200 / 1.0099 / 1.0150 / 1.0012, zero extractor warnings), so the fence is RETRACTED: the non-passivity was a setup symptom, not an extractor property. Recorded so the withdrawn claim stays auditable."` | — |
| `/provenance/no_preflight_note` | `"compute_waveguide_s_matrix runs its own extractor passivity self-check (warnings are part of this record) but no sim.preflight(); operating-point guarantees are the raster asserts in run_point."` | `"compute_waveguide_s_matrix runs its own extractor passivity self-check (warnings are part of this record) but no sim.preflight(); operating-point guarantees are the realized-geometry asserts in run_point, which read realized_pec_edge_masks through validation/crossval/_wr90_iris_realized.py."` | — |
| `/raw_extraction_record/0/aperture_cells` | `23` | *absent* | — |
| `/raw_extraction_record/0/aperture_wall_nodes` | *absent* | `[3, 27]` | 0–1 |
| `/raw_extraction_record/0/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/raw_extraction_record/0/max_colpow` | `1.0007` | `0.9996` | — |
| `/raw_extraction_record/0/max_gap_abs` | `0.0535` | `0.0244` | — |
| `/raw_extraction_record/0/realized_aperture_cells` | *absent* | `24` | — |
| `/raw_extraction_record/0/realized_thickness_cells` | *absent* | `2` | — |
| `/raw_extraction_record/0/s11` | array[29], SHA256 `a69794f8f943` | array[29], SHA256 `6359557acac1` | 0–28 |
| `/raw_extraction_record/0/s21` | array[29], SHA256 `ecaa082dd176` | array[29], SHA256 `07ed0cb8427e` | 0–28 |
| `/raw_extraction_record/0/t_mm` | *absent* | `1.524` | — |
| `/raw_extraction_record/0/thickness_cells` | `2` | *absent* | — |
| `/raw_extraction_record/0/wall_s` | `30.4` | `10.3` | — |
| `/raw_extraction_record/1/aperture_cells` | `15` | *absent* | — |
| `/raw_extraction_record/1/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/raw_extraction_record/1/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/raw_extraction_record/1/max_colpow` | `1.0004` | `1.0007` | — |
| `/raw_extraction_record/1/max_gap_abs` | `0.0543` | `0.0254` | — |
| `/raw_extraction_record/1/realized_aperture_cells` | *absent* | `16` | — |
| `/raw_extraction_record/1/realized_thickness_cells` | *absent* | `2` | — |
| `/raw_extraction_record/1/s11` | array[29], SHA256 `a795ed721597` | array[29], SHA256 `b9fd3bc6fb7e` | 0–28 |
| `/raw_extraction_record/1/s21` | array[29], SHA256 `277fe667de81` | array[29], SHA256 `05a5ae7528be` | 0–28 |
| `/raw_extraction_record/1/t_mm` | *absent* | `1.524` | — |
| `/raw_extraction_record/1/thickness_cells` | `2` | *absent* | — |
| `/raw_extraction_record/1/wall_s` | `30.8` | `9.9` | — |
| `/raw_extraction_record/2/aperture_cells` | `9` | *absent* | — |
| `/raw_extraction_record/2/aperture_wall_nodes` | *absent* | `[10, 20]` | 0–1 |
| `/raw_extraction_record/2/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/raw_extraction_record/2/max_colpow` | `1.0001` | `1.0016` | — |
| `/raw_extraction_record/2/max_gap_abs` | `0.0205` | `0.0088` | — |
| `/raw_extraction_record/2/realized_aperture_cells` | *absent* | `10` | — |
| `/raw_extraction_record/2/realized_thickness_cells` | *absent* | `2` | — |
| `/raw_extraction_record/2/s11` | array[29], SHA256 `04878a5cf129` | array[29], SHA256 `2e4f41e8ab2f` | 0–28 |
| `/raw_extraction_record/2/s21` | array[29], SHA256 `a2d27b295880` | array[29], SHA256 `7f1a158afbae` | 0–28 |
| `/raw_extraction_record/2/t_mm` | *absent* | `1.524` | — |
| `/raw_extraction_record/2/thickness_cells` | `2` | *absent* | — |
| `/raw_extraction_record/2/wall_s` | `30.1` | `12.1` | — |
| `/schema_version` | `1` | `2` | — |
| `/truncation_witness/3/shift_abs` | `1e-05` | `0.0` | — |

## validation/crossval/_05_patch_results/control_witness_2b/control_baseline_d990e18c.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/cases/0/captured/0/call` | *absent* | `"run"` | — |
| `/cases/0/captured/0/conductor_free` | *absent* | `true` | — |
| `/cases/0/captured/0/eps_r/dtype` | *absent* | `"float32"` | — |
| `/cases/0/captured/0/eps_r/max` | *absent* | `12.0` | — |
| `/cases/0/captured/0/eps_r/min` | *absent* | `1.0` | — |
| `/cases/0/captured/0/eps_r/sha256` | *absent* | `"a693e67131469a9dd556f709c9712ec6acc7a5d3ee67726f43be593c8d5a6f4c"` | — |
| `/cases/0/captured/0/eps_r/shape` | *absent* | `[181, 181, 1]` | 0–2 |
| `/cases/0/captured/0/grid_shape` | *absent* | `[181, 181, 1]` | 0–2 |
| `/cases/0/captured/0/n_pec_sheets` | *absent* | `0` | — |
| `/cases/0/captured/0/n_pec_wires` | *absent* | `0` | — |
| `/cases/0/captured/0/pec_cells` | *absent* | `0` | — |
| `/cases/0/captured/0/pec_edges` | *absent* | `0` | — |
| `/cases/0/captured/0/sigma/dtype` | *absent* | `"float32"` | — |
| `/cases/0/captured/0/sigma/max` | *absent* | `0.0` | — |
| `/cases/0/captured/0/sigma/min` | *absent* | `0.0` | — |
| `/cases/0/captured/0/sigma/sha256` | *absent* | `"81205f6f74a3487c5c5afc32af6a3c4008706ce3afc801f7ff241474db1715e7"` | — |
| `/cases/0/captured/0/sigma/shape` | *absent* | `[181, 181, 1]` | 0–2 |
| `/cases/0/case` | *absent* | `"01_waveguide_bend"` | — |
| `/cases/0/error` | *absent* | `null` | — |
| `/cases/1/captured/0/call` | *absent* | `"run"` | — |
| `/cases/1/captured/0/conductor_free` | *absent* | `true` | — |
| `/cases/1/captured/0/eps_r/dtype` | *absent* | `"float32"` | — |
| `/cases/1/captured/0/eps_r/max` | *absent* | `11.5600004196167` | — |
| `/cases/1/captured/0/eps_r/min` | *absent* | `1.0` | — |
| `/cases/1/captured/0/eps_r/sha256` | *absent* | `"40dbec5bbf922728f3b1add478f8b046f82d27c1c5fda11e7c54975901caccb4"` | — |
| `/cases/1/captured/0/eps_r/shape` | *absent* | `[162, 162, 1]` | 0–2 |
| `/cases/1/captured/0/grid_shape` | *absent* | `[162, 162, 1]` | 0–2 |
| `/cases/1/captured/0/n_pec_sheets` | *absent* | `0` | — |
| `/cases/1/captured/0/n_pec_wires` | *absent* | `0` | — |
| `/cases/1/captured/0/pec_cells` | *absent* | `0` | — |
| `/cases/1/captured/0/pec_edges` | *absent* | `0` | — |
| `/cases/1/captured/0/sigma/dtype` | *absent* | `"float32"` | — |
| `/cases/1/captured/0/sigma/max` | *absent* | `0.0` | — |
| `/cases/1/captured/0/sigma/min` | *absent* | `0.0` | — |
| `/cases/1/captured/0/sigma/sha256` | *absent* | `"e006234d0697ae9cbb706dc1b14e87ef89fe2d64a48360fc76895adfd68402a1"` | — |
| `/cases/1/captured/0/sigma/shape` | *absent* | `[162, 162, 1]` | 0–2 |
| `/cases/1/case` | *absent* | `"02_ring_resonator"` | — |
| `/cases/1/error` | *absent* | `null` | — |
| `/cases/2/captured/0/call` | *absent* | `"run"` | — |
| `/cases/2/captured/0/conductor_free` | *absent* | `true` | — |
| `/cases/2/captured/0/eps_r/dtype` | *absent* | `"float32"` | — |
| `/cases/2/captured/0/eps_r/max` | *absent* | `12.0` | — |
| `/cases/2/captured/0/eps_r/min` | *absent* | `1.0` | — |
| `/cases/2/captured/0/eps_r/sha256` | *absent* | `"f12b4af1ea61604e16dcf436aa53b69391644484ec84cb58e250c68c7d0d6a0e"` | — |
| `/cases/2/captured/0/eps_r/shape` | *absent* | `[201, 131, 1]` | 0–2 |
| `/cases/2/captured/0/grid_shape` | *absent* | `[201, 131, 1]` | 0–2 |
| `/cases/2/captured/0/n_pec_sheets` | *absent* | `0` | — |
| `/cases/2/captured/0/n_pec_wires` | *absent* | `0` | — |
| `/cases/2/captured/0/pec_cells` | *absent* | `0` | — |
| `/cases/2/captured/0/pec_edges` | *absent* | `0` | — |
| `/cases/2/captured/0/sigma/dtype` | *absent* | `"float32"` | — |
| `/cases/2/captured/0/sigma/max` | *absent* | `0.0` | — |
| `/cases/2/captured/0/sigma/min` | *absent* | `0.0` | — |
| `/cases/2/captured/0/sigma/sha256` | *absent* | `"309d6171b688ed096719b571b776d2235c8c136f0775575423560be94a4da2b1"` | — |
| `/cases/2/captured/0/sigma/shape` | *absent* | `[201, 131, 1]` | 0–2 |
| `/cases/2/case` | *absent* | `"03_straight_waveguide_flux"` | — |
| `/cases/2/error` | *absent* | `null` | — |
| `/repo` | *absent* | `"/root/workspace/byungkwan-workspace/research/rfx-baseline-d990e18c"` | — |
| `/repo_commit` | *absent* | `"d990e18ce0870ac7a893a86627e7232d1cf92c7b"` | — |
| `/repo_dirty` | *absent* | `false` | — |
| `/rfx_module` | *absent* | `"/root/workspace/byungkwan-workspace/research/rfx-baseline-d990e18c/rfx/__init__.py"` | — |
| `/rfx_version` | *absent* | `"1.8.0"` | — |

## validation/crossval/_05_patch_results/control_witness_2b/control_merged_b4961b56.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/cases/0/captured/0/call` | *absent* | `"run"` | — |
| `/cases/0/captured/0/conductor_free` | *absent* | `true` | — |
| `/cases/0/captured/0/eps_r/dtype` | *absent* | `"float32"` | — |
| `/cases/0/captured/0/eps_r/max` | *absent* | `12.0` | — |
| `/cases/0/captured/0/eps_r/min` | *absent* | `1.0` | — |
| `/cases/0/captured/0/eps_r/sha256` | *absent* | `"a693e67131469a9dd556f709c9712ec6acc7a5d3ee67726f43be593c8d5a6f4c"` | — |
| `/cases/0/captured/0/eps_r/shape` | *absent* | `[181, 181, 1]` | 0–2 |
| `/cases/0/captured/0/grid_shape` | *absent* | `[181, 181, 1]` | 0–2 |
| `/cases/0/captured/0/n_pec_sheets` | *absent* | `0` | — |
| `/cases/0/captured/0/n_pec_wires` | *absent* | `0` | — |
| `/cases/0/captured/0/pec_cells` | *absent* | `0` | — |
| `/cases/0/captured/0/pec_edges` | *absent* | `0` | — |
| `/cases/0/captured/0/sigma/dtype` | *absent* | `"float32"` | — |
| `/cases/0/captured/0/sigma/max` | *absent* | `0.0` | — |
| `/cases/0/captured/0/sigma/min` | *absent* | `0.0` | — |
| `/cases/0/captured/0/sigma/sha256` | *absent* | `"81205f6f74a3487c5c5afc32af6a3c4008706ce3afc801f7ff241474db1715e7"` | — |
| `/cases/0/captured/0/sigma/shape` | *absent* | `[181, 181, 1]` | 0–2 |
| `/cases/0/case` | *absent* | `"01_waveguide_bend"` | — |
| `/cases/0/error` | *absent* | `null` | — |
| `/cases/1/captured/0/call` | *absent* | `"run"` | — |
| `/cases/1/captured/0/conductor_free` | *absent* | `true` | — |
| `/cases/1/captured/0/eps_r/dtype` | *absent* | `"float32"` | — |
| `/cases/1/captured/0/eps_r/max` | *absent* | `11.5600004196167` | — |
| `/cases/1/captured/0/eps_r/min` | *absent* | `1.0` | — |
| `/cases/1/captured/0/eps_r/sha256` | *absent* | `"40dbec5bbf922728f3b1add478f8b046f82d27c1c5fda11e7c54975901caccb4"` | — |
| `/cases/1/captured/0/eps_r/shape` | *absent* | `[162, 162, 1]` | 0–2 |
| `/cases/1/captured/0/grid_shape` | *absent* | `[162, 162, 1]` | 0–2 |
| `/cases/1/captured/0/n_pec_sheets` | *absent* | `0` | — |
| `/cases/1/captured/0/n_pec_wires` | *absent* | `0` | — |
| `/cases/1/captured/0/pec_cells` | *absent* | `0` | — |
| `/cases/1/captured/0/pec_edges` | *absent* | `0` | — |
| `/cases/1/captured/0/sigma/dtype` | *absent* | `"float32"` | — |
| `/cases/1/captured/0/sigma/max` | *absent* | `0.0` | — |
| `/cases/1/captured/0/sigma/min` | *absent* | `0.0` | — |
| `/cases/1/captured/0/sigma/sha256` | *absent* | `"e006234d0697ae9cbb706dc1b14e87ef89fe2d64a48360fc76895adfd68402a1"` | — |
| `/cases/1/captured/0/sigma/shape` | *absent* | `[162, 162, 1]` | 0–2 |
| `/cases/1/case` | *absent* | `"02_ring_resonator"` | — |
| `/cases/1/error` | *absent* | `null` | — |
| `/cases/2/captured/0/call` | *absent* | `"run"` | — |
| `/cases/2/captured/0/conductor_free` | *absent* | `true` | — |
| `/cases/2/captured/0/eps_r/dtype` | *absent* | `"float32"` | — |
| `/cases/2/captured/0/eps_r/max` | *absent* | `12.0` | — |
| `/cases/2/captured/0/eps_r/min` | *absent* | `1.0` | — |
| `/cases/2/captured/0/eps_r/sha256` | *absent* | `"f12b4af1ea61604e16dcf436aa53b69391644484ec84cb58e250c68c7d0d6a0e"` | — |
| `/cases/2/captured/0/eps_r/shape` | *absent* | `[201, 131, 1]` | 0–2 |
| `/cases/2/captured/0/grid_shape` | *absent* | `[201, 131, 1]` | 0–2 |
| `/cases/2/captured/0/n_pec_sheets` | *absent* | `0` | — |
| `/cases/2/captured/0/n_pec_wires` | *absent* | `0` | — |
| `/cases/2/captured/0/pec_cells` | *absent* | `0` | — |
| `/cases/2/captured/0/pec_edges` | *absent* | `0` | — |
| `/cases/2/captured/0/sigma/dtype` | *absent* | `"float32"` | — |
| `/cases/2/captured/0/sigma/max` | *absent* | `0.0` | — |
| `/cases/2/captured/0/sigma/min` | *absent* | `0.0` | — |
| `/cases/2/captured/0/sigma/sha256` | *absent* | `"309d6171b688ed096719b571b776d2235c8c136f0775575423560be94a4da2b1"` | — |
| `/cases/2/captured/0/sigma/shape` | *absent* | `[201, 131, 1]` | 0–2 |
| `/cases/2/case` | *absent* | `"03_straight_waveguide_flux"` | — |
| `/cases/2/error` | *absent* | `null` | — |
| `/repo` | *absent* | `"/root/workspace/byungkwan-workspace/research/rfx-931-XA-crossval-a"` | — |
| `/repo_commit` | *absent* | `"b4961b56059d051e83efc6700be88424444f8871"` | — |
| `/repo_dirty` | *absent* | `false` | — |
| `/rfx_module` | *absent* | `"/root/workspace/byungkwan-workspace/research/rfx-931-XA-crossval-a/rfx/__init__.py"` | — |
| `/rfx_version` | *absent* | `"1.8.0"` | — |

## validation/crossval/_05_patch_results/cv05_run_openems_369367259142.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/analytic_resonance_hz` | *absent* | `2423509824.7011304` | — |
| `/claim_scope` | *absent* | `"diagnostic-reporter (demoted from claims-bearing 2026-07-15): coarse-mesh probe-fed patch resonance integration check, reported not gated; the 20% openEMS smoke bound passes over two different substrate geometries (#325). Patch-accuracy evidence is delegated to committed tests (manifest gate_paths). Since #931 both conductors are declared PEC SHEETS on the substrate floor and top node planes and the six-cell fine substrate block is preserved by the z-profile builder, so the realized cavity is the declared 1.5 mm laminate; realized_stack below carries the measured planes and footprints from realized_pec_edge_masks."` | — |
| `/declared_modes_hz/TM010` | *absent* | `1914913485.2015615` | — |
| `/declared_modes_hz/TM100` | *absent* | `2423509824.7011304` | — |
| `/declared_modes_hz/TM110` | *absent* | `3088736558.2434664` | — |
| `/mode_identification_tol` | *absent* | `0.1249878021081674` | — |
| `/openems_freqs_hz` | *absent* | array[201], SHA256 `c07faa067817` | 0–200 |
| `/openems_harminv_hz` | *absent* | `2190000000.0` | — |
| `/openems_mode_assignment` | *absent* | array[7], SHA256 `0982e15e829c` | 0–6 |
| `/openems_mode_id_gated` | *absent* | `false` | — |
| `/openems_mode_id_ok` | *absent* | `false` | — |
| `/openems_mode_id_reasons` | *absent* | `["declared member TM010 is claimed by 2 measured modes ([1.7707, 1.9304] GHz) -- identification is ambiguous, refuse to name a resonance", "declared member TM100 is claimed by 3 measured modes ([2.1942, 2.2589, 2.4762] GHz) -- identification is ambiguous, refuse to name a resonance", "declared member TM110 is claimed by 2 measured modes ([3.2115, 3.3174] GHz) -- identification is ambiguous, refuse to name a resonance"]` | 0–2 |
| `/openems_modes_hz` | *absent* | array[7], SHA256 `c3112cfb3933` | 0–6 |
| `/openems_s11` | *absent* | array[201], SHA256 `0742b0777c30` | 0–200 |
| `/openems_s11_dip_hz` | *absent* | `2190000000.0` | — |
| `/openems_s11_min_db` | *absent* | `-0.533575290640742` | — |
| `/realized_stack/cavity_eps_r` | *absent* | array[7], SHA256 `c6fd6f01b1f4` | 0–6 |
| `/realized_stack/cavity_node_to_node_mm` | *absent* | `1.5000000000000013` | — |
| `/realized_stack/cavity_physical_mm` | *absent* | `1.5` | — |
| `/realized_stack/ground/declared_x_mm` | *absent* | `60.0` | — |
| `/realized_stack/ground/declared_y_mm` | *absent* | `55.0` | — |
| `/realized_stack/ground/declared_z_mm` | *absent* | `11.999999999999993` | — |
| `/realized_stack/ground/k` | *absent* | `22` | — |
| `/realized_stack/ground/realized_x_mm` | *absent* | `60.0` | — |
| `/realized_stack/ground/realized_y_mm` | *absent* | `55.0` | — |
| `/realized_stack/ground/x_edge_cells` | *absent* | `60` | — |
| `/realized_stack/ground/x_index_range` | *absent* | `[18, 77]` | 0–1 |
| `/realized_stack/ground/y_edge_cells` | *absent* | `55` | — |
| `/realized_stack/ground/y_index_range` | *absent* | `[18, 72]` | 0–1 |
| `/realized_stack/patch/declared_x_mm` | *absent* | `29.5` | — |
| `/realized_stack/patch/declared_y_mm` | *absent* | `38.0` | — |
| `/realized_stack/patch/declared_z_mm` | *absent* | `13.499999999999995` | — |
| `/realized_stack/patch/k` | *absent* | `28` | — |
| `/realized_stack/patch/realized_x_mm` | *absent* | `28.0` | — |
| `/realized_stack/patch/realized_y_mm` | *absent* | `37.0` | — |
| `/realized_stack/patch/x_edge_cells` | *absent* | `28` | — |
| `/realized_stack/patch/x_index_range` | *absent* | `[34, 61]` | 0–1 |
| `/realized_stack/patch/y_edge_cells` | *absent* | `37` | — |
| `/realized_stack/patch/y_index_range` | *absent* | `[27, 63]` | 0–1 |
| `/realized_stack/sheet_plane_delta` | *absent* | `0` | — |
| `/realized_stack/substrate_cells_between` | *absent* | `6` | — |
| `/realized_stack_with_port/cavity_eps_r` | *absent* | array[7], SHA256 `c6fd6f01b1f4` | 0–6 |
| `/realized_stack_with_port/cavity_node_to_node_mm` | *absent* | `1.5000000000000013` | — |
| `/realized_stack_with_port/cavity_physical_mm` | *absent* | `1.5` | — |
| `/realized_stack_with_port/ground/declared_x_mm` | *absent* | `60.0` | — |
| `/realized_stack_with_port/ground/declared_y_mm` | *absent* | `55.0` | — |
| `/realized_stack_with_port/ground/declared_z_mm` | *absent* | `11.999999999999993` | — |
| `/realized_stack_with_port/ground/k` | *absent* | `22` | — |
| `/realized_stack_with_port/ground/realized_x_mm` | *absent* | `60.0` | — |
| `/realized_stack_with_port/ground/realized_y_mm` | *absent* | `55.0` | — |
| `/realized_stack_with_port/ground/x_edge_cells` | *absent* | `60` | — |
| `/realized_stack_with_port/ground/x_index_range` | *absent* | `[18, 77]` | 0–1 |
| `/realized_stack_with_port/ground/y_edge_cells` | *absent* | `55` | — |
| `/realized_stack_with_port/ground/y_index_range` | *absent* | `[18, 72]` | 0–1 |
| `/realized_stack_with_port/patch/declared_x_mm` | *absent* | `29.5` | — |
| `/realized_stack_with_port/patch/declared_y_mm` | *absent* | `38.0` | — |
| `/realized_stack_with_port/patch/declared_z_mm` | *absent* | `13.499999999999995` | — |
| `/realized_stack_with_port/patch/k` | *absent* | `28` | — |
| `/realized_stack_with_port/patch/realized_x_mm` | *absent* | `28.0` | — |
| `/realized_stack_with_port/patch/realized_y_mm` | *absent* | `37.0` | — |
| `/realized_stack_with_port/patch/x_edge_cells` | *absent* | `28` | — |
| `/realized_stack_with_port/patch/x_index_range` | *absent* | `[34, 61]` | 0–1 |
| `/realized_stack_with_port/patch/y_edge_cells` | *absent* | `37` | — |
| `/realized_stack_with_port/patch/y_index_range` | *absent* | `[27, 63]` | 0–1 |
| `/realized_stack_with_port/sheet_plane_delta` | *absent* | `0` | — |
| `/realized_stack_with_port/substrate_cells_between` | *absent* | `6` | — |
| `/rfx_freqs_hz` | *absent* | array[101], SHA256 `db01591c4bc5` | 0–100 |
| `/rfx_harminv_hz` | *absent* | `2446496652.4277554` | — |
| `/rfx_internal_pct` | *absent* | `1.3694417909379908` | — |
| `/rfx_mode_assignment` | *absent* | `[[1900346670.4241853, "TM010", -0.007607035456143829], [2446496652.4277554, "TM100", 0.009484932758405451], [3186911011.360049, "TM110", 0.0317846638149073], [3758955223.270803, null, null]]` | 0–3 |
| `/rfx_mode_id_ok` | *absent* | `true` | — |
| `/rfx_modes_hz` | *absent* | `[1900346670.4241853, 2446496652.4277554, 3186911011.360049, 3758955223.270803]` | 0–3 |
| `/rfx_s11` | *absent* | array[101], SHA256 `b264aca4a46e` | 0–100 |
| `/rfx_s11_dip_hz` | *absent* | `2480000000.0` | — |
| `/rfx_s11_max_abs` | *absent* | `0.7756921648979187` | — |
| `/rfx_s11_min_db` | *absent* | `-9.62222671508789` | — |
| `/rfx_s11_passive` | *absent* | `true` | — |
| `/rfx_vs_analytic_pct` | *absent* | `0.9484932758405353` | — |
| `/rfx_vs_openems_harminv_pct` | *absent* | `11.712175909943166` | — |
| `/sheet_plane_delta` | *absent* | `0` | — |
| `/status` | *absent* | `"passed"` | — |

## validation/crossval/_06b_msl_notch_results/cv06b_build_falsifiers_summary.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/criterion_A_baseline/bw_ratio` | `0.9683988760970414` | `0.9991318709167973` | — |
| `/criterion_A_baseline/err_pct` | `1.4530155863890297` | `2.16493899357691` | — |
| `/criterion_A_baseline/f_notch_analytic_hz` | `3678954217.3889155` | `3678954217.388916` | — |
| `/criterion_A_baseline/f_notch_bin_hz` | `3627272704.0` | `3754545408.0` | — |
| `/criterion_A_baseline/f_notch_refined_hz` | `3625498439.194138` | `3758601331.797011` | — |
| `/criterion_A_baseline/notch_depth_db` | `-43.49096352273952` | `-39.44347175145582` | — |
| `/criterion_A_baseline/solve_s` | `312.70808577537537` | `315.2233588695526` | — |
| `/criterion_A_baseline/sub_bin_shift_bins` | `-0.027881365271800063` | `0.06373557243745055` | — |
| `/criterion_A_baseline/witness_bins` | `0.3175061099379293` | `0.4468865583740074` | — |
| `/criterion_A_baseline/z0_median_ohm` | `46.4818172454834` | `48.19204521179199` | — |
| `/stub_1cell/bin_argmin_delta_pct` | `0.0` | `1.6949251929249807` | — |
| `/stub_1cell/refined_delta_pct` | `0.1447171958022862` | `0.8227703133018797` | — |
| `/stub_1cell/solve_s` | `179.45163202285767` | `180.48277926445007` | — |
| `/stub_1cell/true_shift_bins` | `0.30308192957038443` | `0.31420644578857854` | — |
| `/stub_1cell/true_shift_pct` | `0.5319817366899832` | `0.5319817366899701` | — |
| `/stub_1cell/visible` | `false` | `true` | — |
| `/stub_narrow/bw_frac` | `0.13629883213270969` | `0.1378026444047796` | — |
| `/stub_narrow/bw_ratio` | `0.6481963159149952` | `0.6553479954953043` | — |
| `/stub_narrow/err_pct` | `0.2080765529334878` | `6.438768666807615` | — |
| `/stub_narrow/gates/G1 notch freq vs analytic` | `true` | `false` | — |
| `/stub_narrow/notch_depth_db` | `-35.09758588046899` | `-28.258988020426564` | — |
| `/stub_narrow/solve_s` | `307.8088598251343` | `310.7757911682129` | — |
| `/verdict/all_ok` | `false` | `true` | — |
| `/verdict/criterion_B_sub_bin_visible` | `false` | `true` | — |

## validation/crossval/_06b_msl_notch_results/cv06b_falsifier_baseline.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/bin_hz` | `63636224.0` | `63636736.0` | — |
| `/bw_bins` | `11` | `13` | — |
| `/bw_frac` | `0.20362910527242928` | `0.21009145502515864` | — |
| `/bw_hi` | `3987166887.9623547` | `4162172232.9811854` | — |
| `/bw_lo` | `3248909884.6226635` | `3372522210.3244524` | — |
| `/bw_ratio` | `0.9683988760970414` | `0.9991318709167973` | — |
| `/depth_gate_blind_margin_db` | `-21.193619173049164` | `-21.5065288589727` | — |
| `/err_pct` | `1.4530155863890297` | `2.16493899357691` | — |
| `/err_pct_bin` | `1.4047881635666493` | `2.054692343106507` | — |
| `/f_notch_analytic` | `3678954217.3889155` | `3678954217.388916` | — |
| `/f_notch_bin` | `3627272704.0` | `3754545408.0` | — |
| `/f_notch_refined` | `3625498439.194138` | `3758601331.797011` | — |
| `/notch_depth_db` | `-43.49096352273952` | `-39.44347175145582` | — |
| `/re_z0` | array[100], SHA256 `37d3e0af6f2c` | array[100], SHA256 `52102d6307fb` | 0–99 |
| `/s21_mag` | array[100], SHA256 `a6822330f653` | array[100], SHA256 `d5f19eec944a` | 0–99 |
| `/solve_s` | `312.70808577537537` | `315.2233588695526` | — |
| `/sub_bin_shift` | `-0.027881365271800063` | `0.06373557243745055` | — |
| `/witness_bins` | `0.3175061099379293` | `0.4468865583740074` | — |
| `/worst_sampled_depth_db` | `-31.193619173049164` | `-31.5065288589727` | — |
| `/z0_median` | `46.4818172454834` | `48.19204521179199` | — |

## validation/crossval/_06b_msl_notch_results/cv06b_falsifier_stub_1cell.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/bw_bins` | `11` | `12` | — |
| `/bw_frac` | `0.20423661193042172` | `0.20978884241548154` | — |
| `/bw_hi` | `4006964652.10055` | `4183963537.2144976` | — |
| `/bw_lo` | `3265433562.070325` | `3388963266.9409556` | — |
| `/bw_ratio` | `0.9712879953319086` | `0.9976927362178946` | — |
| `/depth_gate_blind_margin_db` | `-21.20617212167638` | `-21.577729646243753` | — |
| `/err_pct` | `1.8326336145580286` | `2.460450896121476` | — |
| `/err_pct_bin` | `1.926521159534441` | `3.2352503564693165` | — |
| `/f_notch_bin` | `3627272704.0` | `3818182144.0` | — |
| `/f_notch_refined` | `3630745158.8691955` | `3789525987.750406` | — |
| `/notch_depth_db` | `-39.66168331327439` | `-31.794482496176716` | — |
| `/re_z0` | array[100], SHA256 `0084e4baccfc` | array[100], SHA256 `4076bde23e61` | 0–99 |
| `/s21_mag` | array[100], SHA256 `bdcfc0e308e1` | array[100], SHA256 `7718eca87c4b` | 0–99 |
| `/solve_s` | `179.45163202285767` | `180.48277926445007` | — |
| `/sub_bin_shift` | `0.054567267680675866` | `-0.45031201489255807` | — |
| `/witness_bins` | `0.4237993203339697` | `0.6252941337575424` | — |
| `/worst_sampled_depth_db` | `-31.20617212167638` | `-31.577729646243753` | — |
| `/z0_median` | `46.476009368896484` | `48.190359115600586` | — |

## validation/crossval/_06b_msl_notch_results/cv06b_falsifier_stub_narrow.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/bw_frac` | `0.13629883213270969` | `0.1378026444047796` | — |
| `/bw_hi` | `3928758306.569712` | `4184217052.4393773` | — |
| `/bw_lo` | `3426277770.1053743` | `3644604831.6091323` | — |
| `/bw_ratio` | `0.6481963159149952` | `0.6553479954953043` | — |
| `/depth_gate_blind_margin_db` | `-21.33868159383999` | `-21.8623216902829` | — |
| `/err_pct` | `0.2080765529334878` | `6.438768666807615` | — |
| `/err_pct_bin` | `0.3249486105203327` | `7.243916582365864` | — |
| `/f_notch_analytic` | `3678954217.3889155` | `3678954217.388916` | — |
| `/f_notch_bin` | `3690908928.0` | `3945454592.0` | — |
| `/f_notch_refined` | `3686609258.5084596` | `3915833568.804351` | — |
| `/gates/G1 notch freq vs analytic` | `true` | `false` | — |
| `/notch_depth_db` | `-35.09758588046899` | `-28.258988020426564` | — |
| `/re_z0` | array[100], SHA256 `445d0cd96a27` | array[100], SHA256 `9a57c390ed2c` | 0–99 |
| `/s21_mag` | array[100], SHA256 `ddb7fff5de02` | array[100], SHA256 `083a4bbed132` | 0–99 |
| `/solve_s` | `307.8088598251343` | `310.7757911682129` | — |
| `/sub_bin_shift` | `-0.06756611131760686` | `-0.4654723704964401` | — |
| `/witness_bins` | `0.461050988950462` | `0.6280088415546268` | — |
| `/worst_sampled_depth_db` | `-31.33868159383999` | `-31.8623216902829` | — |
| `/z0_median` | `46.43851280212402` | `48.03964805603027` | — |

## validation/crossval/_07_sheen_results/rfx.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/energy_sum` | array[120], SHA256 `62e1fb7f092d` | array[120], SHA256 `211cbfd76716` | 0–119 |
| `/fidelity_report` | *absent* | `"=== INPUT-FIDELITY REPORT (declared vs solved; input units only) ===\n  6 entities, 10 findings\n\n  domain (the solved box)\n    x: declared [0.0, 27472.0] um -> realized [0.0, 27600.0] um &#124; face residuals (0.0, 128.0) um &#124; extent 27472.0 -> 27600.0 um\n    y: declared [0.0, 26320.0] um -> realized [0.0, 26400.0] um &#124; face residuals (0.0, 80.0) um &#124; extent 26320.0 -> 26400.0 um\n    z: declared [0.0, 3794.0] um -> realized [0.0, 3800.0] um &#124; face residuals (0.0, 6.0) um &#124; extent 3794.0 -> 3800.0 um\n\n  geometry[0] 'duroid'\n    cells: 72864\n    x: declared [0.0, 27472.0] um -> realized [0.0, 27600.0] um &#124; face residuals (0.0, 128.0) um &#124; extent 27472.0 -> 27600.0 um\n    y: declared [0.0, 26320.0] um -> realized [0.0, 26400.0] um &#124; face residuals (0.0, 80.0) um &#124; extent 26320.0 -> 26400.0 um\n    z: declared [0.0, 794.0] um -> realized [0.0, 800.0] um &#124; face residuals (0.0, 6.0) um &#124; extent 794.0 -> 800.0 um\n    ! [inside-absorber] x,y: the realized body extends beyond the declared domain on x,y — those cells sit in the CPML pad, where the update equations are the absorber's\n      remedy: keep the body inside the domain, or enlarge the domain so the absorber stays empty\n    ! [off-lattice-face] z: PLACEMENT off by 6.0 um (0.76% of the declared 794.0 um extent); SIZE 800.0 um (+6.0 um, +0.76%)\n      remedy: place a mesh node on this face (non-uniform profile) or choose the cell size / origin commensurate with it\n\n  geometry[1] 'pec' — PEC sheet on node plane z = 800.00 um (756 nodes, closed footprint, zero thickness, no cell; in-plane E zeroed on that plane, normal E live)\n    sheet nodes: 756 (owns no cell)\n    x: declared [0.0, 12466.0] um -> realized [0.0, 12400.0] um &#124; face residuals (0.0, 66.0) um &#124; extent 12466.0 -> 12400.0 um\n    y: declared [8650.0, 11063.0] um -> realized [8800.0, 11000.0] um &#124; face residuals (150.0, 63.0) um &#124; extent 2413.0 -> 2200.0 um\n    z: declared [794.0, 794.0] um -> realized [800.0, 800.0] um &#124; face residuals (6.0, 6.0) um &#124; extent 0.0 -> 0.0 um\n    ! [off-lattice-face] x: PLACEMENT off by 66.0 um (0.53% of the declared 12466.0 um extent); SIZE 12400.0 um (-66.0 um, -0.53%)\n      remedy: place a mesh node on this face (non-uniform profile) or choose the cell size / origin commensurate with it\n    ! [off-lattice-face] y: PLACEMENT off by 150.0 um (6.22% of the declared 2413.0 um extent); SIZE 2200.0 um (-213.0 um, -8.83%)\n      remedy: place a mesh node on this face (non-uniform profile) or choose the cell size / origin commensurate with it\n\n  geometry[2] 'pec' — PEC sheet on node plane z = 800.00 um (1326 nodes, closed footprint, zero thickness, no cell; in-plane E zeroed on that plane, normal E live)\n    sheet nodes: 1326 (owns no cell)\n    x: declared [12466.0, 15006.0] um -> realized [12600.0, 15000.0] um &#124; face residuals (134.0, 6.0) um &#124; extent 2540.0 -> 2400.0 um\n    y: declared [3000.0, 23320.0] um -> realized [3000.0, 23200.0] um &#124; face residuals (0.0, 120.0) um &#124; extent 20320.0 -> 20200.0 um\n    z: declared [794.0, 794.0] um -> realized [800.0, 800.0] um &#124; face residuals (6.0, 6.0) um &#124; extent 0.0 -> 0.0 um\n    ! [off-lattice-face] x: PLACEMENT off by 134.0 um (5.28% of the declared 2540.0 um extent); SIZE 2400.0 um (-140.0 um, -5.51%)\n      remedy: place a mesh node on this face (non-uniform profile) or choose the cell size / origin commensurate with it\n    ! [off-lattice-face] y: PLACEMENT off by 120.0 um (0.59% of the declared 20320.0 um extent); SIZE 20200.0 um (-120.0 um, -0.59%)\n      remedy: place a mesh node on this face (non-uniform profile) or choose the cell size / origin commensurate with it\n\n  geometry[3] 'pec' — PEC sheet on node plane z = 800.00 um (744 nodes, closed footprint, zero thickness, no cell; in-plane E zeroed on that plane, normal E live)\n    sheet nodes: 744 (owns no cell)\n    x: declared [15006.0, 27472.0] um -> realized [15200.0, 27400.0] um &#124; face residuals (194.0, 72.0) um &#124; extent 12466.0 -> 12200.0 um\n    y: declared [15257.0, 17670.0] um -> realized [15400.0, 17600.0] um &#124; face residuals (143.0, 70.0) um &#124; extent 2413.0 -> 2200.0 um\n    z: declared [794.0, 794.0] um -> realized [800.0, 800.0] um &#124; face residuals (6.0, 6.0) um &#124; extent 0.0 -> 0.0 um\n    ! [inside-absorber] x: the realized body extends beyond the declared domain on x — those cells sit in the CPML pad, where the update equations are the absorber's\n      remedy: keep the body inside the domain, or enlarge the domain so the absorber stays empty\n    ! [off-lattice-face] x: PLACEMENT off by 194.0 um (1.56% of the declared 12466.0 um extent); SIZE 12200.0 um (-266.0 um, -2.13%)\n      remedy: place a mesh node on this face (non-uniform profile) or choose the cell size / origin commensurate with it\n    ! [off-lattice-face] y: PLACEMENT off by 143.0 um (5.93% of the declared 2413.0 um extent); SIZE 2200.0 um (-213.0 um, -8.83%)\n      remedy: place a mesh node on this face (non-uniform profile) or choose the cell size / origin commensurate with it\n\n  NOT AUDITED by this report\n    ! [out-of-scope] 2 MSL port(s) — these declare geometry and/or materials that this report does not walk (port pins, shields, end caps, source cells)\n      remedy: audit those with their own port preflight checks; do not read this report as covering them\n\n"` | — |
| `/passivity_correction` | array[120], SHA256 `1e5740733173` | array[120], SHA256 `60e8f4752631` | 0–18, 26–48, 103–104, 106–110 |
| `/preflight` | `"  [PREFLIGHT] pec_faces={z_lo} creates an INFINITE PEC boundary AND the geometry contains finite PEC objects. For antennas or finite-GP structures, the pec_faces boundary makes the ground plane cover the entire domain face, which changes the physics (cavity vs radiating antenna). If you need a finite ground plane, remove pec_faces and use an explicit PEC Box instead.\n  [PREFLIGHT] all dielectric(s) ['duroid'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)\n  [PREFLIGHT] MSL port 'p1' at x=2.50mm, direction='+x': distance to nearest x-CPML = 900µm (domain edge + 1.6mm calibrated CPML buffer) < recommended 1588µm (= 2·h_sub). Source-side CPML reflection may inflate &#124;S11&#124;. Move port further from boundary OR increase domain x-extent.\n  [PREFLIGHT] MSL port 'p2' at x=24.97mm, direction='-x': distance to nearest x-CPML = 900µm (domain edge + 1.6mm calibrated CPML buffer) < recommended 1588µm (= 2·h_sub). Source-side CPML reflection may inflate &#124;S11&#124;. Move port further from boundary OR increase domain x-extent.\n"` | `"  [PREFLIGHT] Zero-thickness geometry 'pec' along z-axis. On non-uniform mesh this may produce empty rasterization. Consider giving it at least one cell of thickness (200µm).\n  [PREFLIGHT] Zero-thickness geometry 'pec' along z-axis. On non-uniform mesh this may produce empty rasterization. Consider giving it at least one cell of thickness (200µm).\n  [PREFLIGHT] Zero-thickness geometry 'pec' along z-axis. On non-uniform mesh this may produce empty rasterization. Consider giving it at least one cell of thickness (200µm).\n  [PREFLIGHT] pec_faces={z_lo} creates an INFINITE PEC boundary AND the geometry contains finite PEC objects. For antennas or finite-GP structures, the pec_faces boundary makes the ground plane cover the entire domain face, which changes the physics (cavity vs radiating antenna). If you need a finite ground plane, remove pec_faces and use an explicit PEC Box instead.\n  [PREFLIGHT] all dielectric(s) ['duroid'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)\n  [PREFLIGHT] 1 congruent-conductor group(s) rasterize to UNEQUAL cell counts on this lattice (design-identical solids, different meshes). Worst group (Box, sorted extents 0mm x 2.413mm x 12.47mm): geometry[1] 'pec' 756 cells, lo-corner sub-lattice offsets (x,y,z)=(0.000, 0.250, 0.970) cells; geometry[3] 'pec' 744 cells, lo-corner sub-lattice offsets (x,y,z)=(0.030, 0.285, 0.970) cells. OBSERVED: cell-count spread 12 > tolerance 1 cell (one cell along the group's smallest extent, 0mm). WHY: congruent solids whose faces sit at different sub-cell offsets are sampled by different node sets, so the mesh invents an asymmetry the design does not have — a mirror pair whose mirror plane is off-lattice rasterizes asymmetrically with the same sign in every pair. COST (measured, issue #703): mirror pairs 173 vs 183 cells (5.6%) from a mirror plane 0.26 cells off the lattice; an A/B run pair differing ONLY by a 13µm lattice-origin slide moved &#124;S11&#124; up to 3.5 dB per bin and improved every aggregate agreement metric. REMEDY: place the members at positions congruent modulo the cell size (no candidate origin slide improved the spread — the members' offsets differ on more than one axis, so equalizing them needs a geometry move). COVERAGE: examined 3 conductor entr(y/ies) that report a bounding box, in 1 congruence group(s) of >=2 members on the uniform lane; skipped 0 conductor entr(y/ies) whose shape reports no bounding box (no congruence key) and 0 thin-conductor sheet(s) (not congruence-grouped). STALE IF: re-rasterizing the named members gives equal counts (spread <= tolerance), or conductors stop being sampled on the E-node lattice.\n  [PREFLIGHT] _assemble_materials (uniform lane): PEC sheets/wires were classified but the caller passed no pec_sheets/pec_wires collector, so they are absent from the returned pec_mask (a sheet owns no cell, #931 §1.3). A caller that steps fields must pass collectors and realize them with rfx.boundaries.pec.realized_pec_edge_masks.\n  [PREFLIGHT] 5 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 5 listed): geometry[3] 'pec' y: extent 2.413mm, worst face residual 70µm (2.90% of the extent, df/f ~ 2.90%); geometry[1] 'pec' y: extent 2.413mm, worst face residual 63µm (2.61% of the extent, df/f ~ 2.61%); geometry[2] 'pec' x: extent 2.54mm, worst face residual 66µm (2.60% of the extent, df/f ~ 2.60%); geometry[3] 'pec' x: extent 12.47mm, worst face residual 72µm (0.58% of the extent, df/f ~ 0.58%); geometry[1] 'pec' x: extent 12.47mm, worst face residual 66µm (0.53% of the extent, df/f ~ 0.53%). OBSERVED: distance from each Box face to its nearest E-node on this run's own node coordinates; the rasterized edge quantizes onto a node, so the realized extent can differ from the design by up to the printed residual, and a resonant dimension realized dL off detunes df/f ~ dL/L. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50µm only because every patterned dimension happened to be an exact multiple of 50µm. REMEDY: choose dx commensurate with the patterned dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on the design edges. COVERAGE: examined 6 axis extent(s) on 3 conductor Box entr(y/ies) on the uniform lane; 3 sub-cell axis extent(s) excluded (the node-thin snap is the live-edge/cavity checks' domain); 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: &#124;face - nearest node&#124; on the run's node coordinates does not reproduce the printed residuals, or box faces stop rasterizing on the E-node lattice.\n  [PREFLIGHT] _assemble_materials (uniform lane): PEC sheets/wires were classified but the caller passed no pec_sheets/pec_wires collector, so they are absent from the returned pec_mask (a sheet owns no cell, #931 §1.3). A caller that steps fields must pass collectors and realize them with rfx.boundaries.pec.realized_pec_edge_masks.\n  [PREFLIGHT] MSL port 'p1' at x=2.50mm, direction='+x': distance to nearest x-CPML = 900µm (domain edge + 1.6mm calibrated CPML buffer) < recommended 1588µm (= 2·h_sub). Source-side CPML reflection may inflate &#124;S11&#124;. Move port further from boundary OR increase domain x-extent.\n  [PREFLIGHT] MSL port 'p2' at x=24.97mm, direction='-x': distance to nearest x-CPML = 900µm (domain edge + 1.6mm calibrated CPML buffer) < recommended 1588µm (= 2·h_sub). Source-side CPML reflection may inflate &#124;S11&#124;. Move port further from boundary OR increase domain x-extent.\n"` | — |
| `/re_z0` | array[120], SHA256 `55e9ff37ebd7` | array[120], SHA256 `eb21374a38c9` | 0–119 |
| `/recipe_evidence` | `"num_periods=60: settling witness -70.2/-71.3 dB per driven run (rule < -40; the retired num_periods=20 leg failed it at -24.7/-24.3 dB and carried truncation-artifact &#124;S&#124; poles, PR #468). n_probe_offset=30: clears BOTH port-probe constraints on this short-feed geometry — >= 5*h_sub from the feed (near-field; measured 16-20 GHz contamination collapse 8.75 -> 1.81, PR #468) AND >= lambda_g/4 from the downstream patch discontinuity (which offset=40 violated, preflight-warned). Extractor: the #511/#507-corrected MSL path (PR #516, f95240f) — trace-anchored modal voltage (the one-edge V-span defect is fixed) and per-frequency two-drive S assembly (the single-ratio assembly defect is fixed). S is passivity-enforced: strict &#124;&#124;S&#124;&#124;_2 <= 1, raw excess recorded per bin in passivity_correction (0/120 bins > 0.05, worst 0.015 at 17.38 GHz — the dx=200um coarse-mesh envelope, bounded and recorded rather than quoted). Fitted median Re(Z0): passband (0.5-3 GHz) 50.3 ohm, in-band (5-15 GHz) 52.4 ohm. Preflight output is stored verbatim in this JSON's 'preflight' field."` | `"num_periods=60: settling witness -64.8/-64.8 dB per driven run (rule < -40; the retired num_periods=20 leg failed it at -24.7/-24.3 dB and carried truncation-artifact &#124;S&#124; poles, PR #468). n_probe_offset=30: clears BOTH port-probe constraints on this short-feed geometry — >= 5*h_sub from the feed (near-field; measured 16-20 GHz contamination collapse 8.75 -> 1.81, PR #468) AND >= lambda_g/4 from the downstream patch discontinuity (which offset=40 violated, preflight-warned). Extractor: the #511/#507-corrected MSL path (PR #516, f95240f) — trace-anchored modal voltage (the one-edge V-span defect is fixed) and per-frequency two-drive S assembly (the single-ratio assembly defect is fixed). S is passivity-enforced: strict &#124;&#124;S&#124;&#124;_2 <= 1, raw excess recorded per bin in passivity_correction (3/120 bins > 0.05, worst 0.657 at 17.87 GHz — the dx=200um coarse-mesh envelope, bounded and recorded rather than quoted). Fitted median Re(Z0): passband (0.5-3 GHz) 51.9 ohm, in-band (5-15 GHz) 54.7 ohm. Preflight output is stored verbatim in this JSON's 'preflight' field."` | — |
| `/rfx_provenance` | `"fix/519-regenerate-stale-crossval-legs 003ab97"` | `"unknown"` | — |
| `/runtime_s` | `258.75834035873413` | `267.941597700119` | — |
| `/s11_mag` | array[120], SHA256 `f8f35e56d784` | array[120], SHA256 `e97b2c865a1d` | 0–119 |
| `/s21_mag` | array[120], SHA256 `6006ddd805a5` | array[120], SHA256 `65ec38739e51` | 0–119 |
| `/settling_db` | `[-70.17436315474305, -71.25531835637528]` | `[-64.80486687076369, -64.78452817446357]` | 0–1 |

## validation/crossval/_15_patch_results/rfx.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/f_dip_hz` | `2310000128.0` | `2420000000.0` | — |
| `/f_harminv_hz` | `2313947436.4204316` | `2423039171.4025517` | — |
| `/f_primary_hz` | `2313947436.4204316` | `2423039171.4025517` | — |
| `/feed` | *absent* | `"full_span"` | — |
| `/gain_dbi` | `7.242423057556152` | `7.199780464172363` | — |
| `/max_abs_s11` | `0.7869663834571838` | `0.9905672669410706` | — |
| `/port_extent` | *absent* | `0.003175` | — |
| `/port_z0` | *absent* | `0.0079375` | — |
| `/preflight` | `"  [PREFLIGHT] 'pec' z-extent 793.7µm = 1.0 cells — below 1 cell resolution. A conductor thinner than a cell is modelled as a one-cell PEC surface — tangential E is zeroed on it and the normal component survives as surface charge. That is usually what you want for metal many skin depths thick, and switching to add_thin_conductor() would not change it; but it means the sheet carries no conductor loss and its thickness is not modelled. For band-centre conductor loss use add_thin_conductor(..., surface_impedance_f0=...) (Leontovich surface resistance, issue #669).\n  [PREFLIGHT] Wire port at (0.027000000000000003, 0.041, 0.00873125) (extent 0.0015875, component ez): its +z-side end cell (42, 60, 21) is live (vacuum/dielectric), but the cell immediately beyond it ((42, 60, 22)) is PEC ['pec'] in the assembled geometry. The port terminates in vacuum/dielectric one cell (gap = 1 cell = 0.00079375 m) short of a rasterized conductor, so the feed never galvanically reaches it and coupling is capacitive only (measured signature: &#124;S21&#124; rising with frequency; issue #556, the #488-lane D5 finding). Remedy: extend the port extent by one cell so its end lands on the conductor's rasterized node, or refine dx so the conductor interface aligns with a grid node (e.g. dx = h/N for an interface at height h).\n  [PREFLIGHT] 1 sheet-bounded cavit(y/ies) differ from the physical stack by more than 1% in electrical thickness: [z] geometry[0](node k=17)/geometry[2](k=22) at in-plane column (53,60): sum(d/eps) mesh 2.237mm vs physical 1.443mm (+55.0%); sum(d*sqrt(eps)) mesh 5.503mm vs physical 4.709mm (+16.9%); node-to-node 3.969mm vs face-to-face 3.175mm; the capacitance measure (sum d/eps) governs (gap << lambda at freq_max); of the sum(d/eps) mesh total, 793.8µm is geometry[0]'s OWN cell (793.8µm at eps_r 1.000) — that sheet fills one cell, and rfx zeroes only TANGENTIAL E on a one-cell PEC sheet, so the cell's normal-E edge stays live and sits INSIDE the cavity. OBSERVED: mesh sums run node-to-node across the run's own cells and assembled eps_r; physical sums run face-to-face through the geometry Box spans at the pair's shared column — the difference is the zero-thickness sheet model's honest cost (each sheet's thickness collapses onto its node) plus any off-lattice registration. TWO MECHANISMS, and any pair naming an OWN cell above has the second one: a sheet registered at its MID-PLANE collapses onto one node and the gap reads mid-plane to mid-plane (a modelling trade — which face the plane sits on — not a fixable defect), while a sheet whose two FACES are registered fills one cell, and rfx zeroes only tangential E on a one-cell PEC sheet, so that cell's normal-E edge is live and its permittivity sits inside the cavity. Face registration therefore does not shorten the electrical cavity; it trades a collapsed sheet for a live gap, which reads WORSE when that cell is vacuum. WHY BOTH MEASURES: the same defect class measured 17.3% as a series capacitance and 3.2% as phase length (#703) — a bare percentage invites 'correcting' a right number into a wrong one, so both are printed and the governing one is named. REMEDY: none required for the sheet model itself (this is a quantified limit, not a defect); if the governing measure's delta matters for a claims-bearing number, resolve the sheet thickness with cells or correct the extracted quantity by the printed delta. For an OWN-cell term the eps_r printed for that cell IS addressable: it is whatever the geometry puts on the live edge (issue #702), so a stack whose dielectric abuts the sheet's faces leaves vacuum there — extend the abutting dielectric across the sheet's cell, or register the sheet's mid-plane instead of its faces, and that term goes. COVERAGE: examined 1 adjacent sheet pair(s) from 2 node-thin conductor sheet(s) on the uniform lane; physical stack computed from Box entries only — 0 non-Box dielectric entr(y/ies) ignored (said so, per #703); 0 pair(s) skipped (conductor between). STALE IF: re-summing the printed column disagrees with these numbers, or sheets stop being registered node-thin.\n  [PREFLIGHT] 2 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 2 listed): geometry[2] 'pec' x: extent 40mm, worst face residual 356.2µm (0.89% of the extent, df/f ~ 0.89%); geometry[0] 'pec' x: extent 56mm, worst face residual 293.7µm (0.52% of the extent, df/f ~ 0.52%). OBSERVED: distance from each Box face to its nearest E-node on this run's own node coordinates; the rasterized edge quantizes onto a node, so the realized extent can differ from the design by up to the printed residual, and a resonant dimension realized dL off detunes df/f ~ dL/L. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50µm only because every patterned dimension happened to be an exact multiple of 50µm. REMEDY: choose dx commensurate with the patterned dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on the design edges. COVERAGE: examined 4 axis extent(s) on 2 conductor Box entr(y/ies) on the uniform lane; 2 sub-cell axis extent(s) excluded (the node-thin snap is the live-edge/cavity checks' domain); 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: &#124;face - nearest node&#124; on the run's node coordinates does not reproduce the printed residuals, or box faces stop rasterizing on the E-node lattice.\n  [PREFLIGHT] NTFF face x_lo is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face x_hi is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face y_lo is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face y_hi is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face z_lo is 0.79mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face z_hi is 7.86mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] Far-field pattern advisory: the PEC sheet backing a source (bbox (8.0, 8.0, 7.1)–(64.0, 74.0, 7.9) mm) spans 66.0mm × 56.0mm = 0.55λ × 0.47λ at f_max=2.50GHz — a ground plane under ~1λ across. Expect the radiation pattern to be shaped by ground-plane edge diffraction (broadside dip, off-axis side peaks). This is expected physics, not a solver defect, and the fixture stays fine for resonance / impedance work. For a clean broadside pattern enlarge the ground plane to at least ~1.4λ; if the small ground plane is intentional, interpret the pattern accordingly.\n"` | `"  [PREFLIGHT] 2 PEC sheet(s) realized (lattice ownership contract #931 §1.3: one node plane each, closed footprint, normal E through the plane live), 0 of them off their declared mid-plane; worst first: geometry[0] 'pec' normal z: declared mid-plane 7.938mm, realized node plane 18 at 7.938mm (offset +0.000 cell = 0mm); geometry[2] 'pec' normal z: declared mid-plane 11.11mm, realized node plane 22 at 11.11mm (offset +0.000 cell = 0mm). A sheet snaps to the node plane nearest its declared mid-plane (an exact half-cell tie resolves LOWER); an offset means the declared plane — a laminate face, a ground level — is not on this mesh's node line, and the conductor sits that far from where it was drawn. REMEDY when the offset matters: put a mesh node on the declared plane (dx = h/N for an interface at height h, or a preserved region on the non-uniform lane). COVERAGE: every sheet declaration on the uniform lane (zero-thickness PEC Boxes via add() and PEC add_thin_conductor entries). STALE IF: the named sheet's SheetSpec.plane is not the printed node.\n  [PREFLIGHT] 2 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 2 listed): geometry[2] 'pec' (sheet) x: extent 40mm, worst face residual 356.2µm (0.89% of the extent, df/f ~ 0.89%); geometry[0] 'pec' (sheet) x: extent 56mm, worst face residual 293.7µm (0.52% of the extent, df/f ~ 0.52%). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3), so the realized extent can differ from the design by up to the printed residual, and a resonant dimension realized dL off detunes df/f ~ dL/L. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: choose dx commensurate with the patterned dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on the design edges. COVERAGE: examined 4 axis extent(s) on 2 conductor Box declaration(s) on the uniform lane; 2 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: &#124;face - nearest node&#124; on the run's node coordinates does not reproduce the printed residuals.\n  [PREFLIGHT] NTFF face x_lo is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face x_hi is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face y_lo is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face y_hi is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face z_lo is 1.59mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face z_hi is 8.65mm from geometry 'sub' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] Far-field pattern advisory: the PEC sheet backing a source (bbox (8.0, 8.0, 7.9)–(64.0, 74.0, 7.9) mm) spans 66.0mm × 56.0mm = 0.55λ × 0.47λ at f_max=2.50GHz — a ground plane under ~1λ across. Expect the radiation pattern to be shaped by ground-plane edge diffraction (broadside dip, off-axis side peaks). This is expected physics, not a solver defect, and the fixture stays fine for resonance / impedance work. For a clean broadside pattern enlarge the ground plane to at least ~1.4λ; if the small ground plane is intentional, interpret the pattern accordingly.\n"` | — |
| `/q_harminv` | `18.897386958978245` | `10.060504212741462` | — |
| `/runtime_s` | `206.5058035850525` | `74.90003871917725` | — |
| `/s11_dip_db` | `-4.429839134216309` | `-19.048009872436523` | — |
| `/s11_im` | array[181], SHA256 `e625150069c0` | array[181], SHA256 `9acd79ec2492` | 0–180 |
| `/s11_mag` | array[181], SHA256 `6753d04228c9` | array[181], SHA256 `235c25386a6c` | 0–180 |
| `/s11_re` | array[181], SHA256 `c9f0021d1100` | array[181], SHA256 `ba8d12edd63d` | 0–180 |
| `/settle_db` | `-53.9897794557981` | `-68.69142333390357` | — |
| `/stack_check/ground_realization` | `"two_plane"` | `"sheet"` | — |
| `/stack_check/n_distinct_eps` | *absent* | `2` | — |
| `/stack_check/patch_realization` | *absent* | `"sheet"` | — |

## validation/crossval/_15_patch_results/rfx_decomposition_feed_pre931.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/dx_um` | *absent* | `793.75` | — |
| `/f_analytic_hz` | *absent* | `2415595433.5060616` | — |
| `/f_dip_hz` | *absent* | `2380000000.0` | — |
| `/f_harminv_hz` | *absent* | `2377431249.047172` | — |
| `/f_primary_hz` | *absent* | `2377431249.047172` | — |
| `/feed` | *absent* | `"pre931"` | — |
| `/freqs_hz` | *absent* | array[181], SHA256 `6dab18cef94e` | 0–180 |
| `/gain_dbi` | *absent* | `7.199788570404053` | — |
| `/max_abs_s11` | *absent* | `1.0000054836273193` | — |
| `/n_steps` | *absent* | `12389` | — |
| `/n_sub_cells` | *absent* | `4` | — |
| `/num_periods` | *absent* | `45.0` | — |
| `/port_extent` | *absent* | `0.0015875` | — |
| `/port_z0` | *absent* | `0.00873125` | — |
| `/preflight` | *absent* | `"  [PREFLIGHT] Wire port at (0.027000000000000003, 0.041, 0.00873125) (extent 0.0015875, component ez): its -z-side end node (42, 60, 19) (z = 8.731mm) carries no realized PEC wall, but the node one cell further (z = 7.938mm) is a realized wall plane of ['pec']. The port terminates in vacuum/dielectric one cell (gap = 0.00079375 m) short of the conductor, so the feed never galvanically reaches it and coupling is capacitive only (measured signature: &#124;S21&#124; rising with frequency; issue #556, the #488-lane D5 finding). Remedy: extend the port extent by one cell so its end node lands on that wall plane, or draw the conductor so its face (a volume) or its plane (a sheet) lands on the wire's end node — e.g. dx = h/N for an interface at height h.\n  [PREFLIGHT] Wire port at (0.027000000000000003, 0.041, 0.00873125) (extent 0.0015875, component ez): its +z-side end node (42, 60, 21) (z = 10.32mm) carries no realized PEC wall, but the node one cell further (z = 11.11mm) is a realized wall plane of ['pec']. The port terminates in vacuum/dielectric one cell (gap = 0.00079375 m) short of the conductor, so the feed never galvanically reaches it and coupling is capacitive only (measured signature: &#124;S21&#124; rising with frequency; issue #556, the #488-lane D5 finding). Remedy: extend the port extent by one cell so its end node lands on that wall plane, or draw the conductor so its face (a volume) or its plane (a sheet) lands on the wire's end node — e.g. dx = h/N for an interface at height h.\n  [PREFLIGHT] 2 PEC sheet(s) realized (lattice ownership contract #931 §1.3: one node plane each, closed footprint, normal E through the plane live), 0 of them off their declared mid-plane; worst first: geometry[0] 'pec' normal z: declared mid-plane 7.938mm, realized node plane 18 at 7.938mm (offset +0.000 cell = 0mm); geometry[2] 'pec' normal z: declared mid-plane 11.11mm, realized node plane 22 at 11.11mm (offset +0.000 cell = 0mm). A sheet snaps to the node plane nearest its declared mid-plane (an exact half-cell tie resolves LOWER); an offset means the declared plane — a laminate face, a ground level — is not on this mesh's node line, and the conductor sits that far from where it was drawn. REMEDY when the offset matters: put a mesh node on the declared plane (dx = h/N for an interface at height h, or a preserved region on the non-uniform lane). COVERAGE: every sheet declaration on the uniform lane (zero-thickness PEC Boxes via add() and PEC add_thin_conductor entries). STALE IF: the named sheet's SheetSpec.plane is not the printed node.\n  [PREFLIGHT] 2 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 2 listed): geometry[2] 'pec' (sheet) x: extent 40mm, worst face residual 356.2µm (0.89% of the extent, df/f ~ 0.89%); geometry[0] 'pec' (sheet) x: extent 56mm, worst face residual 293.7µm (0.52% of the extent, df/f ~ 0.52%). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3), so the realized extent can differ from the design by up to the printed residual, and a resonant dimension realized dL off detunes df/f ~ dL/L. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: choose dx commensurate with the patterned dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on the design edges. COVERAGE: examined 4 axis extent(s) on 2 conductor Box declaration(s) on the uniform lane; 2 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: &#124;face - nearest node&#124; on the run's node coordinates does not reproduce the printed residuals.\n  [PREFLIGHT] NTFF face x_lo is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face x_hi is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face y_lo is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face y_hi is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face z_lo is 1.59mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face z_hi is 8.65mm from geometry 'sub' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] Far-field pattern advisory: the PEC sheet backing a source (bbox (8.0, 8.0, 7.9)–(64.0, 74.0, 7.9) mm) spans 66.0mm × 56.0mm = 0.55λ × 0.47λ at f_max=2.50GHz — a ground plane under ~1λ across. Expect the radiation pattern to be shaped by ground-plane edge diffraction (broadside dip, off-axis side peaks). This is expected physics, not a solver defect, and the fixture stays fine for resonance / impedance work. For a clean broadside pattern enlarge the ground plane to at least ~1.4λ; if the small ground plane is intentional, interpret the pattern accordingly.\n"` | — |
| `/q_harminv` | *absent* | `18.29681634549151` | — |
| `/runtime_s` | *absent* | `76.83319139480591` | — |
| `/s11_dip_db` | *absent* | `-0.009698154404759407` | — |
| `/s11_im` | *absent* | array[181], SHA256 `00174b90b4b7` | 0–180 |
| `/s11_mag` | *absent* | array[181], SHA256 `10953cf0dc97` | 0–180 |
| `/s11_re` | *absent* | array[181], SHA256 `194c4e12afaa` | 0–180 |
| `/settle_db` | *absent* | `-55.419193263752284` | — |
| `/settled` | *absent* | `true` | — |
| `/solver` | *absent* | `"rfx"` | — |
| `/stack_check/eps_between` | *absent* | `[2.200000286102295, 2.200000286102295, 2.200000286102295, 2.200000286102295]` | 0–3 |
| `/stack_check/ground_realization` | *absent* | `"sheet"` | — |
| `/stack_check/ground_wall_z` | *absent* | `0.0079375` | — |
| `/stack_check/n_distinct_eps` | *absent* | `2` | — |
| `/stack_check/n_sub_cells` | *absent* | `4` | — |
| `/stack_check/patch_realization` | *absent* | `"sheet"` | — |
| `/stack_check/patch_wall_z` | *absent* | `0.0111125` | — |

## validation/crossval/_15_patch_results/rfx_pre931_two_plane_ground_1f005d0d.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/dx_um` | *absent* | `793.75` | — |
| `/f_analytic_hz` | *absent* | `2415595433.5060616` | — |
| `/f_dip_hz` | *absent* | `2310000128.0` | — |
| `/f_harminv_hz` | *absent* | `2313947436.4204316` | — |
| `/f_primary_hz` | *absent* | `2313947436.4204316` | — |
| `/freqs_hz` | *absent* | array[181], SHA256 `6dab18cef94e` | 0–180 |
| `/gain_dbi` | *absent* | `7.242423057556152` | — |
| `/max_abs_s11` | *absent* | `0.7869663834571838` | — |
| `/n_steps` | *absent* | `12389` | — |
| `/n_sub_cells` | *absent* | `4` | — |
| `/num_periods` | *absent* | `45.0` | — |
| `/preflight` | *absent* | `"  [PREFLIGHT] 'pec' z-extent 793.7µm = 1.0 cells — below 1 cell resolution. A conductor thinner than a cell is modelled as a one-cell PEC surface — tangential E is zeroed on it and the normal component survives as surface charge. That is usually what you want for metal many skin depths thick, and switching to add_thin_conductor() would not change it; but it means the sheet carries no conductor loss and its thickness is not modelled. For band-centre conductor loss use add_thin_conductor(..., surface_impedance_f0=...) (Leontovich surface resistance, issue #669).\n  [PREFLIGHT] Wire port at (0.027000000000000003, 0.041, 0.00873125) (extent 0.0015875, component ez): its +z-side end cell (42, 60, 21) is live (vacuum/dielectric), but the cell immediately beyond it ((42, 60, 22)) is PEC ['pec'] in the assembled geometry. The port terminates in vacuum/dielectric one cell (gap = 1 cell = 0.00079375 m) short of a rasterized conductor, so the feed never galvanically reaches it and coupling is capacitive only (measured signature: &#124;S21&#124; rising with frequency; issue #556, the #488-lane D5 finding). Remedy: extend the port extent by one cell so its end lands on the conductor's rasterized node, or refine dx so the conductor interface aligns with a grid node (e.g. dx = h/N for an interface at height h).\n  [PREFLIGHT] 1 sheet-bounded cavit(y/ies) differ from the physical stack by more than 1% in electrical thickness: [z] geometry[0](node k=17)/geometry[2](k=22) at in-plane column (53,60): sum(d/eps) mesh 2.237mm vs physical 1.443mm (+55.0%); sum(d*sqrt(eps)) mesh 5.503mm vs physical 4.709mm (+16.9%); node-to-node 3.969mm vs face-to-face 3.175mm; the capacitance measure (sum d/eps) governs (gap << lambda at freq_max); of the sum(d/eps) mesh total, 793.8µm is geometry[0]'s OWN cell (793.8µm at eps_r 1.000) — that sheet fills one cell, and rfx zeroes only TANGENTIAL E on a one-cell PEC sheet, so the cell's normal-E edge stays live and sits INSIDE the cavity. OBSERVED: mesh sums run node-to-node across the run's own cells and assembled eps_r; physical sums run face-to-face through the geometry Box spans at the pair's shared column — the difference is the zero-thickness sheet model's honest cost (each sheet's thickness collapses onto its node) plus any off-lattice registration. TWO MECHANISMS, and any pair naming an OWN cell above has the second one: a sheet registered at its MID-PLANE collapses onto one node and the gap reads mid-plane to mid-plane (a modelling trade — which face the plane sits on — not a fixable defect), while a sheet whose two FACES are registered fills one cell, and rfx zeroes only tangential E on a one-cell PEC sheet, so that cell's normal-E edge is live and its permittivity sits inside the cavity. Face registration therefore does not shorten the electrical cavity; it trades a collapsed sheet for a live gap, which reads WORSE when that cell is vacuum. WHY BOTH MEASURES: the same defect class measured 17.3% as a series capacitance and 3.2% as phase length (#703) — a bare percentage invites 'correcting' a right number into a wrong one, so both are printed and the governing one is named. REMEDY: none required for the sheet model itself (this is a quantified limit, not a defect); if the governing measure's delta matters for a claims-bearing number, resolve the sheet thickness with cells or correct the extracted quantity by the printed delta. For an OWN-cell term the eps_r printed for that cell IS addressable: it is whatever the geometry puts on the live edge (issue #702), so a stack whose dielectric abuts the sheet's faces leaves vacuum there — extend the abutting dielectric across the sheet's cell, or register the sheet's mid-plane instead of its faces, and that term goes. COVERAGE: examined 1 adjacent sheet pair(s) from 2 node-thin conductor sheet(s) on the uniform lane; physical stack computed from Box entries only — 0 non-Box dielectric entr(y/ies) ignored (said so, per #703); 0 pair(s) skipped (conductor between). STALE IF: re-summing the printed column disagrees with these numbers, or sheets stop being registered node-thin.\n  [PREFLIGHT] 2 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 2 listed): geometry[2] 'pec' x: extent 40mm, worst face residual 356.2µm (0.89% of the extent, df/f ~ 0.89%); geometry[0] 'pec' x: extent 56mm, worst face residual 293.7µm (0.52% of the extent, df/f ~ 0.52%). OBSERVED: distance from each Box face to its nearest E-node on this run's own node coordinates; the rasterized edge quantizes onto a node, so the realized extent can differ from the design by up to the printed residual, and a resonant dimension realized dL off detunes df/f ~ dL/L. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50µm only because every patterned dimension happened to be an exact multiple of 50µm. REMEDY: choose dx commensurate with the patterned dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on the design edges. COVERAGE: examined 4 axis extent(s) on 2 conductor Box entr(y/ies) on the uniform lane; 2 sub-cell axis extent(s) excluded (the node-thin snap is the live-edge/cavity checks' domain); 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: &#124;face - nearest node&#124; on the run's node coordinates does not reproduce the printed residuals, or box faces stop rasterizing on the E-node lattice.\n  [PREFLIGHT] NTFF face x_lo is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face x_hi is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face y_lo is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face y_hi is 1.65mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face z_lo is 0.79mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] NTFF face z_hi is 7.86mm from geometry 'pec' — below λ/4 = 29.98mm at f_max=2.50GHz. NTFF will integrate reactive near-field; directivity / pattern likely corrupted. Move NTFF box ≥ λ/2 from any radiating/scattering structure (Huygens-equivalence rule).\n  [PREFLIGHT] Far-field pattern advisory: the PEC sheet backing a source (bbox (8.0, 8.0, 7.1)–(64.0, 74.0, 7.9) mm) spans 66.0mm × 56.0mm = 0.55λ × 0.47λ at f_max=2.50GHz — a ground plane under ~1λ across. Expect the radiation pattern to be shaped by ground-plane edge diffraction (broadside dip, off-axis side peaks). This is expected physics, not a solver defect, and the fixture stays fine for resonance / impedance work. For a clean broadside pattern enlarge the ground plane to at least ~1.4λ; if the small ground plane is intentional, interpret the pattern accordingly.\n"` | — |
| `/q_harminv` | *absent* | `18.897386958978245` | — |
| `/runtime_s` | *absent* | `206.5058035850525` | — |
| `/s11_dip_db` | *absent* | `-4.429839134216309` | — |
| `/s11_im` | *absent* | array[181], SHA256 `e625150069c0` | 0–180 |
| `/s11_mag` | *absent* | array[181], SHA256 `6753d04228c9` | 0–180 |
| `/s11_re` | *absent* | array[181], SHA256 `c9f0021d1100` | 0–180 |
| `/settle_db` | *absent* | `-53.9897794557981` | — |
| `/settled` | *absent* | `true` | — |
| `/solver` | *absent* | `"rfx"` | — |
| `/stack_check/eps_between` | *absent* | `[2.200000286102295, 2.200000286102295, 2.200000286102295, 2.200000286102295]` | 0–3 |
| `/stack_check/ground_realization` | *absent* | `"two_plane"` | — |
| `/stack_check/ground_wall_z` | *absent* | `0.0079375` | — |
| `/stack_check/n_sub_cells` | *absent* | `4` | — |
| `/stack_check/patch_wall_z` | *absent* | `0.0111125` | — |

## validation/crossval/_18_wr90_iris_results/aperture_resolution.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/config/pooled_fine_gate_abs` | `0.04` | `0.02` | — |
| `/pairs/0/committed_fine_gap_abs` | `0.0122` | `0.0079` | — |
| `/pairs/0/fine_gate_abs` | `0.019` | `0.012` | — |
| `/pairs/0/nearest_offset_fine_cells` | `0.5` | `0.0` | — |
| `/pairs/0/one_cell_defect/over/fine_gap_abs` | `0.0446` | `0.0245` | — |
| `/pairs/0/one_cell_defect/over/fine_margin_x` | `2.349` | `2.043` | — |
| `/pairs/0/one_cell_defect/over/richardson_dev_abs` | `0.0069` | `0.0061` | — |
| `/pairs/0/one_cell_defect/under/fine_gap_abs` | `0.0225` | `0.0426` | — |
| `/pairs/0/one_cell_defect/under/fine_margin_x` | `1.185` | `3.553` | — |
| `/pairs/0/one_cell_defect/under/richardson_dev_abs` | `0.0061` | `0.0054` | — |
| `/pairs/0/oracle_distance_abs/+0.0` | `0.0122` | `0.0079` | — |
| `/pairs/0/oracle_distance_abs/+0.5` | `0.0045` | `0.0243` | — |
| `/pairs/0/oracle_distance_abs/+1.0` | `0.0203` | `0.0404` | — |
| `/pairs/0/oracle_distance_abs/-0.5` | `0.0291` | `0.009` | — |
| `/pairs/0/oracle_distance_abs/-1.0` | `0.0469` | `0.0268` | — |
| `/pairs/1/committed_fine_gap_abs` | `0.0223` | `0.0101` | — |
| `/pairs/1/fine_gate_abs` | `0.034` | `0.016` | — |
| `/pairs/1/nearest_offset_fine_cells` | `0.5` | `0.0` | — |
| `/pairs/1/one_cell_defect/over/fine_gap_abs` | `0.0648` | `0.0336` | — |
| `/pairs/1/one_cell_defect/over/fine_margin_x` | `1.906` | `2.101` | — |
| `/pairs/1/one_cell_defect/over/richardson_dev_abs` | `0.0066` | `0.0062` | — |
| `/pairs/1/one_cell_defect/under/detected_by_fine_gate` | `false` | `true` | — |
| `/pairs/1/one_cell_defect/under/fine_gap_abs` | `0.0269` | `0.0538` | — |
| `/pairs/1/one_cell_defect/under/fine_margin_x` | `0.791` | `3.362` | — |
| `/pairs/1/one_cell_defect/under/richardson_dev_abs` | `0.0061` | `0.0056` | — |
| `/pairs/1/oracle_distance_abs/+0.0` | `0.0223` | `0.0101` | — |
| `/pairs/1/oracle_distance_abs/+0.5` | `0.0048` | `0.0321` | — |
| `/pairs/1/oracle_distance_abs/+1.0` | `0.0257` | `0.0536` | — |
| `/pairs/1/oracle_distance_abs/-0.5` | `0.0429` | `0.0121` | — |
| `/pairs/1/oracle_distance_abs/-1.0` | `0.0638` | `0.0342` | — |
| `/pairs/2/committed_fine_gap_abs` | `0.0097` | `0.0034` | — |
| `/pairs/2/fine_gate_abs` | `0.015` | `0.006` | — |
| `/pairs/2/nearest_offset_fine_cells` | `0.5` | `0.0` | — |
| `/pairs/2/one_cell_defect/over/fine_gap_abs` | `0.0265` | `0.0134` | — |
| `/pairs/2/one_cell_defect/over/fine_margin_x` | `1.77` | `2.233` | — |
| `/pairs/2/one_cell_defect/over/richardson_dev_abs` | `0.003` | `0.0027` | — |
| `/pairs/2/one_cell_defect/under/detected_by_fine_gate` | `false` | `true` | — |
| `/pairs/2/one_cell_defect/under/fine_gap_abs` | `0.0035` | `0.0167` | — |
| `/pairs/2/one_cell_defect/under/fine_margin_x` | `0.235` | `2.778` | — |
| `/pairs/2/one_cell_defect/under/richardson_dev_abs` | `0.0019` | `0.0016` | — |
| `/pairs/2/one_cell_defect/under/scores_better_than_undefected` | `true` | `false` | — |
| `/pairs/2/oracle_distance_abs/+0.0` | `0.0097` | `0.0034` | — |
| `/pairs/2/oracle_distance_abs/+0.5` | `0.0018` | `0.0115` | — |
| `/pairs/2/oracle_distance_abs/+1.0` | `0.0071` | `0.0203` | — |
| `/pairs/2/oracle_distance_abs/-0.5` | `0.0167` | `0.0036` | — |
| `/pairs/2/oracle_distance_abs/-1.0` | `0.0229` | `0.0098` | — |
| `/pairs/3/committed_fine_gap_abs` | `0.0145` | `0.0102` | — |
| `/pairs/3/fine_gate_abs` | `0.022` | `0.016` | — |
| `/pairs/3/nearest_offset_fine_cells` | `0.5` | `0.0` | — |
| `/pairs/3/one_cell_defect/over/fine_gap_abs` | `0.0458` | `0.026` | — |
| `/pairs/3/one_cell_defect/over/fine_margin_x` | `2.084` | `1.623` | — |
| `/pairs/3/one_cell_defect/over/richardson_dev_abs` | `0.0064` | `0.0063` | — |
| `/pairs/3/one_cell_defect/under/fine_gap_abs` | `0.023` | `0.0417` | — |
| `/pairs/3/one_cell_defect/under/fine_margin_x` | `1.045` | `2.608` | — |
| `/pairs/3/one_cell_defect/under/richardson_dev_abs` | `0.0058` | `0.0056` | — |
| `/pairs/3/oracle_distance_abs/+0.0` | `0.0145` | `0.0102` | — |
| `/pairs/3/oracle_distance_abs/+0.5` | `0.0075` | `0.024` | — |
| `/pairs/3/oracle_distance_abs/+1.0` | `0.0209` | `0.0395` | — |
| `/pairs/3/oracle_distance_abs/-0.5` | `0.0306` | `0.0111` | — |
| `/pairs/3/oracle_distance_abs/-1.0` | `0.0481` | `0.0282` | — |
| `/pairs/4/committed_fine_gap_abs` | `0.0232` | `0.0106` | — |
| `/pairs/4/fine_gate_abs` | `0.035` | `0.016` | — |
| `/pairs/4/nearest_offset_fine_cells` | `0.5` | `0.0` | — |
| `/pairs/4/one_cell_defect/over/fine_gap_abs` | `0.0649` | `0.0337` | — |
| `/pairs/4/one_cell_defect/over/fine_margin_x` | `1.855` | `2.108` | — |
| `/pairs/4/one_cell_defect/over/richardson_dev_abs` | `0.0066` | `0.0062` | — |
| `/pairs/4/one_cell_defect/under/detected_by_fine_gate` | `false` | `true` | — |
| `/pairs/4/one_cell_defect/under/fine_gap_abs` | `0.027` | `0.054` | — |
| `/pairs/4/one_cell_defect/under/fine_margin_x` | `0.771` | `3.373` | — |
| `/pairs/4/one_cell_defect/under/richardson_dev_abs` | `0.006` | `0.0056` | — |
| `/pairs/4/oracle_distance_abs/+0.0` | `0.0232` | `0.0106` | — |
| `/pairs/4/oracle_distance_abs/+0.5` | `0.0049` | `0.0321` | — |
| `/pairs/4/oracle_distance_abs/+1.0` | `0.0258` | `0.0536` | — |
| `/pairs/4/oracle_distance_abs/-0.5` | `0.0433` | `0.0122` | — |
| `/pairs/4/oracle_distance_abs/-1.0` | `0.0639` | `0.0341` | — |
| `/pairs/5/committed_fine_gap_abs` | `0.0097` | `0.0035` | — |
| `/pairs/5/fine_gate_abs` | `0.015` | `0.006` | — |
| `/pairs/5/nearest_offset_fine_cells` | `0.5` | `0.0` | — |
| `/pairs/5/one_cell_defect/over/fine_gap_abs` | `0.0265` | `0.0134` | — |
| `/pairs/5/one_cell_defect/over/fine_margin_x` | `1.767` | `2.228` | — |
| `/pairs/5/one_cell_defect/over/richardson_dev_abs` | `0.003` | `0.0027` | — |
| `/pairs/5/one_cell_defect/under/detected_by_fine_gate` | `false` | `true` | — |
| `/pairs/5/one_cell_defect/under/fine_gap_abs` | `0.0036` | `0.0167` | — |
| `/pairs/5/one_cell_defect/under/fine_margin_x` | `0.237` | `2.783` | — |
| `/pairs/5/one_cell_defect/under/richardson_dev_abs` | `0.0018` | `0.0016` | — |
| `/pairs/5/one_cell_defect/under/scores_better_than_undefected` | `true` | `false` | — |
| `/pairs/5/oracle_distance_abs/+0.0` | `0.0097` | `0.0035` | — |
| `/pairs/5/oracle_distance_abs/+0.5` | `0.0018` | `0.0115` | — |
| `/pairs/5/oracle_distance_abs/+1.0` | `0.0072` | `0.0203` | — |
| `/pairs/5/oracle_distance_abs/-0.5` | `0.0167` | `0.0036` | — |
| `/pairs/5/oracle_distance_abs/-1.0` | `0.0229` | `0.0098` | — |
| `/pairs/6/committed_fine_gap_abs` | `0.0222` | `0.01` | — |
| `/pairs/6/fine_gate_abs` | `0.034` | `0.015` | — |
| `/pairs/6/nearest_offset_fine_cells` | `0.5` | `0.0` | — |
| `/pairs/6/one_cell_defect/over/fine_gap_abs` | `0.0647` | `0.0335` | — |
| `/pairs/6/one_cell_defect/over/fine_margin_x` | `1.903` | `2.236` | — |
| `/pairs/6/one_cell_defect/over/richardson_dev_abs` | `0.0067` | `0.0062` | — |
| `/pairs/6/one_cell_defect/under/detected_by_fine_gate` | `false` | `true` | — |
| `/pairs/6/one_cell_defect/under/fine_gap_abs` | `0.0269` | `0.0538` | — |
| `/pairs/6/one_cell_defect/under/fine_margin_x` | `0.79` | `3.585` | — |
| `/pairs/6/one_cell_defect/under/richardson_dev_abs` | `0.0061` | `0.0057` | — |
| `/pairs/6/oracle_distance_abs/+0.0` | `0.0222` | `0.01` | — |
| `/pairs/6/oracle_distance_abs/+0.5` | `0.0048` | `0.0321` | — |
| `/pairs/6/oracle_distance_abs/+1.0` | `0.0257` | `0.0536` | — |
| `/pairs/6/oracle_distance_abs/-0.5` | `0.0429` | `0.0122` | — |
| `/pairs/6/oracle_distance_abs/-1.0` | `0.0639` | `0.0342` | — |
| `/pairs/7/committed_fine_gap_abs` | `0.0222` | `0.01` | — |
| `/pairs/7/fine_gate_abs` | `0.034` | `0.015` | — |
| `/pairs/7/nearest_offset_fine_cells` | `0.5` | `0.0` | — |
| `/pairs/7/one_cell_defect/over/fine_gap_abs` | `0.0647` | `0.0336` | — |
| `/pairs/7/one_cell_defect/over/fine_margin_x` | `1.902` | `2.24` | — |
| `/pairs/7/one_cell_defect/over/richardson_dev_abs` | `0.0067` | `0.0063` | — |
| `/pairs/7/one_cell_defect/under/detected_by_fine_gate` | `false` | `true` | — |
| `/pairs/7/one_cell_defect/under/fine_gap_abs` | `0.0268` | `0.0538` | — |
| `/pairs/7/one_cell_defect/under/fine_margin_x` | `0.788` | `3.587` | — |
| `/pairs/7/one_cell_defect/under/richardson_dev_abs` | `0.0061` | `0.0058` | — |
| `/pairs/7/oracle_distance_abs/+0.0` | `0.0222` | `0.01` | — |
| `/pairs/7/oracle_distance_abs/+0.5` | `0.0047` | `0.032` | — |
| `/pairs/7/oracle_distance_abs/+1.0` | `0.0256` | `0.0535` | — |
| `/pairs/7/oracle_distance_abs/-0.5` | `0.0429` | `0.0122` | — |
| `/pairs/7/oracle_distance_abs/-1.0` | `0.0638` | `0.0342` | — |
| `/summary/nearest_offset_fine_cells_values` | `[0.5]` | `[0.0]` | 0 |
| `/summary/nearest_offset_is_positive_at_all_pairs` | `true` | `false` | — |
| `/summary/over_aperture_min_margin_x` | `1.767` | `1.623` | — |
| `/summary/under_aperture_detected` | `2` | `8` | — |
| `/summary/under_aperture_detected_configs` | `["18.288&#124;0.20&#124;0.50", "18.288&#124;0.20&#124;0.42"]` | array[8], SHA256 `3d12a67f8a33` | 1–7 |
| `/summary/under_aperture_max_margin_x` | `1.185` | `3.587` | — |
| `/summary/under_aperture_min_margin_x` | *absent* | `2.608` | — |
| `/summary/under_aperture_scores_better_configs` | `["7.620&#124;0.20&#124;0.50", "7.620&#124;0.20&#124;0.42"]` | `[]` | 0–1 |

## validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/config/fine_gate_abs_per_config` | `0.015` | `0.006` | — |
| `/config/pooled_fine_gate_abs` | `0.04` | `0.02` | — |
| `/measured/fine_gap_abs` | `0.02842` | `0.01246` | — |
| `/measured/per_config_margin_x` | `1.895` | `2.077` | — |
| `/measured/richardson_dev_abs` | `0.00588` | `7e-05` | — |
| `/rows/0/aperture_wall_nodes` | *absent* | `[20, 41]` | 0–1 |
| `/rows/0/iris_wall_nodes` | *absent* | `[380, 384]` | 0–1 |
| `/rows/0/nominal_aperture_cells` | *absent* | `20` | — |
| `/rows/0/nominal_aperture_nodes` | `19` | *absent* | — |
| `/rows/0/realized_aperture_cells` | *absent* | `21` | — |
| `/rows/0/realized_aperture_nodes` | `20` | *absent* | — |
| `/rows/0/realized_thickness_cells` | *absent* | `4` | — |
| `/rows/0/s11` | array[29], SHA256 `d58528f8dea5` | array[29], SHA256 `85c03ea8b60e` | 0–28 |
| `/rows/0/s21` | array[29], SHA256 `0244ab7dacab` | array[29], SHA256 `c641ad393ab6` | 0–28 |
| `/rows/0/thickness_cells` | `4` | *absent* | — |
| `/rows/0/wall_s` | `882.5` | `491.6` | — |
| `/rows/1/aperture_wall_nodes` | *absent* | `[10, 21]` | 0–1 |
| `/rows/1/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/rows/1/nominal_aperture_cells` | *absent* | `10` | — |
| `/rows/1/nominal_aperture_nodes` | `9` | *absent* | — |
| `/rows/1/realized_aperture_cells` | *absent* | `11` | — |
| `/rows/1/realized_aperture_nodes` | `10` | *absent* | — |
| `/rows/1/realized_thickness_cells` | *absent* | `2` | — |
| `/rows/1/s11` | array[29], SHA256 `e28a8927cbe2` | array[29], SHA256 `61ee7229ccf6` | 0–28 |
| `/rows/1/s21` | array[29], SHA256 `27dfe939596d` | array[29], SHA256 `2aa69b6f7e39` | 0–28 |
| `/rows/1/thickness_cells` | `2` | *absent* | — |
| `/rows/1/wall_s` | `73.9` | `36.4` | — |

## validation/crossval/_18_wr90_iris_results/rfx.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/claim_scope` | `"One symmetric inductive PEC iris (t = 1.524 mm = exactly 2 coarse / 4 fine cells; apertures 18.288/12.192/7.62 mm, grid-exact) in WR-90 over 8.2-12.4 GHz on 29 frequency points, flux-normalized &#124;S11&#124; vs a twice-implemented TEn0 mode-matching cascade oracle (self-witnesses: unitarity 1.1e-16, mode convergence 4.3e-5, Marcuvitz cot^2 thin-limit anchor 10.8% with the inductive sign, d->a and deep-constriction limits; the PR #480 review reproduced the oracle with a formulation-independent 2-D H-plane FDFD to 6e-4 and measured rfx's same-geometry agreement at <= 0.02 — attributed, not imported). GATED: fine rung dx = a/60 within 0.04 abs = round-up(measured envelope 0.0232 x 1.5) over 8 configs (3 apertures x {centred, iris off-centre at 0.42 of the guide} + 2 extra guide lengths; every config lands within 0.023, so no single configuration sets the envelope), and the Richardson extrapolation 2*S(a/60) - S(a/30) on the oracle within 0.01 abs (envelope 0.0051) at EVERY one of those 8 pairs, which cross-confirms the oracle and the first-order attribution (gap ratios 0.527-0.604 = textbook first order). REPORTED, NOT GATED: the coarse rung dx = a/30 (0.018-0.043 abs); the raw normalize=False record, which is WORSE than flux (gaps 0.021-0.054) with a pointwise &#124;raw - flux&#124; difference up to 0.033 at the wide aperture; residual detrended ripple, i.e. a quadratic detrend of &#124;S11&#124; MINUS the oracle so the oracle's own curvature is not counted (fine <= 0.0077, coarse <= 0.0158, both at the wide aperture with the iris off-centre, down from the 0.0706 the PR #480 review measured on the same basis before the absorber fix); and phase (magnitude-only lane posture). FENCED, never gated: everything beyond ONE symmetric inductive iris — multi-iris filters, posts, septa and off-centre apertures stay EXPERIMENTAL per docs/guides/support_matrix.md. THREE SETUP DEFECTS were found during this campaign, each having corrupted an earlier revision's numbers and each now fenced by an assert or a derived setting: (1) a parasitic wall-slot (fins drawn to the NOMINAL guide width leave a 1-cell gap at the actual grid wall); (2) node-plane box corners are half-ulp fragile because the volume mask is half-open over NODE coordinates — one fine config rasterized 3 thickness nodes instead of 4, and an apparent +/-0.07 'domain sensitivity' was that ulp lottery; (3) the fin footprint made the ELECTRICAL aperture d + 2*dx instead of d, which alone inflated the envelope 4-6x, and a 0.5*lambda_g absorber left the envelope set by CPML reflection rather than discretization (PR #480 review B2/B3; CPML is now 0.75*lambda_g at the band edge = 60 coarse / 120 fine). RETRACTED: an earlier revision fenced normalize=True modal extraction on the strength of a measured column power 1.112-1.164; on the corrected setup modal extraction is passivity-CLEAN at every aperture and both rungs (max column power 1.0207 at d = 7.62 mm / a-30, 1.0013 at d = 18.288 mm, ZERO extractor warnings), so that non-passivity was a symptom of defects (1)-(3) and not a reflector-inflation property of the extractor — the fence is withdrawn and the measurement is committed as modal_extraction_witness, which also records modal ACCURACY so the retraction does not rest on passivity alone: modal &#124;S11&#124; gaps come out comparable to flux and consistently a little worse, which is why flux still carries the gate. Palace WavePort corroboration (stage S2) and a published multi-iris filter (stage S3) are follow-on stages, not claimed here. APERTURE RESOLUTION (issue #812 re-gate, 2026-09-01): the fine gate is now per-CONFIGURATION -- gate = round-up(that configuration's own committed envelope x 1.5) at quantum 1000, giving 0.019/0.034/0.015/0.022/0.035/0.015/0.034/0.034 for the eight configs -- because the pooled 0.04 was set by the worst configuration and then spent at all eight. Measured against those gates, a one-cell aperture error at each rung (the smallest the grid-snapped geometry can express, and the campaign's own setup defect (3) at half its size) is detected as an OVER-aperture at every configuration and as an UNDER-aperture at only two of the eight, both at the weak aperture and both below the repo's own 1.5x margin. Those counts, the margins and the per-configuration oracle distances are COMMITTED rather than restated here: validation/crossval/_18_wr90_iris_results/aperture_resolution.json, keys summary.over_aperture_detected, summary.over_aperture_min_margin_x, summary.under_aperture_detected, summary.under_aperture_detected_configs and summary.under_aperture_max_margin_x, with the per-configuration rows under pairs[*] (pairs[2] is d = 7.620 mm centred); each one is re-derived from the committed traces by an INDEPENDENT oracle in tests/crossval/test_wr90_iris_modematch_gates.py. A one-cell under-aperture is therefore NOT resolved with margin at any configuration, and at d = 12.192 and d = 7.620 mm it is not resolved at all: at both d = 7.620 configurations the modelled under-aperture defect scores BETTER than the undefected committed row (summary.under_aperture_scores_better_configs; pairs[*].one_cell_defect.under.scores_better_than_undefected), because the fine rung's own staircase error is an effective aperture WIDER than nominal rather than narrower -- over the declared offset grid the committed fine trace's NEAREST oracle sits at d PLUS half a fine cell at all eight configurations (summary.nearest_offset_fine_cells_values; pairs[*].oracle_distance_abs) -- so narrowing the geometry by one cell moves it TOWARD the trace instead of away. CORRECTION (issue #812 round 2): an earlier revision of this paragraph asserted the opposite sign, that the committed fine trace sat closer to the oracle one fine cell NARROW, quoting the under-aperture DEFECT metric as if it were that distance; that claim was mis-sourced and sign-inverted, and aperture_resolution.json is now the only source for this class. The Richardson witness is blind to this whole class in both signs at all eight configurations BY CONSTRUCTION: an aperture error of one cell at each rung is proportional to dx, which is exactly what 2*S(a/60) - S(a/30) is built to remove, so no tightening of its 0.01 gate can catch it and none is attempted. The calibration this case supplies to any downstream multi-iris filter is aperture-resolved to +1 fine cell, NOT to -1. The three declared apertures are now pinned as claims (G18-C): each must be an exact and EVEN integer cell count at BOTH rungs, a geometric condition no one-fine-cell relabel can satisfy."` | `"One symmetric inductive PEC iris (t = 1.524 mm = exactly 2 coarse / 4 fine cells; apertures 18.288/12.192/7.62 mm, grid-exact) in WR-90 over 8.2-12.4 GHz on 29 frequency points, flux-normalized &#124;S11&#124; vs a twice-implemented TEn0 mode-matching cascade oracle (self-witnesses: unitarity 1.1e-16, mode convergence 4.3e-5, Marcuvitz cot^2 thin-limit anchor 10.8% with the inductive sign, d->a and deep-constriction limits; the PR #480 review reproduced the oracle with a formulation-independent 2-D H-plane FDFD to 6e-4 and measured rfx's same-geometry agreement at <= 0.02 — attributed, not imported). GATED: fine rung dx = a/60 within 0.02 abs = round-up(measured envelope 0.0106 x 1.5) over 8 configs (3 apertures x {centred, iris off-centre at 0.42 of the guide} + 2 extra guide lengths; every config lands within 0.011, so no single configuration sets the envelope), and the Richardson extrapolation 2*S(a/60) - S(a/30) on the oracle within 0.01 abs (envelope 0.0046) at EVERY one of those 8 pairs, which cross-confirms the oracle and the first-order attribution (gap ratios 0.407-0.440 = textbook first order). REPORTED, NOT GATED: the coarse rung dx = a/30 (0.008-0.025 abs); the raw normalize=False record, which is WORSE than flux (gaps 0.009-0.025) with a pointwise &#124;raw - flux&#124; difference up to 0.0068 at the wide aperture; residual detrended ripple, i.e. a quadratic detrend of &#124;S11&#124; MINUS the oracle so the oracle's own curvature is not counted (fine <= 0.0076, coarse <= 0.0152, both at the wide aperture with the iris off-centre, down from the 0.0706 the PR #480 review measured on the same basis before the absorber fix); and phase (magnitude-only lane posture). FENCED, never gated: everything beyond ONE symmetric inductive iris — multi-iris filters, posts, septa and off-centre apertures stay EXPERIMENTAL per docs/guides/support_matrix.md. THREE SETUP DEFECTS were found during this campaign, each having corrupted an earlier revision's numbers and each now fenced by an assert or a derived setting: (1) a parasitic wall-slot (fins drawn to the NOMINAL guide width leave a 1-cell gap at the actual grid wall); (2) node-plane box corners were half-ulp fragile under the pre-#931 half-open NODE mask — one fine config rasterized 3 thickness nodes instead of 4, and an apparent +/-0.07 'domain sensitivity' was that ulp lottery — so every corner was moved half a cell OFF the node planes. The #931 lattice ownership contract INVERTS that: a PEC volume is sampled at cell CENTRES, so a node-plane corner selects whole cells and is the well-defined position while a half-cell offset lands exactly on a centre. The corners are back on the node planes and the footprint asserts read the realized edge set rather than a sigma mask; (3) the fin footprint made the ELECTRICAL aperture d + 2*dx instead of d, which alone inflated the envelope 4-6x, and a 0.5*lambda_g absorber left the envelope set by CPML reflection rather than discretization (PR #480 review B2/B3; CPML is now 0.75*lambda_g at the band edge = 60 coarse / 120 fine). THE #931 THICKNESS CORRECTION: until the contract landed, this case fed its oracle the drawn t = 1.524 mm while the lattice realized (t_c - 1)*dx — 0.762 mm at a/30 and 1.143 mm at a/60, a 50% / 25% thickness deficit — because a body's far face was never a wall. Nothing in this case measured that: every assert counted MASKED PLANES, a quantity that agreed with the drawing by construction. Under the contract the realized thickness is the drawn thickness and the oracle input is correct for the first time; the whole record was regenerated on the corrected geometry and every gate re-derived from the new envelopes. The case also gains the contract's one-cell volume witness (one_cell_volume_witness): an iris-thickness sweep t = 1..8 cells against the lattice-blind mode-matching oracle, so that a one-cell PEC body standing two walls has an independent check rather than a thin-limit anchor that only speaks about t -> 0. Each rung is asked to IDENTIFY its own thickness -- the oracle at t-1, t and t+1 cells, argmin on t -- and all seven do; at t = 1 the pre-#931 one-wall alternative (a zero-thickness screen) is 4.32x worse than the two-wall one, so the contract's rule at one cell is measured rather than assumed. The witness's first-stated criterion (t = 1 inside the t = 2..8 range) is RETIRED as vacuous and its verdict kept: the residual is monotone in t, so t = 1 is the extremum whatever the physics does, and a perfect 0.0000 would fail it too. RETRACTED: an earlier revision fenced normalize=True modal extraction on the strength of a measured column power 1.112-1.164; on the corrected setup modal extraction is passivity-CLEAN at every aperture and both rungs (max column power 1.0200 at d = 7.62 mm / a-30, 1.0012 at d = 18.288 mm, ZERO extractor warnings), so that non-passivity was a symptom of defects (1)-(3) and not a reflector-inflation property of the extractor — the fence is withdrawn and the measurement is committed as modal_extraction_witness, which also records modal ACCURACY so the retraction does not rest on passivity alone: modal &#124;S11&#124; gaps come out comparable to flux and consistently a little worse, which is why flux still carries the gate. Palace WavePort corroboration (stage S2) and a published multi-iris filter (stage S3) are follow-on stages, not claimed here. APERTURE RESOLUTION (issue #812 re-gate 2026-09-01, RE-MEASURED under #931 2026-09-07): the fine gate is per-CONFIGURATION -- gate = round-up(that configuration's own committed envelope x 1.5) at quantum 1000, giving 0.012/0.016/0.006/0.016/0.016/0.006/0.015/0.015 for the eight configs -- because the pooled gate is set by the worst configuration and then spent at all eight. All eight moved DOWN when the thickness deficit closed (0.019/0.034/0.015/0.022/0.035/0.015/0.034/0.034 before it), which is a re-derivation of the same rule on a better geometry, not a re-tuning. Measured against those gates, a one-cell aperture error at each rung (the smallest the grid-snapped geometry can express, and the campaign's own setup defect (3) at half its size) is now detected in BOTH signs at every one of the eight configurations, at worst 1.623x the gate for an over-aperture and 2.608x for an under-aperture. Those counts, the margins and the per-configuration oracle distances are COMMITTED rather than restated here: validation/crossval/_18_wr90_iris_results/aperture_resolution.json, keys summary.over_aperture_detected, summary.over_aperture_min_margin_x, summary.under_aperture_detected, summary.under_aperture_detected_configs, summary.under_aperture_min_margin_x and summary.under_aperture_max_margin_x, with the per-configuration rows under pairs[*] (pairs[2] is d = 7.620 mm centred); each one is re-derived from the committed traces by an INDEPENDENT oracle in tests/crossval/test_wr90_iris_modematch_gates.py. WHAT #931 CHANGED HERE, and it is the whole paragraph: before the contract, the committed fine trace's NEAREST oracle over the declared offset grid sat at d PLUS half a fine cell at all eight configurations, an apparent effective aperture WIDER than nominal; a one-cell under-aperture therefore moved the geometry TOWARD the trace, scored BETTER than the undefected row at both d = 7.620 configurations, and was detected at only two of the eight. That half-cell offset was not an aperture property at all -- it was the thickness deficit ((t_c - 1)*dx instead of t) reading out on the aperture axis, the two being the only free dimensions of a symmetric iris. With the realized thickness equal to the drawn one, the nearest oracle sits at the DECLARED d at all eight configurations (summary.nearest_offset_fine_cells_values == [0.0]), no defect scores better than the undefected row (summary.under_aperture_scores_better_configs == []), and the asymmetry between the two signs is gone. CORRECTION HISTORY (issue #812 round 2): an earlier revision of this paragraph asserted that the committed fine trace sat closer to the oracle one fine cell NARROW, quoting the under-aperture DEFECT metric as if it were that distance; that claim was mis-sourced and sign-inverted, and aperture_resolution.json is the only source for this class. The Richardson witness is blind to this whole class in both signs at all eight configurations BY CONSTRUCTION: an aperture error of one cell at each rung is proportional to dx, which is exactly what 2*S(a/60) - S(a/30) is built to remove, so no tightening of its 0.01 gate can catch it and none is attempted. The calibration this case supplies to any downstream multi-iris filter is aperture-resolved to one fine cell in both signs. The three declared apertures are pinned as claims (G18-C): each must be an exact and EVEN integer cell count at BOTH rungs, a geometric condition no one-fine-cell relabel can satisfy."` | — |
| `/coarse_diagnostic/0/aperture_cells` | `23` | *absent* | — |
| `/coarse_diagnostic/0/aperture_wall_nodes` | *absent* | `[3, 27]` | 0–1 |
| `/coarse_diagnostic/0/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/coarse_diagnostic/0/max_gap_abs` | `0.0202` | `0.0193` | — |
| `/coarse_diagnostic/0/realized_aperture_cells` | *absent* | `24` | — |
| `/coarse_diagnostic/0/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/0/richardson_dev_abs` | `0.0042` | `0.0034` | — |
| `/coarse_diagnostic/0/s11` | array[29], SHA256 `5fd05eebb8c1` | array[29], SHA256 `bda599c3f8c3` | 0–28 |
| `/coarse_diagnostic/0/s21` | array[29], SHA256 `37f78dfc0e6e` | array[29], SHA256 `3d5cbf3112ed` | 0–28 |
| `/coarse_diagnostic/0/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/0/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/0/wall_s` | `66.0` | `25.2` | — |
| `/coarse_diagnostic/1/aperture_cells` | `15` | *absent* | — |
| `/coarse_diagnostic/1/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/coarse_diagnostic/1/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/coarse_diagnostic/1/max_gap_abs` | `0.0405` | `0.0246` | — |
| `/coarse_diagnostic/1/realized_aperture_cells` | *absent* | `16` | — |
| `/coarse_diagnostic/1/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/1/richardson_dev_abs` | `0.005` | `0.0045` | — |
| `/coarse_diagnostic/1/s11` | array[29], SHA256 `cf29bc66aa1e` | array[29], SHA256 `40f1fb558306` | 0–28 |
| `/coarse_diagnostic/1/s21` | array[29], SHA256 `f3bfa41f708a` | array[29], SHA256 `399c1f6476a8` | 0–28 |
| `/coarse_diagnostic/1/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/1/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/1/wall_s` | `65.8` | `27.7` | — |
| `/coarse_diagnostic/2/aperture_cells` | `9` | *absent* | — |
| `/coarse_diagnostic/2/aperture_wall_nodes` | *absent* | `[10, 20]` | 0–1 |
| `/coarse_diagnostic/2/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/coarse_diagnostic/2/max_gap_abs` | `0.0184` | `0.0081` | — |
| `/coarse_diagnostic/2/realized_aperture_cells` | *absent* | `10` | — |
| `/coarse_diagnostic/2/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/2/richardson_dev_abs` | `0.001` | `0.0012` | — |
| `/coarse_diagnostic/2/s11` | array[29], SHA256 `5a94f9871f44` | array[29], SHA256 `bd791aada6ca` | 0–28 |
| `/coarse_diagnostic/2/s21` | array[29], SHA256 `a9b6858dcc87` | array[29], SHA256 `5940957773af` | 0–28 |
| `/coarse_diagnostic/2/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/2/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/2/wall_s` | `65.6` | `27.6` | — |
| `/coarse_diagnostic/3/aperture_cells` | `23` | *absent* | — |
| `/coarse_diagnostic/3/aperture_wall_nodes` | *absent* | `[3, 27]` | 0–1 |
| `/coarse_diagnostic/3/iris_wall_nodes` | *absent* | `[169, 171]` | 0–1 |
| `/coarse_diagnostic/3/max_gap_abs` | `0.0256` | `0.0232` | — |
| `/coarse_diagnostic/3/realized_aperture_cells` | *absent* | `24` | — |
| `/coarse_diagnostic/3/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/3/richardson_dev_abs` | `0.0039` | `0.0036` | — |
| `/coarse_diagnostic/3/s11` | array[29], SHA256 `5f2b042f96e4` | array[29], SHA256 `9f851d1128f2` | 0–28 |
| `/coarse_diagnostic/3/s21` | array[29], SHA256 `682abde4f3cf` | array[29], SHA256 `695c9f6244fb` | 0–28 |
| `/coarse_diagnostic/3/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/3/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/3/wall_s` | `67.4` | `25.7` | — |
| `/coarse_diagnostic/4/aperture_cells` | `15` | *absent* | — |
| `/coarse_diagnostic/4/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/coarse_diagnostic/4/iris_wall_nodes` | *absent* | `[169, 171]` | 0–1 |
| `/coarse_diagnostic/4/max_gap_abs` | `0.043` | `0.0252` | — |
| `/coarse_diagnostic/4/realized_aperture_cells` | *absent* | `16` | — |
| `/coarse_diagnostic/4/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/4/richardson_dev_abs` | `0.005` | `0.0045` | — |
| `/coarse_diagnostic/4/s11` | array[29], SHA256 `6eecae5815e2` | array[29], SHA256 `b7ecfa16b9ea` | 0–28 |
| `/coarse_diagnostic/4/s21` | array[29], SHA256 `4fe7c4d61ef5` | array[29], SHA256 `2e8f99f77179` | 0–28 |
| `/coarse_diagnostic/4/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/4/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/4/wall_s` | `64.7` | `26.2` | — |
| `/coarse_diagnostic/5/aperture_cells` | `9` | *absent* | — |
| `/coarse_diagnostic/5/aperture_wall_nodes` | *absent* | `[10, 20]` | 0–1 |
| `/coarse_diagnostic/5/iris_wall_nodes` | *absent* | `[169, 171]` | 0–1 |
| `/coarse_diagnostic/5/max_gap_abs` | `0.0183` | `0.0082` | — |
| `/coarse_diagnostic/5/realized_aperture_cells` | *absent* | `10` | — |
| `/coarse_diagnostic/5/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/5/richardson_dev_abs` | `0.001` | `0.0013` | — |
| `/coarse_diagnostic/5/s11` | array[29], SHA256 `6aaa7ef5849a` | array[29], SHA256 `c313a64c04db` | 0–28 |
| `/coarse_diagnostic/5/s21` | array[29], SHA256 `b70219bb9423` | array[29], SHA256 `85c2523747f3` | 0–28 |
| `/coarse_diagnostic/5/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/5/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/5/wall_s` | `63.8` | `24.2` | — |
| `/coarse_diagnostic/6/aperture_cells` | `15` | *absent* | — |
| `/coarse_diagnostic/6/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/coarse_diagnostic/6/iris_wall_nodes` | *absent* | `[164, 166]` | 0–1 |
| `/coarse_diagnostic/6/max_gap_abs` | `0.0405` | `0.0246` | — |
| `/coarse_diagnostic/6/realized_aperture_cells` | *absent* | `16` | — |
| `/coarse_diagnostic/6/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/6/richardson_dev_abs` | `0.0051` | `0.0045` | — |
| `/coarse_diagnostic/6/s11` | array[29], SHA256 `aa9fb71568aa` | array[29], SHA256 `77ba4951bc5a` | 0–28 |
| `/coarse_diagnostic/6/s21` | array[29], SHA256 `d693a294d6f1` | array[29], SHA256 `f6a127703d6d` | 0–28 |
| `/coarse_diagnostic/6/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/6/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/6/wall_s` | `60.5` | `17.5` | — |
| `/coarse_diagnostic/7/aperture_cells` | `15` | *absent* | — |
| `/coarse_diagnostic/7/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/coarse_diagnostic/7/iris_wall_nodes` | *absent* | `[217, 219]` | 0–1 |
| `/coarse_diagnostic/7/max_gap_abs` | `0.0401` | `0.0245` | — |
| `/coarse_diagnostic/7/realized_aperture_cells` | *absent* | `16` | — |
| `/coarse_diagnostic/7/realized_thickness_cells` | *absent* | `2` | — |
| `/coarse_diagnostic/7/richardson_dev_abs` | `0.005` | `0.0046` | — |
| `/coarse_diagnostic/7/s11` | array[29], SHA256 `510999ea23ac` | array[29], SHA256 `74c027fbec56` | 0–28 |
| `/coarse_diagnostic/7/s21` | array[29], SHA256 `ce37344265a9` | array[29], SHA256 `1ce987a21f76` | 0–28 |
| `/coarse_diagnostic/7/t_mm` | *absent* | `1.524` | — |
| `/coarse_diagnostic/7/thickness_cells` | `2` | *absent* | — |
| `/coarse_diagnostic/7/wall_s` | `71.1` | `28.4` | — |
| `/gated_fine/0/aperture_cells` | `47` | *absent* | — |
| `/gated_fine/0/aperture_wall_nodes` | *absent* | `[6, 54]` | 0–1 |
| `/gated_fine/0/fine_gate_abs` | *absent* | `0.012` | — |
| `/gated_fine/0/iris_wall_nodes` | *absent* | `[380, 384]` | 0–1 |
| `/gated_fine/0/max_gap_abs` | `0.0122` | `0.0079` | — |
| `/gated_fine/0/realized_aperture_cells` | *absent* | `48` | — |
| `/gated_fine/0/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/0/s11` | array[29], SHA256 `305f4b231b58` | array[29], SHA256 `54c1a47f3087` | 0–28 |
| `/gated_fine/0/s21` | array[29], SHA256 `8cd9fc99a011` | array[29], SHA256 `a7a0f4c88505` | 0–28 |
| `/gated_fine/0/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/0/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/0/wall_s` | `823.4` | `424.3` | — |
| `/gated_fine/1/aperture_cells` | `31` | *absent* | — |
| `/gated_fine/1/aperture_wall_nodes` | *absent* | `[14, 46]` | 0–1 |
| `/gated_fine/1/fine_gate_abs` | *absent* | `0.016` | — |
| `/gated_fine/1/iris_wall_nodes` | *absent* | `[380, 384]` | 0–1 |
| `/gated_fine/1/max_gap_abs` | `0.0223` | `0.0101` | — |
| `/gated_fine/1/realized_aperture_cells` | *absent* | `32` | — |
| `/gated_fine/1/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/1/s11` | array[29], SHA256 `594f86e4ae94` | array[29], SHA256 `2dd8dd57effe` | 0–28 |
| `/gated_fine/1/s21` | array[29], SHA256 `f9a9b7e47dd7` | array[29], SHA256 `878ceed4b42c` | 0–28 |
| `/gated_fine/1/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/1/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/1/wall_s` | `766.5` | `418.9` | — |
| `/gated_fine/2/aperture_cells` | `19` | *absent* | — |
| `/gated_fine/2/aperture_wall_nodes` | *absent* | `[20, 40]` | 0–1 |
| `/gated_fine/2/fine_gate_abs` | *absent* | `0.006` | — |
| `/gated_fine/2/iris_wall_nodes` | *absent* | `[380, 384]` | 0–1 |
| `/gated_fine/2/max_gap_abs` | `0.0097` | `0.0034` | — |
| `/gated_fine/2/realized_aperture_cells` | *absent* | `20` | — |
| `/gated_fine/2/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/2/s11` | array[29], SHA256 `5ac0f4d88e80` | array[29], SHA256 `e4635c6aa431` | 0–28 |
| `/gated_fine/2/s21` | array[29], SHA256 `0dc81ce656cb` | array[29], SHA256 `066b30fd703b` | 0–28 |
| `/gated_fine/2/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/2/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/2/wall_s` | `756.3` | `415.9` | — |
| `/gated_fine/3/aperture_cells` | `47` | *absent* | — |
| `/gated_fine/3/aperture_wall_nodes` | *absent* | `[6, 54]` | 0–1 |
| `/gated_fine/3/fine_gate_abs` | *absent* | `0.016` | — |
| `/gated_fine/3/iris_wall_nodes` | *absent* | `[338, 342]` | 0–1 |
| `/gated_fine/3/max_gap_abs` | `0.0145` | `0.0102` | — |
| `/gated_fine/3/realized_aperture_cells` | *absent* | `48` | — |
| `/gated_fine/3/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/3/s11` | array[29], SHA256 `b5fc974d7452` | array[29], SHA256 `a56501b6921f` | 0–28 |
| `/gated_fine/3/s21` | array[29], SHA256 `69dc06b62af9` | array[29], SHA256 `acb5ce12a592` | 0–28 |
| `/gated_fine/3/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/3/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/3/wall_s` | `761.4` | `413.7` | — |
| `/gated_fine/4/aperture_cells` | `31` | *absent* | — |
| `/gated_fine/4/aperture_wall_nodes` | *absent* | `[14, 46]` | 0–1 |
| `/gated_fine/4/fine_gate_abs` | *absent* | `0.016` | — |
| `/gated_fine/4/iris_wall_nodes` | *absent* | `[338, 342]` | 0–1 |
| `/gated_fine/4/max_gap_abs` | `0.0232` | `0.0106` | — |
| `/gated_fine/4/realized_aperture_cells` | *absent* | `32` | — |
| `/gated_fine/4/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/4/s11` | array[29], SHA256 `2c6df57b63d9` | array[29], SHA256 `09176314a5e0` | 0–28 |
| `/gated_fine/4/s21` | array[29], SHA256 `07c6295d6743` | array[29], SHA256 `5b3d74897c05` | 0–28 |
| `/gated_fine/4/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/4/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/4/wall_s` | `758.2` | `419.7` | — |
| `/gated_fine/5/aperture_cells` | `19` | *absent* | — |
| `/gated_fine/5/aperture_wall_nodes` | *absent* | `[20, 40]` | 0–1 |
| `/gated_fine/5/fine_gate_abs` | *absent* | `0.006` | — |
| `/gated_fine/5/iris_wall_nodes` | *absent* | `[338, 342]` | 0–1 |
| `/gated_fine/5/max_gap_abs` | `0.0097` | `0.0035` | — |
| `/gated_fine/5/realized_aperture_cells` | *absent* | `20` | — |
| `/gated_fine/5/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/5/s11` | array[29], SHA256 `9ae33598b24f` | array[29], SHA256 `df07ed9053e2` | 0–28 |
| `/gated_fine/5/s21` | array[29], SHA256 `04331e70b6f6` | array[29], SHA256 `9c341ca9ec7f` | 0–28 |
| `/gated_fine/5/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/5/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/5/wall_s` | `757.7` | `427.0` | — |
| `/gated_fine/6/aperture_cells` | `31` | *absent* | — |
| `/gated_fine/6/aperture_wall_nodes` | *absent* | `[14, 46]` | 0–1 |
| `/gated_fine/6/fine_gate_abs` | *absent* | `0.015` | — |
| `/gated_fine/6/iris_wall_nodes` | *absent* | `[328, 332]` | 0–1 |
| `/gated_fine/6/max_gap_abs` | `0.0222` | `0.01` | — |
| `/gated_fine/6/realized_aperture_cells` | *absent* | `32` | — |
| `/gated_fine/6/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/6/s11` | array[29], SHA256 `d9e5237bd0de` | array[29], SHA256 `d797aca93eb9` | 0–28 |
| `/gated_fine/6/s21` | array[29], SHA256 `2af47ec230c3` | array[29], SHA256 `b2cc66f58bab` | 0–28 |
| `/gated_fine/6/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/6/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/6/wall_s` | `696.4` | `352.1` | — |
| `/gated_fine/7/aperture_cells` | `31` | *absent* | — |
| `/gated_fine/7/aperture_wall_nodes` | *absent* | `[14, 46]` | 0–1 |
| `/gated_fine/7/fine_gate_abs` | *absent* | `0.015` | — |
| `/gated_fine/7/iris_wall_nodes` | *absent* | `[433, 437]` | 0–1 |
| `/gated_fine/7/max_gap_abs` | `0.0222` | `0.01` | — |
| `/gated_fine/7/realized_aperture_cells` | *absent* | `32` | — |
| `/gated_fine/7/realized_thickness_cells` | *absent* | `4` | — |
| `/gated_fine/7/s11` | array[29], SHA256 `96c7aa45fd71` | array[29], SHA256 `e52f4b46c7c6` | 0–28 |
| `/gated_fine/7/s21` | array[29], SHA256 `0a0dea4e5cdd` | array[29], SHA256 `fe9a2f7fc9ed` | 0–28 |
| `/gated_fine/7/t_mm` | *absent* | `1.524` | — |
| `/gated_fine/7/thickness_cells` | `4` | *absent* | — |
| `/gated_fine/7/wall_s` | `840.7` | `496.3` | — |
| `/gates/fine_gate_abs` | `0.04` | `0.02` | — |
| `/gates/fine_gate_abs_per_config/12.192|0.16|0.50` | `0.034` | `0.015` | — |
| `/gates/fine_gate_abs_per_config/12.192|0.20|0.42` | `0.035` | `0.016` | — |
| `/gates/fine_gate_abs_per_config/12.192|0.20|0.50` | `0.034` | `0.016` | — |
| `/gates/fine_gate_abs_per_config/12.192|0.24|0.50` | `0.034` | `0.015` | — |
| `/gates/fine_gate_abs_per_config/18.288|0.20|0.42` | `0.022` | `0.016` | — |
| `/gates/fine_gate_abs_per_config/18.288|0.20|0.50` | `0.019` | `0.012` | — |
| `/gates/fine_gate_abs_per_config/7.620|0.20|0.42` | `0.015` | `0.006` | — |
| `/gates/fine_gate_abs_per_config/7.620|0.20|0.50` | `0.015` | `0.006` | — |
| `/gates/fine_measured_envelope_abs` | `0.0232` | `0.0106` | — |
| `/gates/first_order_ratios` | array[8], SHA256 `bf6e482da00f` | array[8], SHA256 `3ea7b0725f7d` | 0–7 |
| `/gates/posture` | `"gate = round-UP(measured envelope x 1.5), enforced as EXACT equality by the write-fixture self-check (PR #475 convention, PR #480 tightening); coarse rung, raw extraction, ripple and phase are reported, never gated; modal extraction is no longer fenced (retracted, see provenance) but structures beyond one symmetric inductive iris remain fenced, never gated; issue #812 re-gate: the BINDING fine gate is now per-configuration, gate = round-UP(that configuration's own envelope x 1.5) at quantum 1000, with the pooled 0.04 retained unchanged as a ceiling and the one-cell aperture detection table gated as its own claim"` | `"gate = round-UP(measured envelope x 1.5), enforced as EXACT equality by the write-fixture self-check (PR #475 convention, PR #480 tightening); coarse rung, raw extraction, ripple and phase are reported, never gated; modal extraction is no longer fenced (retracted, see provenance) but structures beyond one symmetric inductive iris remain fenced, never gated; issue #812 re-gate: the BINDING fine gate is now per-configuration, gate = round-UP(that configuration's own envelope x 1.5) at quantum 1000, with the pooled 0.02 kept as a ceiling and the one-cell aperture detection table gated as its own claim; #931: pooled 0.04 -> 0.02 and all eight per-config gates re-derived DOWN from the corrected-thickness envelopes (VESSL 369367259159), never widened"` | — |
| `/gates/richardson_measured_envelope_abs` | `0.0051` | `0.0046` | — |
| `/modal_extraction_witness/rows/0/max_colpow` | `1.0207` | `1.02` | — |
| `/modal_extraction_witness/rows/0/max_gap_abs` | `0.0189` | `0.0119` | — |
| `/modal_extraction_witness/rows/0/s11` | array[29], SHA256 `bff29b1b2dfe` | array[29], SHA256 `d1a5d8f68e37` | 0–28 |
| `/modal_extraction_witness/rows/1/max_colpow` | `1.0101` | `1.0099` | — |
| `/modal_extraction_witness/rows/1/max_gap_abs` | `0.0099` | `0.0059` | — |
| `/modal_extraction_witness/rows/1/s11` | array[29], SHA256 `63f5e974bfc1` | array[29], SHA256 `db46109e24b7` | 0–28 |
| `/modal_extraction_witness/rows/2/max_colpow` | `1.0144` | `1.015` | — |
| `/modal_extraction_witness/rows/2/max_gap_abs` | `0.046` | `0.0269` | — |
| `/modal_extraction_witness/rows/2/s11` | array[29], SHA256 `721985c5b000` | array[29], SHA256 `210aa5aae51e` | 0–28 |
| `/modal_extraction_witness/rows/3/max_colpow` | `1.0013` | `1.0012` | — |
| `/modal_extraction_witness/rows/3/max_gap_abs` | `0.0234` | `0.019` | — |
| `/modal_extraction_witness/rows/3/s11` | array[29], SHA256 `77bc9ae97010` | array[29], SHA256 `fb2f88671f91` | 0–28 |
| `/one_cell_aperture_detection_witness/0/config` | *absent* | `"18.288&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/0/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/0/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/0/fine_gap_abs` | *absent* | `0.0245` | — |
| `/one_cell_aperture_detection_witness/0/fine_gate_abs` | *absent* | `0.012` | — |
| `/one_cell_aperture_detection_witness/0/richardson_dev_abs` | *absent* | `0.0061` | — |
| `/one_cell_aperture_detection_witness/0/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/1/config` | *absent* | `"18.288&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/1/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/1/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/1/fine_gap_abs` | *absent* | `0.0426` | — |
| `/one_cell_aperture_detection_witness/1/fine_gate_abs` | *absent* | `0.012` | — |
| `/one_cell_aperture_detection_witness/1/richardson_dev_abs` | *absent* | `0.0054` | — |
| `/one_cell_aperture_detection_witness/1/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/2/config` | *absent* | `"12.192&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/2/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/2/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/2/fine_gap_abs` | *absent* | `0.0336` | — |
| `/one_cell_aperture_detection_witness/2/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/2/richardson_dev_abs` | *absent* | `0.0062` | — |
| `/one_cell_aperture_detection_witness/2/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/3/config` | *absent* | `"12.192&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/3/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/3/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/3/fine_gap_abs` | *absent* | `0.0538` | — |
| `/one_cell_aperture_detection_witness/3/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/3/richardson_dev_abs` | *absent* | `0.0056` | — |
| `/one_cell_aperture_detection_witness/3/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/4/config` | *absent* | `"7.620&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/4/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/4/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/4/fine_gap_abs` | *absent* | `0.0134` | — |
| `/one_cell_aperture_detection_witness/4/fine_gate_abs` | *absent* | `0.006` | — |
| `/one_cell_aperture_detection_witness/4/richardson_dev_abs` | *absent* | `0.0027` | — |
| `/one_cell_aperture_detection_witness/4/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/5/config` | *absent* | `"7.620&#124;0.20&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/5/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/5/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/5/fine_gap_abs` | *absent* | `0.0167` | — |
| `/one_cell_aperture_detection_witness/5/fine_gate_abs` | *absent* | `0.006` | — |
| `/one_cell_aperture_detection_witness/5/richardson_dev_abs` | *absent* | `0.0016` | — |
| `/one_cell_aperture_detection_witness/5/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/6/config` | *absent* | `"18.288&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/6/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/6/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/6/fine_gap_abs` | *absent* | `0.026` | — |
| `/one_cell_aperture_detection_witness/6/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/6/richardson_dev_abs` | *absent* | `0.0063` | — |
| `/one_cell_aperture_detection_witness/6/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/7/config` | *absent* | `"18.288&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/7/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/7/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/7/fine_gap_abs` | *absent* | `0.0417` | — |
| `/one_cell_aperture_detection_witness/7/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/7/richardson_dev_abs` | *absent* | `0.0056` | — |
| `/one_cell_aperture_detection_witness/7/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/8/config` | *absent* | `"12.192&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/8/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/8/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/8/fine_gap_abs` | *absent* | `0.0337` | — |
| `/one_cell_aperture_detection_witness/8/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/8/richardson_dev_abs` | *absent* | `0.0061` | — |
| `/one_cell_aperture_detection_witness/8/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/9/config` | *absent* | `"12.192&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/9/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/9/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/9/fine_gap_abs` | *absent* | `0.054` | — |
| `/one_cell_aperture_detection_witness/9/fine_gate_abs` | *absent* | `0.016` | — |
| `/one_cell_aperture_detection_witness/9/richardson_dev_abs` | *absent* | `0.0056` | — |
| `/one_cell_aperture_detection_witness/9/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/10/config` | *absent* | `"7.620&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/10/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/10/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/10/fine_gap_abs` | *absent* | `0.0134` | — |
| `/one_cell_aperture_detection_witness/10/fine_gate_abs` | *absent* | `0.006` | — |
| `/one_cell_aperture_detection_witness/10/richardson_dev_abs` | *absent* | `0.0027` | — |
| `/one_cell_aperture_detection_witness/10/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/11/config` | *absent* | `"7.620&#124;0.20&#124;0.42"` | — |
| `/one_cell_aperture_detection_witness/11/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/11/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/11/fine_gap_abs` | *absent* | `0.0167` | — |
| `/one_cell_aperture_detection_witness/11/fine_gate_abs` | *absent* | `0.006` | — |
| `/one_cell_aperture_detection_witness/11/richardson_dev_abs` | *absent* | `0.0016` | — |
| `/one_cell_aperture_detection_witness/11/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/12/config` | *absent* | `"12.192&#124;0.16&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/12/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/12/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/12/fine_gap_abs` | *absent* | `0.0335` | — |
| `/one_cell_aperture_detection_witness/12/fine_gate_abs` | *absent* | `0.015` | — |
| `/one_cell_aperture_detection_witness/12/richardson_dev_abs` | *absent* | `0.0062` | — |
| `/one_cell_aperture_detection_witness/12/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/13/config` | *absent* | `"12.192&#124;0.16&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/13/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/13/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/13/fine_gap_abs` | *absent* | `0.0538` | — |
| `/one_cell_aperture_detection_witness/13/fine_gate_abs` | *absent* | `0.015` | — |
| `/one_cell_aperture_detection_witness/13/richardson_dev_abs` | *absent* | `0.0057` | — |
| `/one_cell_aperture_detection_witness/13/sign` | *absent* | `-1` | — |
| `/one_cell_aperture_detection_witness/14/config` | *absent* | `"12.192&#124;0.24&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/14/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/14/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/14/fine_gap_abs` | *absent* | `0.0336` | — |
| `/one_cell_aperture_detection_witness/14/fine_gate_abs` | *absent* | `0.015` | — |
| `/one_cell_aperture_detection_witness/14/richardson_dev_abs` | *absent* | `0.0063` | — |
| `/one_cell_aperture_detection_witness/14/sign` | *absent* | `1` | — |
| `/one_cell_aperture_detection_witness/15/config` | *absent* | `"12.192&#124;0.24&#124;0.50"` | — |
| `/one_cell_aperture_detection_witness/15/detected_by_fine_gate` | *absent* | `true` | — |
| `/one_cell_aperture_detection_witness/15/detected_by_richardson_gate` | *absent* | `false` | — |
| `/one_cell_aperture_detection_witness/15/fine_gap_abs` | *absent* | `0.0538` | — |
| `/one_cell_aperture_detection_witness/15/fine_gate_abs` | *absent* | `0.015` | — |
| `/one_cell_aperture_detection_witness/15/richardson_dev_abs` | *absent* | `0.0058` | — |
| `/one_cell_aperture_detection_witness/15/sign` | *absent* | `-1` | — |
| `/one_cell_volume_witness/aperture_mm` | *absent* | `12.192` | — |
| `/one_cell_volume_witness/cells_per_a` | *absent* | `30` | — |
| `/one_cell_volume_witness/identified_every_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/monotone_range_criterion/status` | *absent* | `"RETIRED — vacuous for a monotone residual"` | — |
| `/one_cell_volume_witness/monotone_range_criterion/verdict` | *absent* | `false` | — |
| `/one_cell_volume_witness/multi_cell_gap_range_abs` | *absent* | `[0.0109, 0.0246]` | 0–1 |
| `/one_cell_volume_witness/note` | *absent* | `"#931 lattice ownership contract, design note 20260906 section 5: a PEC volume one cell thick is a filled slab with a tangential wall at BOTH faces, at every thickness, with no flag. Before #931 the far face was never a wall and a two_plane flag put it back for t = 1 only; nothing independent said which was right AT ONE CELL, because the thin-limit anchor is a t -> 0 statement rather than a t = dx one. Here the mode-matching oracle — which takes the physical t and knows nothing about the lattice — is run against rfx at t = 1..8 cells on the coarse rung at the worst-gap aperture. GATE: every swept rung must IDENTIFY its own drawn thickness — the oracle is evaluated at t-1, t and t+1 cells and the residual argmin must land on t. At t = 1 the t-1 alternative is precisely the pre-#931 realization (one wall, a zero-thickness screen), so the rule in dispute is decided by a measurement rather than by a convention. RETIRED, and recorded rather than deleted (monotone_range_criterion): the original statement — the t = 1 residual lies inside the range t = 2..8 spans — turned out to be vacuous once the residual was measured to be monotone decreasing in t, because then t = 1 is the extremum for every possible outcome, a perfect 0.0000 included. It was retired for having no power in either direction, not for its verdict."` | — |
| `/one_cell_volume_witness/one_cell_gap_abs` | *absent* | `0.0312` | — |
| `/one_cell_volume_witness/one_cell_two_wall_vs_one_wall_x` | *absent* | `4.317` | — |
| `/one_cell_volume_witness/passed` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/0/identification/argmin_t_cells` | *absent* | `1` | — |
| `/one_cell_volume_witness/rows/0/identification/gap_at_t` | *absent* | `0.0312` | — |
| `/one_cell_volume_witness/rows/0/identification/gap_at_t_minus_1` | *absent* | `0.1347` | — |
| `/one_cell_volume_witness/rows/0/identification/gap_at_t_plus_1` | *absent* | `0.04` | — |
| `/one_cell_volume_witness/rows/0/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/0/identification/margin_vs_runner_up_x` | *absent* | `1.282` | — |
| `/one_cell_volume_witness/rows/0/iris_wall_nodes` | *absent* | `[191, 192]` | 0–1 |
| `/one_cell_volume_witness/rows/0/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/0/max_gap_abs` | *absent* | `0.0312` | — |
| `/one_cell_volume_witness/rows/0/oracle_s11` | *absent* | array[29], SHA256 `bd0c9992cfbe` | 0–28 |
| `/one_cell_volume_witness/rows/0/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/0/realized_thickness_cells` | *absent* | `1` | — |
| `/one_cell_volume_witness/rows/0/s11` | *absent* | array[29], SHA256 `775b78889ddd` | 0–28 |
| `/one_cell_volume_witness/rows/0/t_cells` | *absent* | `1` | — |
| `/one_cell_volume_witness/rows/0/t_mm` | *absent* | `0.762` | — |
| `/one_cell_volume_witness/rows/0/wall_s` | *absent* | `26.1` | — |
| `/one_cell_volume_witness/rows/1/identification/argmin_t_cells` | *absent* | `2` | — |
| `/one_cell_volume_witness/rows/1/identification/gap_at_t` | *absent* | `0.0246` | — |
| `/one_cell_volume_witness/rows/1/identification/gap_at_t_minus_1` | *absent* | `0.0942` | — |
| `/one_cell_volume_witness/rows/1/identification/gap_at_t_plus_1` | *absent* | `0.0329` | — |
| `/one_cell_volume_witness/rows/1/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/1/identification/margin_vs_runner_up_x` | *absent* | `1.337` | — |
| `/one_cell_volume_witness/rows/1/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/one_cell_volume_witness/rows/1/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/1/max_gap_abs` | *absent* | `0.0246` | — |
| `/one_cell_volume_witness/rows/1/oracle_s11` | *absent* | array[29], SHA256 `bed1e9c4e3a2` | 0–28 |
| `/one_cell_volume_witness/rows/1/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/1/realized_thickness_cells` | *absent* | `2` | — |
| `/one_cell_volume_witness/rows/1/s11` | *absent* | array[29], SHA256 `40f1fb558306` | 0–28 |
| `/one_cell_volume_witness/rows/1/t_cells` | *absent* | `2` | — |
| `/one_cell_volume_witness/rows/1/t_mm` | *absent* | `1.524` | — |
| `/one_cell_volume_witness/rows/1/wall_s` | *absent* | `27.0` | — |
| `/one_cell_volume_witness/rows/2/identification/argmin_t_cells` | *absent* | `3` | — |
| `/one_cell_volume_witness/rows/2/identification/gap_at_t` | *absent* | `0.0207` | — |
| `/one_cell_volume_witness/rows/2/identification/gap_at_t_minus_1` | *absent* | `0.0767` | — |
| `/one_cell_volume_witness/rows/2/identification/gap_at_t_plus_1` | *absent* | `0.0277` | — |
| `/one_cell_volume_witness/rows/2/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/2/identification/margin_vs_runner_up_x` | *absent* | `1.338` | — |
| `/one_cell_volume_witness/rows/2/iris_wall_nodes` | *absent* | `[190, 193]` | 0–1 |
| `/one_cell_volume_witness/rows/2/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/2/max_gap_abs` | *absent* | `0.0207` | — |
| `/one_cell_volume_witness/rows/2/oracle_s11` | *absent* | array[29], SHA256 `57665830614d` | 0–28 |
| `/one_cell_volume_witness/rows/2/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/2/realized_thickness_cells` | *absent* | `3` | — |
| `/one_cell_volume_witness/rows/2/s11` | *absent* | array[29], SHA256 `396803b8a62e` | 0–28 |
| `/one_cell_volume_witness/rows/2/t_cells` | *absent* | `3` | — |
| `/one_cell_volume_witness/rows/2/t_mm` | *absent* | `2.286` | — |
| `/one_cell_volume_witness/rows/2/wall_s` | *absent* | `22.5` | — |
| `/one_cell_volume_witness/rows/3/identification/argmin_t_cells` | *absent* | `4` | — |
| `/one_cell_volume_witness/rows/3/identification/gap_at_t` | *absent* | `0.0179` | — |
| `/one_cell_volume_witness/rows/3/identification/gap_at_t_minus_1` | *absent* | `0.0647` | — |
| `/one_cell_volume_witness/rows/3/identification/gap_at_t_plus_1` | *absent* | `0.0236` | — |
| `/one_cell_volume_witness/rows/3/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/3/identification/margin_vs_runner_up_x` | *absent* | `1.318` | — |
| `/one_cell_volume_witness/rows/3/iris_wall_nodes` | *absent* | `[189, 193]` | 0–1 |
| `/one_cell_volume_witness/rows/3/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/3/max_gap_abs` | *absent* | `0.0179` | — |
| `/one_cell_volume_witness/rows/3/oracle_s11` | *absent* | array[29], SHA256 `226f8c0145f4` | 0–28 |
| `/one_cell_volume_witness/rows/3/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/3/realized_thickness_cells` | *absent* | `4` | — |
| `/one_cell_volume_witness/rows/3/s11` | *absent* | array[29], SHA256 `43cc528859fa` | 0–28 |
| `/one_cell_volume_witness/rows/3/t_cells` | *absent* | `4` | — |
| `/one_cell_volume_witness/rows/3/t_mm` | *absent* | `3.048` | — |
| `/one_cell_volume_witness/rows/3/wall_s` | *absent* | `24.6` | — |
| `/one_cell_volume_witness/rows/4/identification/argmin_t_cells` | *absent* | `5` | — |
| `/one_cell_volume_witness/rows/4/identification/gap_at_t` | *absent* | `0.0157` | — |
| `/one_cell_volume_witness/rows/4/identification/gap_at_t_minus_1` | *absent* | `0.0555` | — |
| `/one_cell_volume_witness/rows/4/identification/gap_at_t_plus_1` | *absent* | `0.0202` | — |
| `/one_cell_volume_witness/rows/4/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/4/identification/margin_vs_runner_up_x` | *absent* | `1.287` | — |
| `/one_cell_volume_witness/rows/4/iris_wall_nodes` | *absent* | `[189, 194]` | 0–1 |
| `/one_cell_volume_witness/rows/4/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/4/max_gap_abs` | *absent* | `0.0157` | — |
| `/one_cell_volume_witness/rows/4/oracle_s11` | *absent* | array[29], SHA256 `c75b34c3a600` | 0–28 |
| `/one_cell_volume_witness/rows/4/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/4/realized_thickness_cells` | *absent* | `5` | — |
| `/one_cell_volume_witness/rows/4/s11` | *absent* | array[29], SHA256 `27599ab95c1f` | 0–28 |
| `/one_cell_volume_witness/rows/4/t_cells` | *absent* | `5` | — |
| `/one_cell_volume_witness/rows/4/t_mm` | *absent* | `3.81` | — |
| `/one_cell_volume_witness/rows/4/wall_s` | *absent* | `25.2` | — |
| `/one_cell_volume_witness/rows/5/identification/argmin_t_cells` | *absent* | `6` | — |
| `/one_cell_volume_witness/rows/5/identification/gap_at_t` | *absent* | `0.0139` | — |
| `/one_cell_volume_witness/rows/5/identification/gap_at_t_minus_1` | *absent* | `0.0481` | — |
| `/one_cell_volume_witness/rows/5/identification/gap_at_t_plus_1` | *absent* | `0.0173` | — |
| `/one_cell_volume_witness/rows/5/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/5/identification/margin_vs_runner_up_x` | *absent* | `1.245` | — |
| `/one_cell_volume_witness/rows/5/iris_wall_nodes` | *absent* | `[188, 194]` | 0–1 |
| `/one_cell_volume_witness/rows/5/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/5/max_gap_abs` | *absent* | `0.0139` | — |
| `/one_cell_volume_witness/rows/5/oracle_s11` | *absent* | array[29], SHA256 `0edc599036e2` | 0–28 |
| `/one_cell_volume_witness/rows/5/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/5/realized_thickness_cells` | *absent* | `6` | — |
| `/one_cell_volume_witness/rows/5/s11` | *absent* | array[29], SHA256 `b121a5cbbd7c` | 0–28 |
| `/one_cell_volume_witness/rows/5/t_cells` | *absent* | `6` | — |
| `/one_cell_volume_witness/rows/5/t_mm` | *absent* | `4.572` | — |
| `/one_cell_volume_witness/rows/5/wall_s` | *absent* | `27.0` | — |
| `/one_cell_volume_witness/rows/6/identification/argmin_t_cells` | *absent* | `8` | — |
| `/one_cell_volume_witness/rows/6/identification/gap_at_t` | *absent* | `0.0109` | — |
| `/one_cell_volume_witness/rows/6/identification/gap_at_t_minus_1` | *absent* | `0.0367` | — |
| `/one_cell_volume_witness/rows/6/identification/gap_at_t_plus_1` | *absent* | `0.0131` | — |
| `/one_cell_volume_witness/rows/6/identification/identified_own_thickness` | *absent* | `true` | — |
| `/one_cell_volume_witness/rows/6/identification/margin_vs_runner_up_x` | *absent* | `1.202` | — |
| `/one_cell_volume_witness/rows/6/iris_wall_nodes` | *absent* | `[187, 195]` | 0–1 |
| `/one_cell_volume_witness/rows/6/max_colpow` | *absent* | `1.0` | — |
| `/one_cell_volume_witness/rows/6/max_gap_abs` | *absent* | `0.0109` | — |
| `/one_cell_volume_witness/rows/6/oracle_s11` | *absent* | array[29], SHA256 `278c32a67858` | 0–28 |
| `/one_cell_volume_witness/rows/6/realized_aperture_cells` | *absent* | `16` | — |
| `/one_cell_volume_witness/rows/6/realized_thickness_cells` | *absent* | `8` | — |
| `/one_cell_volume_witness/rows/6/s11` | *absent* | array[29], SHA256 `6f7db1192fff` | 0–28 |
| `/one_cell_volume_witness/rows/6/t_cells` | *absent* | `8` | — |
| `/one_cell_volume_witness/rows/6/t_mm` | *absent* | `6.096` | — |
| `/one_cell_volume_witness/rows/6/wall_s` | *absent* | `25.8` | — |
| `/provenance/modal_fence_retraction_2026_07_28` | `"An earlier revision of this case FENCED normalize=True modal extraction, citing measured max column power 1.112 (later 1.15374 at driven port 0 coarse / 1.16407 at driven port 1 fine, with a second per-frequency advisory referencing issue #337). Those runs carried the d + 2*dx electrical aperture and a 0.5*lambda_g absorber. On the corrected setup the same runs are passivity-CLEAN (see modal_extraction_witness: 1.0207 / 1.0101 / 1.0144 / 1.0013, zero extractor warnings), so the fence is RETRACTED: the non-passivity was a setup symptom, not an extractor property. Recorded so the withdrawn claim stays auditable."` | `"An earlier revision of this case FENCED normalize=True modal extraction, citing measured max column power 1.112 (later 1.15374 at driven port 0 coarse / 1.16407 at driven port 1 fine, with a second per-frequency advisory referencing issue #337). Those runs carried the d + 2*dx electrical aperture and a 0.5*lambda_g absorber. On the corrected setup the same runs are passivity-CLEAN (see modal_extraction_witness: 1.0200 / 1.0099 / 1.0150 / 1.0012, zero extractor warnings), so the fence is RETRACTED: the non-passivity was a setup symptom, not an extractor property. Recorded so the withdrawn claim stays auditable."` | — |
| `/provenance/no_preflight_note` | `"compute_waveguide_s_matrix runs its own extractor passivity self-check (warnings are part of this record) but no sim.preflight(); operating-point guarantees are the raster asserts in run_point."` | `"compute_waveguide_s_matrix runs its own extractor passivity self-check (warnings are part of this record) but no sim.preflight(); operating-point guarantees are the realized-geometry asserts in run_point, which read realized_pec_edge_masks through validation/crossval/_wr90_iris_realized.py."` | — |
| `/raw_extraction_record/0/aperture_cells` | `23` | *absent* | — |
| `/raw_extraction_record/0/aperture_wall_nodes` | *absent* | `[3, 27]` | 0–1 |
| `/raw_extraction_record/0/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/raw_extraction_record/0/max_colpow` | `1.0007` | `0.9996` | — |
| `/raw_extraction_record/0/max_gap_abs` | `0.0535` | `0.0244` | — |
| `/raw_extraction_record/0/realized_aperture_cells` | *absent* | `24` | — |
| `/raw_extraction_record/0/realized_thickness_cells` | *absent* | `2` | — |
| `/raw_extraction_record/0/s11` | array[29], SHA256 `a69794f8f943` | array[29], SHA256 `6359557acac1` | 0–28 |
| `/raw_extraction_record/0/s21` | array[29], SHA256 `ecaa082dd176` | array[29], SHA256 `07ed0cb8427e` | 0–28 |
| `/raw_extraction_record/0/t_mm` | *absent* | `1.524` | — |
| `/raw_extraction_record/0/thickness_cells` | `2` | *absent* | — |
| `/raw_extraction_record/0/wall_s` | `30.4` | `10.3` | — |
| `/raw_extraction_record/1/aperture_cells` | `15` | *absent* | — |
| `/raw_extraction_record/1/aperture_wall_nodes` | *absent* | `[7, 23]` | 0–1 |
| `/raw_extraction_record/1/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/raw_extraction_record/1/max_colpow` | `1.0004` | `1.0007` | — |
| `/raw_extraction_record/1/max_gap_abs` | `0.0543` | `0.0254` | — |
| `/raw_extraction_record/1/realized_aperture_cells` | *absent* | `16` | — |
| `/raw_extraction_record/1/realized_thickness_cells` | *absent* | `2` | — |
| `/raw_extraction_record/1/s11` | array[29], SHA256 `a795ed721597` | array[29], SHA256 `b9fd3bc6fb7e` | 0–28 |
| `/raw_extraction_record/1/s21` | array[29], SHA256 `277fe667de81` | array[29], SHA256 `05a5ae7528be` | 0–28 |
| `/raw_extraction_record/1/t_mm` | *absent* | `1.524` | — |
| `/raw_extraction_record/1/thickness_cells` | `2` | *absent* | — |
| `/raw_extraction_record/1/wall_s` | `30.8` | `9.9` | — |
| `/raw_extraction_record/2/aperture_cells` | `9` | *absent* | — |
| `/raw_extraction_record/2/aperture_wall_nodes` | *absent* | `[10, 20]` | 0–1 |
| `/raw_extraction_record/2/iris_wall_nodes` | *absent* | `[190, 192]` | 0–1 |
| `/raw_extraction_record/2/max_colpow` | `1.0001` | `1.0016` | — |
| `/raw_extraction_record/2/max_gap_abs` | `0.0205` | `0.0088` | — |
| `/raw_extraction_record/2/realized_aperture_cells` | *absent* | `10` | — |
| `/raw_extraction_record/2/realized_thickness_cells` | *absent* | `2` | — |
| `/raw_extraction_record/2/s11` | array[29], SHA256 `04878a5cf129` | array[29], SHA256 `2e4f41e8ab2f` | 0–28 |
| `/raw_extraction_record/2/s21` | array[29], SHA256 `a2d27b295880` | array[29], SHA256 `7f1a158afbae` | 0–28 |
| `/raw_extraction_record/2/t_mm` | *absent* | `1.524` | — |
| `/raw_extraction_record/2/thickness_cells` | `2` | *absent* | — |
| `/raw_extraction_record/2/wall_s` | `30.1` | `12.1` | — |
| `/schema_version` | `1` | `2` | — |
| `/truncation_witness/3/shift_abs` | `1e-05` | `0.0` | — |

## validation/crossval/_19_iris_filter_results/rfx.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/absorber_depth_witness/d_hi_mhz` | `0.06` | `0.01` | — |
| `/absorber_depth_witness/d_lo_mhz` | `0.02` | `0.0` | — |
| `/absorber_depth_witness/deep/d_hi_mhz` | `0.063` | `0.014` | — |
| `/absorber_depth_witness/deep/d_lo_mhz` | `0.015` | `0.004` | — |
| `/absorber_depth_witness/deep/hi` | `11140753280.781218` | `11140910475.46354` | — |
| `/absorber_depth_witness/deep/lo` | `10800389034.893412` | `10800344622.687653` | — |
| `/absorber_depth_witness/deep/s11` | array[131], SHA256 `80701c043a3b` | array[131], SHA256 `835ba4548bb8` | 0–8, 10–15, 17–21, 24–99, 102–108, 110–114, 116–119, 121–122, 125–130 |
| `/absorber_depth_witness/deep/s21` | array[131], SHA256 `f98624952c50` | array[131], SHA256 `cf2a439b0afb` | 0–11, 13–114, 116–130 |
| `/absorber_depth_witness/deep/wall_s` | `1483.5` | `1033.7` | — |
| `/absorber_depth_witness/gated/hi` | `11140816346.186275` | `11140924560.619234` | — |
| `/absorber_depth_witness/gated/lo` | `10800373669.774708` | `10800340317.655672` | — |
| `/absorber_depth_witness/mid/d_hi_mhz` | `0.051` | `0.009` | — |
| `/absorber_depth_witness/mid/d_lo_mhz` | `0.011` | `0.003` | — |
| `/absorber_depth_witness/mid/hi` | `11140765734.608944` | `11140915351.637445` | — |
| `/absorber_depth_witness/mid/lo` | `10800385111.490433` | `10800343078.352245` | — |
| `/absorber_depth_witness/mid/s11` | array[131], SHA256 `ec4862e79b34` | array[131], SHA256 `e803a62b9b4c` | 1–19, 21, 23, 25, 27–96, 100–105, 107–112, 114–116, 118–120, 123, 125–128, 130 |
| `/absorber_depth_witness/mid/s21` | array[131], SHA256 `90f014a0e3ff` | array[131], SHA256 `a96bbee72b56` | 0–10, 12–14, 16–111, 113–114, 116–130 |
| `/absorber_depth_witness/mid/wall_s` | `1324.4` | `940.9` | — |
| `/b_invariance_witness/0/bw` | `340442676.4115677` | `340584242.963562` | — |
| `/b_invariance_witness/0/f0` | `10970595007.980492` | `10970632439.137453` | — |
| `/b_invariance_witness/0/hi` | *absent* | `11140924560.619234` | — |
| `/b_invariance_witness/0/lo` | *absent* | `10800340317.655672` | — |
| `/b_invariance_witness/0/s11` | array[131], SHA256 `19dd17dae901` | array[131], SHA256 `a8d2053b1cec` | 0–18, 20–26, 28–99, 101–106, 108–113, 115–122, 124–130 |
| `/b_invariance_witness/0/s21` | array[131], SHA256 `74e9a501b715` | array[131], SHA256 `b14514c8010c` | 0–130 |
| `/b_invariance_witness/0/wall_s` | `1160.3` | `850.0` | — |
| `/b_invariance_witness/1/bw` | `340442419.3242817` | `340584420.32989883` | — |
| `/b_invariance_witness/1/f0` | `10970594978.610989` | `10970632624.384037` | — |
| `/b_invariance_witness/1/hi` | *absent* | `11140924834.548986` | — |
| `/b_invariance_witness/1/lo` | *absent* | `10800340414.219088` | — |
| `/b_invariance_witness/1/max_dev_vs_b4` | `29.369503021240234` | `185.24658393859863` | — |
| `/b_invariance_witness/1/s11` | array[131], SHA256 `c1cc2ac94cf2` | array[131], SHA256 `9319cb5d9744` | 0–16, 18, 20–23, 25–26, 28–99, 101–107, 109, 111–130 |
| `/b_invariance_witness/1/s21` | array[131], SHA256 `1f66898b2053` | array[131], SHA256 `368ba5c1b004` | 0–130 |
| `/b_invariance_witness/1/wall_s` | `1526.9` | `1037.1` | — |
| `/b_invariance_witness/2/bw` | `340442195.4324913` | `340583617.81036377` | — |
| `/b_invariance_witness/2/f0` | `10970594855.77844` | `10970632206.62834` | — |
| `/b_invariance_witness/2/hi` | *absent* | `11140924015.533522` | — |
| `/b_invariance_witness/2/lo` | *absent* | `10800340397.723158` | — |
| `/b_invariance_witness/2/max_dev_vs_b4` | `152.20205116271973` | `232.50911331176758` | — |
| `/b_invariance_witness/2/s11` | array[131], SHA256 `e9e31388046c` | array[131], SHA256 `a9e1bfc92400` | 0–7, 9–16, 18–26, 28–99, 102–104, 106–109, 111–122, 124–130 |
| `/b_invariance_witness/2/s21` | array[131], SHA256 `b3770b632d22` | array[131], SHA256 `27a9f50e765a` | 0–130 |
| `/b_invariance_witness/2/wall_s` | `1938.9` | `1374.6` | — |
| `/claim_scope` | `"A published 4th-order WR-90 inductive-iris bandpass filter (Aghanim et al., E3S Web of Conferences 351, 01059 (2022), CC BY 4.0, Table 6 optimized: five irises t = 2.00 mm, apertures 10.27/6.65/6.18/6.65/10.27 mm, cavities 14.29/15.73/15.73/14.29 mm) built at dx = a/90 and compared against a TEn0 mode-matching cascade oracle over 10.40-11.70 GHz on 131 points at 10 MHz. Stage S3 of the waveguide-obstacle campaign and the first RESONANT multi-obstacle case in the lane: unlike the single iris of S1, a per-face geometry error here is a passband shift rather than a magnitude tolerance. GATED: centre frequency f0 within 19 MHz = round-up(measured envelope 12.1230 x 1.5); the structural reflection-zero COUNT (an integer, depth-independent); and passband CONTIGUITY as a regression lock (span_holes <= 1, the committed envelope) -- added after a post-merge joint review showed f0 is computed from the OUTERMOST -10 dB crossings, so a future regeneration whose passband split into separated resonances could have shipped green with its bridged midpoint inside the f0 gate. All against the oracle evaluated on the AS-REALIZED geometry. Measured d_f0 = +12.08 MHz, zeros 3 vs 3, one interior hole bin. The zero-count gate is additionally witnessed ROBUST to the unsettled iris-thickness convention: an oracle-side sweep of t_elec across 8.00-8.50 cells (covering both candidate conventions; committed as iris_thickness_zero_count_sweep) holds the count at 3 throughout while bandwidth moves 20 MHz across the same band, so the gated integer does not depend on which convention the comparator picks. The envelope is a population of NINE configurations over four setup axes, not a single run, and each axis carries an INTERIOR sample as well as an endpoint: guide height b = 4/6/8 cells, run length num_periods 400/600/800, port standoff 3.05/7.62/15.24 mm, absorber depth 0.75/1.00/1.25 lambda_g. The interior samples are the point rather than decoration: a one-alternative-per-axis envelope cannot detect NON-MONOTONIC sensitivity, which is exactly the failure of PR #475, where three sampled clearances passed while 9 of 13 exceeded the gate and the passing three were the sampled ones. Every population member carries its own committed &#124;S11&#124; trace, so each residual is recomputable rather than a free-floating scalar whose integrity is borrowed from asserts living in other tests. WHAT THAT GATE IS AND IS NOT, stated because the phrasing invites more than it delivers: the population makes the envelope ROBUST rather than resting on one datum, but it does not make the gate independent of the datum. The spread is 0.06 MHz while every member's &#124;d_f0&#124; is about 12.08 MHz, so the envelope is dominated by the RESIDUAL and not by the scatter, and gate = round-up(env x 1.5) is therefore 1.5x the measured agreement. This is a REGRESSION LOCK with 50 percent headroom, not an independent accuracy bound, exactly as the merged case 18's gate is; what gives the measured agreement meaning is not the gate but the comparison of that agreement against an external scale, namely the reference's own 21.9 MHz f0 spread between two independent commercial codes. That tightness is the substance of the result: the residual is a reproducible systematic difference rather than a setup artifact, and at the measured cavity sensitivity of -105 MHz/cell it corresponds to about 0.12 cell of cavity length. The num_periods = 200 run is EXCLUDED from the envelope rather than folded in, because it fails the settling criterion at column power 1.207; it stays committed as the evidence that the settling gate can fire. WHY f0 AND NOT BANDWIDTH, which is the correction this case exists to record: the oracle must be fed the geometry that was BUILT, not the geometry that was DRAWN, and the three legs of that convention are not equally settled. The transverse aperture leg d_c*dx is confirmed to better than 0.05 cell by an independent refit of 16 committed case-18 configurations during the #499 review -- a session measurement; the committed corroboration is the per-run raster assert on the open-node count and the exact-mask FDFD agreement. The cavity leg (L_c + 1)*dx - the distance between the bounding zeroed node planes - is confirmed to 0.04-0.17 cell and carries about 105 of the 107.5 MHz that separates a drawn-count oracle from a realized-geometry one. But the IRIS-THICKNESS leg is NOT (t_c - 1)*dx: four independent FDTD runs at drawn t_c = 2/4/6/8 give a flat offset of -0.66/-0.68/-0.68/-0.70 cell, i.e. t_elec is about (t_c - 0.68)*dx, matching neither this case's earlier rule nor the merged case 18's t_c*dx, with a residual 10-33x below both. That leaves an irreducible comparator-input uncertainty of order half a cell, and the gated observable is therefore chosen by SENSITIVITY to it: per cell of convention error, f0 moves about 2.4 MHz, bandwidth about 40 MHz, and individual band edges 22-30 MHz. So f0 and the zero count are gated; band edges and bandwidth are REPORTED, because a +/-20 MHz input uncertainty cannot honestly sit under a 15 MHz gate. Adopting the fitted -0.68 cell would absorb the disagreement into a free parameter, after which the residual would measure nothing - the tautological-validation failure this campaign has hit repeatedly - so the offset is recorded as an uncertainty and NOT adopted. Handing the oracle the drawn cell counts instead of the realized ones biases f0 by +107.5 MHz, five times the reference's own 21.9 MHz CST-vs-HFSS spread, and the envelope-times-1.5 rule does NOT catch that class because the rule bounds SCATTER and this is BIAS. The realized lengths are read back off the rasterized metal and re-derived again from the committed node indices in the frozen gate test. Total electrical length is a face-continuity CHECK across region types, NOT a uniqueness argument: putting the metal/open interface at sigma*dx beyond the outermost metal node gives total = span - 1 + 2*sigma = the outer extent measured at the same sigma, conserved for EVERY sigma, and sigma = 0.5 is the drawn pairing itself. An earlier revision claimed this was \"the only pairing that conserves total electrical length\"; that is false and is withdrawn. Drawn counts are COMPENSATED (t_c = round(t/dx) + 1, L_c = round(L/dx) - 1) so the stated electrical dimensions land on nominal, which is nearest-representable rounding with zero free parameters and no reference number entering - that, not any mesh comparison, is why it is a statement of intent rather than a fit. It is NOT a monotone improvement: at a/60 it moves f0 from -35.8 to +120.2 MHz, because it converts a uniformly-signed set of per-cavity errors into a mixed-sign one. REPORTED, NEVER GATED: individual band edges (+17.08 / +7.09 MHz) and bandwidth (-9.99 MHz), which are ONE fact and not two - d_bw is identically d_hi - d_lo - so the earlier framing of an \"unexplained asymmetric edge residual\" separate from a bandwidth deficit was an algebraic error; worst in-band return loss; individual ripple levels; every reflection-zero DEPTH (four nominally identical equiripple zeros bottom out across a wide spread in the published figure, so the paper's frequency step and not physics sets those depths - zero FREQUENCIES are meaningful, depths are not values); passband contiguity beyond the span_holes <= 1 lock (hole depth, hole position, longest contiguous run); the coarse a/60 rung; and phase. PASSBAND CONTIGUITY IS RECORDED, NOT ASSUMED: lo and hi are the OUTERMOST interpolated -10 dB crossings, so the span between them is not necessarily a passband. The built filter has one 10 MHz bin at 9.80 dB inside its 340 MHz span (longest contiguous -10 dB run 270 MHz), while the oracle on the same geometry is contiguous over 35 bins - a real difference that an earlier revision hid behind a clamped statistic, because worst-RL had been computed as the minimum over samples already filtered to >= 10 dB and therefore could not report a violation at all. The coarse a/60 rung has no meaningful passband: 16 of 24 bins in its nominal span are above -10 dB with an interior trough at 2.73 dB, i.e. two separated resonances. That, and not any gate comparison, is the evidence that the gated mesh had to be a/90. The a/60 rung's own numbers do not disqualify it cleanly: its zero count matches its oracle (2 vs 2) and its f0 residual (+19.85 MHz) is the same ~0.12-cell offset seen at a/90; against the committed 19 MHz constant it happens to fail by 0.85 MHz, but a self-derived envelope-times-1.5 gate would pass it. The broken passband is the disqualifier. SETUP IS GATED SEPARATELY FROM PHYSICS, because a resonant band read off an unsettled or absorber-limited run is not a measurement. The repo's preferred ENERGY-BASED ring-down witness (terminal energy in dB below the post-source peak, rule < -40 dB) is NOT AVAILABLE on this path: it is implemented for the lumped/MSL S-matrix extractor, but compute_waveguide_s_matrix returns no settling_db, and the null fields are committed in every row rather than papered over (filed as an rfx capability gap). The ENFORCED settling criterion is therefore the independent axis: the num_periods scan 400/600/800 holds f0 and BW to under 0.1 MHz (the gate is the 400 -> 800 doubling within one 10 MHz bin), the gated run is passivity-clean at column power 1.0065, and the criterion demonstrably fires -- the np=200 run is excluded non-passive at 1.207. The feed-clearance and absorber-depth scans each hold the edges to one 10 MHz bin across their interior and outer samples (standoff 3.05 -> 7.62 -> 15.24 mm: 0.0/0.1 MHz; absorber 0.75 -> 1.00 -> 1.25 lambda_g: 0.0/0.1 MHz). Absorber depth is scanned because in S1 it was the envelope-limiting term at 0.5 lambda_g; here 0.75 lambda_g is measurably sufficient, which is a negative result worth recording rather than a rule inherited. Extractor warnings and the passivity footprint (the bins where column power exceeds 1.02, not merely the scalar maximum) are committed per row. Guide height is reduced to 4 cells on a MEASURED b-invariance witness: b = 4 and b = 8 agree to 152 Hz in f0 on THIS resonant five-iris filter, not merely on the single iris where it was first measured, which is the 8x saving that makes the case affordable to generate. THE ORACLE HAS HAD AN ADVERSARIAL PASS, and it did not find the residual. Its own witnesses are unitarity 2.2e-15, reciprocity and mirror symmetry exact, and an L -> 0 collapse closing two thin irises onto one thick one; its N=1 centred limit reproduces the merged case-18 oracle to 1.05e-04 in an independent odd-mode formulation, and THAT object is what PR #480 confirmed against a formulation-independent 2-D H-plane FDFD at 5.8e-4 - a comparison the frozen gate test now EXECUTES rather than asserting in prose. Those witnesses have known limits, stated because an earlier revision leaned on them too hard: unitarity constrains only the propagating sub-block, mirror symmetry holds by construction for a symmetric geometry, and injected overlap-integral errors leave the reduction and collapse axes silent because they share the overlap routine, so the gate test's re-typed cascade agreeing to 0.0e+00 is a REGRESSION LOCK and not a second opinion. An independent review then closed the real gaps. What is COMMITTED: the overlap integral is validated against direct numerical quadrature IN CI (216 combinations across three apertures including the as-realized 6.096 mm -- where n*pi/a equals m*pi/d exactly for six mode pairs -- centred and 0.19 mm off-centre, worst deviation bounded at 1e-12 with the small-denominator guard exercised), and the oracle's truncation is witnessed on the GATED observable at generation time: f0 by bisection moves 0.33 MHz for n_a 90 -> 130 and 1.85 MHz for an aperture-mode-count doubling -- the aperture axis is the sensitive one, and both are an order under the 19 MHz gate. Session measurements recorded in the research notes but NOT in this record put the gauge invariance of the sqrt(Y) normalisation at machine precision, truncation saturation near 1 MHz in bandwidth, and inter-cavity evanescent transport near 1.6 MHz; they are corroborating colour rather than load-bearing, because the formulation-level check below subsumes their role. THE FORMULATION-LEVEL CHECK IS NOW DONE, and it lands the residual on the rfx side. A 2-D H-plane FDFD -- scalar Helmholtz on a finite-difference grid with an exact discrete transparent port condition, sharing only numpy and scipy with the cascade and no rfx code path at all (validation/crossval/comparators/fdfd_hplane.py) -- was run on the same electrical geometry, grid-exact at every refinement level. It is FIRST-order convergent, so no single level is meaningful; the record carries THREE levels (r = 2, 3, 4, whose bandwidth deviations from the extrapolate shrink as 1/r: measured ratios 1.55 and 1.33 against the first-order 1.50 and 1.33) and BOTH Richardson estimates, which agree to 0.37 MHz in centre frequency and 0.36 MHz in bandwidth -- the two-estimate consistency protocol the porting handoff mandates before either estimate is trusted. The finer pair gives f0 = 10.95742 GHz and BW = 351.42 MHz against the cascade's 10.95851 and 350.43: agreement to 1.09 MHz in centre frequency and 0.98 MHz in bandwidth between two formulations that share nothing but their numerical libraries, with the extrapolation's own order-assumption uncertainty, not the two-estimate consistency, bounding how much of that gap the FDFD itself owns: the consistency (0.37/0.36 MHz) is agreement between two fits of the SAME first-order model, not an error bound -- fitting the convergence order to the three committed levels edge-by-edge gives p = 1.26 (lower edge) / 1.24 (upper edge) rather than the assumed 1, and re-extrapolating the finer pair at the fitted order moves f0 by +0.68 MHz and bandwidth by -0.65 MHz, so the FDFD owns roughly 0.7-1.0 MHz of the 1.09/0.98 MHz gap; f0 and bandwidth are also algebraic combinations of the SAME two band edges, one confirmation and not two. rfx differs from the FDFD by +13.17 MHz in f0 and -10.97 MHz in bandwidth, essentially the same as it differs from the cascade (+12.08 / -9.99), so the 12 MHz residual is not an oracle error. The FDFD's gates: lossless unitarity is enforced on EVERY evaluation (worst 4.6e-07 across all levels), and the empty-guide transparency gate -- &#124;S11&#124; = 5.0e-14 with &#124;S21&#124; = 1.000000000000, the test that originally caught a missing 1/h in the discrete propagation constant -- runs once per generation and once per CI pass. One defect in the comparator itself was found by an independent port review and fixed before this record was generated: its aperture mask realized every aperture two fine cells wide of the stated convention, a first-order bias that Richardson cancelled -- making the extrapolated numbers right for the wrong per-level geometry -- and that produced a spurious FOURTH reflection zero at the coarser levels. With the mask exact, every level shows THREE zeros, matching the cascade and rfx. Independently, the cascade's zero count was checked against its own aperture-mode truncation, which nobody had done for the COUNT: it is 3 at nb_scale 1.0, 1.5, 2.0 and 3.0, with f0 moving 1.7 MHz over that 3x range. What remains genuinely unexplained is the ~12 MHz rfx residual itself: it is mesh-invariant when expressed in cells (-0.1169 cell at a/90 against -0.1241 at a/60, where dispersion would have given 0.083), so it behaves like a fixed geometric offset rather than a frequency-dependent solver error, but attributing it to a specific convention leg has FAILED: propagating the independently measured iris thickness (t_c - 0.68)*dx through node-plane length conservation overshoots and flips the sign, taking a/90 from +12.08 to -30.62 MHz. That attribution is recorded as falsified, not as pending. _gamma at exact cutoff (k equal to n*pi/w, where the sqrt argument vanishes) is unreachable at these band edges and untested. FENCED: nothing here promotes the lane beyond S1. Multi-iris filters, posts and septa remain EXPERIMENTAL; this measures one published design on one mesh with one gated observable, and certifies neither arbitrary filters nor the a/60 rung. Says nothing about phase, group delay, loss, higher-order-mode ports, or fabrication tolerance. The reference is an ANCHOR, not a solver run: the CST and HFSS scalars are digitized from the paper's Fig. 5, no external solver is invoked here, and the case does not compare rfx against either commercial code on any geometry - so the 12.1 MHz f0 residual against this case's own analytic oracle must not be read as an accuracy claim relative to CST or HFSS. Reported for context and not as a yardstick beaten: the oracle on nominal dimensions sits -6.2 MHz from CST and -28.2 MHz from HFSS in f0, +14.7 MHz from both in bandwidth (against a published inter-solver bandwidth spread of only 0.4 MHz), -0.4 and -1.1 dB in worst return loss, and up to 25.4 and 61.9 MHz in individual reflection-zero frequencies. The bandwidth and zero-frequency misses are larger than the f0 miss and are stated here because quoting only f0 would be selective. The built structure is a SNAPPED Aghanim filter: its centre frequency is within the reference's own solver scatter of CST (though not of HFSS), one of the four structural reflection zeros is lost (4 -> 3, confirmed grid-robust by refining the oracle to 1 MHz, with the loss occurring in the upper band), and worst in-band return loss degrades from 13.82 dB to 10.65 dB by rasterization alone, oracle to oracle, with rfx at 9.80 dB. Say snapped, not equivalent. OBSERVABLE PRIORITY for a resonant structure, as this case measures it: the structural reflection-zero COUNT first (an integer, depth-independent, and shown grid-robust), then centre frequency (least sensitive of the continuous quantities to the unsettled convention, ~2.4 MHz per cell), then band edges and bandwidth (~22-40 MHz per cell, hence reported), then worst return loss, and last individual ripple levels and null depths, which are not values at all. TOPOLOGY FIRST, AND f0 IS NOT EXONERATED: the zero count is the most robust observable, but f0 is not thereby safe -- it carries the +12.08 MHz residual this case gates, and at -105 MHz per cell of cavity length it is the quantity a geometry error moves first. A cell snap is inherently non-uniform, since each cavity rounds independently, so every snap figure quoted here was MEASURED on the as-snapped geometry and none may be re-derived by multiplying a sensitivity coefficient by a half cell."` | `"A published 4th-order WR-90 inductive-iris bandpass filter (Aghanim et al., E3S Web of Conferences 351, 01059 (2022), CC BY 4.0, Table 6 optimized: five irises t = 2.00 mm, apertures 10.27/6.65/6.18/6.65/10.27 mm, cavities 14.29/15.73/15.73/14.29 mm) built at dx = a/90 and compared against a TEn0 mode-matching cascade oracle over 10.40-11.70 GHz on 131 points at 10 MHz. Stage S3 of the waveguide-obstacle campaign and the first RESONANT multi-obstacle case in the lane: unlike the single iris of S1, a per-face geometry error here is a passband shift rather than a magnitude tolerance. GATED: centre frequency f0 within 19 MHz = round-up(measured envelope 12.1219 x 1.5); the structural reflection-zero COUNT (an integer, depth-independent); and passband CONTIGUITY as a regression lock (span_holes <= 1, the committed envelope) -- added after a post-merge joint review showed f0 is computed from the OUTERMOST -10 dB crossings, so a future regeneration whose passband split into separated resonances could have shipped green with its bridged midpoint inside the f0 gate. All against the oracle evaluated on the AS-REALIZED geometry. Measured d_f0 = +12.12 MHz, zeros 3 vs 3, one interior hole bin. The zero-count gate is additionally witnessed ROBUST to a perturbation of the comparator's most length-sensitive input: an oracle-side sweep of t_elec across one full cell CENTRED on the realized iris thickness (committed as iris_thickness_zero_count_sweep) holds the count throughout while bandwidth moves across the same band, so the gated integer does not sit on one exact value. Before #931 that window was one-sided, 8.00-8.50 cells, spanning a genuine disagreement about what the iris thickness realized as; the lattice ownership contract removed the disagreement, so the sweep now measures sensitivity rather than ambiguity. The envelope is a population of NINE configurations over four setup axes, not a single run, and each axis carries an INTERIOR sample as well as an endpoint: guide height b = 4/6/8 cells, run length num_periods 400/600/800, port standoff 3.05/7.62/15.24 mm, absorber depth 0.75/1.00/1.25 lambda_g. The interior samples are the point rather than decoration: a one-alternative-per-axis envelope cannot detect NON-MONOTONIC sensitivity, which is exactly the failure of PR #475, where three sampled clearances passed while 9 of 13 exceeded the gate and the passing three were the sampled ones. Every population member carries its own committed &#124;S11&#124; trace, so each residual is recomputable rather than a free-floating scalar whose integrity is borrowed from asserts living in other tests. WHAT THAT GATE IS AND IS NOT, stated because the phrasing invites more than it delivers: the population makes the envelope ROBUST rather than resting on one datum, but it does not make the gate independent of the datum. The spread is 0.02 MHz while every member's &#124;d_f0&#124; is about 12.12 MHz, so the envelope is dominated by the RESIDUAL and not by the scatter, and gate = round-up(env x 1.5) is therefore 1.5x the measured agreement. This is a REGRESSION LOCK with 50 percent headroom, not an independent accuracy bound, exactly as the merged case 18's gate is; what gives the measured agreement meaning is not the gate but the comparison of that agreement against an external scale, namely the reference's own 21.9 MHz f0 spread between two independent commercial codes. That tightness is the substance of the result: the residual is a reproducible systematic difference rather than a setup artifact, and at the measured cavity sensitivity of -105 MHz/cell it corresponds to about 0.12 cell of cavity length. The num_periods = 200 run is EXCLUDED from the envelope rather than folded in, because it fails the settling criterion at column power 1.207; it stays committed as the evidence that the settling gate can fire. WHY f0 AND NOT BANDWIDTH, which is the correction this case exists to record: the oracle must be fed the geometry that was BUILT, not the geometry that was DRAWN. Under the #931 lattice ownership contract those are the same geometry -- a PEC volume drawn on node planes realizes tangential walls at BOTH faces and shorts every normal edge between them, so realized == drawn on all three legs -- and this case reads all three off realized_pec_edge_masks instead of asserting a locally written rule. HISTORY, kept because the posture was built on it and the numbers are still in the record: until #931 a body's far face was never a wall, so the three legs disagreed with the drawing and with each other. The transverse aperture leg d_c*dx was confirmed to better than 0.05 cell by an independent refit of 16 committed case-18 configurations during the #499 review. The cavity leg was (L_c + 1)*dx -- the distance between the bounding zeroed node planes -- confirmed to 0.04-0.17 cell, and carried about 105 of the 107.5 MHz that separated a drawn-count oracle from a realized-geometry one. The IRIS-THICKNESS leg fitted neither rule: four independent FDTD runs at drawn t_c = 2/4/6/8 gave a flat offset of -0.66/-0.68/-0.68/-0.70 cell, t_elec about (t_c - 0.68)*dx, matching neither this case's (t_c - 1)*dx nor the merged case 18's t_c*dx, with a residual 10-33x below both. That was recorded as an irreducible half-cell comparator-input uncertainty and never adopted as a fitted parameter, which was the right call: it was not a physical property but the signature of a missing far face plus a corner recipe that put every face half a cell off the node planes. Both are gone. The gated observable is still chosen by SENSITIVITY -- per cell of iris thickness, f0 moves about 2.4 MHz, bandwidth about 40 MHz, individual band edges 22-30 MHz -- so f0 and the zero count carry the gates. Band edges and bandwidth stay REPORTED, and #931 replaces the reason rather than the posture: the input uncertainty that made a gate on them dishonest is removed by the contract and the envelope a gate needs is now measured (17.0553 and 9.9024 MHz over the nine-configuration population), but that population is single-mesh while lattice rounding dominates these two, so the standing objection is sensitivity, not ambiguity, and it wants its own pre-declaration. Handing the oracle drawn counts under the old realization biased f0 by +107.5 MHz, five times the reference's own 21.9 MHz CST-vs-HFSS spread, and the envelope-times-1.5 rule does NOT catch that class because the rule bounds SCATTER and this is BIAS. The realized lengths are read off the realized edge set at build time and re-derived again from the committed node indices in the frozen gate test. Total realized length is now plain addition -- five irises plus four cavities equals the outer extent between the first and last wall plane -- where the retired rule needed a span - 1 and a face-continuity argument to close. An earlier revision claimed that pairing was \"the only pairing that conserves total electrical length\"; that was false, is withdrawn, and the contract makes the question moot. Drawn counts are the plain roundings t_c = round(t/dx), L_c = round(L/dx), d_c = round(d/dx), with NO compensation: the +1 / -1 this case carried until #931 existed only to cancel the missing far face, and deleting it together with the realization it cancelled leaves the built structure unchanged -- same wall planes, same cavities, same apertures. Snapping is still nearest-representable rounding with zero free parameters and no reference number entering, and it is still NOT a monotone improvement across meshes: a/60 and a/90 land on opposite sides of the nominal design. REPORTED, NEVER GATED: individual band edges (+17.05 / +7.20 MHz) and bandwidth (-9.85 MHz), which are ONE fact and not two - d_bw is identically d_hi - d_lo - so the earlier framing of an \"unexplained asymmetric edge residual\" separate from a bandwidth deficit was an algebraic error. #931 discharges BOTH stated blockers on gating them and still does not gate them: the half-cell comparator-input uncertainty is removed by the contract, and the envelope a gate needs is now measured over the nine-configuration population (17.0553 and 9.9024 MHz, so a gate would be 26.0 and 15.0 MHz, committed as gates.edge_bw_gate_would_be_mhz with applied=false). What is left is lattice rounding, which is dominant for these two at 22-40 MHz per cell against f0's 2.4, and the population is single-mesh - every member is a/90 - so a 1.5x lock over it would pin the mesh choice rather than bound the solver. Re-gating them needs its own pre-declaration and a cross-mesh sensitivity measurement, which is separate work; the envelope committed here is what that pre-declaration starts from. Also reported: worst in-band return loss; individual ripple levels; every reflection-zero DEPTH (four nominally identical equiripple zeros bottom out across a wide spread in the published figure, so the paper's frequency step and not physics sets those depths - zero FREQUENCIES are meaningful, depths are not values); passband contiguity beyond the span_holes <= 1 lock (hole depth, hole position, longest contiguous run); the coarse a/60 rung; and phase. PASSBAND CONTIGUITY IS RECORDED, NOT ASSUMED: lo and hi are the OUTERMOST interpolated -10 dB crossings, so the span between them is not necessarily a passband. The built filter has one 10 MHz bin at 9.84 dB inside its 341 MHz span (longest contiguous -10 dB run 270 MHz), while the oracle on the same geometry is contiguous over 35 bins - a real difference that an earlier revision hid behind a clamped statistic, because worst-RL had been computed as the minimum over samples already filtered to >= 10 dB and therefore could not report a violation at all. The coarse a/60 rung has no meaningful passband: 16 of 24 bins in its nominal span are above -10 dB with an interior trough at 2.73 dB, i.e. two separated resonances. That, and not any gate comparison, is the evidence that the gated mesh had to be a/90. The a/60 rung's own numbers do not disqualify it cleanly: its zero count matches its oracle (2 vs 2) and its f0 residual (+19.87 MHz) is the same ~0.12-cell offset seen at a/90; against the committed 19 MHz constant it happens to fail by 0.87 MHz, but a self-derived envelope-times-1.5 gate would pass it. The broken passband is the disqualifier. SETUP IS GATED SEPARATELY FROM PHYSICS, because a resonant band read off an unsettled or absorber-limited run is not a measurement. The repo's preferred ENERGY-BASED ring-down witness (terminal energy in dB below the post-source peak, rule < -40 dB) is NOT AVAILABLE on this path: it is implemented for the lumped/MSL S-matrix extractor, but compute_waveguide_s_matrix returns no settling_db, and the null fields are committed in every row rather than papered over (filed as an rfx capability gap). The ENFORCED settling criterion is therefore the independent axis: the num_periods scan 400/600/800 holds f0 and BW to under 0.1 MHz (the gate is the 400 -> 800 doubling within one 10 MHz bin), the gated run is passivity-clean at column power 1.0065, and the criterion demonstrably fires -- the np=200 run is excluded non-passive at 1.207. The feed-clearance and absorber-depth scans each hold the edges to one 10 MHz bin across their interior and outer samples (standoff 3.05 -> 7.62 -> 15.24 mm: 0.0/0.1 MHz; absorber 0.75 -> 1.00 -> 1.25 lambda_g: 0.0/0.1 MHz). Absorber depth is scanned because in S1 it was the envelope-limiting term at 0.5 lambda_g; here 0.75 lambda_g is measurably sufficient, which is a negative result worth recording rather than a rule inherited. Extractor warnings and the passivity footprint (the bins where column power exceeds 1.02, not merely the scalar maximum) are committed per row. Guide height is reduced to 4 cells on a MEASURED b-invariance witness: b = 4 and b = 8 agree to 233 Hz in f0 on THIS resonant five-iris filter, not merely on the single iris where it was first measured, which is the 8x saving that makes the case affordable to generate. THE ORACLE HAS HAD AN ADVERSARIAL PASS, and it did not find the residual. Its own witnesses are unitarity 2.2e-15, reciprocity and mirror symmetry exact, and an L -> 0 collapse closing two thin irises onto one thick one; its N=1 centred limit reproduces the merged case-18 oracle to 1.05e-04 in an independent odd-mode formulation, and THAT object is what PR #480 confirmed against a formulation-independent 2-D H-plane FDFD at 5.8e-4 - a comparison the frozen gate test now EXECUTES rather than asserting in prose. Those witnesses have known limits, stated because an earlier revision leaned on them too hard: unitarity constrains only the propagating sub-block, mirror symmetry holds by construction for a symmetric geometry, and injected overlap-integral errors leave the reduction and collapse axes silent because they share the overlap routine, so the gate test's re-typed cascade agreeing to 0.0e+00 is a REGRESSION LOCK and not a second opinion. An independent review then closed the real gaps. What is COMMITTED: the overlap integral is validated against direct numerical quadrature IN CI (216 combinations across three apertures including the as-realized 6.096 mm -- where n*pi/a equals m*pi/d exactly for six mode pairs -- centred and 0.19 mm off-centre, worst deviation bounded at 1e-12 with the small-denominator guard exercised), and the oracle's truncation is witnessed on the GATED observable at generation time: f0 by bisection moves 0.33 MHz for n_a 90 -> 130 and 1.85 MHz for an aperture-mode-count doubling -- the aperture axis is the sensitive one, and both are an order under the 19 MHz gate. Session measurements recorded in the research notes but NOT in this record put the gauge invariance of the sqrt(Y) normalisation at machine precision, truncation saturation near 1 MHz in bandwidth, and inter-cavity evanescent transport near 1.6 MHz; they are corroborating colour rather than load-bearing, because the formulation-level check below subsumes their role. THE FORMULATION-LEVEL CHECK IS NOW DONE, and it lands the residual on the rfx side. A 2-D H-plane FDFD -- scalar Helmholtz on a finite-difference grid with an exact discrete transparent port condition, sharing only numpy and scipy with the cascade and no rfx code path at all (validation/crossval/comparators/fdfd_hplane.py) -- was run on the same electrical geometry, grid-exact at every refinement level. It is FIRST-order convergent, so no single level is meaningful; the record carries THREE levels (r = 2, 3, 4, whose bandwidth deviations from the extrapolate shrink as 1/r: measured ratios 1.55 and 1.33 against the first-order 1.50 and 1.33) and BOTH Richardson estimates, which agree to 0.37 MHz in centre frequency and 0.36 MHz in bandwidth -- the two-estimate consistency protocol the porting handoff mandates before either estimate is trusted. The finer pair gives f0 = 10.95742 GHz and BW = 351.42 MHz against the cascade's 10.95851 and 350.43: agreement to 1.09 MHz in centre frequency and 0.98 MHz in bandwidth between two formulations that share nothing but their numerical libraries, with the extrapolation's own order-assumption uncertainty, not the two-estimate consistency, bounding how much of that gap the FDFD itself owns: the consistency (0.37/0.36 MHz) is agreement between two fits of the SAME first-order model, not an error bound -- fitting the convergence order to the three committed levels edge-by-edge gives p = 1.26 (lower edge) / 1.24 (upper edge) rather than the assumed 1, and re-extrapolating the finer pair at the fitted order moves f0 by +0.68 MHz and bandwidth by -0.65 MHz, so the FDFD owns roughly 0.7-1.0 MHz of the 1.09/0.98 MHz gap; f0 and bandwidth are also algebraic combinations of the SAME two band edges, one confirmation and not two. rfx differs from the FDFD by +13.21 MHz in f0 and -10.83 MHz in bandwidth, essentially the same as it differs from the cascade (+12.12 / -9.85), so the 12 MHz residual is not an oracle error. The FDFD's gates: lossless unitarity is enforced on EVERY evaluation (worst 5.0e-07 across all levels), and the empty-guide transparency gate -- &#124;S11&#124; = 5.0e-14 with &#124;S21&#124; = 1.000000000000, the test that originally caught a missing 1/h in the discrete propagation constant -- runs once per generation and once per CI pass. One defect in the comparator itself was found by an independent port review and fixed before this record was generated: its aperture mask realized every aperture two fine cells wide of the stated convention, a first-order bias that Richardson cancelled -- making the extrapolated numbers right for the wrong per-level geometry -- and that produced a spurious FOURTH reflection zero at the coarser levels. With the mask exact, every level shows THREE zeros, matching the cascade and rfx. Independently, the cascade's zero count was checked against its own aperture-mode truncation, which nobody had done for the COUNT: it is 3 at nb_scale 1.0, 1.5, 2.0 and 3.0, with f0 moving 1.7 MHz over that 3x range. What remains genuinely unexplained is the ~12 MHz rfx residual itself: it is mesh-invariant when expressed in cells (-0.117 cell at a/90 against -0.124 at a/60 on the same measured per-mesh cavity sensitivity, where dispersion would have given 0.083), so it behaves like a fixed geometric offset rather than a frequency-dependent solver error, but attributing it to a specific convention leg had FAILED under the old realization: propagating the independently measured iris thickness (t_c - 0.68)*dx through node-plane length conservation overshot and flipped the sign, taking a/90 from +12.08 to -30.62 MHz, and that attribution was recorded as falsified rather than pending. Under the #931 contract the convention legs are no longer free at all -- realized == drawn on every one -- so a residual that survives the regeneration is not a convention artifact. Whether it survived is stated with the regenerated numbers. _gamma at exact cutoff (k equal to n*pi/w, where the sqrt argument vanishes) is unreachable at these band edges and untested. FENCED: nothing here promotes the lane beyond S1. Multi-iris filters, posts and septa remain EXPERIMENTAL; this measures one published design on one mesh with one gated observable, and certifies neither arbitrary filters nor the a/60 rung. Says nothing about phase, group delay, loss, higher-order-mode ports, or fabrication tolerance. The reference is an ANCHOR, not a solver run: the CST and HFSS scalars are digitized from the paper's Fig. 5, no external solver is invoked here, and the case does not compare rfx against either commercial code on any geometry - so the 12.1 MHz f0 residual against this case's own analytic oracle must not be read as an accuracy claim relative to CST or HFSS. Reported for context and not as a yardstick beaten: the oracle on nominal dimensions sits -6.2 MHz from CST and -28.2 MHz from HFSS in f0, +14.7 MHz from both in bandwidth (against a published inter-solver bandwidth spread of only 0.4 MHz), -0.4 and -1.1 dB in worst return loss, and up to 25.4 and 61.9 MHz in individual reflection-zero frequencies. The bandwidth and zero-frequency misses are larger than the f0 miss and are stated here because quoting only f0 would be selective. The built structure is a SNAPPED Aghanim filter: its centre frequency is within the reference's own solver scatter of CST (though not of HFSS), one of the four structural reflection zeros is lost (4 -> 3, confirmed grid-robust by refining the oracle to 1 MHz, with the loss occurring in the upper band), and worst in-band return loss degrades from 13.82 dB to 10.65 dB by rasterization alone, oracle to oracle, with rfx at 9.84 dB. Say snapped, not equivalent. OBSERVABLE PRIORITY for a resonant structure, as this case measures it: the structural reflection-zero COUNT first (an integer, depth-independent, and shown grid-robust), then centre frequency (least sensitive of the continuous quantities to a length error, ~2.4 MHz per cell of iris thickness), then band edges and bandwidth (~22-40 MHz per cell, hence reported), then worst return loss, and last individual ripple levels and null depths, which are not values at all. TOPOLOGY FIRST, AND f0 IS NOT EXONERATED: the zero count is the most robust observable, but f0 is not thereby safe -- it carries the +12.12 MHz residual this case gates, and at -105 MHz per cell of cavity length it is the quantity a geometry error moves first. A cell snap is inherently non-uniform, since each cavity rounds independently, so every snap figure quoted here was MEASURED on the as-snapped geometry and none may be re-derived by multiplying a sensitivity coefficient by a half cell."` | — |
| `/coarse_diagnostic/aperture_nodes` | `[[17, 42], [22, 37], [23, 37], [22, 37], [17, 42]]` | *absent* | 0–4 |
| `/coarse_diagnostic/aperture_wall_nodes` | *absent* | `[[16, 43], [21, 38], [22, 38], [21, 38], [16, 43]]` | 0–4 |
| `/coarse_diagnostic/band/bw` | `236253485.27890015` | `236295820.43458366` | — |
| `/coarse_diagnostic/band/f0` | `11095241431.043194` | `11095267252.543339` | — |
| `/coarse_diagnostic/band/hi` | `11213368173.682644` | `11213415162.760632` | — |
| `/coarse_diagnostic/band/lo` | `10977114688.403744` | `10977119342.326048` | — |
| `/coarse_diagnostic/band/worst_rl_db` | `2.7303308022352253` | `2.7314608133164775` | — |
| `/coarse_diagnostic/d_bw_mhz` | `-9.3` | `-9.26` | — |
| `/coarse_diagnostic/d_hi_mhz` | `15.2` | `15.24` | — |
| `/coarse_diagnostic/d_lo_mhz` | `24.5` | `24.51` | — |
| `/coarse_diagnostic/extractor_warnings` | array[88], SHA256 `8269bb2c8308` | array[89], SHA256 `2425b81e95ce` | 0, 2, 4–6, 8, 14–16, 18, 22, 24, 26–28, 30, 36–38, 40, 44, 46, 48–50, 52, 58–60, 62, 66, 68, 70–72, 74, 80–82, 84, 88 |
| `/coarse_diagnostic/glen_cells` | `264` | `263` | — |
| `/coarse_diagnostic/grid` | `[413, 61, 5]` | `[412, 61, 5]` | 0 |
| `/coarse_diagnostic/iris_wall_nodes` | *absent* | `[[114, 119], [157, 162], [203, 208], [249, 254], [292, 297]]` | 0–4 |
| `/coarse_diagnostic/iris_x_nodes` | `[[114, 119], [157, 162], [203, 208], [249, 254], [292, 297]]` | *absent* | 0–4 |
| `/coarse_diagnostic/oracle_s11` | array[131], SHA256 `c89658b611e5` | array[131], SHA256 `7eb1d7ec3983` | 3, 10, 12–13, 21, 23, 26, 34, 42–43, 47, 52–53, 56, 66, 73, 79, 87–89, 92, 105–106, 109, 121 |
| `/coarse_diagnostic/s11` | array[131], SHA256 `56fb3869ff2d` | array[131], SHA256 `dde625f5cdfb` | 0–130 |
| `/coarse_diagnostic/s21` | array[131], SHA256 `a52cc3e02f69` | array[131], SHA256 `ce7f9f11c2f5` | 0–113, 115–122, 124–130 |
| `/coarse_diagnostic/settling_db` | `null` | `[-50.4, -50.4]` | 0–1 |
| `/coarse_diagnostic/wall_s` | `445.5` | `255.7` | — |
| `/electrical_geometry/aperture_cells` | *absent* | `[40, 26, 24, 26, 40]` | 0–4 |
| `/electrical_geometry/aperture_wall_nodes` | *absent* | `[[25, 65], [32, 58], [33, 57], [32, 58], [25, 65]]` | 0–4 |
| `/electrical_geometry/compensation` | `"drawn counts are chosen so the ELECTRICAL dimensions land on nominal: t_c = round(t/dx) + 1, L_c = round(L/dx) - 1. At a/90 that puts f0 +3.3 MHz from the paper's exact design (inside its own 21.9 MHz CST-vs-HFSS spread) against -101.4 MHz uncompensated. Compensation is NOT a monotone improvement -- it only picks which side of the sub-cell rounding you land on, and at a/60 it moves f0 from -35.8 to +120.2 MHz."` | `"none. Drawn counts are the plain roundings t_c = round(t/dx), L_c = round(L/dx), d_c = round(d/dx), and under the contract they are also the realized ones. Until #931 this case drew t_c = round(t/dx) + 1 and L_c = round(L/dx) - 1 so the ELECTRICAL dimensions would land on nominal against a realization that lost one plane per body; the compensation and that realization are deleted together and the built structure is unchanged. Snapping remains a rounding and is NOT a monotone improvement across meshes: a/90 and a/60 land on opposite sides of the nominal design."` | — |
| `/electrical_geometry/cost_note` | `"measured 2026-07-29: feeding the oracle the DRAWN cell counts puts its band at 10.9054-11.2267 GHz against 10.7833-11.1337 GHz for the realized geometry, a +107.5 MHz f0 error -- five times the paper's own 21.9 MHz CST-vs-HFSS spread, and it would have had to be absorbed by a ~162 MHz gate (46% of the passband) that pins nothing. First found as +90.0 MHz on the uncompensated counts; compensation changes which pair is confused, not the class."` | `"HISTORICAL, measured 2026-07-29 under the pre-#931 realization: feeding the oracle the DRAWN cell counts put its band at 10.9054-11.2267 GHz against 10.7833-11.1337 GHz for the realized geometry, a +107.5 MHz f0 error -- five times the paper's own 21.9 MHz CST-vs-HFSS spread, and it would have had to be absorbed by a ~162 MHz gate (46% of the passband) that pins nothing. First found as +90.0 MHz on the uncompensated counts; compensation changed which pair was confused, not the class. The contract removes the pair: there is one geometry, drawn and realized, and this number is kept as the record of what the defect was worth."` | — |
| `/electrical_geometry/drawn_aperture_cells` | *absent* | `[40, 26, 24, 26, 40]` | 0–4 |
| `/electrical_geometry/drawn_cavity_cells` | `[55, 61, 61, 55]` | `[56, 62, 62, 56]` | 0–3 |
| `/electrical_geometry/drawn_iris_thickness_cells` | `9` | `8` | — |
| `/electrical_geometry/iris_wall_nodes` | *absent* | `[[150, 158], [214, 222], [284, 292], [354, 362], [418, 426]]` | 0–4 |
| `/electrical_geometry/rule` | `"oracle inputs are READ BACK off the rasterized metal. The CAVITY leg is (L_c + 1)*dx -- the distance between the bounding zeroed node planes -- and is confirmed to 0.04-0.17 cell by the committed residual against the measured cavity sensitivity; the transverse APERTURE leg d_c*dx is confirmed to better than 0.05 cell by an independent refit of 16 committed case-18 configurations. The IRIS-THICKNESS leg is NOT (t_c - 1)*dx: four FDTD runs at drawn t_c = 2/4/6/8 give a flat offset of -0.66/-0.68/-0.68/-0.70 cell, i.e. t_elec ~ (t_c - 0.68)*dx, matching neither this rule nor case 18's t_c*dx. That ~1/3-cell ambiguity is an irreducible comparator-input uncertainty here and it is why bandwidth and individual band edges are REPORTED rather than gated (they move ~40 and ~22-30 MHz per cell of it) while f0 and the zero count are gated (~2.4 MHz per cell, and an integer). Total electrical length is a face-continuity CHECK across region types, NOT a uniqueness argument: putting the interface at sigma*dx beyond the outermost metal node gives total = span - 1 + 2*sigma = the outer extent measured at the same sigma, conserved for EVERY sigma, and sigma = 0.5 is the drawn pairing itself."` | `"oracle inputs are READ BACK off the REALIZED PEC edge set (rfx.boundaries.pec.realized_pec_edge_masks through validation/crossval/_wr90_iris_realized.py), never off the drawn counts. Under the #931 lattice ownership contract a PEC volume drawn on node planes realizes tangential walls at BOTH faces and shorts every normal edge between them, so all three legs are the drawn ones: iris thickness t_c*dx, cavity L_c*dx, aperture d_c*dx. raster_assert asserts that identity per iris, per cavity and per aperture at build time, with no solve. HISTORY: before #931 a body's far face was never a wall, so the realized cavity was (L_c + 1)*dx and the realized iris (t_c - 1)*dx against the drawing, this case carried a +1/-1 compensation in the drawn counts to land the electrical dimensions on nominal, and the iris-thickness leg fitted neither rule (four FDTD runs at drawn t_c = 2/4/6/8 gave t_elec ~ (t_c - 0.68)*dx, matching neither this rule nor case 18's t_c*dx). That ~1/3-cell offset was read as an irreducible comparator-input uncertainty; it is better read as the missing far face plus a corner recipe that placed every face half a cell off the node planes. Both are gone, and with them the compensation and the face-continuity argument that closed the old total-length identity through a span - 1. Total realized length is now plain addition: five irises plus four cavities equals the outer extent between the first and last wall plane."` | — |
| `/fdfd_formulation_independent/d_bw_vs_cascade_mhz` | `0.985` | `0.984` | — |
| `/fdfd_formulation_independent/d_f0_rfx_vs_fdfd_mhz` | `13.171` | `13.209` | — |
| `/fdfd_formulation_independent/levels/2/band/bw` | `346043170.02007294` | `346043218.8118229` | — |
| `/fdfd_formulation_independent/levels/2/band/f0` | `10963397722.378645` | `10963397746.77452` | — |
| `/fdfd_formulation_independent/levels/2/band/hi` | `11136419307.388681` | `11136419356.180431` | — |
| `/fdfd_formulation_independent/levels/2/s11` | array[131], SHA256 `e086d8743a0e` | array[131], SHA256 `394d7de7f64c` | 41, 71, 73, 109 |
| `/fdfd_formulation_independent/levels/2/wall_s` | `385.4` | `121.1` | — |
| `/fdfd_formulation_independent/levels/2/worst_unitarity` | `2.910083445328837e-07` | `2.5891862298621504e-07` | — |
| `/fdfd_formulation_independent/levels/3/band/bw` | `347953193.3703842` | `347953184.3007183` | — |
| `/fdfd_formulation_independent/levels/3/band/f0` | `10961284753.373653` | `10961284757.908485` | — |
| `/fdfd_formulation_independent/levels/3/band/lo` | `10787308156.688461` | `10787308165.758127` | — |
| `/fdfd_formulation_independent/levels/3/s11` | array[131], SHA256 `886aa6f2b8e4` | array[131], SHA256 `dd15808dcadb` | 8, 38, 40, 44, 46–47, 49, 54, 57–58, 64, 70, 72, 75, 77 |
| `/fdfd_formulation_independent/levels/3/wall_s` | `1186.6` | `308.1` | — |
| `/fdfd_formulation_independent/levels/3/worst_unitarity` | `4.2312295167601377e-07` | `3.9601045442871907e-07` | — |
| `/fdfd_formulation_independent/levels/4/band/bw` | `348818859.72759247` | `348818754.1513157` | — |
| `/fdfd_formulation_independent/levels/4/band/f0` | `10960319536.995258` | `10960319521.61694` | — |
| `/fdfd_formulation_independent/levels/4/band/hi` | `11134728966.859055` | `11134728898.692596` | — |
| `/fdfd_formulation_independent/levels/4/band/lo` | `10785910107.131462` | `10785910144.54128` | — |
| `/fdfd_formulation_independent/levels/4/s11` | array[131], SHA256 `15d6d2735350` | array[131], SHA256 `c6d33eb5ed88` | 39–40, 51, 63–64, 68–69, 71, 73, 78, 90 |
| `/fdfd_formulation_independent/levels/4/wall_s` | `2540.5` | `601.9` | — |
| `/fdfd_formulation_independent/levels/4/worst_unitarity` | `4.5658172764806437e-07` | `5.024902829386946e-07` | — |
| `/fdfd_formulation_independent/richardson_23/bw` | `351773240.0710068` | `351773115.27850914` | — |
| `/fdfd_formulation_independent/richardson_23/f0` | `10957058815.36367` | `10957058780.176414` | — |
| `/fdfd_formulation_independent/richardson_23/hi` | `11132945435.399174` | `11132945337.815674` | — |
| `/fdfd_formulation_independent/richardson_23/lo` | `10781172195.328167` | `10781172222.537167` | — |
| `/fdfd_formulation_independent/richardson_34/bw` | `351415858.7992172` | `351415463.70310783` | — |
| `/fdfd_formulation_independent/richardson_34/f0` | `10957423887.860073` | `10957423812.742302` | — |
| `/fdfd_formulation_independent/richardson_34/hi` | `11133131817.259682` | `11133131544.59385` | — |
| `/fdfd_formulation_independent/richardson_34/lo` | `10781715958.460464` | `10781716080.89074` | — |
| `/fdfd_formulation_independent/richardson_consistency_mhz/bw` | `0.357` | `0.358` | — |
| `/fdfd_formulation_independent/self_test/empty_s11` | `4.998689747642886e-14` | `4.9977732337688505e-14` | — |
| `/fdfd_formulation_independent/self_test/empty_s21` | `1.0000000000000018` | `1.0000000000000016` | — |
| `/fdfd_formulation_independent/self_test/unitarity` | `1.4655321400880439e-09` | `2.3153723383018132e-09` | — |
| `/feed_clearance_witness/d_hi_mhz` | `0.11` | `0.05` | — |
| `/feed_clearance_witness/d_lo_mhz` | `0.03` | `0.01` | — |
| `/feed_clearance_witness/gated/hi` | `11140816346.186275` | `11140924560.619234` | — |
| `/feed_clearance_witness/gated/lo` | `10800373669.774708` | `10800340317.655672` | — |
| `/feed_clearance_witness/generous/d_hi_mhz` | `0.109` | `0.001` | — |
| `/feed_clearance_witness/generous/d_lo_mhz` | `0.032` | `0.003` | — |
| `/feed_clearance_witness/generous/hi` | `11140925564.344723` | `11140923080.683405` | — |
| `/feed_clearance_witness/generous/lo` | `10800341884.471735` | `10800337332.294498` | — |
| `/feed_clearance_witness/generous/s11` | array[131], SHA256 `b2a8527e031c` | array[131], SHA256 `29d2f24c0cd1` | 0–13, 15–99, 102, 104–108, 110–117, 119–126, 128–130 |
| `/feed_clearance_witness/generous/s21` | array[131], SHA256 `6c47bbc339ee` | array[131], SHA256 `a3b7e36e7283` | 0–7, 10–118, 120–122, 124–126, 128–130 |
| `/feed_clearance_witness/generous/wall_s` | `1308.9` | `977.5` | — |
| `/feed_clearance_witness/mid/d_hi_mhz` | `0.016` | `0.045` | — |
| `/feed_clearance_witness/mid/d_lo_mhz` | `0.003` | `0.01` | — |
| `/feed_clearance_witness/mid/hi` | `11140800129.524334` | `11140879350.558094` | — |
| `/feed_clearance_witness/mid/lo` | `10800371060.821806` | `10800350360.551697` | — |
| `/feed_clearance_witness/mid/s11` | array[131], SHA256 `295d3b127ebf` | array[131], SHA256 `a2915cf7ab01` | 0–26, 28–96, 98–99, 101–109, 111–113, 115–124, 126–130 |
| `/feed_clearance_witness/mid/s21` | array[131], SHA256 `01ffc558141f` | array[131], SHA256 `9a7f42f34830` | 0–5, 7–11, 13–113, 115–123, 125–130 |
| `/feed_clearance_witness/mid/wall_s` | `1235.5` | `907.8` | — |
| `/gated_rfx/aperture_nodes` | `[[26, 64], [33, 57], [34, 56], [33, 57], [26, 64]]` | *absent* | 0–4 |
| `/gated_rfx/aperture_wall_nodes` | *absent* | `[[25, 65], [32, 58], [33, 57], [32, 58], [25, 65]]` | 0–4 |
| `/gated_rfx/band/bw` | `340442676.4115677` | `340584242.963562` | — |
| `/gated_rfx/band/f0` | `10970595007.980492` | `10970632439.137453` | — |
| `/gated_rfx/band/hi` | `11140816346.186275` | `11140924560.619234` | — |
| `/gated_rfx/band/lo` | `10800373669.774708` | `10800340317.655672` | — |
| `/gated_rfx/band/worst_rl_db` | `9.802406901625071` | `9.83633015873974` | — |
| `/gated_rfx/d_bw_mhz` | `-9.99` | `-9.85` | — |
| `/gated_rfx/d_f0_mhz` | `12.08` | `12.12` | — |
| `/gated_rfx/d_hi_mhz` | `7.09` | `7.2` | — |
| `/gated_rfx/d_lo_mhz` | `17.08` | `17.05` | — |
| `/gated_rfx/extractor_warnings` | array[88], SHA256 `8269bb2c8308` | array[89], SHA256 `a623f30a2587` | 0, 2, 4–6, 8, 14–16, 18, 22, 24, 26–28, 30, 36–38, 40, 44, 46, 48–50, 52, 58–60, 62, 66, 68, 70–72, 74, 80–82, 84, 88 |
| `/gated_rfx/glen_cells` | `357` | `356` | — |
| `/gated_rfx/grid` | `[578, 91, 5]` | `[577, 91, 5]` | 0 |
| `/gated_rfx/iris_wall_nodes` | *absent* | `[[150, 158], [214, 222], [284, 292], [354, 362], [418, 426]]` | 0–4 |
| `/gated_rfx/iris_x_nodes` | `[[150, 158], [214, 222], [284, 292], [354, 362], [418, 426]]` | *absent* | 0–4 |
| `/gated_rfx/oracle_s11` | array[131], SHA256 `8742d7e4cd9e` | array[131], SHA256 `bcb27e2f197d` | 6, 28, 31, 33, 35, 55, 76, 82, 87, 89, 95, 98, 100, 102, 105–106, 117, 130 |
| `/gated_rfx/s11` | array[131], SHA256 `19dd17dae901` | array[131], SHA256 `a8d2053b1cec` | 0–18, 20–26, 28–99, 101–106, 108–113, 115–122, 124–130 |
| `/gated_rfx/s21` | array[131], SHA256 `74e9a501b715` | array[131], SHA256 `b14514c8010c` | 0–130 |
| `/gated_rfx/settling_db` | `null` | `[-59.23, -59.23]` | 0–1 |
| `/gated_rfx/wall_s` | `1160.3` | `850.0` | — |
| `/gates/bw_measured_envelope_mhz` | *absent* | `9.9024` | — |
| `/gates/bw_reported_residual_mhz` | `9.99` | *absent* | — |
| `/gates/edge_bw_envelope_population/0/config` | *absent* | `"gated a/90 np400 b4"` | — |
| `/gates/edge_bw_envelope_population/0/d_bw_mhz` | *absent* | `-9.8471` | — |
| `/gates/edge_bw_envelope_population/0/d_hi_mhz` | *absent* | `7.1982` | — |
| `/gates/edge_bw_envelope_population/0/d_lo_mhz` | *absent* | `17.0453` | — |
| `/gates/edge_bw_envelope_population/1/config` | *absent* | `"ring np600"` | — |
| `/gates/edge_bw_envelope_population/1/d_bw_mhz` | *absent* | `-9.8641` | — |
| `/gates/edge_bw_envelope_population/1/d_hi_mhz` | *absent* | `7.1698` | — |
| `/gates/edge_bw_envelope_population/1/d_lo_mhz` | *absent* | `17.034` | — |
| `/gates/edge_bw_envelope_population/2/config` | *absent* | `"ring np800"` | — |
| `/gates/edge_bw_envelope_population/2/d_bw_mhz` | *absent* | `-9.864` | — |
| `/gates/edge_bw_envelope_population/2/d_hi_mhz` | *absent* | `7.1699` | — |
| `/gates/edge_bw_envelope_population/2/d_lo_mhz` | *absent* | `17.0339` | — |
| `/gates/edge_bw_envelope_population/3/config` | *absent* | `"b=6 cells"` | — |
| `/gates/edge_bw_envelope_population/3/d_bw_mhz` | *absent* | `-9.8469` | — |
| `/gates/edge_bw_envelope_population/3/d_hi_mhz` | *absent* | `7.1985` | — |
| `/gates/edge_bw_envelope_population/3/d_lo_mhz` | *absent* | `17.0454` | — |
| `/gates/edge_bw_envelope_population/4/config` | *absent* | `"b=8 cells"` | — |
| `/gates/edge_bw_envelope_population/4/d_bw_mhz` | *absent* | `-9.8477` | — |
| `/gates/edge_bw_envelope_population/4/d_hi_mhz` | *absent* | `7.1976` | — |
| `/gates/edge_bw_envelope_population/4/d_lo_mhz` | *absent* | `17.0454` | — |
| `/gates/edge_bw_envelope_population/5/config` | *absent* | `"mid feed"` | — |
| `/gates/edge_bw_envelope_population/5/d_bw_mhz` | *absent* | `-9.9024` | — |
| `/gates/edge_bw_envelope_population/5/d_hi_mhz` | *absent* | `7.153` | — |
| `/gates/edge_bw_envelope_population/5/d_lo_mhz` | *absent* | `17.0553` | — |
| `/gates/edge_bw_envelope_population/6/config` | *absent* | `"generous feed"` | — |
| `/gates/edge_bw_envelope_population/6/d_bw_mhz` | *absent* | `-9.8456` | — |
| `/gates/edge_bw_envelope_population/6/d_hi_mhz` | *absent* | `7.1967` | — |
| `/gates/edge_bw_envelope_population/6/d_lo_mhz` | *absent* | `17.0423` | — |
| `/gates/edge_bw_envelope_population/7/config` | *absent* | `"mid absorber"` | — |
| `/gates/edge_bw_envelope_population/7/d_bw_mhz` | *absent* | `-9.8591` | — |
| `/gates/edge_bw_envelope_population/7/d_hi_mhz` | *absent* | `7.189` | — |
| `/gates/edge_bw_envelope_population/7/d_lo_mhz` | *absent* | `17.0481` | — |
| `/gates/edge_bw_envelope_population/8/config` | *absent* | `"deep absorber"` | — |
| `/gates/edge_bw_envelope_population/8/d_bw_mhz` | *absent* | `-9.8655` | — |
| `/gates/edge_bw_envelope_population/8/d_hi_mhz` | *absent* | `7.1841` | — |
| `/gates/edge_bw_envelope_population/8/d_lo_mhz` | *absent* | `17.0496` | — |
| `/gates/edge_bw_gate_would_be_mhz/applied` | *absent* | `false` | — |
| `/gates/edge_bw_gate_would_be_mhz/bw` | *absent* | `15.0` | — |
| `/gates/edge_bw_gate_would_be_mhz/edges` | *absent* | `26.0` | — |
| `/gates/edge_bw_gate_would_be_mhz/why_not` | *absent* | `"the population is single-mesh (every member a/90) while lattice rounding is the dominant term for these two observables at 22-40 MHz per cell, so a 1.5x lock over it would pin the mesh choice rather than bound the solver; re-gating needs its own pre-declaration and a cross-mesh sensitivity measurement"` | — |
| `/gates/edge_measured_envelope_mhz` | *absent* | `17.0553` | — |
| `/gates/edge_reported_residual_mhz` | `17.08` | *absent* | — |
| `/gates/f0_envelope_population/0/d_f0_mhz` | `12.0843` | `12.1217` | — |
| `/gates/f0_envelope_population/1/d_f0_mhz` | `12.064` | `12.1019` | — |
| `/gates/f0_envelope_population/2/d_f0_mhz` | `12.0641` | `12.1019` | — |
| `/gates/f0_envelope_population/3/d_f0_mhz` | `12.0843` | `12.1219` | — |
| `/gates/f0_envelope_population/4/d_f0_mhz` | `12.0842` | `12.1215` | — |
| `/gates/f0_envelope_population/5/d_f0_mhz` | `12.0749` | `12.1042` | — |
| `/gates/f0_envelope_population/6/d_f0_mhz` | `12.123` | `12.1195` | — |
| `/gates/f0_envelope_population/7/d_f0_mhz` | `12.0647` | `12.1185` | — |
| `/gates/f0_envelope_population/8/d_f0_mhz` | `12.0605` | `12.1169` | — |
| `/gates/f0_measured_envelope_mhz` | `12.123` | `12.1219` | — |
| `/gates/f0_population_excluded/0/max_colpow` | `1.207` | `1.2071` | — |
| `/gates/posture` | `"gate = round-UP(measured envelope x 1.5) over a MULTI-CONFIGURATION population, enforced as EXACT equality by the write-fixture self-check. That makes the envelope robust rather than resting on one datum, but it does NOT make the gate independent of the datum: the population spread is 0.06 MHz while every member is about 12.08 MHz from the oracle, so the envelope is dominated by the residual and the gate is 1.5x the measured agreement. It is a REGRESSION LOCK with 50% headroom, not an independent accuracy bound. What gives the agreement meaning is its comparison against an external scale (the reference's own 21.9 MHz f0 spread between two independent commercial codes), not the gate. GATED: centre frequency f0; the structural zero COUNT (witnessed invariant across the 8.00-8.50-cell iris-thickness ambiguity band, see iris_thickness_zero_count_sweep); and passband contiguity as a regression lock (span_holes <= 1, the committed envelope -- the f0 gate alone cannot see a split passband because band edges are the outermost crossings), against the oracle on as-realized geometry. REPORTED, never gated: individual band edges and bandwidth (their comparator-input uncertainty from the unsettled iris-thickness convention, ~20 MHz, exceeds any defensible gate on them, and d_bw is identically d_hi - d_lo so they are one fact), worst-case RL, ripple levels, zero depths, contiguity detail beyond the span_holes lock, the coarse rung and phase"` | `"gate = round-UP(measured envelope x 1.5) over a MULTI-CONFIGURATION population, enforced as EXACT equality by the write-fixture self-check. That makes the envelope robust rather than resting on one datum, but it does NOT make the gate independent of the datum: the population spread is 0.02 MHz while every member is about 12.12 MHz from the oracle, so the envelope is dominated by the residual and the gate is 1.5x the measured agreement. It is a REGRESSION LOCK with 50% headroom, not an independent accuracy bound. What gives the agreement meaning is its comparison against an external scale (the reference's own 21.9 MHz f0 spread between two independent commercial codes), not the gate. GATED: centre frequency f0; the structural zero COUNT (witnessed invariant across a one-cell iris-thickness band centred on the realized thickness, see iris_thickness_zero_count_sweep); and passband contiguity as a regression lock (span_holes <= 1, the committed envelope -- the f0 gate alone cannot see a split passband because band edges are the outermost crossings), against the oracle on as-realized geometry. REPORTED, never gated: individual band edges and bandwidth -- #931 removes the half-cell input uncertainty that was the original reason, and the regenerated record now MEASURES their envelope (17.0553 and 9.9024 MHz over the nine-configuration population, a gate would be 26.0 and 15.0, committed as edge_bw_gate_would_be_mhz with applied=false), but the population is single-mesh while lattice rounding dominates these two at 22-40 MHz per cell, so a lock over it would pin the mesh rather than bound the solver; re-gating needs its own pre-declaration and a cross-mesh sensitivity measurement. d_bw is identically d_hi - d_lo so they are one fact. Also reported: worst-case RL, ripple levels, zero depths, contiguity detail beyond the span_holes lock, the coarse rung and phase"` | — |
| `/iris_thickness_zero_count_sweep/note` | `"oracle-side robustness witness for the ZERO-COUNT gate (post-merge joint review, N3): the iris-thickness electrical leg is the one unsettled convention input, so the gated integer must not depend on which convention the comparator picks. t_elec swept 8.00-8.50 cells -- the built (t_c - 1)*dx rule at 8.00, the measured (t_c - 0.68)*dx offset at 8.32 -- on the committed frequency grid; oracle evaluations only, no FDTD."` | `"oracle-side robustness witness for the ZERO-COUNT gate (post-merge joint review, N3; re-centred for #931): the gated integer must survive a perturbation of the comparator's most length-sensitive input. t_elec swept over one full cell CENTRED on the realized iris thickness (realized -0.5 .. +0.5 cell, eleven points) on the committed frequency grid; oracle evaluations only, no FDTD. Before #931 the window was one-sided, 8.00-8.50 cells, spanning the disagreement between this case's built (t_c - 1)*dx rule and the measured (t_c - 0.68)*dx offset; the lattice ownership contract makes the realized thickness exact, so there is no ambiguity band left to span and the remaining question is symmetric sensitivity."` | — |
| `/iris_thickness_zero_count_sweep/rows/0/bw_hz` | `350431353` | `371605947` | — |
| `/iris_thickness_zero_count_sweep/rows/0/f0_hz` | `10958510693` | `10957110862` | — |
| `/iris_thickness_zero_count_sweep/rows/0/t_elec_cells` | `8.0` | `7.5` | — |
| `/iris_thickness_zero_count_sweep/rows/1/bw_hz` | `348350618` | `367450743` | — |
| `/iris_thickness_zero_count_sweep/rows/1/f0_hz` | `10958634569` | `10957252317` | — |
| `/iris_thickness_zero_count_sweep/rows/1/t_elec_cells` | `8.05` | `7.6` | — |
| `/iris_thickness_zero_count_sweep/rows/2/bw_hz` | `346177876` | `363001206` | — |
| `/iris_thickness_zero_count_sweep/rows/2/f0_hz` | `10958798136` | `10957613410` | — |
| `/iris_thickness_zero_count_sweep/rows/2/t_elec_cells` | `8.1` | `7.7` | — |
| `/iris_thickness_zero_count_sweep/rows/3/bw_hz` | `343911491` | `358268053` | — |
| `/iris_thickness_zero_count_sweep/rows/3/f0_hz` | `10959001732` | `10958203195` | — |
| `/iris_thickness_zero_count_sweep/rows/3/t_elec_cells` | `8.15` | `7.8` | — |
| `/iris_thickness_zero_count_sweep/rows/4/bw_hz` | `341547717` | `354311209` | — |
| `/iris_thickness_zero_count_sweep/rows/4/f0_hz` | `10959246710` | `10958386496` | — |
| `/iris_thickness_zero_count_sweep/rows/4/t_elec_cells` | `8.2` | `7.9` | — |
| `/iris_thickness_zero_count_sweep/rows/5/bw_hz` | `339161490` | `350431353` | — |
| `/iris_thickness_zero_count_sweep/rows/5/f0_hz` | `10959575668` | `10958510693` | — |
| `/iris_thickness_zero_count_sweep/rows/5/t_elec_cells` | `8.25` | `8.0` | — |
| `/iris_thickness_zero_count_sweep/rows/6/bw_hz` | `337525422` | `346177876` | — |
| `/iris_thickness_zero_count_sweep/rows/6/f0_hz` | `10959529952` | `10958798136` | — |
| `/iris_thickness_zero_count_sweep/rows/6/t_elec_cells` | `8.3` | `8.1` | — |
| `/iris_thickness_zero_count_sweep/rows/7/bw_hz` | `335790673` | `341547717` | — |
| `/iris_thickness_zero_count_sweep/rows/7/f0_hz` | `10959532008` | `10959246710` | — |
| `/iris_thickness_zero_count_sweep/rows/7/t_elec_cells` | `8.35` | `8.2` | — |
| `/iris_thickness_zero_count_sweep/rows/8/bw_hz` | `333963749` | `337525422` | — |
| `/iris_thickness_zero_count_sweep/rows/8/f0_hz` | `10959578303` | `10959529952` | — |
| `/iris_thickness_zero_count_sweep/rows/8/t_elec_cells` | `8.4` | `8.3` | — |
| `/iris_thickness_zero_count_sweep/rows/9/bw_hz` | `332050107` | `333963749` | — |
| `/iris_thickness_zero_count_sweep/rows/9/f0_hz` | `10959665799` | `10959578303` | — |
| `/iris_thickness_zero_count_sweep/rows/9/t_elec_cells` | `8.45` | `8.4` | — |
| `/ring_down_witness/0/bw` | `339091610.2254982` | `339279358.9056721` | — |
| `/ring_down_witness/0/f0` | `10969497751.822296` | `10969517916.62199` | — |
| `/ring_down_witness/0/hi` | *absent* | `11139157596.074827` | — |
| `/ring_down_witness/0/lo` | *absent* | `10799878237.169155` | — |
| `/ring_down_witness/0/max_colpow` | `1.207` | `1.2071` | — |
| `/ring_down_witness/0/s11` | array[131], SHA256 `4fa6437fb50f` | array[131], SHA256 `7e96c261ee82` | 0–11, 13–33, 35–130 |
| `/ring_down_witness/0/s21` | array[131], SHA256 `c6fbb213fe4b` | array[131], SHA256 `083a4f61cbaf` | 0–130 |
| `/ring_down_witness/0/wall_s` | `586.3` | `427.7` | — |
| `/ring_down_witness/1/bw` | `340442676.4115677` | `340584242.963562` | — |
| `/ring_down_witness/1/f0` | `10970595007.980492` | `10970632439.137453` | — |
| `/ring_down_witness/1/hi` | *absent* | `11140924560.619234` | — |
| `/ring_down_witness/1/lo` | *absent* | `10800340317.655672` | — |
| `/ring_down_witness/1/s11` | array[131], SHA256 `19dd17dae901` | array[131], SHA256 `a8d2053b1cec` | 0–18, 20–26, 28–99, 101–106, 108–113, 115–122, 124–130 |
| `/ring_down_witness/1/s21` | array[131], SHA256 `74e9a501b715` | array[131], SHA256 `b14514c8010c` | 0–130 |
| `/ring_down_witness/1/wall_s` | `1160.3` | `850.0` | — |
| `/ring_down_witness/2/bw` | `340426072.2811279` | `340567212.6795101` | — |
| `/ring_down_witness/2/f0` | `10970574707.84292` | `10970612585.738922` | — |
| `/ring_down_witness/2/hi` | *absent* | `11140896192.078676` | — |
| `/ring_down_witness/2/lo` | *absent* | `10800328979.399166` | — |
| `/ring_down_witness/2/s11` | array[131], SHA256 `45d75d62dde3` | array[131], SHA256 `99756a759dc8` | 0–3, 5, 7–9, 11, 15, 17, 19, 21–97, 99–101, 104, 106–107, 109–115, 118–124, 126–128, 130 |
| `/ring_down_witness/2/s21` | array[131], SHA256 `cac2961e684f` | array[131], SHA256 `fde1f8f94b2b` | 0–130 |
| `/ring_down_witness/2/wall_s` | `1728.3` | `1262.2` | — |
| `/ring_down_witness/3/bw` | `340426165.5448818` | `340567333.74594116` | — |
| `/ring_down_witness/3/f0` | `10970574763.497368` | `10970612623.710629` | — |
| `/ring_down_witness/3/hi` | *absent* | `11140896290.5836` | — |
| `/ring_down_witness/3/lo` | *absent* | `10800328956.837658` | — |
| `/ring_down_witness/3/s11` | array[131], SHA256 `e4c759b2c17e` | array[131], SHA256 `f45ff7f82b0b` | 0–4, 7–10, 15–19, 21–22, 24–101, 103–107, 109–110, 112, 114, 117, 119–122, 124, 126–130 |
| `/ring_down_witness/3/s21` | array[131], SHA256 `673831a61700` | array[131], SHA256 `5e8be8492798` | 0–130 |
| `/ring_down_witness/3/wall_s` | `2302.6` | `1698.9` | — |
| `/schema_version` | `1` | `2` | — |

## validation/crossval/_22_dispersive_results/rfx.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/arms/debye/meep/gates/precheck_passed` | *absent* | `true` | — |
| `/arms/debye/run/elapsed_s` | `0.840024471282959` | `0.4232821464538574` | — |
| `/arms/debye/tail/fit_note` | `"fitted_rate_* recomputed from the stored envelope with the fit starting at n_pulse_end (review finding 1); no FDTD rerun"` | *absent* | — |
| `/arms/debye/tail/fit_reliable` | *absent* | `true` | — |
| `/arms/debye/tail/fitted_rate_scat_refl_1_s` | `8821615702.722586` | `8821615702.72259` | — |
| `/arms/debye/tail/fitted_rate_total_trans_1_s` | `8051643183.107595` | `8051643183.107597` | — |
| `/arms/drude/meep/gates/precheck_passed` | *absent* | `true` | — |
| `/arms/drude/run/elapsed_s` | `0.7948386669158936` | `0.4261960983276367` | — |
| `/arms/drude/tail/fit_note` | `"fitted_rate_* recomputed from the stored envelope with the fit starting at n_pulse_end (review finding 1); no FDTD rerun"` | *absent* | — |
| `/arms/drude/tail/fit_reliable` | *absent* | `true` | — |
| `/arms/drude/tail/fitted_rate_scat_refl_1_s` | `13065106004.435501` | `13065106004.435495` | — |
| `/arms/drude/tail/fitted_rate_total_trans_1_s` | `16042555653.222027` | `16042555653.222025` | — |
| `/arms/lorentz/meep/gates/precheck_passed` | *absent* | `true` | — |
| `/arms/lorentz/run/elapsed_s` | `0.838709831237793` | `0.4339315891265869` | — |
| `/arms/lorentz/tail/fit_note` | `"fitted_rate_* recomputed from the stored envelope with the fit starting at n_pulse_end (review finding 1); no FDTD rerun"` | *absent* | — |
| `/arms/lorentz/tail/fit_reliable` | *absent* | `true` | — |
| `/arms/lorentz/tail/fitted_rate_scat_refl_1_s` | `7073777515.481599` | `7073777515.481603` | — |
| `/arms/lorentz/tail/fitted_rate_total_trans_1_s` | `8970592127.455292` | `8970592127.4553` | — |
| `/date_utc` | `"2026-09-02T10:57:37+00:00"` | `"2026-09-07T12:50:44+00:00"` | — |

## validation/crossval/_24_nu_cavity_results/rfx.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/arms/multi_band/cost/wall_energy_s` | `5.661684036254883` | `3.0806050300598145` | — |
| `/arms/multi_band/cost/wall_harminv_s` | `61.59971880912781` | `128.35460257530212` | — |
| `/arms/multi_band/cost/wall_run_s` | `5.970173358917236` | `3.985015392303467` | — |
| `/arms/multi_band/measured/energy/metric_swap_in_scope` | *absent* | `false` | — |
| `/arms/multi_band/measured/per_mode/TE011/amp_max` | `96748.36509617846` | `96748.36509657347` | — |
| `/arms/multi_band/measured/per_mode/TE011/error_max` | `2.7640778554882672e-11` | `2.7639668331858047e-11` | — |
| `/arms/multi_band/measured/per_mode/TE011/f_hz` | `6244040436.688446` | `6244040436.6884365` | — |
| `/arms/multi_band/measured/per_mode/TE101/amp_max` | `129435.23758901736` | `129435.23758769906` | — |
| `/arms/multi_band/measured/per_mode/TE101/error_max` | `3.5188585378875814e-10` | `3.5188851832401724e-10` | — |
| `/arms/multi_band/measured/per_mode/TE101/f_hz` | `4798343770.952076` | `4798343770.952088` | — |
| `/arms/multi_band/measured/per_mode/TE102/amp_max` | `22090.642500422822` | `22090.642500507853` | — |
| `/arms/multi_band/measured/per_mode/TE102/error_max` | `2.7316957584844204e-09` | `2.7316948703060007e-09` | — |
| `/arms/multi_band/measured/per_mode/TE102/f_hz` | `8070689031.169416` | `8070689031.16942` | — |
| `/arms/multi_band/measured/per_mode/TE201/amp_max` | `35366.147246192035` | `35366.147246051296` | — |
| `/arms/multi_band/measured/per_mode/TE201/error_max` | `2.2608243055444177e-09` | `2.2608233063436955e-09` | — |
| `/arms/multi_band/measured/per_mode/TE201/f_hz` | `7067827217.133965` | `7067827217.133914` | — |
| `/arms/multi_band/measured/per_mode/TM110/amp_max` | `120103.47855217254` | `120103.47855145385` | — |
| `/arms/multi_band/measured/per_mode/TM110/error_max` | `7.341871555155421e-10` | `7.341857122256101e-10` | — |
| `/arms/multi_band/measured/per_mode/TM110/f_hz` | `5825297801.628358` | `5825297801.628375` | — |
| `/arms/multi_band/measured/per_mode/TM111/amp_max` | `70972.96764306964` | `70972.96764307808` | — |
| `/arms/multi_band/measured/per_mode/TM111/error_max` | `1.5070054204535666e-08` | `1.5070054537602573e-08` | — |
| `/arms/multi_band/measured/per_mode/TM210/amp_max` | `29777.227911303085` | `29777.227911206923` | — |
| `/arms/multi_band/measured/per_mode/TM210/error_max` | `7.853507622002098e-09` | `7.853506844845981e-09` | — |
| `/arms/multi_band/measured/per_mode/TM210/f_hz` | `7801774909.3333435` | `7801774909.33337` | — |
| `/arms/multi_band/measured/stationarity/TE011` | `1.5982075022395575e-08` | `1.5982086477410098e-08` | — |
| `/arms/multi_band/measured/stationarity/TE101` | `6.1340067424938176e-09` | `6.134040331366789e-09` | — |
| `/arms/multi_band/measured/stationarity/TE102` | `5.438967121913395e-08` | `5.438967062830809e-08` | — |
| `/arms/multi_band/measured/stationarity/TE201` | `3.234761904255376e-08` | `3.23476083829455e-08` | — |
| `/arms/multi_band/measured/stationarity/TM110` | `5.976126028106317e-08` | `5.97612691215403e-08` | — |
| `/arms/multi_band/measured/stationarity/TM111` | `2.680067219331596e-08` | `2.680068389642e-08` | — |
| `/arms/multi_band/measured/stationarity/TM210` | `1.65301691630456e-07` | `1.6530167720635697e-07` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE011/amp_max` | `96587.32428688025` | `96587.32428642485` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE011/error_max` | `5.69539304606792e-10` | `5.695404148298167e-10` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE011/f_hz` | `6244040507.289372` | `6244040507.289479` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE101/amp_max` | `129397.66066688432` | `129397.66066781367` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE101/error_max` | `7.470164486989006e-10` | `7.470144502974563e-10` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE101/f_hz` | `4798343816.535418` | `4798343816.535296` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE102/amp_max` | `22364.792681186962` | `22364.792681146882` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE102/error_max` | `3.0344478041399725e-09` | `3.0344480261845774e-09` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE102/f_hz` | `8070689648.099824` | `8070689648.099788` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE201/amp_max` | `35421.45796755824` | `35421.45796745361` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE201/error_max` | `6.649131245595186e-10` | `6.649123474034013e-10` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TE201/f_hz` | `7067827391.938916` | `7067827391.938855` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM110/amp_max` | `120370.04078013355` | `120370.04077958128` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM110/error_max` | `7.198920348727711e-10` | `7.198909246497465e-10` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM110/f_hz` | `5825297615.509593` | `5825297615.509565` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM111/amp_max` | `70830.30539928336` | `70830.30539983942` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM111/error_max` | `8.495005943132128e-08` | `8.495005698883062e-08` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM111/f_hz` | `6926565472.466625` | `6926565472.466591` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM210/amp_max` | `30223.15613966631` | `30223.15613930209` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM210/error_max` | `1.4168444195661323e-10` | `1.4168133333214428e-10` | — |
| `/arms/multi_band/measured/windows/A/per_mode/TM210/f_hz` | `7801774544.977457` | `7801774544.977551` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE011/amp_max` | `97327.0024809329` | `97327.00247921226` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE011/error_max` | `5.24290055814447e-10` | `5.24285503900046e-10` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE011/f_hz` | `6244040407.49665` | `6244040407.496685` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE101/amp_max` | `129641.38080470842` | `129641.38080541353` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE101/error_max` | `7.064133722423094e-10` | `7.064118179300749e-10` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE101/f_hz` | `4798343845.968491` | `4798343845.968531` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE102/amp_max` | `22875.301829383374` | `22875.301829451808` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE102/error_max` | `4.861654301357987e-09` | `4.861654967491802e-09` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE102/f_hz` | `8070689209.137701` | `8070689209.13767` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE201/amp_max` | `35685.223261445906` | `35685.22326144236` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TE201/f_hz` | `7067827163.311534` | `7067827163.311548` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM110/amp_max` | `120090.79440247714` | `120090.7944024593` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM110/error_max` | `2.8357027836989346e-10` | `2.8357005632528853e-10` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM110/f_hz` | `5825297963.636731` | `5825297963.636755` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM111/amp_max` | `71592.2961131397` | `71592.29611299008` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM111/error_max` | `4.0095536224526995e-08` | `4.009553866701765e-08` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM111/f_hz` | `6926565658.103237` | `6926565658.103284` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM210/amp_max` | `29153.739334204973` | `29153.739334019287` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM210/error_max` | `2.287104727827227e-09` | `2.2871030624926902e-09` | — |
| `/arms/multi_band/measured/windows/B/per_mode/TM210/f_hz` | `7801775834.624047` | `7801775834.624029` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE011/amp_max` | `96748.36509617846` | `96748.36509657347` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE011/error_max` | `2.7640778554882672e-11` | `2.7639668331858047e-11` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE011/f_hz` | `6244040436.688446` | `6244040436.6884365` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE101/amp_max` | `129435.23758901736` | `129435.23758769906` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE101/error_max` | `3.5188585378875814e-10` | `3.5188851832401724e-10` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE101/f_hz` | `4798343770.952076` | `4798343770.952088` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE102/amp_max` | `22090.642500422822` | `22090.642500507853` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE102/error_max` | `2.7316957584844204e-09` | `2.7316948703060007e-09` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE102/f_hz` | `8070689031.169416` | `8070689031.16942` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE201/amp_max` | `35366.147246192035` | `35366.147246051296` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE201/error_max` | `2.2608243055444177e-09` | `2.2608233063436955e-09` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TE201/f_hz` | `7067827217.133965` | `7067827217.133914` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TM110/amp_max` | `120103.47855217254` | `120103.47855145385` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TM110/error_max` | `7.341871555155421e-10` | `7.341857122256101e-10` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TM110/f_hz` | `5825297801.628358` | `5825297801.628375` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TM111/amp_max` | `70972.96764306964` | `70972.96764307808` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TM111/error_max` | `1.5070054204535666e-08` | `1.5070054537602573e-08` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TM210/amp_max` | `29777.227911303085` | `29777.227911206923` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TM210/error_max` | `7.853507622002098e-09` | `7.853506844845981e-09` | — |
| `/arms/multi_band/measured/windows/full/per_mode/TM210/f_hz` | `7801774909.3333435` | `7801774909.33337` | — |
| `/arms/single_band/cost/wall_energy_s` | `5.083554744720459` | `3.0125932693481445` | — |
| `/arms/single_band/cost/wall_harminv_s` | `64.04675269126892` | `128.30529594421387` | — |
| `/arms/single_band/cost/wall_run_s` | `5.501209259033203` | `3.9258182048797607` | — |
| `/arms/single_band/measured/energy/metric_swap_in_scope` | *absent* | `false` | — |
| `/arms/single_band/measured/per_mode/TE011/amp_max` | `96739.94002324167` | `96739.94002313593` | — |
| `/arms/single_band/measured/per_mode/TE011/error_max` | `6.434778265784757e-10` | `6.434774935115684e-10` | — |
| `/arms/single_band/measured/per_mode/TE011/f_hz` | `6244008632.387905` | `6244008632.387892` | — |
| `/arms/single_band/measured/per_mode/TE101/amp_max` | `132757.00762994145` | `132757.00762820817` | — |
| `/arms/single_band/measured/per_mode/TE101/error_max` | `4.4339132365678324e-10` | `4.433946543258571e-10` | — |
| `/arms/single_band/measured/per_mode/TE101/f_hz` | `4798302388.165375` | `4798302388.1653` | — |
| `/arms/single_band/measured/per_mode/TE102/amp_max` | `21412.93977911847` | `21412.939779247314` | — |
| `/arms/single_band/measured/per_mode/TE102/error_max` | `1.148810502282771e-08` | `1.1488103468515476e-08` | — |
| `/arms/single_band/measured/per_mode/TE102/f_hz` | `8065528884.140119` | `8065528884.140081` | — |
| `/arms/single_band/measured/per_mode/TE201/amp_max` | `36256.830125604676` | `36256.830125457695` | — |
| `/arms/single_band/measured/per_mode/TE201/error_max` | `2.6698172561623323e-09` | `2.6698161459393077e-09` | — |
| `/arms/single_band/measured/per_mode/TE201/f_hz` | `7067799115.336304` | `7067799115.336333` | — |
| `/arms/single_band/measured/per_mode/TM110/amp_max` | `120105.7824701793` | `120105.78247180484` | — |
| `/arms/single_band/measured/per_mode/TM110/error_max` | `9.569167680467672e-11` | `9.568790204639299e-11` | — |
| `/arms/single_band/measured/per_mode/TM110/f_hz` | `5825297849.221689` | `5825297849.22169` | — |
| `/arms/single_band/measured/per_mode/TM111/amp_max` | `71404.32271883919` | `71404.32271902883` | — |
| `/arms/single_band/measured/per_mode/TM111/error_max` | `3.104127921371713e-08` | `3.104127987985095e-08` | — |
| `/arms/single_band/measured/per_mode/TM111/f_hz` | `6926536873.845411` | `6926536873.845436` | — |
| `/arms/single_band/measured/per_mode/TM210/amp_max` | `29782.16759171055` | `29782.167591915004` | — |
| `/arms/single_band/measured/per_mode/TM210/error_max` | `1.3273335763841487e-09` | `1.3273353527409881e-09` | — |
| `/arms/single_band/measured/per_mode/TM210/f_hz` | `7801775308.145749` | `7801775308.14574` | — |
| `/arms/single_band/measured/stationarity/TE011` | `2.372796656476227e-08` | `2.3727961982733184e-08` | — |
| `/arms/single_band/measured/stationarity/TE101` | `3.1440867404718724e-10` | `3.144108603240525e-10` | — |
| `/arms/single_band/measured/stationarity/TE102` | `5.982835225457299e-08` | `5.982834823438721e-08` | — |
| `/arms/single_band/measured/stationarity/TE201` | `1.0980398011624492e-08` | `1.0980390860213102e-08` | — |
| `/arms/single_band/measured/stationarity/TM110` | `6.185421740530157e-08` | `6.185422690062898e-08` | — |
| `/arms/single_band/measured/stationarity/TM111` | `3.115266360317501e-08` | `3.1152653139179775e-08` | — |
| `/arms/single_band/measured/stationarity/TM210` | `1.0799974692852881e-07` | `1.0799975047343432e-07` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE011/amp_max` | `96565.83333098027` | `96565.83333130399` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE011/error_max` | `7.508316191007225e-10` | `7.508307309223028e-10` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE011/f_hz` | `6244008694.470599` | `6244008694.470619` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE101/amp_max` | `132665.37372098488` | `132665.37372063185` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE101/error_max` | `4.6514680995812796e-10` | `4.6514747609194274e-10` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE101/f_hz` | `4798302443.753766` | `4798302443.753856` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE102/amp_max` | `21457.138232632024` | `21457.13823281402` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE102/error_max` | `4.4353702932653505e-09` | `4.435368072819301e-09` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE102/f_hz` | `8065529644.6684065` | `8065529644.668402` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE201/amp_max` | `36345.082938519525` | `36345.08293871587` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE201/error_max` | `3.582688923309263e-09` | `3.5826902555768925e-09` | — |
| `/arms/single_band/measured/windows/A/per_mode/TE201/f_hz` | `7067799267.566257` | `7067799267.56624` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM110/amp_max` | `120373.26637215777` | `120373.26637132413` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM110/error_max` | `2.3631510082111618e-09` | `2.3631492318543224e-09` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM110/f_hz` | `5825297630.492821` | `5825297630.492884` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM111/amp_max` | `71265.82585283596` | `71265.82585286179` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM111/error_max` | `8.777990023745019e-08` | `8.777990090358401e-08` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM111/f_hz` | `6926536833.375786` | `6926536833.375835` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM210/amp_max` | `30230.661526453525` | `30230.66152602714` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM210/error_max` | `3.255807179414205e-09` | `3.255803515678224e-09` | — |
| `/arms/single_band/measured/windows/A/per_mode/TM210/f_hz` | `7801775145.446669` | `7801775145.446605` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE011/amp_max` | `97326.03191041594` | `97326.03191033896` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE011/error_max` | `9.432281622423488e-10` | `9.432280512200464e-10` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE011/f_hz` | `6244008546.312971` | `6244008546.31302` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE101/amp_max` | `132960.34927794483` | `132960.34927782975` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE101/error_max` | `3.2747595746229763e-09` | `3.2747597966675812e-09` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE101/f_hz` | `4798302442.245138` | `4798302442.245217` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE102/amp_max` | `22210.341367416815` | `22210.341367320707` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE102/error_max` | `5.052340767974783e-10` | `5.052330775967562e-10` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE102/f_hz` | `8065529162.121103` | `8065529162.121131` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE201/amp_max` | `36559.3678097172` | `36559.36780941609` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE201/error_max` | `3.4844007679168953e-09` | `3.484398769515451e-09` | — |
| `/arms/single_band/measured/windows/B/per_mode/TE201/f_hz` | `7067799189.95901` | `7067799189.9590435` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM110/amp_max` | `120086.22723777377` | `120086.22723851734` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM110/error_max` | `3.214513988325507e-09` | `3.2145124340132725e-09` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM110/f_hz` | `5825297990.81206` | `5825297990.812179` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM111/amp_max` | `72012.77714645029` | `72012.77714631337` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM111/error_max` | `6.809771568860867e-08` | `6.809771546656407e-08` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM111/f_hz` | `6926537049.155859` | `6926537049.155836` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM210/amp_max` | `29158.737065876063` | `29158.737065783367` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM210/error_max` | `1.0979607334427044e-08` | `1.0979608222605464e-08` | — |
| `/arms/single_band/measured/windows/B/per_mode/TM210/f_hz` | `7801775988.0364275` | `7801775988.036391` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE011/amp_max` | `96739.94002324167` | `96739.94002313593` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE011/error_max` | `6.434778265784757e-10` | `6.434774935115684e-10` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE011/f_hz` | `6244008632.387905` | `6244008632.387892` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE101/amp_max` | `132757.00762994145` | `132757.00762820817` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE101/error_max` | `4.4339132365678324e-10` | `4.433946543258571e-10` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE101/f_hz` | `4798302388.165375` | `4798302388.1653` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE102/amp_max` | `21412.93977911847` | `21412.939779247314` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE102/error_max` | `1.148810502282771e-08` | `1.1488103468515476e-08` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE102/f_hz` | `8065528884.140119` | `8065528884.140081` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE201/amp_max` | `36256.830125604676` | `36256.830125457695` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE201/error_max` | `2.6698172561623323e-09` | `2.6698161459393077e-09` | — |
| `/arms/single_band/measured/windows/full/per_mode/TE201/f_hz` | `7067799115.336304` | `7067799115.336333` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM110/amp_max` | `120105.7824701793` | `120105.78247180484` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM110/error_max` | `9.569167680467672e-11` | `9.568790204639299e-11` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM110/f_hz` | `5825297849.221689` | `5825297849.22169` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM111/amp_max` | `71404.32271883919` | `71404.32271902883` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM111/error_max` | `3.104127921371713e-08` | `3.104127987985095e-08` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM111/f_hz` | `6926536873.845411` | `6926536873.845436` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM210/amp_max` | `29782.16759171055` | `29782.167591915004` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM210/error_max` | `1.3273335763841487e-09` | `1.3273353527409881e-09` | — |
| `/arms/single_band/measured/windows/full/per_mode/TM210/f_hz` | `7801775308.145749` | `7801775308.14574` | — |
| `/arms/uniform/cost/wall_energy_s` | `4.130686283111572` | `2.321579694747925` | — |
| `/arms/uniform/cost/wall_harminv_s` | `20.245747566223145` | `43.76630115509033` | — |
| `/arms/uniform/cost/wall_run_s` | `3.6896908283233643` | `2.043273687362671` | — |
| `/arms/uniform/measured/energy/metric_swap_in_scope` | *absent* | `false` | — |
| `/arms/uniform/measured/per_mode/TE011/amp_max` | `0.0004492369243079374` | `0.00044923692430700663` | — |
| `/arms/uniform/measured/per_mode/TE011/error_max` | `4.1370011860664135e-09` | `4.137000297887994e-09` | — |
| `/arms/uniform/measured/per_mode/TE011/f_hz` | `6244727276.861334` | `6244727276.861398` | — |
| `/arms/uniform/measured/per_mode/TE101/amp_max` | `0.0006165167935786657` | `0.0006165167935781801` | — |
| `/arms/uniform/measured/per_mode/TE101/error_max` | `1.3069961779521577e-08` | `1.3069962001566182e-08` | — |
| `/arms/uniform/measured/per_mode/TE101/f_hz` | `4798620984.719757` | `4798620984.719786` | — |
| `/arms/uniform/measured/per_mode/TE102/amp_max` | `9.911622294843942e-05` | `9.911622294841254e-05` | — |
| `/arms/uniform/measured/per_mode/TE102/error_max` | `3.09336238846214e-08` | `3.0933623773599095e-08` | — |
| `/arms/uniform/measured/per_mode/TE102/f_hz` | `8067960136.21369` | `8067960136.213679` | — |
| `/arms/uniform/measured/per_mode/TE201/amp_max` | `0.00016824910248643696` | `0.00016824910248692637` | — |
| `/arms/uniform/measured/per_mode/TE201/error_max` | `1.9848086507678886e-08` | `1.9848087506879608e-08` | — |
| `/arms/uniform/measured/per_mode/TE201/f_hz` | `7068847205.616574` | `7068847205.616583` | — |
| `/arms/uniform/measured/per_mode/TM110/amp_max` | `0.0005578096095607718` | `0.0005578096095598407` | — |
| `/arms/uniform/measured/per_mode/TM110/error_max` | `8.888955682628819e-09` | `8.888956348762633e-09` | — |
| `/arms/uniform/measured/per_mode/TM110/f_hz` | `5825890095.150942` | `5825890095.150917` | — |
| `/arms/uniform/measured/per_mode/TM111/amp_max` | `0.0003315510680286656` | `0.0003315510680298078` | — |
| `/arms/uniform/measured/per_mode/TM111/error_max` | `1.667236360081148e-07` | `1.6672363845060545e-07` | — |
| `/arms/uniform/measured/per_mode/TM111/f_hz` | `6927520306.002395` | `6927520306.002429` | — |
| `/arms/uniform/measured/per_mode/TM210/amp_max` | `0.00013820892387804244` | `0.0001382089238782515` | — |
| `/arms/uniform/measured/per_mode/TM210/error_max` | `1.194777726754026e-07` | `1.194777722313134e-07` | — |
| `/arms/uniform/measured/per_mode/TM210/f_hz` | `7803195765.821351` | `7803195765.821363` | — |
| `/arms/uniform/measured/stationarity/TE011` | `2.237091880423147e-08` | `2.237091193197848e-08` | — |
| `/arms/uniform/measured/stationarity/TE101` | `8.07638972567181e-09` | `8.076403836158229e-09` | — |
| `/arms/uniform/measured/stationarity/TE102` | `2.8838675799047446e-08` | `2.883867839956045e-08` | — |
| `/arms/uniform/measured/stationarity/TE201` | `3.0506791809741964e-08` | `3.050679680149641e-08` | — |
| `/arms/uniform/measured/stationarity/TM110` | `5.2704767044230826e-08` | `5.270478194055786e-08` | — |
| `/arms/uniform/measured/stationarity/TM111` | `6.26575041656587e-08` | `6.26575074696088e-08` | — |
| `/arms/uniform/measured/stationarity/TM210` | `1.1543419698814544e-07` | `1.1543421006524276e-07` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE011/amp_max` | `0.00044870203251969336` | `0.00044870203251967775` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE011/f_hz` | `6244728031.740266` | `6244728031.74021` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE101/amp_max` | `0.0006162380326577173` | `0.0006162380326578643` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE101/f_hz` | `4798621293.950386` | `4798621293.950315` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE102/amp_max` | `9.984292226383969e-05` | `9.984292226383537e-05` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE102/error_max` | `7.312240812851201e-09` | `7.3122410348958056e-09` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE102/f_hz` | `8067963632.15795` | `8067963632.157979` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE201/amp_max` | `0.00016874263296800014` | `0.00016874263296828363` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE201/error_max` | `3.428044514919293e-10` | `3.4280400740271944e-10` | — |
| `/arms/uniform/measured/windows/A/per_mode/TE201/f_hz` | `7068847633.627587` | `7068847633.627603` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM110/amp_max` | `0.000559034681003017` | `0.0005590346809973617` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM110/error_max` | `1.8006633961675789e-09` | `1.8006598434539e-09` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM110/f_hz` | `5825888754.843991` | `5825888754.8439865` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM111/amp_max` | `0.0003311982971084991` | `0.0003311982971075458` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM111/error_max` | `1.1568753288671019e-07` | `1.1568753299773249e-07` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM111/f_hz` | `6927522950.77296` | `6927522950.772945` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM210/amp_max` | `0.00014047090414645542` | `0.00014047090414568924` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM210/error_max` | `2.1769374081159754e-09` | `2.1769352986922286e-09` | — |
| `/arms/uniform/measured/windows/A/per_mode/TM210/f_hz` | `7803195716.700983` | `7803195716.700927` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE011/amp_max` | `0.00045192016049802597` | `0.00045192016049599336` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE011/error_max` | `1.2481935485197937e-09` | `1.2481921052298617e-09` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE011/f_hz` | `6244727892.039979` | `6244727892.039966` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE101/amp_max` | `0.0006174176800027186` | `0.00061741768000273` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE101/f_hz` | `4798621332.705919` | `4798621332.705916` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE102/amp_max` | `0.00010279964447995751` | `0.00010279964448021013` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE102/error_max` | `2.1890635970578387e-09` | `2.189062708879419e-09` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE102/f_hz` | `8067963399.488664` | `8067963399.488671` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE201/amp_max` | `0.00016953118753532408` | `0.0001695311875357321` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE201/error_max` | `5.123693469499813e-09` | `5.12369424665593e-09` | — |
| `/arms/uniform/measured/windows/B/per_mode/TE201/f_hz` | `7068847417.979737` | `7068847417.979717` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM110/amp_max` | `0.0005577667866110828` | `0.00055776678661584` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM110/error_max` | `4.741489423309986e-10` | `4.741520509554675e-10` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM110/f_hz` | `5825889061.896172` | `5825889061.896254` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM111/amp_max` | `0.0003344668374561368` | `0.0003344668374564238` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM111/error_max` | `7.794700473962024e-08` | `7.79470027412188e-08` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM111/f_hz` | `6927523384.834092` | `6927523384.834101` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM210/amp_max` | `0.00013534787480846814` | `0.00013534787480864424` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM210/error_max` | `3.725397768050698e-11` | `3.725353359129713e-11` | — |
| `/arms/uniform/measured/windows/B/per_mode/TM210/f_hz` | `7803196617.45662` | `7803196617.456666` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE011/amp_max` | `0.0004492369243079374` | `0.00044923692430700663` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE011/error_max` | `4.1370011860664135e-09` | `4.137000297887994e-09` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE011/f_hz` | `6244727276.861334` | `6244727276.861398` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE101/amp_max` | `0.0006165167935786657` | `0.0006165167935781801` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE101/error_max` | `1.3069961779521577e-08` | `1.3069962001566182e-08` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE101/f_hz` | `4798620984.719757` | `4798620984.719786` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE102/amp_max` | `9.911622294843942e-05` | `9.911622294841254e-05` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE102/error_max` | `3.09336238846214e-08` | `3.0933623773599095e-08` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE102/f_hz` | `8067960136.21369` | `8067960136.213679` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE201/amp_max` | `0.00016824910248643696` | `0.00016824910248692637` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE201/error_max` | `1.9848086507678886e-08` | `1.9848087506879608e-08` | — |
| `/arms/uniform/measured/windows/full/per_mode/TE201/f_hz` | `7068847205.616574` | `7068847205.616583` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM110/amp_max` | `0.0005578096095607718` | `0.0005578096095598407` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM110/error_max` | `8.888955682628819e-09` | `8.888956348762633e-09` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM110/f_hz` | `5825890095.150942` | `5825890095.150917` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM111/amp_max` | `0.0003315510680286656` | `0.0003315510680298078` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM111/error_max` | `1.667236360081148e-07` | `1.6672363845060545e-07` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM111/f_hz` | `6927520306.002395` | `6927520306.002429` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM210/amp_max` | `0.00013820892387804244` | `0.0001382089238782515` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM210/error_max` | `1.194777726754026e-07` | `1.194777722313134e-07` | — |
| `/arms/uniform/measured/windows/full/per_mode/TM210/f_hz` | `7803195765.821351` | `7803195765.821363` | — |
| `/arms/uniform_fine/cost/wall_energy_s` | `34.12033152580261` | `21.215805530548096` | — |
| `/arms/uniform_fine/cost/wall_harminv_s` | `45.222601890563965` | `97.87563300132751` | — |
| `/arms/uniform_fine/cost/wall_run_s` | `34.8805365562439` | `22.03135848045349` | — |
| `/arms/uniform_fine/measured/energy/metric_swap_in_scope` | *absent* | `false` | — |
| `/arms/uniform_fine/measured/per_mode/TE011/amp_max` | `0.00011211971474787309` | `0.00011211971474870486` | — |
| `/arms/uniform_fine/measured/per_mode/TE011/error_max` | `4.8814261632834643e-08` | `4.8814259412388594e-08` | — |
| `/arms/uniform_fine/measured/per_mode/TE011/f_hz` | `6245438451.9756` | `6245438451.975603` | — |
| `/arms/uniform_fine/measured/per_mode/TE101/amp_max` | `0.00015406939537789143` | `0.0001540693953775816` | — |
| `/arms/uniform_fine/measured/per_mode/TE101/error_max` | `4.860663416206279e-08` | `4.860663360695128e-08` | — |
| `/arms/uniform_fine/measured/per_mode/TE101/f_hz` | `4798922541.201998` | `4798922541.2019825` | — |
| `/arms/uniform_fine/measured/per_mode/TE102/amp_max` | `2.4607150852739873e-05` | `2.4607150852646293e-05` | — |
| `/arms/uniform_fine/measured/per_mode/TE102/error_max` | `3.759662303703948e-08` | `3.759662181579415e-08` | — |
| `/arms/uniform_fine/measured/per_mode/TE102/f_hz` | `8071114161.793128` | `8071114161.793144` | — |
| `/arms/uniform_fine/measured/per_mode/TE201/amp_max` | `4.208184297487528e-05` | `4.208184297458329e-05` | — |
| `/arms/uniform_fine/measured/per_mode/TE201/error_max` | `5.595446783690505e-08` | `5.5954465727481306e-08` | — |
| `/arms/uniform_fine/measured/per_mode/TE201/f_hz` | `7070158709.593542` | `7070158709.593548` | — |
| `/arms/uniform_fine/measured/per_mode/TM110/amp_max` | `0.00013930843992203924` | `0.00013930843992196676` | — |
| `/arms/uniform_fine/measured/per_mode/TM110/error_max` | `5.45027357690131e-08` | `5.4502735991057705e-08` | — |
| `/arms/uniform_fine/measured/per_mode/TM110/f_hz` | `5826660375.596754` | `5826660375.596762` | — |
| `/arms/uniform_fine/measured/per_mode/TM111/amp_max` | `8.154779294501709e-05` | `8.154779294484468e-05` | — |
| `/arms/uniform_fine/measured/per_mode/TM111/error_max` | `4.946553500140283e-07` | `4.946553485707383e-07` | — |
| `/arms/uniform_fine/measured/per_mode/TM111/f_hz` | `6927818052.895847` | `6927818052.895831` | — |
| `/arms/uniform_fine/measured/per_mode/TM210/amp_max` | `3.45022247457572e-05` | `3.450222474559585e-05` | — |
| `/arms/uniform_fine/measured/per_mode/TM210/error_max` | `7.207896490779575e-08` | `7.207896646210799e-08` | — |
| `/arms/uniform_fine/measured/per_mode/TM210/f_hz` | `7804432307.653516` | `7804432307.653458` | — |
| `/arms/uniform_fine/measured/stationarity/TE011` | `3.24790101349856e-07` | `3.2479010104445714e-07` | — |
| `/arms/uniform_fine/measured/stationarity/TE101` | `4.091375638188538e-07` | `4.0913757594118734e-07` | — |
| `/arms/uniform_fine/measured/stationarity/TE102` | `1.0257506194066907e-06` | `1.025750579468966e-06` | — |
| `/arms/uniform_fine/measured/stationarity/TE201` | `8.485406307259218e-08` | `8.485406401680292e-08` | — |
| `/arms/uniform_fine/measured/stationarity/TM110` | `3.041852535767511e-07` | `3.0418525292205366e-07` | — |
| `/arms/uniform_fine/measured/stationarity/TM111` | `3.389808804817593e-07` | `3.3898088309727507e-07` | — |
| `/arms/uniform_fine/measured/stationarity/TM210` | `9.334670445728337e-07` | `9.334670366300681e-07` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE011/amp_max` | `0.00011201685210967646` | `0.00011201685210980594` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE011/error_max` | `2.0321059612804504e-08` | `2.032105983484911e-08` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE011/f_hz` | `6245437904.808926` | `6245437904.808884` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE101/amp_max` | `0.0001540048227470442` | `0.00015400482274783347` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE101/error_max` | `2.7811877356676007e-08` | `2.7811876024408377e-08` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE101/f_hz` | `4798920539.95315` | `4798920539.9531145` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE102/amp_max` | `2.4985485470535488e-05` | `2.4985485470371394e-05` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE102/error_max` | `1.8374917631902576e-07` | `1.8374917454266892e-07` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE102/f_hz` | `8071107830.384832` | `8071107830.384814` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE201/amp_max` | `4.203438909407722e-05` | `4.2034389093796304e-05` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE201/error_max` | `7.267233903540671e-08` | `7.267234081176355e-08` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TE201/f_hz` | `7070156237.904547` | `7070156237.904599` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM110/amp_max` | `0.00013962527334665116` | `0.00013962527334788094` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM110/error_max` | `2.727553516379544e-09` | `2.7275510738888897e-09` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM110/f_hz` | `5826659599.235274` | `5826659599.235245` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM111/amp_max` | `8.141719899905505e-05` | `8.141719899902407e-05` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM111/error_max` | `3.713552212403215e-07` | `3.713552210182769e-07` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM111/f_hz` | `6927819882.420602` | `6927819882.420551` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM210/amp_max` | `3.505511832373266e-05` | `3.5055118323561814e-05` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM210/error_max` | `6.435825061767986e-08` | `6.435825194994749e-08` | — |
| `/arms/uniform_fine/measured/windows/A/per_mode/TM210/f_hz` | `7804428754.713709` | `7804428754.713838` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE011/amp_max` | `0.00011281849191349558` | `0.00011281849191321207` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE011/error_max` | `1.674384075833757e-09` | `1.6743834096999421e-09` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE011/f_hz` | `6245439933.265513` | `6245439933.26547` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE101/amp_max` | `0.0001542605426708385` | `0.00015426054266980542` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE101/error_max` | `1.1814169531554342e-08` | `1.1814171307911181e-08` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE101/f_hz` | `4798922503.372627` | `4798922503.37265` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE102/amp_max` | `2.5519779164856012e-05` | `2.551977916509205e-05` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE102/error_max` | `1.808296545835475e-08` | `1.80829632379087e-08` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE102/f_hz` | `8071116109.335183` | `8071116109.334843` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE201/amp_max` | `4.236536673942597e-05` | `4.2365366739248716e-05` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE201/error_max` | `6.59895915688935e-09` | `6.5989580466663256e-09` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TE201/f_hz` | `7070156837.83624` | `7070156837.836299` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM110/amp_max` | `0.0001393095174681351` | `0.00013930951746713607` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM110/error_max` | `6.3273226658111525e-09` | `6.327324664212597e-09` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM110/f_hz` | `5826661371.619438` | `5826661371.619405` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM111/amp_max` | `8.229461171672637e-05` | `8.229461171617786e-05` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM111/error_max` | `6.514694310677527e-07` | `6.514694321779757e-07` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM111/f_hz` | `6927817534.022738` | `6927817534.02267` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM210/amp_max` | `3.37995850331423e-05` | `3.3799585033219575e-05` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM210/error_max` | `1.2188286135916826e-07` | `1.2188286180325747e-07` | — |
| `/arms/uniform_fine/measured/windows/B/per_mode/TM210/f_hz` | `7804436039.89407` | `7804436039.894136` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE011/amp_max` | `0.00011211971474787309` | `0.00011211971474870486` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE011/error_max` | `4.8814261632834643e-08` | `4.8814259412388594e-08` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE011/f_hz` | `6245438451.9756` | `6245438451.975603` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE101/amp_max` | `0.00015406939537789143` | `0.0001540693953775816` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE101/error_max` | `4.860663416206279e-08` | `4.860663360695128e-08` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE101/f_hz` | `4798922541.201998` | `4798922541.2019825` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE102/amp_max` | `2.4607150852739873e-05` | `2.4607150852646293e-05` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE102/error_max` | `3.759662303703948e-08` | `3.759662181579415e-08` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE102/f_hz` | `8071114161.793128` | `8071114161.793144` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE201/amp_max` | `4.208184297487528e-05` | `4.208184297458329e-05` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE201/error_max` | `5.595446783690505e-08` | `5.5954465727481306e-08` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TE201/f_hz` | `7070158709.593542` | `7070158709.593548` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM110/amp_max` | `0.00013930843992203924` | `0.00013930843992196676` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM110/error_max` | `5.45027357690131e-08` | `5.4502735991057705e-08` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM110/f_hz` | `5826660375.596754` | `5826660375.596762` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM111/amp_max` | `8.154779294501709e-05` | `8.154779294484468e-05` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM111/error_max` | `4.946553500140283e-07` | `4.946553485707383e-07` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM111/f_hz` | `6927818052.895847` | `6927818052.895831` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM210/amp_max` | `3.45022247457572e-05` | `3.450222474559585e-05` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM210/error_max` | `7.207896490779575e-08` | `7.207896646210799e-08` | — |
| `/arms/uniform_fine/measured/windows/full/per_mode/TM210/f_hz` | `7804432307.653516` | `7804432307.653458` | — |
| `/commit` | `"83b0756ed9ecd144fab040251f65189afdd6a835"` | `"unknown"` | — |
| `/date_utc` | `"2026-09-02T13:59:23+00:00"` | `"2026-09-07T12:53:11+00:00"` | — |
| `/evaluations/multi_band/rows/TE011/allowance_bound` | `0.000582490844878473` | `0.000582490844868259` | — |
| `/evaluations/multi_band/rows/TE011/dev_raw` | `-0.00026190465056541434` | `-0.00026190465056685763` | — |
| `/evaluations/multi_band/rows/TE011/dev_spatial` | `-0.0003784322222868397` | `-0.000378432222288283` | — |
| `/evaluations/multi_band/rows/TE011/f_meas_hz` | `6244040436.688446` | `6244040436.6884365` | — |
| `/evaluations/multi_band/rows/TE011/resid_lattice` | `-4.135279563222127e-09` | `-4.135281117534362e-09` | — |
| `/evaluations/multi_band/rows/TE011/stationarity` | `1.5982075022395575e-08` | `1.5982086477410098e-08` | — |
| `/evaluations/multi_band/rows/TE101/allowance_bound` | `0.0005526282718047595` | `0.0005526282717986532` | — |
| `/evaluations/multi_band/rows/TE101/dev_raw` | `-0.0001410942174593366` | `-0.00014109421745678308` | — |
| `/evaluations/multi_band/rows/TE101/dev_spatial` | `-0.0002099180252798094` | `-0.00020991802527703385` | — |
| `/evaluations/multi_band/rows/TE101/f_meas_hz` | `4798343770.952076` | `4798343770.952088` | — |
| `/evaluations/multi_band/rows/TE101/resid_lattice` | `3.961832195642501e-09` | `3.96183486017776e-09` | — |
| `/evaluations/multi_band/rows/TE101/stationarity` | `6.1340067424938176e-09` | `6.134040331366789e-09` | — |
| `/evaluations/multi_band/rows/TE102/allowance_bound` | `0.0018395267501703704` | `0.0018395267501719248` | — |
| `/evaluations/multi_band/rows/TE102/dev_raw` | `-0.00018210000144303073` | `-0.00018210000144258665` | — |
| `/evaluations/multi_band/rows/TE102/dev_spatial` | `-0.0003767897085522609` | `-0.00037678970855192784` | — |
| `/evaluations/multi_band/rows/TE102/f_meas_hz` | `8070689031.169416` | `8070689031.16942` | — |
| `/evaluations/multi_band/rows/TE102/resid_lattice` | `-7.022076586871151e-08` | `-7.02207653136e-08` | — |
| `/evaluations/multi_band/rows/TE102/stationarity` | `5.438967121913395e-08` | `5.438967062830809e-08` | — |
| `/evaluations/multi_band/rows/TE201/allowance_bound` | `0.000700235225812539` | `0.0007002352258116509` | — |
| `/evaluations/multi_band/rows/TE201/dev_raw` | `-0.0003908815593682835` | `-0.00039088155937549995` | — |
| `/evaluations/multi_band/rows/TE201/dev_spatial` | `-0.0005401640344043779` | `-0.0005401640344115943` | — |
| `/evaluations/multi_band/rows/TE201/f_meas_hz` | `7067827217.133965` | `7067827217.133914` | — |
| `/evaluations/multi_band/rows/TE201/resid_lattice` | `9.982889404014372e-09` | `9.982882298587015e-09` | — |
| `/evaluations/multi_band/rows/TE201/stationarity` | `3.234761904255376e-08` | `3.23476083829455e-08` | — |
| `/evaluations/multi_band/rows/TM110/allowance_bound` | `0.00038330484533377217` | `0.00038330484533821306` | — |
| `/evaluations/multi_band/rows/TM110/dev_raw` | `-0.00027805468038444303` | `-0.00027805468038155645` | — |
| `/evaluations/multi_band/rows/TM110/dev_spatial` | `-0.00037947582325725815` | `-0.00037947582325437157` | — |
| `/evaluations/multi_band/rows/TM110/f_meas_hz` | `5825297801.628358` | `5825297801.628375` | — |
| `/evaluations/multi_band/rows/TM110/resid_lattice` | `-1.488704659546869e-08` | `-1.4887043708888825e-08` | — |
| `/evaluations/multi_band/rows/TM110/stationarity` | `5.976126028106317e-08` | `5.97612691215403e-08` | — |
| `/evaluations/multi_band/rows/TM111/allowance_bound` | `0.0005052708754791538` | `0.0005052708754741578` | — |
| `/evaluations/multi_band/rows/TM111/stationarity` | `2.680067219331596e-08` | `2.680068389642e-08` | — |
| `/evaluations/multi_band/rows/TM210/allowance_bound` | `0.000579467222820536` | `0.0005794672228187596` | — |
| `/evaluations/multi_band/rows/TM210/dev_raw` | `-0.0003935482628516507` | `-0.000393548262848209` | — |
| `/evaluations/multi_band/rows/TM210/dev_spatial` | `-0.0005754422746072629` | `-0.0005754422746040433` | — |
| `/evaluations/multi_band/rows/TM210/f_meas_hz` | `7801774909.3333435` | `7801774909.33337` | — |
| `/evaluations/multi_band/rows/TM210/resid_lattice` | `2.7664418089656806e-08` | `2.766442142032588e-08` | — |
| `/evaluations/multi_band/rows/TM210/stationarity` | `1.65301691630456e-07` | `1.6530167720635697e-07` | — |
| `/evaluations/single_band/rows/TE011/allowance_bound` | `0.0005180132460579959` | `0.0005180132460477819` | — |
| `/evaluations/single_band/rows/TE011/dev_raw` | `-0.00026699686147735324` | `-0.00026699686147946267` | — |
| `/evaluations/single_band/rows/TE011/dev_spatial` | `-0.00038352265263430496` | `-0.00038352265263619234` | — |
| `/evaluations/single_band/rows/TE011/f_meas_hz` | `6244008632.387905` | `6244008632.387892` | — |
| `/evaluations/single_band/rows/TE011/resid_lattice` | `-4.583902257770944e-09` | `-4.583904367194691e-09` | — |
| `/evaluations/single_band/rows/TE011/stationarity` | `2.372796656476227e-08` | `2.3727961982733184e-08` | — |
| `/evaluations/single_band/rows/TE101/allowance_bound` | `0.00044341824670232535` | `0.0004434182466962191` | — |
| `/evaluations/single_band/rows/TE101/dev_raw` | `-0.0001497173903298732` | `-0.0001497173903454163` | — |
| `/evaluations/single_band/rows/TE101/dev_spatial` | `-0.00021853941750005212` | `-0.00021853941751559525` | — |
| `/evaluations/single_band/rows/TE101/f_meas_hz` | `4798302388.165375` | `4798302388.1653` | — |
| `/evaluations/single_band/rows/TE101/resid_lattice` | `2.3745194699387184e-09` | `2.3745039268163737e-09` | — |
| `/evaluations/single_band/rows/TE101/stationarity` | `3.1440867404718724e-10` | `3.144108603240525e-10` | — |
| `/evaluations/single_band/rows/TE102/allowance_bound` | `0.0015307259895359017` | `0.001530725989537456` | — |
| `/evaluations/single_band/rows/TE102/dev_raw` | `-0.0008213524056172039` | `-0.0008213524056217558` | — |
| `/evaluations/single_band/rows/TE102/dev_spatial` | `-0.0010156689303580935` | `-0.0010156689303625344` | — |
| `/evaluations/single_band/rows/TE102/f_meas_hz` | `8065528884.140119` | `8065528884.140081` | — |
| `/evaluations/single_band/rows/TE102/resid_lattice` | `-5.723885487807223e-08` | `-5.7238859541008935e-08` | — |
| `/evaluations/single_band/rows/TE102/stationarity` | `5.982835225457299e-08` | `5.982834823438721e-08` | — |
| `/evaluations/single_band/rows/TE201/allowance_bound` | `0.0006499249895293952` | `0.0006499249895285071` | — |
| `/evaluations/single_band/rows/TE201/dev_raw` | `-0.00039485602172051326` | `-0.0003948560217162944` | — |
| `/evaluations/single_band/rows/TE201/dev_spatial` | `-0.0005441367161681532` | `-0.0005441367161640454` | — |
| `/evaluations/single_band/rows/TE201/f_meas_hz` | `7067799115.336304` | `7067799115.336333` | — |
| `/evaluations/single_band/rows/TE201/resid_lattice` | `9.525269906163203e-09` | `9.525273902966092e-09` | — |
| `/evaluations/single_band/rows/TE201/stationarity` | `1.0980398011624492e-08` | `1.0980390860213102e-08` | — |
| `/evaluations/single_band/rows/TM110/allowance_bound` | `0.00038330484533377217` | `0.00038330484533821306` | — |
| `/evaluations/single_band/rows/TM110/dev_raw` | `-0.0002780465125449272` | `-0.00027804651254481616` | — |
| `/evaluations/single_band/rows/TM110/dev_spatial` | `-0.0003794676579036427` | `-0.0003794676579031986` | — |
| `/evaluations/single_band/rows/TM110/f_meas_hz` | `5825297849.221689` | `5825297849.22169` | — |
| `/evaluations/single_band/rows/TM110/resid_lattice` | `-6.716935452288908e-09` | `-6.7169353412666055e-09` | — |
| `/evaluations/single_band/rows/TM110/stationarity` | `6.185421740530157e-08` | `6.185422690062898e-08` | — |
| `/evaluations/single_band/rows/TM111/allowance_bound` | `0.0004528671053064644` | `0.0004528671053014684` | — |
| `/evaluations/single_band/rows/TM111/dev_raw` | `-0.00019909904324244554` | `-0.00019909904323889283` | — |
| `/evaluations/single_band/rows/TM111/dev_spatial` | `-0.0003425004344885796` | `-0.00034250043448491585` | — |
| `/evaluations/single_band/rows/TM111/f_meas_hz` | `6926536873.845411` | `6926536873.845436` | — |
| `/evaluations/single_band/rows/TM111/resid_lattice` | `-1.2570714469362088e-09` | `-1.2570677832002275e-09` | — |
| `/evaluations/single_band/rows/TM111/stationarity` | `3.115266360317501e-08` | `3.1152653139179775e-08` | — |
| `/evaluations/single_band/rows/TM210/allowance_bound` | `0.000579467222820536` | `0.0005794672228187596` | — |
| `/evaluations/single_band/rows/TM210/dev_raw` | `-0.000393497164805634` | `-0.00039349716480685526` | — |
| `/evaluations/single_band/rows/TM210/dev_spatial` | `-0.0005753912044542675` | `-0.0005753912044554887` | — |
| `/evaluations/single_band/rows/TM210/f_meas_hz` | `7801775308.145749` | `7801775308.14574` | — |
| `/evaluations/single_band/rows/TM210/resid_lattice` | `7.878258290183737e-08` | `7.878258179161435e-08` | — |
| `/evaluations/single_band/rows/TM210/stationarity` | `1.0799974692852881e-07` | `1.0799975047343432e-07` | — |
| `/evaluations/uniform/rows/TE011/dev_raw` | `-0.00015193414457415066` | `-0.0001519341445639366` | — |
| `/evaluations/uniform/rows/TE011/dev_spatial` | `-0.00038505804841704183` | `-0.0003850580484068278` | — |
| `/evaluations/uniform/rows/TE011/f_meas_hz` | `6244727276.861334` | `6244727276.861398` | — |
| `/evaluations/uniform/rows/TE011/resid_lattice` | `-1.4973263495754452e-07` | `-1.497326247434927e-07` | — |
| `/evaluations/uniform/rows/TE011/stationarity` | `2.237091880423147e-08` | `2.237091193197848e-08` | — |
| `/evaluations/uniform/rows/TE101/dev_raw` | `-8.332956622703058e-05` | `-8.33295662211464e-05` | — |
| `/evaluations/uniform/rows/TE101/dev_spatial` | `-0.00022099819649745722` | `-0.000220998196491351` | — |
| `/evaluations/uniform/rows/TE101/f_meas_hz` | `4798620984.719757` | `4798620984.719786` | — |
| `/evaluations/uniform/rows/TE101/resid_lattice` | `-1.0187778398673686e-07` | `-1.0187777799153253e-07` | — |
| `/evaluations/uniform/rows/TE101/stationarity` | `8.07638972567181e-09` | `8.076403836158229e-09` | — |
| `/evaluations/uniform/rows/TE102/dev_raw` | `-0.000520162589831008` | `-0.0005201625898323403` | — |
| `/evaluations/uniform/rows/TE102/dev_spatial` | `-0.0009091244682669641` | `-0.0009091244682685184` | — |
| `/evaluations/uniform/rows/TE102/f_meas_hz` | `8067960136.21369` | `8067960136.213679` | — |
| `/evaluations/uniform/rows/TE102/resid_lattice` | `-4.7617291587531696e-07` | `-4.761729172075846e-07` | — |
| `/evaluations/uniform/rows/TE102/stationarity` | `2.8838675799047446e-08` | `2.883867839956045e-08` | — |
| `/evaluations/uniform/rows/TE201/dev_raw` | `-0.000246623676910529` | `-0.00024662367690930775` | — |
| `/evaluations/uniform/rows/TE201/dev_spatial` | `-0.0005453045169631077` | `-0.0005453045169622195` | — |
| `/evaluations/uniform/rows/TE201/f_meas_hz` | `7068847205.616574` | `7068847205.616583` | — |
| `/evaluations/uniform/rows/TE201/resid_lattice` | `-7.332136109372556e-08` | `-7.332135998350253e-08` | — |
| `/evaluations/uniform/rows/TE201/stationarity` | `3.0506791809741964e-08` | `3.050679680149641e-08` | — |
| `/evaluations/uniform/rows/TM110/dev_raw` | `-0.00017640685863151706` | `-0.0001764068586357359` | — |
| `/evaluations/uniform/rows/TM110/dev_spatial` | `-0.0003793048453337722` | `-0.00037930484533821307` | — |
| `/evaluations/uniform/rows/TM110/f_meas_hz` | `5825890095.150942` | `5825890095.150917` | — |
| `/evaluations/uniform/rows/TM110/resid_lattice` | `1.5622221627431543e-07` | `1.5622221205546793e-07` | — |
| `/evaluations/uniform/rows/TM110/stationarity` | `5.2704767044230826e-08` | `5.270478194055786e-08` | — |
| `/evaluations/uniform/rows/TM111/dev_raw` | `-5.7146957446962965e-05` | `-5.714695744196696e-05` | — |
| `/evaluations/uniform/rows/TM111/dev_spatial` | `-0.0003440595649610856` | `-0.0003440595649560896` | — |
| `/evaluations/uniform/rows/TM111/f_meas_hz` | `6927520306.002395` | `6927520306.002429` | — |
| `/evaluations/uniform/rows/TM111/resid_lattice` | `-4.308125148844155e-07` | `-4.308125099994342e-07` | — |
| `/evaluations/uniform/rows/TM111/stationarity` | `6.26575041656587e-08` | `6.26575074696088e-08` | — |
| `/evaluations/uniform/rows/TM210/dev_raw` | `-0.00021150028929295406` | `-0.00021150028929139975` | — |
| `/evaluations/uniform/rows/TM210/dev_spatial` | `-0.000575467222820536` | `-0.0005754672228187596` | — |
| `/evaluations/uniform/rows/TM210/f_meas_hz` | `7803195765.821351` | `7803195765.821363` | — |
| `/evaluations/uniform/rows/TM210/resid_lattice` | `2.693732348291178e-09` | `2.6937339026034124e-09` | — |
| `/evaluations/uniform/rows/TM210/stationarity` | `1.1543419698814544e-07` | `1.1543421006524276e-07` | — |
| `/evaluations/uniform_fine/rows/TE011/allowance_bound` | `0.0003890580484170418` | `0.00038905804840682777` | — |
| `/evaluations/uniform_fine/rows/TE011/dev_raw` | `-3.806735248568227e-05` | `-3.806735248512716e-05` | — |
| `/evaluations/uniform_fine/rows/TE011/dev_spatial` | `-9.637130112538639e-05` | `-9.637130112483128e-05` | — |
| `/evaluations/uniform_fine/rows/TE011/f_meas_hz` | `6245438451.9756` | `6245438451.975603` | — |
| `/evaluations/uniform_fine/rows/TE011/resid_lattice` | `-1.3450261504566186e-07` | `-1.3450261460157265e-07` | — |
| `/evaluations/uniform_fine/rows/TE011/stationarity` | `3.24790101349856e-07` | `3.2479010104445714e-07` | — |
| `/evaluations/uniform_fine/rows/TE101/allowance_bound` | `0.0002249981964974572` | `0.00022499819649135098` | — |
| `/evaluations/uniform_fine/rows/TE101/dev_raw` | `-2.0492481441647392e-05` | `-2.049248144486704e-05` | — |
| `/evaluations/uniform_fine/rows/TE101/dev_spatial` | `-5.4917194373116374e-05` | `-5.4917194376113976e-05` | — |
| `/evaluations/uniform_fine/rows/TE101/f_meas_hz` | `4798922541.201998` | `4798922541.2019825` | — |
| `/evaluations/uniform_fine/rows/TE101/resid_lattice` | `3.09987040481019e-07` | `3.099870371503499e-07` | — |
| `/evaluations/uniform_fine/rows/TE101/stationarity` | `4.091375638188538e-07` | `4.0913757594118734e-07` | — |
| `/evaluations/uniform_fine/rows/TE102/allowance_bound` | `0.0009131244682669641` | `0.0009131244682685184` | — |
| `/evaluations/uniform_fine/rows/TE102/dev_raw` | `-0.00012943371659646097` | `-0.00012943371659446257` | — |
| `/evaluations/uniform_fine/rows/TE102/dev_spatial` | `-0.00022679679782544593` | `-0.00022679679782355855` | — |
| `/evaluations/uniform_fine/rows/TE102/f_meas_hz` | `8071114161.793128` | `8071114161.793144` | — |
| `/evaluations/uniform_fine/rows/TE102/resid_lattice` | `4.2542572087356234e-07` | `4.254257228719638e-07` | — |
| `/evaluations/uniform_fine/rows/TE102/stationarity` | `1.0257506194066907e-06` | `1.025750579468966e-06` | — |
| `/evaluations/uniform_fine/rows/TE201/allowance_bound` | `0.0005493045169631077` | `0.0005493045169622195` | — |
| `/evaluations/uniform_fine/rows/TE201/dev_raw` | `-6.11364976158546e-05` | `-6.113649761507745e-05` | — |
| `/evaluations/uniform_fine/rows/TE201/dev_spatial` | `-0.00013585329880994035` | `-0.00013585329880938524` | — |
| `/evaluations/uniform_fine/rows/TE201/f_meas_hz` | `7070158709.593542` | `7070158709.593548` | — |
| `/evaluations/uniform_fine/rows/TE201/resid_lattice` | `4.7625492993752516e-07` | `4.762549308257036e-07` | — |
| `/evaluations/uniform_fine/rows/TE201/stationarity` | `8.485406307259218e-08` | `8.485406401680292e-08` | — |
| `/evaluations/uniform_fine/rows/TM110/allowance_bound` | `0.00038330484533377217` | `0.00038330484533821306` | — |
| `/evaluations/uniform_fine/rows/TM110/dev_raw` | `-4.42133996956251e-05` | `-4.421339969429283e-05` | — |
| `/evaluations/uniform_fine/rows/TM110/dev_spatial` | `-9.496033591926967e-05` | `-9.496033591771535e-05` | — |
| `/evaluations/uniform_fine/rows/TM110/f_meas_hz` | `5826660375.596754` | `5826660375.596762` | — |
| `/evaluations/uniform_fine/rows/TM110/resid_lattice` | `-8.451840560752544e-08` | `-8.451840438628011e-08` | — |
| `/evaluations/uniform_fine/rows/TM110/stationarity` | `3.041852535767511e-07` | `3.0418525292205366e-07` | — |
| `/evaluations/uniform_fine/rows/TM111/allowance_bound` | `0.0003480595649610856` | `0.00034805956495608957` | — |
| `/evaluations/uniform_fine/rows/TM111/dev_raw` | `-1.4169114730844257e-05` | `-1.4169114733175725e-05` | — |
| `/evaluations/uniform_fine/rows/TM111/dev_spatial` | `-8.591114742317885e-05` | `-8.591114742528827e-05` | — |
| `/evaluations/uniform_fine/rows/TM111/f_meas_hz` | `6927818052.895847` | `6927818052.895831` | — |
| `/evaluations/uniform_fine/rows/TM111/resid_lattice` | `5.015990289791716e-09` | `5.015988069345667e-09` | — |
| `/evaluations/uniform_fine/rows/TM111/stationarity` | `3.389808804817593e-07` | `3.3898088309727507e-07` | — |
| `/evaluations/uniform_fine/rows/TM210/allowance_bound` | `0.000579467222820536` | `0.0005794672228187596` | — |
| `/evaluations/uniform_fine/rows/TM210/dev_raw` | `-5.3067726492939116e-05` | `-5.306772650037761e-05` | — |
| `/evaluations/uniform_fine/rows/TM210/dev_spatial` | `-0.0001441101810618628` | `-0.0001441101810694123` | — |
| `/evaluations/uniform_fine/rows/TM210/f_meas_hz` | `7804432307.653516` | `7804432307.653458` | — |
| `/evaluations/uniform_fine/rows/TM210/resid_lattice` | `-2.2267610355619638e-07` | `-2.2267611099469065e-07` | — |
| `/evaluations/uniform_fine/rows/TM210/stationarity` | `9.334670445728337e-07` | `9.334670366300681e-07` | — |

## validation/crossval/_issue812_phase_identity/regate_evidence.json

| JSON field | main | branch | Changed array indices |
|---|---|---|---|
| `/cv20/blindness/audit_construction_e1_max_phase_dev_deg` | `0.24144025314175127` | `0.0646838247297135` | — |
| `/cv20/blindness/dispersion_corrected_residual_max_abs_change_deg_k2` | `1.0658141036401503e-14` | `8.770761894538737e-15` | — |
| `/cv20/blindness/dispersion_corrected_residual_max_abs_deg_baseline` | `0.7153450725324438` | `0.7917114796063136` | — |
| `/cv20/blindness/dispersion_corrected_residual_max_abs_deg_k2` | `0.7153450725324506` | `0.7917114796063096` | — |
| `/cv20/criterion_b/0/e1_max_phase_dev_deg` | `0.12072012657087881` | `0.032341912364863105` | — |
| `/cv20/criterion_b/0/e2_analytic_beta_rfx_dev_frac` | `0.4956433472800128` | `0.4936334711257333` | — |
| `/cv20/criterion_b/0/e2_failure_reason` | `"[perturbed] analytic-beta witness failed for solver 'rfx': its MEASURED beta disagrees with the Hammerstad-Jensen quasi-static closed form for this board (eps_eff=2.832693) by 0.495643 > 0.020000. The reference here contains NO quantity from the run, so -- unlike the self-consistency witness -- a coherent phase-velocity error cannot satisfy it."` | `"[perturbed] analytic-beta witness failed for solver 'rfx': its MEASURED beta disagrees with the Hammerstad-Jensen quasi-static closed form for this board (eps_eff=2.872970) by 0.493633 > 0.020000. The reference here contains NO quantity from the run, so -- unlike the self-consistency witness -- a coherent phase-velocity error cannot satisfy it."` | — |
| `/cv20/criterion_b/0/e4_cross_solver_max_abs_raw_phase_diff_deg` | `22.18826368420616` | `22.067283269346667` | — |
| `/cv20/criterion_b/0/e4_failure_reason` | `"[perturbed] cross-solver phase witness failed: max &#124;angle(S21_rfx) - angle(S21_openems)&#124; = 22.1883 deg > 3.0000 deg over the 3.0-4.5 GHz gate band. ONE OF THE TWO SOLVERS' de-embedded phase is wrong beyond the +-1-cell mesh and +-4-cell reference-plane budget -- this witness does NOT say which; read the analytic-beta witness for that."` | `"[perturbed] cross-solver phase witness failed: max &#124;angle(S21_rfx) - angle(S21_openems)&#124; = 22.0673 deg > 3.0000 deg over the 3.0-4.5 GHz gate band. ONE OF THE TWO SOLVERS' de-embedded phase is wrong beyond the +-1-cell mesh and +-4-cell reference-plane budget -- this witness does NOT say which; read the analytic-beta witness for that."` | — |
| `/cv20/criterion_b/1/e1_max_phase_dev_deg` | `0.12072012657087564` | `0.032341912364863105` | — |
| `/cv20/criterion_b/1/e2_analytic_beta_rfx_dev_frac` | `1.0187644160757117` | `1.0282436367527041` | — |
| `/cv20/criterion_b/1/e2_failure_reason` | `"[perturbed] analytic-beta witness failed for solver 'rfx': its MEASURED beta disagrees with the Hammerstad-Jensen quasi-static closed form for this board (eps_eff=2.832693) by 1.018764 > 0.020000. The reference here contains NO quantity from the run, so -- unlike the self-consistency witness -- a coherent phase-velocity error cannot satisfy it."` | `"[perturbed] analytic-beta witness failed for solver 'rfx': its MEASURED beta disagrees with the Hammerstad-Jensen quasi-static closed form for this board (eps_eff=2.872970) by 1.028244 > 0.020000. The reference here contains NO quantity from the run, so -- unlike the self-consistency witness -- a coherent phase-velocity error cannot satisfy it."` | — |
| `/cv20/criterion_b/1/e4_cross_solver_max_abs_raw_phase_diff_deg` | `44.814580677298274` | `45.72707401242633` | — |
| `/cv20/criterion_b/1/e4_failure_reason` | `"[perturbed] cross-solver phase witness failed: max &#124;angle(S21_rfx) - angle(S21_openems)&#124; = 44.8146 deg > 3.0000 deg over the 3.0-4.5 GHz gate band. ONE OF THE TWO SOLVERS' de-embedded phase is wrong beyond the +-1-cell mesh and +-4-cell reference-plane budget -- this witness does NOT say which; read the analytic-beta witness for that."` | `"[perturbed] cross-solver phase witness failed: max &#124;angle(S21_rfx) - angle(S21_openems)&#124; = 45.7271 deg > 3.0000 deg over the 3.0-4.5 GHz gate band. ONE OF THE TWO SOLVERS' de-embedded phase is wrong beyond the +-1-cell mesh and +-4-cell reference-plane budget -- this witness does NOT say which; read the analytic-beta witness for that."` | — |
| `/cv20/eps_eff_hammerstad_jensen_realized_board` | `2.832692749102272` | `2.8326927491022724` | — |
| `/cv20/eps_eff_hammerstad_jensen_rfx_board_post_931` | *absent* | `2.872970226316938` | — |
| `/cv20/run2_openems_with_current_rfx_fixture/all_three_passed` | *absent* | `true` | — |
| `/cv20/run2_openems_with_current_rfx_fixture/analytic_beta_openems_max_abs_dev_frac` | *absent* | `0.0030679160312854226` | — |
| `/cv20/run2_openems_with_current_rfx_fixture/analytic_beta_rfx_max_abs_dev_frac` | *absent* | `0.014121818376352069` | — |
| `/cv20/run2_openems_with_current_rfx_fixture/analytic_beta_tol_frac` | *absent* | `0.02` | — |
| `/cv20/run2_openems_with_current_rfx_fixture/cross_solver_margin_x` | *absent* | `5.651464842989515` | — |
| `/cv20/run2_openems_with_current_rfx_fixture/cross_solver_max_abs_raw_phase_diff_deg` | *absent* | `0.5308358245776609` | — |
| `/cv20/run2_openems_with_current_rfx_fixture/cross_solver_tol_deg` | *absent* | `3.0` | — |
