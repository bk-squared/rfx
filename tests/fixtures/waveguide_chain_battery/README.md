# waveguide_chain_battery — fixture JSON schema

`fixture.json` in this directory is written by `tests/test_waveguide_chain_battery.py` on
its first run (a later PR). It does not exist yet on purpose: the pre-declaration
`docs/design_notes/waveguide_chain_battery_predeclaration.md` and the builder
`tests/_waveguide_chain_battery_fixture.py` are committed first so that every tolerance,
position and drive setting provably predates the first measured S-parameter. This file fixes
the schema the measurement writes, so the writer and the later replay gate cannot drift.

Units: metres, hertz, seconds, S/m, decibels, degrees. Complex values are written as
`[re, im]` pairs. Every measured number carries the provenance block that produced it.

## Top level

| key | type | meaning |
|---|---|---|
| `schema` | string | `"rfx.waveguide_chain_battery"` |
| `schema_version` | int | starts at 1; bump on any key change |
| `predeclaration` | string | path of the design note, plus its commit sha in `predeclaration_sha` |
| `predeclaration_sha` | string | the commit the note was read at — must predate `provenance.commit` |
| `generated_at` | string | ISO-8601 UTC |
| `provenance` | object | see below |
| `fixture` | object | the constants of the builder, restated for the record (see below) |
| `cells` | array | one entry per (dut, dx, lane) — see below |
| `ladder` | object | per-observable ladder results, witnesses and interpretability verdict |
| `plane_shift` | object | leg (b): rotations and gradient invariance / covariance |
| `ad_vs_fd` | array | leg (a): one entry per (dut, objective, theta_kind, lane) |
| `referee` | object | leg (d): PEC-short bounds and slab-vs-Airy per rung |
| `physics_gates` | object | column power, magnitude and complex reciprocity, power closure |
| `verdicts` | object | per gate: `"pass"`, `"fail"`, `"report_only"`, `"skipped"`, `"not_interpretable"` |

Nothing under `verdicts` is derived by the replay test from the numbers alone; the replay test
recomputes each gate from the stored values and compares with the stored verdict, so a
disagreement is itself a failure.

## `provenance`

| key | type | meaning |
|---|---|---|
| `commit` | string | rfx commit sha the battery ran at |
| `run_id` | string | the VESSL run id (`vessl run ...` number) or the GitHub Actions run id; `"local"` only for a developer dry run, which is never claims-bearing |
| `run_lane` | string | `"vessl"`, `"ci-fast"`, `"ci-slow:<shard>"` or `"local"` |
| `wall_time_s` | number | whole-battery wall time, the number that decides fast vs slow lane (30 s) |
| `jax_version`, `numpy_version`, `jax_default_backend` | string | as in the broad-E5 fixtures |
| `jax_enable_x64` | bool | process default; FD legs record their own per-test context in `ad_vs_fd[].x64_context` |
| `precision` | string | the `Simulation` precision argument |
| `recapture_command` | string | the exact command (or the tracked VESSL YAML path) that regenerates this file from a clean checkout |

## `fixture`

The builder constants, copied at write time so the JSON is self-describing:
`a_m`, `b_m`, `dx_ladder_m`, `n_ladder`, `domain_x_m`, `port_planes_m` (left, right),
`reference_planes_default_m`, `reference_planes_shifted_m`, `probe_planes_m`,
`pec_short_x_m`, `slab_x_m`, `slab_eps_r`, `pec_short_window_x_m`, `freqs_hz`, `f0_hz`,
`bandwidth`, `band_centre_bin`, `num_periods`, `lanes`, `boundary` (`"cpml-x, pec-y, pec-z"`).
The replay gate asserts these equal the builder's live constants; a drift means the fixture
was measured on a different geometry than the one now declared.

## `cells[]` — one per (dut, dx, lane)

| key | type | meaning |
|---|---|---|
| `dut` | string | `"thru"`, `"pec_short"`, `"slab"` |
| `dx_m` | number | rung |
| `lane` | string | `"false"` or `"flux"` (`normalize` argument as a string) |
| `cpml_layers` | int | derived count (17 / 34 / 68) |
| `fc_te10_numerical_hz` | number | from `numerical_te10_cutoff_hz` on this cell's grid |
| `n_steps` | int | realised from `num_periods` |
| `dt_s` | number | grid time step |
| `grid_shape` | [int, int, int] | padded grid |
| `guide_cells_yz` | [int, int] | 9/18/36 and 4/8/16 |
| `dut_cells` | int or null | rasterized DUT cell count (null for thru) |
| `dut_runs_xyz` | [int, int, int] or null | per-axis run lengths |
| `preflight` | array of {`code`, `severity`, `message`} | every finding, verbatim, in emission order; empty list when clean |
| `settling_db` | object | `{port_name: value_db}` per driven port — the ring-down witness, mandatory |
| `settling_rerun` | object or null | present when a drive exceeded −40 dB: `{num_periods, settling_db, s_params}` of the doubled-record rerun |
| `s_params` | object | `{"S11": [[re, im], ...], "S21": ..., "S12": ..., "S22": ...}` per frequency bin, at the default reference planes |
| `reference_planes_m` | [number, number] | as reported by `WaveguideSMatrixResult.reference_planes` |
| `column_power_max` | number | `max_f Σ_i |S_ij|²` |
| `reciprocity_mag_mean` | number | `mean_f |S21|-|S12| / max` |
| `reciprocity_complex_max` | number | `max_f |S21 − S12| / max|S|` |
| `power_closure_max` | number | `max_f |1 − Σ_i |S_ij|²|` (slab only meaningful) |
| `non_vacuity_max_s11` | number | `max_f |S11|`; must exceed 0.20 on both reflecting DUTs |

## `ladder`

Keyed by observable (`slab_s11_mag`, `slab_s21_mag`, `slab_s21_phase_deg`,
`pec_short_s11_mag`, `pec_short_s11_phase_deg`) and lane. Each holds:
`values_by_rung` (per bin, per rung), `coarse_delta_worst`, `fine_delta_worst`,
`worst_bin_hz`, `floor` (0.005 or 1.0), `gate_pass` (bool),
`monotone_fraction_of_bins`, `successive_ratio_per_bin`, `successive_ratio_worst`,
`interpretable` (bool, window [0.15, 0.70]), `richardson` = per adjacent pair
`{pair: [dx_a, dx_b], estimate_per_bin, oracle_per_bin, max_abs_diff}` (absent for
`pec_short_s11_mag`), and `verdict` ∈ `{"pass", "fail", "not_interpretable"}`. If a fourth
rung a/72 was added, it appears in `values_by_rung` and `fixture.dx_ladder_m` records it.

## `plane_shift`

Per (dut, lane): `abs_s_max_diff` (whole matrix, base vs shifted),
`rotation_deg` = `{S11: {predicted_yee, predicted_continuous, measured, resid_yee_max,
resid_cont_max, wrong_sign_resid_min}, S22: ..., S21: ..., S12: ...}` per bin,
`gradient_invariance` = per objective `{kind: "magnitude"|"complex", value_base,
value_shifted, rotated_base (complex only), rel_change, report_bar: 0.01,
pinned_gate: null|number}`. `pinned_gate` stays null on the first run and is filled by the
same PR from `gate_from_envelope(measured, quantum=1000)`.

## `ad_vs_fd[]`

`dut`, `lane`, `objective` (`s11_mag2`, `s21_mag2`, `re_s21`, `im_s21`, `re_s11`, `im_s11`),
`theta_kind` (`"eps"` or `"sigma"`), `theta0`, `h`, `x64_context` (bool), `loss_dtype`,
`f_plus`, `f_minus`, `fd_ulp_span`, `ulp_floor` (1e4), `g_ad`, `g_fd`, `rel`, `gate` (0.05),
`verdict` ∈ `{"pass", "fail", "skipped_under_ulp_floor"}`, and `forward_identity_max_diff`
(the θ = 0 traced-vs-untraced check, gate rtol 1e-5 / atol 1e-7).

## `referee`

`pec_short`: per rung and lane `{min_s11, max_s11, mean_s11, bins_above_1_03: [...]}` against
0.99 / 1.03 / 0.02. `slab_airy`: per rung and lane `{max_mag_abs_diff, max_phase_diff_deg,
worst_bin_hz, oracle_shift_convention: "exp(-2j beta_v d_L) / exp(-1j beta_v (d_L+d_R))",
d_left_m, d_right_m}` against 0.05 / 15°. `broad_e5_replay`: the five fixture paths and their
gate test, listed for the 3(d) set, not re-run here.

## `physics_gates`

Fine-rung values with their gates: `column_power` (1.02), `reciprocity_mag` (0.01),
`reciprocity_complex` (0.01, first measurement report-only if above), `power_closure`
(report-only here; gated in WP3), `settling_all_below_minus_40_db` (bool).
