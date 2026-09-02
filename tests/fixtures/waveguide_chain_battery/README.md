# waveguide_chain_battery — fixture JSON schema

`fixture.json` in this directory is written by the measurement driver
`scripts/diagnostics/waveguide_chain_battery_measure.py` (one JSON per case persisted as it
finishes, then assembled) and replayed by `tests/test_waveguide_chain_battery.py`. The
pre-declaration `docs/design_notes/waveguide_chain_battery_predeclaration.md` and the builder
`tests/_waveguide_chain_battery_fixture.py` were committed first (PR #861) so that every
tolerance, position and drive setting provably predates the first measured S-parameter. This
file fixes the schema the measurement writes, so the writer and the replay gate cannot drift.
Gate arithmetic shared by the writer and the replay: `tests/_waveguide_chain_battery_gates.py`.

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
| `port_cutoff` | object | report-only mechanism witness: the port config's `f_cutoff` (the β / Z_TE the extractor uses) against the guide's cutoff fitted from the thru's S21 phase, per rung and lane (see below) |
| `legs_rung` | string | the rung the AD/FD and plane-shift legs ran at (`"fine"`, the claims rung) |
| `readme` | string | this file's path |
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
| `recapture_command` | string | the exact command that regenerates this file from a clean checkout |
| `recapture_entry_point`, `recapture_vessl_yaml` | string | the tracked driver path and the tracked VESSL YAML the measurement ran under (bare paths, `git ls-files`-resolvable) |
| `wall_time_note` | string | what `wall_time_s` sums |

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
| `rung` | string | `"coarse"`, `"mid"`, `"fine"` |
| `cpml_layers` | int | derived count (17 / 34 / 68) |
| `fc_te10_numerical_hz` | number | from `numerical_te10_cutoff_hz` on this cell's grid (preflight's wall-to-wall reader) |
| `port_f_cutoff_hz` | [number, number] | `WaveguidePortConfig.f_cutoff` of the two ports — the cutoff the extractor's β / Z_TE actually use (`port_cutoff_hz` in the builder) |
| `fc_discrete_guide_hz` | number | TE10 cutoff of the Yee-discretized 9/18/36-cell guide, `kc = (2/dx)·sin(π·dx/2a)` |
| `num_periods` | number | 40 |
| `n_steps` | int | realised from `num_periods` |
| `dt_s` | number | grid time step |
| `grid_shape` | [int, int, int] | padded grid |
| `guide_cells_yz` | [int, int] | 9/18/36 and 4/8/16 |
| `dut_cells` | int or null | rasterized DUT cell count (null for thru) |
| `dut_runs_xyz` | [int, int, int] or null | per-axis run lengths |
| `preflight` | array of {`code`, `severity`, `message`} | every finding, verbatim, in emission order; empty list when clean |
| `settling_db` | object | `{port_name: value_db}` per driven port — the ring-down witness, mandatory |
| `settling_records` | array | per extractor call, per port and record (`v_probe_t`, `v_ref_t`, `i_probe_t`, `i_ref_t`): `peak`, `end` (mean of the last 10 %), `db`, `n_nonzero`, `peak_is_zero` — the numbers the witness is built from, so a degenerate record (peak exactly 0 → the witness reads 0 dB) is visible |
| `settling_rerun` | object or null | present when a drive exceeded −40 dB: `{num_periods, n_steps, settling_db, settling_records, s_params, max_abs_s_shift_vs_40_periods, ...metrics}` of the doubled-record rerun |
| `warnings` | array of {`message`, `count`} | every Python warning emitted by the solve, verbatim, deduplicated with a count |
| `s_params` | object | `{"S11": [[re, im], ...], "S21": ..., "S12": ..., "S22": ...}` per frequency bin, at the default reference planes |
| `reference_planes_m` | [number, number] | as reported by `WaveguideSMatrixResult.reference_planes` |
| `column_power_max` | number | `max_f Σ_i |S_ij|²` |
| `reciprocity_mag_mean` | number | `mean_f |S21|-|S12| / max` |
| `reciprocity_complex_max` | number | `max_f |S21 − S12| / max|S|` |
| `power_closure_max` | number | `max_f |1 − Σ_i |S_ij|²|` (slab only meaningful) |
| `non_vacuity_max_s11` | number | `max_f |S11|`; must exceed 0.20 on both reflecting DUTs |

## `ladder`

Keyed `"<observable>|<lane>"` with observable ∈ (`slab_s11_mag`, `slab_s21_mag`,
`slab_s21_phase_deg`, `pec_short_s11_mag`, `pec_short_s11_phase_deg`). Each holds:
`values_by_rung` (per bin, per rung), `coarse_delta_per_bin`, `fine_delta_per_bin`,
`coarse_delta_worst`, `fine_delta_worst`, `excess_worst` (= max over bins of
fine_delta − coarse_delta; the gate is `excess ≤ floor` on EVERY bin), `worst_bin_hz`, `floor`
(0.005 or 1.0), `gate_pass` (bool), `monotone_fraction_of_bins`, `successive_ratio_per_bin`
(null where the coarse delta is inside the floor — the ratio is not conditioned there),
`successive_ratio_worst` (the conditioned ratio farthest, in log space, from the window's centre),
`successive_ratio_worst_bin_hz`, `n_conditioned_bins`, `ratio_window`, `interpretable` (bool: every
conditioned ratio inside [0.15, 0.70]; vacuously true with no conditioned bin), `richardson` = per
adjacent pair `{pair: [dx_a, dx_b], estimate_per_bin, oracle_per_bin, abs_diff_per_bin,
max_abs_diff, max_abs_diff_bin_hz, finer_rung_abs_diff_max, oracle_continuous_per_bin,
max_abs_diff_continuous (PEC-short phase only)}` (absent for `pec_short_s11_mag`),
`richardson_max_abs_diff`, `pinned_richardson_gate` / `pinned_monotone_fraction_min` (null on the
first run, filled by the pin commit), and `verdict` ∈ `{"pass", "fail", "not_interpretable"}`. If a
fourth rung a/72 was added, it appears in `values_by_rung` and `fixture.dx_ladder_m` records it.

## `plane_shift`

Keyed `"<dut>|<lane>"` (the legs rung): `dx_m`, `rung`, `reference_planes_base_m`,
`reference_planes_shifted_m` (as reported by the result), `shift_m`, `s_params_shifted`,
`settling_db_shifted`, `abs_s_max_diff` and `abs_s_allclose` (whole matrix, base vs shifted),
`fc_port_hz` / `fc_predeclared_hz`,
`rotation_deg` = `{S11: {predicted_yee, predicted_continuous, predicted_port_beta, measured,
resid_yee_max, resid_cont_max, resid_port_beta_max, wrong_sign_resid_min}, S22: ..., S21: ...,
S12: ...}` per bin (`predicted_port_beta` uses the extractor's own `f_cutoff`; `resid_port_beta_max`
is the mechanism witness that the shift is a pure `exp(∓jβ_port·s)`), the matrix-wide
`resid_yee_max`, `resid_cont_max`, `resid_port_beta_max`, `wrong_sign_resid_min`, and
`gradient_invariance` keyed `"<theta_kind>:<objective>"` (`s21_complex` / `s11_complex` combine the
re/im objectives) = `{kind: "magnitude"|"complex", value_base, value_shifted, rotated_base
(complex only), phi_measured_deg, phi_predeclared_deg, rel_change, rel_change_predeclared_phi,
report_bar: 0.01, pinned_gate: null|number}` or `{skipped_under_ulp_floor: true, reason}` when the
base FD leg could not resolve the gradient. `rel_change` (the tested quantity) rotates by the
measured `∠(S_shift/S_base)` at the band-centre bin — the unit-modulus factor the extractor
applied — so it isolates the gradient property from the value of β; `rel_change_predeclared_phi`
uses `2β_yee(c/2a)Δ`. `pinned_gate` stays null on the first run and is filled by the same PR from
`gate_from_envelope(measured, quantum=1000)`.

`plane_shift.cheap_refute` records the §8 refute (plane-shift stage under a local copy of
`_shift_modal_waves` with the shift sign flipped): `resid_yee_min_over_entries`,
`resid_yee_max_over_entries`, `rotation_gate_would_pass` (must be false), `abs_s_still_invariant`,
`per_case`, `rung`.

## `ad_vs_fd[]`

`dut`, `lane`, `dx_m`, `rung`, `objective` (`s11_mag2`, `s21_mag2`, `re_s21`, `im_s21`,
`re_s11`, `im_s11`), `theta_kind` (`"eps"` or `"sigma"`), `theta0`, `h`, `x64_context` (bool),
`loss_dtype`, `s_dtype_fd`, `checkpoint_segments`, `value_at_theta0`, `grad_dtype`,
`expected_ulp_floor_skip` (bool), `f_plus`, `f_minus`, `fd_ulp_span`, `ulp_floor` (1e4), `g_ad`,
`g_fd`, `rel`, `gate` (0.05), `verdict` ∈ `{"pass", "fail", "skipped_under_ulp_floor"}`,
`forward_identity` = `{max_abs_diff, max_scaled_diff, worst_entry, abs_s_at_worst, rtol, atol,
pass}` (the θ0 reverse-mode-traced primal vs the untraced call; `max_scaled_diff ≤ 1` is the
rtol 1e-5 / atol 1e-7 gate), `forward_identity_concrete_override_vs_plain` (eps legs: the concrete
no-op override vs the plain call, the form of `tests/test_waveguide_flux_ad.py:104`), and
`x64_witness` (report-only, first objective of each (dut, lane, kind) and every non-finite float32
gradient): `{g_ad_x64, value_x64, forward_identity_x64}` from the same reverse-mode call under a
scoped x64 context.

## `port_cutoff`

`length_between_declared_planes_m` (0.08128) and `per_rung["<rung>|<lane>"]` =
`{fc_fit_hz, rms_deg_at_fit, const_deg_at_fit, fc_c_over_2a_hz, rms_deg_at_c_over_2a,
fc_discrete_guide_hz, rms_deg_at_discrete_guide, fc_port_hz, rms_deg_at_port_cutoff,
port_cutoff_effective_width_cells}` — the guide's TE10 cutoff fitted from the thru's S21 phase
(`unwrap(∠S21) = −β_yee(f; fc)·L + const`) against the cutoff the port config carries. Report-only.

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
