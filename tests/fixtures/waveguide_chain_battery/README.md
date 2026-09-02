# waveguide_chain_battery — fixture JSON schema

`fixture.json` in this directory is written by the measurement driver
`scripts/diagnostics/waveguide_chain_battery_measure.py` (one JSON per case persisted as it
finishes, then assembled) and replayed by `tests/oracle/test_waveguide_chain_battery.py`. The
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
| `pins` | object | written by the pin step (`--stages pin`): `gradient_invariance_envelope` / `gradient_invariance_gate` (quantum 1000), `richardson_quantum` (100 for magnitudes, 10 for degrees), `monotone_quantum` (100), `policy` |
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
no-op override vs the plain call, the form of `tests/unit/autodiff/test_waveguide_flux_ad.py:104`), and
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

## Closure witness — `closure_witness.json`

Written by `scripts/diagnostics/waveguide_chain_battery_closure_measure.py`, replayed by
`tests/oracle/test_waveguide_chain_battery_closure.py`. It answers the one thing `fixture.json`
cannot answer about itself: `physics_gates[*].power_closure_gate` reads
`"report-only (WP3)"` because the flux lane's `1 − Σ|S|²` is built from Poynting integrals
taken at the two PORT PROBE planes, so port column power and the S-matrix are ONE witness.
This file records the same power balance measured at two planes the extractor never samples.

**Routes.** A (port): `1 − |S11|² − |S21|²` from
`compute_waveguide_s_matrix(normalize="flux", num_periods=40)`, column 0. B (interior): two
full-plane `add_flux_monitor` planes at `x = 0.04572 m` (`guide_in`, 18 coarse cells) and
`x = 0.07620 m` (`guide_out`, 30 coarse cells) — 4 coarse cells outside the nearer slab face,
strictly between the port probe planes (0.03810 / 0.08382 m) and the slab — read on two
single-port solves, the `slab` device run and a `thru` reference run:
`closure_M = [F_dev(guide_in) − F_dev(guide_out)] / F_ref(guide_in)`.
Gate: `max_f |closure_S − closure_M| ≤ 0.02`, the column-power tolerance named by the plan's
WP3 Falsifier. Independence scope: the PLANE INDEX only. Both routes integrate the same
transverse window with the same uniform `dA = dx²` (measured `u[0,10)`, `v[0,5)`,
`dA = 6.4516003e-06`) through `rfx/probes/probes.py::flux_spectrum` — the port's `aperture_dA`,
which drops the +face PEC cell, serves the modal V/I integral and reaches neither monitor. So a
shared-kernel defect that scales every plane equally, and any area-weighting error, cancel in
both ratios and are not caught. The reference-plane de-embedding is not caught either: both
routes are magnitude-only, `|S| = sqrt(P_num / P_inc)` is built from flux alone and `ref_shifts`
reaches only `jnp.angle(ratio)`. Measured with a positive control that the shift really acted:
moving the left reference plane five coarse cells (0.02032 → 0.03302 m) swings `∠S11` by 277.3°
while `closure_S` moves 1.09e-07 and `|S|` moves 7.0e-08, against a 0.02 gate — so the one-cell
port aperture cutoff error (issue #868) cannot reach this witness. What IS caught: any failure of power transport in the
guide between the two plane pairs. A wrong plane index is caught only when it mis-snaps into the
slab or the absorber, because the closure residual is plane-invariant in a lossless source-free
region — the port planes (k = 15/33) give `max|closure_S| = 9.033e-05` and the interior planes
(k = 18/30) give `max|closure_M| = 6.887e-05`, agreeing to 2.146e-05 across a three-cell move.

**Rung and cost.** Coarse rung only: `dx = 2.54 mm`, 17 CPML layers, grid 83 × 10 × 5,
713 steps at `num_periods = 40`, `precision="float32"`, jax 0.6.2 on CPU. Wall time
9.80 s (flux lane) + 1.16 s (device) + 1.20 s (reference) = 13.5 s total, so the live
re-measurement stays in the fast lane (contract criterion 3, "fast lane when ≤ 30 s").

**Measured** (`closure_witness.json` keys in brackets):

| quantity | band centre, 10.00 GHz | worst bin, 8.60 GHz |
|---|---|---|
| route A, port planes [`closure_s_per_bin`] | 7.067e-06 | 9.033e-05 |
| route B, interior planes [`closure_m_per_bin`] | 3.839e-06 | 6.887e-05 |
| \|A − B\| [`abs_diff_per_bin`] | 3.228e-06 | **2.146e-05** [`max_abs_diff`] |

Verdict `pass` [`verdict`]: 2.146e-05 against the 0.02 gate. Read honestly — both routes put
the closure residual at ~1e-05, the float32 field-noise floor of this rung, so the measurement
BOUNDS the disagreement rather than resolving a physical closure defect. Three checks say the
bound is real rather than an artefact of two identical computations:

- the interior planes reproduce the magnitudes separately, not only their sum:
  `1 − F_dev(in)/F_ref(in)` vs `|S11|²` differs by at most 2.674e-05, `F_dev(out)/F_ref(in)`
  vs `|S21|²` by at most 2.250e-05;
- the empty guide transports power between the two monitor planes to
  `max_f |F_ref(out)/F_ref(in) − 1| = 7.147e-06`, so neither plane is mis-snapped;
- re-summing the SAME complex64 accumulators in float64 (`flux_exact_f64`, the sanctioned
  remedy for the issue-#304 subnormal flush) moves `closure_M` by at most 3.080e-07, so the
  2.146e-05 route difference is field-level float32 noise, not host-side cancellation. The
  interior fluxes are ~1e-24 W, far above the float32 minimum normal, and no bin reads zero.

A 5 % scaling of `F_dev(guide_out)` drives the gate red (`test_a_perturbed_interior_flux_makes_the_gate_red`),
which is what says the gate measures the balance rather than passing on agreement of noise.

**Settling witness** [`flux_lane.settling_db`, `device_run.settling_db`,
`reference_run.settling_db`], all at `num_periods = 40`, all below the −40 dB line:
flux lane left −81.35 dB / right −79.54 dB; device monitor run −86.65 dB; reference monitor
run −98.72 dB.

**Preflight** [`flux_lane.preflight`, `device_run.preflight`, `reference_run.preflight`],
verbatim and part of the result. Both slab solves emit the same four advisories as
`fixture.json`'s `("slab", "coarse")` cell:

> dielectric 'diel' on x: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.

> dielectric 'diel' on y: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.

> dielectric 'diel' on z: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.

> all dielectric(s) ['diel'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)

The `thru` reference run is preflight-clean, matching `fixture.json`'s `("thru", "coarse")`
cell. Every Python warning of every solve is stored verbatim and deduplicated under each
run's `warnings` key.

**What this witness does NOT bound.** Both routes divide by a reference run's net flux, so a
reflection at the far absorber biases `F_ref` and the flux lane's `F_ref[i]` by the same
factor and cancels in the comparison. A travelling backward wave carries the same power at
every plane, so the `F_ref(out)/F_ref(in) = 1` check above cannot see it either. The absorber
is bounded by the battery's own thru gates, not here. Likewise, a defect in `flux_spectrum`
that scales every plane equally is invisible to this test by construction.

**Report-only, from committed artifacts and no extra solve.** The same interior fluxes place
the `normalize=False` (modal V/I) lane's non-passivity where it belongs. At the coarse rung
`fixture.json`'s `slab|coarse|false` cell has column-0 closure residual 1.759e-02, while the
interior monitors put the physical power imbalance at 6.887e-05 —
`max_f |closure_S(false) − closure_M| = 1.756e-02`, 800x the flux lane's disagreement and
still inside 0.02. The `thru|coarse|false` cell shows 1.825e-02 on an EMPTY guide, where no
DUT can absorb anything, which corroborates the reading: the V/I lane's ~1.8 % is extractor
error, not lost power. The specific mechanism — a Yee `Z_TE` magnitude error — is INFERRED here,
not instrumented. What supports it: the extractor's own docstring records ~3 % `Z_TE` error on
`normalize=False` S11 at `dx/λ = 0.07`, and the excess falls as `dx²` across the battery's rungs
(1.8253e-02, 4.0817e-03, 9.8341e-04 at dx = 2.54, 1.27, 0.635 mm; ratios 4.47 and 4.15). The
missing step is a direct comparison of the lane's modal `Z_TE` against the analytic value at the
coarse rung. Tracked in issue #873. Not gated — it was not pre-declared, and the plan's WP3
comparison is against the flux lane.

**Provenance.** `provenance.run_lane` reads `"local"` — this measurement ran on a CPU
developer box, not on VESSL. Unlike `fixture.json`, that does not make it
non-claims-bearing: the whole measurement is 13.5 s, so
`test_live_closure_routes_agree_within_the_column_power_tolerance` re-runs it on every
fast-lane invocation and re-asserts the gate against physics. The artifact is a replay
convenience and a record of the intermediates, not the only evidence.

**Schema.** `schema` / `schema_version`, `declaration` (gate, gate source, both route
formulas, monitor positions in metres and in coarse cells, the independence scope sentence,
`settling_db_max`, `non_vacuity_min_s11`), `rung` / `dx_m` / `dut` / `reference_dut` /
`num_periods` / `band_centre_bin` / `freqs_hz`, `flux_lane` (`preflight`, `warnings`,
`settling_db`, `reference_planes_m`, `s_params` in the `fixture.json` `[re, im]` form,
`wall_time_s`), `device_run` and `reference_run` (`dut`, `dx_m`, `n_steps`, `grid_shape`,
`cpml_layers`, `preflight`, `warnings`, `settling_db`, `flux`, `flux_exact_f64`,
`wall_time_s`), the per-bin dumps `s11_mag2_per_bin` / `s21_mag2_per_bin` /
`closure_s_per_bin` / `closure_m_per_bin` / `abs_diff_per_bin`, the headline
`max_abs_diff` / `worst_bin_index` / `worst_bin_hz` / `closure_*_at_worst` /
`closure_*_at_centre` / `abs_diff_at_centre` / `non_vacuity_max_s11`, `verdict`,
`wall_time_s`, `generated_at`, `provenance` and `driver`. The replay test recomputes both
routes from the raw per-bin intermediates and compares with the stored headline, so a
hand-edited summary cannot pass.
