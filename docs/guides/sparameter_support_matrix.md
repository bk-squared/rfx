# S-Parameter Calculation Support Matrix

This matrix answers three separate questions for each public primitive:

1. Does it define a port impedance and reference plane?
2. Which API computes its S-parameters?
3. What RF evidence and restrictions apply to that calculation now?

The machine-readable companion is
`docs/guides/sparameter_support_matrix.json`. Evidence levels are defined in
`docs/guides/physics_validation_evidence_rule.md`; raw voltage/current replay is
defined in `docs/guides/sparameter_dump_replay.md`.

Status terms on this page are **supported**, **limited**, **experimental**,
**not documented**, and **unsupported** as defined in
`docs/guides/support_matrix.md`.

## Result and metric convention

Full matrices use
`S[receiver_port, driven_port, frequency_index]` with shape
`(n_ports, n_ports, n_freqs)`. Frequencies are in Hz with shape `(n_freqs,)`.
Unless a metric name or sentence explicitly says dB, `*_mag_abs_diff` is an
absolute difference in linear magnitude.

Passivity, reciprocity, replay, and warning checks are necessary diagnostics;
none alone proves calibrated RF accuracy. Apply only the evidence listed for
the selected port, mode, mesh, geometry, and frequency range.

Automatic guards also differ by calculator. MSL and waveguide matrices receive
the full column-power check. Lumped/wire results are checked only for non-finite
or excessive individual `|S|` values, not column power. The public
`compute_coaxial_line_reflection(...)` result has no shared automatic passivity
guard. An absent warning therefore cannot be compared across port families.
(`compute_coaxial_two_port(...)`, below, does route through the shared guard.)

## API summary

| Primitive | Calculation API | Result | Current status |
|---|---|---|---|
| Lumped `add_port(..., extent=None)` | `run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | **limited** — one-cell impedance model; E2/E3/E4-partial evidence |
| Lumped `add_port(..., extent=None)` | `forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `.freqs` (S11 vectors) | **limited** — uniform, single-device AD path; inherits the lumped-port RF limits |
| Wire `add_port(..., extent=...)` | `run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | **limited** — multi-cell discrete feed across `extent`; magnitude evidence is stronger than absolute calibration evidence; nonuniform use is experimental |
| Wire `add_port(..., extent=...)` | `forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `.freqs` (S11 vectors) | **limited** — uniform, single-device AD path |
| `add_msl_port(...)` | `compute_msl_s_matrix(...)` | `MSLSMatrixResult.S`, `.freqs`, `.Z0`, `.beta`, `.port_names`, `.reliable` | **limited** — E5-narrow / eigenmode-blocked; external notch agreement is characterized, not tight; `eps_override` AD checked against an f64 referee on the band-mean `\|S21\|^2` objective (rel_err 0.0026 at the gate's num_periods=20 fixture, threshold 0.03; issue #530, superseding the pre-#530 `sum\|S_ij\|^2` objective and its 0.0331/0.10 figures); nonuniform mode is experimental |
| `add_waveguide_port(...)` | `compute_waveguide_s_matrix(...)` | `WaveguideSMatrixResult.s_params`, `.freqs`, `.port_names`, `.port_directions`, `.reference_planes` | **limited** — broad magnitude evidence for documented uniform single-mode rectangular guides; phase and junction evidence are narrower; nonuniform configurations outside the passed Palace `normalize=flux` WR-90 cases remain experimental |
| `add_waveguide_port(...)` | `run(...)` | `Result.waveguide_sparams[name]` | **limited diagnostic** — per-port output, not the full multi-port matrix API |
| `add_coaxial_port(...)` | `compute_coaxial_line_reflection(...)` | `CoaxialLineReflectionResult` | **limited** — exactly one `face="top"` port; broad-E5 analytic and broad-E4 MEEP evidence for the documented TEM-line result |
| `add_coaxial_port(...)` | `compute_coaxial_s_matrix(...)` | `CoaxialSMatrixResult` | **experimental and deprecated** — older single-plane V/I path; can produce non-physical `\|S11\| > 1` for a lossless short |
| `add_coaxial_port(...)` | `compute_coaxial_two_port(...)` | `CoaxialTwoPortResult` | **validated with scope** (issue #489, PI decision 2026-08-06) — two-drive through-line 2-port solve on one coax geometry family; bracketed by an external openEMS referee (`validation/crossval/21_coax_two_port_referee.py`, VESSL run-3 `369367251629` + VESSL `369367252220`) on `\|S21\|` and, via the port's own measured `beta`, phase, plus a mesh-refinement convergence witness (VESSL `369367251845`, `p ~= 1.5`) and a `GRAD_SAFE` `eps_scale` AD gate; every DUT it can currently gate against is still azimuthally symmetric (TM0n only) — coax<->planar transitions are the separate `compute_coax_msl_transition(...)` lane (own row below, own status, unaffected by this promotion) |
| `add_coaxial_port(...)` + `add_msl_port(...)` | `compute_coax_msl_transition(...)` | `CoaxMSLTransitionResult` | **experimental, diagnostic-only** (issue #489 leg 4) — coax-to-microstrip transition, two-drive; two committed fixtures plus a settled VESSL run (`369367252283`, `n_steps=135000`) of attempt 2's own fixture. gamma-vs-beta is CONFIRMED (three in-band checkpoints, the last at full settling); reciprocity (91.4% worst deviation) and passivity are now MEASURED/ATTRIBUTED at full settling (the earlier passivity-guard trip was a truncation artifact); the MSL-driven column power is a SHARPENED open question (~99% missing at 6/8 GHz, ~20% at 10 GHz) with a named discriminating check, and whether the junction's own physical reflection also contributes is genuinely unresolved — see the section below — do not treat as a validated transition |
| `add_port(...)` + `add_msl_port(...)` | `compute_mixed_s_matrix(...)` | `MixedSMatrixResult` | **experimental**, diagnostic, not in the validated set — off-diagonal magnitudes from Poynting flux; internal reciprocity witness 9.0% (flux channel) vs 55% (wave channel) on the probe-fed MSL fixture; absolute \|S\| is NOT validated (no external-solver referee has been run); with `enforce_passivity=True` (default) the returned diagonal is a joint SVD-projected value — read `S_raw`/`passivity_correction` for what was measured |
| `add_floquet_port(...)` | no documented high-level S-parameter API | none | **experimental** — broadside diagnostic helpers only; no calibrated Floquet-port result |
| Sources, TFSF, probes, DFT planes, flux monitors | none | field, resonance, or flux results | **not a port** — no impedance or S-matrix reference plane |

## Lumped port

**Use:** a one-cell feed or load with a positive scalar reference impedance.
It is not a transmission-line mode. R/L/C/RLC entries below are synthetic
extractor checks, not capabilities of `Simulation.add_port(...)`; circuit
elements use the separate `add_lumped_rlc(...)` API and do not themselves
define an S-parameter port.

**RF evidence (E2/E3/E4-partial):**

- Closed-form open, short, matched, resistive, capacitor, inductor, series-RLC,
  and parallel-RLC extractor checks have `max_abs_diff 7.91e-8` against a
  `2.20e-6` tolerance.
- A real two-port V/I replay covers 9 frequencies and 2 ports with
  `max_abs_diff 1.13e-7` against `9.84e-7`.
- A three-case uniform-grid replay/passivity/reciprocity check has maximum replay
  difference `1.58e-7`, maximum column power `0.971`, and maximum reciprocity
  difference `3.02e-7`.
- The rfx/OpenEMS PEC-box magnitude checks cover three port-position cases. The
  largest per-case linear-magnitude differences are `0.11835` maximum and
  `0.06466` mean. These cases do not cover a broad matched/open/short/load set.

**Restrictions:**

- `forward(port_s11_freqs=...)` is uniform and single-device only.
- Nonuniform lumped-port S-parameter extraction is unsupported.
- TFSF, waveguide ports, and mixed port families cannot share this S-parameter
  calculation.
- Analytic extractor and V/I replay checks validate algebra and reproducibility;
  they do not establish a generally calibrated lumped-port result.

Relevant implementations and tests include `tests/unit/sparams/test_sparam.py`,
`tests/unit/sparams/test_port_dump_replay.py`,
`scripts/diagnostics/report_lumped_analytic_oracles.py`, and
`scripts/diagnostics/build_lumped_openems_sweep_comparison.py`.

## Wire port

**Use:** a one-cell transverse probe/wire feed. Use `add_msl_port(...)` when the
intended model is a distributed microstrip line.

**RF evidence (E2/E3/E4-partial):**

- A real midpoint-cell two-port V/I replay covers 7 frequencies and 2 ports with
  `max_abs_diff 7.82e-8` against `9.80e-7`.
- A three-case uniform-grid replay/passivity/reciprocity check has maximum replay
  difference `8.20e-8`, maximum column power `0.979`, and maximum reciprocity
  difference `1.24e-6`.
- The patch/OpenEMS comparison over 1.5--3.4 GHz has
  `max_mag_abs_diff 0.05318` and `mean_mag_abs_diff 0.02750`. Phase is not
  gated for this comparison because the two solvers' reference planes have
  not been aligned for the wire/patch configuration -- not because their
  time conventions differ. Both tools accumulate `exp(-j*2*pi*f*t)` (rfx
  `update_dft_plane_probe`; openEMS `DFT_time2freq`), differing only by
  real positive normalization factors that cannot change a phase, so there
  is no `e^{+-j*omega*t}` mismatch to resolve here (verified at source,
  issue #490 Lane 2). Aligning the planes for this configuration is
  untried; see the microstrip-line section for the one configuration
  where it has been done.
- A three-case OpenEMS mesh/length comparison covers `dx` of 1--2 mm, wire
  lengths of 4--8 mm, and 0.8--1.8 GHz, with
  `max_mag_abs_diff_across_cases 0.05212`.

**Restrictions:**

- Absolute S-matrix calibration remains limited by the one-cell feed and current
  per-cell impedance convention. Do not treat the replay as modal-port
  calibration.
- The nonuniform wire calculation is experimental; regression and AD coverage
  are not external RF validation.
- `forward(port_s11_freqs=...)` is uniform and single-device only.

Relevant checks include `validation/crossval/05_patch_antenna.py`,
`tests/unit/sparams/test_twoport_wire_port.py`, `tests/unit/sparams/test_wire_port_sparams_forward.py`, and
`scripts/diagnostics/report_wire_replay_sweep.py`.

## Microstrip-line port

**API:** `compute_msl_s_matrix(...)` with the laplace/quasi-TEM model.

**RF evidence (E5-narrow / eigenmode-blocked):**

- The uniform thru-line check uses `|S21|` in `(0.90, 1.05)` and
  `Re(Z0)` in `(40, 65) ohm` for the cited `dx=80 um` setup.
- The analytic quarter-wave-notch case
  (`validation/crossval/06b_msl_notch_filter_uniform.py`) ships at
  `dx = 63.5 um = h_sub/4` (issue #723). It ran at `dx = 80 um` through
  2026-08, where the declared `254 um` substrate rasterized to `320 um` and
  the `600 um` trace to `560 um`, so the analytic references were computed on
  a board that mesh did not solve; that run is history, not current evidence.
  The committed 2026-08-27 GPU run log
  (`validation/crossval/_06b_notch_uniform_logs/20260827T131217Z_run.log`)
  reports `1.40%` frequency error against the analytic notch evaluated on the
  realized `635.0 um` trace width, `-43.3 dB` notch depth, and median
  `Re(Z0)=46.5 ohm` (port 0, median over the 100 bins); it passes the listed
  gates of frequency error `<15%`, notch depth `<-10 dB`, and median
  `Re(Z0)` in `(40, 65) ohm`. That `Re(Z0)` sits `-2.9%` from
  Hammerstad-Jensen on the DESIGN board (`600/254 um`, `47.90 ohm`) and
  reproduces the independent `msl_z0_bias_floor_sweep` "aligned h_sub/4"
  point (`46.098 ohm`) to `0.87%`. Read it with that run's own warnings, not
  without them: a standing-wave null flags 9 bins in `[3.6273, 7.0000] GHz`
  as unreliable for the wave split -- which starts at the reported notch --
  `63 of 100` bins were non-passive as extracted (worst `sigma_max = 1.006`),
  and the per-port argmax `Z0` deviations against Hammerstad-Jensen are
  `61.02 ohm` (`msl_0`) and `39.90 ohm` (`msl_1`).
- The `dx = 50 um` OpenEMS notch comparison below is NOT the same board:
  its rfx leg (`scripts/diagnostics/build_msl_notch_rfx_dx50.py`) still
  realizes `h_sub = 300 um` at its own `dx = 50 um` and is deferred, not
  fixed, by #723.
- The committed matched-geometry OpenEMS comparison at `dx=50 um` reports a
  `5.8%` notch-frequency difference, linear `|S21|` mean difference `0.1147`,
  and maximum difference `0.2078` over 2.5--6 GHz. This is a characterized
  external check, not a tight cross-solver match. See
  `tests/fixtures/msl_notch_e4/comparison_summary.json`.
- `scripts/diagnostics/replay_msl_3probe_dump.py`'s independent 3-probe
  replay is SUPERSEDED and does not run against current dumps: it expects
  the retired 3-probe `raw_v123` schema (schema v1) and recomputes S by
  the single-ratio rule the multi-drive solve replaced (issue #507), so it
  cannot be compared against `production_smatrix` on a `schema_version >=
  3` dump (it now raises a clear error instead of a bare `KeyError`). The
  current independent check on the production V/I extraction is
  `scripts/diagnostics/msl_vi_flux_oracle.py`.
- The `eps_override` gradient is checked against an f64 AD-vs-FD referee on
  band-mean `\|S21\|^2` (issue #530; this REPLACES the prior `sum_ij\|S_ij\|^2`
  objective, which was 99.96% a passivity-pinned structural constant — see
  `tests/unit/autodiff/test_msl_ad_fd_converged.py`'s docstring for the full replacement
  rationale). Tracked run log:
  `scripts/diagnostics/msl_ad_band_mean_owner_measurement/owner_runs_20260804.md`
  (both VESSL runs' full measurement tables plus the actual pytest gate's
  own PASS output — the raw logs live only under the primary checkout's
  gitignored `.omx/`, this is the tracked copy). Headline: rel_err `0.0026`
  at `num_periods=20` through the full extraction, on the gate's own fixture
  at its own h=1e-3 (gpu-rtx4090, VESSL 369367251813/369367251827; a
  5-point h-sweep over h in [3e-4, 1e-2] reads
  rel_err 0.0002-0.0146 with a 1.583% FD spread), against a `0.03` gate
  threshold derived via `tests._gate_policy.gate_from_envelope` from the
  sweep's worst point. A planted issue-#483-class defect (`eps_override`
  frozen before tracing, so the traced parameter never reaches the AD tape)
  reads rel_err `1.0000` on the same fixture — the gate reds at 33x the
  threshold, confirming it discriminates a real defect and not only
  comparator noise. This supersedes the pre-#530 objective's `0.0331`/`0.10`
  figures (issue #527, closed; that run: VESSL 369367250775, gate's own
  measured run VESSL 369367250794) and the older pre-#516 `0.000110` figure
  (issues #483/#486): the #507/#511 fixes had shrunk the OLD objective's
  differentiated signal about 50x, exposing the f32 comparator's own
  resolving-power floor as the dominant cause of an intermediate 0.8519
  mismatch. The new objective is NOT immune to the same shape of failure — an
  extractor fix that shrinks `\|S11\|` would shrink this gradient too — but
  MEASURED (not narrated): the level dropped 16x on this fixture (16.00599 to
  0.99787211), cutting the loss's float32 ULP 32x and lifting f32 resolving
  power from 4.45 to 53.8 ULP at the gate's h, and the residue from unity
  (~2.5e-3, order `\|S11\|^2` with `\|S11\|` ~ 0.05 here) is now a physical
  observable rather than a unitarity-violation artifact. That risk is
  CONTAINED — by the f64 comparator's 2.9e6x resolving-power headroom above
  `_MIN_FD_ULP_SPAN` and by the resolving-power floor assert (issue #527's
  fix, unchanged by the objective swap) reporting a comparator failure loudly
  instead of silently — not eliminated (see `tests/unit/autodiff/test_msl_ad_fd_converged.py`
  for the ULP-resolving-power derivation and `tests/_msl_ad_objective.py` for
  the full statement, including what mechanism drives the gradient — a
  reference-plane artifact against the wave split's frozen Hammerstad-Jensen
  `z0_hj`, or genuine beta/reflection physics. **RESOLVED, issue #560,
  2026-08-06**: a decisive probe
  (`scripts/diagnostics/msl_ad_z0_anchor_probe.py`, run log
  `scripts/diagnostics/msl_ad_z0_anchor_probe_run_20260806.md`) swapped the
  frozen analytic `z0_hj` anchor for a frozen per-port FITTED z0 (measured
  at alpha=1, held constant) and re-measured `|g_ad|` on this same fixture
  (CPU/float32): it collapsed `1.602236e-03` (bit-identical across 2
  repeats) -> `6.885110e-05` (headline value from an un-repeated run,
  killed mid-way by a background-task duration limit; the
  bit-identical-2/2 value under a CLI-rounded anchor is `6.884444e-05`,
  agreeing to 4 significant figures). By issue #560's own QUALITATIVE
  criterion ("drops toward the FD-unresolvable floor"): the estimated FD
  signal for this g_b at the gate's h is only ~1.16 ULP of a float32 loss,
  below the 4.449 ULP issue #527 measured for the retired objective's
  comparator and declared untrustworthy — g_b is noise-floor by this
  repo's own established standard. (As a secondary check, this is ~23.3x,
  4.6x past this PR's own pre-declared 5x threshold — NOT a quote from
  #560, whose body contains no such number; see the probe script's
  docstring.)
  The reference-plane artifact (not beta/reflection physics) is the
  dominant channel — this does not change the `rel_err`/threshold numbers
  above, only their physical reading. Separately, anchor B's own loss
  exceeded 1 (a passivity violation, attributed to the raw unprojected
  `eps_override` channel — see the probe's run log), which is evidence the
  fitted anchor is not self-evidently "more correct"; whether
  `compute_msl_s_matrix`'s PRODUCTION wave split should anchor on it is a
  SEPARATE, undecided design question this PR does not settle. The #515 AD smoke
  shares this same objective function (`tests/_msl_ad_objective.py`) so the
  two tests cannot drift apart. The launch fixture derives from registered
  materials on both the FD and AD sides; staticness is regression-locked by
  `tests/unit/ports/test_msl_source_fixture_static.py` (pre-fix `0.126` vs gate `0.03`
  at `num_periods=1`).
- MSL de-embedded phase now has an external referee (issue #490 Lane 2,
  openEMS, VESSL run 369367251705). With both solvers' measurement planes
  placed at the same physical coordinate (rfx's probe-0 plane), the raw
  cross-solver `angle(S21)` difference is `<= 0.304 degrees` across the
  gated 3.0--4.5 GHz band and `<= 1 degree` in 22 of 30 bins over
  0.5--5 GHz. A conjugate (time-convention) mismatch would have produced
  ~102 degrees of disagreement at 5 GHz, where the unwrapped phase is
  `-51 degrees`; the measured disagreement there is `0.131 degrees`. Gated
  per-solver self-consistency -- each solver's own `angle(S21)` against its
  own measured `beta` -- is `0.642 degrees` (openEMS) and `0.121 degrees`
  (rfx) against a 3-degree budget derived from a +-4-cell plane-position
  allowance. Scope: one thru geometry at `dx=50 um`, single drive. Bins
  below ~1 GHz are excluded from the comparison: the line is only 0.03
  guide wavelengths long at 0.5 GHz, where three-point port extraction is
  poorly conditioned and the openEMS side returns `|S21|` up to `1.0087`.
  openEMS's `CalcPort(ref_plane_shift=...)` rotation is not exercised by
  this run -- the effective shift is 0 by construction -- so the plane
  match comes from stencil placement, not from the referral transform.
  Issue #812 P1 (2026-09-01): the per-solver self-consistency figures above
  are an **E1** leg -- both sides come from one field solve, so a coherent
  phase-velocity error cancels and a factor-2 error reads
  `validation/crossval/_issue812_phase_identity/regate_evidence.json::cv20.blindness.audit_construction_e1_max_phase_dev_deg = 0.241`
  degrees against that same 3-degree gate. Two independent-reference gates now run
  alongside it and are wired into the script's own pass/fail: each solver's
  measured `beta` against the Hammerstad-Jensen closed form of the realized
  board (E2, 2.0% tolerance; measured 0.94% rfx / 0.31% openEMS), and the
  **raw** cross-solver `angle(S21)` difference quoted above (E4, 3-degree
  tolerance; measured `0.342 degrees` on the #723 realized-board re-run),
  which was previously reported rather than gated. Do not cite the
  dispersion-corrected residual as the cross-solver number: it subtracts a
  term built from `beta_rfx` and is provably blind to this error class.
  See `validation/crossval/20_msl_phase_referee.py` (manifest entry
  `20_msl_phase_referee`), `tests/crossval/test_msl_phase_referee_header.py`, and
  `docs/design_notes/issue812_phase_identity_predeclaration.md`.

`MSLSMatrixResult.reliable` is available during normal execution and is `None`
during JAX tracing. Under the multi-drive solve (issue #507) a `False` entry at
any port contaminates the entire frequency slice `S[:, :, k]`, so the only safe
per-bin filter is `np.all(reliable, axis=0)`; the per-port index tells you which
probe plane collapsed. Details and the filtering example:
[Low-signal MSL bins](../public/guide/probes-sparams.mdx#low-signal-msl-bins).
A `True` entry is not an accuracy guarantee.

**Restrictions:**

- Nonuniform `mode="laplace"` and `mode="uniform"` have internal settled-S11
  regression coverage but no external nonuniform comparison; treat them as
  experimental.
- `mode="eigenmode"` is unsupported and raises `NotImplementedError`.
- SBP-SAT subgridding, ADI, TFSF, and mixed port families are unsupported for
  this calculation.
- Surface-impedance sheets (`add_thin_conductor(...,
  surface_impedance_f0=...)`) are applied on this lane's device runs (#679):
  every FDTD dispatch goes through `run()`/`forward()`, which realize the
  sheet node-thin via the #677 per-step operator. The trace itself must stay
  a PEC `Box` (an f0 sheet never enters the PEC mask the Ampere-loop current
  and V span anchor on, and the lane raises if no PEC trace is found).
  Combination refusals (dispersive substrate, subpixel/conformal, UPML,
  ADI/subgridded/distributed) fire at the run-lane entry. A sheet lying
  inside a probed span biases the lossless-line N-probe `Z0`/`q` fit (the
  honesty guard may warn) while the V-I S extraction stays valid. The mixed
  (wire + MSL junction) lane still refuses f0 sheets.
- Strong-reflector `|S11|`'s roughly 0.16--0.22 "staircase-Z0 floor" this
  document used to cite is **RETIRED** (issue #487) — it was substantially
  the #511 modal-voltage span and #507 far-port-echo single-ratio assembly
  extractor defects, fixed in PR #516 (`f95240f`), not a mesh property. A
  smaller floor survives whose mechanism is the mismatch between the
  rasterized line's own Z0 and the analytic Hammerstad-Jensen anchor `S` is
  normalized against, tracking `|Gamma_implied| = |(Z0-Z0_HJ)/(Z0+Z0_HJ)|`
  within ~1.3x over 5 of 6 points of the #487 re-sweep
  (`scripts/diagnostics/msl_z0_bias_floor_sweep.py`, committed JSON). No
  single envelope number is published, for two reasons: (1) below
  `|Gamma_implied| ~ 0.006` (the finest aligned sweep point) the sweep
  cannot resolve whether the mechanism still holds — it compares one
  band-mean Z0 against a band-mean `|S11|(f)`, with no per-bin trace to
  exclude a Jensen's-inequality artifact, against a fitted-Z0 estimator
  the library's own honesty guard calls healthy only to +/-10% — so this
  is reported as a resolution limit of the sweep, not a confirmed second
  mechanism (see the script for the full breakdown); (2) even where the
  mechanism does hold, the measured floors are specific to that one thru
  fixture, and generalizing them to a dB promise for arbitrary MSL ports
  would itself be the overclaim the next sentence forbids. Do not
  generalize the matched/thru/notch evidence.
- The auto `n_probe_offset` solves the upstream/downstream clearance interval
  at driver time (midpoint with a reflector, unchanged without one) and warns
  loudly when a short feed cannot satisfy both clearances (#469). Library
  witness probes are excluded from preflight advisories (#470).

## Rectangular-waveguide port

**API:** use `compute_waveguide_s_matrix(...)` for a full matrix. At least two
ports are required. `run()` provides only per-port diagnostics.

**Uniform single-mode magnitude evidence:**

- Analytic Airy checks cover WR-28, WR-62, WR-15, WR-340, and WR-10 dielectric
  slabs. With `normalize="flux"`, maximum per-band linear `|S11|` differences
  are 0.005--0.041 for the cited cases.
- The Palace WR-90 comparison covers empty guide, PEC short, and dielectric slab
  from 8.2--12.4 GHz. Across five compared terms, the maximum and mean
  linear-magnitude differences are `0.0707` and `0.00943`.
  > **STALE — 2026-06-16 numbers; quote them with their date (2026-08-31,
  > issue #812 Phase 0).** They come from
  > `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`,
  > which was committed at `b0322c1` (2026-06-16, PR #181) and never
  > regenerated. Its **provenance is settled**: feeding
  > `git show b0322c1:tests/fixtures/waveguide_broad_e5/cv11_wr90_fresh_stdout.txt`
  > (that stdout as it stood at the artifact's own commit — it was overwritten
  > later, at `20e5533`) to the artifact's own builder
  > (`scripts/diagnostics/build_waveguide_wr90_rectangular_broad_e4_comparison.py`
  > `--reference-column Palace_r_h2`; pure text post-processing, **no FDTD**)
  > rebuilds it field for field — 54 numeric fields, 52 bit-identical, 2
  > differing at ~1 ulp (`5.3e-18`, `1.7e-18`).
  >
  > What is wrong with it is its **age**. The cv11 stdouts committed in that
  > directory today are from 2026-08-28 (`20e5533`, #724/#730), and the same
  > builder on them rebuilds the slab `S11` `max_mag_abs_diff` to `0.0186` /
  > `0.0194` / `0.0193` against the artifact's `0.0707` (3.6x-3.8x better) and
  > the slab `S11` `mean_mag_abs_diff` to `0.007705`-`0.007771` against
  > `0.043976`. The artifact's slab-`S11` rfx magnitude range
  > `[0.0397, 0.5924]` reads `[0.0018, 0.5243--0.5251]` on the current runs,
  > while the Palace reference column is identical throughout — the delta is
  > entirely on the rfx leg, which is what a code change between June and
  > August looks like.
  >
  > **Direction matters: this understates the family.** Every current run is
  > *better*, and `0.0707` is inside the artifact's own `max_mag_abs_tol` of
  > `0.1`. No gate is at risk and no rectangular-waveguide physics verdict is
  > challenged. Two minor warts remain: `source_cv11_stdout` records a `/tmp`
  > path even though a file of that basename is committed beside the artifact,
  > and there is no `setup` block (no commit, `dx`, `NUM_PERIODS` or
  > `CPML_LAYERS`).
  >
  > **Correction:** commit `2d05212`, earlier on 2026-08-31, labelled this leg
  > PROVENANCE-DISPUTED and stated it "does not reproduce from any committed
  > run of its own producing script". That is withdrawn — it rebuilt only from
  > the working-tree revision of those stdouts, never from their content at the
  > artifact's commit. Settling this needed `git show`, not an FDTD run. What
  > remains open is only whether to refresh the artifact, and any refresh must
  > *explain* the 3.7x delta rather than silently re-pin it. Full record: the
  > artifact's own `provenance` key and
  > `docs/design_notes/20260831_cv11_broad_e4_artifact_provenance.md`.
- The validation battery requires empty-guide `max |S11| < 0.02`, maximum column
  power `< 1.02`, symmetric-obstacle mean reciprocity error `< 0.01`, and a
  PEC-short result with `min |S11| >= 0.99` and `max |S11| < 1.03`.
- The cv11 cross-solver gates use a band-mean linear-magnitude difference of
  `0.10` for S11 and `0.07` for S21, a masked band-mean phase difference of
  `60 degrees` where reference magnitude is at least `0.30`, and a maximum
  complex-S difference of `0.30`.

This supports broad magnitude use inside the documented uniform, single-mode
rectangular-guide limits. Phase evidence covers fewer configurations; do not
infer equally broad phase accuracy from the magnitude status for anything
outside the single-dielectric-slab configuration described next.

**Phase and group-delay evidence (issue #490, Lanes 1 and 3):**

- Lane 1 adds an analytic-Airy **phase** envelope over the SAME five WR bands
  and 20 mesh/geometry cases as the magnitude envelope above, re-analyzing the
  npz/manifest pair that produced the committed magnitude envelope (restored
  locally for this analysis -- `.omx` is gitignored and not present in the
  tree; no new FDTD run for that 5-band/20-case sweep -- phase and magnitude
  are two projections of the same measurement). Reference-plane phase is
  corrected per a written convention
  (`e^{+jwt}` time convention; S11 reflection is a round trip through the
  vacuum gap between the reference plane and the slab, `exp(-2j*beta*d)`; S21
  transmission is a single forward pass through the gap on BOTH sides of the
  slab, `exp(-1j*beta*(d_left+d_right))`) documented in
  `scripts/diagnostics/build_waveguide_band_broad_e5_phase_envelope.py`.
  Maximum phase difference across all 20 cases is `11.99 degrees` against a
  measured-envelope gate of `15.0 degrees`. A planted-defect falsifier (the
  wrong S21 reference-plane formula that was sitting, unexercised, in the
  magnitude-only envelope builder before this session -- invisible to a
  magnitude gate because `|exp(i*anything)| == 1`) reds at `179.87 degrees`.
  A fresh domain-size invariance run (WR-340, domain grown `+100 mm`) holds
  the pass verdict (`8.92 -> 7.73 degrees`). See
  `tests/crossval/test_waveguide_broad_e5.py` and
  `tests/fixtures/waveguide_broad_e5/phase_falsifier_and_domain_invariance.json`.
  Caveat: `d_left == d_right` in all five slab fixtures (the slab sits
  centered between symmetric reference planes), so this evidence cannot
  distinguish the S21 formula's `d_left` and `d_right` terms separately --
  only their sum (`d_left + d_right`) is exercised.
- Lane 3 adds a **group-delay** gate on a purpose-built, empty WR-340 fixture
  near cutoff (`f/fc` in `[1.152, 1.498]`, chosen far enough from cutoff to
  keep the CPU run affordable and honest -- true near-cutoff divergence is
  out of scope). `tau_g = -d(unwrap(angle(S21)))/d(omega)` (central
  difference, 9 interior points; 2 band-edge points use a lower-order
  one-sided difference and are reported but not gated) is compared against
  the analytic `L_eff / v_g(f)` oracle (`L_eff = 0.320 m`) run through the
  SAME finite-difference stencil, so the comparator is like-for-like (the
  stencil's own truncation error, 0.0072 ns at the worst point, has the
  opposite sign from the true residual and would otherwise flatter it by
  29%): max interior diff `0.0320 ns` against a pinned-constant gate of
  `0.042 ns` (the raw diff against the exact closed-form derivative,
  `0.0248 ns`, is recorded for context but is not the gated number). A
  record-length settling witness (`num_periods` 60 vs 120; at the time of
  that run `compute_waveguide_s_matrix` had no built-in energy-based
  `settling_db` -- since issue #538 the uniform single-mode waveguide
  lanes carry the same energy-based `settling_db` field and -40 dB
  aggregate warning as the MSL calculator; the nonuniform lane adopted
  the same witness with the #827 waveguide-instance fix, with the
  multimode branch still un-adopted) agrees to `0.000 ns`. A domain-size invariance run (`+100 mm` growth) holds the pass
  verdict (`0.0266 ns`, still under the gate). Three genuinely independent
  falsifiers (skipping the phase-unwrap step, dropping the leading minus
  sign, and using the wrong `L_eff` -- domain length instead of
  reference-plane separation -- in the analytic comparator) all red at
  `>= 1.5 ns`. See `tests/oracle/test_waveguide_group_delay_near_cutoff.py`,
  `tests/oracle/test_waveguide_group_delay_tolerance_envelope.py`, and
  `tests/fixtures/waveguide_group_delay/wr340_near_cutoff_group_delay_envelope.json`.
- Neither lane covers PEC-short, T-junction, nonuniform, or multimode
  configurations for phase or group delay -- those remain uncharacterized.
- MSL de-embedded phase vs an external (openEMS) referee with the convention
  mismatch resolved (issue #490 Lane 2) is explicitly OUT of scope for both
  lanes above; the microstrip-line section below is unchanged.

**Nonuniform transverse mesh:** single-mode `normalize=True` and
`normalize="flux"` run. Analytic Airy fixtures cover grading ratios 1--3,
relative permittivity 2 and 4, and 8.2--12.4 GHz with a maximum
linear-magnitude difference of `0.001081` (was `0.01561` before the
#574 regeneration promoted below). A passed Palace magnitude comparison
covers `normalize="flux"`, a graded-`dy` ratio of 2, and WR-90
empty/PEC-short/dielectric-slab cases over 8.2--12.4 GHz; its maximum and mean
linear-magnitude differences are `0.008529` and `0.000709` (they were `0.07009` and `0.01042` until this lane's absorber was derived from lambda_g rather than hard-coded at 0.33 of it — see #576/#496). This is external RF
evidence for that configuration, not for other profiles, bands, phase,
multimode extraction, or arbitrary junctions. The calculation remains
experimental outside those stated results. `eps_override` and `sigma_override`
differentiation is implemented only with `normalize="flux"`.
`tests/unit/autodiff/test_waveguide_nu_flux_ad.py` finite-difference-checks `eps_override`;
there is no corresponding nonuniform `sigma_override` AD-vs-FD test. Neither
implementation nor gradient regression is RF validation.
Dispatch history (#811, fixed 2026-09-01): until that fix
`compute_waveguide_s_matrix` reached this nonuniform lane only when
`dx_profile` or `dy_profile` was set — a `dz_profile`-only simulation was
silently solved on the uniform grid built from the scalar `dx` while
preflight described the graded mesh. A `dz_profile` now dispatches here
under the same restrictions. No dz-graded accuracy evidence exists yet
(#810): the Palace comparison above is graded-`dy` only, so dz-graded
results are dispatch-correct and unvalidated. The nonuniform lane emits
the same `settling_db` ring-down witness and -40 dB aggregate warning as
the uniform single-mode lane (#827 waveguide instance).

**The nonuniform E5 fixture was regenerated and promoted (#574, closing the
staleness #562/#564 recorded).** It now carries post-#562 geometry AND the
#576/#496 absorber (`cpml_layers` 183 = 0.75 lambda_g, replacing a hardcoded 24
= 0.099), measured on GPU over all 16 configs:

| | committed (pre-#576) | promoted |
|---|---|---|
| `max_mag_abs_diff_across_cases` | 0.015609 | **0.001081** (14.4x) |
| `mean_max_mag_abs_diff_across_cases` | 0.012636 | **0.000644** |
| gate `max_mag_abs_tol` | 0.05 (flat, underived) | **0.002** (derived) |

Every one of the 16 improved, by 11.9x--35.7x. The gate is no longer a flat
number: it is `gate_from_envelope(0.001081, quantum=1000)` through the repo-wide
multiplier (`tests/_gate_policy.py`), and the old 0.05 had been sitting 46x
above the residual it was supposed to bound. The measured envelope is
additionally capped by a literal pinned OUTSIDE the artifact, because a gate
derived from a number living inside the artifact it guards would let a degraded
regeneration re-derive its own looser gate; that ceiling is blind below a 1.203x
degradation, measured by mutation in
`tests/crossval/test_waveguide_nu_broad_e5_envelope_gates.py`.

The improvement is not a backend artifact: the superseded envelope was itself
produced on GPU, same script and same 16 geometries, the only difference being
the absorber, and the two backends agree to 0.6% relative on the excess at the
committed configuration.

Settling witness (mandatory for a fixed-`num_periods` open-CPML claim, and a
named step of #574 after #576 recorded it as not-run): at the promoted absorber,
doubling the record window 60 -> 120 periods moves `max|S11|` by 2.8e-05
(PEC short) and 7.8e-07 (slab), with column power passive to 9.3e-05 — the
window does not bind. At the superseded 24-cell absorber the same doubling moved
the PEC short by 1.46e-02, 520x more, which is why #576 calls absorber depth and
record length co-conditions and why the witness is only meaningful when read at
the absorber actually promoted.

The earlier grading-ratio-independence reading stands as a **hypothesis, not a
demonstrated cause**: #562 changed the node count and the centre-to-node
coordinate convention in the same commit, and no ablation separates them.

The E4 comparison fixture is unchanged and still cannot be regenerated in this
workspace — its producer needs the gitignored Palace dataset at
`REPO.parent/microwave-energy/...`, which is present in neither workspace. It
stays replayable because its reference values are embedded (#574 scope item 3).

**Setup restrictions:**

- Prefer `normalize="flux"`; it uses a matched reference run for Poynting-flux
  normalization and modal V/I phase. It costs `2 * n_ports` FDTD runs.
- The cited five-band fixtures use 24 CPML cells and band-specific `dx` values
  from 25 um to 1.5 mm. Every case has at least 60 cells per vacuum wavelength
  at the highest sampled frequency. This is not a 60-cell guarantee inside the
  dielectric: the coarsest WR-340, `eps_r=4` case has about 30 cells per bulk
  dielectric wavelength.
- Port, reference-plane, and discontinuity distances are band-specific in the
  fixtures. Do not infer a one-guided-wavelength rule from them; preserve those
  coordinates or establish mesh, domain, and port-placement convergence for a
  different geometry.
- Choose slab length and frequency samples away from Airy reflection nulls;
  otherwise relative error is dominated by the numerical noise floor.
- Choose `dx` so the slab length is an integer number of cells; staircase
  quantization directly perturbs the round-trip phase.
- Draw interior PEC obstacles (irises, septa, posts) with their interior faces
  on **cell midpoints**, keep the metal depth an exact number of cells, and
  assert the realized footprint. `Box` rasterizes half-open `[lo, hi)` over node
  coordinates, so a box drawn between two node planes occupies one cell fewer
  than drawn, asymmetrically at the `hi` face. Two facing fins drawn to leave a
  nominal opening `d` therefore leave an electrical opening of `d + dx` **or
  `d + 2*dx`, and which one is not predictable from the nominal dimensions**.
  The pair's two interior faces are different corner types: the lo fin's is a
  `hi` corner, which half-openness always drops, so it always retreats one
  cell; the hi fin's is a `lo` corner, which is kept unless float32 rounding
  puts the node just below it. One retreat gives `d + dx` with the opening
  **asymmetric** (centre `dx/2` low); two retreats give `d + 2*dx`, **centred**.
  Measured on WR-90 at both a/30 and a/60: 7.620 mm and 18.288 mm give
  `d + dx` off-centre, 12.192 mm gives `d + 2*dx` centred. **Transverse** to
  the propagation direction the electrical dimension is the span between the
  innermost zeroed planes, `(n_open + 1) * dx` — the measure that reproduces
  `a = cells * dx` exactly, and an independent refit of 16 committed
  single-iris configurations across two meshes pins the realized aperture to
  within 1/20 of a cell of it (a stage-S3 / issue #499 review observation; no
  committed record carries the refit yet — a caution-grade number, like the
  longitudinal one below). **That identity does not carry into the
  propagation direction:** an obstacle's electrical *thickness* is set by field
  interaction with the discontinuity rather than by a cutoff, is measured to
  fall between `t_cells * dx` and `(t_cells - 1) * dx` so neither integer rule
  holds, and is not settled — treat it as an unknown of order half a cell and
  fold the sensitivity into the reported envelope instead of picking a rule.
  (This measured *effective* thickness is a different quantity from a cascade
  comparator's electrical-length bookkeeping — issue #499's comparator draws
  `t_c = round(t/dx) + 1` so `(t_c - 1)*dx` conserves total electrical
  length; that choice answers a different question and is not contradicted
  here.)
  Offsetting each
  interior face half a cell the wrong way retreats both faces by construction
  rather than by luck, giving `d + 2*dx` deterministically at every aperture;
  that is the drawing case 18's blocked revision used. In
  the WR-90 single-iris lane this inflated the `|S11|` difference against an
  analytic mode-matching oracle by 4-6x (0.0193 to 0.1262 at `d = 7.620 mm`,
  a/30). Because the error scales with `dx` it mimics first-order convergence,
  and on a resonant structure it shifts the passband instead of widening a
  magnitude tolerance. Midpoint corners are rounding-independent, and with
  `(cells - d_cells)` even the realized opening equals the nominal one exactly;
  at odd parity a symmetric opening of that width is not representable on the
  grid and costs one cell **more** however it is drawn. Odd parity is a fork
  rather than a dead end — change `dx`/the aperture so the parity works, or
  place the fins asymmetrically on purpose and accept a recorded half-cell
  offset instead of rounding the aperture (the quantity that sets the cutoff)
  to the wrong parity. Neither is recommended here, because the cost of the
  offset has not been measured; what is required is that the offset be recorded
  and representable by the comparator, since an off-centre aperture compared
  against a centred oracle silently becomes comparator error. See the `Box`
  docstring for the
  arithmetic and `run_point` in
  `validation/crossval/18_wr90_iris_modematch.py` for the assert pattern.
- Size the absorber from the guide wavelength at the **lowest** measured
  frequency, where `lambda_g` is longest and the `cpml_layers=16` default is
  weakest. `compute_waveguide_s_matrix` documents `>= 0.5 * lambda_g` and now
  emits an advisory when the configured stack is thinner. That floor is a
  minimum, not a target: at WR-90 band edge (8.2 GHz, `lambda_g` about 61 mm)
  a 0.30 `lambda_g` stack left residual `|S11|` ripple 0.0706, 0.50 `lambda_g`
  left 0.0366, and 0.75 `lambda_g` left 0.0093, so a 0.5 `lambda_g` absorber
  can still set the accuracy envelope instead of discretization. Case 18
  derives its depth as `ceil(0.75 * lambda_g(band edge) / dx)`; the validated
  fixtures use 20-24 cells rather than the default 16.
- Multimode `normalize=True` is unsupported.
- Branch, T-junction, and septum calculations require per-port matched straight-
  guide references and far-port placement. Compact arbitrary junctions are not
  covered by the broad uniform-guide magnitude result.

## Coaxial port

Use `compute_coaxial_line_reflection(...)` for the documented TEM-line result.
The simulation must contain exactly one coaxial port, it must use `face="top"`,
and no other port family may be registered. It also requires `mode="3d"`,
`solver="yee"`, `precision="float32"`, `stencil_order=2`, a uniform grid, and
`boundary="cpml"` with `cpml_layers > 0`. Other settings raise before grid
construction. Both z faces must have positive CPML thickness, the method must
use its default `cpml_axes="z"`, all six `BoundarySpec` face tokens must be
`cpml`, and periodic boundary axes are unsupported.
The calculator constructs the line, TEM source, DFT planes, and termination.
Do not register separate geometry, thin conductors, lumped RLC elements,
probes, field monitors, NTFF boxes, or `add_coaxial_*` termination helpers.
The registered port contributes x/y, `face`, radii, and waveform. Its z
coordinate, `pin_length`, and `impedance` do not set the internally derived
line layout or loads; use `feed_impedance=` and use `dut_impedance=` only for
`termination="matched"`. `probe_count` must be an integer of at least three,
and all requested planes must fit between the DUT and source. Increase the z
domain or reduce the count, start, or spacing if the method reports that they
do not fit; it does not silently use fewer planes.

**RF evidence (broad-E5 analytic, broad-E4 external):**

- The analytic check covers 4--12 GHz, short/open/matched/resistive 25 and
  100 ohm terminations, characteristic impedances 48.6 and 63 ohm, and a mesh
  sweep. For method-gated cases, maximum `|Gamma|` deviation is `0.0372` against
  a `0.05` tolerance and maximum recurrence residual is `0.00588` against
  `0.03`.
- Use about four or more annulus cells; the committed gate requires at least
  3.5. Coarser cases are reported as under-resolved.
- The matched-load fixture reaches `|Gamma|` deviation `0.0929` because of the
  single-cell annular resistor and is reported separately rather than used as a
  method gate.
- The MEEP short/open comparison over 4--12 GHz has maximum and mean
  linear-magnitude differences `0.0628` and `0.0235`.
- End-to-end differentiation is checked through the `eps_scale` dielectric
  channel; the cited AD-versus-finite-difference discrepancy is `2.6%`.

See `tests/fixtures/coax_broad_e5/`, `tests/fixtures/coax_broad_e4/`, and
`tests/unit/autodiff/test_coax_end_to_end_ad.py`.

This API is not a general multi-port coaxial-network solver and does not cover
arbitrary launches, mixed port families, nonuniform meshes, TFSF, Floquet, or
SBP-SAT. PEC, UPML, zero-layer CPML, ADI, two-dimensional, and fourth-order
configurations are also unsupported. Mixed precision is unsupported.
Boundary specifications without positive CPML on both z faces, non-z
`cpml_axes` selections, mixed boundary-face tokens, and periodic axes are
unsupported. `run()` and `forward()` reject high-level coaxial S-parameter
requests.
The older `compute_coaxial_s_matrix(...)` path is deprecated and experimental.

`compute_coaxial_two_port(...)` (issue #489 stage 2) extends the same
transmission-line method to two ports: it builds a single through line with a
matched annular-resistor feed near each z end, drives each end's own TEM TFSF
source in turn (two FDTD runs), and assembles the S-matrix via a two-drive
solve that does not assume the non-driven port sees zero incident wave. **This
method is VALIDATED WITH SCOPE** (issue #489, PI decision 2026-08-06): an
external openEMS referee (`validation/crossval/21_coax_two_port_referee.py`,
VESSL run-3 `369367251629` and the first default-scale green promoted-lane
run VESSL `369367252220`) brackets — it does not judge — this method's own
`|S21|` on the through-line class, and, via the port's own measured `beta`
(not an idealized analytic one), its phase; a mesh-refinement convergence
witness (VESSL `369367251845`) moved the measured/analytic `beta` ratio from
`1.1208` to `1.0662` (annulus `3.79` -> `5.68` cells, implied convergence
order `p ~= 1.5`, two-point, from a single 1.5x step); the `eps_scale` AD
channel below is `GRAD_SAFE`;
issue #812 P1 (2026-09-01) adds the leg that phase bracketing was missing --
the referee's phase-vs-own-`beta` witness is **E1** (a coherent
phase-velocity error cancels in it exactly), so the port's measured `beta`
and the group delay taken from `S21` alone are now also gated against the
**exact continuum coax TEM** values inside a mesh-dependent staircase
envelope derived from the refinement law above (bound `0.157` at the
registered mesh and `0.086` at the 1.5x refinement, against measured
`0.121` and `0.066`); note this leg is **E2, not E4** -- the referee's
Stage B reads no rfx S-parameters, so no cross-solver phase comparison
exists for this case. Round-2 review (2026-09-01) extended that finding to
the whole referee: it imports no rfx module and reads no rfx fixture, so
**no leg inside `21_coax_two_port_referee` is E4** and its manifest
`evidence_levels` is now `["E1", "E2"]`; the E4 in this method's evidence
chain is **this paragraph's own comparison** -- rfx's `compute_coaxial_two_port`
against the referee's committed openEMS output -- and it is owned here, not
by the referee's registration. This method's own reciprocity/`cond(A)`
are measured below. **Scope that remains outside this evidence**: every DUT
it can currently gate against (none, a matched feed, or a coaxial dielectric
plug) is azimuthally symmetric and excites only TM0n modes, while the
transition discontinuities issue #489 targets excite TE11 (cutoff 25.17 GHz
on the validated SMA line, evanescently surviving to the first probe plane);
nor does this evidence generalize beyond this single coax geometry family.
The measured single-run envelope (60 mm /
40 GHz fixture, 4-12 GHz): `|S21|`,`|S12|` 0.74-0.96, `|S11|`,`|S22|`
`<= 0.051` (measured max `0.0502` at 12 GHz,
`tests/unit/sparams/test_coax_two_port_smatrix.py:699`; the committed gate itself is the
wider inherited 1-port envelope `<= 0.08`), reciprocity within `0.3%`
magnitude / `0.21` degree phase, `cond(A) <= 1.11`. See
`tests/unit/sparams/test_coax_two_port_smatrix.py` for the full measured envelope and its
provenance.

Coax<->planar transitions are a SEPARATE lane, `compute_coax_msl_transition(...)`
(below), and are unaffected by this promotion — that lane's own status is
tracked independently in its own section below, not restated here.

`compute_coaxial_two_port(...)` now has the same `eps_scale` differentiable
channel as the 1-port method above (issue #489 leg 3): the gate compares
float32 AD (as shipped) against a float64-loss central finite difference —
the FDTD fields stay float32 (same baseline as the MSL f64 referee), but a
scoped `enable_x64()` around the forward call makes the DFT-accumulator-and-
downstream math run at float64 (`rfx/probes/probes.py` keys the accumulator
dtype off `jax.config.x64_enabled`, independent of the `precision="float32"`
pin). Measured rel_err 0.51% at `h=2e-3` against a 2% gate; owner-platform
(GPU) re-measurement pending. See `tests/unit/autodiff/test_coax_two_port_ad.py` and
`scripts/coax_two_port_ad_fd_f64_referee.py`.

Numerical line attenuation at the validated 3.79-cell annulus gives `|S21|`
0.96 -> 0.74 on the 60 mm / 40 GHz fixture over 4-12 GHz even though `|S11|`
stays at or below 0.05 throughout. A post-hoc consistency check (run after
this measurement; the estimator was chosen after seeing an own/other-drive
split described below, not predeclared) found the `|S21|` deficit equals
what the extractor's own matrix-pencil-fitted `Re(gamma)` predicts over the
reported port separation (within about 2% at every measured frequency,
`|S21|` against `exp(-Re(gamma)*L12)`). That check is sensitive to
SCALE-type deficits (amplitude mis-normalization, mode conversion, a bad
wave split — `gamma` is fit from shape, not scale) and is structurally
BLIND to reference-plane referral errors (a referral error scales the wave
amplitude and `L12` by the same factor, so the compensation cancels it
exactly — verified to five decimal places at +30 cells of injected error).
The under-resolved-annulus recipe (at least 4 cells) that this repo already
documents for reflection accuracy applies to transmission magnitude here
too, even when `status` reports `"passed"` (`annulus_cells`
only gates below 3.5).

## Floquet/Bloch and non-port observables

`add_floquet_port(...)` has broadside modal bookkeeping, field-dump replay, and
analytic empty-space/slab diagnostics. The internal differences: the
empty-space analytic-null check has `max_abs_diff 0.06067` (against `0.07`) and
`mean_abs_diff 0.05306` (against `0.06`); the homogeneous-slab analytic oracle
covers 8 cases x 3 frequencies with maximum power-balance error `4.44e-16`; the
rfx-FDTD homogeneous-slab magnitude check has `max_mag_abs_diff 0.06212`
(against `0.07`) and `mean_mag_abs_diff 0.03209` (against `0.04`) over 3
frequencies; the synthetic specular-TE check has S11 difference `2.89e-7`, S21
difference `6.37e-7`, and power-balance error `1.02e-6`; the real-FDTD Ex/Hy
DFT-plane replay has maximum S difference `2.23e-7`. These are not RCWA or
independent full-wave validation, and there is no documented high-level Floquet
S-parameter API. Treat the result as experimental.

`add_source(...)`, polarized sources, TFSF, point probes, DFT plane probes, and
flux monitors do not define a port impedance or S-matrix reference. Validate
their field, resonance, far-field, or flux observable directly; do not document
them as port substitutes.

## Rejection and preflight behavior

Explicit S-parameter requests outside the matching API must fail rather than
returning `None` or silently omitting a feature:

- `run(compute_s_params=True)` accepts only lumped/wire `add_port(...)`.
- `compute_msl_s_matrix(...)` accepts only MSL-port simulations.
- `compute_waveguide_s_matrix(...)` accepts only waveguide-port simulations.
- `compute_coaxial_line_reflection(...)` accepts only exactly one `face="top"`
  coaxial port with no mixed port family, within its documented line setup.
- `forward(port_s11_freqs=...)` accepts only uniform, single-device lumped/wire
  port setups.

Check routing before an expensive run:

```python
sim.preflight_sparameters(calculator="run")
sim.preflight_sparameters(calculator="forward")
sim.preflight_sparameters(calculator="msl")
sim.preflight_sparameters(calculator="waveguide")
```

The returned issues check API compatibility, not mesh/time convergence or RF
accuracy. Use `strict=True` when the setup should fail on any reported issue.

## Mixed port families — EXPERIMENTAL, not in the validated set

The per-extractor restrictions above are accurate: `compute_msl_s_matrix`, the
lumped/wire scan driver, and the coaxial calculators each reject foreign port
families. They should not be read as "rfx cannot compute across port families
at all" — a separate, explicitly experimental method exists:

```python
res = sim.compute_mixed_s_matrix(freqs=freqs)   # lumped/wire + MSL, uniform mesh
```

`compute_mixed_s_matrix` drives each port in turn, takes its diagonals from the
per-family validated extractors, and takes off-diagonal **magnitudes** from
Poynting flux (`magnitude_channel="flux"`, the default), normalizing incident
power as `P_net / (1 - |S_jj|^2)`. No characteristic-impedance anchor enters the
magnitude.

What this is and is not:

| aspect | status |
|---|---|
| off-diagonal magnitude | experimental; internal reciprocity witness 9.0% on the probe-fed MSL fixture at `dx=63.5um`, settling `-101`/`-100` dB (55% on the wave channel) |
| absolute magnitude | **NOT validated** — no external-solver referee has been run |
| off-diagonal phase | provisional (mixes two reference-plane conventions) |
| per-column power | an algebraic identity on the flux channel — **not** a passivity check |

Because the flux normalization makes column power identically 1 whenever the
arriving power equals the net launched power, a green passivity result on this
lane carries no information. Reciprocity is the only independent internal
witness, and the method warns when it exceeds `reciprocity_tol` (default
`0.06`, set below the 9% reference-fixture residual so that fixture warns
rather than passing silently).

Everything outside the first supported pair is rejected loudly: waveguide,
Floquet, coaxial and TFSF registrations, bare sources and 0-ohm ports, mixed
lumped+wire sets, `reference_plane_cells`, non-uniform meshes, SBP-SAT, and ADI.
The flux channel additionally requires a PEC `z_lo` ground and vertical
(`component="ez"`) lumped/wire ports, since the per-port flux box omits its
bottom face and treats the port extent as a height.

Two further cautions on this lane, both measured rather than anticipated:

- **Neither diagonal is verified.** The wire port-cell V·I accounting was measured
  undercounting delivered power ~3× against an independent Poynting referee, and on an
  end-fed fixture the MSL probe plane's local `V/I` was ~591 Ω and strongly reactive
  while the reported `|S22|` was 0.03. Those cannot both be right, and which one is
  wrong is open.
- **The returned diagonal is not always the measured one.** With
  `enforce_passivity=True` (the default), `_project_passive` is a joint SVD clip: when
  any entry is non-passive it rewrites others as a side effect. On the committed test
  fixture the shipped MSL diagonal came out ~4× its unprojected value. Read `S_raw` and
  `passivity_correction` for what was actually measured.

## Coax<->MSL transition — EXPERIMENTAL, diagnostic-only (issue #489 leg 4)

```python
res = sim.compute_coax_msl_transition(junction_x=..., eps_r_sub=...)
```

Generalizes the mixed-family idea above to a coax<->microstrip launch instead
of lumped/wire<->MSL, but by a DIFFERENT route: rather than extending
`compute_mixed_s_matrix`'s own machinery (which is deeply specific to the
lumped/wire family — flux-box geometry, port-cell V·I, the analytic
Hammerstad-Jensen anchor baked into its guards), this method combines the
LESS-INVASIVE half of each existing family's own validated machinery
unchanged: the coax side is built exactly like `compute_coaxial_two_port`'s
own single-ended stub (TEM source, matched annular-resistor feed, probe
array — reused verbatim, just one end instead of two); the MSL side is
consumed exactly like `compute_mixed_s_matrix` consumes its own MSL ports
(arbitrary caller-registered substrate/trace/ground-plane/pin-post geometry
via ordinary `sim.add(Box(...)/Cylinder(...), material=...)`). Both ports'
forward/backward wave amplitudes come from the SAME extractor,
`coaxial_line_reflection_from_plane_voltages` (a Z0-free matrix-pencil fit) —
applied to the coax probe ladder AND, deliberately, to the MSL probe ladder
too, in place of the mixed lane's own N-probe SVD fit (`extract_msl_nprobe`),
which that lane's own docstring documents as diagnostic-only with an
unresolved branch-sign instability. Each port's raw modal-voltage wave is
converted to a POWER wave via `a = V+ / sqrt(Z0)` (Z0 is already a real
analytic value on both sides, so no `Re()` is taken anywhere; coax: analytic TEM
Z0; MSL: analytic Hammerstad-Jensen Zc) before the same two-drive solve
`compute_coaxial_two_port` uses (`solve_two_port_from_wave_amplitudes`) —
this per-port `sqrt(Z0)` step is the fix for the "impedance-convention
mismatch" failure mode anticipated for this leg: solving directly on raw
(un-normalized) volt-wave amplitudes leaves the diagonal correct but scales
each off-diagonal entry by `sqrt(Z0_i/Z0_j)`, invisible on a coax-coax
through line (equal Z0 cancels) but not invisible here. This normalization is
independently unit-tested against a PLANTED, analytically known S-matrix
under UNEQUAL port impedances (`tests/unit/sparams/test_coax_msl_transition.py`), including
a dedicated regression test that reproduces the exact defect the fix
prevents.

The registered `impedance=` on either port is NOT the reference impedance of
the returned `s_params` (issue #581 review N2): `add_coaxial_port(impedance=...)`
/ `add_msl_port(impedance=...)` size only the feed resistor / termination and
(for coax) the TEM source amplitude calibration, while the power-wave
normalization (`z0_ref`, the `sqrt(Z0)` step above) always uses the analytic
TEM / Hammerstad-Jensen Z0 the method computes itself. When either port's
registered impedance diverges more than 5% from the analytic Z0 actually used,
`compute_coax_msl_transition(...)` emits a divergence advisory naming both
values — pass a matching `pin_radius`/`outer_radius` (coax) or
`width`/`height`/`eps_r_sub` (msl), or reconcile the mismatch, before trusting
a specific reference-impedance interpretation.

**One fixture has been run, and it is diagnostic, not validated.** The
committed fixture (a vertical coax landing on a grounded substrate edge via a
thin, unmatched pin-to-trace post — "no intermediate matching structure", per
the leg's own scoping) is internally self-consistent (finite, deterministic,
settles below −40 dB) but trips its own pre-declared reciprocity falsifier
badly: `|S12|` vs `|S21|` disagree by 94-100% across the three measured
frequencies (0.6/3.3/6.0 GHz), with the two-drive solve's raw condition
number `cond_a` in the 1e3-1e7 range at every bin (precise per-frequency
values: `cond_a` (raw) `7.0e4` / `4.6e3` / `4.3e7`). Own-drive incident-wave
magnitudes `a_inc` are 5-9 orders of magnitude apart: coax
`5.8e-9`/`5.8e-8`/`3.0e-9`, msl `8.2e-14`/`1.3e-11`/`7.1e-17`. The MSL probe
ladder spans `1.000` mm. Settling: `-43.9` dB (coax drive) / `-63.6` dB (msl
drive) at `N_STEPS=8000`.

**Attribution (corrected after adversarial PR review; see PR #581 review
findings B2/B3).** The first attribution written for this finding —
"near-degenerate two-drive amplification from strong junction reflection" —
did not survive its own data and was retracted. Three checks on this
fixture's own numbers refute it: (i) `cond_a` is almost entirely a per-drive
amplitude SCALE artifact — after per-column equilibration,
`cond_a_equilibrated` is 1.0004/1.0001/1.0040 (near 1) and the two drives'
incident-wave columns are near-ORTHOGONAL (normalized overlap ~1e-4 to
~4e-3), the opposite of "nearly parallel"; (ii) the "both ports strongly
reflecting" premise fails on this fixture's own measured `|S22|`
(0.000/0.000/0.500 — not uniformly near 1); (iii) the signature that DOES
match is a drive-amplitude mismatch between the two unrelated source
constructions (the MSL drive's own incident amplitude is 5-9 orders of
magnitude smaller than the coax drive's), exactly the second branch of the
two-drive solve's own warning ("one drive that failed to excite"), not the
first ("nearly linearly dependent" in the geometric sense). The PREDECLARED
alternative explanation — an MSL wave-extraction instrument-scoping limit,
not junction physics — is positively supported instead: the MSL probe
ladder on this fixture spans only 0.34%-3.37% of the guided wavelength
across the three measured frequencies, and the fitted propagation constant
on the MSL array does not track the analytic Hammerstad-Jensen beta
(21.2/116.4/211.7 rad/m) at all — the coax-driven fit gives
673.0/853.6/885.0 rad/m (4-32x too high and nearly frequency-flat, where
the true beta is not), and the MSL's own-drive fit gives 4.5/36.2/2881.3
rad/m with an implied decay length near one grid cell (not a real
propagating/decaying wave). Both discriminants are locked as test
assertions, not just prose (`tests/unit/sparams/test_coax_msl_transition.py`). Whether
the junction's own physical reflection also contributes is genuinely
UNRESOLVED by this one fixture — `sim.preflight()` on the same fixture
independently names a THIRD, also-unruled-out candidate mechanism (the MSL
port sits only 200um from its own x-CPML face, "recommended 600um... source-
side CPML reflection may inflate |S11|") — but the extraction-class
explanation is the better-supported one, and it NAMES the implementation
defect (probe ladder far shorter than any reasonable fraction of a guided
wavelength) that would justify a future, separately pre-declared retry (a
longer MSL probe ladder, e.g. >= 0.25 lambda_g) — not attempted in this PR.
`sim.preflight()` on this fixture also independently flags the same general
resolution class from a different angle: the pin post's 4-cell diameter is
under the ≥5-cell PEC-volume floor, and the 3-cell substrate is flagged for
>5% Z0 staircase bias.

| aspect | status |
|---|---|
| reciprocity | **FALSIFIER TRIPPED** — 94-100% deviation on the one committed fixture; root-caused to the MSL probe ladder being far too short for the matrix-pencil wave split (instrument scoping), not the assembler, not a missing-PEC defect, and not confirmed junction physics |
| off-diagonal magnitude | not validated — the one fixture measured near-zero, unreliable transmission |
| power-wave normalization (`sqrt(Z0)` step) | validated by planted-voltage unit tests, independent of any FDTD run |
| AD | out of scope for this leg (no `eps_scale`-style channel) |

Do not extend this lane's gates to "pass" without a NEW pre-declared falsifier
or an identified implementation defect in this fixture (repo R2 rule). This
finding DOES name such a defect (the MSL probe ladder's length, per the
attribution above), which is R2's own written escape clause for a second
attempt: a future retry should lengthen the MSL probe ladder (e.g. to
>= 0.25 lambda_g at the lowest measured frequency) BEFORE touching the
junction geometry itself — the junction's own physical dimensions (matching
structure, via width/length) remain a separate, still-open question this
fixture cannot yet speak to.

### Attempt 2 (PI-directed, R2's escape clause) — ladder fix CONFIRMED-provisional; reciprocity UNMEASURED at this settling

Attempt 1's own named defect (MSL probe ladder spanning only 0.34%-3.37% of
the guided wavelength) authorized exactly one retry. Attempt 2 lengthens the
MSL probe ladder 1.000mm → 8.000mm and widens the MSL port's x-CPML clearance
200um → 1500um, shifts the measured band {0.6,3.3,6.0}GHz → {6.0,8.0,10.0}GHz
(forced by the ladder-length requirement — 0.25·lambda_g at the old lowest bin
is 74.3mm, infeasible), and keeps the junction geometry (coax pin/outer
radius, PTFE fill, ground node, substrate, trace, pin-to-trace post) **byte-
identical** to attempt 1 (asserted by
`test_attempt2_junction_geometry_is_byte_identical_to_attempt1`, not just
claimed). This required a genuine, small API extension — the two families'
probe ladders were coupled through one shared set of
`compute_coax_msl_transition(...)` kwargs in attempt 1, which cannot host a
short coax ladder and a long MSL ladder at once; new `msl_probe_count` /
`msl_probe_start_cells` / `msl_probe_spacing_cells` parameters (default
`None` → falls back to the coax value, preserving attempt 1's exact
behavior — its own slow_physics test still reproduces its exact numbers
unchanged) decouple them.

**A second adversarial review (issue #585) found the FIRST write-up of
attempt 2's own result overclaimed.** That write-up attributed a still-broken
reciprocity finding to a coax/MSL drive-amplitude gap and locked an assertion
on it. This is **mathematically impossible** — per-drive (column) rescaling
of the two-drive solve leaves the S-matrix EXACTLY invariant by construction
(`A'=A·D, B'=B·D ⇒ B'·inv(A')=B·inv(A)=S` for any invertible diagonal `D`;
verified numerically at this attempt's own measured gap, deviation ~3e-16),
and the "amplitude ratio" invoked turned out to equal raw `cond_a` to 8
significant figures — the exact quantity this lane's own `cond_a_equilibrated`
split (issue #581 review finding B2) already says not to read as a
degeneracy witness. **Retracted; this is the THIRD retracted attribution on
this lane** (after attempt 1's own "near-degenerate two-drive amplification").
The corrected read below applies a **run-length invariance test** (the
repo's own discriminator for "settled physical quantity" vs "still-evolving
transient") across the two measured checkpoints (20000 → 45000 steps) and
reports a **column-power witness** (scaling-invariant, unlike the retracted
amplitude-gap story) in its place.

**Verdict, run-length invariance test applied — the two predeclared
discriminants SPLIT, but not the way the first write-up said:**

| discriminant | attempt 1 | attempt 2, 20000 steps | attempt 2, 45000 steps | invariance verdict |
|---|---|---|---|---|
| gamma-vs-beta ratio (coax-driven fit) | 4-32x off, non-monotonic | 1.085 / 0.826 / 0.976 | 1.128 / 0.854 / 1.071 | **STABLE, in [0.8,1.3] both times → CONFIRMED, provisional** |
| `cond_a_equilibrated` | 1.0004 / 1.0001 / 1.0040 | (not separately logged) | 1.00238 / 1.00244 / 1.00549 | near 1, consistent with attempt 1 |
| reciprocity worst deviation | 94-100% | 82.4% (0.824) | 93.8% (0.938) | **moved AWAY from any acceptance → UNMEASURED at this settling** |
| `\|S22\|` | 1e-8 to 1e-11 at 2/3 bins | 0.043 / 0.141 / 0.451 | 0.102 / 0.109 / 1.104 (×2.4 at top bin) | **still growing → UNMEASURED at this settling** |
| max`\|S\|` | 0.99 (well under limit) | 0.9933 | 1.1038 (crosses the 1.10 passivity-guard hard limit) | **crosses unity between checkpoints → UNMEASURED at this settling** |
| MSL-driven column power `Σ\|S_ij\|²` (NEW, issue #585 finding B5) | not computed | 0.0018 / 0.0199 / 0.204 | 0.0104 / 0.0119 / 1.218 | **THE open question — mostly ≪1 on a lossless structure, one bin >1 at the less-settled checkpoint** |
| `settling_db` | -43.9 / -63.6 dB (past -40) | -12.3 / -10.7 dB | -19.7 / -17.9 dB | improving direction only; neither checkpoint clears -40 dB |

**Verdict: gamma-vs-beta ratio is CONFIRMED, PROVISIONAL pending a settled
run** — the one discriminant that passes the run-length invariance test,
stable and in-band at both checkpoints. **Reciprocity, `|S22|`, and max`|S|`
are UNMEASURED AT THIS SETTLING**, not "refuted with cause identified" — all
three fail the same invariance test, moving in the wrong direction between
checkpoints. The passivity guard fires on the 45000-step run itself
("column power 1.218 exceeds limit 1.1... UNRELIABLE... do not interpret as
physics") — exercised directly in the committed test via the same shared
guard function `strict_passivity=True` would invoke, so the guard is not
silently bypassed even though the top-level `strict_passivity` flag on this
particular call is `False` (issue #585 review finding B3: no silent gate
removal). **The productive open question is the column-power witness**: on a
nominally lossless structure, the MSL-driven column's measured power is far
below 1 at most bins (power going somewhere unaccounted-for) with one bin
above 1 at the less-settled checkpoint (itself evidence of non-convergence,
since sum-power > 1 is impossible for a genuinely passive structure).

| aspect | status |
|---|---|
| gamma-vs-beta (MSL wave extraction) | **CONFIRMED** — passes the run-length invariance test across THREE checkpoints (20000/45000/135000 steps), the last at full settling; see the settled-run section below |
| reciprocity / `\|S22\|` / max`\|S\|` | **RESOLVED to MEASURED / ATTRIBUTED at full settling** — see the settled-run section below; do not cite the pre-settling checkpoint numbers above as physics |
| MSL-driven column power | **SHARPENED OPEN QUESTION** — the settled measurement below narrows it to a frequency-dependent effect, still not adjudicated |
| drive-amplitude gap "explanation" | **RETRACTED** (issue #585 B1) — mathematically cannot affect S; do not repeat |
| settled run | **RUN** — VESSL `369367252283`, `n_steps=135000`, both drives clear -40 dB; see the settled-run section below |

Per R2, attempt 2 itself stopped at the two checkpoints above (no third
ladder/clearance change in that PR). The settled VESSL run below is the named
next step that PR called for, not a new instrument change — it reuses the
SAME attempt-2 fixture unmodified.

### Settled run (VESSL `369367252283`, `n_steps=135000`, 2026-08-06) — gamma-vs-beta CONFIRMED; reciprocity and passivity now MEASURED

The same attempt-2 fixture (`_build_coax_msl_transition_sim_attempt2`,
unmodified), driven at `n_steps=135000` on VESSL `369367252283` (git SHA
`38a002c`), clears the -40 dB ring-down rule for the first time on this lane:
`settling_db` `-45.94` / `-44.17` dB. Tracked run log and result JSON:
`scripts/diagnostics/_coax_msl_transition_settled_run_logs/settled_run_369367252283_run.log`
/ `..._result.json`. Fill-contract record:
`tests/unit/sparams/test_coax_msl_transition.py::SETTLED_RUN_RECORD` (`status: "RUN"`).

| discriminant | attempt 2, 20000 steps | attempt 2, 45000 steps | settled, 135000 steps | verdict |
|---|---|---|---|---|
| gamma-vs-beta ratio (coax-driven fit) | 1.085 / 0.826 / 0.976 | 1.128 / 0.854 / 1.071 | 1.148 / 0.859 / 1.051 | **CONFIRMED** — third in-band `[0.8,1.3]` checkpoint, at full settling |
| reciprocity worst deviation | 82.4% (0.824) | 93.8% (0.938) | 91.4% (0.914) | **MEASURED at full settling** — not attributed to any of this lane's three retracted explanations |
| `\|S22\|` | 0.043 / 0.141 / 0.451 | 0.102 / 0.109 / 1.104 | 0.0808 / 0.1048 / 0.8937 | **MEASURED at full settling** |
| max`\|S\|` | 0.9933 | 1.1038 (crosses guard limit) | 0.9933 | **ATTRIBUTED** — the 1.1038 reading was a truncation artifact of an unsettled run; the settled guard is clean |
| MSL-driven column power `Σ\|S_ij\|²` | 0.0018 / 0.0199 / 0.204 | 0.0104 / 0.0119 / 1.218 | 0.00653 / 0.01098 / 0.79865 | **SHARPENED** — 99.3% / 98.9% / 20.1% of incident power unaccounted for, dropping sharply at 10 GHz |
| `settling_db` | -12.3 / -10.7 dB | -19.7 / -17.9 dB | -45.94 / -44.17 dB | **clears -40 dB for the first time** |

**gamma-vs-beta: CONFIRMED**, no longer provisional — a THIRD checkpoint,
this one at full settling, lands inside the predeclared `[0.8, 1.3]` band.
Three in-band checkpoints in a row is the strongest confirmation this lane's
own run-length-invariance discipline can produce.

**Passivity: ATTRIBUTED.** Settled max`\|S\|` = `0.9933` and the shared
passivity guard (strict semantics) raises nothing on this result. The earlier
passivity-guard trip (max`\|S\|` `1.1038`, column power `1.218`, both measured
at the 45000-step checkpoint and both impossible for a passive structure) is
now attributable to TRUNCATION — those readings were taken before ring-down
had actually settled, not evidence of a real passivity violation or an
extractor defect. This resolves the earlier "UNMEASURED AT THIS SETTLING"
label for max`\|S\|`.

**Reciprocity: now MEASURED**, not merely disclosed — worst deviation `91.4%`
(pair `(0, 1)`) AT FULL SETTLING. This is not a still-evolving transient (the
-40 dB rule is cleared) and is not explained by any of this lane's THREE
previously RETRACTED attributions (near-degenerate two-drive amplification;
the drive-amplitude gap, proven mathematically impossible; the
MSL-ladder-too-short instrument-scoping limit, which the gamma-vs-beta pass
above shows is resolved, not the cause of this number). What `91.4%` IS: a
real, settled property of THIS fixture's own measurement — a passive
reciprocal structure cannot physically have `\|S12\|` differ from `\|S21\|` by
an order of magnitude, so this deviation measures either (a) genuine
un-modeled loss/coupling asymmetry in how the extraction sees the two drives,
or (b) an instrument limitation in the extraction itself that survives
settling. Per this lane's own retraction history, this is deliberately NOT
adjudicated here.

**THE SHARPENED OPEN QUESTION (MSL-driven column power, at full settling):**
`0.00653` / `0.01098` / `0.79865` at 6/8/10 GHz — `99.3%` / `98.9%` / `20.1%`
of incident power is unaccounted for (neither transmitted nor reflected) at
the two lower frequencies, dropping sharply by 10 GHz. Derived from the same
result: `\|S12\|²` (column power minus `\|S22\|²`) stays of order `1e-7` at
all three bins — essentially nothing transmits to the coax side — while
`\|S22\|²` alone accounts for nearly all the retained power. Two named
candidates, NOT adjudicated:

- **(a) physical** — the unmatched vertical coax-to-trace launch (no
  intermediate matching structure, per this leg's own scoping) radiates the
  MSL drive's power into the CPML absorber at low frequency; consistent with
  the frequency trend (coupling back to the ports improves sharply toward
  10 GHz).
- **(b) instrument** — the MSL-side outgoing-wave (`b`) extraction misses
  non-quasi-TEM content generated near the junction discontinuity, which a
  quasi-TEM matrix-pencil fit on the MSL probe ladder cannot resolve.

A DISCRIMINATING check (named, not run): a closed-box variant of this fixture
(PEC walls in place of CPML, no absorber) — if the missing power reappears in
the port accounting once there is nowhere to radiate, that supports (a); if
it is still missing, that supports (b). Per this lane's three-retraction
history (near-degenerate drives; ladder-too-short, later resolved by
measurement; drive-amplitude gap, proven impossible), do not attribute
without running this or an equivalent falsifier.

Per R2, this stops here: no further ladder/clearance/geometry attempt in this
PR. The named next step is the closed-box discriminating check above, tracked
by issue #589.
