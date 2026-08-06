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
| Wire `add_port(..., extent=...)` | `run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | **limited** — multi-cell discrete feed across `extent`; magnitude evidence is stronger than absolute calibration evidence |
| Wire `add_port(..., extent=...)` | `forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `.freqs` (S11 vectors) | **limited** — uniform, single-device AD path |
| `add_msl_port(...)` | `compute_msl_s_matrix(...)` | `MSLSMatrixResult.S`, `.freqs`, `.Z0`, `.beta`, `.port_names`, `.reliable` | **limited** — E5-narrow / eigenmode-blocked; external notch agreement is characterized, not tight; `eps_override` AD checked against an f64 referee on the band-mean `\|S21\|^2` objective (rel_err 0.0026 at the gate's num_periods=20 fixture, threshold 0.03; issue #530, superseding the pre-#530 `sum\|S_ij\|^2` objective and its 0.0331/0.10 figures) |
| `add_waveguide_port(...)` | `compute_waveguide_s_matrix(...)` | `WaveguideSMatrixResult.s_params`, `.freqs`, `.port_names`, `.port_directions`, `.reference_planes` | **limited** — broad magnitude evidence for documented uniform single-mode rectangular guides; phase and junction evidence are narrower |
| `add_waveguide_port(...)` | `run(...)` | `Result.waveguide_sparams[name]` | **limited diagnostic** — per-port output, not the full multi-port matrix API |
| `add_coaxial_port(...)` | `compute_coaxial_line_reflection(...)` | `CoaxialLineReflectionResult` | **limited** — exactly one `face="top"` port; broad-E5 analytic and broad-E4 MEEP evidence for the documented TEM-line result |
| `add_coaxial_port(...)` | `compute_coaxial_s_matrix(...)` | `CoaxialSMatrixResult` | **experimental and deprecated** — older single-plane V/I path; can produce non-physical `\|S11\| > 1` for a lossless short |
| `add_coaxial_port(...)` | `compute_coaxial_two_port(...)` | `CoaxialTwoPortResult` | **experimental** (issue #489 stage 2) — two-drive through-line 2-port solve; every DUT it can currently gate against is azimuthally symmetric (TM0n only); an external openEMS referee is now REGISTERED (`validation/crossval/21_coax_two_port_referee.py`, promoted 2026-08-04) bracketing the through-line class — it builds and runs its own independent openEMS model offline and does not execute rfx in-process; EXPERIMENTAL status stands until the transition/AD legs close; no phase claim |
| `add_coaxial_port(...)` + `add_msl_port(...)` | `compute_coax_msl_transition(...)` | `CoaxMSLTransitionResult` | **experimental, diagnostic-only** (issue #489 leg 4) — coax-to-microstrip transition, two-drive; two committed fixtures (attempt 1 + a longer-ladder attempt 2). Attempt 2's gamma-vs-beta discriminant is CONFIRMED, PROVISIONAL (passes a run-length invariance test); reciprocity/`\|S22\|`/max`\|S\|` are UNMEASURED at this settling (fail that same test — a settled VESSL run is predeclared, not yet run); a scaling-invariant column-power witness is the OPEN question — see the section below — do not treat as a validated transition |
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

Relevant implementations and tests include `tests/test_sparam.py`,
`tests/test_port_dump_replay.py`,
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
`tests/test_twoport_wire_port.py`, `tests/test_wire_port_sparams_forward.py`, and
`scripts/diagnostics/report_wire_replay_sweep.py`.

## Microstrip-line port

**API:** `compute_msl_s_matrix(...)` with the laplace/quasi-TEM model.

**RF evidence (E5-narrow / eigenmode-blocked):**

- The uniform thru-line check uses `|S21|` in `(0.90, 1.05)` and
  `Re(Z0)` in `(40, 65) ohm` for the cited `dx=80 um` setup.
- The analytic quarter-wave-notch case
  (`validation/crossval/06b_msl_notch_filter_uniform.py`) last reported
  `1.63%` frequency error, `-34.3 dB` notch depth, and median
  `Re(Z0)=48.6 ohm`. That run predates the #511/#507 extractor fixes
  (PR #516 / `f95240f`) and has not been regenerated since — there is no
  committed RESULT ARTIFACT from a post-#511 rerun, though the producer
  script itself is committed and manifest-registered (claims-bearing,
  `pr-fast`/`gpu-manual` tiers). Treat this number as describing the
  superseded extractor until issue #519 re-runs it and commits a
  refreshed result.
- The committed matched-geometry OpenEMS comparison at `dx=50 um` reports a
  `5.8%` notch-frequency difference, linear `|S21|` mean difference `0.105`,
  and maximum difference `0.2172` over 2.5--6 GHz. This is a characterized
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
  `tests/test_msl_ad_fd_converged.py`'s docstring for the full replacement
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
  figures (issue #527, closed) and the older pre-#516 `0.000110` figure
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
  instead of silently — not eliminated (see `tests/test_msl_ad_fd_converged.py`
  for the ULP-resolving-power derivation and `tests/_msl_ad_objective.py` for
  the full statement, including the open question of what mechanism drives
  the gradient — a reference-plane artifact against the wave split's frozen
  Hammerstad-Jensen `z0_hj`, or genuine beta/reflection physics — tracked in
  **issue #560**, which this PR ships without resolving). The #515 AD smoke
  shares this same objective function (`tests/_msl_ad_objective.py`) so the
  two tests cannot drift apart. The launch fixture derives from registered
  materials on both the FD and AD sides; staticness is regression-locked by
  `tests/test_msl_source_fixture_static.py`.
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
  See `validation/crossval/20_msl_phase_referee.py` (manifest entry
  `20_msl_phase_referee`) and `tests/test_msl_phase_referee_header.py`.

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
  `tests/test_waveguide_broad_e5_phase_gates.py` and
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
  record-length settling witness (`num_periods` 60 vs 120 --
  `compute_waveguide_s_matrix` has no built-in energy-based `settling_db`
  for the waveguide port family, unlike the MSL calculator) agrees to
  `0.000 ns`. A domain-size invariance run (`+100 mm` growth) holds the pass
  verdict (`0.0266 ns`, still under the gate). Three genuinely independent
  falsifiers (skipping the phase-unwrap step, dropping the leading minus
  sign, and using the wrong `L_eff` -- domain length instead of
  reference-plane separation -- in the analytic comparator) all red at
  `>= 1.5 ns`. See `tests/test_waveguide_group_delay_near_cutoff.py`,
  `tests/test_waveguide_group_delay_tolerance_envelope.py`, and
  `tests/fixtures/waveguide_group_delay/wr340_near_cutoff_group_delay_envelope.json`.
- Neither lane covers PEC-short, T-junction, nonuniform, or multimode
  configurations for phase or group delay -- those remain uncharacterized.
- MSL de-embedded phase vs an external (openEMS) referee with the convention
  mismatch resolved (issue #490 Lane 2) is explicitly OUT of scope for both
  lanes above; the microstrip-line section below is unchanged.

**Nonuniform transverse mesh:** single-mode `normalize=True` and
`normalize="flux"` run. Analytic Airy fixtures cover grading ratios 1--3,
relative permittivity 2 and 4, and 8.2--12.4 GHz with a maximum
linear-magnitude difference of `0.01561`. A passed Palace magnitude comparison
covers `normalize="flux"`, a graded-`dy` ratio of 2, and WR-90
empty/PEC-short/dielectric-slab cases over 8.2--12.4 GHz; its maximum and mean
linear-magnitude differences are `0.07009` and `0.01042`. This is external RF
evidence for that configuration, not for other profiles, bands, phase,
multimode extraction, or arbitrary junctions. The calculation remains
experimental outside those stated results. `eps_override` and `sigma_override`
differentiation is implemented only with `normalize="flux"`.
`tests/test_waveguide_nu_flux_ad.py` finite-difference-checks `eps_override`;
there is no corresponding nonuniform `sigma_override` AD-vs-FD test. Neither
implementation nor gradient regression is RF validation.

**Those nonuniform fixtures are STALE and pending regeneration (issue #562).**
`tests/fixtures/waveguide_nu_broad_e5/waveguide_wr90_nu_flux_broad_e5_envelope.json`
and
`tests/fixtures/waveguide_nu_broad_e4/waveguide_wr90_nu_flux_broad_e4_comparison.json`
were generated while the nonuniform grid realized every axis one cell short of
the requested domain, so their guide is `a - dy_edge` wide rather than `a`, and
their gate tests replay the frozen numbers rather than re-running FDTD — the
tests therefore pass while describing the earlier geometry. Nothing was
re-blessed: the numbers above stand exactly as measured, and the lane stays
experimental, so the matrix is conservative rather than wrong. Regenerating
them is expected to *improve* agreement (the guide finally being the requested
width); until it happens, treat the cited magnitude differences as an upper
bound obtained on a slightly narrow guide, and do not compare them against
newly generated nonuniform numbers.

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
`tests/test_coax_end_to_end_ad.py`.

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
solve that does not assume the non-driven port sees zero incident wave. It is
**experimental**: every DUT it can currently gate against (none, a matched
feed, or a coaxial dielectric plug) is azimuthally symmetric and excites only
TM0n modes, while the transition discontinuities issue #489 targets excite
TE11 (cutoff 25.17 GHz on the validated SMA line, evanescently surviving to
the first probe plane). No external referee has run against this method and
no phase claim is made. See `tests/test_coax_two_port_fdtd.py` for the
measured single-run envelope and its provenance.

`compute_coaxial_two_port(...)` now has the same `eps_scale` differentiable
channel as the 1-port method above (issue #489 leg 3): the gate compares
float32 AD (as shipped) against a float64-loss central finite difference —
the FDTD fields stay float32 (same baseline as the MSL f64 referee), but a
scoped `enable_x64()` around the forward call makes the DFT-accumulator-and-
downstream math run at float64 (`rfx/probes/probes.py` keys the accumulator
dtype off `jax.config.x64_enabled`, independent of the `precision="float32"`
pin). Measured rel_err 0.51% at `h=2e-3` against a 2% gate; owner-platform
(GPU) re-measurement pending. See `tests/test_coax_two_port_ad.py` and
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
analytic empty-space/slab diagnostics. Representative internal differences are
`max_abs_diff 0.06067` for the empty-space analytic-null check and
`max_mag_abs_diff 0.06212` for the three-frequency homogeneous-slab magnitude
check. These are not RCWA or independent full-wave validation, and there is no
documented high-level Floquet S-parameter API. Treat the result as experimental.

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
| off-diagonal magnitude | experimental; internal reciprocity witness 9% on the probe-fed MSL fixture (55% on the wave channel) |
| absolute magnitude | **NOT validated** — no external-solver referee has been run |
| off-diagonal phase | provisional (mixes two reference-plane conventions) |
| per-column power | an algebraic identity on the flux channel — **not** a passivity check |

Because the flux normalization makes column power identically 1 whenever the
arriving power equals the net launched power, a green passivity result on this
lane carries no information. Reciprocity is the only independent internal
witness, and the method warns when it exceeds `reciprocity_tol`.

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
under UNEQUAL port impedances (`tests/test_coax_msl_transition.py`), including
a dedicated regression test that reproduces the exact defect the fix
prevents.

**One fixture has been run, and it is diagnostic, not validated.** The
committed fixture (a vertical coax landing on a grounded substrate edge via a
thin, unmatched pin-to-trace post — "no intermediate matching structure", per
the leg's own scoping) is internally self-consistent (finite, deterministic,
settles below −40 dB) but trips its own pre-declared reciprocity falsifier
badly: `|S12|` vs `|S21|` disagree by 94-100% across the three measured
frequencies, with the two-drive solve's raw condition number `cond_a` in the
1e3-1e7 range at every bin.

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
assertions, not just prose (`tests/test_coax_msl_transition.py`). Whether
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
| reciprocity worst deviation | 94-100% | 82.4% | 93.8% | **moved AWAY from any acceptance → UNMEASURED at this settling** |
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
| gamma-vs-beta (MSL wave extraction) | **CONFIRMED, PROVISIONAL** — passes the run-length invariance test; locked as a real test assertion |
| reciprocity / `\|S22\|` / max`\|S\|` | **UNMEASURED at this settling** — recorded (both checkpoints) but NOT gated; do not cite either checkpoint's number as physics |
| MSL-driven column power | **OPEN QUESTION** — scaling-invariant, regression-locked as a measured observation, not yet explained |
| drive-amplitude gap "explanation" | **RETRACTED** (issue #585 B1) — mathematically cannot affect S; do not repeat |
| settled run | **PREDECLARED, UNRUN** — `SETTLED_RUN_RECORD` in the test file targets VESSL (~135000 steps, extrapolated); not attempted in this PR |

Per R2, this stops here: no third ladder/clearance attempt in this PR. The
named next step is the settled VESSL run (predeclared UNRUN, fill-contract
pattern) — reciprocity, `|S22|`, and the column-power question all need that
settled data before any further attribution, not a new instrument change.
