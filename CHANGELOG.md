# Changelog

All notable changes to `rfx-fdtd` that affect user-visible behaviour are
recorded here. Dates follow local (KST) convention. Version bumps follow
SemVer — **BREAKING** entries are flagged in upper-case.

## [Unreleased]

### Changed — empty-window gradient + `add_lumped_rlc`-is-not-a-port pinned in docs (blind docs-only audit, doc-pin R2-STOP)

- A blind, docs-only agent (validated across two model families) trusted a
  finite-difference-verified gradient whose loss was `~1e-7` — orders of
  magnitude too small — because the reflection it minimized never landed in
  `minimize_reflected_energy`'s late-time split window. AD and FD agreed because
  both differentiate the **same empty window**; the docs prescribed the FD check
  as the trust ritual but never warned it cannot detect an empty observable.
- Documented the round-trip **precondition** for `minimize_reflected_energy`'s
  split window (docstring + [Inverse Design](/rfx/guide/inverse-design/)) and the
  "a passing finite-difference check is necessary, not sufficient — check the loss
  **magnitude** against a physical expectation" caveat in
  [Autodiff & Adjoint](/rfx/guide/autodiff-adjoint/) and
  [Gradient Behavior](/rfx/guide/gradient-behavior/).
- Clarified in [Sources & Ports](/rfx/api/sources-ports/) that
  `add_lumped_rlc(...)` is a circuit element, **not** a reflection-referenced
  port: a single-cell lumped element reports its own self-interaction, not a
  `Z0`-referenced `S11`; drive with `add_port(..., impedance=Z0)` to measure a
  load reflection coefficient.
- R2-STOP on a preflight advisory: the split-window premise depends on the
  geometry's round-trip time and the run length, which `preflight()` cannot know,
  so the guidance lives in the docs, not a call-time gate. Locked by
  `tests/test_empty_window_gradient_caveat_docpin.py`. No physics or preflight
  behaviour changed.

### Changed — lossless-cavity infinite-Q pinned in the harminv docstrings (LLM-naive-usage audit item #4, doc-pin R2-STOP)

- Measuring a resonator Q via `harminv` / `harminv_from_probe` on a **lossless
  closed (PEC) cavity** gives a window-length artefact (infinite physical Q),
  not physics — the frequency is fine, the Q is not. Documented with a
  `.. warning::` in both docstrings (add a realistic `sigma`/`tan_delta` loss
  or use an open CPML boundary before trusting a Q).
- R2-STOP on a preflight advisory: a "closed + lossless + soft source" trigger
  has NO clean call-time discriminator — legitimate lossless-cavity resonance
  sims (e.g. the NU stage-1 cavity physics gate, memory-planning cavities) use
  the same config and would be false-alarmed. Per "a false-alarming preflight
  erodes trust worse than the silent gap," the guidance lives in the API docs,
  not in `preflight()`. No physics or preflight behaviour changed.


### Changed — RCS bistatic-pattern validation scope pinned in docs (LLM-naive-usage audit item #2, doc-pin R2-STOP)

- `compute_rcs` returns a full bistatic `rcs_dbsm` / `rcs_linear` pattern, but
  only the `monostatic_rcs` (backscatter) bin is cross-validated against the
  exact Mie series (~0.06 dB at ka~1). At the auto-placed default NTFF box
  (`ntff_offset=1`, ~1 cell off the TFSF boundary, deep in the reactive near
  field) the off-backscatter bins can be several dB to ~20 dB off — the
  committed ka~1 PEC sphere shows a spurious forward-oblique lobe near 25-55 deg
  scattering angle measured ~10 dB high vs Mie. This was already recorded
  non-gated in `tests/fixtures/rcs_sphere_mie/`; the audit asked whether it
  should surface at call time.
- R2-STOPPED to a doc-pin (no runtime advisory added). `monostatic_rcs` is
  computed at the exact backscatter direction INDEPENDENT of the
  `theta_obs`/`phi_obs` grid, and every validated monostatic test passes a full
  observation grid, so there is no call-time signal that separates
  "monostatic-only" from "bistatic" intent to key an advisory on without
  false-alarming those tests (the same trap as item #3). Wiring the
  `Simulation` NTFF near-field guard into `compute_rcs` would likewise fire on
  every validated monostatic RCS test.
- Pinned the caveat in the `compute_rcs` / `RCSResult` docstrings, the module
  docstring, and `docs/public/guide/farfield-rcs.md`. Verify-first finding also
  recorded: enlarging `ntff_offset` alone does NOT close the oblique gap at test
  scale (offset 1->2 leaves the 25-55 deg error ~10 dB and worsens backscatter;
  a larger domain did not help), so the docs steer users away from that dead
  end. No physics changed; no gate touched. Doc-contract witness in
  `tests/test_rcs_bistatic_caveat_docpin.py`.

### Added — waveguide S-matrix soft over-unity advisory (LLM-naive-usage audit item #5)

- `compute_waveguide_s_matrix(normalize=False)` now emits a SEPARATE, humble
  ADVISORY (warning, never raise) when a passive extraction's max column power
  lands in `(2.25, hard_limit]` — above the documented single-run overshoot
  envelope but below the `passivity_tol` extractor-broken hard limit. On the
  loose `normalize=False` tol (2.0 → column-power limit 3.0) the window
  `(~2.0, 3.0]` was previously unguarded: a validated PEC short sits at column
  power ~2.0 there (a documented Yee/near-cutoff artifact, `|S21|≈1` not
  cancelled by the single-run decomposition), so the existing self-check
  (`_warn_if_nonpassive_smatrix` in `rfx/api/_sparams.py`) stayed silent for a
  passive result materially above that envelope. A real coarse-mesh (`dx=2 mm`)
  WR-90 PEC short landing at column power ~2.51 reproduced the silent gap.
- The floor is column power 2.25 (`|S| ~ 1.5` for a 1-port): above the ~2.0
  documented envelope AND the committed `normalize=False` PEC short (column
  power ~2.00, `test_pec_short_s11_magnitude` /
  `test_normalize_aware_tol_tolerates_documented_overshoot`) with margin. The
  window is EMPTY on the tight-tol path (`normalize='flux'`/`True` → tol 0.10 →
  hard limit 1.10 < 2.25), so the advisory only fires for `normalize=False`. No
  physics changed; the `tol=2.0` hard threshold and every committed gate are
  untouched. Message wording is distinct from the hard `UNRELIABLE` error.
  Witness + false-positive gates in `tests/test_sparam_passivity_guard.py`.

### Changed — pec-open-radiator advisory kept NTFF-gated (LLM-naive-usage audit item #3, R2-STOP)

- The audit asked whether `_validate_cfg_pec_boundary_open_structure`
  (`rfx/api/_preflight.py`) should be ungated from its `self._ntff is not None`
  condition so an open radiator read via a near-field probe / S11 alone also
  warns. R2-STOPPED after investigation: NTFF is the SOLE radiation-intent
  signal on the `Simulation` config, so a source (and/or a finite PEC object)
  inside a `boundary="pec"` domain is config-IDENTICAL between an open radiator
  that mistakenly used PEC and a legitimate closed cavity / internal-PEC
  numerics test (e.g. `test_adi.py::test_simulation_adi_internal_pec_geometry_masks_ez`,
  `test_conformal.py::test_api_conformal_flag`,
  `test_extract_s_matrix_pec_mask.py`). Any broadening that caught the footgun
  would false-alarm that legion of valid closed-structure sims, and a
  false-alarming preflight erodes trust worse than the silent gap. The advisory
  is unchanged; the decision is locked by regression tests in
  `tests/test_preflight_false_positives.py` (NTFF radiator still warns; valid
  closed structures stay silent).

### Added — lumped R/L/C component values as a differentiable design variable (WP 4-E)

- `Simulation.forward()` now processes `add_lumped_rlc(...)` elements on the
  uniform single-device lane and gained a keyword-only `rlc_values_override`
  injection surface, so a lumped element's R/L/C can be a `jax.grad` design
  variable. `LumpedRLCSpec` stores plain floats, so a component value enters
  the AD tape AS a tracer via
  `forward(rlc_values_override={element_index: {"R": R, "C": C, "L": L}}, ...)`;
  a missing index or key falls back to the registered float. A traced meta
  builder (`rfx.lumped.build_rlc_meta_traced` / `setup_rlc_materials_traced`)
  drops the `float()` coercions and keeps element topology decisions static, so
  `jax.grad(|S11|^2)` w.r.t. R and C is finite, nonzero and FD-consistent
  (rel < 5%, scoped x64) — gated in `tests/test_lumped_rlc_ad.py`.
- **Fixed a silent no-op**: previously a registered `add_lumped_rlc` element was
  IGNORED by `forward()` (the differentiable lane never iterated
  `self._lumped_rlc`), so a sim's `forward()` |S11| was byte-identical with or
  without the element. `forward()` now correctly reflects the element even
  without an override. Because `run(compute_s_params=True)` extracts its
  S-matrix through the same `_forward_from_materials` path, that lane was
  IGNORING a co-located RLC too — it now reflects it as well (witness-locked in
  the same test file). The main field solve `run()` was always correct and is
  BYTE-IDENTICAL (the concrete `build_rlc_meta` / `run(lumped_rlc=...)` path is
  untouched; golden regression-locked).
- The lumped/wire port S11 DFT accumulators
  (`forward(port_s11_freqs=...)` / `run(compute_s_params=True)`) now derive
  their complex dtype so they promote under a scoped `jax_enable_x64` instead of
  tripping the `lax.scan` carry-dtype contract — this makes the wave-decomp S11
  AD lane (WP 4-E and the pre-existing `eps_override` channel) usable under
  scoped x64. Byte-identical with x64 off (accumulators stay complex64).
- Scope: uniform single-device forward lane only. `rlc_values_override` on the
  non-uniform / distributed forward lanes raises `NotImplementedError` (those
  lanes do not process `add_lumped_rlc` in `forward()`).

### Added — opt-in multi-start / best-iterate / step-clamp knobs on `optimize()` (WP 4-C)

- `optimize()` gained four keyword-only, DEFAULT-OFF parameters promoted
  from the MSL open-stub inverse-design example (`_multistart_adam`, issue
  #171): `n_starts` (best-of random restarts), `best_iterate` (return the
  lowest-loss visited iterate instead of the last), `step_clamp` (bound the
  L2 norm of each Adam latent update), and `seed` (reproducible restart
  inits).
- Defaults (`n_starts=1`, `best_iterate=False`, `step_clamp=None`) reproduce
  the previous single-run Adam loop BYTE-for-BYTE — start 0 always uses the
  caller's `init_latent`, extra restarts draw i.i.d. standard-normal latents
  from `seed`, and the legacy NaN/Inf gradient guard (warn + return the last
  finite design) is unchanged. A bit-identity gate against a verbatim copy of
  the legacy loop is committed in `tests/test_optimize_multistart.py`.
- The step-clamp is GENERIC (an exact whole-step L2-norm rescale, direction
  preserved) rather than the example's physical-length bisection, which stays
  specialized in the example. Multi-start fails closed (raises) only when
  every restart is non-finite; a single start keeps the fail-soft contract.
- Scope is optimizer-loop plumbing only — no objective or numerics change.
  Extra restarts help only on a genuinely multimodal loss surface.

### Fixed — `compute_rcs` monostatic extraction pointed at −z broadside, not backscatter (issue #276)

- `RCSResult.monostatic_rcs` was argmin-extracted at (θ=π, φ=0), which under
  the far-field convention (`r_hat = [sinθcosφ, sinθsinφ, cosθ]`) is the −z
  BROADSIDE direction — not the −x backscatter of the +x TFSF incidence
  (θ=π/2, φ=π). Measured on a validated exact-Mie PEC-sphere falsifier
  (ka≈1.0, dx=λ/40): the shipped number was 10.06 dB off Mie; the same run
  re-extracted at the true backscatter direction is 0.06 dB off.
- The backscatter direction is now derived from the incident propagation
  unit vector (b_hat = −k_hat) and the far field is evaluated EXACTLY at
  that direction on the already-accumulated NTFF data — no observation-grid
  snapping (the default φ grid `[0, π/2]` does not even contain φ=π).
  `monostatic_rcs` is therefore independent of `theta_obs`/`phi_obs` and is
  now always computed (previously `None` for empty observation grids).
- New committed evidence + gate: `tests/fixtures/rcs_sphere_mie/`
  (exact-Mie oracle self-validated by four rfx-independent witnesses —
  Rayleigh 9(ka)^4, GO→1, term-doubling convergence, bistatic-bridge — plus
  fixture JSON with the full H-plane trace) and
  `tests/test_rcs_mie_fixture.py` (live recompute at the committed
  resolution, gate |Δ| ≤ 1.0 dB vs Mie; measured 0.06 dB). Claim scope is
  MONOSTATIC magnitude at the committed resolution only — the same run
  shows a NON-GATED spurious forward-oblique lobe (25–55°, ~10 dB high vs
  Mie; TFSF/NTFF forward-face contamination suspected) and a ~1.6 dB
  forward-scatter delta, documented per-angle in the fixture rather than
  hidden.

### Fixed — `estimate_ad_memory` counts the live-segment rematerialization tape (#277)

- Both segmented paths (`checkpoint_segments` and `checkpoint_every`) now model
  peak reverse-mode AD memory as
  `(2 x active_segments + live_tape_steps) x field_bytes + forward + ntff`.
  The previous formula (issue #39) counted only the segment-boundary
  carry + cotangent term; during the backward pass each segment is
  rematerialized as a unit, so one segment's per-step field tape
  (`n_steps // checkpoint_segments`, or `checkpoint_every` on the padded
  non-uniform path, capped at the active step count) is resident on top of it.
  At segment counts far from `sqrt(n_steps)` the old estimate under-counted
  peak by up to `~(segment_len / (2 x segments))x` (a hypothetical
  far-from-sqrt(N) knob: ~272x at `n_steps=6999`, `checkpoint_segments=3`;
  the committed desk ladder's sqrt(N) path shifts only ~1.5x); the VESSL 369367233509
  inverse-design gradient peaks (5.84 GB at chunk=100/n_steps=10000 vs the old
  ~3.1 GB estimate) confirm the missing term.
- Planning numbers shift accordingly: `ad_segmented_gb` grows, is minimized
  near `sqrt(2 x n_steps)` steps per segment (no longer monotone in the knob),
  and `preflight`/`estimate_ad_memory` VRAM warnings fire more often — that is
  the estimate becoming honest, not a regression. Warnings now point at the
  dominant term (increase or reduce the knob toward `sqrt(n_steps)`).
- `explain_ad_memory` decomposes the new term as a
  `segmented_live_segment_tape` component; `plan_ad_memory`'s does-not-fit
  diagnostics now report the true least-memory candidate (balanced segment
  size) instead of `checkpoint_segments=1` / `checkpoint_every=n_steps`, which
  are ~full-AD-sized under the corrected model.

### Added — waveguide multi-port junction references (`port_reference_sims`)

- `compute_waveguide_s_matrix(..., port_reference_sims=[...])` exposes public
  plumbing for interior-PEC multi-port structures (T-junctions / branches /
  septa). Each `port_reference_sims[i]` is a `Simulation` describing the matched
  STRAIGHT continuation of driven port `i`'s guide (same domain / `dx` /
  boundary, no junction); its PEC-folded materials feed the flux extractor's
  per-port incident-power reference so the guided `P_inc` is correct. The
  default shared VACUUM reference strips the interior PEC and mis-normalizes
  `P_inc`, inflating every `|S|` (measured on the compact 3-port T-junction:
  `normalize=True` max|S|~230; `normalize='flux'` max|S|~9.8, |S11|~1.9).
- Only valid with `normalize='flux'` (raises otherwise); single-mode ports,
  uniform mesh only; not combinable with `eps_override` / `sigma_override`;
  each reference grid must match the device grid (shape + `dx`); one reference
  per waveguide port.
- Two in-method advisories (pure NumPy, no FDTD) warn when the **far-port
  discipline** is not met: probe clearance < 5 evanescent decay lengths of the
  next higher mode, and CPML thickness < ~0.5 guide wavelengths at band centre.
- **Far-port numbers (verified 2026-07-06).** On a far-port geometry (arms
  90/90/70 mm, 48 mm CPML, dx 1.0/0.667 mm) the matched-reference flux path
  reaches passivity 1.006/1.002, reciprocity 0.001, mesh-convergence 0.0297 and
  0.087 vs MEEP (r2000, cross-device). **Necessary-but-not-sufficient caveat:**
  on COMPACT geometry it fixes |S11| (1.86 → 0.49) but the overall matrix stays
  non-physical (residual max|S|~3.9); this enables junction measurements only
  under the documented discipline, not for arbitrary compact junctions.
- Companion committed evidence: `tests/fixtures/waveguide_tjunction_e4/` +
  gate test `tests/test_waveguide_tjunction_e4e5_gates.py`; guard / advisory /
  A/B-witness coverage in `tests/test_waveguide_port_reference_sims.py`.

### Added — coaxial S-parameters: AD-traceable + end-to-end differentiable + `broad_e5_passed` (PRs #260, #261, #262)

- `compute_coaxial_line_reflection(...)` is now **end-to-end differentiable** via a new
  `eps_scale` design channel: `grad(|S11|**2)` w.r.t. the dielectric flows through
  FDTD -> DFT plane accumulators -> modal voltage -> matrix-pencil reflection -> Gamma.
  Pass a scalar or `(nx, ny, nz)` `eps_scale` to optimize a coaxial reflection under
  `jax.grad`; `eps_scale=None` is byte-identical to the validated numpy path.
- The reflection extractor (`coaxial_line_reflection_from_plane_voltages`) is now
  `jax.numpy`-traceable (dual-path: concrete -> numpy float64, traced -> jnp), and a
  differentiable voltage line-integral `coaxial_line_plane_voltage_jnp` was added.
- **`coaxial_port` promoted to `broad_e5_passed`**: with the committed broad-E5 analytic
  envelope + broad-E4 MEEP comparison (PRs #256/#259) and the passing composition
  AD-vs-FD gate (`tests/test_coax_end_to_end_ad.py`, 2.6%), the clean-checkout auditor
  (`check_port_external_references.py`) returns `coaxial_port` PASSED. Tolerances unchanged.

### Fixed — multi-port wire `run(compute_s_params=True)` S-matrix (item-5 Stage 4)

- Multi-port wire ports now return the FULL S-matrix from `run(compute_s_params=True)`
  on a uniform mesh. Previously the uniform fast-path filled only the diagonal
  `S[j,j]` and silently dropped the off-diagonal (a 2-port wire `S21` came out
  identically `0`); multi-port wire now routes through the production-scan driver
  (`compute_lumped_wire_s_matrix_via_scan`), which fills the full matrix. The
  well-conditioned single-port wire fast-path is unchanged.
- **Behaviour change**: a MIXED lumped + wire port set with `compute_s_params=True`
  now raises `NotImplementedError` (their wave-decomposition conventions differ)
  instead of silently returning a wire-only diagonal matrix that dropped the
  lumped ports. Use a homogeneous all-lumped or all-wire port set.
- The eager `extract_s_matrix_wire` is no longer on the `run()` hot path (kept for
  diagnostics / openEMS-crossval tooling), removing the last hand-maintained
  second-FDTD-loop drift-class root (siblings of #203/#205/#206).

### Added — MSL S-parameters on non-uniform meshes (PRs #238, #239)

- `compute_msl_s_matrix()` now runs on non-uniform (`dz_profile`/`dx_profile`/
  `dy_profile`) meshes with the default `mode="laplace"` (and `"uniform"`) feed,
  routed through the non-uniform runner. `mode="eigenmode"` on a non-uniform
  mesh raises `NotImplementedError` (previously ALL non-uniform meshes were
  rejected). The extractor math (probe abscissae, transverse integrals) is
  NonUniformGrid-aware (#238); uniform-mesh results are byte-identical.
  Scope: the NU lane is internally gated (settled-S11 GPU physics gate,
  `tests/test_msl_nu_sparam_gate.py`); external cross-solver validation of the
  NU lane is still outstanding — see `docs/guides/sparameter_support_matrix.md`.

### Fixed — MSL V·I current DFT leapfrog half-step (PR #240)

- The MSL port's Hy/Hz current DFT now applies the Yee leapfrog `exp(+jω·dt/2)`
  half-step correction (H fields live at half-integer time steps). Reported MSL
  S-parameters change slightly (~0.3–0.7° phase-scale over the validated bands).

### Deprecated — removal slated for rfx v2.0

Both functions below already carried `DeprecationWarning`s; this release only
pins the removal version (v2.0) in the warning text and here.

- `compute_coaxial_s_matrix()` (single-plane V/I in a closed PEC box; reports
  non-physical `|S11| > 1` for a lossless short) — use
  `compute_coaxial_line_reflection()` (validated coax-line method).
- `minimize_s11_at_freq()` (time-gating heuristic biased for short-round-trip
  antennas, issue #72) — use `minimize_s11_at_freq_wave_decomp` +
  `Simulation.forward(port_s11_freqs=...)`.
- (Matching the existing v2.0 removal notices on `pec_faces=` and
  `set_periodic_axes()`.)

### Added — AD-memory planning explainability (PR #231; diagnostic/planning-only, no numerics-path change)

- `Simulation.explain_ad_memory(...)` decomposes the selected AD-memory estimate
  into named contributors (field tape, segment-boundary carries, CPML/material,
  NTFF) with a component-sum invariant. New public types `ADMemoryComponent` and
  `ADMemoryExplainabilityReport` (exported from `rfx`).
- Evidence-class labels on AD-memory artifacts (`static_estimate` /
  `calibrated_conservative_plan` / `static_ad_explainability`) so a planning
  estimate is never confused with profiler evidence or a bounded certificate.

### Changed — AD-memory planning (PR #231)

- Checkpoint knobs separated: `checkpoint_every` (non-uniform scan-of-scan chunk
  length) vs `checkpoint_segments` (uniform segmented-scan count; must divide
  `n_steps`; mutually exclusive).
- Conservative `AD_MEMORY_FIT_SAFETY_FACTOR` (1.30) applied before the
  full-AD / segmented fit flags; `design_mask` is recorded but does NOT reduce
  the estimate until masked-state memory has observed calibration. Strict input
  validation, strict-JSON (`allow_nan=False`) serialization, and MB-aware
  formatting (sub-10 MB no longer rendered as `0.00 GB`).

### Added — AD compiled-memory certificate, saved-residual diagnostics, Pareto + checkify tooling (diagnostic/planning-only, no numerics-path change)

- `Simulation.ad_memory_preflight(...)` composes the static AD-memory planner,
  explainability, and mesh-intelligence reports (plus an optional saved-residual
  diagnostic) into one `ADMemoryPreflightReport` with actionable
  `ADMemoryActionHint`s. Does not run FDTD.
- `Simulation.ad_memory_compiled_certificate(...)` reads a caller-supplied
  compiled executable's `Compiled.memory_analysis()` once and fails closed into
  an `ADCompiledMemoryCertificate` bounded to one exact scope. The verdict is
  estimate-framed (`compiler_estimate_within_budget` /
  `compiler_estimate_exceeds_budget`, boolean `estimate_within_budget`): it is a
  JAX compiler **estimate**, not a runtime peak-memory guarantee — it excludes
  allocator fragmentation/scratch, and the fit recommendation reports the
  estimate's utilization of the target budget.
- AD saved-residual introspection: `inspect_ad_saved_residuals`,
  `diagnose_ad_saved_residuals`, `parse_saved_residual_line` parse JAX's
  saved-residual output into JSON artifacts (`ADResidualInspection`,
  `ADSavedResidualDiagnosticReport`, …). Read-only; not a runtime profile.
- Multi-objective sweep tooling: `pareto_front`, `pareto_mask`,
  `weighted_scalarization`, `epsilon_constraint_mask`,
  `select_epsilon_constrained` (+ `SweepResult.pareto_front`) — optimizer-agnostic
  numpy utilities returning JSON-serializable Pareto artifacts. `atol` is a
  strict-improvement tolerance (a strict partial order: a larger `atol` retains
  more points and never empties a non-empty front).
- Opt-in JAX invariant checks: `checkify_invariants` with `check_finite`,
  `check_positive`, `check_bounds`, `check_courant_number` wrap
  `jax.experimental.checkify` so RF/design invariants survive jit/grad/vmap/scan.

### Changed — `maximize_directivity` objective defaults (behaviour change)

- `maximize_directivity(...)` now defaults to **`log_ratio=True`** — the full,
  sign-correct quotient gradient `U'/U - P'/P` — instead of the legacy
  `-U/stop_gradient(P)` mode, which is wrong-sign for any degree of freedom that
  changes total radiated power (PEC/conductor topology, lossy/σ, magnitude-only
  dielectric reshape; GitHub #129). Pass `log_ratio=False` for the old behaviour.
  The loss is now `-(log U - log P)` (positive below ~11 dBi), not the old
  fixed-negative ratio.
- The directivity denominator `P_rad` is now integrated over the **full sphere**
  (matching `farfield.directivity()` / `antenna._total_radiated_power`) rather
  than the upper hemisphere, so the optimized quantity is the true directivity;
  the hemisphere-only integral inflated it (~+3 dB) for any radiator with back
  radiation. The shipped tap-paper beam-steering example is unaffected (it builds
  its own full-sphere `4π U/P_rad` objective and never called this function).

## [1.6.6] - 2026-06-24

Maintenance release: a behaviour-preserving internal refactor (extract the
waveguide mode solver into its own module — byte-identical, GPU-suite-confirmed)
plus repo hygiene (remove dangling internal-doc references from shipped code and
add a CI guard). **No user-visible behaviour or public-API change.** Per-release
GPU gate green: `pytest -m gpu` on the release commit = 187 passed / 62 skipped /
2 xfailed (VESSL gpu-rtx4090).

### Internal — refactor + repo hygiene (no user-visible behaviour change; PRs #218, #219)

- Extracted the rectangular-waveguide transverse mode solver and mode-profile
  linear algebra (10 pure-NumPy helpers) out of the 3364-line
  `rfx/sources/waveguide_port.py` into a new sibling module
  `rfx/sources/_waveguide_modes.py`, re-imported so every existing import path
  (`rfx.sources.waveguide_port.*`, the `rfx/eigenmode.py` private imports, the
  public re-exports) is unchanged (`waveguide_port.py` 3364 → 3090 LOC). A
  verbatim, behaviour-preserving move: the helpers' outputs are byte-identical
  pre/post (verified by output digest on both `main` and the branch), the
  `jax.grad` S-parameter tape is unaffected, and the GPU suite is unchanged
  (187 passed / 62 skipped / 2 xfailed). A new contract test
  (`tests/test_waveguide_modes_extraction_contract.py`) pins the re-export
  surface and the pure-NumPy (no-import-cycle) invariant. (PR #219)
- Removed 23 dangling references to gitignored internal docs
  (`docs/agent-memory/…`, `docs/research_notes/…`, `.claude/…`, `CLAUDE.md`) from
  shipped code and examples — public clones do not contain those paths — and
  repointed the two with public equivalents to `docs/agent/recipe-*.mdx`. Added a
  CI guard (`.github/workflows/lint.yml` `agent-docs-hygiene`) so the class cannot
  regress. Docstring / comment / warning-message text only; no behaviour change.
  (PR #218)

## [1.6.5] - 2026-06-19

Highlights: a validation-framework **reframe** — the `broad_e5_passed`
port-external-reference verdict now means magnitude envelope + an (approximately)
convention-free phase witness + an AD-vs-FD differentiability moat + a live-physics
anchor + committed-artifact enforcement (it was a magnitude-only checkmark) — plus
machine-readable per-port physics-set validation ceilings, several
documentation-honesty corrections, a public-docs surface narrowing, and MSL /
Floquet / non-uniform / distributed test hardening. This is mostly validation rigor
and honesty; the only user-visible runtime change is a per-port reported-Z0 sign
normalization (S-parameter-invariant).

### Changed — port-external-reference validation reframe (T0–T2.5; PRs #184, #190, #186, #187, #188, #189)

- `scripts/diagnostics/check_port_external_references.py`'s `broad_e5_passed`
  verdict now requires, beyond the magnitude envelope: a documented numeric
  breadth floor (≥4 cases, ≥2 mesh, ≥2 geometry/eps, a freq-span ratio, all cases
  pass) enforced for **every** family (was waveguide-only); an approximately
  convention-free TE10 propagation-phase witness (`tests/test_waveguide_phase_gate.py`);
  an **AD-vs-FD differentiability gate wired into the verdict** (a family whose
  extractor is numpy / not-traceable — e.g. coaxial — cannot pass); a live-physics
  anchor that runs a real `compute_waveguide_s_matrix` PEC-short / empty-guide check
  rather than only replaying a frozen JSON; physics-derived (measured-envelope)
  tolerances; and git-committed-artifact enforcement (`--require-committed`) so
  evidence living only in gitignored `.omx/` no longer counts. `missing_evidence`
  now gates the verdict. (T2.4 note: the planned `C·(k·dx)²` dispersion-tolerance
  model was falsified by the committed data and replaced with a measured envelope —
  a stop-and-redesign, not a tweak.)

### Added — per-port physics-set target ceilings + usage rules (PR #191)

- The port-external manifest and auditor now declare, per family, a
  machine-readable `target_ceiling` (the validation ceiling the port can physically
  reach) and `usage_rule`, from a controlled vocabulary. broad-E5 is **not** a
  universal goal: `rectangular_waveguide_port` = broad-E5 (achieved);
  `microstrip_line_port` = broad-E5 matched-regime only; `coaxial_port` = broad-E5
  pending a differentiable API; lumped/wire = E4 natural ceiling (single-cell feeds
  have no transmission-line oracle — "validated to ceiling", not a failure);
  `floquet_port` = broadside structural-partial; generalized-planar = unimplemented.
  Descriptive context emitted alongside the verdict — it does **not** change the
  pass/block gate logic.

### Changed — public documentation surface narrowed (PR #198)

- Public `/rfx/` docs were narrowed to maintained workflows + bounded support
  envelopes: the route inventory was trimmed, generated-API and agent-deploy-sync
  surfaces removed, and public wording rewritten around documented evidence
  envelopes (temporary surfaces such as SBP-SAT subgridding stay out of user docs).
  No library behaviour change.

### Fixed — documentation honesty (T0; PRs #192, #193)

- Corrected a stale claim that the coaxial family was *"the current validated"*
  method and *"the only family currently passing the clean-checkout port
  external-reference audit"* — true only before the rectangular-waveguide broad-E5
  evidence was committed (PR #181, v1.6.4). Coaxial evidence still lives in
  gitignored `.omx/`, so the auditor reports `coaxial_port` BLOCKED and
  `rectangular_waveguide_port` is the single passing family
  (`coaxial_port` → `broad_e5_demonstrated_evidence_uncommitted` across README,
  support matrix, port-selection guide, evidence-rule doc, manifest, CHANGELOG, and
  the reference-lane doc).
- The MSL thru-line openEMS smoke comparator
  (`compare_msl_thru_openems_reference.py`) no longer reports a bare `passed` when
  its |S11| channel is non-discriminating (on a matched line the reference |S11| ≈
  the tolerance, so a degenerate output would have "passed"); it now flags the S11
  channel informational and rests on transmission. Added a committed
  estimator-level test for `rfx.harminv` (synthetic known-frequency recovery,
  float32 robustness), and corrected the cv05 metric label (a Harminv-vs-Harminv
  resonance-frequency agreement, not an S11-vs-S11 match).

### Fixed — microstrip-line reported characteristic impedance sign (issue #140, PR #194)

- `compute_msl_s_matrix` now reports a positive `Re(Z0)` on **both** ports. A `-x`
  port previously reported a negative Z0 (it inherited the sign of the
  direction-aware closed-Ampère loop current), which also false-fired the |Z0|
  honesty guard at ~228% deviation. This is **S-parameter-invariant** — the reported
  Z0 never enters S11/S21 (those use the static analytic Hammerstad-Jensen Z0); the
  genuine ~20–27% Yee-staircase Z0 warning correctly remains. A new `@slow`
  thru-line test locks |Z0| length-invariance + the positive-sign behaviour. The
  earlier PR-#134 alarm (non-physical Z0 corrupting S-params, runaway passivity) was
  verified-and-refuted; **issue #140 closed**.

### Added — validation test coverage (PRs #195, #196)

- Floquet: an extractor-level AD-vs-FD **agreement** test for
  `compute_floquet_s_params` (the differentiability moat — `jax.grad` agrees with
  central finite-difference to <1e-2 at a fixed step; previously only a finiteness
  smoke existed).
- Non-uniform mesh: an analytic-gated NonUniformGrid accuracy test against a
  **graded-axis-dependent** mode — an air PEC cavity TM111 resonance (p=1, whose
  closed-form frequency moves with the graded *z* extent, unlike the existing
  `test_stage1_nu_physics_gate` TM110 p=0 gate, which is z-independent) reproduced to
  ~2.7% on a genuinely graded mesh. This gates the graded axis against a number it
  actually changes.

### Fixed — distributed tests on a single-GPU pod (issue #162, PR #197)

- The multi-GPU `tests/test_distributed.py` tests are now device-count-adaptive
  (shard across `min(4, jax.device_count())`) and skip cleanly when <2 devices are
  present, instead of failing a hardcoded `len(devices)==4` assert on a single-GPU
  pod (where the host-device-count sentinel does not add virtual devices). The
  distributed runner is verified equivalent to single-device by the committed tests
  (rel-err ≤1e-3) — this was test brittleness, not a runner defect. The GPU suite
  goes green because the multi-device tests now SKIP cleanly on the single-GPU pod;
  reliable multi-device CI coverage is environment-gated (a known follow-up — needs
  an isolated pytest lane). **issue #162 closed**.

## [1.6.4] - 2026-06-16

Highlights: the rectangular-waveguide-port **broad-E5 close** (committed
analytic-Airy band envelopes + an rfx-vs-Palace-FEM external comparison, with the
port-external-reference audit GREEN for `rectangular_waveguide_port` on a clean
checkout) and removal of the orphaned legacy 3-probe MSL extractor — on top of the
accumulated correctness, preflight, AD-tape, and validation-lane work since 1.6.3.

### Added — rectangular waveguide port broad-E5 evidence, committed (PR #181, 2026-06-16)

- `compute_waveguide_s_matrix(normalize='flux')` broad-E5 evidence now survives a
  clean checkout: five WR-band (WR-28/62/15/340/10, eps_r 2 & 4) analytic-Airy flux
  envelopes (20/20 cases, max |S| diff ≤ 0.0414) committed under
  `tests/fixtures/waveguide_broad_e5/`, plus an rfx-vs-Palace-FEM broad-E4 external
  comparison across the empty / PEC-short / dielectric-slab geometry axis (max |S|
  diff 0.0707, gate 0.10). Previously this evidence lived only in gitignored
  `.omx/` outputs, so `scripts/diagnostics/check_port_external_references.py`
  reported the family `blocked` on a clean checkout while the manifest claimed
  `broad_e5_passed`; the auditor now reports `rectangular_waveguide_port` passed.
  New gate `tests/test_waveguide_broad_e5_envelope_gates.py` re-derives both
  verdicts from the committed fixtures and mirrors the auditor's broad-E5/E4
  acceptance. R5 note: coarse Meep (res 3/4) gives a non-physical PEC-short
  |S11|>1, so the converged Palace high-order FEM reference is used for that
  geometry; rfx itself is exact (|S11|=1.0000).

### Removed — orphaned legacy 3-probe MSL extractor (2026-06-15)

- Deleted the pre-issue-#80 closed-form 3-probe MSL de-embedding helpers
  from `rfx/sources/msl_port.py`: `_solve_3probe`, `msl_forward_amplitude`,
  `compute_s21`, and the unused `_integrate_v` / `_integrate_i` line
  integrals.  They had **zero callers in `rfx/`** — the production MSL
  S-matrix path uses the closed Ampère-loop current (`msl_loop_current`,
  retained) plus the SVD N-probe wave decomposition (`extract_msl_nprobe`)
  — and were only kept alive by one unit test
  (`tests/test_msl_port.py::test_compute_s21_round_trip`), removed with its
  imports.  None were exported (no `__all__`; absent from top-level `rfx`),
  so the public surface and the api-reference inventory are unchanged.
  Closes architect-review item NEW-1.

### Fixed — normalize='flux' waveguide S-matrix joins the AD tape (issue #148, 2026-06-12)

- `extract_waveguide_s_matrix_flux` is now jnp-native end to end: the
  `np.array(flux_spectrum(...))` concretizations and the in-place numpy
  S-matrix assembly are gone, so
  `compute_waveguide_s_matrix(normalize='flux', eps_override=<traced>)`
  works under `jax.grad` (previously: `TracerArrayConversionError`) —
  design loops can optimize directly through the production-recommended
  power-flux extraction instead of the normalize=False-then-validate
  workaround.
- The rewrite adds double-where guards at the two genuine gradient
  singularities (sqrt of a zero power ratio at a perfect match/null,
  angle of a zero modal ratio); primal values are preserved exactly.
- Forward regression: S-matrix unchanged vs the numpy path within the
  float-reassociation envelope (measured max|diff| 1.1e-7 on the WR-90
  fixture). New CI gates in `tests/test_waveguide_flux_ad.py`
  (composition-level grad finite + central-FD agreement ≤5% + forward
  no-op-override equivalence); support matrix `ad_evidence` updated.

### Fixed — MSL N-probe extractor NaN gradient at tiny field scales (2026-06-12)

- `extract_msl_nprobe`'s β-refinement (`_estimate_beta`) produced `nan`
  gradients when the plane-integrated probe voltages were very small
  (|V| ~ 1e-14, measured on the density-PEC/Kottke forward path): the
  float32 residual curve over the β scan went numerically flat, the
  parabolic second-difference collapsed below its 1e-20 guard, and the
  **single-where** division guard leaked `0 * nan = nan` through the
  backward pass — the exact trap class the module's `_solve_q`
  custom-JVP comment documents, reintroduced by the lstsq rewrite.
  Forward values were always finite (the failure was invisible to
  value-level checks and to unit-scale AD tests — composition-level
  only). Fixed with the double-where idiom plus scale-normalizing the
  β-estimate input (`v/max|v|`; α/γ/Z0 keep absolute scale via the raw
  final lstsq). Found by the msl_stub G2 re-run (VESSL 369367242390:
  Adam grad=nan from iter 0 while the 17-point brute scan stayed
  finite). Regression-locked by
  `test_nprobe_grad_finite_and_scale_invariant_at_tiny_v` (fails on the
  old code; locks finiteness at scale 1e-14 + scale-invariance + FD
  match).

### Fixed — cv03 flux-region congruence (issue #160, 2026-06-12)

- `examples/crossval/03_straight_waveguide_flux.py`: the rfx flux monitors
  are now bounded to the same `2*wg_width` region the Meep `FluxRegion`
  measures, instead of the full y-plane (UPML padding included). The
  full-plane `flux_in` additionally integrated the line source's radiation
  cone — power that physically exits through the transverse absorber before
  `flux_out` — so the self-transmission read 0.913 against the [0.95, 1.05]
  gate with **no flux-normalization bug present**. Measured matrix
  (resolution 10/15/20): full-plane 0.913 / 0.986 / 0.958 (non-monotonic,
  not a convergence curve); bounded 0.974 / 1.011 / 0.997 — passing at every
  resolution including the recipe mesh. Truncation witness: bounded
  resolution-10 T(f_peak) = 0.977 at 3x run length. Gate unchanged.
  Falsifier matrix: `scripts/diagnostics/cv03_flux/sweep_t_deficit.py`.
- Second comparator defect (same script, surfaced by the lane's first real
  Meep execution): the rfx integration time was slaved to Meep's
  `stop_when_fields_decayed` wall clock — when Meep stopped at t=200 the
  rfx flux DFT was truncated mid-tail and read T=1.155. rfx now runs a
  fixed 400 a/c0 units (measured band: 0.9736 at 1x, 0.9772 at 3x).
  `until_decay=1e-5` was tried and rejected for this geometry: the point
  stopper triggers at ~2200 steps while the eps=12 guide's slow tail is
  still carrying flux (T=0.745) — point-field decay is not a
  flux-convergence witness here (filed as issue #169).
- Gate statistic re-specified to the **central-band mean** T (fcen ±
  0.15·df), tolerances unchanged at 1.0 ± 0.05 and cross-diff < 0.05.
  Measured first (sweep matrix + lane runs 27393931821/27394439174): at
  the recipe mesh — 11.5 cells/λ_eff at freq_max, below the preflight's
  own ≥20 floor for flux extraction — rfx's per-bin T(f) carries the
  preflight-documented ±5-10% coarse-mesh ripple while Meep's curve is
  smooth, so the old single-bin gate sampled at Meep's peak bin landed
  in ripple valleys (0.902 at f=0.1510) even at resolutions where the
  band-energy transmission is clean (band-mean 0.966 / 1.005 / 0.989 at
  resolution 10/15/20). The band mean is the physically meaningful
  energy-transmission estimator; peak-bin values remain printed for
  information.

### Fixed — preflight 2D false positive + unit-adaptive warning text (issue #166, 2026-06-12)

- **`absorber_overlap` no longer false-trips on the collapsed z axis in 2D
  modes.** The preflight thickness mirror assumed an absorber on every
  non-PEC/PMC/periodic axis, but 2D grids collapse z to a single cell with
  no absorber at all (`Grid` sets `pad_z = 0` and strips z from
  `cpml_axes`) — so every 2D source/probe, necessarily at z=0, warned
  "near/inside UPML region" (cv03: one line per line-source point, 20 lines
  of spam per preflight). Real x/y overlap in 2D and all 3D behaviour are
  regression-locked unchanged.
- **Scale-sensitive preflight messages now pick units adaptively**
  (`_fmt_len` / `_fmt_freq`): the mesh-resolution and absorber-placement
  warnings printed fixed mm/GHz, rendering optical-scale setups as
  `dx=0.000mm`, `lo=0.0mm`, and `freq_max=74950.00GHz` — values that read
  like bugs while being correct (cv03: 100nm, 2µm, 74.95THz). Remaining
  RF-lane messages (NTFF, MSL, ports) keep mm/GHz, which is correct at
  their scale, and can migrate incrementally.
- New gates in `tests/test_preflight_structured_and_guards.py` (2D-z no
  false positive, 2D-x/y still fires, formatter units, optical-scale
  mesh-warning text).

### Fixed — waveguide-port default source spectrum (issue #150, 2026-06-12)

- **`f0=None` now defaults to the center of the requested DFT band** instead
  of the unrelated `freq_max / 2`. The old fallback could land at or below
  the port mode's cutoff (canonical WR-90 toy: 6 GHz < fc_TE10 = 6.56 GHz),
  launching an evanescent near-cutoff crawl whose extracted S-parameters were
  physically meaningless and **grew with `n_steps`** (max column power
  20 → 57 → 114 over 600 → 2400 steps on the recorded #150 toy; post-fix
  1.107, identical at 4800 and 9600 steps). Explicit-`f0` setups unchanged.
- **New preflight guards** `port_source_below_cutoff` and
  `port_freqs_below_cutoff`: the resolved source center or any requested
  measurement bin at/below the excited mode's cutoff is now flagged loudly
  (below-cutoff bins also NaN gradients under `jax.grad`).
- `examples/inverse_design/differentiable_s11_design.py` setup corrected
  (issue #149): ports moved clear of the CPML, measurement band kept inside
  the TE20 contamination bound, preflight now runs visibly and aborts on
  issues. AD↔FD relative error after the cleanup: 2.8e-4.
- New gates: `tests/test_waveguide_port_spectrum_guard.py` (preflight codes,
  f0-omitted empty-guide transmission |S21|≈1, and the #150 toy's
  column-power growth signature locked at the measured envelope).

### Current main status (2026-06-10)

- **Recommended public lane remains uniform Cartesian Yee RF/FDTD.**
  Non-uniform mesh, distributed execution, Floquet/Bloch, guarded
  subgridding, and broad inverse-design workflows remain lane-scoped and must
  be described through their support/evidence envelopes.
- **All-port broad-E5 is still incomplete.**  The external-reference audit is
  intentionally blocked until lumped, wire, MSL, Floquet, generalized planar,
  and clean-checkout waveguide artifact tracking satisfy the manifest.  Do not
  turn one port-family promotion into a blanket S-parameter claim.

### Removed — dead multimode waveguide extractor (2026-06-11)

- Deleted the internal helper
  `rfx.sources.waveguide_port.extract_multimode_s_params_normalized`.  It
  was the would-be `normalize=True` multi-mode waveguide S-matrix extractor
  but was never wired into the public API: `compute_waveguide_s_matrix`
  raises for `normalize=True` with `n_modes > 1` (cross-mode channels hit a
  0/0 in the two-run normalization) and routes multi-mode work to
  `extract_multimode_s_matrix` or `extract_multimode_s_matrix_flux` instead.
  Verified zero callers across `rfx/ tests/ examples/ scripts/ docs/`
  (only its own definition plus two docstring cross-references, now
  repointed at `extract_waveguide_s_params_normalized`).  The function was
  not exported (no `__all__` entry, absent from `rfx/api/_sparams.py`
  imports), so the public surface is unchanged.

### Added — preflight, finite-result, and automation guards (2026-06-10)

- `Simulation.preflight()` and `preflight_sparameters()` now return coded
  `PreflightReport` / `PreflightIssue` records while preserving legacy
  list-of-string behaviour.  Automation can gate on `.errors`,
  `.warnings`, `.by_code(...)`, `.raise_for_failure()`, `.to_dict()`, and
  `.to_json()` instead of scraping warning text.
- `run()` now uses the consolidated preflight path with `skip_preflight=...`,
  while preserving hard failures for structurally impossible configurations
  such as UPML + refinement and Floquet + non-uniform z.
- `Result.assert_finite()`, run/forward non-finite warnings, S-matrix
  passivity guards, and optimizer NaN-gradient recovery now surface bad
  states before they silently contaminate inverse-design loops.  Sweeps run
  preflight once and avoid repeated per-case warning floods.
- Added PR/CI guard coverage for the preflight/guard suites and re-enabled the
  tree-wide ruff lint gate.

### Added — coaxial line reflection evidence envelope (2026-06-08)

- `Simulation.compute_coaxial_line_reflection(...)` is the coaxial
  transmission-line reflection path.  Its broad-E5 physics was *demonstrated*
  (analytic Γ envelope over short/open/matched plus resistive 25/100 Ω loads,
  two characteristic impedances and mesh-resolution cases, max |Γ| dev 0.037)
  plus an independent broad-E4 MEEP power-flux short/open comparison over
  4–12 GHz (max |S11| diff 0.063).  **The evidence artifacts live in gitignored
  `.omx/` and are not committed to the repo**, so
  `check_port_external_references.py` reports `coaxial_port` BLOCKED on a clean
  checkout; do not cite this path as `broad_e5_passed` until the artifacts are
  committed and the auditor returns PASSED.  (See the 2026-06-17 documentation
  honesty correction under [Unreleased].)
- `Simulation.compute_coaxial_s_matrix(...)` remains available for backward
  compatibility but is deprecated as the older single-plane V/I path; it is
  not the promoted coaxial claims surface.

### Added — waveguide S-matrix memory control (2026-06-09)

- `Simulation.compute_waveguide_s_matrix(checkpoint_segments=...)` now
  threads segmented checkpointing through the uniform waveguide extractors.
  Regression tests pin bit-identical forward S-matrices for
  `normalize=False`, `normalize=True`, and `normalize="flux"`, finite
  gradients through `eps_override`, rejection for non-divisor segment counts,
  and a loud `NotImplementedError` on non-uniform meshes.

### Fixed — public analysis and objective correctness (2026-06-08 to 2026-06-10)

- Finite-size `FluxMonitor` bookkeeping is regression-locked as
  machine-precision equivalent to summing the same full-plane integrand window
  in standing-wave-heavy fields.  The older "finite-size flux is less stable"
  caveat is superseded by a coverage distinction: finite monitors intentionally
  exclude cells outside their requested window.
- `maximize_directivity(..., log_ratio=True)` / the log-ratio directivity
  objective fixes the wrong-sign gradient for power-changing design variables
  by differentiating the full `log(U_target) - log(P_rad)` ratio instead of a
  partially stopped absolute-power proxy.
- The MSL S-parameter AD tests are restored to CPU CI, with checkpointed tape
  usage so the lane remains covered without requiring GPU-only memory budgets.

### Fixed — waveguide port extractor correctness (2026-04-22)

- **`_co_located_current_spectrum` sign flip.**  The H-derived DFT
  correction was `exp(-jω·dt/2)`; the correct sign from the leapfrog
  timing derivation is `exp(+jω·dt/2)`.  On a lossless empty WR-90 the
  mean `∠(Z_formula / Z_actual)` on a pure forward wave drops from
  −8° to −1°.  No public API change.
- **`_shift_modal_waves` direction-awareness.**  Added a
  `step_sign: int = 1` parameter (+1 for `+x/y/z` ports, −1 for
  `-x/y/z` ports).  Previously the shift formula silently applied the
  `+x` convention regardless of port direction, producing the wrong
  sign for negative-direction ports.  Two-run normalized S-matrices
  are unaffected (the shift cancels in the device/reference ratio),
  but any external caller of `extract_waveguide_port_waves` that
  captured the single-amplitude output for a negative-direction port
  now sees the physically correct sign.
- **`_compute_mode_impedance` below-cutoff sentinel.**  Returns
  `1e30` for TE / `0.0` for TM below cutoff, replacing `jnp.inf`.
  `inf × complex(r, 0)` generated a NaN in the imaginary component
  on most NumPy/JAX implementations and cascaded into NaN
  S-parameters on any multi-mode frequency sweep that straddled a
  higher-mode cutoff.  Regression-locked by
  `tests/test_waveguide_port_validation_battery.py::test_below_cutoff_z_mode_no_nan`.
- **`_compute_beta` Yee-discrete branch.**  Optional `dt, dx` kwargs;
  when both are positive the Yee 3-D dispersion relation is used
  instead of the analytic continuous form.  Now threaded through
  `_compute_mode_impedance`, `extract_waveguide_sparams`,
  `extract_waveguide_port_waves`, and `extract_waveguide_sparams_overlap`
  so Z and β stay internally consistent.

### Changed — defaults (2026-04-22)

- **BREAKING:** `Simulation.compute_waveguide_s_matrix()` now returns
  S-parameters referenced to each port's user-facing `x_position`
  (the plane passed to `add_waveguide_port`).  Previously the default
  was the internal `reference_x_m = x_position + ref_offset·dx`
  (three cells inward), which silently phase-shifted the returned
  complex S-matrix by `exp(-jβ·3·dx)` relative to the user's port
  plane.  Users who explicitly passed `reference_plane=` on the
  port entry see no change.  Magnitude (|S_ij|) is unaffected; the
  phase convention now matches what users get from Meep's
  `get_eigenmode_coefficients` at a monitor placed at the same
  absolute position, and what any analytic formula written against
  the port plane produces.
- **`Simulation(cpml_layers=8)` → `cpml_layers=16`.**  Waveguide-mode
  CPML back-reflection measured on an empty WR-90 scales as 11.7%
  (10 layers) → 4.2% (20) → 1.8% (40).  The previous default was
  tuned for free-space simulations and was inadequate for guided-
  mode absorption.  A parallel `kappa_max` sweep confirmed
  `kappa_max > 1` degrades guided absorption in the current CFS
  formulation, so `kappa_max = 1` is preserved.  Free-space
  simulations see a modest (but monotonic) improvement in peak
  CPML absorption from the thicker default.
- **CPML polynomial order 2 → 3** (`_cpml_profile(order=3)` default
  in `rfx/boundaries/cpml.py`).  Matches the Taflove & Hagness 3rd
  ed. §7.9 recommendation for guided-mode absorbers.

### Added

- **Preflight P2.8 — waveguide-port reference-plane sanity.**
  `Simulation.preflight()` now verifies each waveguide port's
  effective reference plane lies inside the domain, outside the
  CPML absorbing region, and does not intersect a geometry box.
  Raises `ValueError` for out-of-domain; emits `UserWarning` for
  CPML or device overlap with an actionable remediation message.
- **Validation battery** (`tests/test_waveguide_port_validation_battery.py`):
  nine tests locking physical-correctness invariants with Meep-class
  gates where achievable and explicitly-ratcheted gates where the
  extractor still has a known residual.  Supersedes the older loose
  `test_passivity_*` / `test_unitarity_*` gates in
  `tests/test_conservation_laws.py` (those remain as broad
  regression detectors with a loosened lower bound to reflect the
  real extractor accuracy ceiling, documented inline).
- **Two-port contract lock** (`tests/test_waveguide_twoport_contract_v1.py`):
  three tests fixing the v1 normalized two-port invariants (empty
  preservation, PEC-short strong reflection, reference-plane
  invariance on empty guide).
- **WR-90 crossval skeleton** (`examples/crossval/11_waveguide_port_wr90.py`):
  a diagnostic reporter against analytic Airy and (when present) a
  Meep reference JSON.  Not a regression gate — gates live in the
  battery above.
- **Diagnostic scripts** under `scripts/`:
  `waveguide_port_canonical_diagnostics.py` (before/after snapshot
  harness with `--json`), `isolate_extractor_vs_engine.py`
  (empty-guide V/I spillover diagnosis, scriptable CPML and
  kappa_max sweeps), and `slab_physical_diagnostics.py` (per-freq
  rfx vs analytic vs Meep with magnitude and phase breakdown).

### Known issues (carried from earlier sessions)

- `|S11|` at resonance nulls (e.g. dielectric-slab quarter-wave
  minimum) remains ~0.05–0.10 on the default grid; halving `dx`
  locally via `dx_profile=` cuts this by ~30%.  Meep at equivalent
  resolution shows a comparable ~0.05 floor, so this is a
  shared-FDTD discretization limit, not an rfx-specific bug.
- rfx vs Meep `∠S21` residual on the WR-90 crossval slab case
  (`examples/crossval/11_waveguide_port_wr90.py`) fits a linear
  `Δφ(rad) = slope·β + intercept` model to RMS 2.3° with
  slope ≈ −5.9 mm and intercept ≈ −57°
  (`scripts/phase_offset_beta_sweep.py`).  Applying this correction
  back to the slab rfx S21 reduces the RMS phase diff from 113° to
  2.3° — VERIFIED (`scripts/verify_phase_alignment.py`).

  Decomposition against physics:
  - `exp(−j(β_slab − β_empty)·L_slab)` (material-contrast phase that
    appears in two-run normalization through a dielectric slab):
    linear-in-β fit gives slope ≈ −1.95 mm, intercept ≈ −44°
    (range −56° to −69° over the band).
  - Measured: slope ≈ −5.87 mm, intercept ≈ −57°.
  - Physics explains ~1/3 of the slope and most of the intercept;
    a residual of **−3.9 mm slope + −13° intercept** remains
    unexplained.

  Experiment 1 (2026-04-22): rfx `dx` 1.0 mm → 0.5 mm, Meep unchanged.
  Result: slope −5.87 mm → −6.0 mm, intercept −57.3° → −58.2°
  (essentially identical). **Cell-snapping / Yee-discretization
  hypothesis FALSIFIED** — the residual does not scale with rfx mesh
  size.  Remaining candidates: (a) Meep's `get_eigenmode_coefficients`
  α⁺ is referenced to a different plane than rfx assumes (cell
  centre vs monitor plane); (b) implementation difference in how
  either code handles the E/H overlap at a material-discontinuity
  edge during the two-run device/reference pair.

  The same (slope, intercept) also does NOT transfer to the
  PEC-short case (`|S11|` RMS stays 104° → 103°), so the offset is
  slab-geometry-specific rather than a universal convention shift.
  Magnitude agreement (|S21|) remains within 3–5% across the band.
  Practical guidance: compare rfx per-geometry against analytic
  Airy (where rfx matches |S21| within ≈ 5% on the slab); do not
  expect bit-level phase agreement with an external Meep script
  that has its own monitor / source-pulse conventions.

### Do-not-repeat log (carry-over from diagnosis)

- Do **not** retune `kappa_max` above 1 in pursuit of better guided-
  mode absorption under the current CFS-CPML formulation — the
  effect is **negative** (sweep evidence 2026-04-22).  Use thicker
  CPML instead.
- Do **not** treat `max|Ez|` over the full simulation window as a
  source-directionality metric — it is dominated by CPML round-trip
  reflection and has led multiple sessions in a circle.  Use an
  early-time-windowed envelope; the regression-locked version lives
  in `test_source_directionality_early_time`.

---

## [1.6.3] - 2026-04-17

(reconstructed from commit log)

### Fixed

- **Periodic boundary + CPML allocation** (`#68`): `set_periodic_axes` was
  not honoured during CPML layer allocation, causing CPML to be placed on
  periodic faces.  Preflight now detects and rejects this configuration.
  (`fix(boundary): #68 honor set_periodic_axes in CPML allocation + preflight`)

### Added

- **`distributed=True` threaded through `optimize()` and
  `progressive_optimize()`** (`#69`): the `distributed` keyword introduced
  in v1.6.2 for `forward()` is now propagated to the higher-level
  optimisation entry points.
  (`feat(optimize): #69 thread distributed=True through optimize + progressive_optimize`)

---

## [1.6.2] - 2026-04-17

(reconstructed from commit log)

### Added

- **`Simulation.forward(distributed=True)` public API** (`#44`, Phase 3):
  opt-in multi-device execution via `distributed=True`.  Covers the full
  non-uniform runner: sharded grid metadata, ghost exchange, CPML on
  x-slabs, Debye/Lorentz ADE ordering contract, soft-PEC occupancy
  sharding, segmented remat + warmup + `design_mask` + `emit_ts`.
- **`progressive_optimize` multi-resolution orchestrator** (`#42`):
  `Simulation.progressive_optimize(...)` chains resolution levels with
  geometry transfer.  API demo added as crossval 08.
- **`design_mask` stop-gradient on non-design cells** (`#41`): cells
  outside the design region are hard-stopped so gradients cannot escape
  the design volume.
- **Non-uniform sentinel hardening** (`#45`): tracer-safe `dz_profile`,
  soft-PEC occupancy, and bit-identical CPML path verified against the
  uniform runner.

### Fixed

- **Multi-device grad NaN at rank-0 corners** (`#44` Phase 4): corner
  cells at rank-0 were not receiving a ghost exchange contribution,
  producing NaN gradients in distributed training runs.
- **`Simulation.__init__` host-coercion** (`#44`): closes NU sentinel #2 —
  non-uniform grid parameters are coerced to host arrays at construction
  time, preventing JIT-time shape errors.

---

## [1.6.1] - 2026-04-16

(reconstructed from commit log)

### Added

- **Preflight auto-run before `forward()` / `optimize()` /
  `topology_optimize()`** (`#66`): preflight is now invoked automatically
  by all three execution entry points; pass `skip_preflight=True` to
  suppress for benchmarking.
- **`optimize()` routed through `sim.forward()`** (`#64`): optimizer now
  uses the single differentiable forward path, removing a separate
  code branch and unifying the differentiable-path surface.
- **Simulation-time breakdown utility** (`#58`): `Simulation.profile()`
  returns a per-phase timing breakdown (CPML init, Yee loop, probe
  accumulation, post-processing).
- **Patch antenna ground-plane size sweep** (`#59`): parametric test
  verifying that ground-plane size does not affect resonance frequency
  within the validated range.

### Fixed

- **Preflight: PEC inside CPML region** (`#61`): raises `ValueError` when
  any PEC geometry box overlaps the CPML absorbing region.
- **Preflight: Taflove dispersion check** replaces the earlier ratio
  heuristic with the exact Yee dispersion criterion from Taflove &
  Hagness Ch. 4.
- **Preflight: probe-in-PEC check**; aspect-ratio threshold tightened to
  2.0:1.
- **Preflight: Courant asymmetry warning** for non-uniform grids with
  per-axis `ν` values.
- **Preflight: NU cell aspect-ratio warning** at > 2.5:1.
- **Decimated Harminv** fixes F3 post-processing OOM on fine-mesh
  convergence runs.

---

## [1.6.0] - 2026-04-16

(reconstructed from commit log)

### Added

- **Memory-efficient inverse design on non-uniform mesh** (`#35`, `#36`):
  segmented scan + remat path; `estimate_ad_memory()` gains
  `ad_segmented_gb` field (`#39`).
- **Non-uniform runner feature parity + inverse-design unblock** (`#34`):
  NU runner now supports all source/boundary types available in the
  uniform runner relevant to inverse design.
- **Per-cell `dx`/`dy` profile support** for non-uniform grids.
- **`n_warmup` stop-gradient split** (`#40`, `#56`): warmup steps are
  detached from the AD tape, preventing spurious gradients from the
  initial transient.
- **Streaming multi-frequency NTFF sweep** (`#43`, `#55`): a single
  forward pass accumulates DFT data at multiple frequencies without
  storing full time-series.
- **3D structure + far-field visualisation API** (`#38`, `#54`):
  `Simulation.visualize_structure()` and `visualize_far_field()`.
- **`minimize_s11_at_freq` objective** (`#50`, `#52`): single-frequency
  S11 proxy usable directly inside `forward()`.
- **2-port wire-port S-matrix with passive loads and direction** (`#34`
  area): `add_wire_port` supports two-port extraction with explicit
  termination impedance and `+`/`-` direction.

### Fixed

- **Physics-based resolution thresholds in preflight** (`#37`, `#53`):
  replaces heuristic cell-count thresholds with wavelength/skin-depth
  criteria.
- **Thin-PEC on non-uniform mesh** (`#48`, `#51`): rasterisation fix +
  preflight warning + mesh-aligned patch visualisation.
- **`excite=False` guards + `forward()` profile check + preflight**:
  sources with `excite=False` no longer contribute to the excitation
  sum used by normalisation.
- **`NameError` `base_materials` in differentiable forward path**.

---

Earlier releases (v1.0.0–v1.3.0) predate this changelog's version sections; see git tags.
