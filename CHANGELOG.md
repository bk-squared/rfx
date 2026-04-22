# Changelog

All notable changes to `rfx-fdtd` that affect user-visible behaviour are
recorded here. Dates follow local (KST) convention. Version bumps follow
SemVer — **BREAKING** entries are flagged in upper-case.

## [Unreleased]

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
