# Changelog

All notable changes to rfx are documented here.

---

## v1.0.0 (2026-04-02)

### New Features

- **PyPI publication** as `rfx-fdtd` -- package is now installable from PyPI
  while keeping the Python import name `rfx`.
- **SBP-SAT subgridding** with JIT performance via `jax.lax.scan` -- provably
  stable local mesh refinement with impedance coupling at coarse/fine
  interfaces.
- **Lumped RLC elements** (series and parallel) via auxiliary differential
  equation (ADE) update -- model discrete components in the FDTD grid.
- **Via/through-hole geometry helper** (`rfx.geometry.Via`) -- cylindrical
  plated-through-hole primitive for PCB simulations.
- **Curved patch geometry** (`rfx.geometry.CurvedPatch`) with staircase
  approximation for conformal antenna structures.
- **Field animation export** (MP4/GIF) via `rfx.animation` -- render
  time-domain field evolution directly from simulation results.
- **Oblique TFSF** with 2D auxiliary grid -- total-field/scattered-field
  plane-wave injection at arbitrary incidence angles (leakage < 1%).
- **Improved PML defaults** -- CFS-CPML with `kappa_max=5.0`, default
  thickness `0.4 * lambda`, reflectivity < -40 dB across the band.
- **Source auto-selection** in `auto_configure()` -- automatically chooses
  between Gaussian pulse and modulated Gaussian based on bandwidth and
  port type.
- **Lumped port S11 wave decomposition fix** -- correct V/I sampling order
  (sample before source injection) for accurate port impedance extraction.

### Improvements

- **5-line minimal workflow** via `Simulation.auto()` -- patch antenna from
  `freq_max` and `domain` in five lines of code.
- **Centralized TFSF 2D dispatch** -- `isinstance`-based routing replaces
  scattered conditional logic; dead oblique 1D code removed.
- **Dead code cleanup** in TFSF module after 2D auxiliary grid rewrite.
- **Reorganized examples** -- 8 clean scripts with visualization covering
  cavity resonance, waveguide S-params, inverse design, patch antenna,
  materials gallery, far-field radiation, auto-configure, and GPU benchmark.
- **CI timeout** increased to 45 min to accommodate full SBP-SAT test suite.

### Breaking Changes

- **PML default thickness increased** -- simulations may use slightly more
  memory due to thicker CPML padding (0.4 * lambda vs previous default).
- **`auto_configure()` returns additional fields** -- `source_type` and
  `source_info` are now included in the configuration dict.
- **Lumped port S-parameter extraction order changed** -- probe sampling now
  occurs before source injection within each time step. User code that
  relied on the previous (incorrect) ordering may see different S11 values.

---

## v0.1.0 (2026-03-30)

Initial public release.

### Core Simulator

- 3D and 2D (TE/TM) Yee FDTD with CFS-CPML absorbing boundaries.
- JAX JIT compilation -- ~3,000 Mcells/s on RTX 4090.
- `jax.grad` autodiff through full time-stepping with checkpointed
  reverse-mode AD (O(sqrt(n)) memory).
- Non-uniform Yee mesh with automatic cell-size selection.
- Auto-configuration from `freq_max` (cell size, time step, CPML, source).

### Sources and Ports

- Gaussian pulse and modulated Gaussian waveforms.
- TFSF plane-wave source (normal incidence).
- Wire port and lumped port with S-parameter extraction.
- Waveguide port with overlap-integral modal S-matrix.
- Multi-port N-port assembly and per-axis boundary control.

### Post-Processing

- Harminv resonance extraction (Matrix Pencil Method).
- DFT frequency-domain probes.
- Near-to-far-field transform, radiation patterns, and RCS pipeline.
- Field decay convergence criterion.
- S-parameter conservation law validation.

### Materials

- 11 built-in RF materials (FR4, Rogers4003C, alumina, silicon, copper, PEC,
  and more).
- Custom materials with arbitrary `eps_r`, `sigma`, `mu_r`.
- Debye dispersive model (water, biological tissue).
- Lorentz/Drude dispersive model (metals, plasmas).
- Anisotropic subpixel smoothing for dielectric interfaces.

### Optimization

- Inverse design with Adam optimizer and `jax.grad`.
- Design region permittivity parameterization.
- Convergence tests for optimizer stability.

### Validation

- Comprehensive three-way cross-validation against Meep and OpenEMS.
- 260+ tests, 0.000% deviation on analytical benchmarks.
- Patch antenna < 1% error vs commercial solvers.
- Waveguide TE101 0.98% error.
- Generalized validation across 4 antenna designs.

### Infrastructure

- GitHub Actions CI with automated test suite.
- MIT license.
- 8 example scripts with visualization.
- Full documentation suite (9 guide pages).
