# 2026-03-28: Stage 2 — TFSF Source + Enhanced Tests Complete

## What was done

### TFSF Plane-Wave Source (`rfx/sources/tfsf.py`)
- 1D auxiliary FDTD with CPML for incident field generation
- Split into `update_tfsf_1d_h` / `update_tfsf_1d_e` for correct leapfrog interleaving with 3D grid
- `apply_tfsf_e` / `apply_tfsf_h` for 3D boundary corrections
- Normal +x incidence, Ez polarization

### Periodic BC (`rfx/core/yee.py`)
- `periodic: tuple = (False, False, False)` parameter on `update_h`/`update_e`
- Uses `jnp.roll` for periodic axes, zero-padded shifts for non-periodic
- `static_argnums=(4,)` ensures JIT retraces per periodic configuration

### Per-axis CPML (`rfx/boundaries/cpml.py`)
- `axes: str = "xyz"` parameter on `apply_cpml_e`/`apply_cpml_h`
- Enables CPML on x only while y/z use periodic BC (needed for TFSF plane wave)

### Enhanced Test Suite (29 tests, all passing)
- Fresnel normal incidence via TFSF: 3.4% error
- Fresnel oblique TE 30° via effective-eps: <10% error
- Two-port reciprocity (S21 ~ S12)
- CPML grazing incidence (<-20 dB)
- Mesh convergence (2nd-order verified)
- Late-time stability (5000 steps, <5% drift)
- Dielectric cavity resonance (<0.5% error)
- Meep + openEMS cross-validation (3 tests)
- Unit tests: grid, geometry, cavity, CPML, S-param (17 tests)

### Benchmark (CPU, `jax.lax.fori_loop`)
- 27x speedup at small grids (1.3k cells), 6x at 133k cells vs NumPy
- Python-loop JIT slower than NumPy on CPU (dispatch overhead dominates)
- GPU expected to give much larger gains

## Bugs found and fixed (do not repeat)
1. **1D FDTD H update sign**: Must be `h += (dt/MU_0) * de` (positive). The 3D code gets this right via double negation in curl structure, but 1D must use `+` explicitly. Wrong sign causes exponential blowup.
2. **TFSF leapfrog interleaving**: Must split 1D update into H and E halves interleaved with 3D updates. Single `update_tfsf_1d()` call causes wrong time-level references at corrections.
3. **Dielectric at TFSF boundary**: TFSF E/H corrections assume vacuum. Dielectric extending to or past x_lo/x_hi causes massive spurious scattered field (R=6.34 instead of 0.33).
4. **Back-face reflection**: For finite dielectric slab, simulation must end before round-trip reflection from far edge arrives at probe. Use timing calculation: `t_safe = t_front + 2*slab_thick/(C0/sqrt(eps_r))`.

## Files changed
- `rfx/sources/tfsf.py` (new, then sign fix)
- `rfx/core/yee.py` (periodic BC)
- `rfx/boundaries/cpml.py` (per-axis filtering)
- `tests/test_physics.py` (7 enhanced physics tests)
- `examples/benchmark_jax_vs_numpy.py` (fori_loop benchmark)

## Commits
- `5ed1f16` Add enhanced physics tests and JAX vs NumPy benchmark
- `ae7194d` Add periodic BC, per-axis CPML, and TFSF-based Fresnel tests

## Next steps (priority order)
1. **Waveguide port** — modal excitation/extraction for rectangular waveguide, needed for filter/antenna S-parameters
2. **DFT probes** — frequency-domain field monitors (running DFT accumulation during timestepping)
3. **Lossy conductors** — finite conductivity σ in materials, surface impedance BC for thin conductors
4. **Dispersive materials** — Debye/Lorentz/Drude models via auxiliary differential equation (ADE)
5. **Stage 3: Differentiation MVP** — `jax.checkpoint` + reverse-mode AD, custom VJP, gradient validation
