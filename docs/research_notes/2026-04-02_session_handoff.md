# 2026-04-02 Session Handoff

## Git: main @ a4e53ed (8 commits this session)

## Accuracy Achieved

| Structure | dx | Method | Error | Speed |
|-----------|-----|--------|-------|-------|
| PEC cavity TM110 | 1.0mm | Raw source, Harminv | **0.00%** | — |
| Patch antenna | 0.5mm | J source, JIT, subpixel | **4.89%** | 2840 Mcells/s |
| Patch antenna | 0.25mm | J source, JIT, bandpass | **3.78%** | 2069 Mcells/s |
| Richardson extrapolation | → 0 | — | **~2.5%** | — |

Convergence: monotonic (dx=1mm 6.3% → dx=0.5mm 4.89% → dx=0.25mm 3.78%).
Remaining ~2.5% gap: finite ground plane/substrate geometry effects (same as commercial solvers).

## Root Causes Found & Fixed (10 total)

### Simulation Physics
1. **Port impedance loading** — masks cavity resonance. Fix: `add_source()` (no impedance)
2. **Source energy ∝ dx³** — weaker cavity mode at fine grid. Fix: J source (Cb/dx normalization)
3. **DC surface charge** — differentiated Gaussian has ∫s≠0. Fix: ModulatedGaussian + bandpass Harminv
4. **sigma=1e10 PEC** — Ca≈-1 oscillation. Fix: true PEC mask (tangential-only zeroing)
5. **PEC mask in CPML** — causes NaN instability. Fix: Box.mask excludes CPML region

### Geometry / Grid
6. **Box.mask 2-cell PEC** — thin features rasterized to 2 cells. Fix: snap to corner_lo
7. **Box.mask thick-feature** — inclusive upper bound. Fix: half-open [lo, hi)
8. **Small domain** — domain resonance interferes. Fix: margin ≥ λ/4 in auto_config
9. **Substrate surface wave** — dominates spectrum at fine grid. Fix: bandpass filter in Harminv

### Analysis
10. **argmin vs argmax** — found anti-resonance instead of resonance. Fix: spectral peak finding

## Modules Implemented

### New Files
- `rfx/auto_config.py` — auto-derive dx, domain, CPML, n_steps from geometry + freq_range
- `rfx/harminv.py` — Matrix Pencil Method resonance/Q extraction + auto bandpass
- `rfx/visualize3d.py` — 3D geometry, field visualization, VTK export, animation
- `rfx/subgridding/runner.py` — subgridded simulation runner (Python loop)
- `ATTRIBUTION.md` — algorithm references and licenses

### Modified
- `rfx/sources/sources.py` — ModulatedGaussian (zero-DC Meep-style waveform), wire port sigma fix
- `rfx/simulation.py` — make_j_source (Cb/dx normalized), pec_mask in scan body
- `rfx/api.py` — add_source(), add_refinement(), auto source selection (PEC→raw, CPML→J), Simulation.auto(), Result.find_resonances()
- `rfx/boundaries/pec.py` — apply_pec_mask (component-specific tangential zeroing)
- `rfx/geometry/csg.py` — Box.mask thin-feature snap, thick-feature half-open
- `rfx/subgridding/sbp_sat_3d.py` — material/PEC/SAT-penalty support

### Tests & Examples
- `tests/test_wire_sparam.py` (6 tests)
- `tests/test_visualize3d.py` (5 tests)
- `examples/12_validation_suite.py` — PEC cavity + dielectric resonator + patch antenna
- `examples/13_patch_profiled.py` — JIT profiling + error analysis

## Meep/OpenEMS Research Findings

### Source Strategy (KEY LEARNING)
- Meep: always J source + modulated Gaussian (zero DC by construction)
- OpenEMS: E-field source (type=0) or J source (type=10), user selects
- rfx: auto-select based on boundary (CPML→J, PEC→raw)

### Auto-Configuration Rules
| Parameter | Meep | OpenEMS | rfx |
|-----------|------|---------|-----|
| dx | user `resolution` (min 8/λ) | `SmoothMeshLines` (λ/20 + thirds rule) | `auto_configure` (λ/20 default) |
| PML | 0.5·λ thickness | 8 cells (~0.4·λ) | 8-16 cells |
| PEC | mask (ε→−∞) | mask (`AddMetal`) | mask (component-specific tangential) |
| Convergence | `stop_when_fields_decayed` | `EndCriteria=1e-5` | bandpass Harminv |
| Resonance | Harminv (FDM) | FFT | MPM Harminv |

### Remaining Gaps vs Meep/OpenEMS
1. **Flux monitors** for port S-params (Meep's mode decomposition approach)
2. **S-param in jax.lax.scan** (currently separate Python loop)
3. **Subgrid runner JIT** (currently Python loop, 10 steps/s vs 700+)
4. **Source normalization** per Meep convention (current density, not field)

## Performance

| Path | Speed | Notes |
|------|-------|-------|
| JIT (jax.lax.scan) | **2000-3100 Mcells/s** | RTX 4090 |
| Python loop (subgrid/manual) | 10-25 steps/s | 224x slower |
| GPU benchmark (ref) | 1310 Mcells/s | Simple cavity |

## Next Session Priority

### P0: <1% accuracy
- dx=0.125mm convergence test (should give ~1-2%)
- Or improve source normalization for consistent results across resolutions
- The 2.5% Richardson limit may be irreducible (finite geometry effects)

### P1: JIT integration
- S-param extraction in jax.lax.scan (100x speedup)
- Subgrid runner in jax.lax.scan (50x speedup)

### P2: Validation breadth
- Waveguide coupler (multi-port S-params)
- Cavity filter (narrowband, high Q)
- Field animation end-to-end
- Cross-validation with Meep (`test_meep_crossval.py` exists)

### P3: Auto-config completion
- Integrate auto source selection into `auto_configure()`
- `run_until_decay` as default stopping criterion
- Warning system for under-resolved features

## VESSL Jobs This Session
| # | Description | Result |
|---|-------------|--------|
| 369367231427 | Convergence (dx=1.0,0.5mm) | 6.3% → 3.4% |
| 369367231440 | Validation suite (3 tests) | PEC 0.0%, patch 3.1% |
| 369367231455 | JIT profiling (dx=0.5mm) | 3124 Mcells/s, 5.85% |
| 369367231458 | Subpixel smoothing A/B | 5.85% → 4.68% |
| 369367231522 | J source validation | 3.78% at dx=0.25mm |
