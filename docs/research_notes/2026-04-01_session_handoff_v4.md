# 2026-04-01 Session Handoff v4

## What was done

### Features Implemented
1. **WirePort S-param extraction** — V/I DFT probes, extract_s_matrix_wire, API integration
2. **True PEC geometry mask** — apply_pec_mask with component-specific zeroing (tangential only)
3. **3D visualization** — plot_geometry_3d, plot_field_3d, save_field_vtk, save_screenshot
4. **SBP-SAT subgridding integration** — materials, PEC mask, CPML, sources, probes via add_refinement() API
5. **Subgridding runner** — Python-loop runner with progress logging
6. **Wire port sigma fix** — 1/(Z0*dx*N) → N/(Z0*dx) for series resistors

### Bug Fixes
- **PEC mask component-specific zeroing**: apply_pec_mask now only zeros tangential E, preserving normal E at thin PEC surfaces. Uses neighbor analysis: `mask_ez = pec(i,j,k) AND (pec(i,j,k-1) OR pec(i,j,k+1))`
- **Spectral peak finding**: Cavity probe resonance is a spectral PEAK (cavity amplifies at resonance), not MIN. Previous code used argmin → found anti-resonance.
- **Wire port impedance**: Per-cell sigma = N/(Z0*dx), not 1/(Z0*dx*N)
- **PEC mask port exclusion**: Wire/lumped port cells excluded from PEC mask

## Root Cause Analysis: Patch Antenna Frequency Error

### Issue chain identified through systematic testing:

1. **Domain too small** (original margin=15mm = 0.12λ)
   - Domain resonance (~1.3 GHz) dominates over patch resonance (2.4 GHz)
   - Fix: margin ≥ λ/4 = 31mm minimum, ideally λ/2 = 62mm
   - VESSL #369367231386 confirmed: even with λ/4 margin, results improve

2. **PEC thickness couples to dx**
   - Ground PEC Box height = dx → different effective substrates at each resolution
   - dx=1mm: ground eats 62% of 1.6mm substrate (only 0.6mm left)
   - dx=0.5mm: ground eats 31% (1.1mm left)
   - Non-convergent behavior: finer grid gives worse results
   - Fix: PEC thickness should be independent of dx, OR use subgridding

3. **Insufficient substrate resolution** (without subgridding)
   - h=1.6mm / dx=0.5mm = 3.2 cells → TM010 mode not properly resolved
   - h=1.6mm / dx_f=0.25mm = 6.4 cells → adequate with subgridding

4. **Insufficient simulation time**
   - dt scales with dx → finer grid = shorter T_sim for same n_steps
   - Patch cavity Q ~ 10-20 (FR4) → need T_sim ≥ 10ns for ring-down
   - dx=0.25mm: dt=4.8e-13 → need ~21K steps for 10ns

5. **SBP-SAT coupling creates reflections** (partially fixed)
   - Original: hard weighted-average replacement → significant reflections
   - Fixed: SAT penalty corrections (additive) → reduced reflections
   - Still needs: tuning of penalty coefficients, possibly H-field coupling

### Convergence test results (VESSL #369367231386, T_sim=10ns):
| dx | h/dx | Peak | Error |
|----|------|------|-------|
| 1.0mm | 2 | 1.911 GHz | 20.4% |
| 0.5mm | 3 | 1.361 GHz | 43.3% |

Non-convergent due to PEC thickness coupling. Both results unreliable.

### Subgridding result (VESSL #369367231354, coarse 1.5mm, fine 0.25mm):
- 1.73 GHz (with old hard coupling) → dominated by coupling reflections
- 6 substrate cells at fine resolution

## Files Changed
- `rfx/boundaries/pec.py` — apply_pec_mask with component-specific tangential zeroing
- `rfx/probes/probes.py` — wire port V/I, S-param extraction
- `rfx/sources/sources.py` — wire port sigma fix
- `rfx/api.py` — PEC mask, add_refinement(), _run_subgridded(), wire port integration
- `rfx/simulation.py` — pec_mask parameter in run/run_until_decay
- `rfx/subgridding/sbp_sat_3d.py` — material/PEC support, SAT penalty coupling
- `rfx/subgridding/runner.py` — NEW: subgridded simulation runner with CPML+sources+probes
- `rfx/visualize3d.py` — NEW: 3D visualization module
- `rfx/__init__.py` — exports

## Test Status
- 247 tests passed (before subgridding changes; need to re-run full suite)
- New tests: test_wire_sparam.py (6), test_visualize3d.py (5)

## Priority Next Steps

### P0: Fix patch antenna convergence
1. **Decouple PEC thickness from dx**: Use a fixed physical thickness (e.g., 0.1mm) or auto-set to min(dx, h/10)
2. **Increase domain**: margin ≥ λ/4 with proper CPML (≥ λ/20 thick)
3. **Run convergence study** with fixed PEC, large domain, sufficient T_sim

### P1: Optimize SBP-SAT coupling
1. Tune SAT penalty coefficients (alpha_c, alpha_f) for minimal reflection
2. Add H-field interpolation at interface
3. Validate: plane wave through interface should have < 1% reflection

### P2: JIT-compile subgridded runner
- Current Python loop: ~13 steps/s on CPU, ~19 steps/s on GPU
- JIT via jax.lax.scan: expected ~100x speedup
- Required for practical antenna simulation (10K+ steps)

### P3: Calibrated S11 for antenna ports
- Wave decomposition fails for thin substrates (V >> Z0*I)
- Options: coaxial port model, de-embedding, or time-domain matched-load calibration

## Active VESSL Jobs
- All completed. Recent: #369367231386 (convergence), #369367231385 (convergence v3)

## Git: main (uncommitted changes — need to commit this session's work)
