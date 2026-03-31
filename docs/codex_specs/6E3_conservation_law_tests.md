# Codex Spec 6E3: Conservation Law Validation Tests (Tier 2)

## Goal
Create a test suite that validates rfx against physics invariants that are
independent of any external tool. These are the strongest possible tests
because they depend only on Maxwell's equations, not on Meep or OpenEMS.

## Context
Current rfx validation relies on:
- Analytical solutions (Tier 1) — cavity, Fresnel, velocity ✓
- Cross-validation vs Meep/OpenEMS (Tier 3) — bakes in their errors

Missing: **Tier 2 conservation law tests** that catch errors without
referencing any other simulator.

## Deliverable
`tests/test_conservation_laws.py` with 5 tests:

### Test 1: `test_passivity_two_port_empty_waveguide`
For a lossless empty waveguide with two ports:
- `Σ_i |S_ij(f)|² ≤ 1.0` at every frequency point for each column j
- This is energy conservation: power out ≤ power in

Setup:
- Domain (0.12, 0.04, 0.02), CPML on x, PEC on yz
- Two waveguide ports: +x at 0.01, -x at 0.09
- TE10 mode, freqs = linspace(4.5e9, 8e9, 20)
- Use `compute_waveguide_s_matrix(num_periods=40)` for adequate convergence

Assertion:
- `max(Σ_i |S_ij|²) < 1.05` (allowing 5% numerical margin)
- Print actual values for visibility

### Test 2: `test_unitarity_lossless_waveguide`
For a lossless system, total scattered power equals incident power:
- `Σ_i |S_ij(f)|² ≈ 1.0` (not just ≤, but approximately equal)

Same setup as Test 1 but with a dielectric obstacle:
- Box((0.04, 0.0, 0.0), (0.06, 0.04, 0.02)) with eps_r=4
- All materials lossless (sigma=0)

Assertion:
- `0.8 < mean(Σ_i |S_ij|²) < 1.05` at above-cutoff frequencies
- This validates that power is neither created nor excessively lost

### Test 3: `test_reciprocity_asymmetric_structure`
Reciprocity (S_ij = S_ji) must hold for any linear passive structure,
even asymmetric ones. Use an asymmetric obstacle to make this non-trivial.

Setup:
- Same waveguide as above
- Asymmetric dielectric: Box((0.03, 0.0, 0.0), (0.05, 0.02, 0.02)) eps_r=6
  (only covers half the y-cross-section, breaks geometric symmetry)

Assertion:
- `mean(||S21| - |S12|| / max(|S21|, |S12|)) < 0.05` (5% relative)
- Print S21 and S12 spectra for visibility

### Test 4: `test_mesh_convergence_s21`
S-parameters must converge as mesh is refined. This establishes the "true"
answer without any external tool.

Setup:
- Waveguide with dielectric slab, two ports
- Run at 3 resolutions: dx=3mm, 2mm, 1mm
- Extract |S21| at a single frequency (6 GHz, above cutoff)

Assertion:
- |S21(2mm) - S21(1mm)| < |S21(3mm) - S21(2mm)| (convergence is monotonic)
- |S21(2mm) - S21(1mm)| < 0.05 (converged to within 5%)
- Print all three values

### Test 5: `test_causality_no_signal_before_source`
No field should appear at a distant probe before the wavefront could
physically arrive. This validates basic FDTD causality.

Setup:
- Long domain (0.20, 0.04, 0.02), CPML on x
- TFSF source at x_lo, Ez polarization
- Probe at far end (x = 0.18)
- Compute arrival time: t_min = distance / c

Assertion:
- For all steps where t < t_min - 2*dt: |Ez(probe)| < 1e-10
- After t_min: signal appears (|Ez| > 1e-6 at some point)

## Constraints
- Tests 1-3: may be slow due to `compute_waveguide_s_matrix` (up to 300s each)
- Test 4: runs 3 resolutions, may be 180s total
- Test 5: should be fast (< 30s)
- Do NOT modify any source files — only create the test file
- Use existing high-level API (`Simulation`, `compute_waveguide_s_matrix`)
  for tests 1-3, and manual loop for test 5

## Verification
Run: `pytest -xvs tests/test_conservation_laws.py`
All 5 tests must pass.
