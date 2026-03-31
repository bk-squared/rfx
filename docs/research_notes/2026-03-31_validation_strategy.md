# 2026-03-31: Validation Strategy — Beyond Cross-Validation

## Problem
Using Meep/OpenEMS as validation references bakes their errors into our test
suite. If both tools agree at 0.97 (true answer 1.0), a cross-validation test
that passes at 1% tolerance actually masks a 3% systematic error in both.

Function-level tests each tolerate 3-5% error. Through 6 chained stages
(mu_r → CPML → modal extraction → DFT → de-embedding → S-param), compound
error can reach 10-15%.

## Solution: 3-Tier Validation Hierarchy

### Tier 1: Analytical Ground Truth
No dependence on any external tool. These are the primary accuracy gates.
- Cavity resonance vs formula
- Fresnel reflection vs `R = (η₂-η₁)/(η₂+η₁)`
- Phase velocity vs `v = c/√(μ_r·ε_r)`
- Cutoff frequency vs `f_c = c/(2a)`
- Waveguide impedance vs `Z_TE = ωμ/β`

### Tier 2: Conservation Laws (Physics Invariants)
Tool-independent physical constraints that any correct simulator must satisfy.
- **Passivity**: `Σ|S_ij|² ≤ 1` at every frequency (lossless system)
- **Unitarity**: `Σ|S_ij|² = 1` for lossless system (strict energy conservation)
- **Reciprocity**: `|S_ij| = |S_ji|` for linear passive media
- **Convergence**: mesh refinement → S-params converge to a stable value
- **Causality**: no signal before source onset at probe location

### Tier 3: Cross-Validation (Report Only)
Meep/OpenEMS results are logged for human review but do NOT gate pass/fail.
- rfx vs Meep cavity resonance
- rfx vs Meep waveguide S21 (report, not assert)
- rfx vs openEMS cavity resonance

## Compound Error Budget
Final S-parameter accuracy is judged end-to-end, not per function:
- Straight waveguide: `|S21| = 1.0 ± 0.03` via passivity + convergence
- With obstacle: `Σ|S|² ≤ 1.02` (allowing 2% numerical margin)
- Reciprocity: `|S_ij - S_ji| / |S_ij| < 3%` for all port pairs

## Action Items
1. Implement Tier 2 conservation law test suite
2. Add mesh convergence test
3. Convert existing cross-val tests to report-only
4. Normalization run (6E1) should fix passivity violation
