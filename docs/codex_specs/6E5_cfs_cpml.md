# Codex Spec 6E5: Complex Frequency Shifted CPML (CFS-CPML)

## Goal
Upgrade the existing CPML to CFS-CPML for better absorption of evanescent
and low-frequency fields, especially important for waveguide simulations
near cutoff.

## Background
Standard CPML stretching function: `s(ω) = 1 + σ/(jω)`
CFS-CPML stretching: `s(ω) = κ + σ/(α + jω)`

The `α` parameter (real, positive) dramatically improves absorption of:
- Evanescent fields from waveguide modes below cutoff
- Low-frequency propagating fields
- Fields from sources near the PML boundary

Meep uses UPML which is inferior for evanescent absorption — this is where
rfx can be definitively better.

## Current State
`rfx/boundaries/cpml.py` implements standard CPML with:
- `b = exp(-(σ + α) * dt / ε₀)` where α is already present but typically small
- `c = σ * (b - 1) / (σ + α)`
- 12 psi fields (6 E + 6 H components)
- Per-axis graded polynomial σ profile

The existing code already has the `α` parameter infrastructure. The upgrade
is primarily about:
1. Adding the `κ` stretching parameter
2. Tuning the α/κ/σ profiles for better performance
3. Validating the improvement

## Deliverable

### 1. Modify `init_cpml()` in `rfx/boundaries/cpml.py`

Add `cfs_alpha` and `cfs_kappa` parameters:

```python
def init_cpml(
    grid,
    *,
    sigma_max: float | None = None,
    alpha_max: float = 0.05,    # existing, increase default
    kappa_max: float = 1.0,     # NEW: κ stretching (1.0 = no stretch)
    polynomial_order: int = 3,
) -> tuple[CPMLConfig, CPMLState]:
```

The CFS profiles should be graded:
- `σ(ρ) = σ_max * ρ^m` (existing polynomial)
- `α(ρ) = α_max * (1 - ρ)` (existing, peaks at outer edge)
- `κ(ρ) = 1 + (κ_max - 1) * ρ^m` (NEW: graded from 1 to κ_max)

In the CPML update equations, `κ` modifies the spatial derivative:
- `∂E/∂x → (1/κ) * ∂E/∂x + ψ` (instead of `∂E/∂x + ψ`)
- The `b` and `c` coefficients remain the same
- `κ` is applied as a divisor on the curl term inside the PML region

### 2. Modify `apply_cpml_e()` and `apply_cpml_h()`

The κ modification applies to the spatial derivative:
```python
# Before (standard CPML):
psi_new = b * psi + c * dE
H += (dt/mu0) * psi_new

# After (CFS-CPML):
psi_new = b * psi + c * dE
H += (dt/mu0) * psi_new
# Plus: divide the main curl contribution inside PML by κ
```

Actually, the simplest correct implementation: store `kappa` profile in
CPMLConfig and divide the curl contribution in the PML region by κ.

### 3. Tests in `tests/test_cfs_cpml.py`

**Test 1: `test_cfs_cpml_evanescent_absorption`**
- Waveguide below cutoff (f < f_c)
- Standard CPML: measure residual field after PML
- CFS-CPML (kappa_max=5): measure residual field
- Assert: CFS residual < standard residual (at least 10x improvement)

**Test 2: `test_cfs_cpml_backward_compatible`**
- kappa_max=1.0 should give identical results to standard CPML
- Run same simulation with both, assert allclose

**Test 3: `test_cfs_cpml_no_regression_above_cutoff`**
- Above-cutoff waveguide: CFS-CPML should be at least as good as standard
- Compare CPML reflection coefficient for both variants
- Assert CFS is not worse

## Constraints
- Backward compatible: `kappa_max=1.0` (default) = standard CPML behavior
- Must not break any existing tests
- Store κ profile in CPMLConfig alongside existing b/c arrays
- Each test < 60 seconds

## Verification
Run: `pytest -xvs tests/test_cfs_cpml.py`
Then: full suite regression check
