# Codex Spec 6E6: Subpixel Smoothing (Anisotropic ε Tensor)

## Goal
Implement anisotropic subpixel smoothing at dielectric material interfaces
to achieve second-order convergence, matching/exceeding Meep's accuracy at
equivalent resolution.

## Background (from Meep research)
When a Yee voxel straddles a dielectric interface, raw stairstepping gives
only first-order convergence. Meep uses:
- **Parallel to interface**: arithmetic mean `ε_∥ = f·ε₁ + (1-f)·ε₂`
- **Perpendicular to interface**: harmonic mean `ε_⊥ = [f/ε₁ + (1-f)/ε₂]⁻¹`

where `f` is the filling fraction and the interface normal determines
which field components are parallel vs perpendicular.

This gives second-order (quadratic) convergence — equivalent accuracy at
6x lower resolution.

## Approach
Since rfx uses CSG geometry (`Box`, `Sphere`, `Cylinder`) with analytical
level-set functions, we can compute exact filling fractions and surface
normals at each boundary voxel.

The key insight: the Yee grid has different E-field components at different
positions. For Ex at (i+½,j,k), Ey at (i,j+½,k), Ez at (i,j,k+½), the
effective ε seen by each component depends on the interface orientation.

### Simplified approach (for initial implementation)
Rather than a full tensor, use **component-wise averaging**:
- For each E-component, compute the filling fraction at that component's
  Yee position
- Use harmonic mean for the component perpendicular to the interface normal
- Use arithmetic mean for components parallel to the interface

## Deliverable

### 1. `compute_smoothed_eps()` in `rfx/geometry/smoothing.py` (new file)

```python
def compute_smoothed_eps(
    grid: Grid,
    shapes: list[tuple[Shape, float]],  # (shape, eps_r) pairs
    background_eps: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute smoothed permittivity arrays for each E-field component.

    Returns (eps_ex, eps_ey, eps_ez) each of shape grid.shape.
    At interface voxels, uses anisotropic arithmetic/harmonic averaging.
    Interior voxels get the bulk permittivity value.
    """
```

For each shape boundary voxel:
1. Compute filling fraction `f` using the shape's signed distance function
   (or by sub-sampling if SDF not available)
2. Estimate local interface normal from the gradient of the SDF
3. For each E-component, determine if it's parallel or perpendicular to normal
4. Apply arithmetic (parallel) or harmonic (perpendicular) mean

### 2. Integration with `Simulation.run()`

Add `subpixel_smoothing: bool = False` parameter to `Simulation.run()`.
When enabled, replace the scalar `eps_r` with per-component arrays before
time-stepping.

This requires modifying the Yee E-update to support per-component ε:
- Currently: `E_new = Ca * E + Cb * curl_H` where Ca/Cb depend on `eps_r[i,j,k]`
- With smoothing: Ex uses `eps_ex[i,j,k]`, Ey uses `eps_ey[i,j,k]`, etc.

### 3. Tests in `tests/test_subpixel.py`

**Test 1: `test_convergence_order_dielectric_sphere`**
- Dielectric sphere (eps_r=4, radius=0.01) in PEC cavity
- Run at 3 resolutions: 20, 30, 40 cells per wavelength
- Measure resonance frequency error vs analytical
- Without smoothing: error ∝ 1/N (first-order)
- With smoothing: error ∝ 1/N² (second-order)
- Assert: smoothing convergence rate > 1.5 (between 1st and 2nd order)

**Test 2: `test_smoothing_reduces_stairstepping_error`**
- Dielectric slab at non-grid-aligned angle (tilted 15°)
- Measure Fresnel reflection with and without smoothing
- Smoothing should reduce error vs analytical by at least 2x at same resolution

**Test 3: `test_smoothing_disabled_matches_original`**
- `subpixel_smoothing=False` (default) gives identical results to current code
- Backward compatibility check

## Constraints
- Backward compatible: disabled by default
- The E-update modification should use per-component ε only when smoothing
  is enabled (avoid overhead when disabled)
- Focus on dielectric interfaces (not metal/PEC — those use stairstepping by design)
- Each test < 120 seconds
- Create new file `rfx/geometry/smoothing.py`, modify `rfx/core/yee.py` minimally

## Verification
Run: `pytest -xvs tests/test_subpixel.py`
Then: full suite regression check
