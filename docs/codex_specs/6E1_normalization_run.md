# Codex Spec 6E1: Two-Run Normalization for Waveguide S-Parameters

## Goal
Implement a two-run normalization pattern that cancels Yee-grid numerical
dispersion from waveguide S-parameter measurements, achieving |S21| > 0.95
for a straight empty waveguide above cutoff.

## Background (from Meep research)
Meep achieves high S-parameter accuracy by running two simulations:
1. **Normalization run**: empty waveguide (no device), records incident modal
   coefficient `a_inc(f)` at the input port
2. **Device run**: actual structure, records coefficients at all ports

S-parameters are then: `S21 = a_fwd_out(f) / a_inc(f)`. This cancels
numerical dispersion bias common to both runs.

## Current State
- `rfx/sources/waveguide_port.py` has `extract_waveguide_sparams()` and
  `extract_waveguide_s_matrix()` that compute S-params from a single run
- Current |S21| for empty straight waveguide: mean 0.74 (should be ~1.0)
- The error comes from Yee-grid dispersion + finite-time DFT truncation

## Deliverable

### 1. `extract_waveguide_s_params_normalized()` in `rfx/sources/waveguide_port.py`

```python
def extract_waveguide_s_params_normalized(
    grid,
    materials,          # device materials
    ref_materials,      # reference (empty waveguide) materials
    port_cfgs: list[WaveguidePortConfig],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str = "x",
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
) -> jnp.ndarray:
    """Two-run normalized waveguide S-matrix.

    Run 1: reference (empty waveguide) → incident coefficients
    Run 2: device → port coefficients
    S_ij = b_i(device) / a_j(reference)
    """
```

The function should:
1. Run the reference simulation with `ref_materials` (same ports, same n_steps)
2. Extract incident wave `a_ref_j(f)` at each port from the reference run
3. Run the device simulation with `materials`
4. Extract outgoing waves `b_i(f)` at each port from the device run
5. Compute `S_ij(f) = b_i_device(f) / a_j_reference(f)`

### 2. High-level API: `Simulation.compute_waveguide_s_matrix(normalize=True)`

Add `normalize: bool = True` parameter to `compute_waveguide_s_matrix()`.
When True, automatically creates a reference run with vacuum in all design
regions (or with the base waveguide geometry only).

For the initial implementation, the reference materials should be the same
as the device materials but with all user-added geometry shapes removed
(just the base waveguide walls). The simplest approach: store the materials
state before and after user geometry is applied.

### 3. Tests in `tests/test_normalization.py`

**Test 1: `test_normalized_s21_straight_waveguide`**
- Empty straight waveguide, two ports (+x, -x)
- Without normalization: |S21| mean ~ 0.74 (current)
- With normalization: |S21| mean > 0.95
- Assert improvement and accuracy

**Test 2: `test_normalized_s_matrix_with_obstacle`**
- Waveguide with a dielectric block between ports
- Normalized S-matrix should satisfy passivity: Σ_i |S_i,j|² ≤ 1.05 at each frequency
- Reciprocity: |S21| ≈ |S12| within 5%

**Test 3: `test_normalization_preserves_reciprocity`**
- Verify normalization doesn't break reciprocity
- Normalized |S12| - |S21| error < 5%

## Constraints
- Do NOT break existing non-normalized API — `normalize=False` keeps old behavior
- Each test < 180 seconds (two runs per test)
- Read existing `extract_waveguide_s_matrix()` carefully before modifying

## Verification
Run: `pytest -xvs tests/test_normalization.py`
All tests must pass. Then run full suite to check regression.
