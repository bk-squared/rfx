# Codex Spec 6F: Oblique TFSF via Dispersion-Matched 1D Auxiliary Grid

## Goal
Fix the oblique TFSF vacuum leakage (currently 27%, marked xfail) by using
a dispersion-matched 1D auxiliary grid instead of analytic waveforms.

## Background
Current oblique TFSF uses analytic `E = f(t - k·r/c)` at the TFSF boundary.
This satisfies continuous Maxwell but not the discrete Yee curl equations,
causing 27% vacuum leakage.

### Schneider's approach (Taflove Ch. 5.6)
For oblique angle θ, the 1D auxiliary grid uses modified parameters:
- `dx_1d = dx / cos(θ)` — matches the projected cell size along k
- The 1D grid naturally has the same numerical dispersion as the 3D grid
  along the propagation direction

At each TFSF boundary cell, a transverse phase correction is applied:
- `E_inc(x, y) = E_1d(x') * exp(j·k_y·y)` where `k_y = (ω/c)·sin(θ)`
- The phase factor accounts for the wavefront tilt

## Current State
- `rfx/sources/tfsf.py` has `init_tfsf(angle_deg=...)`
- When `sin_theta > 1e-8`, the 1D aux grid is bypassed (`update_tfsf_1d_h/e`
  return early) and analytic fields are used in `apply_tfsf_h/e`
- `test_oblique_tfsf_fresnel` is marked `xfail`

## Deliverable

### 1. Modify `init_tfsf()` for oblique angles

When `angle_deg != 0`:
- Compute `dx_1d = dx / cos(theta)`
- Recompute 1D CPML profiles with the new `dx_1d`
- The `dt` remains the same as the 3D grid
- Store `dx_1d` in TFSFConfig

### 2. Modify `update_tfsf_1d_h()` and `update_tfsf_1d_e()`

Remove the `if abs(cfg.sin_theta) > 1e-8: return st` early exit.
Instead, use `cfg.dx_1d` (which equals `dx/cos(theta)` for oblique,
or `dx` for normal incidence) in the 1D update equations.

### 3. Modify `apply_tfsf_e()` and `apply_tfsf_h()`

For oblique incidence, replace analytic waveforms with 1D-grid values
plus transverse phase correction:

```python
# For Ez polarization at x_lo boundary:
# e1d gives E at the x_lo position (same index mapping as normal incidence)
# Add phase tilt along y for each cell:
k_y = (2 * pi * f0 / c) * sin_theta  # approximate — use instantaneous
phase_y = jnp.exp(1j * k_y * (jnp.arange(ny) - grid_pad) * dx)
E_inc_lo = e1d[i0] * phase_y.real  # real part for real-valued fields
```

Actually, the correct approach for real-valued FDTD:
- The 1D grid carries the amplitude envelope
- The transverse phase is applied as a time delay per y-cell:
  `E_inc(x_lo, y_j, t) = e1d[i0](t - y_j * sin(θ) / c)`
- This requires interpolating the 1D waveform at retarded times

Simplest correct implementation: store a short history buffer of e1d/h1d
values and interpolate for each y-cell's retarded time.

### 4. Update tests

- Remove `@pytest.mark.xfail` from `test_oblique_tfsf_fresnel`
- The test should now pass with vacuum leakage < 5%
- Fresnel reflection error < 15%

## Constraints
- Must not break normal-incidence TFSF (angle_deg=0 path unchanged)
- Keep `dx_1d = dx` for normal incidence (no behavioral change)
- All existing TFSF tests must continue to pass
- Oblique test < 120 seconds

## Verification
Run: `pytest -xvs tests/test_verification.py::test_oblique_tfsf_fresnel`
Must pass (not xfail). Then run full suite.
