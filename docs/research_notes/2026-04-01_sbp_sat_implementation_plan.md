# SBP-SAT Subgridding Implementation Plan for rfx

## Key References

1. **[Cheng et al., arXiv:2110.09054, J. Comput. Phys. 494, 2023](https://arxiv.org/abs/2110.09054)**
   "A Stable FDTD Subgridding Scheme with SBP-SAT for Transient Electromagnetic Analysis"
   - Original 1D/TM formulation, provably stable
   - Adds boundary field components for SBP property

2. **[Cheng et al., arXiv:2202.10770, IEEE TGRS, 2022](https://arxiv.org/abs/2202.10770)**
   "SBP-SAT FDTD Subgridding Using Staggered Yee's Grids Without Modifying Field Components"
   - **Most relevant for rfx**: uses standard Yee grid
   - Field extrapolation at boundaries instead of adding components
   - No dissipation in the whole domain → long-time stability guaranteed
   - TM analysis demonstrated

3. **[Cheng et al., IEEE TAP, 2023](https://ieeexplore.ieee.org/document/10242252/)**
   "Toward the 2-D Stable FDTD Subgridding Method With SBP-SAT and Arbitrary Grid Ratio"
   - Extends to arbitrary (not just integer) grid ratios
   - Interpolation matrices for coupling different mesh sizes

4. **[Cheng et al., IEEE TAP, 2022](https://www.semanticscholar.org/paper/Toward-the-Development-of-a-3-D-SBP-SAT-FDTD-Theory-Cheng-Liu/f5d5f3ff58662c3c5b3608447d420de127386e77)**
   "Toward the Development of a 3-D SBP-SAT FDTD Method: Theory and Validation"
   - 3D extension, theoretically stable
   - Matrix-free, same accuracy as FDTD with negligible overhead

5. **[Cheng et al., IEEE AP-S 2024](https://ieeexplore.ieee.org/document/10686530)**
   "A Theoretically Stable FDTD Subgridding Method with Arbitrary Grid Ratios and Material Transitions"
   - Handles material interfaces crossing the subgrid boundary
   - Critical for practical use (scatterer straddles coarse/fine)

## Algorithm Summary (from Ref [2] — staggered Yee variant)

### Core idea

Standard FDTD Yee update:
```
H^{n+1/2} = H^{n-1/2} + (dt/mu) * curl(E^n)
E^{n+1}   = E^n       + (dt/eps) * curl(H^{n+1/2})
```

SBP-SAT modification at the coarse-fine interface:
```
H^{n+1/2} = H^{n-1/2} + (dt/mu) * curl(E^n) + SAT_H(E_coarse, E_fine)
E^{n+1}   = E^n       + (dt/eps) * curl(H^{n+1/2}) + SAT_E(H_coarse, H_fine)
```

The SAT terms are **penalty terms** that weakly enforce continuity of
tangential fields across the interface:
```
SAT_H = -tau_H * P^{-1} * (H_boundary - H_interpolated_from_other_grid)
SAT_E = -tau_E * P^{-1} * (E_boundary - E_interpolated_from_other_grid)
```

Where:
- `P` = diagonal SBP norm matrix (cell volumes)
- `tau` = penalty parameter (chosen for stability, typically tau = 0.5)
- Interpolation is done via derived interpolation matrices `R` that map
  between coarse and fine grid boundary nodes

### Why it's stable

The SBP property guarantees:
```
d/dt (||E||² + ||H||²) ≤ 0
```

The discrete energy is non-increasing, so fields cannot grow unboundedly.
This is a **mathematical proof**, not an empirical observation.

### Interpolation matrices

For grid ratio r:1 (e.g., 3:1):
- `R_c2f`: maps coarse boundary nodes to fine grid positions (upsampling)
- `R_f2c`: maps fine boundary nodes to coarse grid positions (downsampling)
- These are derived analytically to preserve SBP stability

## Implementation Plan for rfx

### Phase 1: 1D Prototype (validate stability + JAX compatibility)

Create `rfx/subgridding/sbp_sat_1d.py`:

```python
class SubgriddedDomain1D:
    """Two coupled 1D Yee grids with SBP-SAT interface."""
    coarse_grid: Grid1D     # dx_c = r * dx_f
    fine_grid: Grid1D       # dx_f
    ratio: int              # r (3 recommended)
    R_c2f: jnp.ndarray     # interpolation coarse → fine
    R_f2c: jnp.ndarray     # interpolation fine → coarse
    tau: float              # penalty parameter

def step_subgridded_1d(state_c, state_f, dt, config):
    """One timestep of coupled coarse+fine grids."""
    # 1. Standard FDTD update on both grids
    state_c = update_1d(state_c, dt)
    state_f = update_1d(state_f, dt)  # r sub-steps for temporal ratio

    # 2. SAT penalty at interface
    e_diff = state_c.e[boundary] - R_f2c @ state_f.e[boundary]
    h_diff = state_c.h[boundary] - R_f2c @ state_f.h[boundary]
    state_c.e[boundary] -= tau * e_diff / P_c
    state_f.e[boundary] += tau * (R_c2f @ e_diff) / P_f
    # (same for H)

    return state_c, state_f
```

Test: run 10⁶ steps, verify energy doesn't grow.

### Phase 2: 2D TM Extension

Extend to 2D TMz mode (Ez, Hx, Hy) with rectangular refinement region.
The interface now has four sides (top/bottom/left/right).

### Phase 3: 3D Integration with jax.lax.scan

Key design decision: **single scan with compound carry**

```python
carry = {
    "state_coarse": FDTDState,
    "state_fine": FDTDState,
    "cpml_coarse": CPMLState,
    "cpml_fine": CPMLState,
}

def scan_body(carry, xs):
    # Coarse step
    carry["state_coarse"] = update_h(carry["state_coarse"], ...)
    carry["state_coarse"] = update_e(carry["state_coarse"], ...)

    # Fine sub-steps (r steps per coarse step)
    for _ in range(ratio):
        carry["state_fine"] = update_h(carry["state_fine"], ...)
        carry["state_fine"] = update_e(carry["state_fine"], ...)

    # SAT interface coupling
    carry = apply_sat_interface(carry, config)

    return carry, outputs
```

The fine sub-steps can use `jax.lax.fori_loop` inside the scan body.

### Phase 4: High-level API

```python
sim = Simulation(freq_max=10e9, domain=(0.10, 0.10, 0.10),
                 boundary="cpml", dx=0.002)

# Add a refinement region (3:1 ratio, dx_fine = 0.002/3)
sim.add_refinement(
    Box((0.03, 0.03, 0.03), (0.07, 0.07, 0.07)),
    ratio=3,
)

# Everything else works the same
sim.add_port(...)
result = sim.run(n_steps=1000)
```

## JAX Differentiability

- SBP operators: matrix-vector multiply → differentiable
- SAT penalty: linear operation on fields → differentiable
- Interpolation R_c2f, R_f2c: fixed matrices → differentiable
- jax.checkpoint works across the compound carry
- **Conclusion: SBP-SAT subgridding is fully differentiable with jax.grad**

## Contribution Statement

If published, the rfx subgridding contribution would be:

> **First differentiable FDTD implementation with provably stable subgridding.**
> We integrate the SBP-SAT framework of Cheng et al. [1-5] into a JAX-based
> FDTD solver, enabling gradient-based inverse design of multi-scale RF
> structures with local mesh refinement. The automatic differentiation graph
> flows correctly through the subgrid interface penalty terms, allowing
> gradient computation for structures spanning coarse and fine grids.

## Timeline Estimate

| Phase | Effort | Output |
|-------|--------|--------|
| 1: 1D prototype | 2 days | Stability validation |
| 2: 2D TM | 3 days | Interface convergence |
| 3: 3D + scan | 5 days | Full integration |
| 4: API + tests | 2 days | User-facing feature |
| **Total** | **~2 weeks** | |
