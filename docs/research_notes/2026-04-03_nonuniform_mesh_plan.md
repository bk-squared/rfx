# Non-Uniform Yee Mesh Implementation Plan

## Why
- SBP-SAT subgridding has coupling coefficient issues (37-57% error)
- Uniform fine grid OOMs or has source detection issues
- Non-uniform mesh is what CST/OpenEMS actually use
- Only ~50 lines of core code change

## What Changes

### Core: `rfx/core/yee.py`
Replace scalar `dx` with per-axis arrays `(dx_arr, dy_arr, dz_arr)`.

H update: use mean of adjacent cell sizes
```python
inv_dz_H = 1.0 / (0.5 * (dz[:-1] + dz[1:]))  # averaged spacing
Hx -= (dt/mu) * ((Ez[:,1:,:]-Ez[:,:-1,:]) * inv_dy_H - ...)
```

E update: use local cell size
```python
inv_dz_E = 1.0 / dz  # local spacing
Ex += (dt/eps) * ((Hz[:,1:,:]-Hz[:,:-1,:]) * inv_dy_E - ...)
```

### Grid: `rfx/grid.py`
Add non-uniform grid support: `Grid(dx_profile=(dx_arr, dy_arr, dz_arr))`

### Auto-config: `rfx/auto_config.py`
Choose dz profile that snaps to substrate boundaries:
```python
dz = snap_profile(features=[h, ...], max_dz=lambda/20, grading=1.4)
```

### Source normalization
```python
dV_source = dx[i] * dy[j] * dz[k]  # actual source cell volume
J_amplitude = waveform(t) / dV_source
```

## CFL
dt = 0.99 / (c * sqrt(1/dx_min² + 1/dy_min² + 1/dz_min²))

## Expected Impact
- Substrate: dz=0.4mm (h/dx=4 exact)
- Air: dz=2mm (coarse, saves cells)
- Total cells: ~5-10x less than uniform fine grid
- Accuracy: <1% (substrate perfectly resolved, no discretization error)
