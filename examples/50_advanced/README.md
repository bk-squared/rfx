# Advanced Research-Grade Examples

Six advanced examples demonstrating rfx's differentiable FDTD capabilities
for research-level RF/microwave design, plus a comprehensive visualization
showcase.

All examples are designed to run on CPU with small grids and short runs.
For production use, increase grid resolution (`dx`) and iteration counts.

## Examples

### 01_patch_bandwidth_opt.py — Patch Antenna Bandwidth Optimization
Density-based topology optimization of a 2.4 GHz patch antenna on FR4.
Optimizes the patch shape (binary PEC/air) using `topology_optimize()`
with `minimize_reflected_energy` as the objective.
Shows: optimized geometry, loss convergence, beta continuation schedule.

### 02_waveguide_filter_inverse.py — Waveguide Filter Inverse Design
Topology optimization of iris geometry inside a WR-90 waveguide for
bandpass filter design. Uses `maximize_transmitted_energy` to shape
the transmission response. Shows: iris geometry, eps distribution,
convergence, reference signals.

### 03_broadband_matching.py — Broadband Matching Network
Parametric sweep of lumped RLC matching network values for 50-ohm
impedance matching at 2.4 GHz. Visualizes S11 vs inductance, frequency
response, Smith chart trajectory, and multi-sweep S11 overlay.

### 04_array_mutual_coupling.py — Antenna Array Mutual Coupling
Study of mutual coupling between patch antenna elements as a function
of inter-element spacing. Sweeps spacing from 0.2 to 1.0 wavelengths
and extracts coupling coefficients. Compares with 1/r^2 reference.

### 05_dielectric_lens.py — Dielectric Lens Beam Shaping
Gradient-based inverse design of a graded-index dielectric lens using
`optimize()` with `DesignRegion`. The optimizer tunes local permittivity
(eps_r = 1..10) to maximize transmitted energy. Shows: optimized eps
distribution, convergence, permittivity histogram.

### 06_material_characterization.py — S-param Material Characterization
Generates synthetic S-parameters from a known Debye material (water at
20C), then recovers the dispersive poles via `differentiable_material_fit`.
Compares recovered vs original permittivity spectra. Demonstrates end-to-end
gradient flow through the FDTD simulation for material fitting.

### 07_visualization_showcase.py — Comprehensive Visualization
All visualization capabilities in one script:
- Field slices at multiple timesteps
- S-parameter magnitude (dB) and phase plots
- Smith chart with frequency markers
- Radiation pattern (E-plane and H-plane polar plots)
- Mesh convergence study
- Parametric sweep
- Field animation (GIF)

Saves 7 PNG files + 1 GIF.

## Running

```bash
cd /path/to/rfx
python examples/50_advanced/01_patch_bandwidth_opt.py
python examples/50_advanced/07_visualization_showcase.py
# etc.
```

## Requirements

- rfx (installed)
- matplotlib, numpy, jax
- optax (for topology optimization examples 01, 02)
- Pillow (for GIF animation in example 07)
