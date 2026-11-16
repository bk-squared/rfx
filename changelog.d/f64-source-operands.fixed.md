### Fixed — a float64 run built its source tables, TF/SF lines and smoothing operands in float32 (#PR)

- With `precision="float64"` inside `jax.enable_x64()`, source tables, the TF/SF auxiliary lines and their
  corrections, the subpixel-smoothing and Kottke operands and the waveguide reference medium are now built in
  float64. Before, each carried float32 rounding of up to about 6e-8 relative.
- Float32 runs are unchanged bit for bit. In a float64 run these arrays use twice the memory.
