### Fixed — a float64 run built its grid widths, CPML profiles, source tables and smoothing operands in float32 (#PR)

- With `precision="float64"` inside `jax.enable_x64()`, the graded mesh's cell widths and metrics, the CPML grading
  profiles, source tables, the TF/SF auxiliary profiles and their corrections, the subpixel-smoothing and Kottke
  operands and the waveguide reference materials are now built in float64. Before, each carried float32 rounding,
  which moves these values by up to about 6e-8 relative (estimate).
- A check that runs on the host during a traced call (boundary, time-step and shape checks under `jax.jit` or
  `jax.grad`) now judges the run's float64 values. Before, the host received them rounded to float32.
- Float32 and mixed float16 runs keep their arithmetic, also with JAX x64 enabled. The affected arrays double in
  memory in float64 runs.
