### Fixed — a float64 run built its graded cell widths, CPML profiles and host-side checks in float32 (#PR)

- With `precision="float64"` inside `jax.enable_x64()`, the graded mesh's cell widths and metrics and the CPML
  grading profiles are now built in float64. Before, they carried float32 rounding of up to about 6e-8 relative.
- A check that runs on the host during a traced call (boundary, time-step and shape checks under `jax.jit` or
  `jax.grad`) now judges the run's float64 values. Before, the host received them rounded to float32.
- Float32 runs are unchanged bit for bit. In a float64 run these arrays use twice the memory.
