### Fixed — float64 source and smoothing operands retain their precision (#PR)

- Float64 runs retain float64 source tables, TF/SF auxiliary profiles and
  corrections, smoothing and Kottke operands, and waveguide reference materials.
  Removing intermediate float32 rounding can change these values by up to
  about 6e-8 relative (estimate).
- Float32 and mixed float16 runs retain their existing arithmetic, including
  with JAX x64 enabled. The affected arrays double in memory in float64 runs.
