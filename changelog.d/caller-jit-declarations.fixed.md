### Fixed — two declaration methods crashed when the scene was declared inside a user's `jax.jit` (#PR)

- `add_thin_conductor(..., surface_impedance_f0=...)` raised `ConcretizationTypeError` from its warning text, and
  `add_waveguide_port` given a NumPy `freqs` array raised `TracerArrayConversionError`, even with every input
  concrete. Both now run, and the scene gives the same fields as the eager build.
- Warnings still point at the user's line, and traced inputs stay traced.
