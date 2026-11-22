### Fixed — declaring a scene inside a user's `jax.jit` no longer crashes on concrete inputs (#PR)

- Every public declaration method (`add` and each `add_*`) now evaluates what reads only concrete inputs at trace
  time, as `forward()` and `run()` already did, so a scene built inside `jax.jit` gives the same fields as the eager
  build.
- Two methods crashed before: `add_thin_conductor(..., surface_impedance_f0=...)` (`ConcretizationTypeError` from its
  warning text) and `add_waveguide_port` given a NumPy `freqs` array (`TracerArrayConversionError`).
- Traced inputs stay traced: values and gradients through declaration arguments are unchanged.
