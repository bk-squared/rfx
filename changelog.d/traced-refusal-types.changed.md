### Changed — Refusals of traced and unsupported inputs carry rfx's own error types (#PR)

- These refusals now raise `rfx.errors.RfxRefusal` or `rfx.errors.RfxUnsupported` instead of a plain
  `ValueError` or `NotImplementedError`. They are still `ValueError` or `NotImplementedError` subclasses with the
  same messages, so existing `except` clauses still catch them; `except ValueError` now also catches the
  `RfxUnsupported` ones, and a logged `type(exc).__name__` reads the new class name.
- Covered: auto mesh with traced geometry, traced graded profiles and `dt`, traced waveguide-port inputs, a
  Floquet port on a graded z mesh, the coaxial drivers' grid and geometry limits, traced pole and thin-conductor
  parameters, a traced shape dimension under `jax.grad`, a traced `OrientedBox` or wire-port radius,
  `rlc_values_override` off the uniform forward lane, `gradient='adjoint'` limits, and closed-box or
  `CustomWaveform` TF/SF under a transform. Catch `rfx.errors.RfxError` to handle every rfx refusal at once.
- `Simulation(dx=...)` with a traced `dx` raises `TracedStructureRefusal` (an `RfxUnsupported`) at the declaration instead of a raw JAX tracer error in grid building, preflight or `add_msl_port`.
