### Fixed — Empty traced shape refusal under `jax.jit` (#PR)

- A dielectric shape with a traced parameter (for example a radius under `jax.jit` or `grad`) that owns no cell is refused with the "rasterizes to ZERO cells" message, as it is outside a trace. Before, the refusal surfaced as an `UnexpectedTracerError` inside a `JaxRuntimeError`, with the message lost.
- For a Box, Sphere or Cylinder the message prints the drawn thickness (for example 0.0002 m for a 0.2 mm disc between two cell centres); for other shapes, such as a `PolylineWire`, it prints the thickness as unavailable.
- Eager refusals and traced shapes that own cells are unchanged.
