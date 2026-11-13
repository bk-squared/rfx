### Fixed — Empty traced shape refusal under `jax.jit` (#PR)

- A dielectric shape whose radius is traced (for example under `jax.jit` or `jax.vmap`) and that owns no cell is refused with the "rasterizes to ZERO cells" message, as it is outside a trace. Before, the refusal surfaced as an `UnexpectedTracerError` inside a `JaxRuntimeError`, with the message lost.
- For a Box, Sphere or Cylinder the message prints the drawn thickness (about 0.0002 m for a 0.2 mm disc between two cell centres, read from float32 values); for other shapes, such as a `PolylineWire`, it prints the thickness as unavailable.
- Eager refusals and traced shapes that own cells are unchanged. A gradient with respect to a shape dimension is still refused by name before this check, and a traced centre still fails earlier, at trace time, for a separate cause.
