### Fixed — Refuse unsupported traced PEC dimensions by name (#PR)

- A traced PEC Cylinder radius, height or centre, or a traced PEC Sphere centre, under `jax.jit` or `jax.vmap` on a concrete mesh previously stopped with JAX's `ConcretizationTypeError`.
- These cases now raise `TracedShapeDimensionError`, which is still a `ConcretizationTypeError`, and name the traced fields.
- On a traced mesh these cases keep running under `jax.jit`.
- Dielectric shapes are unchanged.
