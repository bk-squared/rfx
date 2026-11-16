### Fixed — Advisory preflight reports a check it could not judge under a transform (#PR)

- Under `jax.jit`, `jax.grad`, `jax.vmap` or `jax.jvp`, an advisory preflight check that reads a traced value used to stop preflight with a raw JAX error, or to be skipped without a word. It now gives one `not_judged_traced` warning per check, naming the traced inputs, for example `not judged: _validate_mesh_quality reads a traced input (add(shape))`.
- A concrete model gets the same issues as before, and no extra work: the inputs are named only when a check cannot judge.
- Named rfx refusals and other validator errors keep their existing handling.
- `preflight(strict=True)` escalates these warnings like every other issue. `run()` and `forward()` use non-strict preflight, so they are unaffected.
