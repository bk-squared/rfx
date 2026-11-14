### Changed — the refusal of more than 8 traced dispersion poles names the limit and the routes (#PR)

- Under `jax.jit` or `jax.vmap`, when the cells of Drude, Lorentz or Debye media are traced (a traced mesh profile, a traced `Sphere` radius), the time-step admission refuses more than 8 distinct poles. Its message now states that limit and why.
- It gives three routes: equal pole values (or one pole object) for poles of one medium, a concrete axis or shape, or the gradient with `jax.grad` or `jax.jvp` without `jax.jit` or `jax.vmap`.
- No result changes.
