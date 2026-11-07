### Fixed — a gradient over a material's permittivity now passes through subpixel smoothing (#PR)

- `jax.grad`, `jax.jit` and `jax.vmap` over `eps_r` with `subpixel_smoothing` on stopped with
  `TypeError: unhashable type`: the smoother grouped shapes by the permittivity value. Shapes are now
  grouped by medium: equal concrete values as before (concrete runs unchanged bit for bit), a traced
  value by its tracer, so one traced material is one medium.
- A traced `eps_r` now runs with `True` (uniform and graded) and `"kottke_pec"` (uniform); `"full_tensor"`
  refuses it by name until #2441 (under a trace it solves a different model on a face on grid nodes).
- Two materials given different tracers are two media even when their values are equal: their shared
  face is smoothed as an interface, and where they overlap the one declared last wins.
- Under a trace the curved high-contrast advisory warns that it was not judged and names the material.
- `forward()` still needs `skip_preflight=True` with a traced `eps_r` (known limitations).
