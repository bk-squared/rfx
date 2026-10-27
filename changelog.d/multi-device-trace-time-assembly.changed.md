### Changed — Graded multi-device runs inside a user `jax.jit` build their model once while tracing; results are unchanged (#PR)

`forward(distributed=True)` and `run(devices=...)` on a graded mesh, called
inside a user `jax.jit`, now evaluate the grid, materials, ports, drives and
slab records at trace time instead of staging them into the compiled program.
Returned arrays and permittivity gradients are bit for bit unchanged (four
scenes, two CPU devices, eager and jit).
