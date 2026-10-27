### Fixed — a Drude/Lorentz medium on two devices under an outer `jax.jit` is refused instead of running unchecked (#PR)

`run(devices=...)` and `forward(distributed=True)` with a Drude or Lorentz medium whose material values or cell sizes
are traced by an outer `jax.jit` or `jax.vmap` now raise a `ValueError` naming the cause. Before, they ran without the
time-step stability check, so a medium above its limit could grow without bound. Calls without the outer transform
(including `jax.grad` and `jax.jvp` alone) and one-device runs are checked as before.
