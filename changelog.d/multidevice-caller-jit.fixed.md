### Fixed — multi-device CPU runs inside a caller's `jax.jit` (#PR)

- `run(devices=..., report_every=...)` inside `jax.jit` with a graded wire port
  no longer ends the Python process (an XLA CPU compiler abort on jax 0.10.2 and
  0.11.1). With a graded lumped port, its S-parameters now equal the eager call's.
- On CPU devices under a caller's `jax.jit`, multi-device fields, S-parameters and
  records equal the eager call bit for bit, apart from the 64-bit host fields the
  docstrings name. On GPUs the call stays refused by name, as before.
- The compiled multi-device step no longer holds every device's current-moment
  weights (61,600 B per device on a 2-device example) or a zero buffer the size
  of the largest device's snapshot. Plain calls are unchanged bit for bit.
