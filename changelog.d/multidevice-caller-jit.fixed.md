### Fixed — multi-device runs inside a caller's `jax.jit`, and one grading check on both graded multi-device entries (#PR)

- `run(devices=..., report_every=...)` inside `jax.jit` with a graded wire port no longer ends the Python process (an XLA CPU compiler abort on jax 0.10.2 and 0.11.1). With a graded lumped port, its S-parameters now equal the eager call's.
- On CPU devices under a caller's `jax.jit`, multi-device fields, S-parameters and records equal the eager call bit for bit, apart from the 64-bit host fields the docstrings name. On GPUs the call stays refused by name, as before.
- The compiled multi-device step no longer holds every device's current-moment weights (61,600 B per device on a 2-device example) or a zero buffer the size of the largest device's snapshot. Plain calls are unchanged bit for bit.
- A traced cell-size profile on `run(devices=...)` or `forward(distributed=True)` on a graded mesh now gets the named `TracedStructureRefusal`, with or without preflight, instead of jax's `TracerArrayConversionError`.
- Both graded entries apply one grading check, with or without `skip_preflight`: a grading ratio above 5 is refused and one above 3 warns. Before, `forward(distributed=True, skip_preflight=True)` ran a ratio above 5, and `forward()` never warned above 3.
