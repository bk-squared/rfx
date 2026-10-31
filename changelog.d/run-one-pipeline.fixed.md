### Fixed — run() shares forward() setup under a user jit (#PR)

- Fixed-step single-device uniform, graded and ADI runs now evaluate declaration
  setup like `forward()` under `jax.jit`, avoiding setup-driven probe differences
  and graded-port tracing errors.
- Jitted `run()` port-S extraction now points to `forward(port_s11_freqs=...)`
  or a plain call. Jitted multi-device closed-box TF/SF calls now request a plain call.
