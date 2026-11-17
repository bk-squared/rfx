### Fixed — graded multi-device `run()` and `forward()` admit the same mesh (#PR)

- A traced cell-size profile on `run(devices=...)` or `forward(distributed=True)` on
  a graded mesh now gets the named `TracedStructureRefusal`, with or without preflight,
  instead of jax's `TracerArrayConversionError`.
- Both entries apply one grading check, with or without `skip_preflight`: a grading
  ratio above 5 is refused and one above 3 warns. Before, `forward(distributed=True,
  skip_preflight=True)` ran a ratio above 5, and `forward()` never warned above 3.
