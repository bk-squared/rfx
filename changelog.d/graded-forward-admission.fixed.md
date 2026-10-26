### Fixed — graded multi-device forward() refuses microstrip ports and lumped RLC with the same reason as run() (#PR)

- Graded multi-device `forward()` and `run()` now report the same unsupported-feature
  reason for microstrip ports and lumped RLC elements, with guidance for each call.
