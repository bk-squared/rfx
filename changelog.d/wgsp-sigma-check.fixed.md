### Fixed — Warn on excessive returned waveguide reflection (#PR)

`run()` and `forward()` now warn when a waveguide port's returned `|s11|`
exceeds `1 + 1e-2`. `s21` is the lead-in propagation factor between the
reference and probe planes and is not judged. S and result fields are unchanged.
