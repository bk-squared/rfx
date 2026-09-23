### Fixed — a dispersive model split across devices no longer builds the whole domain's Debye/Lorentz arrays on the first device; Kerr materials are refused on that lane instead of running linear (#PENDING)

- `sim.run(devices=...)` computed the Debye/Lorentz coefficient and polarization-state arrays
  for the whole domain on the first device of every process, then cut and shipped them. With
  one Debye and one Lorentz pole that is about twenty per-cell arrays, so on three A6000 a
  96 M-cell model peaked at 21 GB on the first GPU against 6.9 GB on the others. Each device
  now computes its own slab's coefficients from its slab of the material and mask arrays and
  allocates its own zero state; physical-boundary ghost values are unchanged. Measured on the
  same three A6000: 8.3 / 7.0 / 7.8 GB. Results are bit-identical.
- The multi-device lane dropped a material's Kerr `chi3` without a warning and ran it as
  linear: a χ³ = 1e-2 block gave the same probe trace as χ³ = 0, about 17 % away from the
  single-device run. `sim.run(devices=...)` and both distributed runners now refuse Kerr
  materials with `NotImplementedError`.
