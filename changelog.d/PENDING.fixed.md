### Fixed — a dispersive model split across devices no longer builds the whole domain's Debye/Lorentz arrays on the first device (#PENDING)

- `sim.run(devices=...)` computed the Debye/Lorentz coefficient and polarization-state arrays
  for the whole domain on the first device of every process, then cut and shipped them. With
  one Debye and one Lorentz pole that is about twenty per-cell arrays, so on three A6000 a
  96 M-cell model peaked at 21 GB on the first GPU against 6.9 GB on the others. Each device
  now computes its own slab's coefficients from its slab of the material and mask arrays and
  allocates its own zero state; physical-boundary ghost values are unchanged.
- Results are bit-identical (six dispersive models on 2, 3 and 4 devices, both topologies).
