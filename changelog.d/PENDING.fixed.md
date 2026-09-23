### Fixed — a multi-device run dropped every lumped RLC element without a word; it now refuses them (#PENDING)

- `sim.run(devices=[...])` (and both distributed runners called directly) solved the
  structure as if every `add_lumped_rlc(...)` element were absent: on a 12 mm PEC box with a
  10 Ω element between source and probe, the multi-device probe amplitude was bit-identical
  to the box without the element and 43 % above the single-device result (#1239). The
  multi-device lanes have no lumped-element update, so they now raise `NotImplementedError`
  naming the elements; remove them or run on one device. `forward(distributed=True)`
  already refused them.
