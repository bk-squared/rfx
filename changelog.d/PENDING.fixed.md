### Fixed — a multi-device run dropped lumped RLC elements and subgrid refinements without a word; it now refuses them (#PENDING)

- `sim.run(devices=[...])` (and both distributed runners called directly) solved the
  structure as if every `add_lumped_rlc(...)` element were absent: on a 12 mm PEC box with a
  10 Ω element between source and probe, the multi-device probe peak was bit-identical to
  the box without the element and 76 % above the single-device peak, where the element
  lowers it by 43 % (#1239). The same held for `add_refinement(...)`: the refined box ran
  bit-identical to the unrefined one. The multi-device lanes implement neither, so they now
  raise `NotImplementedError` naming the feature; remove it or run on one device.
  `forward(distributed=True)` already refused lumped elements.
- A direct call to either distributed runner also refuses DFT plane probes now, as
  `sim.run(devices=...)` already did.
