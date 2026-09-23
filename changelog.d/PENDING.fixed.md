### Fixed — a multi-device run dropped lumped RLC elements, MSL ports and subgrid refinements without a word; it now refuses them (#PENDING)

- `sim.run(devices=[...])` (and both distributed runners called directly) solved the
  structure as if every `add_lumped_rlc(...)` element were absent: on a 12 mm PEC box with a
  10 Ω element between source and probe, the multi-device probe peak was bit-identical to
  the box without the element and 76 % above the single-device peak, where the element
  lowers it by 43 % (#1239). The same held for `add_refinement(...)` (the refined box ran
  bit-identical to the unrefined one) and for `add_msl_port(...)` (a microstrip fed only by
  its MSL port read exactly zero under the trace, against a peak of 22 on one device). The
  multi-device lanes implement none of these, so they now raise `NotImplementedError`
  naming the feature; remove it or run on one device.
- A direct call to either distributed runner also refuses DFT plane probes now, as
  `sim.run(devices=...)` already did.
- A contract test records how the multi-device lanes treat every attribute a `Simulation`
  carries; a new attribute fails it until its multi-device handling is decided.
