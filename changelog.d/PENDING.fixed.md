### Fixed — a `surface_impedance_f0` thin conductor whose drawn edges lie on grid nodes is solved at its drawn width, no longer one cell wider (#PENDING)

- A resistive strip drawn 8 cells wide was solved 9 cells wide by the f0 (Leontovich) sheet and 8 cells wide by the
  DC sheet. Both are now 8: for R_s = 377 ohm/sq filling 0.8 of a line, |S21| moves from 0.690 to 0.714 (0.30 dB),
  the quasi-static value 1/(1 + f/2). The difference falls as 1/width (0.15 dB at 16 cells, 0.10 dB at 24).
- The edge rows on a drawn free edge now carry half of the sheet conductance; the set of loaded edges is unchanged.
  A sheet that runs wall to wall, and an edge drawn between two nodes, are unchanged. Uniform and graded meshes.
- Measured on a patch with R_s = 1.07 ohm/sq (two of its four edges on nodes): the resonance stays at 25.399 GHz
  and Q moves from 27.304 to 27.309.
- `sim.realized_geometry()` reports an f0 conductor by its node footprint and solved extent; it was one cell too
  long on every in-plane axis.
