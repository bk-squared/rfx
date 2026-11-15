### Fixed — drives and RLC elements read the permittivity the field update uses (#PR)

- On an E edge whose update uses subpixel smoothing, the full tensor, conformal PEC
  weights, the Kottke inverse permittivity or UPML with smoothing, lumped, wire and
  MSL ports, current-kind soft sources and lumped RLC elements now use that edge's
  update permittivity. They used the cell permittivity before. On a smoothed
  dielectric face, a 50-ohm port injected 0.81 of its declared current, a soft
  source 0.64, and a series R-L coupled at 0.64. They now inject 1.000. Edges with
  no tensor are unchanged.
- A drive on an edge that `subpixel_smoothing="kottke_pec"` freezes is now refused
  by name. Before, it was admitted and injected field into an edge the metal holds
  at zero.
