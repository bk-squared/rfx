### Fixed — a lumped RLC element under UPML without smoothing read the wrong permittivity (#NNNN)

- Under `boundary='upml'` with no subpixel smoothing, a lumped RLC element solved its
  current with the four-cell mean permittivity of its edge, while the UPML step uses
  the permittivity of the cell that owns the edge. Where those differ, the element's
  update denominator was scaled by their ratio (1.75/4.0 on an edge with one eps_r 4
  cell among four), and its series resistance with it.
  The element now reads the same cell-owned permittivity and conductivity as the step.
  Runs with smoothing, with `boundary='cpml'`, and edges inside one material are unchanged.
