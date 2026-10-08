### Fixed — Source amplitudes use consistent material coefficients across execution paths (#PENDING)

Field sources inject their declared waveform with zero material tangent.
Current sources on periodic material sweeps use the wrapped edge coefficient,
and subgrid sources and driven ports account for loads declared after them.
Current sources on experimental disjoint subgrids use the vacuum update coefficient.
