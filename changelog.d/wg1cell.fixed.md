### Fixed — discrete waveguide modes support a one-cell transverse axis (#wg1cell)

- Discrete waveguide profiles now average the two boundary derivatives on a one-cell axis without indexing a missing neighbour. Scalar eigenmode profile conversion also handles singleton axes.
- Added one- and two-cell gradient checks, transverse profile coverage, TE10 cutoff and impedance comparisons across 1/2/4-cell heights, and a 22 mm empty-guide two-port regression.
