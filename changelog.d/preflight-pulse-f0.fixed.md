### Fixed — preflight judges waveguide pulses at the driven source centre (#PR)

- For a waveguide port with `f0` omitted, preflight's unresolved/undriven-pulse
  check now uses the measurement band centre that the run drives, instead of
  `freq_max/2`, on uniform and graded meshes.
