### Fixed — omitted waveguide source centres agree across meshes (#PR)

- A waveguide port with `f0` omitted on a graded mesh is now driven at the centre
  of its measurement band, as on a uniform mesh. Previously it used `freq_max/2`:
  6.2 GHz at `freq_max=12.4 GHz`, below the WR-90 TE10 cutoff, with `|S11|` up to
  2.0 on a matched guide.
