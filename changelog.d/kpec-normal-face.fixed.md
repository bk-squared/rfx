### Fixed — normal fields at off-grid PEC Box faces (#PR)

- `subpixel_smoothing="kottke_pec"` now uses the declared dielectric and
  exact Box intervals for normal components of partially filled PEC cells.
- Normal fields on a metal face retain their finite series-capacitor value;
  tangential fields and the existing PEC freeze threshold are unchanged.
