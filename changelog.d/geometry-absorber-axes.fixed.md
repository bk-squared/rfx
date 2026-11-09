### Fixed — geometry crossing multiple absorber axes names every crossed axis (#PR)

- A shape crossing the absorber on two axes now gets a finding for each axis.
  Previously only the first crossing in x, y, z order was named.
- Findings still aggregate geometry entries per axis and name the worst overshoot.
