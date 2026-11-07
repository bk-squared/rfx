### Fixed — forward() refuses wire reference planes (#PR)

- `forward()` now refuses `reference_plane_cells`, which it accepted without using; its S21 phase differed from `run()` by 14–33 degrees on the issue's 16 mm thru.
