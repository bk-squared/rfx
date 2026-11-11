### Fixed — Occupancy of a watertight imported mesh is counted per grid line, not per grid point (#PR)

Before the first time step, rfx decides which cells lie inside an imported mesh (`MeshShape`). For a watertight
mesh it now counts surface crossings once per grid line instead of casting a ray from every cell centre, so the
cost no longer grows as (cells) x (triangles). On eleven spheres and grids it marks exactly the same cells as before.
Before, a 20,480-triangle sphere at 2 mm cells took 9 min and 34 GiB, and a 5,120-triangle sphere at 1.4 mm cells
5.6 min and 25 GiB. A mesh that is not watertight is handled as before: refused by default, and the old test under
`validate=False`.
