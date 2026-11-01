### Fixed — material Box faces follow the nearest-node rule (#PR)

A dielectric, magnetic, lossy or dispersive Box face between nodes now goes to the
nearest node, as PEC faces do; before, it moved up to one cell toward +. A patch whose
substrate top lay 0.3 cell above a node resonated 1.2-1.8 % low (two meshes).
A material volume holding no cell centre now refuses before stepping (the old rule
inflated a 0.3-cell film 3.3 times). There is no sub-cell dielectric film model:
refine until a cell centre lies inside the layer, or use `add_thin_conductor` for a
conducting sheet. A volume Box drawn to the declared domain's high face now fills the
last cell when the grid is one cell longer than the declared domain.
`interface_eps='dual_average'` now gives the same coefficients as the default.
Mixed Box and non-Box CSG now samples each operand as it would alone (it may now refuse).
