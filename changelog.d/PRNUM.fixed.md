### Fixed — material Box faces follow the nearest-node rule

A dielectric, magnetic, lossy or dispersive Box with a face between nodes is now
placed on the nearest node, as PEC boxes are; before, each such face moved up to
one cell toward +. On a patch whose substrate top lay 0.3 cell above a node this
lowered the resonance by 1.2-1.8 % (two meshes).

A material volume that contains no primal-cell centre now refuses before stepping,
including zero-thickness dielectric Boxes, dielectric CurvedPatch shapes and
unresolved dielectric Vias. Previously, the node rule inflated a 0.3-cell film
to one full cell (3.3 times its drawn thickness). There is no sub-cell dielectric
film model: refine until a cell centre lies inside the layer. For a conducting
sheet, use `add_thin_conductor`.
