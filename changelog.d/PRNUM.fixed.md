### Fixed — material Box faces follow the nearest-node rule

A dielectric, magnetic, lossy or dispersive Box with a face between nodes is now
placed on the nearest node, as PEC boxes are; before, each such face moved up to
one cell toward +. On a patch whose substrate top lay 0.3 cell above a node this
lowered the resonance by 1.2-1.8 % (two meshes).
