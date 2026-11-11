### Fixed — Read declared content beyond smoothed dielectric faces (#PR)

Smoothing integrates the outside part of each normal support, so thin gaps and
neighbouring layers enter the Yee and full-tensor operands. Box, Sphere and
Cylinder intersections preserve declaration order and periodic images.
Full-tensor edges and dual faces use their own length and area weights.
