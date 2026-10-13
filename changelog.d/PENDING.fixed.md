Cell widths given as float32 values (`np.float32(h)` in a `dx_profile`, or `dx=float(np.float32(h))`) no longer move a
Box declared in metres by one whole cell. Its faces, conductor sheets, port and probe positions and the domain cell
count now treat a coordinate within float32 resolution of a node as on that node, on uniform and graded meshes. Before,
a substrate at 0.381 mm cells could sit one cell off on every axis with no warning.
