### Fixed — Reduce unnecessary mesh refinement near sheet edges and fixed faces. (#PENDING)

Nearby faces and domain ends now participate in sheet-edge fitting, avoiding
sliver cells left by repeated shrinking. Faces that land on an edge-cell
boundary stay paired when the mesh refines that cell.
