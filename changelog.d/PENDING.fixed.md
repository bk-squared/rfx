### Fixed — Graded-mesh preflight uses local cells for MSL reflector and waveguide reference-plane tolerances (#PENDING)

MSL reflector screening reads the primal cell at the feed on the propagation
and trace-width axes. Waveguide mirror auditing uses half the smaller primal
cell at the two reference planes, with the adjacent boundary cell for a plane
outside the mesh. Previously both checks used the boundary x-cell size and
could omit findings inside a refined band. Uniform preflight output is unchanged.
