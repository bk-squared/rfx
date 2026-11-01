### Changed — conductors continue into absorbers on traced meshes (#PR)

- Box volumes and axial Cylinders now use the declared nominal mesh to continue through absorbing faces when a mesh profile is differentiated or compiled. A run-time check refuses a changed outermost-layer occupancy or a pad beyond the continuation; re-declare `nominal_profiles` at the current design or keep the axis concrete. Missing nominal profiles and lattice-dependent continuations still refuse.
