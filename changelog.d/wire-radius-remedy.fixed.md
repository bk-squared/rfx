### Fixed — wire-port radius advice uses the finest mesh in a refinement ladder (#PR)

- The radius-less wire-port warning now tells users to choose a radius at most 0.20 x the finest transverse cell they will run, so one fixed radius is admitted on every mesh of the ladder. Previously it named the coarsest mesh, whose radius finer meshes refuse.
