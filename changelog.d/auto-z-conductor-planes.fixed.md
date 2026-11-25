### Fixed — Automatic z meshes retain conductor planes (#PR)

Previously, the plate under FR-4 was refused at +6.81 % extent miss,
and a sheet in air solved 0.17 mm off its declared plane with a warning.
Automatic z meshes now rebuild around conductor planes when their z
realization would be refused. Sheets and lossy films are checked on their
normal axis against the neighbouring layer extents, including explicit meshes.
Close planes merge with an explanation and an explicit mesh remedy;
x/y refusals print the automatic dx and profiles needed to replace the mesh.
A model that passes today keeps its profile.
