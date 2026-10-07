### Added — realized geometry reports coaxial calculator conductors

- After a coaxial reflection, two-port, or coax-to-microstrip calculation,
  `realized_geometry()` includes the stamped shell, pin, and dielectric,
  with port provenance, area-equivalent radii, and realized axial extents.
  The conductor owner supplies the kernel's unchanged PEC edge masks.
  Stamped geometry check findings are recorded without adding acceptance gates.
