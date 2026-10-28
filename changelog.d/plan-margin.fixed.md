### Fixed — a graded z axis solves the declared length: planner margins, auto mesh and a 1 % sum rule (#PR)

- `plan_mesh` / `auto_configure` with a graded z put a board's lower face on the z-lo absorber and solved about half
  the z they reported (+2 dB near field over 1.5–3.5 GHz on a 1.6 mm FR4 board). The z profile now holds one margin
  below and one above; add every shape translated by the new `offset`.
- The auto mesh (`Simulation` without `dx`) replaced the declared z by geometry top + margin (a declared 8 mm box ran
  76.5 mm tall); it now builds the declared domain.
- A `dz_profile` that missed the declared z built silently: a 4.1 mm profile under a 10 mm declaration solved 4.1 mm,
  so an NTFF box drawn up to 6 mm read outside it and its far field came out 2.97x below the full column's. More than
  1 % off is now refused (`snap="strict"`) or warned (`snap="declared"`), also in `make_nonuniform_grid(domain=)`.
- New advisory `body_flush_on_absorber`: a static board lying flush on an absorbing face, continued there as a half-space.
