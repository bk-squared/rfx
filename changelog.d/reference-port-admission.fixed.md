### Fixed — a `port_reference_sims` reference is admitted on its own model, like a direct entry (#PR)

- A reference whose port section lies inside a PEC block is refused with the same reason as a direct
  entry; before, it ran, and a driven port there would launch its mode inside the conductor.
- A reference with a Debye or Lorentz material is refused; before, the reference run dropped its poles
  and ran it as a constant permittivity.
- On the compact T-junction the buried undriven ports changed nothing: S is identical with closed
  pockets (max |S| 3.68 either way). References that keep an undriven port inside metal now need a pocket.
