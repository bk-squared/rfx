### Changed — a probe, point source or port inside the CPML pad is refused unless the object opts in (#PR)

- Points in the CPML pad (probes, point sources, lumped and wire ports), previously warned about, are now refused at preflight and before any step. Set `allow_in_absorber=True` on the individual add call to keep the pad position; the attenuation warning remains. Positions beyond the grid are always refused.
- On graded grids these pad points previously moved to the interior edge; the opt-in now uses the actual padded node index.
- The same rule covers batched material sweeps and differentiable material fits; the opt-in survives design export/load and automatic ringdown probes.
- Float32-rounded domain endpoints are admitted within the existing on-node band.
