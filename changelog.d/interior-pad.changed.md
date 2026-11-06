Points in the CPML pad (probes, point sources, lumped and wire ports), previously
warned about, are now refused when the grid is resolved at preflight/solve.
Set `allow_in_absorber=True` on the individual add call to keep the pad position;
the attenuation warning remains. Positions beyond the grid are always refused.
On graded grids these pad points previously moved to the interior edge;
the opt-in now uses the actual padded node index.
