# T1 → preflight: rule (ii) reads cells, so a sheet ground is invisible

Owner: `rfx/api/_preflight.py` (group P). Written, not applied.

`_check_coaxial_port_junction_aperture` reads `self._port_pec_mask`, a primal
CELL mask. Two consequences measured on
`tests/unit/geometry/test_fidelity_topology_findings.py::_junction_sim`
(VESSL 369367259135, commit 299b0f9f):

1. TODAY, with the ground still drawn by the `_half_cell` midpoint recipe, the
   foil realizes on cell 24 (walls on planes 24 and 25) while the coaxial
   port's junction plane index is 25. `pec_mask[:, :, 25]` is empty there, so
   `coaxial_port_junction_short` no longer fires on the shorted copy and the
   four `test_rule_ii_*` tests are red.
2. AFTER the fixture migrates (the ground is a sheet at node plane 25, which is
   what it means), the cell mask is empty on every plane and the check sees
   nothing at all.

Both are fixed the same way: the junction-plane metal census is
`realized_wall_planes` / the sheet footprints unioned with the volume cells, on
the NODE line the port axis sits on — not `pec_mask`. The node reading also
restores the historic counts exactly: annulus 36/36, first lattice ring 16/16,
lip 32/32, pinned green in
`test_fidelity_topology_findings.py::test_junction_plane_metal_under_a_sheet_ground`.

The pin-footprint precondition in
`test_rule_ii_is_silent_on_a_correctly_built_hole_with_the_pin_present`
(11 nodes at `r <= PIN_R`) reads 10 under centre sampling of the pin Cylinder
and will need re-deriving with the rest.
