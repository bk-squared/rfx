# T2 → owner of `rfx/api/_preflight.py`

Four tests in `tests/unit/sparams` / `tests/unit/ports` fail on branch
`feat/931-t2-sparams-ports` until preflight stops measuring metal from the
primal CELL mask (design note §6, "Not yet implemented"). All four are
pre-declared falsifiers for that migration; none is a tolerance to widen. The
expected post-migration numbers are written into the tests already, so the
preflight change flips them green without anyone re-deriving anything.

They are marked `@pytest.mark.xfail(strict=True)`, not left red. Strict is the
whole point: the moment preflight lands, the test XPASSes and pytest FAILS the
run, so the marker cannot be forgotten and the assertions cannot quietly stop
meaning anything. A red that everybody learns to skip is not a falsifier. When
you land the change, delete the marker — the body underneath is already the
post-migration expectation.

Sections 1 and 2 below are the two whose numbers this branch measured; section
3 covers the two added 2026-09-07 (`test_thru_preflight_code_set_is_the_contract_set`
in `test_lumped_twoport_vi_validation_battery.py`, and the referee-record gate,
whose owner is a diagnostics script and which has its own note).

## 1. `test_port_aperture_rasterization.py::test_sub_aperture_guide_is_measured_from_the_pec_walls`

`_port_transverse_spans` measures `guide` wall-to-wall on the assembled
`pec_mask` — it takes the OUTERMOST OCCUPIED CELL index as the wall. Under the
ownership contract a volume's drawn face IS a wall, one node beyond its last
occupied cell, so the reading is one cell short on each side.

Measured on that fixture (two PEC blocks filling y ∈ [0, 0.04] and [0.08, 0.12]
at dx = 2 mm, a declared 40 mm gap):

| quantity | pec_mask cells (today) | realized wall planes |
|---|---|---|
| inner indices | 29 and 50 | 30 and 50 |
| guide | 42.0000 mm | 40.0000 mm |
| fc_TE20 = c/guide | 7.138 GHz | 7.495 GHz |
| 0.90·fc_TE20 | 6.424 GHz | 6.745 GHz |

The test pins the right-hand column and the fixture band was raised to 7.0 GHz
so the `port_evanescent` advisory still has something to catch. The
build-time witness beside it,
`test_sub_aperture_walls_are_realized_where_they_are_drawn`, passes today: it
reads `realized_wall_planes` directly and confirms the inner walls are 20 cells
apart.

The docstring of `_port_transverse_spans` (rfx/api/_preflight.py, the
"Review measurement" paragraph) states the 42.0000 mm figure as the correct
one. It is the pre-#931 measurement and needs rewriting with the reading above;
`tests/unit/sparams/test_waveguide_port_reference_sims.py` carries the same
geometry and now asserts the 40.0 mm guide at the source.

Suggested rule: `guide` = the distance between the innermost realized wall
planes on the port's transverse line, from
`realized_wall_planes(edge_masks, axis, ij=<the port's own column>)`, with the
run's own periodic flags — the same call the test makes.

## 2. `test_mixed_port_sparam.py::test_wire_port_end_gap_advisory_fires_on_a_declared_one_cell_gap`

The #556 end-gap advisory (`_validate_cfg_port_inside_pec`, the
`wire_port_end_gap_to_conductor` branch) decides contact by reading
`mask_np[nb]` — "is the cell immediately beyond the port's end cell PEC?". A
SHEET owns no cell, so that question is false for every foil, and the advisory
cannot fire on a microstrip feed at all. Not a false negative on one fixture:
it is structurally blind to the conductor class the contract introduces.

The signature restated on the realization, which is what the test's ground
truth now uses:

* **contact** — the port's end NODE (`cells[-1][axis] + 1`) is a realized wall
  plane on the port's own column;
* **one-cell gap** — the realized wall plane is at `end_node + 1`, i.e. exactly
  one live edge separates the feed from the conductor;
* neither — nothing to say.

Both read from
`realized_wall_planes(edge_masks, axis, ij=(the two in-plane indices of the
port's cells), periodic=<the run's flags>)`.

The fixture: `_base_sim()` on an on-lattice board (h_sub = 254 µm,
dx = h_sub/3), foil sheet on node 3, feed drawn with `extent = dx` so its cells
are 0..1 and its top node is 2 — one live Ez edge (cell 2) short of the wall.
The silent controls beside it are `extent = 2·dx` (top node ON the wall:
contact) and a port in open vacuum.

One consequence worth stating in the advisory's own text: contact with a sheet
leaves NO dead cell, because the port drives Ez and Ez is normal to a
z-normal sheet. So the old partition — "#556 owns the gap, #319 owns contact
because contact means a dead end cell" — no longer holds, and the control
`test_wire_port_end_gap_advisory_silent_when_extent_reaches_conductor` now
asserts that BOTH are silent on a correct galvanic feed. That is the #929
finding in miniature: a feed that reaches its ground is correct geometry, and
the remedy text must never tell a user to move it off.

---

## 3. Two findings a correctly-declared sheet provokes today

Added 2026-09-07 from `tests/unit/sparams/test_lumped_twoport_vi_validation_battery.py`,
whose air-microstrip THRU trace became a sheet on this branch. Its preflight
report is now three findings where the contract says one
(`pec_faces_finite_pec`, intended — the infinite ground plane IS the microstrip
return). The extra two are preflight's, and the test that pins the
one-finding set is `test_thru_preflight_code_set_is_the_contract_set`,
`xfail(strict=True)` with both quoted.

### 3a. `mesh_resolution` tells the user to un-declare their sheet

Verbatim:

> Zero-thickness geometry 'pec' along z-axis. On non-uniform mesh this may
> produce empty rasterization. Consider giving it at least one cell of
> thickness (500µm).

Under §1.5 a zero-thickness PEC Box **is** the sheet declaration — "zero
thickness is a statement of intent, not an inference" — and it does not
rasterize to nothing: it realizes one wall plane. Following the remedy turns a
foil into a one-cell metal slab, which for this fixture is 0.5 mm of solid
metal on a 1.0 mm ground-to-trace gap. The advisory is exactly backwards now.

Replacement: fire only for a zero-thickness NON-PEC (dielectric) shape, whose
node-half-open sampling genuinely can produce an empty region. For a PEC shape
the zero-extent axis is a declaration; if anything is worth saying it is the
`sheet_plane_realized` NOTICE from §3 (declared plane, realized plane, offset),
not a thickness remedy.

### 3b. The uncoded sheets-dropped warning is preflight's own assemble call

Verbatim:

> `_assemble_materials` (uniform lane): PEC sheets/wires were classified but
> the caller passed no pec_sheets/pec_wires collector, so they are absent from
> the returned pec_mask (a sheet owns no cell, #931 §1.3). A caller that steps
> fields must pass collectors and realize them with
> `rfx.boundaries.pec.realized_pec_edge_masks`.

This is the warning working as designed, aimed at preflight itself (design note
§6, "preflight's own `_assemble_materials(grid)` calls take the sheets-dropped
warning"). It reaches the user's report as an uncoded finding, so every
sheet-declared board preflights with a warning about preflight's internals.

Replacement: pass the collectors at the three call sites (`_preflight.py:566`,
`1880`, `4883`) and realize the edges from them. Until then it should at least
not be surfaced as a finding — it is not an input-fidelity statement about the
user's geometry, which is what preflight is for
(`feedback_preflight_input_fidelity_only`).
