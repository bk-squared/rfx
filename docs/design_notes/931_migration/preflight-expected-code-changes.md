# #931 preflight stage — replacement text for files the preflight owner does not edit

Branch `feat/931-preflight`, worktree `research/rfx-931-P-preflight`. Everything
below is measured on that branch (commands in the stage report); the merge /
ingest agents apply it. Nothing here is a number to hand-edit into a fixture —
where a golden moves, the producer is named.

## 1. Preflight finding codes: what changed and why

| code | change | reason |
|---|---|---|
| `pec_box_subcell` | NEW, severity `error` | design note §3 / §1.5: a PEC shape thinner than one local cell is refused at assembly; preflight reports the classifier's own message (physical thickness + local cell) so the whole configuration is audited before `run()` raises. |
| `pec_zero_cells` | NEW, `error` | §3: a PEC volume no cell centre falls inside, or a sheet footprint reaching no node (#369 class). |
| `pec_realization_refused` | NEW, `error` | every other classifier refusal (a line/point Box, a sheet declared > ½ cell from the node line, `add_thin_conductor` thicker than one cell). Not in the note; added so no refusal is uncoded. |
| `pec_box_one_cell` | NEW, `warning`, aggregated | §3: a PEC volume exactly one cell thick along some axis — a filled slab with walls on both faces; names the sheet declaration as the foil remedy. Fires on every unmigrated foil-as-1-cell-Box by design. |
| `sheet_plane_realized` | NEW, `info` (aggregated, one per run); a second message at `warning` when a sheet mid-plane sits on an exact half-cell tie | §3: declared mid-plane, realized node plane, offset per sheet; the tie warning names both candidate planes and the chosen (lower) one. |
| `sheet_slot_vacuum` | NEW, `warning` (message starts `ERROR-GRADE:`) | §3: the sheet's node plane carries vacuum while the cells on both sides carry dielectric (a stack drawn with a slot for the foil). Replaces what the #702 resample used to absorb. Not `error` severity: the run is well-defined, the stack is just not the drawn one. |
| `sheet_live_edge_material_mismatch` | REMOVED | #703 check 2 guarded the #702 resample, deleted by §2. A sheet owns no cell, so there is no "own cell" material to compare. |
| `congruent_conductor_rasterization_parity` | kept; compares realized PEC **edge** counts, message says `N PEC edges` / `edge-count spread`; gate constant renamed `_CONGRUENCE_SPREAD_TOL_CELLS` → `_CONGRUENCE_SPREAD_TOL_EDGES` (still 1); census keys on `(shape class, realization kind, extents)`; PEC thin conductors are in the census | §1.7: the edge set is what decides whether the lattice kept a symmetry; a sheet has no cell count. |
| `sheet_cavity_electrical_thickness` | kept; census = sheet normal planes ∪ volume faces per axis (from `realized_wall_planes` on each entry's own edges); cavity = adjacent planes of two different conductors over a shared footprint; mesh Σ over the cells strictly between the planes vs the physical stack between the DECLARED faces; message prints each plane's snap (`snap -400µm`) and no longer has the "OWN cell" / "TWO MECHANISMS" text | §3 / #767: the far face of a volume is a realized wall the check can now see; the only mechanisms left are the plane snap and the slot. |
| `off_lattice_design_edges` | kept; no sub-cell exclusion; sheets' in-plane axes examined, their normal axis reported by `sheet_plane_realized` | §3: there is no node-thin snap any more. |
| `campaign_statics_unavailable` | kept; text renamed to the three surviving #703 checks + the §3 findings; silent when a refusal already explains why the assembly failed | |
| `port_in_pec` | kept; rule = an E component is dead iff its own edge is realized PEC, an H component iff all four E edges of its curl loop are PEC; the `≤ 1.5·dx` thin-PEC exemption is gone; text now `Port/source/probe at … (ey) is frozen by realized PEC geometry [...]: its own E edge is a realized PEC edge …` (was `is inside PEC geometry '…'. Field will be zero.`); runs on the NU lane too | #929 / §1.9: liveness from the realized edge set. An Hy probe at the centre of a 1-cell VOLUME trace now (correctly) warns; beside a SHEET trace it stays silent. |
| `wire_port_dead_extent_cells` | kept; classification through `_wire_port_live_cells` on the realized edges (sheets visible); text `… of which N have their ez edge inside realized PEC [...]` (was `land inside PEC geometry`); remedy tail rewritten (no "thin PEC sheet snaps to its nearest grid NODE") | §1.9 |
| `wire_port_end_gap_to_conductor` | kept; ONE definition of contact: the wire's end node lies on a realized wall plane of its own column (`realized_wall_planes(ij=)`); fires only when the end node carries no wall and the node one cell further does; text names `realized wall plane of [...]`; remedy `… so its end node lands on that wall plane, or draw the conductor so its face (a volume) or its plane (a sheet) lands on the wire's end node` | #556 / #929: a port starting ON a volume face or a sheet plane is galvanic and draws nothing. |
| `port_evanescent` / `port_aperture_snap` | kept; the guide is the distance between realized wall planes bracketing the aperture | #868: the sub-aperture fixture now reads **40.0000 mm** (was 42.0000). |
| `coaxial_port_junction_short` | kept; counts NODES carrying a tangential wall (realized set) in the ring `a + dx/√2 < r ≤ min(a + dx/√2 + dx, shell_inner)` (was `a + dx/2 … a + 3dx/2` on the cell mask) | the pin is centre-sampled now and its own nodes reach `a + dx/√2`; the old bound would count the pin's own rim. Needs the coax–MSL fixture re-run to re-pin 0/16 vs 16/16 (owner: tests-sparams). |
| `ntff_pec_overlap` | kept; closed on the entry's realized wall planes along the normal axis; PEC thin conductors in the census | §3 |
| `ntff_small_ground_plane`, floating single-cell port | kept; PEC thin conductors added to the census | |
| `thin_metal_nu_mesh` | kept; examines each REALIZED plane (sheet plane; each face of a slab ≤ 2 cells thick) and its two neighbouring cells; text `Thin PEC geometry[i] ('pec', realized as a sheet) has a metal plane at z = … (node k) …` | §1.3: the plane is read from the realization, not a bbox midpoint. |
| `mesh_resolution` (PEC branches) | zero-thickness PEC Box: silent (it IS the sheet declaration); sub-cell PEC: hint text names the §1.5 refusal and the sheet declaration (was "modelled as a one-cell PEC surface … add_thin_conductor would not change it"); 3–5 cell PEC text `a 1-2 cell slab is realized as drawn, with walls on both faces` | #929 class |
| `pec_faces_finite_pec` | REMEDY names the sheet for a foil and a Box volume for a plate | |
| `msl_port_geometry` (2/2b/2c) | `_msl_realized_substrate` starts the eps walk ABOVE the realized ground (skips a volume ground's own cells, returns `ground_cells`); text "a PEC Box is a VOLUME … a foil trace declared as a SHEET … neither enters pec_occupancy_override" | §1.9 |
| `graded_box_rasterization` (#325) | PEC Boxes counted with the §1.1 centre rule (`_box_axis_volume`); dielectrics unchanged | |

Preflight's four `_assemble_materials(grid)` sites now all go through
`_assemble_realized(grid, nonuniform=)` (sheet/wire collectors + the run's
periodic flags), so preflight never triggers `_warn_uncollected_pec` any more.

## 2. `tests/unit/ports/test_port_aperture_rasterization.py` (owner: tests-sparams-ports)

`test_sub_aperture_guide_is_measured_from_the_pec_walls`: measured on this branch,
the fixture at `freqs = linspace(4.5e9, 6.5e9, 3)` emits NO `port_evanescent`
finding — the realized guide is 40.0000 mm (walls at nodes 20 and 40 = the drawn
y = 40 / 80 mm faces), fc_TE20 = 7.495 GHz, threshold 6.745 GHz, and 6.5 GHz is
below it. Replacement (measured with `linspace(4.5e9, 7.0e9, 3)`):

```
Waveguide port 'left': max measurement frequency 7.000 GHz exceeds 0.90 × fc_next=6.745 GHz
on the REALIZED guide (40.0000 mm (y, pec_walls) x 20.0000 mm (z, aperture))
(fc_TE10=3.747 GHz, fc_TE01=7.495 GHz). …
```

so: raise the fixture's top frequency to 7.0 GHz, assert `"40.0000"`, `"7.495"`,
`"6.745"` (and that `"42.0000"` is absent), and rewrite the comment block: the
lower Box `[0, 0.04]` realizes its far-face wall AT y = 40 mm, the upper Box its
near face at 80 mm (#931 §1.2); the 42 mm number was the cell-mask scan's
(#868 class). `test_margin_heuristic_is_evaluated_on_the_rasterized_guide` is
unchanged (domain-face guide, measured 24.0000 × 12.0000 mm, 11.242 GHz).

## 3. `tests/_waveguide_chain_battery_fixture.py::transverse_spans` (shared fixture)

`sim._port_pec_mask(grid)` is kept on this branch as a NAMED ALIAS that returns
the realized conductor set (`_RealizedPEC`), not a cell mask, so the fixture runs
unchanged. Replace the two lines with

```python
    realized = sim._port_realized_edges(grid)
    ...
    spans = sim._port_transverse_spans(entry, grid, realized)
```

and delete the alias from `rfx/api/_preflight.py`. The docstring of
`numerical_te10_cutoff_hz` ("the wall-to-wall extent measured on the assembled
PEC mask") becomes "the distance between the realized wall planes".

## 4. `tests/data/example_fidelity_snapshot.json` (owner: examples, regenerated LAST)

Rows that move because of this stage (all preflight-code rows):

* every `sheet-own-cell-live` / `sheet_live_edge_material_mismatch` row: gone;
* every model with a sheet gains one `sheet_plane_realized` (info) row; a
  face-registered 1-cell foil migrated to `add_thin_conductor` with faces gains
  the tie WARNING as well;
* every unmigrated 1-cell PEC Box gains `pec_box_one_cell`;
* `port_in_pec` rows: Hy probes inside 1-cell VOLUME traces now warn; probes
  beside SHEET traces do not; message text changed (see §1);
* `wire_port_dead_extent_cells` / `wire_port_end_gap_to_conductor` text changed;
  the end-gap predicate is the wall-plane one — re-read the D5 fixture rows;
* `port_evanescent` guide labels on PEC-walled fixtures move by one cell;
* `msl_port_geometry` rows: text changed, and on fixtures whose ground was a
  1-cell Box drawn from the port plane the substrate walk now starts above the
  ground's top face (was: degraded to the scalar estimate after #702 deletion);
* `coaxial_port_junction_short` rows: ring bounds moved (§1) — re-pin from the
  regenerated snapshot, not by hand.

## 5. Other test files (owners named)

* `tests/unit/geometry/test_fidelity_report.py:76` asserts `"sheet-own-cell-live"
  not in _kinds(ground)` — still true (the kind no longer exists); keep or delete.
* `tests/unit/geometry/test_fidelity_topology_findings.py:63` docstring names
  `self._port_pec_mask`; rename to `_port_realized_edges` when the alias goes.
* `tests/unit/materials/test_sheet_impedance.py:1449` quotes the old
  `is inside PEC geometry` text in a comment — update the quote (`is frozen by
  realized PEC geometry`).
* `tests/unit/sparams/test_coax_msl_transition.py:2799` names the
  `sheet-own-cell-live` kind in a message string; and its 11 cell-mask reds
  (F.md) need the junction check's new ring (§1) — re-pin from the re-run.
* `tests/unit/sparams/test_mixed_port_sparam.py:1073-1164` (D5 end-gap): the
  predicate is now "end node on a realized wall plane"; on the D5 board
  (dx = 80 µm, h_sub = 254 µm) the trace plane moves (critic.json, k = 4 → 3),
  so re-measure which geometry fires; do not translate the old expectation.
* `tests/contracts/test_example_fidelity_contract.py:23` lists
  `sheet_cavity_electrical_thickness` among preflight kinds — unchanged name.

## 6. `rfx/api/_compile.py::_warn_uncollected_pec` — flipped to a raise (done)

This section asked for the flip once preflight passed collectors everywhere,
and listed the collector-less callers that remained in files this stage did not
own. Those callers were threaded on branch `feat/931-core-collectors`: the
helper is now `_refuse_uncollected_pec` and raises `ValueError` naming the
caller (file, line and function), so a future collector-less caller cannot
appear. Callers that read cells only pass `pec_sheets=[], pec_wires=[]` and
drop the result; the two lanes that cannot realize a sheet — the coaxial
S-parameter lanes behind `_build_materials`, and the SBP-SAT subgrid validator
— refuse it by name instead. See §1.9 of the design note.

## 7. CHANGELOG lines (owner: docs)

Under `## [Unreleased — 2.0.0]` / preflight:

* NEW findings `pec_box_subcell`, `pec_zero_cells`, `pec_realization_refused`
  (errors), `pec_box_one_cell` (warning), `sheet_plane_realized` (info; warning
  on a half-cell tie), `sheet_slot_vacuum` (warning, error-grade) — the lattice
  ownership contract's per-declaration realization report (#931 §3).
* REMOVED `sheet_live_edge_material_mismatch` (#703 check 2) with the #702
  resample it guarded; `_LIVE_EDGE_RTOL`, `_CAVITY_SHEET_CELL_FILL_FRAC`,
  `_CAMPAIGN_SUBCELL_FACTOR` gone; `_CONGRUENCE_SPREAD_TOL_CELLS` →
  `_CONGRUENCE_SPREAD_TOL_EDGES`.
* Every conductor-reading preflight check (`port_in_pec`, the wire-port
  advisories, the waveguide guide width, the coax junction ring, the NTFF wall
  test, the sheet-cavity report, thin metal on NU) reads
  `realized_pec_edge_masks` / `realized_wall_planes` / `edge_is_pec` instead of
  a cell mask or a bounding box; the `≤ 1.5·dx` thin-PEC H exemption is gone
  (#929); a sub-aperture waveguide fixture's guide reads 40 mm, not 42 (#868).
* Preflight builds its conductor context once per configuration
  (`Simulation._campaign_ctx`).

## 8. Recompute

None owed by this stage. Preflight is input-fidelity only and owns no solved
artifact; no VESSL run was submitted. The artifacts that move because of the
codes above (`tests/data/example_fidelity_snapshot.json`, the coax–MSL
junction re-pin, the D5 end-gap rows) are regenerated by their owners
(examples, tests-sparams) with their own producers, after this branch merges.
