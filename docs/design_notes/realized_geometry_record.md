# Public realized geometry (#1138, part A)

`Simulation.realized_geometry()` returns a cached, frozen, host-side
`RealizedGeometry`. `Result.realized_geometry` holds that same object for the
configuration used by `run()`. Lengths are metres; node ranges are inclusive,
cell ranges half-open, and face residuals are signed solved minus declared.
Arrays have immutable byte backing. No stepping kernel or tracing path changes.

The record contains per-entity kind, declared bounds, realized axis ranges and
extents, cell count, masks, PEC edges and wall planes. Sheets carry their plane
and `solved_sheet_span`, including free-edge offsets and seams. Material masks
are collected by identity at the production write site, so a skipped PEC row
cannot shift another material's mask. Curved materials use the production cell
centre sampler; the former fidelity node sampler was a separate computation.
Refused declarations carry diagnostic occupancy (`occupancy_role="diagnostic"`)
and their refusal; this occupancy is not solved metal and contributes no edges.

Domain rows give interior cell counts, realized lengths, declared lengths and
per-face padding. Lumped/wire ports list driven edges and waveguide ports list
apertures. MSL, coaxial and Floquet rows explicitly say `not represented`:
family-specific aperture/drive assembly is not exposed by this part.

Fidelity, the test realization wrapper and the ports/S-parameters, RCS, patch
and nonuniform-patch tutorials read this record. Preflight retains its shared
`_EntryRealization` rows, which also supply the record's conductor data. The
legacy fidelity `_entity_mask` name is a record accessor, not a rasterizer.
The record is cached per mesh/geometry/material context and port configuration;
configuration edits obtain a new record and existing Results retain the old one.
The runner owns its unsupported-continuation warning; host record construction
does not emit another copy of that warning.

Parts B/C can build reader and user-facing geometry operations on this public
record; they are not implemented here. PI decisions: the record is public on
Result (2026-09-24); part 3 will default to snapping and emit an error finding
above 1% residual (2026-10-02). This part exposes measurements, not that future
snapping policy.
