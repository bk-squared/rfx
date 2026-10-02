# Public realized geometry (#1138, part A)

`Simulation.realized_geometry()` lazily builds a cached, frozen host diagnostic
record. `Result.realized_geometry` is a separate compact record built from the
run's own assembly, including when `skip_preflight=True`. The assemblers collect
original entry identities and their actual cells, sheets and wires; the Result
builder neither assembles again nor consults preflight. No stepping kernel changes.
For traced mesh coordinates, the run returns `realized_geometry=None`; other
record errors propagate. NU waveguide apertures use the runner’s node helper.
Assembly refusals still raise. Custom runner result objects are returned unchanged.

Lengths are metres, node ranges inclusive, cell ranges half-open, and residuals
signed solved minus declared. Entities carry kind, declared bounds, realized
ranges/extents, cell/node and component-edge counts, edge ranges, sheet plane,
wall planes and continued faces. Comparison extents/counts exclude absorber
continuation. Sheet spans use `solved_sheet_span`, including free edges and seams.
Result records retain no dense entity/global masks, sheet/wire specs or material
arrays. Diagnostic arrays are available by requesting `sim.realized_geometry()`.
Refused diagnostic rows retain occupancy and errors; execution still raises from
its assembly. Immutable arrays have byte backing.

The cache keys include geometry, thin conductors, pinned sheets, materials, mesh
and every port family. Port registration therefore invalidates even a context
built before the ports. Material masks are keyed by original entry identity,
independent of collection order. Rule (i) compares material cells to conductor
cells; a sheet contacts a cell iff its plane bounds that cell and all four
in-plane corners of the contacting face are metal. Both adjacent layers count.
The junction hole control has zero contacts; its solid-ground copy has 104
(52 cells on each side), versus the former 52 coincident cell/node indices.

Domain rows give cells, lengths and per-face padding. Lumped/wire ports identify
driven edges, waveguide ports apertures. MSL/coaxial/Floquet rows explicitly say
`not represented` for family-specific aperture/drive data. `lane` identifies the
runner; subgrid records describe the coarse grid and explicitly mark the refined
region unrepresented. Per-entity planes remain accessible in compact records;
whole-model spatial wall queries require the diagnostic record.

Fidelity, the test wrapper and the ports/S-parameters, RCS, patch and nonuniform
patch tutorials read the diagnostic record. Preflight retains `_EntryRealization`
rows; diagnostic records reuse those rows. Runner records use the same summary
code with entry products collected directly from that run. Contracts capture
runner arguments and compare their PEC edges/ranges/counts, including stale
preflight controls on uniform and nonuniform lanes.

Parts B/C will build reader/user geometry operations on this API. PI decisions:
public Result record (2026-09-24); part 3 defaults to snap and an error above 1%
residual (2026-10-02). Those future operations are not implemented here.
