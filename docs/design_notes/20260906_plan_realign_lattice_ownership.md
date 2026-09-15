# Lattice ownership contract for conductors — design + migration plan (2026-09-06)

Status: **DECIDED by the PI 2026-09-06** (issue #931). This note is normative for the
implementation branch `feat/931-lattice-ownership`. It amends the 2026-09-05 post-v1.8
plan's item 1 from "visualize the rasterization" to "define and enforce the ownership
contract"; the visualization work then reads the contract instead of re-deriving.

## 0. The defect in one paragraph

`Box.mask_on_coords` samples a Box half-open `[lo, hi)` at E-NODE coordinates and the
array is called `cell_mask`. `rfx.boundaries.pec.tangential_edge_masks` zeroes `E_a` at
entry `c` iff `c` is masked and a neighbour of `c` along axis `a` is masked (the #677
"thin-sheet neighbour rule"). Net effect, measured on synthetic slabs (issue #930/#931
tables): a PEC body is realized as a stack of sheets, one tangential wall per masked node
plane, at that plane. The far (`hi`) face is never a wall at any thickness; in-plane, the
`hi` row of a footprint is missing the same way. `two_plane` (#706) puts the far plane back
for `t = 1` only. #702 re-samples a 1-node sheet's "own cell" material at its live edge.
cv19 feeds its oracle `(L_c + 1)·dx`; cv15 draws the ground one cell below the substrate
floor and needs `two_plane` on one sheet and not on the other. Every one of those is a local
repair of the same undeclared semantics. The rule itself was a correct sheet rule; it was
never declared as one, and the primitive it was attached to is called `Box`.

## 1. The contract (normative)

A conductor is exactly one of three things, and the declaration says which.

| kind | what it is | declared by | E edge is PEC iff |
|---|---|---|---|
| **volume** | a set of primal cells | `sim.add(shape, material=<pec>)` — Box, Sphere, Cylinder | the edge is incident to an occupied cell |
| **sheet** | a footprint on ONE node plane, zero thickness | `sim.add_thin_conductor(shape, ...)` (PEC by default; lossy with `surface_impedance_f0`) | the edge lies in the plane and both of its end nodes are in the footprint |
| **wire** | a 1-D path of edges | `PolylineWire` | the edge lies on the path |

One sentence covers all three: **an E component is PEC iff its own location is inside the
closed conductor region.** The three rows are that sentence evaluated for a 3-D, 2-D and
1-D region on the Yee lattice.

### 1.1 Index conventions (unchanged, now written down)

* Node `i` sits at `x_i = (i − pad)·dx` (uniform) or at the cumulative edge position (NU).
  Primal cell `i` spans `[x_i, x_{i+1}]`; node `i` is its LOWER corner.
* `Ex[i,j,k]` is at `(x_{i+½}, y_j, z_k)`, `Ey[i,j,k]` at `(x_i, y_{j+½}, z_k)`,
  `Ez[i,j,k]` at `(x_i, y_j, z_{k+½})`.
* The occupancy array `C[i,j,k]` means **primal cell `(i,j,k)` is conductor**. For PEC
  VOLUMES it is sampled at **cell centres**, half-open: cell `i` is occupied iff
  `lo ≤ x_{i+½} < hi` (Box), or the centre lies inside the shape (Sphere, Cylinder). On node
  planes this gives exactly the cells the node sampler gives (`i0 .. i1−1`); off-lattice it
  rounds each face to the NEAREST plane instead of always outward (node-half-open + far
  face) or always inward (the old rule), and a corner drawn on a cell midpoint — the cv18
  "midpoint recipe" — lands on the plane it intended (lo inclusive, hi exclusive at the tie).
  A Sphere centred on a node realizes symmetric about that node (the old rule was one cell
  short on every `+` side). **Dielectric sampling is untouched** (node, half-open), so
  every dielectric-only fixture stays bit-identical; for node-aligned corners PEC and
  dielectric cells coincide by construction. NU grids: centres are `node + d/2` from the
  per-cell size arrays; a traced mesh keeps the traced path.

### 1.2 Volume realization

```
Mx[i,j,k] = C[i,j,k] | C[i,j-1,k] | C[i,j,k-1] | C[i,j-1,k-1]
My[i,j,k] = C[i,j,k] | C[i-1,j,k] | C[i,j,k-1] | C[i-1,j,k-1]
Mz[i,j,k] = C[i,j,k] | C[i-1,j,k] | C[i,j-1,k] | C[i-1,j-1,k]
```

The backward shifts use the #689 boundary convention already spelled in
`_axis_neighbors`: explicit zero pad on a non-periodic axis, wrap on a periodic axis and on
a length-1 axis. Consequences that follow, and that the contract test pins:

* a Box drawn `z_a → z_b` on node planes realizes tangential walls at BOTH `z_a` and `z_b`
  and shorts every normal edge between them; realized thickness = drawn thickness;
* a 1-cell PEC Box is a filled slab with two faces (what `two_plane` produced), at every
  thickness, on every axis, with no flag;
* a body touching the domain face at cell `n−1` has its far face on plane `n`, which the
  domain BC owns (same as today);
* the rule is invariant under mirror and axis permutation of the geometry.

### 1.3 Sheet realization

A sheet is `(normal_axis a, plane index k, footprint F)` where `k` is a **static integer**
and `F` is a boolean node mask on the plane. For a Box shape the footprint is sampled
**closed** `[lo, hi]` on the two in-plane axes (so the drawn rectangle is realized exactly,
including its `hi` row); for any other shape (a patterned `MeshShape`, a `Cylinder` pad,
an imported outline) the footprint is the shape's **cross-section at its own mid-plane**
along `a` — `F = shape.mask_on_coords(x, y, [z_mid])` — and `k` is the node plane nearest
that mid-plane (tie → lower). The footprint never depends on how many node planes the
shape's thickness happens to straddle, so a 17 µm foil declared on a mesh with nodes at
both its faces still realizes as ONE plane. Realized edges: for `a = z`,

```
Mx[i,j,k] = F[i,j] & F[i+1,j]      (edge from node i to i+1 at row j)
My[i,j,k] = F[i,j] & F[i,j+1]
Mz                                  unchanged — normal E through a sheet stays live
```

and cyclically for `a = x, y`. A sheet owns NO cell: it adds nothing to `C`, writes no
`eps_r`/`sigma`, and is realized at exactly one plane. Plane selection for a declared
sheet: the node plane nearest the shape's mid-plane along `a`; an exact half-cell tie
resolves to the LOWER plane (today's `n_vol == 1` rule for a face-registered 1-cell Box,
kept so existing declarations land where they land today). `fidelity_report` and preflight
print the realized plane in physical units.

Footprints of all sheets on the same `(a, k)` plane are UNIONED before the edge rule is
applied, so two abutting sheets realize seamlessly (per-sheet application would leave the
shared edge live — a slit). Sheets on adjacent planes stay two films with a live normal
edge between them (#690 semantics). A zero-thickness Box (`lo == hi` on the normal axis)
is the canonical way to say "sheet on this plane". A sheet whose declared mid-plane sits on
an exact half-cell tie (a face-registered one-cell Box) gets a preflight WARNING naming
both candidate planes and the one chosen. On a length-1 axis (2-D lane) a sheet whose
normal is that axis has no tangential components; it is realized as a 2-D volume of its
footprint cells, with a notice.

**Off-lattice interfaces.** A sheet meant to sit on a dielectric interface must have that
interface on a node plane. If it does not (the canonical dx = 80 µm / h_sub = 254 µm
microstrip: nodes at 240 and 320 µm), the sheet snaps to the nearest node — 240 µm, inside
the substrate — while the old rule happened to put its single wall at 320 µm, on top of the
substrate the mesh realized as four cells. Neither is the declared board. The contract does
not paper over this: the assembly warns when a sheet plane lies strictly inside one
dielectric (same material on both sides), preflight reports the declared-vs-realized
offset, and the fixtures are redrawn ON-LATTICE — the #325/#802 class, now made visible
instead of absorbed by a tie rule.

**On-lattice is two conditions, not one.** `dx = h_sub / n` SIZES the laminate: it makes
the thickness a whole number of cells. It does not PLACE it — the stack's own origin has
to sit on a node line too, or every face in it is off by the same fraction of a cell. Both
conditions, or neither is satisfied: a ground declared at 4 mm on a 196.75 µm mesh realizes
at 3935 µm however exactly `h_sub` divides (measured 2026-09-07 on the public
materials-geometry example, which stated the shorthand and demonstrated the failure it
warns about). Say the plane in cells (`Z_GND = 20 * DX`) and the question does not arise.

**Mesh lifetime during construction (2026-09-08).** A grid preview is provisional
when automatic resolution is active: adding geometry can change its spacing and
profiles. Before placing geometry from a resolved grid, explicitly finalize it
with `grid = sim.freeze_mesh()`, after adding the geometry/materials that must
drive resolution. Spacing, profiles and extent then stay fixed through subsequent
additions. Those additions still face the same sub-cell and realization refusals;
freezing is not a substitute for resolving a feature. Use a new Simulation to
remesh. Explicit `dx=` and fixed profiles from the outset remain valid. Read
physical node coordinates on NU grids, and register ports/boundaries before
retaining array indices since they can change padding. Ordinary grid reads and
preflight remain previews; they must not silently freeze automatic resolution.

A lossy (`surface_impedance_f0`) sheet uses the SAME footprint and the SAME edge set; the
#677 G4 identity ("f0 toggles loss, never geometry") is then true by construction, not by
a test that compares two rules.

### 1.4 Wire realization

`PolylineWire` with `radius ≥ ½` local cell is a volume (centre-sampled tube). Below that
it is a filament: the E edges of the axis-aligned lattice path joining the nearest nodes of
consecutive vertices. Only axis-aligned segments are supported in this change (a diagonal
segment raises; today such a wire silently rasterizes to disconnected nodes and realizes
nothing). No crossval or example uses `PolylineWire`; four tests do.

### 1.5 What `sim.add(Box, material=pec)` refuses

* A PEC Box with **exactly one zero-extent axis** (`lo == hi` there) IS a sheet
  declaration — zero thickness is a statement of intent, not an inference — and is realized
  exactly as `add_thin_conductor` would realize the same Box (plane = nearest node to the
  declared plane, tie → lower; an off-node plane is reported as a NOTICE with its offset).
  This keeps the documented five-line patch example and `first-patch.mdx` valid. Two or
  three zero-extent axes raise (a line or a point is not a conductor; use `PolylineWire`).
* A PEC Box with `0 < extent < one local cell` along any axis raises
  `ValueError: ... a Box is a volume; declare a sheet (a zero-thickness Box or
  add_thin_conductor) or resolve the thickness`. Nothing is inferred from raster thickness
  or drawing direction.
* A PEC Sphere / Cylinder / Box that rasterizes to ZERO cells (a via thinner than ~0.7 cell,
  a post between cell centres) raises, naming `PolylineWire` for a filament and the minimum
  radius for a volume — the #369 silently-vaporized-metal class, now an error.
* `add_thin_conductor` with a shape thicker than one local cell along its normal raises
  ("not a sheet; use add() for a volume").
* `two_plane` is gone. Passing it is a `TypeError` (unknown keyword), not a deprecation.
* There are no per-entry realization knobs. A test greps `rfx/` for `two_plane` and for any
  `realization=` style keyword on geometry entries and fails on a hit.

### 1.6 Soft (differentiable) path

`apply_pec_occupancy(state, occ, periodic)` uses the noisy-OR of the four incident cells,
`M = 1 − Π(1 − o_c)`, with the same #689 shifts. At binary occupancy it is bit-identical to
1.2 (pinned on a shape battery that includes non-periodic seams). Sheets enter the soft
path as static masks OR'd in (their plane is not a traced quantity — the `argmin` cliff).
`rfx/topology.py` gets no sheet or two_plane logic.

### 1.7 One source, every consumer

`rfx.boundaries.pec.realized_pec_edge_masks(occupancy, sheets, periodic) → (Mx, My, Mz)`
is the only function that turns geometry into PEC edges. Consumers that today re-derive
from `pec_mask` switch to it or to the two helpers built on it:

* `realized_wall_planes(axis, region=None)` — sorted node-plane indices where a tangential
  wall exists (preflight #703/#767 cavity checks, #729 declared-vs-realized, cv15
  `assert_realized_stack`, oracles);
* `edge_is_pec(component, i, j, k)` — wire-port live-cell logic (#556 end-gap, #929
  `port_in_pec`), probes, sources.

Consumers: `simulation.py` / `nonuniform.py` step functions, `runners/distributed_nu.py`
shmap twins (the three edge masks are sharded along x like the cell mask — a PEC sheet must
not vanish on the distributed lane), `visualize.py`, `fidelity.py`, `_preflight.py`,
`sources/sources.py` (`_wire_port_live_cells`), `sources/coaxial_port.py`,
`probes/probes.py`, `probes/sparam_driver.py`, `probes/msl_wave_decomp.py` (finds the MSL
trace by scanning a `pec_mask` column above the substrate — a sheet trace is not in
`pec_mask`, so it must read `realized_wall_planes`), `api/_execute.py` (2-D lane
`pec_mask[:, :, 0]`), `api/_sparams.py`, `materials/thin_conductor.py`
(`build_sheet_impedance_ctx`), `vmap_sweep.py`, `interop/_design.py` (an IR document
carrying `two_plane` is rejected with a message, not ignored).

### 1.8 Scope fences (each is a decision, not an omission)

* **Domain-boundary PEC** (`BoundarySpec` faces, `apply_pec` / `apply_pec_faces`) is NOT a
  body and keeps its convention (E_tan = 0 on the face plane at index 0 / N). cv09, cv10,
  cv14, cv24, `adi_solver_demo`, `hello_world`, `resonance_harminv` are the controls: they
  must not move.
* **Sigma-fill conductors** — `rasterize(..., sigma=1e7)` in cv16 / `rcs_scattering`,
  `stamp_coaxial_line`'s shell and pin, the DC-fold `add_thin_conductor` (`sigma_bulk <
  1e6`, no `f0`) — are a LOSSY VOLUME model (fields decay inside a conductive cell); they
  are not PEC realization and are unchanged here. Their own node-vs-cell debts (the coax
  shell one cell inside `b`, `r_os = b + 2dx`) get a follow-up issue. A test pins that the
  two models are not silently equated.
* **Kottke Stage-2 (`subpixel_smoothing='kottke_pec'`) and Dey–Mittra Stage-1 conformal**
  paths are subpixel models with their own interior selection; unchanged. Their binary
  parts are a follow-up (they already realize both faces for node-aligned Boxes).
  **The Kottke rule, stated (2026-09-08, review finding P1-2).** "Unchanged" is not
  self-executing: the step body applies `(Mx, My, Mz)` and on the Kottke lane that
  overwrote the tensor's fractional selection. On `kottke_pec` an E edge is zeroed iff
  the inverse-permittivity tensor froze it (`inv < 1e-9`) **or** a SHEET or a WIRE owns
  it — `rfx.boundaries.pec.kottke_fenced_edge_masks`, called once in
  `_build_step_setup`. Two terms, two reasons: the tensor is the volume's only owner
  there, and an edge it left with a positive fraction is the subpixel answer, not a
  cell to freeze; sheets and wires own no cell, so they never reach
  `compute_inv_eps_tensor_diag` and would vanish under an `inv`-only rule. Port clearing
  survives — the fence only removes entries from a set that arrives already cleared.
  Measured on a PEC `Sphere(r = 2.1 mm)` at `dx = 1 mm`: 16 edges per component carried a
  positive Kottke inverse permittivity and were hard-zeroed; `Ex[4,3,4]`, at
  `inv = 0.03291842`, read exactly 0 after 60 steps and reads 0.00444051 fenced. It does
  NOT read the pre-#931 rule's 0.00926519 and must not: that rule zeroed its own wrong
  subset of fractional edges. The fence applies only to the Stage-2 tensor built from
  the run's own `pec_shapes`; the occupancy-derived tensor (`aniso_inv_eps_smooth`, the
  `RFX_PEC_OCC_KOTTKE` lane) keeps the realized edges, because there the static
  declaration and the traced override are two different geometries and the intersection
  would silently delete declared metal. Pinned by
  `test_kottke_owns_its_own_volume_the_ownership_rule_does_not_overwrite_it` and
  `test_the_kottke_fence_keeps_sheets_and_wires_the_tensor_cannot_see`. The CONFORMAL
  amendment in §6 is a different path and is untouched by this: Dey–Mittra is an
  update-coefficient model that realizes no geometry of its own, so a conformal run
  still applies the realized edges in full.
* **Dielectric sampling** (node, half-open) and the `DesignRegion` / `eps_override` index
  mapping (`optimize.py`, `topology.py`, inverse-design examples) are unchanged. The design
  region's inclusive `+1` is a separate, documented debt (#729 class), not touched here.
* **`forward(pec_mask_override=)`** is a VOLUME override (cells) and stays one.

### 1.9 Consumers that today scan `pec_mask` for metal (all switch to §1.7)

* wire-port live/dead (`_wire_port_live_cells`): live iff the port component's edge at that
  index is not in `(Mx, My, Mz)`;
* port "clearing" (wire live cells, lumped cell, MSL cross-section cells): today sets
  `pec_mask[c] = False`; becomes `clear_edges(edge_masks, cells)` — the three E entries at
  those indices are un-zeroed. Same for `pec_occupancy` clearing;
* MSL trace detection (`probes/msl_wave_decomp.py`, `_preflight._msl_realized_substrate`):
  read `realized_wall_planes` on the column;
* waveguide guide width (`_preflight._port_transverse_spans`): distance between realized
  wall planes (today measures a 40 mm guide as 42 mm — the #868 class);
* two-run S-matrix reference (`strip_interior_pec`): strips sheets too, or a sheet iris
  reads S11 = 0;
* reference-plane conductor footprint (`_execute._refplane_conductor_mask`, `refplane.py`),
  `conductor_mask()` / `conductor_footprint`: cells of volumes ∪ footprints of sheets;
* `auto_configure` z-feature detection: sheet planes are features;
* the YAML front-end (`rfx/config/loader.py`): gains a sheet entry (`thin_conductor`);
* `interop/_design.py` + IR schema: `two_plane` (currently a REQUIRED boolean) is removed,
  sheets are added, IR version bumped; a document carrying `two_plane` is refused.

## 2. Deleted

| what | where | why |
|---|---|---|
| `two_plane` kwarg, `_GeometryEntry.two_plane`, IR field, `_two_plane_cell_mask`, `_refuse_two_plane`, `two_plane_extension_masks`, `_place_at_next_plane`, ctx fields on every lane | `api/__init__.py`, `api/_spec.py`, `api/_execute.py`, `boundaries/pec.py`, `simulation.py`, `nonuniform.py`, `runners/*`, `vmap_sweep.py`, `interop/_design.py`, `probes/refplane.py`, `fidelity.py`, `visualize.py`, IR schema, Studio | t=1-only patch; subsumed by 1.2 |
| `resample_sheet_node_materials`, `sheet_normal_live_axis_masks`, `_subcell_box_axis_window`, `_statics_on_coords`, `collect_thin_conductor_sheet_inputs` | `geometry/rasterize_grid.py` and its callers in `api/_compile.py`, `nonuniform.py` | a sheet owns no cell, so there is no "own cell" to re-sample. The one physical case it served — a stack-up that leaves a slot for the foil — becomes a preflight notice (§3) |
| `tests/locks/test_two_plane_pec_slab.py`, `tests/unit/materials/test_sheet_node_permittivity.py` (resample tests) | tests | pin deleted mechanics; replaced by the contract tests in §5 |
| oracle-side compensations: cv19 `(L_c + 1)·dx`, `t_c = round(t/dx) + 1`, `L_c = round(L/dx) − 1`; cv18/csg "midpoint recipe" as a requirement; cv15 `Box(z_sub_lo − DX → z_sub_lo)` ground | crossval scripts + fixtures | under 1.2 drawn = realized, so the oracle takes the drawn value |

The full per-site list (every crossval, example, test, doc) is the inventory in §4.

## 3. Added

* `realized_pec_edge_masks`, `realized_wall_planes`, `edge_is_pec` (§1.7).
* `rasterize_geometry` returns `sheets: list[SheetSpec]` beside `pec_mask`; PEC thin
  conductors go to `sheets`, never to `pec_mask`.
* Preflight (input-fidelity only, per `feedback_preflight_input_fidelity_only`):
  * `pec_box_subcell` — ERROR: the §1.5 refusal, with the physical thickness and local cell;
  * `pec_box_one_cell` — WARNING: "PEC Box '<name>' is one cell thick along z: realized as a
    filled slab with walls at z = … and z = …. If this is foil, declare it with
    `add_thin_conductor`" (every crossval foil today is drawn this way, so this fires on
    unmigrated scripts by design);
  * `pec_zero_cells` — ERROR: the §1.5 zero-cell refusal (also reported here for shapes the
    rasterizer clipped to nothing);
  * `sheet_plane_realized` — NOTICE per sheet: declared mid-plane, realized plane, offset;
    WARNING when the mid-plane is an exact half-cell tie (both candidate planes named);
  * `sheet_slot_vacuum` — ERROR-grade WARNING: the node plane of a sheet carries vacuum while
    both neighbouring cells carry dielectric (a stack-up drawn with a slot for the foil);
    the one E edge that reads that node sits half a cell into the dielectric, so the cavity
    gains a vacuum cell in series (the #702 measurement: 17 % on a 127 µm stack). Remedy:
    extend the dielectric boxes to the sheet plane. Nothing is re-sampled silently;
  * every existing PEC/cavity/port validator reads `realized_wall_planes` / `edge_is_pec`.
* `fidelity_report`: per PEC entry, drawn extent vs realized wall planes per axis, in input
  units.
* `CHANGELOG`: this is a breaking API change (a public kwarg removed, a realization
  change). It ships as **2.0.0** under the #825 umbrella, not as a 1.x minor.

## 4. Migration (filled from the exhaustive inventory)

The 18-reader inventory that this section was filled from — every PEC site in
`rfx/`, `validation/`, `examples/`, `tests/`, `docs/`, `scripts/` with intent,
thickness, downstream artifacts, action and recompute cost — was a session
working artifact and was never committed. An earlier draft of this line pointed
at `docs/design_notes/20260906_lattice_ownership_inventory.md`, which does not
exist; the pointer is removed rather than left dangling in a normative document.
What survived the inventory is below, plus the per-group migration manifests
under `docs/design_notes/931_migration/`, which carry the same information for
the sites that actually changed.

Migration rules, in priority order:

1. **Foil drawn as a 1-cell PEC Box** (ground planes, patches, traces) → `add_thin_conductor`
   with the SAME physical corners. Realized plane = nearest node to the mid-plane (tie →
   lower), which for a face-registered 1-cell Box is the plane it lands on today.
2. **Foil drawn one cell OUTSIDE its interface** to park the wall on the interface (cv15
   ground) → draw it AT the interface as a sheet; delete the `two_plane` flag.
3. **Walls, irises, posts, plates drawn as volumes** → unchanged drawing; the realization
   gains its far face. Every number derived from today's realization (oracle inputs,
   fixture values, locks, gate bounds) is recomputed from the drawn geometry, and the
   compensation that produced it is deleted, not re-tuned.
4. **Tests pinning the old mechanics** → rewritten against §1, or deleted when the mechanic
   no longer exists.
5. **Docs** stating the old rule → rewritten from §1.

## 5. Verification (pre-declared)

Contract tests (`tests/contracts/test_lattice_ownership_contract.py`):

* slab battery: Box `t = 1, 2, 3` on each axis, each realizes walls at `lo` AND `hi` and
  no live normal edge inside; a sheet realizes one plane and a live normal edge;
* footprint battery: a patch Box realizes its drawn rectangle exactly (closed), as volume
  and as sheet;
* mirror / axis-permutation invariance of the realized edge set;
* soft ≡ hard at binary occupancy on the battery, including bodies on faces 0 and `n−1` of a
  non-periodic axis and the periodic seam;
* PEC thin conductor ≡ sub-cell sheet ≡ f0 sheet footprint (G4, by construction);
* `grep two_plane rfx/` = 0; no realization keyword on geometry entries;
* distributed-NU shmap parity with the single-device lane on the battery.

Physics falsifiers, pre-declared before any recompute:

* **cv19 (WR-90 iris filter)**: the fixture's cavity leg goes from `(L_c + 1)·dx` to the
  drawn `L_c·dx` and the iris-thickness offset from `−0.68` to `≈ +0.32` cell. If the
  recomputed residual does not move by about one cell, the diagnosis is wrong and the
  volume rule is not merged.
* **cv15 (RT5880 patch)**: `assert_realized_stack` passes with NO flag: walls at `z_sub_lo`
  and `z_sub_hi`, four substrate cells between. Realized in-plane patch = drawn.
* **One-cell volume witness (new)**: cv18's iris-thickness sweep gains `t_c = 1`; the
  mode-matching oracle at `t = dx` must sit on the same residual curve as `t_c = 2..8`. Today
  no independent witness says the two-wall rule is right at one cell.
* **cv16 (PEC sphere Mie)**: UNCHANGED — its sphere is a `rasterize(Sphere, eps 1, sigma 1e7)`
  material fill, which §1.8 fences out of this contract, so it is a control, not a recompute
  (this line previously contradicted §1.8; settled 2026-09-07 with group X-C). Measured
  price of the other choice, for the follow-up issue: at ka = 0.5 the node-sampled sigma
  fill occupies N = 1082 cells (a_eff/a 0.988032) while a declared PEC volume of the same
  sphere is centre-sampled to N = 1123 (a_eff/a 1.000357), 259 cells differing — bringing
  the RCS family under the contract moves a_eff by ~1.2 % and needs the fixture and both
  gate constants regenerated (~30 min CPU). #820's translation variance is then the first
  thing to re-measure.
* **Dielectric-only cases** (cv04, cv17, cv22, cv23 and every example without a conductor
  body): bit-identical results before/after — the change must not touch them.
* **Multilayer board A/B (memory 2026-08-28, VESSL 369367256724)**: the old `two_plane` arm
  correlated +0.265 with CST, one-plane +0.829. Under the contract the foils are sheets, so
  the board must stay at the one-plane figure; a drop below 0.8 blocks the default.

R3 for every commit on the branch: `R3: memory=rfx-known-issues.md("top face plane is never
zeroed" 2026-08-28; two_plane A/B verdict) | R2-attempts=0 (redesign, not a repeat) |
falsifier=<the §5 item exercised>`.

## 6. Amendments from implementation and review (2026-09-07)

**Auto-mesh resolution (2026-09-08).** Grid-dependent decisions consume a
resolved view of the declared mesh inputs, before choosing a lane. The common
mesh-field read boundary covers execution, inspection, optimization, builders,
and export; the uniform builder refuses a resolved nonuniform mesh instead of
silently building a surrogate. Preflight preserves declared dx/domain/profiles
and caches the same resolution execution uses. Geometry/material replacement
invalidates that cache. Auto-mesh selection warnings are outside preflight's
legality findings. Static preflight diagnostics and NU grid construction stay
host-side under an outer JIT, while explicitly traced profiles retain the
differentiable mesh path.
The §1.5 refusal and its numerical tolerance are unchanged.

**Declaration validation is a different question (2026-09-08 regression
audit).** `_declared_mesh` snapshots the caller's dx/domain/profiles without
planning a mesh; the existing mesh fields still answer what execution will
use. Floquet registration validates an explicit profile, while preflight
validates the completed model's resolved profile. The complete-line coaxial
drivers validate declared profiles before rejecting registered geometry they
do not consume; they must not auto-mesh that forbidden geometry just to reject
it for an inferred profile. Numerical capability checks, grid builders and
inspection keep the resolved view, including the uniform builder's refusal.
Tests that need a uniform comparator construct a separate uniform model;
tests of assembly follow the selected lane. The #655 coarse-mesh face-material
fixture explicitly pins its vacuum-default dx, so adding geometry cannot move
its physical sample or refine away the stress case. No dielectric sampling or
CPML material-extension rule changes in this repair. The regression audit and
blast-radius validation are recorded in
`20260908_automesh_regressions.md`.

Where the branch and this note disagreed, the branch is right and the wording
below replaces the earlier text. Each item names what actually shipped.

**§1.9 port clearing — one component, not three.** The note said "the three E
entries at those indices are un-zeroed". Implemented literally, that opens the
two edges TANGENTIAL to the port at its foot, which wherever the foot stands on
a conductor's node plane are that conductor's wall: measured, a wire port on a
PEC block removed 2 of 40 wall edges from the block's top face, and an MSL feed
released 7/7 Ex and 7/7 Ey along the port width on the ground plane. Corrected
rule: **a port releases the ONE component it drives, at its own cells.**
`clear_edges(edge_masks, cells, component=)`. A wire port releases nothing at
all — a cell is live exactly when the port component's own edge is not PEC, so
the release is a no-op by construction. An MSL port releases the
substrate-normal component over its cross-section.

**§1.5 sub-cell refusal is shape-agnostic.** The note's prose said "a PEC Box
with `0 < extent < one local cell`"; the rule is on the DRAWN bounding box of
any shape that has one (Cylinder pad, thin Sphere, imported outline).
PolylineWire is exempt — §1.4 decides filament vs volume on the radius.

**§1.3 sheet plane must be on the node line.** A sheet declared further than
half a local cell from the nearest node is refused, not clamped onto an end
plane.

**§1.3 2-D lane.** A sheet whose normal is a length-1 axis is realized by the
node-footprint rule (identical to the volume rule for a rectangular footprint,
verified), and no notice is emitted — notices are preflight's surface and
preflight is a separate stage.

**§3 out-parameters.** `rasterize_geometry` / `_assemble_materials` do not
return `sheets` in the positional tuple; sheets and wires come back through the
existing `sheet_specs`-style collector keywords, so the positional tuple
(pec_mask at index 3) stays unpacked by three validation scripts and many
tests. Omitting the collectors on a model that HAS a sheet or a wire is a
`ValueError` naming the caller (`_refuse_uncollected_pec`) — it was a
`UserWarning` only while the consumers were being migrated. A caller that
steps fields realizes what it collected or refuses the lane by name; a caller
that reads cells only still passes `pec_sheets=[], pec_wires=[]` and drops the
result, so "cells only" is a decision at the call site.

**§4 rule 2 for stack-ups.** `Stackup.to_shapes` puts a foil sheet on the
dielectric INTERFACE it bounds, not on the foil's mid-plane. On the mid-plane
the sheet sits half a foil thickness off the laminate and `auto_configure`'s
sheet-plane cut opens a 17.5 µm cell in series with the board — a ~23x dt
collapse and the #702 slot geometry. `_uniform_run` also merges a cut within
dx/4 of an existing mesh line rather than splitting there.

**§1.8 conformal.** "Conformal unchanged" is not literally true: the waveguide
lane's conformal path now applies the realized PEC edges AND Dey–Mittra, where
before it applied a sigma fold and Dey–Mittra. Not a regression (the sigma fold
was the thing §1.7 replaces), but no test pins a curved conformal body, so this
stays an untested edge.

**§1.9 consumers — the collectors at every assembler caller (done).** The
nine collector-less callers outside preflight's files now collect:
`optimize.py` (both lanes and the gradient check), `vmap_sweep.py`'s
pre-conductor pad assembly, `fidelity.py`, three sites in `visualize.py`,
`api/__init__.py`'s subgrid validator, `_execute.py`'s MSL static-eps read and
`_compile.py`'s `_build_materials`. Lanes that cannot realize a sheet refuse
it by name: `_build_materials` (the three coaxial S-parameter lanes) raises
`NotImplementedError`, and `validate_subgrid` reports
`subgrid_pec_sheet_or_wire_unsupported` in the same words the SBP-SAT runner
uses. `conductor_mask()` also collects wires and unions a filament's path
nodes (`rfx.boundaries.pec.wire_node_footprint`). Pinned by the entry-point
battery in `tests/contracts/test_lattice_ownership_contract.py`: tangential E
on a declared sheet plane is exactly zero through `run()`, `forward()`,
`vmap_material_sweep` and `optimize`'s step, with an off-sheet control probe.

**Preflight (§3 findings, §1.9 consumers) — LANDED 2026-09-07/08.** This
paragraph read "not yet implemented, owned by the preflight stage that follows
this branch" until 2026-09-08. It shipped on this branch instead, and leaving
the note saying otherwise had a cost worth recording: a `xfail(strict=True)`
test whose reason cited this very paragraph kept its marker after its
pre-declared falsifier fired, and under strict an XPASS is a red
(`test_thru_preflight_code_set_is_the_contract_set`). A design note that
describes a state the code has left is not merely out of date; it is load
bearing for anything that quotes it.

What is in place, verified by reading the shipped code rather than the plan:

* the four findings named here — `pec_box_one_cell`, `sheet_plane_realized`,
  `sheet_slot_vacuum`, `pec_zero_cells` — plus `pec_box_subcell`,
  `pec_realization_refused` and `pec_face_short_of_domain_wall`;
* `_port_transverse_spans` reads realized planes (the #868 "40 mm guide reads
  42 mm" case), and `_check_coaxial_port_junction_aperture` inherits that
  through `_port_pec_mask`;
* `_msl_realized_substrate` takes the realized conductor from
  `_msl_assemble_once`'s cache, starts its walk at the first cell ABOVE the
  realized ground, and reports `ground_cells` so a port sitting under its own
  ground can be named;
* the wire-port end-gap advisory fires from realized wall planes. Its MESSAGE
  wording is still the pre-contract spelling, which is why
  `test_wire_port_end_gap_advisory_fires_on_a_declared_one_cell_gap` remains a
  strict xfail with that stated as its reason. The premise it owns passes; the
  wording is the open half.

`_validate_cfg_sheet_live_edge_materials` is not migrated because it is GONE:
`sheet_live_edge_material_mismatch` (#703 check 2) guarded the #702 resample and
§2 deletes it. A sheet owns no cell, so there is no "own cell" material to
compare against — the check had no subject under the contract.

`sheet_plane_realized` is conditional: it reports only when a declared sheet
lands OFF its declared mid-plane, or when a tie had to be broken. A board whose
sheets land exactly where they were declared prints nothing, which is the
correct silence and which invalidated two test expectations that pinned the
LINE rather than the fact.

**§1.1 nearest-plane rounding meets a ceil-realized domain (2026-09-07, the
cv11 pec-short "core regression").** cv11's pec-short |S11| deficit went
0.0146 → 0.0560 after this branch (VESSL 369367259194) and was filed against
the waveguide lane's stage-C change. Adjudicated with per-bin dumps and port
time records on both checkouts (`scripts/diagnostics/pec_short_lane_ab.py`):
it was the FIXTURE'S DRAWING, made visible by §1.1. The plug was drawn to the
declared 22.86 × 10.16 mm; `Grid` realizes the guide by ceil as 23 × 11 mm,
a volume's face rounds to the nearest node, so the plug's top landed at
10.000 mm under a wall at 11.000 mm — a one-cell vacuum slot along the top
broad wall, a parallel-plate line for Ez, transmitting |S21| 0.22–0.33 past
the "short". The pre-#931 node-half-open sampler included node 10 by
accident and the sigma fill covered the full height. Drawn to the realized
walls the leg reads [0.9980, 1.0019] / 3.26°, identical to the no-trim
baseline to four decimals; the far face, the window and the reference run do
not matter; a sheet drawn to the walls reproduces the closed volume (the lane
applies sheets — the 0b6f1239 "sheet does not work" reading was the same
rim-short footprint as a resonant slot). Rule for fixtures: a conductor meant
to reach a domain wall is drawn to the grid's REALIZED wall plane
(`tests/_realized_geometry.domain_wall_positions`), and a shorting plug
asserts a full-cross-section front wall at build time. Owed by the preflight
stage: a finding for a conductor face drawn within a cell of a domain wall
that rounds away from it — **landed 2026-09-07** as
`pec_face_short_of_domain_wall` (WARNING,
`_validate_cfg_pec_face_short_of_domain_wall`): a PEC VOLUME whose own realized
wall plane sits exactly one node inside a NON-absorbing domain face, computed
from `realized_wall_planes` on the entry's own edges against the grid's
`interior` slices. On the cv11 drawing at dx = 1 mm it fires once, on `z_hi`
(10.16 mm rounds DOWN off the 11 mm wall) and not on `y_hi` (22.86 mm rounds UP
onto the 23 mm wall) — which is why it reads realized planes instead of
comparing declared numbers. The validation battery's `test_pec_short_s11_magnitude`
(auto mesh, same slot) is green again with its gate untouched; the chain
battery's `pec_short` DUT is on-lattice at every rung and its T6 red
(max|ΔS| 0.938) is a different reading — S22's phase, the far face one cell
further from the right port — not this one.

**§1.8 gains a fence it did not have: the ADI lane cannot carry interior PEC
(2026-09-08).** Not a scope decision like the other fences — a measured limit.
Handing that lane its sheets and wires, which §1.9 requires, exposed an
instability `run()` already had and `forward()` had been hiding. 104
configurations on a 20 mm cube, before anything was decided:

| lane | conductor | factor | steps | peak probe |
|---|---|---|---|---|
| 3-D | none | 5.0 | 200 | 0.0023 |
| 3-D | sheet | 5.0 | 200 | 4.08e30, 19 non-finite |
| 3-D | none | 1.0 | 4000 | 0.0083 |
| 3-D | sheet | 1.0 | 4000 | 4.33e6 |
| 3-D | one-cell volume | 1.0 | 4000 | 29.1 |
| 3-D | sheet | 0.5 | 4000 | 0.0100 |

With no interior conductor the lane is bounded at every factor tested, 0.5
through 5.0, out to 4000 steps — the unconditional-stability property is real
for the homogeneous lossless split with compatible domain boundaries. With any
interior conductor it diverges, and the class changes the growth rate rather
than the outcome. Factor 1 survives 800 steps and fails by 4000, so a short run
hides it. Factor 0.5 stayed bounded, but that is below the Yee limit and
therefore not an operating point — and it is an observation over one horizon,
not a derived bound, so it does not license a clamp.

The rule: **interior PEC on `solver="adi"` is REFUSED, at every factor, on 3-D
and 2-D TMz** (`adi_interior_pec_unsupported`), enforced in `run()`, `forward()`
and the low-level step functions so `skip_preflight=True` does not get past it.
Domain-face PEC without interior conductors is unaffected.

Why refuse rather than warn, clamp, or fix. A warning leaves a user holding
4e30. A clamp implies a safe factor that the measurement does not supply. And an
unconditionally stable scheme losing stability at a material discontinuity is a
numerics question — the split's stability proof assumes a homogeneous lossless
medium, and the PEC projection applied between the two half-steps is outside it.
Answering that needs constrained-operator analysis, not a branch fix.

This fence differs from the others in §1.8 in one way that matters: the
conductor is not silently absent, it is loudly refused. A declared conductor
vanishing is the defect this note exists to remove, and "make the number finite
by dropping the metal" was explicitly rejected as a resolution. Measurements:
`docs/design_notes/adi_pec_stability/`. Reasoning and the test that pins the
shipped default: `docs/design_notes/20260908_adi_interior_pec_guard.md`.

**§1.9 consumers — one that had gone silent (2026-09-08).**
`_warn_junction_probe_clearance` compared a device against its straight-guide
references through `sigma` alone. Once §1.7 moved PEC out of `sigma` into
realized edges, a PEC junction and its references read as identical vacuum and
the advisory stopped firing — on the lane whose purpose is catching a probe
plane too close to a junction. It compares realized edges componentwise now,
and deliberately does NOT union the three masks first: that would hide two
differently oriented sheets occupying one node plane. The device edges are
captured before the Kottke path, which encodes PEC into inverse permittivity and
clears the solver masks. After the fix, clearances 0 / 2 / 2 mm against an
unchanged 28.11 mm minimum. Worth stating as a general lesson for §1.9: a
consumer that reads `sigma` to find metal does not fail loudly under this
contract, it goes quiet, and a quiet advisory looks exactly like a clean model.

**§1.1 an off-lattice corner splits a PEC volume from a dielectric drawn on it
(2026-09-08, fresh-eyes review).** §1.1 says PEC volumes are centre-sampled and
dielectric sampling is untouched, and §1.3 works the off-lattice case for
SHEETS. The volume case was never written down, and it is not free.

`eps_r` is node-sampled: node `k` carries the material when `k·dx` is inside
the drawn extent. A PEC volume is centre-sampled: cell `k` is occupied when
`(k + ½)·dx` is. Measured on a 6-cell box whose lo corner sits `frac` of a cell
above a node, `dx = 1 mm`:

| `frac` | dielectric nodes | PEC cells | lo offset |
|---|---|---|---|
| 0.0 | 4..9 | 4..9 | 0 |
| 0.2 | 5..10 | 4..9 | −1 |
| 0.3 | 5..10 | 4..9 | −1 |
| 0.5 | 5..10 | 4..9 | −1 |
| 0.7 | 5..10 | 5..10 | 0 |
| 0.9 | 5..10 | 5..10 | 0 |

They agree on a node (`frac = 0`) and above the half cell, and differ by a full
cell for `frac` in `(0, ½]` — node sampling takes the ceiling of the corner,
centre sampling takes the nearest cell whose middle is inside. So a microstrip
whose laminate face is drawn a fifth of a cell off the node line puts its trace
one cell INSIDE its own substrate, and both realizations are individually
correct for the rule they follow.

This is not a defect to fix by making one sampler match the other: a volume owns
cells and a dielectric fills them, and the two questions are genuinely different.
It is a declaration hazard, and the remedy is §1.3's remedy for sheets — put a
mesh node on the interface (`dx = h/N`, or a preserved region on the non-uniform
lane). Preflight already reports it as `off_lattice_design_edges` with a
quantified residual, so it is visible rather than silent; what was missing was
this note saying that the residual has a one-cell CONSEQUENCE for a volume, not
only a sub-cell placement error.
