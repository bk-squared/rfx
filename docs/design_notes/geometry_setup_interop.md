# Geometry / setup export–import interop

Status: **PROVISIONAL.** The rfx⇄rfx design document and the first external
emitter (openEMS) are implemented and contract-tested; the import direction is
researched only. **Nothing here is a validated claim about agreement with
another solver** — the openEMS emitter is proven *executable*, not *agreeing*,
and the port models are known to differ by ~0.20 in |S| before any physics is in
question (see the ceiling section below).
Date: 2026-07-25. Branch: `feat/geometry-setup-interop`.

> **Current-status note (2026-08-11):** The rfx design document and openEMS
> emitter described here remain implemented and contract-tested; import from a
> foreign solver remains research-only. Paths and line numbers below record the
> 2026-07-25 implementation snapshot, while the named symbols and decisions are
> the durable contract. Use `rfx/interop/` and its contract tests for the
> current layout; this note is not evidence of cross-solver physics agreement.

## Question

Can an rfx design (geometry **and** setup — materials, excitations,
observables, mesh directives) be exported to another EM solver, and can a
design authored in another tool be brought into rfx?

## Premise check first — what already exists

Two premises that looked true from a stale checkout turned out to be false, and
both change the plan. They are recorded here because the wrong version was
briefly acted on.

**1. rfx already has CAD import.** `rfx/geometry/mesh_import.py` provides
`MeshShape`, which loads watertight STL / OBJ / PLY through `trimesh` and
STEP / STP through the optional `cascadio` OpenCASCADE backend (`cad` extra in
`pyproject.toml`). Landed via #453 / #456; issue #358 is closed. The asymmetry
is therefore **import exists, export does not** — not "no CAD support".

**2. rfx already has four setup-serialisation layers.** Every one of them is
box-or-bbox-level, so none can rebuild its own input:

| layer | geometry fidelity |
|---|---|
| `rfx/io.py:869` `export_geometry_json` | class name + bounding box |
| `rfx/artifacts.py:274,311` `build_scene_artifact` | pops `shape`, keeps `shape_type` + bounding box |
| `rfx/config/_shapes.py:15` | `_SUPPORTED_SHAPES = ("box",)` |
| `rfx/experiments/canonical.py:752` | `"P0 supports box geometry"`, geometry as `bounds_m` |

Measured demonstration (run on this branch):

```python
sim.add(Cylinder(center=(0.010, 0.006, 0.0008), radius=3e-4,
                 height=1.5e-3, axis="z"), material="pec")
```

`build_scene_artifact(sim)["geometry"][1]` →

```json
{"id": "geometry-1", "kind": "geometry", "type": "_GeometryEntry",
 "shape_type": "Cylinder",
 "bounding_box": [[0.0097, 0.0057, 5.0e-05], [0.0103, 0.0063, 0.00155]],
 "material_name": "pec"}
```

`radius`, `height`, `axis` and `center` are absent; the cylinder is
indistinguishable from a `Box` with the same bounds. That is the concrete gap
this work closes. (The quoted entry is the second geometry entry of a two-entry
scene — a substrate box plus this cylinder — hence `geometry-1`.)

## Decisions

**D1 — The primitive-level codec is a shared leaf, not a fifth layer.**
All four layers above lack the same thing: a way to record a shape's
*constructor parameters*. `rfx/interop/_shapes.py` supplies it for all six CSG
primitives, and `rfx/interop/_materials.py` does the same for `MaterialSpec`
including Debye/Lorentz pole parameters (which `artifacts.py:245` reduces to
`{"present", "count"}`). Consolidating the four layers onto these codecs is
follow-up work, not a prerequisite.

**D2 — Vocabulary: `kind` + snake_case.** The repo already names shapes twice —
`config/_shapes.py` uses `shape: "box"`, `canonical.py` uses `kind: "box"`. The
codec uses `kind` with snake_case names (`box`, `cylinder`, `sphere`,
`polyline_wire`, `via`, `curved_patch`) so a third spelling is not created.
Parameter names stay the *constructor* names, because that is what makes the
registry pinnable against each live class signature; per-layer spellings of a
box (`bounds`, `bounds_m`) remain adapter concerns for those layers.

**D3 — Two profiles, two names.** The rfx→rfx document is round-trip complete
and its gate is structural equality. An external-solver projection is a
*separate*, explicitly lossy artifact that carries its own list of
approximations. There is no single schema with a "portable-ish" flag, because a
reader cannot tell which fields survived.

**D4 — Refuse, never approximate.** Anything not representable exactly raises
`UnsupportedDesignFeature` naming the entry and index. No `warnings.warn`-and-
drop, no bounding-box fallback. `MeshShape` is refused by the codec today: a
triangle mesh is not parameter-describable the way a CSG primitive is, and
degrading it to a bounding box would produce a different structure under the
same name.

**D4a — Geometry order is semantic state, but the rule is not uniform
last-write-wins.** Two distinct composition rules run in the same loop
(`rfx/api/_compile.py:162-174`, the live path — note `rasterize()`'s
"later shapes overwrite earlier ones" docstring at `rfx/geometry/csg.py:321`
describes a function the simulation path never calls):

- **Dielectric paint is last-write-wins.** `eps_r`/`sigma`/`mu_r` are written
  with `jnp.where(mask, ...)`, so a later overlapping entry overwrites an
  earlier one.
- **PEC is a union and is order-independent.** When `mat.sigma >=
  _PEC_SIGMA_THRESHOLD` the loop does `pec_mask = pec_mask | mask` and **skips
  the dielectric paint entirely**; `pec_mask` is initialised once
  (`_compile.py:139`) and only ever OR'd again (`:168`, `:234` for thin
  conductors). So a dielectric added after an overlapping PEC shape does **not**
  erase it — PEC wins regardless of order.

An emitter built on uniform last-write-wins would therefore mistranslate every
PEC-over-dielectric overlap, which is most PCB stackups. The conclusion stands
and gets stronger: entry order is semantic state the document must preserve
exactly, and an emitter targeting a boolean solid modeller (HFSS) must ask or
refuse rather than silently pick a subtraction order. The boolean helpers
`union` / `difference` / `intersection` are defined at `csg.py:298,302,306` and
have **zero call sites anywhere in `rfx/`** — they are exercised only in
`tests/test_geometry.py`.

**D5 — The container decision is deferred, deliberately.**
`CanonicalExperimentSpec` (`rfx-experiment/v2`) is the strongest existing
foundation — versioned, with schema migrations, a closed-world validator, an
sha256 + semantic fingerprint, a working importer (`build_simulation()`), and
Python emission. Folding the design document into it is the likely endpoint,
but it is a governed schema with a published JSON Schema and existing
consumers, so widening its geometry beyond `box` deserves an explicit decision
rather than a side effect of this branch.

## Non-portable state (must be marked, not hidden)

These round-trip correctly inside rfx but are meaningless to another solver.
An external emitter that ships them silently would produce a comparison against
a structure the other solver never built.

- `_cpml_kappa_max` and the CPML profile constants hard-coded in
  `rfx/boundaries/cpml.py` — absorber behaviour is not portable; CPML layer
  *counts* have per-solver cell semantics.
- `_coaxial_terminations`, `_coaxial_open_terminations`,
  `_coaxial_pec_end_caps` — offsets are **cell-relative**, not physical
  coordinates.
- `_msl_ports.n_probe_offset` / `n_probe_spacing` — cell counts derived from
  `_dx` at registration time.
- `_refinement` — the SBP-SAT subgrid path, experimental and falsified in 3D
  (PR #90). Round-trip for rfx; never project outward.
- `_precision`, `_solver` (`yee`/`adi`), `_adi_cfl_factor`, `_stencil_order` —
  rfx solver controls with no counterpart.
- `_tfsf` — a plane-wave projection onto CST/HFSS is an approximation with a
  different absorber story, not a translation.

## Boundaries that constrain honest claims

- **`MeshShape` is off the AD path.** It rasterises host-side through
  `trimesh.contains` (`rfx/geometry/mesh_import.py:149`), so it cannot be
  traced or JIT-compiled. A CAD-sourced design can round-trip, but it cannot
  carry gradients — which matters because the project's standing tie-breaker is
  that AD-traceability is non-negotiable.
- **Traced mesh profiles cannot be exported.** `_dx_profile` / `_dy_profile`
  may be a JAX tracer on the differentiable-mesh path
  (`rfx/api/__init__.py:387-393`); exporting a tracer is meaningless, so it
  must refuse.
- **The S-parameter drivers rewrite builder state.** `rfx/api/_sparams.py`
  temporarily mutates `_dz_profile`, `_msl_ports`, `_ports`, `_dft_planes` and
  `_waveguide_ports`, then restores them. Exporting mid-driver captures a
  synthetic configuration.
- **Derived state is never a source of truth.** `Grid` / `NonUniformGrid`,
  `dt`, `axis_pads` and preflight verdicts are excluded from the document. A
  design description that carries `dt` invites a reader to trust it.
- **Config-layer fences are narrower than the Python API.** The YAML path
  supports `box` only, defers waveguide / coaxial / floquet / msl / tfsf ports
  (`rfx/config/loader.py:47-53`), and takes scalar boundary strings rather than
  a per-face `BoundarySpec`. Any claim of "full-fidelity export through the
  config layer" would be an overclaim today.

## Why this earns a slot under "correctness > feature sprawl"

The cross-solver validation campaign's cost centre is building each structure
twice — once in rfx and once by hand in openEMS / Meep / Palace — and keeping
the two identical. `scripts/diagnostics/build_sheen_lpf_palace_referee.py` and
the Palace mesh builder are exactly that hand-porting. The project's standing rules (never hand-roll an external-solver crossval
setup; comparator first — see `docs/guides/simulation_methodology.md` and the
crossval manifest's evidence rule) identify comparator/fixture divergence —
not solver physics — as the dominant historical failure class. A
generated, single-source setup attacks that class directly.

## Target order, and why openEMS is first

Measured in this environment, not assumed:

- **openEMS is installed and executes.** `/usr/bin/openEMS`, Python bindings
  importable, and a 10-timestep 9261-cell run completes (1.24 MCells/s, writes
  `et`/`ht`). It is therefore the **only** target whose generated script we can
  verify end to end here. CST and HFSS cannot be verified at all without a
  licence.
- Boundary-condition string is `'PML_<N>'`, **not** `'PML-<N>'` — the hyphen form
  raises "Unknown boundary condition". Confirmed by running it. `PML_<N>` is
  parameterised, so `cpml_layers=16` → `'PML_16'`.
- **CST is the closest *semantic* match** even though it cannot be verified
  here: `DispModelEps` accepts `Debye1st`/`Debye2nd`/`Lorentz`/`NonLinearKerr`,
  `FloquetPort` takes scan θ/φ directly, `Port.ReferencePlaneDistance` is
  exactly rfx's de-embedding, and `Solver.SteadyStateLimit` is rfx's settling
  criterion. But the CST Python API is a *driver*: models are built by pushing
  **VBA** through `add_to_history`, so a CST emitter must emit VBA.
- **Generated artifacts must be plain-text scripts, runnable later on a
  licensed machine.** PyAEDT requires a legally licensed local AEDT, so a
  live-session generator would licence-gate CI and be unusable for the actual
  workflow. Generation must need no licence.

### The highest-severity translation hazard, measured

**rfx adds CPML *outside* the user domain; openEMS PML consumes cells *inside*
the mesh extent.** A naive translation therefore buries the structure in the
absorber and compares against a different problem — the dominant historical
failure class in this repo (comparator/fixture divergence, not solver physics).

Measured with `domain=(0.01, 0.01, 0.01)`, `dx=1e-3`:

| `cpml_layers` | grid shape | `axis_pads` |
|---|---|---|
| 0 | (11, 11, 11) | (0, 0, 0) |
| 8 | (27, 27, 27) = 11 + 2×8 | (8, 8, 8) |
| 16 | (43, 43, 43) = 11 + 2×16 | (16, 16, 16) |

So the user's `domain` is entirely physical and the absorber is extra cells
beyond it, consistent with `rfx/grid.py:133` — `(idx - axis_pads[ax]) * dx`
recovers user-domain coordinates, i.e. index 0 lies outside the domain. An
openEMS emitter must span `domain + 2 × cpml_layers × dx` per absorbing axis and
let `PML_<N>` eat the added margin.

(Noted while verifying: the subgrid PML-overlap warning at
`rfx/api/__init__.py:627-650` tests `z_lo < pml_thickness` / `z_hi > domain_z -
pml_thickness`, which reads the absorber as living *inside* the domain — the
opposite frame from the grid builder. The subgrid path is experimental and
parked (PR #90), so this is recorded rather than chased.)

### The ceiling on any emitted-setup agreement claim

Port models differ **structurally** between rfx and openEMS, and the repo has
already measured the size of it. This bounds what an emitter may claim, before
any physics is in question:

- rfx's wire/lumped port is a **point feed** (one vertical cell column from
  ground to trace, the `extent`); openEMS's `MSLPort` occupies a **span** along
  the propagation axis (6 cells in the upstream `MSL_NotchFilter.py` tutorial).
  `port_w_cells` has **no rfx counterpart**, so it is an emitter parameter and an
  assumption the reader must be shown, not a derived quantity.
- The **direction convention inverts**: rfx's `direction` names the outward
  (away-from-the-line) normal, while openEMS derives propagation from
  `sign(stop - start)` with both ports pointing *into* the shared trace. A wrong
  sign yields a plausible-looking S21 with the wrong reference sense.
- **≈0.20 in |S| is the measured port-convention gap**
  (`scripts/diagnostics/build_wire_openems_broad_envelope.py:12-16`: the
  tolerance is deliberately loose "because the wire-port mid-cell convention has
  known mismatch with openEMS's lumped-port multi-cell averaging", and broad
  calibrated E5 still requires resolving the wire-port absolute calibration
  convention).

That 0.20 is **looser than the cv06b mean gate (0.13)** and comparable to its max
gate (0.25). So: a passing comparison of an emitted setup is not evidence that
the port models agree, and **structural equivalence of a generated setup is not
evidence of physics agreement** at all. The emitter's job is to remove
hand-porting divergence, not to produce agreement.

### What the openEMS emitter was actually gated on

**Not cv06b.** The emitter is gated on **executability**, on a coarse two-port
PEC cavity: an emitted script is written to disk, run as a subprocess under
openEMS, and the artifact is asserted to be finite, passive (≤ 1.05), to have a
non-empty `approximations` list, to show a non-zero incident peak (so the ports
really are on-grid — openEMS silently drops an off-grid excitation), and to give
`|S11| > 0.9` as a lossless closed cavity must. Every one of those is a statement
about the emitted artifact, not about rfx.

A physics-agreement comparison against the committed cv06b reference was
**deliberately not attempted**: dx = 50 µm over that domain is ~1.6 M cells ×
600 k timesteps, and a coarse stand-in would not be evidence. The next
well-defined step is cheaper and more diagnostic anyway — compare a *generated*
openEMS setup against the repo's *hand-written* one at the same mesh, where any
difference is emitter fidelity rather than solver physics.

### The cv06b reference that a future physics comparison would use

`tests/crossval/test_msl_notch_e4_comparison_gates.py` + `tests/fixtures/msl_notch_e4/`
already hold a committed physical openEMS dx = 50 µm reference for the cv06b
microstrip open-stub notch, alongside the rfx result at matched geometry. The
test runs no FDTD (the ~65 min rfx run is committed as a fixture) and the whole
gate family is green on this branch (13 tests, 0.12 s).

Committed gates: notch-frequency agreement **≤ 7 %** (characterised ~6 %),
off-notch |S21| mean abs diff **≤ 0.13** and max **≤ 0.25** over 2.5–6 GHz,
passivity **≤ 1.05** for both solvers, notch depth **< −20 dB**, plus a sign
constraint (rfx notch above openEMS — part of the committed characterisation).
A Palace FEM referee lands at ~3.631 GHz, closest to rfx (+0.1 %), which
retired the earlier open-end-fringing interpretation.

The fixture's `meta` block carries the full parametric geometry
(`eps_r=3.66`, `h_sub=0.254 mm`, `w_trace=0.6 mm`, `l_line=5.0 mm`,
`l_stub=12.0 mm`, `dx=50 µm`, 2–7 GHz × 50 points, `nrts=600000`,
`end_criteria=1e-4`, domain 7.0 × 16.232 × 1.754 mm), so the emitter has a
numeric target to reproduce rather than a narrative one.

## Decisions that need a human, not a default

Both come from the target-API survey and neither has a defensible default:

1. **HFSS paint-order → boolean rewriting.** rfx's overlapping paint (D4a) has
   no boolean equivalent; choosing a subtraction order silently would change the
   structure. Ask or refuse.
2. **The MSL port recipe.** rfx extracts via an N-probe spatial fit; HFSS offers
   wave vs lumped ports with v/h factors and a choice of `Zpi` / `Zwave` /
   `Zpv`. Which of those rfx's number should be compared against is a physics
   decision, not a mapping detail.

## Verification status of the scaffolding (measured on this branch)

- `ruff check rfx/ tests/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402` → clean
- codec contract tests → 71 passed (grew from 53 as the two in-development
  reviews' value-validation and stray-key fixes landed with their tests)
- `python scripts/check_api_reference.py` → `api reference surface: OK`
  (the pinned surface is untouched because nothing was added to `rfx/__init__.py`)
- wheel build ships the package: `rfx/interop/{__init__,_design,_errors,_materials,_shapes,_validate}.py`
  and `rfx/interop/emitters/{__init__,openems}.py` (setuptools `include = ["rfx*"]`
  auto-discovery picks up both subpackages, so no packaging change was needed)
- cv06b gate family → 13 passed (the baseline the emitter work must not disturb)

## Status of each piece

| piece | status |
|---|---|
| shape codec, 6/6 primitives, registry pinned to live signatures | implemented, contract-tested |
| material codec incl. dispersion poles | implemented, contract-tested |
| `MeshShape` / subclass / impostor-class refusals | implemented, tested |
| design document (`Simulation` ⇄ document, `rfx-design-ir/v1`) | implemented; round trip gated over 16 fixtures by an all-attribute diff whose non-vacuity is itself proven by 7 mutation tests |
| completeness ledgers | implemented: a new `Simulation.__init__` attribute or `add_*` method reds a test until a decision is recorded |
| published JSON Schema for the document | implemented, pinned to the code and validated against every design fixture |
| openEMS emitter | implemented; **executability** proven by an emitted script that actually runs under openEMS and returns finite, passive S-parameters. Shapes are limited to `box`/`cylinder` — the rest refuse because no translation of them has ever been checked against a known-good result in this repo |
| CST VBA / PyAEDT / Meep emitters | **not started** (CST needs VBA emission; neither CST nor HFSS can be verified here without a licence) |
| import direction (GDSII, CST history list, PyAEDT enumeration) | **researched only** |
| consolidation of the four existing serialisers | **not started** |
| physics agreement against `tests/fixtures/msl_notch_e4/` | **not attempted** — cv06b at dx = 50 µm is ~1.6M cells × 600k steps; a coarse run would not be evidence |

## Corrections to the target-API survey (found while implementing)

The survey report was the emitter's specification, and implementing against it
surfaced three errors in it. Recorded here because the report itself is a
scratch artifact and these are the durable parts.

1. **The two rfx port builders use OPPOSITE direction conventions.** The survey
   claimed `add_msl_port(direction=...)` maps to an openEMS `MSLPort` whose
   `stop - start` sign is the negation of the rfx direction. That is true for
   `add_port` but **not** for `add_msl_port`:
   - `add_port` — direction is the **outward normal**: *"Outward-normal
     direction of the port (from the port cell into the external world)"*
     (`rfx/api/__init__.py:1116`). openEMS propagation points *into* the line,
     so the negation is correct here.
   - `add_msl_port` — direction is the **propagation direction**: *"Direction
     the launched wave propagates away from the feed plane"*
     (`rfx/sources/msl_port.py:51`). No negation.
   Getting this wrong yields a plausible-looking S21 with the wrong reference
   sense, so it is pinned by a test asserting the physical consequence (both
   spans land strictly inside the port-to-port interval) rather than an
   arithmetic sign.
2. **The survey's recommended excitation guard `uf_inc <= 1e-9` would
   false-fire.** Measured `incident_peak_abs` on runs that produced correct
   S-parameters: **1.93e-12** (PEC cavity) and **1.79e-13** (MSL thru). That
   threshold is live in
   `scripts/diagnostics/build_coaxial_line_openems_broad_comparison.py:214-223`
   and is worth auditing before it is reused. The emitter instead tests for
   zero/non-finite plus an independent port time-trace witness.
3. **Pinning mesh lines at conductor faces is wrong for a faithful uniform-mesh
   projection.** rfx rasterises geometry against the uniform grid, so inserting
   lines at conductor faces hands openEMS a mesh rfx never used. Only *port*
   edges are pinned (openEMS drops an off-grid excitation outright), snapped
   with the same `round(pos/dx)` rule as `Grid.position_to_index`.

Also worth knowing, because it is indistinguishable from a real failure in the
log: openEMS prints `Unused primitive (type: Box) ... msl_feed_N` whenever the
design's own trace already covers the MSL port span. Benign when both are PEC —
but it is the *same* log line as the port-was-dropped failure mode, so the
incident-peak witness is what actually discriminates.
