# Geometry / setup export–import interop

Status: **PROVISIONAL — under construction.** The primitive-level codecs are
implemented and contract-tested; the design document, the external-solver
emitters, and the import direction are not yet landed. Nothing here is a
validated claim about agreement with another solver.
Date: 2026-07-25. Branch: `feat/geometry-setup-interop`.

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

`radius`, `height`, `axis` and `center` are absent; the via is indistinguishable
from a `Box` with the same bounds. That is the concrete gap this work closes.

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

**D4a — Geometry order is semantic state, not presentation.** `Simulation.add`
appends to an **ordered, last-write-wins paint list**: `rfx/geometry/csg.py:321`
— *"Applied in order; later shapes overwrite earlier ones."* The boolean helpers
`union` / `difference` / `intersection` exist in `rfx/geometry/csg.py` but are
**not** used by the simulation path (no call sites in `rfx/api/` or `rfx/core/`;
they are exported for direct use and exercised only in `tests/test_geometry.py`).
Consequences: the document must preserve entry order exactly, and an emitter
targeting a solver with a boolean solid model (HFSS) cannot translate overlaps
mechanically — overlapping paint has no boolean equivalent, so the projection
must either ask or refuse rather than silently pick a subtraction order.

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
the Palace mesh builder are exactly that hand-porting. The repo's own rules
(`.claude/rules/rfx-feature-discovery.md`: never hand-roll an external-solver
crossval setup; `CLAUDE.md`: comparator first) identify comparator/fixture
divergence — not solver physics — as the dominant historical failure class. A
generated, single-source setup attacks that class directly.

## Status of each piece

| piece | status |
|---|---|
| shape codec, 6/6 primitives, registry pinned to live signatures | implemented, contract-tested |
| material codec incl. dispersion poles | implemented, contract-tested |
| `MeshShape` / subclass / impostor-class refusals | implemented, tested |
| design document (`Simulation` ⇄ document) | **in progress** |
| published JSON Schema for the document | **in progress** |
| external-solver emitters (PyAEDT / CST VBA / openEMS / Meep) | **not started** |
| import direction (GDSII, CST history list, PyAEDT enumeration) | **researched only** |
| consolidation of the four existing serialisers | **not started** |
