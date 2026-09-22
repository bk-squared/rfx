# One grid for the uniform and the non-uniform lane — step 0 pre-declaration

**Status:** pre-declaration for the core change the PI approved on 2026-09-22
after an independent design review (Fable, separate instance; its text and the
two scripts it ran are kept off-repo under `bk-workspace/rfx-nu/notes/`). This
note fixes what step 0 builds, what it must not change, and how each part is
judged, before any code. Later parts (the mesher, step 1; the physics witnesses
G14/G16; the differentiable map) get their own notes.
**Lane:** nu-mesh. **Trackers:** #810 (graded-mesh observables), the program
document's gap map G9/G11/G13/G16/G17 (`20260913_nu_full_functionality_program.md`).
**Written by** the NU lane leader (bk-workspace, Fable).

## 0. What is wrong, and what changes for a user

rfx keeps two grids. `rfx/grid.py::Grid` (uniform) carries one scalar `dx`.
`rfx/nonuniform.py::NonUniformGrid` carries per-axis cell arrays
(`dx_arr`, `dy_arr`, `dz`) and ALSO a scalar `dx` that means "the boundary
cell". A consumer that reads `grid.dx` is right on the uniform grid and
silently wrong on a graded one wherever the local cell differs from the
boundary cell. On main 5b8724be, 167 sites in 15 modules outside the NU
module read the scalar, and 8 modules branch on `isinstance(grid,
NonUniformGrid)` and implement the same thing twice.

What a user gets today because of it: a waveguide port placed inside a fine
band injects at half the local metric (G13, `rfx/nonuniform.py:2091, 2134,
2202`); the x/y absorber is calibrated with the boundary cell whatever cells
sit under it (G11, `rfx/boundaries/cpml.py:192-206`; the six per-face slots
exist at `:505-510` and are filled with the one scalar); preflight
tolerances count cells of the wrong size (G17). Every graded-mesh result so
far had to keep ports and absorbers on uniform runways to stay out of these
three — the notch-filter arm moved its line 609 µm for that reason.

After step 0 those three are closed by construction: there is one place a
consumer can ask for a cell size, and it answers per cell. What step 0 does
NOT close: the TFSF auxiliary 1-D grid must itself run on the graded cells to
keep numerical dispersion matched (G14, `rfx/nonuniform.py:2048-2062`), and a
transition next to an absorber is a normal-direction medium change inside
the PML (G16, pad replication `rfx/nonuniform.py:122-149`) — both need
physics witnesses and are separate declarations.

## 1. Facts this note rests on (who checked, how)

| fact | checked by |
|---|---|
| 167 scalar `grid.dx` read sites outside `rfx/nonuniform.py`/`rfx/grid.py`; 8 `isinstance(NonUniformGrid)` modules | leader, `grep` on main 5b8724be |
| E and H use different metrics: `inv_d_h[k] = 1/d[k]` (`[N-1] = 0`), `inv_d_e[k] = 2/(d[k-1]+d[k])` (`[0] = 1/d[0]`), bounding node appended | leader read `rfx/nonuniform.py:151-246` |
| the sheet conductance uses the dual spacing, source `dV` and RLC the primal | reviewer read `thin_conductor.py:380-384`, `sources.py:70-100`; `sources.py:36-100` (#691) is the pattern to copy |
| nodes derived from `i·dx` do not reproduce the cells: 47 of 50 cells differ from `dx` by 6e-15; the present raster lane identity (#807) comes from a closed-form special case for uniform axes (`rasterize_grid.py:117-120`) | reviewer RAN |
| the two lanes' `dt` differ in the last bit on the same grid (`grid.py:151` vs `nonuniform.py:416-421`) | reviewer RAN; leader read both formulas |
| the two lanes' source units differ: NU injects a current (`E += dt/ε · I/dV`), uniform adds a field increment; the same script gives probe peaks `dt/(ε₀dx³) = 2.15e8` apart, 2.9e-5 after normalization | reviewer RAN (`notes/kernel_norm.py`) |
| at a constant profile the NU kernel is slower on CPU: 48³ 1.45 vs 1.17 ms/step, 96³ 4.87 vs 4.14 (one host, PEC box, one run each); GPU not measured | reviewer RAN (`notes/kernel_time.py`) |
| in float32, `a/d` and `a·(1/d)` agree bit-for-bit in only 70 % of cases, so a division kernel and a reciprocal-multiply kernel cannot be bit-identical | reviewer RAN |
| the uniform lane alone has: the GPU baked-coefficient path (`simulation.py:1531, 2500-2545`), the 4th-order stencil, periodic/Bloch, UPML, 2-D modes, float64 storage | reviewer read; leader read `simulation.py` |
| the distributed runners carry their own copies: `distributed_nu.py:88-139` reads `inv_dx` directly, `_distributed_common.py:902-956` has its own curl, `distributed_v2.py:727` reads `grid.dx` | reviewer read |
| the traced-profile `position_to_index` falls back to a uniform nominal mesh and returns the wrong cell silently (G9, `rfx/nonuniform.py:474-507`) | program document, reviewer read |
| the uniform `dx` is public surface: `freeze_mesh().dx` (`test_frozen_mesh_construction.py:22`), `checkpoint.py:58`, `io.py:963` write it; `fidelity.py:80-96` already handles both grids as sizes/nodes | reviewer read |

## 2. Decisions (fixed here)

1. **The float64 cell arrays are the source of truth.** Nodes are derived
   (cumulative sum; a uniform axis keeps the closed form so today's raster
   identity is preserved), duals and the two inverse-metric arrays are
   derived, and the constructor asserts that the cells between any two
   declared lines sum to their distance. The optional `*_f64` fields of
   `NonUniformGrid` become mandatory for concrete axes.
2. **One interface, both classes, then one class.** `cells(axis)`,
   `duals(axis)`, `index_of(axis, x)`, `node_of(axis, i)`,
   `boundary_cell(axis, side)`, `is_constant(axis)`. Both `Grid` and
   `NonUniformGrid` get it first (0a); consumers move (0b); then
   `NonUniformGrid.dx` raises on a graded axis and the classes merge (0c).
   The uniform public `dx` stays (a constant axis's cell).
3. **Two metrics, one table.** Each consumer's choice of primal vs dual is
   written once, in the interface's docstring table, with the end-entry
   rule for each (`inv_d_e[0] = 1/d[0]`, `inv_d_h[N-1] = 0`, bounding-node
   duplicate). A consumer that needs a third thing (a face-averaged size)
   adds a named accessor, never a formula at the call site.
4. **Two kernels behind one grid.** The uniform kernel stays and is chosen
   by "every axis is constant" (`is_constant`), not by "a profile was
   given" (`api/_mesh.py:153-156`). It takes its scalar from
   `cells(axis)[0]` after asserting constancy, so the metric has one
   source. The two kernels are held together by a constant-grid battery
   under a declared float32 tolerance, not by bit identity. Retiring the
   uniform kernel is a later decision from a GPU measurement (ms/step of
   both kernels and the baked path at the notch-ladder sizes 1 M, 3 M,
   13 M cells); the threshold is the PI's.
5. **Conventions the merged grid inherits from the uniform lane:** the
   `dt` formula (`dx_min/(c√3)·0.99`, and its 2-D form) and the source unit
   (field increment). The NU lane's current-normalized injection becomes an
   explicit `amplitude_kind` on the merged path, never a silent change of
   what a stored record means. Which records the switch touches is listed
   in 0b before it lands.
6. **Concrete and traced axes are marked.** A constant axis is a Python
   float for the uniform kernel to fold; a traced axis is a JAX leaf (#1190
   keeps working). `index_of` on a traced axis raises instead of falling
   back to a nominal uniform mesh (closes G9).
7. **The uniform runner's own scalar reads are not migrated** (six sites in
   `runners/uniform.py`, `simulation.py:913`): the user result is unchanged
   and only fusion order would move. Coaxial-port sites (27) migrate
   mechanically under uniform bit identity only; coax on a graded mesh stays
   unsupported by policy (`support_matrix.md`, the PI's 2026-09-20 decision).

## 3. Scope and judges, per sub-step

**0a — the interface on both classes (no consumer moves).**
Adds the accessors of decision 2 to `Grid` and `NonUniformGrid`, the metric
table (decision 3), the f64 source and the sum-equals-distance assertion
(decision 1), `is_constant`, and the traced-axis refusal in `index_of`.
Judges: (i) every existing test and committed record unchanged — the
accessors are additions; (ii) on every fixture in `tests/unit/nonuniform`
the accessors reproduce `inv_dx`/`inv_dx_h` (`_profile_to_inv_arrays`) and
`e_node_dual_spacing_at` bit-for-bit; (iii) on a uniform `Grid`, `cells`
is constant and `node_of(i)` equals the closed form `i·dx` used by
`coords_from_uniform_grid` bit-for-bit; (iv) mutation: a constructor given
cells that do not sum to a declared line distance refuses; a traced axis
asked `index_of` refuses. Also in 0a: the consumer-contract tracker — an AST
walk over `rfx/` (the pattern of `test_no_module_level_x64.py`) that counts
`.dx` reads on grid objects against an allow-list that starts at today's
sites and may only shrink.

**0b — consumers move, one module per PR, defects first.**
Order: `boundaries/cpml.py` (G11; fills the six per-face slots from
`boundary_cell` and the runway's own cells) → the waveguide-port injection
(`rfx/nonuniform.py:2091, 2134, 2202`, `sources/waveguide_port.py`; G13) →
`preflight/*` (G17) → `probes/probes.py` → `sparams/_common.py` (absorbs
`_msl_cell_profile` and the probe-interval helper as the prototypes they
are) → `sources/*`, `materials/thin_conductor.py`, `geometry/smoothing.py`,
`geometry/rasterize_grid.py`, `api/*` → the distributed and subgridded
runners → `sources/coaxial_port.py` (mechanical). Each PR carries two
judges: (i) uniform fixtures: the six final field arrays and every probe
trace `np.array_equal` against a baseline captured on the same host before
the move (the `tests/locks/test_runner_split_bit_identity.py` method; the
baseline is not committed); (ii) a graded fixture whose boundary cell and
local cell differ by 2× at the consumer's site, on which reviving the
scalar read (helper calls kept) turns a test red — the mutation the PR body
reports. The cpml PR additionally reports the x↔z relabel identity of the
absorber reflection on a symmetric profile, which is the only physics a
metric fix can be asked to preserve; it does NOT claim absorber-adjacent
grading works (G16 stays open).

**0c — the scalar goes.** `NonUniformGrid.dx` returns the cell only when
the axis is constant and raises otherwise, for one release with a
deprecation warning naming `boundary_cell(axis, side)`; the allow-list of
the 0a tracker reaches zero; the two classes merge into one, the kernel
selection reads `is_constant`, and the constant-grid battery of decision 4
is committed with its tolerance. Judges: the whole battery of 0b plus the
tracker at zero.

## 4. What is out of scope here

The mesher (`GridSpec` auto; fixed lines, ratio-capped smoothing, the
0.35-cell sheet rule as a mesher parameter with per-sheet override) — step
1, its own note. The TFSF 1-D auxiliary grid on graded cells with a leakage
witness (G14). The absorber-adjacent transition reflection measurement (G16).
The notch filter's remaining first-order term (R2: FZ-only ladder with
offset 0 and 0.35 arms at fixed FZ) — runs in parallel, it does not depend
on step 0. The differentiable edge→cell map (step 3). Retiring the uniform
kernel (needs the GPU measurement of decision 4).

## 5. Regression requirement, every PR of step 0

`scripts/ci/local.sh` green; `tests/unit/nonuniform`, `tests/contracts`,
`tests/locks` green; the #834 raster lane-identity contract untouched; no
frozen record JSON edited; the runner bit-identity harness re-baselined
only when a PR's own diff explains the change. PR bodies carry the two
mutation results of §3 and `Lane: lane:nu-mesh`; a separate Opus instance
reviews each; PRs over 200 lines in `rfx/` get the two-direction review.

## 6. Who does what

Implementation: one Opus instance per sub-step PR, from this note. Review: a
separate Opus instance per PR. The leader reads every judge's output and
writes any sentence that interprets a number. Documentation mismatches go
to #1171; the support-matrix rows move only in the release pass.
