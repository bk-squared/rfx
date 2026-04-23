# SBP-SAT all-PEC box refinement specification

## Status

This is the **Milestone 5** specification artifact for the future
all-PEC arbitrary 6-face box-refinement lane.

It is an **internal implementation contract**, not a public support claim.
The current public/claims-bearing SBP-SAT surface remains the narrower
all-PEC full-span x/y z-slab lane documented elsewhere.

## Scope

This document specifies what must be true before any implementation widens the
current z-slab-only coupling to an arbitrary all-PEC 3-D refinement box.

In scope:

- a general face-operator contract for `x`, `y`, and `z` oriented interfaces
- a six-face normal/tangential component table
- a deterministic edge/corner ownership policy
- a staged implementation plan
- a benchmark matrix that stresses non-z faces, edges, and corners

Out of scope:

- CPML/UPML coexistence
- PMC or periodic `BoundarySpec` coexistence
- partial-x/y or graded-profile refinement
- port / observable expansion beyond the already documented roadmap
- a proof of long-time stability for the full 6-face, edge, and corner system

## Baseline assumptions

The Milestone 5 box lane remains inside the current research envelope:

- coarse and fine regions share one global `dt`
- the refinement ratio is uniform in `x`, `y`, and `z`
- the box is axis-aligned
- the surrounding simulation boundary is all-PEC
- the current pairwise SAT relaxation remains a candidate coupling model,
  not a finalized proof-backed general SBP-SAT derivation

## Current reusable baseline

Milestone 5 is not a greenfield design.  The current z-slab lane already gives
the reusable seed for the generalized box contract:

- `rfx/subgridding/face_ops.py` — current 2-D z-face prolongation/restriction
  bundle and norm-compatibility checks
- `rfx/subgridding/sbp_sat_3d.py` — current z-face slice helpers, tangential
  extraction/scatter helpers, and pairwise SAT coupling
- `tests/test_sbp_sat_face_ops.py` — operator and trace-shape regression tests
- `tests/test_sbp_sat_api_guards.py` — runtime boundary/misuse hard-fail surface

The Milestone 5 plan generalizes those z-face-only pieces; it does not replace
them with a second unrelated subgridding stack.

## Notation

Let the refined coarse box occupy

- `i in [fi_lo, fi_hi)`
- `j in [fj_lo, fj_hi)`
- `k in [fk_lo, fk_hi)`

with coarse box shape

- `ni = fi_hi - fi_lo`
- `nj = fj_hi - fj_lo`
- `nk = fk_hi - fk_lo`

and fine box shape

- `nx_f = ni * ratio`
- `ny_f = nj * ratio`
- `nz_f = nk * ratio`.

Face names use the refined-box outward normal convention:

- `x_lo -> -x̂`
- `x_hi -> +x̂`
- `y_lo -> -ŷ`
- `y_hi -> +ŷ`
- `z_lo -> -ẑ`
- `z_hi -> +ẑ`

Trace storage order is **axis-order based, not sign based**.  For a face with
normal axis `n`, the tangential axes are stored as `(t1, t2)` in the natural
remaining-axis order used below.  Low/high faces of the same orientation share
that same trace order; sign-sensitive quantities must use `normal_sign`
explicitly rather than flipping array order implicitly.

## General face-operator contract

Each oriented face uses one orientation bundle with the following fields:

- `face`: one of `x_lo`, `x_hi`, `y_lo`, `y_hi`, `z_lo`, `z_hi`
- `normal_axis`: `x`, `y`, or `z`
- `normal_sign`: `-1` for `*_lo`, `+1` for `*_hi`
- `tangential_axes`: ordered pair `(t1, t2)`
- `tangential_e_components`: ordered pair `(E_t1, E_t2)`
- `tangential_h_components`: ordered pair `(H_t1, H_t2)`
- `coarse_shape`: `(n_t1, n_t2)` for that face
- `fine_shape`: `(n_t1 * ratio, n_t2 * ratio)`
- `coarse_slice(...)`: coarse-grid extractor for the face
- `fine_slice(...)`: fine-grid extractor for the face
- `ops`: the 2-D prolongation/restriction/norm bundle for that orientation

### Operator math

For Milestone 5, the face operators stay orientation-agnostic in algebra and
orientation-specific only in shape/slicing:

- build one 1-D cell-centered prolongation matrix on each tangential axis
- define `R_t = P_t^T / ratio` on each tangential axis
- define the 2-D face restriction as `R_face = R_t1 @ face @ R_t2.T`
- define the 2-D face prolongation as `P_face = P_t1 @ face @ P_t2.T`
- define one diagonal norm per oriented face using the tangential cell area

Because the current coarse grid is isotropic in the accepted lane, each face
cell still has coarse area `dx_c^2` and fine area `(dx_c / ratio)^2`.  The
orientation affects only face shape and slice placement, not the scalar area.

### Candidate SAT extension

The current z-face candidate extends to all faces component-wise:

- use the same `alpha_c = tau / (ratio + 1)`
- use the same `alpha_f = tau * ratio / (ratio + 1)`
- couple matching tangential components by restriction/prolongation mismatch
- do **not** encode face-sign-dependent permutations in the trace arrays

If a later derivation shows that sign, material scaling, or time staggering
must differ by face orientation, that is a spec change and must land together
with benchmark and test changes.

## Six-face component table

| Face | Outward normal | Tangential axes `(t1, t2)` | Tangential `E` | Tangential `H` | Coarse face slice | Fine face slice |
|---|---|---|---|---|---|---|
| `x_lo` | `-x̂` | `(y, z)` | `(Ey, Ez)` | `(Hy, Hz)` | `i = fi_lo`, `j:fj_lo->fj_hi`, `k:fk_lo->fk_hi` | `i = 0`, `j:0->ny_f`, `k:0->nz_f` |
| `x_hi` | `+x̂` | `(y, z)` | `(Ey, Ez)` | `(Hy, Hz)` | `i = fi_hi - 1`, `j:fj_lo->fj_hi`, `k:fk_lo->fk_hi` | `i = nx_f - 1`, `j:0->ny_f`, `k:0->nz_f` |
| `y_lo` | `-ŷ` | `(x, z)` | `(Ex, Ez)` | `(Hx, Hz)` | `j = fj_lo`, `i:fi_lo->fi_hi`, `k:fk_lo->fk_hi` | `j = 0`, `i:0->nx_f`, `k:0->nz_f` |
| `y_hi` | `+ŷ` | `(x, z)` | `(Ex, Ez)` | `(Hx, Hz)` | `j = fj_hi - 1`, `i:fi_lo->fi_hi`, `k:fk_lo->fk_hi` | `j = ny_f - 1`, `i:0->nx_f`, `k:0->nz_f` |
| `z_lo` | `-ẑ` | `(x, y)` | `(Ex, Ey)` | `(Hx, Hy)` | `k = fk_lo`, `i:fi_lo->fi_hi`, `j:fj_lo->fj_hi` | `k = 0`, `i:0->nx_f`, `j:0->ny_f` |
| `z_hi` | `+ẑ` | `(x, y)` | `(Ex, Ey)` | `(Hx, Hy)` | `k = fk_hi - 1`, `i:fi_lo->fi_hi`, `j:fj_lo->fj_hi` | `k = nz_f - 1`, `i:0->nx_f`, `j:0->ny_f` |

## Edge and corner policy

A face-only update is **not** allowed to own every perimeter degree of freedom.
If two or three face operators write the same edge/corner trace independently,
that DOF is over-penalized and the scheme loses a clear ownership model.

Milestone 5 therefore uses the following ownership decomposition:

1. **Face interior ownership**
   - Each face SAT operator owns only the strict face interior.
   - In face coordinates `(t1, t2)`, the owned set excludes the perimeter
     indices in both tangential directions.
2. **Edge ownership**
   - Each of the 12 box edges is owned by a dedicated 1-D edge operator.
   - A face operator must not write an edge DOF once edge support is enabled.
   - Edge traces are ordered by the remaining edge axis and carry the field
     components tangential to both adjacent faces.
3. **Corner ownership**
   - Each of the 8 box corners is owned by a dedicated corner operator or an
     explicit corner-resolution rule.
   - Corners must never be updated implicitly by two edge operators plus one
     face operator.
4. **Implementation gate**
   - If dedicated edge and corner ownership is absent, arbitrary 6-face box
     refinement stays disabled and the runtime must keep hard-failing any
     attempt to widen beyond the current z-slab lane.

This policy is intentionally conservative: it forbids “face-only but including
perimeter” implementations until the line/point ownership problem is solved
explicitly.

## Implementation plan

### Phase 5A — orientation-general operator layer

- Introduce an orientation-agnostic `FaceOrientation` description.
- Generalize the current z-face operator bundle into a 2-D face bundle that can
  be instantiated for `(y, z)`, `(x, z)`, and `(x, y)` tangential planes.
- Keep the z-face names as compatibility shims until callers are migrated.

### Phase 5B — x/y/z face extraction and scatter

- Add explicit extraction/scatter helpers for all six faces using the table in
  this document.
- Lock component ordering by tests before any SAT logic is generalized.
- Preserve the current z-face helpers as a specialization of the same contract.

### Phase 5C — face-interior SAT only

- Generalize SAT coupling to x-, y-, and z-oriented **face interiors**.
- Do not include edge or corner DOFs in this step.
- Keep runtime behind an internal flag or blocked path until edge/corner work is
  present.

### Phase 5D — edge operators

- Introduce dedicated 1-D edge operators for the 12 coarse/fine edge pairs.
- Define which field components are shared on each edge and how they are
  prolonged/restricted.
- Add ownership tests showing that face operators leave edge DOFs untouched.

### Phase 5E — corner resolution

- Introduce a dedicated corner rule for the 8 coarse/fine corners.
- Add tests proving that no corner DOF is written more than once per half step.
- Keep the rule all-PEC only.

### Phase 5F — runtime enablement

- Only after Phases 5A-5E land may the runtime/API accept an arbitrary box.
- The initial accepted surface remains:
  - all-PEC outer boundary only
  - one axis-aligned box only
  - no CPML/UPML/PMC/periodic coexistence
  - no port/observable expansion beyond the existing roadmap gates

## Benchmark matrix

All Milestone 5 benchmarks remain **uniform-fine reference comparisons**, not
public physical R/T claims.

### Required fixtures

1. **x-face oblique proxy benchmark**
   - point source positioned so the dominant wavefront crosses an `x`-oriented
     face obliquely before reaching the probe set
   - compare arbitrary-box run vs uniform-fine reference
2. **y-face oblique proxy benchmark**
   - same idea for `y`-oriented faces
3. **z-face regression benchmark**
   - preserve the current z-face proxy lane so the box work does not regress the
     Milestone 3 baseline
4. **edge stress benchmark**
   - source/probe geometry chosen so energy crosses an `x-y`, `x-z`, or `y-z`
     edge neighborhood rather than only face interiors
5. **corner stress benchmark**
   - diagonal propagation through a box corner neighborhood

### Required metrics

For each fixture, record against the uniform-fine reference:

- point-probe DFT amplitude error
- point-probe DFT phase error
- at least one probe near each newly activated face / edge / corner region
- coarse-interior overlap sanity checks (no hidden duplicate dynamics path)

### Initial benchmark targets

Before public promotion, the box lane must first satisfy the same internal proxy
floor already used for the z-slab lane:

- amplitude error `<= 5%`
- phase error `<= 5°`

Those targets are only the **first internal gate**.  Any edge/corner instability,
non-monotone convergence, or face-orientation asymmetry blocks runtime enablement
until the spec or implementation is corrected.

## Implementation gate

Milestone 5 completes when this specification exists and is regression-locked.
It does **not** mean arbitrary-box refinement is implemented.

Implementation remains blocked until all of the following are true:

- the orientation-general face contract is accepted
- edge and corner ownership rules are implemented, not implied
- the benchmark matrix above is implemented and passing
- the runtime still hard-fails unsupported non-PEC / port / observable cases

Until then, the supported SBP-SAT runtime remains the current z-slab-only lane.
