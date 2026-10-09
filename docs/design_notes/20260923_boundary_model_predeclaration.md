# One realized boundary for every kernel — pre-declaration

> Documentation update (2026-10-03): Records moved by #1294 are in internal record 20260924-moved-from-rfx. Historical paths below refer to that archive; reproduction commands that read those records need the archived files.

**Status:** pre-declaration for the boundary-model campaign, the PI's request of 2026-09-22 ("설계 자체를
다시 고민해봐 … 시뮬레이터로서 일반화가 제한적인 영역에서 가능해야해"). The draft had three independent
reviews before this text: a design review (Opus, separate instance, 4 P1 / 9 P2 / 4 P3, seven CPU
reproductions), then the leader's recommendations reviewed by Codex and by a second Opus instance, each
independently (both SOUND WITH CHANGES; their conditions are folded in below and named where they
decided something). Both then rechecked this text (ACCEPT WITH CHANGES each); their changes are in it.
The review texts are kept off-repo under `bk-workspace/.boundary-model/`; the measurements are
committed under `scripts/diagnostics/boundary_model/`, and the reviews' scripts with the output they
printed under its `reviews/` (its README says how they were kept). Written by the absorber-lane leader. Model for the shape: the
NU grid-core note (`20260922_nu_grid_core_predeclaration.md`).
**Lane:** absorber. **Order (PI):** this campaign, then PR #1012; after that the PI chooses #952 or a
periodic unit-cell campaign.

## 0. The physics, what is wrong, and what changes for a user

A finite-difference model ends at six faces, and each face is one of four things: a perfect electric
wall (tangential E = 0 on the face), a perfect magnetic wall (tangential H = 0 on the face — a symmetry
plane: tangential H odd across it, tangential E even), an absorber (the face's cross-section continues
outward through graded loss and ends on a wall), or one half of a periodic / Bloch pair. That is a
property of the MODEL, so every run of one model — `run()`, `forward()`, a sweep, a graded mesh, a
distributed shard, a GPU — has to solve the same six faces, each where it was declared.

rfx declares the boundary per face (`BoundarySpec`) and then flattens it: a scalar (`_boundary` =
'cpml' | 'upml' | 'pec', where 'pec' also means "magnetic" or "periodic only"), a few sets, the grid's
own sets, and axis strings (`pec_axes`, `cpml_axes`) that each entry point derived its own way. Each of
about a dozen step loops then applied walls itself (1518 sites in 79 files read or write a boundary
flag; 92 apply a wall, absorber or wrap inside a step loop — `B0/INVENTORY.md`).

Measured on main at 798ec64e (12 declarations × 8 entry points, CPU and GPU, every wall and absorber
call recorded; instrumentation shown to leave probes bit-identical — `B0/MATRIX.md`, `B0/REPORT.md`),
before PR #1205:
- A magnetic face was realized two ways physically: a magnetic wall half a cell inside the face
  (`run()`, `forward()`, the graded and distributed lanes on CPU), or an electric wall (the vmap sweep,
  subgridding, ADI, and `run()`/`forward()` on GPU, whose baked fast path had no magnetic operation).
  With absorbers elsewhere, the distributed lane also put an absorber on the magnetic faces.
- The distributed lane put an absorber on a declared electric face; the graded lane solved periodic
  faces as electric walls; a plane-wave (TFSF) run wrapped the transverse faces on the uniform lanes
  whatever was declared and absorbed them on the graded lane; a waveguide-port run turned declared
  absorbing transverse faces into electric walls on the uniform lanes and kept them on the graded lane.
- In the all-absorber point-source box `run()` backed every absorber with an electric wall and
  `forward()` did not; a waveguide-port run backed its transverse faces but not its own absorber faces.
- Preflight printed "All checks passed" on every one of these runs.

PR #1205 (f8da752a, another session, 2026-09-22) then fixed one class on the uniform scan and the
graded step: a per-face rule `resolve_wall_faces` (`rfx/boundaries/pec.py`) that never zeroes E on a
magnetic face, the fast path fenced off when a magnetic face is declared, the sweep refusing magnetic
faces. What it left, and what this note is about:

- **A magnetic wall sits half a cell inside the declared face.** A 24 × 20 × 4 mm cavity, magnetic x
  faces, electric y, z faces (`B0c/REPORT.md`): the magnetic-only mode (0,1) converges to 7.4948 GHz
  (analytic 7.4948); mode (1,1) reads 9.9288 / 9.8408 / 9.7981 GHz at dx = 1 / 0.5 / 0.25 mm against
  9.7561 GHz for walls on the declared faces — a derived wall separation of 23.02 / 23.50 / 23.75 mm,
  the declared 24 mm minus one cell. First order in dx: +1.8 % at dx = 1 mm (λ/30), outside the v2 1 %
  bar until dx ≤ 0.5 mm. A symmetry half-model is half a cell narrower than drawn.
- **A periodic axis is one cell longer than declared.** Every axis gets L/dx + 1 nodes and the periodic
  roll wraps all of them, so the realized period is L + dx (`rfx/grid.py:212-227`, `rfx/core/yee.py:171-218`);
  where a requirement turns an absorber into a periodic face the pads are wrapped too (2·pad + 1 cells).
  A declared 24 mm periodic parallel-plate ring resonates at 11.970 GHz at dx = 1 mm, −4.2 % against
  c/24 mm (the continuum shift to c/25 mm is −4.0 %, the rest is dispersion); a `RISUnitCell` 10 mm cell
  at dx = 0.5 mm is a 10.5 mm cell (`reviews/opus_review1/`, CPU; B0b's records show the same
  extra node for its own 7.4948 mm cell).
- **A Floquet port's scan angle never reaches the fields**: at 0 / 15 / 30 / 45° the recorded fields
  are bit-identical, the port's own injection and S-parameter functions have no callers, and
  `rfx.RISUnitCell.sweep_angle` returns a peak-normalized probe spectrum labelled as reflection
  (`B0b/REPORT.md`; the normalization makes the output independent of the field amplitude to 2.5e-16 —
  a synthetic check, `reviews/codex_review/checks.json`). No calibrated reflection comes out of that path at any angle.
- **An oblique Bloch plane wave with subpixel smoothing drops the Bloch phase in its E update** and is
  accepted; with a Debye slab the same run crashes on a scan-carry dtype
  (`reviews/opus_review2/bloch_branches.py`, CPU; the size of the error was not measured). The Bloch rule lives in one difference
  operator and the smoothing branch has its own curl.
- The subgridded and ADI lanes still apply an axis-wide wall and no magnetic operation; the distributed
  lanes consume the magnetic faces but compose them with an electric wall on the face plane, or with an
  absorber when the other faces absorb (`B0/MATRIX.json`, `rfx/runners/_distributed_common.py:510-552`);
  the feature rewrites above are unchanged; in an all-absorber box `run()` still backs the absorbers and
  `forward()` does not.
- What is consistent: the all-electric cube's TM110 is 8.8305 / 8.8322 GHz at dx = 1 / 0.5 mm through
  `run()`, `forward()`, the sweep and the graded lane alike (analytic 8.8327; `B0/WITNESS.md`).

**Why the lattice does this.** rfx stores tangential E on the declared faces and gives every axis
L/dx + 1 nodes so that "PEC walls at index 0 and index N span exactly N·dx" (`rfx/grid.py:212-213`), and
its differences shift in zeros at the array ends. Under that storage convention two pieces are missing,
each a separate discretization choice: a magnetic face has no image, so the nearest tangential-H plane
(half a cell in) is zeroed instead (`rfx/boundaries/pmc.py`: "0.5·dx inside the wall"); and a periodic
axis has no periodic topology, so the fence-post node lengthens the period. Where no wall operation runs
at all, the zero-padded shift is itself a wall: a magnetic wall half a cell outside the lo face and an
electric wall one cell beyond the hi face (`reviews/opus_review2/`, 1-D eigenvalue maps of rfx's
kernels: L + 1.5 dx, quarter-wave family). The missing Floquet k_t and the per-kernel flags are separate data-flow defects.
The half-cell offset was known (#722's "ninth surface", 2026-08-28) and answered with documentation
(`tests/unit/boundaries/test_pmc_plane_convention.py`), while the PMC physics gates (10 % and 3 %
tolerances) cannot see a one-cell shift and #1164's cavity check used the (0,1) mode, which does not
depend on the wall separation. Declared and realized were not compared.

## 1. Facts this note rests on

| fact | checked by |
|---|---|
| the flattening of the spec | leader read `rfx/api/__init__.py:563-606, 699-704` |
| inventory: 1518 sites / 79 files / 92 APPLY in 12 lanes | Codex, `B0/INVENTORY.md`, `B0/inventory.csv` |
| the realized-boundary matrix at 798ec64e (66 measured, 30 refused) | Codex, `B0/MATRIX.md`, `B0/MATRIX.json`, VESSL 369367263554–561 |
| magnetic cavity: CPU at a4aaa862, GPU relaunch (electric walls on GPU: no (0,1), separation 24.02 / 24.005 / 24.001 mm) | Codex, `B0c/REPORT.md`, `B0c/RESULTS.json`, `B0c/gpu_relaunch/`, VESSL 369367263573–579 |
| PEC cube W1, PMC line W3, backing W4 | Codex, `B0/WITNESS.md`, `B0/witness/*.json`, VESSL 369367263562–564 |
| Floquet scan angle has no effect; RIS output is a normalized spectrum | Codex, `B0b/REPORT.md`, `B0b/angle_equalities.json`; `reviews/codex_review/checks.json` (normalization) |
| what #1205 changed | leader read PR #1205 and `rfx/boundaries/pec.py` `resolve_wall_faces` at c2dbf922 |
| the image wall on rfx's kernels: second order on uniform and graded axes and with a dielectric face node; with the (2,4) ribbon its error equals the electric-wall ribbon's at two meshes; a post-step correction with vacuum coefficients is first order | prototypes in the three reviews, `reviews/` (1-D eigenvalue maps; a two-spacing 3-D cavity); Codex: a mirrored-domain comparison gives 0 V/m for vacuum, εr = 4 and εr = 4 with σ, and −0.0550 / −0.0563 / −0.0550 V/m on the Debye / Lorentz / mixed E updates unless each migrates — prototypes, not production-kernel validation |
| the curl is spelled in at least ten places (smoothing/anisotropic, Debye, Lorentz, mixed ADE, UPML, fast path, graded kernels, port current loops, distributed, subgrid) | second Opus review, read with file:line |
| periodic L + dx; RIS 10 → 10.5 mm; index map has no node in [L − dx/2, L] once the fence post goes | `reviews/opus_review1/periodic_period.py`, `ris_grid.py`; `reviews/opus_review2/d2_index.py`; mechanism read by the leader at `rfx/grid.py:212-227`, `rfx/core/yee.py:171-218` |
| `add_tfsf_source` requires `boundary='cpml'` and refuses periodic overrides | second Opus review, `rfx/api/__init__.py:2301-2310` (leader read) |

## 2. Decisions

1. **One boundary object, built in a fixed order and rebuilt at dispatch.** A stable interior mesh plan
   first (independent of exterior pads); then every feature's requirement is collected; then the face
   kinds are resolved (`resolve_kinds`): per face PEC | PMC | ABSORBER | PERIODIC, per axis INVARIANT for
   the 2-D modes, model-level the absorber type and its scalar parameters, per periodic axis whether it
   is Bloch, the conformal flag, the origin of each face; then ONE grid is built from the kinds (a
   periodic axis gets L/dx nodes and no pads, every other axis L/dx + 1 plus its pads); then `realize`
   gives the realized plane of every face in metres, the realized period, the terminal plane and
   condition of every absorber, and `restrict(box | shard)` for subgrids and shards (interior faces are
   SEAM or INTERFACE). `realize` only VALIDATES geometry predicates (a waveguide port's realized aperture)
   and refuses on a mismatch — it never changes a kind. The object is a static, hashable descriptor
   (kinds, pairings, integer counts, capability keys) plus dynamic pytree leaves (traced metrics, k_t),
   so a scan angle vmaps and does not recompile (`bloch` leaves the static argnums of `update_h`/`update_e`).
   Registration order does not change the solved geometry.
2. **The object drives three mechanisms and nothing else applies a boundary.** (i) Boundary-aware
   neighbour and sample operations, staggering- and direction-aware, used by every derivative AND every
   boundary sample: tangential H odd and tangential E even across a magnetic face (N = cells on the
   axis: E nodes 0…N, H samples 0…N−1, stored ghost H[N]; rfx arrays are indexed by stored length, so
   the implementation states its map; low face: D_H[0] = 2H[0]/d[0]; high face: D_H[N] = −2H[N−1]/d[N−1]
   with the stored ghost refreshed or canonicalized), periodic, Bloch (the existing per-neighbour envelope phase exp(−j·k·d), not a
   seam-only phase). (ii) The absorber's auxiliary update (CPML ψ; UPML coefficients; ADI's conductivity
   layer inside its implicit operator), parameters from the object, the kernel choosing the structure.
   (iii) Electric zeroing after the E update on PEC faces and absorber backings. `resolve_wall_faces`
   (#1205) is the seed of its OUTPUT only: it becomes `electric_faces(object)`, computed from each face's
   kind, with no periodic argument (the graded lane passes "no periodic axis" today,
   `rfx/nonuniform.py:2546-2548`, and so solves periodic faces as electric walls), no `pec_axes` argument
   and no default branch. No axis-level wall, no `pec_axes`, no `cpml_axes`. What each
   stored ghost cell holds under each kind is written down, and ghosts are masked from energy, flux,
   DFT and far-field sums. A sum over a plane or volume that meets a magnetic face counts the
   face-node samples with half their dual-cell weight, so every sum is the half domain's value; a
   consumer that reports a full-model quantity (decision 6) multiplies by 2 per mirror face and
   records that it did.
3. **The step order is pinned and tested.** H: update → TFSF/waveguide H → absorber H → Kottke H mask →
   magnetic sources, with magnetic images evaluated at consumption (after every H change). E: update →
   design box / Kerr → TFSF/waveguide E → absorber E → electric zeroing → conformal / PEC edges /
   occupancy → sheet → RLC → pre-injection DFT → soft sources → wire V/I. TFSF auxiliary grids update in
   their own slot. Distributed: walls before the E ghost exchange (#1041). Subgrid: walls after
   coupling. Tests: an edge where an electric and a magnetic face meet keeps E = 0 after every later
   operator; a soft source on an electric face; a magnetic source on the H sample next to a magnetic
   face, and a perpendicular CPML touching that face, each against a mirrored-domain reference (these
   two see a stale stored ghost; the first two cannot).
4. **Features state admissible sets; the declaration is authoritative.** Each feature gives, per face,
   the kinds it can work with (a 1-D plane wave along x with E along z: y ∈ {PERIODIC, PMC},
   z ∈ {PERIODIC, PEC}, x ∈ {ABSORBER}; the open-domain oblique method: y ∈ {ABSORBER}, z ∈ {PERIODIC};
   an oblique Bloch cell: BLOCH on the transverse axes; `compute_rcs` at normal incidence: absorbers;
   a waveguide port: ABSORBER on its axis and, when its REALIZED aperture equals the realized cross
   section, PEC on the transverse faces; the 2-D modes: z INVARIANT whatever scalar token the model was
   given, so a legacy `boundary='cpml', mode='2d_tmz'` keeps working, and an explicit per-face z
   declaration is accepted only as the mode's equivalent wall, PEC for TMz and PMC for TEz). The resolver intersects all features'
   sets; a declared face inside the intersection is kept; an empty intersection or a declared PEC/PMC
   outside it is refused, naming every feature involved. A declared ABSORBER pair outside it is turned into
   the PERIODIC (or Bloch) pair only when (1) that replacement, with its phase, belongs to EVERY active
   feature's admissible set — a postcondition asserted, not assumed — and (2) the realized operator and
   the excitation have the translational (Bloch) symmetry along that axis: materials, conductors, loads,
   localized field updates and the source profile all invariant (the laterally infinite slab under a
   plane wave is the case this admits); a preflight finding names the face. Otherwise refused. A
   full-aperture waveguide declared with absorbing transverse faces is refused and asked to declare PEC:
   its admissible set is {PEC}, and its TE10 profile is not transversely constant even when the
   material is. `Simulation`'s default is `boundary='cpml'`, so this refuses the commonest spelling of a
   waveguide run; the refusal gives the PEC declaration to write, and B5 converts the committed scripts. The add-time guards of `add_tfsf_source` are
   lifted so that the admissible declarations can be written. (Codex disagreed with any rewrite and, on
   recheck, accepted the bounded rule with conditions (1) and (2); the second Opus review agreed under
   the invariance condition; the leader takes the bounded rule because it is exact where it applies and
   refuses everywhere else.)
5. **Each kernel declares its capability, and tests hold it.** Refusals raise at dispatch — also with
   preflight bypassed — naming the face, the feature and the kernel. Enforced by: B0's matrix harness as
   an always-on contract on CPU (two steps; the realized physics class of every cell computed from
   seeded field operations and realized planes, not from call names); a physics mini-battery per kernel
   (the electric cube, the magnetic cavity judged quantitatively, the periodic ring, a CPML box energy
   check, a dielectric face node with subpixel smoothing, a Debye face node); an AST registry allowing
   difference spellings (`_shift_*`, `jnp.roll` differences) and wall/absorber primitives only in the
   operator module and the object-driven entry points (an allow-list that may only shrink); a
   completeness test that every dispatch target has a declaration. The fast path is an optimization:
   its eligibility is the object together with the kernel feature descriptor (the existing material,
   sheet and design-variable exclusions stay), and it falls back rather than refusing.
6. **A magnetic wall sits on the declared face, by image (D1).** Second order (all three reviews). An
   object on a magnetic face's node plane is that object in the FULL symmetric model — the imaged update
   and the metric's `dual[0] = d[0]` already mean that — so a soft source, an E probe, an RLC or a sheet
   there is allowed, and a port there reports the full symmetric port's impedance. Consumers whose own
   spelling lacks the image (the port current loops `_bwd_h` / `_bwd_neighbor`, flux and far-field planes
   on the face) are refused on that plane until each moves. (Codex preferred refusing every object on the
   plane; the second Opus review showed that this refuses the main use of a symmetry plane — a feed on
   a bisected line or patch — and the TEz thin box #1205 recommends. The leader takes "full" with
   per-consumer refusals.)
7. **A periodic axis has the declared period (D2).** L/dx nodes; `position_to_index`/`index_of` wrap
   modulo N on a periodic axis (L maps to 0); a DFT plane at L and at 0 are the same plane (for Bloch
   fields they differ by the period phase, stated); full-period sums count N cells. Commensurability:
   an automatically chosen dx is snapped to L/ceil(L/dx), never coarser than asked (reported); an explicit dx that does not
   divide L within tolerance is refused with the nearest dividing dx; two periodic axes need a common
   dx or are refused with a suggestion. The metric table gets a periodic row (coordinated with the NU
   lane's step 0c, which owns `grid.py`).
8. **One Bloch realization**: the complex-envelope path (#404); real kernels support k_t = 0 only; the
   real-part wrap in `rfx/floquet.py:122-181` goes.
9. **Every absorber is backed by an electric wall** in every kernel (D5; both reviews agree). The object
   records the terminal plane and its condition. Records that move are listed when the change lands.
10. **ADI's absorber is named for what it is (D4):** a `conductivity_layer` absorber kind in the
   declaration, serialization and padding; ADI supports it (all faces) and a declared 'cpml' on ADI is
   refused, naming it. Its reflection goes to the support matrix with a witness. An explicit exception
   to "no new boundary kinds": it names an existing approximation.
11. **Consumers read the object**; the legacy views become derived, tracked by an AST allow-list that
   only shrinks, then go.
12. **The absorber metric per face comes from the NU lane's grid interface** (its step 0b moves
   `boundaries/cpml.py` first); #1012 lands inside the profile function; #952 becomes a kernel change
   over the object's absorber faces.
13. **A waveguide port with magnetic transverse faces is refused (D7)** until magnetic-wall mode profiles
   exist.

## 3. Sub-steps and their judges

- **B1 — the object and the resolver; no kernel moves; a live baseline.** The matrix is re-measured on
  current main (after #1205). Judges: every cell's realized physics class — (a) magnetic realized as
  electric, (b1) face node plane dead, (b2) face node plane shorted, (c) electric face absorbing, (d) periodic realized as
  electric or absorber, (e) period ≠ declared, (f) backing convention, (g) absorber on a reflecting face,
  (h) magnetic wall off its declared plane — is computed from realized planes and seeded field
  operations and compared with the object; each departure is a STRICT expected failure pinned to its
  assertion, removed by the step that fixes it. The AST registry and the mini-battery land here.
- **B1.5 — refusals first, over every kernel, keyed to B1's re-measured cells.** Magnetic faces on
  subgrid, ADI and the distributed lanes; the graded lane with periodic faces; the distributed lane with a
  mixed electric/absorber layout; a Floquet port at a scan angle ≠ 0; any RIS or Floquet reflection output
  without native, reference-normalized S data (at every angle, including capacitance sweeps); an oblique
  Bloch plane wave with subpixel smoothing or a dispersive material; a waveguide port with magnetic
  transverse faces; a declared 'cpml' on ADI (until B4 names the conductivity layer). Each raises its OWN refusal (face, feature and kernel named) with
  preflight enabled and bypassed; an unrelated exception (a scan-carry dtype error) does not count. The PR lists every committed test that flips from "accepts" to "raises" and
  converts it; four of the seven tests that pin H_t = 0 at Yee index 0 are distributed-lane tests
  (`test_boundary_pmc_distributed` ×3, `test_distributed::test_apply_pmc_local_pad_x_targets_real_face`)
  and flip here. `known_limitations.md` entries for the periodic period and the half-cell magnetic wall
  until they land. Does not touch `boundaries/cpml.py`.
- **B2 — the periodic period.** Judges: the periodic ring at c/L (a commensurate fixture), first order
  gone, and the fence-post node restored → red; the index-wrap test; a seam probe and a seam flux case;
  the normal-incidence invariant (a transversely constant broadside field is independent of the period);
  a non-commensurate case that exercises decision 7's snap and its refusal.
- **B3 — the curl spellings consolidated, then the image and the electric zeroing from the object, in the
  uniform scan and `forward()`.** Every uniform E-update branch goes through one boundary-aware curl (the
  graded branches through `curl_h_nu`); the port loops read a boundary-aware neighbour accessor. Judges
  (quantitative, not detection windows): f(0,1) = 7.4948 GHz within a stated tolerance and f(1,1)'s
  derived separation → 24 mm at second order over three dx; a dielectric face node and a Debye face
  node against a mirrored-domain reference, the dielectric arm with subpixel smoothing on a Box that
  extends at least one cell beyond the face (the face node is then interior, so the arm tests the curl
  and not the smoothing); a flux plane crossing a magnetic face against the mirrored-domain reference
  (half the full flux, decision 2's half weight). Mutations: (i) the image disabled (the free
  termination: a mode near 8.05 GHz, which a window would have passed); (ii) electric zeroing restored
  on the x faces with every image call intact → the (0,1) mode disappears; (iii) the half-cell wall
  revived — H_t zeroed at the first and last physical H samples with the image calls intact: f(0,1)
  stays at 7.49 GHz and only the separation judge turns red (23.02 mm at dx = 1 mm); (iv) the face node
  counted at full weight in the crossing flux plane → a first-order excess. B3 also lands a refusal:
  subpixel smoothing with a smoothed material surface within half a cell of a magnetic face plane.
  Under the image that face column becomes live, and a Box ending on the plane gets the fill fraction
  clip(0.5 − sdf/cell, 0, 1) = 0.5 there (`rfx/geometry/smoothing.py`), so its tangential E column
  carries (ε + 1)/2 instead of the mirrored model's ε — a first-order error where a bisected
  microstrip's field is strongest. The refusal lifts when the even extension is designed (§4).
  Tests that encode the half-cell / dead-slab realization are rewritten here with the physics reason:
  #1205's `test_magnetic_wall_faces_not_shorted.py` (its one-cell line measured a line of width dx
  centred on the face, half of it outside the domain, which a closed form happens to fit); the three
  non-distributed tests of the seven pinning H_t = 0 at Yee index 0 (`test_boundary_pmc_runtime` ×2 and
  the composition test oq8; the other four flip at B1.5); the TEz thin-box test (kept, re-judged under
  "full"); and #1162's `tests/unit/ports/test_lumped_port_known_load_line.py` (closed form ± 0.05,
  passivity, lumped == wire, three loads) and `test_lumped_two_port_matched_line.py` (the matched
  two-port, the S21 closed form and a 1° S21 phase gate), whose port sits on the same y = 0 magnetic
  face plane. The image moves that line's x-end walls out by half a cell each, so B3 re-measures these
  and the ports lane re-judges the phase gate. The fast-path arm records whether the fused kernel ran.
- **B4 — the other kernels, one per PR,** each with B1's seeded backing-plane assertion and the mini-battery arms relevant to it; an
  unsupported configuration refuses rather than being skipped as supported (graded, sweep,
  subgrid, distributed ×3 with `restrict`, ADI with its named conductivity layer, the probe reference
  runs, the TFSF auxiliary grids), serialized per module with the NU lane's 0b: its PR for a module lands
  first.
- **B5 — requirements as admissible sets.** Lift the TFSF add-time guards; the invariance-gated rewrite;
  the waveguide realized-aperture predicate (behaviour changes listed: an aperture port inside a larger
  absorbing domain stops being solved in an electric box; the graded lane's waveguide runs with
  CPML-declared transverse faces move; every uniform-lane full-aperture waveguide script that relies on
  the default `boundary='cpml'` flips from accepted to refused, among them the rectangular-waveguide
  port's authoritative gate `tests/unit/sparams/test_waveguide_twoport_contract_v1.py` and
  `tests/crossval/test_waveguide_broad_e5.py` — a heuristic `git grep` finds 17 test and example files
  that call `add_waveguide_port` without declaring PEC; the PR lists and converts each to a PEC
  declaration, and their results must not move); Floquet with k_t (with `tfsf_2d`'s `complex(...)`
  phase helper under a traced angle); RCS; the INVARIANT 2-D axis; `add_waveguide_port`'s grid and
  plane resolution move from registration to dispatch, and a boundary-changing `add_*` after
  `freeze_mesh` invalidates the derived setup or is refused. Judges: an
  admissible invariant slab accepted and a noninvariant finite scatterer refused; an invariant-material
  full-aperture guide whose set excludes PERIODIC refused; a compatible explicitly declared guide keeps
  its faces; each refusal with preflight enabled and bypassed; an independently referenced oblique Bloch
  slab (magnitude and phase); a plane wave spanning PMC transverse faces (the parallel-plate
  arrangement decision 4 admits) against its closed form. Mutations, helpers kept: the invariance predicate made unconditional, and
  a replacement outside the admissible set allowed — each must turn a judge red.
- **B6 — the legacy views go**; the legacy-use allow-list actually empty, and every earlier physical
  judge still green (the tracker's count is accounting, not the physics).

## 4. Out of scope
New boundary kinds other than the named conductivity layer (Mur, higher-order ABCs, impedance walls);
mixing CPML and UPML; Kottke smoothing's even extension at a magnetic face and far-field boxes cut by a
reflecting face (refused until each is designed); the absorber-adjacent graded transition (NU G16); the
continuation contract's own rules (#1167 — this note changes where absorbing faces are read from, not
what continues through them); the periodic unit-cell feature itself (sheet source with the Bloch phase,
TE/TM, higher Floquet orders, witnesses), which the PI schedules after #1012.

## 5. Regression requirement, every PR
`scripts/ci/local.sh` green; `tests/unit/boundaries`, `tests/contracts`, `tests/locks` green; no frozen
record edited without its PR's own explanation; the two mutations of the workspace rule in the body; a
separate Opus instance reviews each PR, two directions over 200 lines of `rfx/` or when a gate moves;
`Lane: lane:absorber`.

## 6. Who does what, and the other lanes
Measurements and implementation: Codex from written briefs; judgments, this note and every interpreting
sentence: the leader. The NU lane owns `boundaries/cpml.py`'s metric (its 0b) and `grid.py` (its 0c): B2
and B3 onwards follow it module by module. #1205 was done in this lane by another session; its rule is
this design's seed, and its tests are rewritten in B3 with the reason. The one-cell PMC line —
#1205's `test_magnetic_wall_faces_not_shorted.py` and the ports lane's two #1162 tests on the same
line (`lane:coax-mixed-port`) — is rebuilt in B3 as a line whose port and load sit at least two cells
from any magnetic face, or on the face under the "full" convention with a migrated current loop.

## 7. Decision record, 2026-10-05: S1 boundaries (the structure plan's realized-model stage)

The PI's 2026-10-04 restructuring makes the remaining kernel work of §3 (B4, B6) part of stage S1:
one declaration, one realized model, every path reads it. For boundaries the duplicated part today
is not the absorber law (one `_cpml_profile` since #1374) but the **reading of the declaration into
per-face depths**: the same rule — a wall face or a face on a non-absorbing axis gets no pad, an
absorbing face gets its declared thickness or the scalar budget — is written out in `grid.py`
(`Grid._face_pad`), in `nonuniform.py` (`_face_pad`, with a range check the uniform copy lacks), in
`init_cpml` (`n_active`), in the multi-device builder (`_distributed_boundary_layers`, which omits the
non-absorbing-axis test), in `api/_compile.py`, and again in several preflight readers. #1346 was two
of these copies disagreeing (the graded grid dropped per-face depths, a 4/8 request built 8/8).

Decisions:
1. The boundary model gains one function that returns, per face, the kind, the declared depth, the
   realized depth (pad cells) and the absorber's terminal plane — no coefficient arrays: `init_cpml`
   stays the one coefficient builder and takes its depths from this record. Every place listed above
   reads the record; no module outside `rfx/boundaries/` derives a per-face depth from the declaration.
2. Two PRs. PR1: the record, the uniform and graded grid builders, `init_cpml`, `api/_compile.py`, and
   the contracts below. PR2: the multi-device builder and the preflight readers. The TF/SF auxiliary
   grids (own spacing, own depth) and the ψ-update copies (stage S3) are out of scope.
3. Judges.
   - Non-regression: every cell of the S0 path-equivalence matrix (#1484) that passes before passes
     after, bit-for-bit on the realized record and within its own bars on fields; no committed test,
     lock or frozen record changes value. Expected to move: nothing. A cell that moves stops the PR
     for a written root cause.
   - One source (structural): a contract test scans `rfx/` and fails if a module outside
     `rfx/boundaries/` reads a declaration's per-face depth (`face_layers.get(…)`, `face_layers[…]`,
     `resolved_lo/hi_thickness`, `lo/hi_thickness`) other than through the record. Mutation (a): the
     scan disabled → red on a seeded file. Mutation (b): restore #1346 in the graded builder (per-face
     depth replaced by the scalar budget while it still calls the record for everything else) →
     the physical judge below goes red.
   - Physical: an asymmetric declaration (x low 8 layers, x high 16, y PEC, z absorbing at 12) on the
     uniform and the graded single-device paths (PR2 adds the two multi-device paths): the plane-wave
     reflection of each x face, measured as a reflection coefficient over a band
     (`tests/_absorber_witness.py`), agrees between the paths within its float32 floor, and the deeper
     face reflects less than the shallower one by the margin the single-device run shows. The
     restored-#1346 mutation makes the graded x-high face read like 8 layers.

Addendum, 2026-10-05 (leader, after the PR1 implementer stopped on a mismatch). The uniform and graded
copies differ on two inputs: a per-face depth above the budget on a face of a non-absorbing axis (the
uniform grid raises, the graded grid ignores it), and a fractional depth on an absorbing face (the
uniform grid truncates 1.5 to 1, the graded grid raises). Neither input reaches either builder through
the public API: `Boundary` refuses a non-integer thickness and a thickness on a non-absorbing face before
any grid is built (checked on 31ecbefb with `Simulation(boundary=…)`, uniform and graded). Only a direct
call of `Grid(…, face_layers=…)` or `make_nonuniform_grid(…, face_layers=…)` reaches them. Decision: the
record applies the strict rule to every face it is given — an integer between 0 and the budget, raising
otherwise — so the two builders agree; no public result changes. A test pins both inputs on both builders.

Addendum 2, 2026-10-05 (leader, second implementer stop). A reachable difference, through the exported
low-level `rfx.run(grid, materials, n_steps, boundary="cpml")`: a uniform grid built with
`cpml_axes="z"` has zero pads on x and y, yet `init_cpml` still builds full 8-layer absorbing profiles on
the x and y faces, and `rfx.run` applies them because its own `cpml_axes` argument defaults to "xyz". The
absorber then sits on the outermost eight cells of the physical domain on x and y. The graded builder
gives no-op profiles there. `test_an_axis_outside_cpml_axes_keeps_its_pre_876_applied_depth` pins the
uniform behaviour as a regression baseline (it guarded #876's clamp, not the physics of the dual knob).
`Simulation` never reaches this: it passes one consistent axis set. Decision for PR1, which promises no
result change: the record's realized depth is the pad (0 on those faces), and `init_cpml` keeps its
present profile depth for faces outside the grid's absorbing axes, labelled in code as the legacy dual
declaration with a pointer here; the structural contract allow-lists exactly that line. The dual
declaration itself (grid axes vs the runner argument) is a separate decision, raised with the lead: the
candidate is that the runner reads the grid's axes and refuses a conflicting explicit argument, with
the #876 test rewritten with its reason. That change moves results on the low-level API and does not
belong in this PR.

Addendum 3, 2026-10-06 (leader, PR2 implementer stop). With a waveguide port, the grid absorbs only on
the port axis (`_waveguide_cpml_axes`, `api/_compile.py`): a model declared `cpml` on every face is
solved with electric walls and no pad on the transverse faces, while preflight's own face reader
(`_preflight_face_layers`, `preflight/absorber.py`) reports the declared absorber there (4 layers on
y and z for a 4-layer x-port guide), so its absorber advisories describe absorbers that do not exist.
Decision for PR2: preflight reads the realized record, as every other reader does; on a waveguide model
its transverse faces report 0 layers. This changes diagnostic output only, never a computed result; the
PR lists every test whose expected advisories move and why. The rewrite itself — a feature silently
turning declared absorbing faces into walls — is the same class as the TF/SF transverse period
(§3 B5, decision 4: a full-aperture waveguide's admissible transverse set is {PEC}) and joins PR3:
a declared absorber the feature cannot keep is refused or must be declared as the wall it becomes.

Addendum 4, 2026-10-06 (leader, after PR2's two reviews). (1) Addendum 3's rule — preflight reports the
realized record — covers every feature that changes the realized faces, not only waveguide ports: with a
Floquet port (which sets the periodic axes itself) preflight reports the periodic faces as unpadded; the
PR lists every advisory that appears or disappears. (2) A face declared `cpml` with zero layers is an
electric wall on one device (the absorber's backing at the domain face). On two devices with the uniform
mesh it gets no wall: the multi-device lane builds walls only from the declared PEC/PMC face lists, and a
probe differs from the one-device run by up to 78 % of its peak (found by the fresh-eyes review; graded
two-device and a declared PEC face agree). Same cause as this PR — a reader not taking the face from the
record — and in a file the PR already touches, so it is fixed here: the multi-device walls come from the
record (PEC faces and zero-depth absorber backings), judged by that probe equal to one device within the
cross-trace bar, with the old wall rule as the mutation. This is the PR's one computed-result change.

Addendum 5, 2026-10-07 (leader, before PR3 starts; scope approved by the lead, [L:245043]). PR3 is the class
Addendum 3 named: a feature that silently rewrites declared faces. Two features do it today.
(a) A waveguide port: `_waveguide_cpml_axes` (`api/_compile.py`) makes the grid absorb only on the port
axes, so a model declared `cpml` on every face (the default) is solved with electric walls on the
transverse faces. `boundaries/depths.py` reads that method — a reader of the feature, not of the declaration.
(b) A TF/SF plane-wave source: `add_tfsf_source` requires `boundary='cpml'` and the run then wraps the
transverse axes periodically, pads included. On the default box a finite scatterer is solved as an array
of period N·dx with the pads (37 mm on the 2026-10-05 record, rfx-archive `20261005-tfsf-transverse-period`,
f8b3b79a: backscatter off by several dB at 7 GHz and 20–30 dB toward c/period).
B5 as written also moves computed results (an aperture port in a larger absorbing domain, the graded
lane's waveguide runs, a pad-free periodic grid, Floquet k_t, RCS, the 2-D invariant axis, port
resolution moving from registration to dispatch). Stage S1's judge is "expected to move: nothing".
Decision: two PRs.
PR3a — no computed result changes; what was silently rewritten is refused or must be declared.
 1. Waveguide. This departs from decision 4, which refused every full-aperture guide whose transverse
    faces are absorbing, the default included. Reason (lead, 2026-10-07): `Simulation`'s `boundary`
    defaults to `cpml`, so that rule breaks every waveguide script over an absorber the user never
    declared. Rule now, for a guide whose REALIZED aperture equals the realized cross-section:
    - `boundary` not passed (the default applied): the transverse faces default to PEC, the guide's
      walls. The face record holds PEC there, and the run says so once (a warning naming the faces until
      stage S2's diagnostics exist). This is the realized model of today; nothing computed moves.
    - an absorbing transverse face passed explicitly (`boundary='cpml'`, or a `Boundary`/`BoundarySpec`
      naming an absorber on that face): refused at dispatch, also with preflight bypassed; the message
      gives the declaration to write (PEC on the transverse faces, absorber on the port axis).
    - PEC transverse faces declared: kept, bit-identical to today.
    Telling "not passed" from "passed 'cpml'" needs a sentinel default on `Simulation(boundary=...)`;
    that change is part of this PR, and `sim._boundary` keeps reading 'cpml' for the default so no other
    reader changes. Committed scripts that pass `boundary='cpml'` explicitly with a full-aperture guide
    are converted to the PEC declaration; their realized record and results must be bit-identical before
    and after. An aperture port smaller than the cross-section keeps today's behaviour in PR3a, labelled
    in code as the legacy rewrite with a pointer here (it moves in PR3b).
 2. TF/SF: a declared transverse absorber pair is kept as today's periodic wrap (pads included, so the
    realized grid does not move) only when decision 4's two conditions hold — the periodic pair is in
    every active feature's admissible set, and materials, conductors, loads, localized field updates and
    the source profile are invariant along that axis; a preflight finding names the face. Otherwise
    refused, naming the face and the feature, with the two ways out (declare the axis periodic for an
    array; `closed_box=True` for a finite scatterer). The add-time guard that demands `boundary='cpml'`
    is lifted so the admissible declarations (periodic, or PMC/PEC per polarization) can be written.
 3. The face record takes the realized faces from the resolved declaration; no module outside
    `rfx/boundaries/` asks a feature which axes absorb (`_waveguide_cpml_axes` loses its readers).
PR3b — everything in B5 that moves a number: the realized-aperture port inside an absorbing domain, the
pad-free periodic grid, Floquet with k_t, RCS, the invariant 2-D axis, dispatch-time port resolution.
Judges (PR3a). Non-regression: the S0 path-equivalence matrix and every committed lock, oracle and frozen
record unchanged; converted scripts bit-identical on the realized record. Refusals, each with preflight
enabled and bypassed, on every path (uniform, graded, both multi-device, subgridded): explicit-`cpml` full-aperture guide refused; the
default (no `boundary` passed) accepted with PEC transverse faces in the record and one warning; the same guide with PEC transverse faces declared runs and equals today's
run bit-for-bit; a laterally invariant slab under a TF/SF plane wave accepted with the finding; a finite
scatterer under the same source refused (the 37 mm array of the record is the reproduction); `closed_box`
unaffected. Mutations, helpers kept: invariance predicate made unconditional; a replacement outside the
admissible set allowed; the silent waveguide rewrite restored — each turns its judge red.
Memory: ledger grep for "waveguide transverse", "TF/SF period", "37 mm" — none; #1221 entries concern PMC
planes. Design note §2 decision 4 and §3 B5; Addendum 3. First-pass count of files calling
`add_waveguide_port` without an explicit boundary declaration: 31 of 158 by text heuristic (the note's
earlier heuristic said 17); the build-only census in the PR is authoritative.

Addendum 5a, 2026-10-07 (leader, after the PR3a implementer stopped on a mismatch). Addendum 5 asked for
the realized face record of a converted waveguide script to be bit-identical before and after. It cannot
be: on main the transverse faces of a full-aperture guide are recorded as kind ABSORBER with realized
depth 0 (the absorber's electric backing wall at the domain face, Addendum 4), and declaring them PEC —
or defaulting them to PEC — changes the recorded kind to PEC (reproduced by the implementer on
`tests/unit/sparams/test_waveguide_twoport_contract_v1.py`: four transverse kinds ABSORBER → PEC, shape
and pads equal). That change of label is the purpose of the PR: the record now says what is solved.
Corrected requirement: before and after, the realized depth of every face, the pad counts, the grid
shape and every face's terminal plane are equal, and every computed result the test reads is
bit-identical; the kind of a transverse face of a full-aperture guide changes from ABSORBER (depth 0) to
PEC and nothing else in the record changes. Readers that branch on the kind (multi-device walls,
preflight, exporters) are in the reach table, each shown to give the same realized walls for
"ABSORBER, depth 0" and "PEC".

Addendum 5b, 2026-10-07 (leader, second implementer stop). Three points the stop raised.
(1) `FaceDepth.declared` on a wall face. On the committed two-port guide the transverse faces read
(ABSORBER, declared 1, realized 0) and after the PEC declaration (PEC, declared 16, realized 0): a PEC
face cannot carry a thickness (`Boundary` refuses it), so the field falls back to the scalar budget. On a
wall that field is the legacy scalar view and means nothing physical. Identity for a converted script is
therefore judged on: realized depth of every face, pad counts, grid shape, the physical position of
every face's terminal plane computed from the REALIZED pads, and every computed result the test reads.
The kind and the `declared` field of a full-aperture guide's transverse faces may change; nothing else.
(2) `model.realize` places `terminal_m` from the DECLARED layers (`rfx/boundaries/model.py:~202`), so for
a face declared absorbing but realized with depth 0 it reports a plane the grid does not have. That is a
reader not on the realized record — the class this stage removes. In PR3a: list every reader of
`terminal_m` in the reach table; if none feeds a computed result, make `realize` take the plane from the
realized depth and show no result moves; if one does, stop on that item and report which.
(3) The S0 path-equivalence TF/SF cell (`tests/contracts/path_equivalence/builders.py`, plane wave over a
finite eps_r 2.5 box, default absorbing transverse faces) is the arrangement PR3a refuses: its realized
permittivity is not invariant along y or z, and declaring y, z periodic is a different grid (no pads; the
periods do not share the cell). The cell stays in the matrix on every path by redrawing its dielectric as
a laterally invariant slab (same x extent and permittivity, spanning the whole transverse extent
including the pads), which keeps the legacy wrap and the realized grid. Its numbers are those of a
different structure and are re-recorded; this is the one fixture whose values change in PR3a, listed in
the PR with the old finite-box arrangement kept as a refusal judge. No other committed value may move.
The same rule applies to any other committed TF/SF case the census finds non-invariant: report each
with its structure before converting; a case that pins a physical number (a lock, an oracle, a
cross-validation record) is NOT converted by the implementer — stop on it and report.

Addendum 5c, 2026-10-08 (leader, after two independent implementations were reviewed; decisions by the
lead, [L:2de396]). Addendum 5 rested on assumptions about the paths that were not checked against the
code; four were wrong. What is assumed now is written per path, and each line is to be confirmed by
reading the code and by a build before it is relied on.
(1) Where the waveguide rule applies. Checked (both reviews, reproduced): on the uniform single-device
path a full-aperture guide declared `cpml` is built with realized transverse depth 0 (electric walls);
on the non-uniform path the same declaration builds real transverse absorbers (8-cell pads on the
reviewers' case), so nothing is rewritten there. Rule: the default-to-PEC and the refusal of an explicit
absorbing transverse face apply on a path if and only if, on main, that path realizes depth 0 on a
declared absorbing transverse face of a full-aperture guide. Every other path keeps today's behaviour in
PR3a, unchanged and unrefused. To be confirmed per path by the implementer (code read + build):
distributed_v2, distributed_nu, subgridded, ADI. That uniform and graded meshes solve different models
for one declaration is a PR3b item; PR3a measures it once (S11 and S21 over the band, one WR-90
full-aperture case, default declaration, uniform against graded) and reports the curves.
(2) A declared PMC or PEC transverse wall under a TF/SF plane wave. On main the wrap overwrites it
silently. PR3a: refused unless the model is invariant along that axis (then wall and wrap give the same
field and today's run is kept bit-identically, with the finding); honouring walls on a non-invariant
model is PR3b. `boundary='upml'` with a TF/SF plane wave stays refused, as on main.
(3) Invariance of conductor masks. The legacy wrap leaves the last transverse edge row of a tangential
conductor mask empty, so a conductor sheet spanning the whole transverse extent is not invariant on the
realized arrays, and main leaks −57.7 dB through such a plate (reviewer, reproduced). The invariance
check ignores that duplicated end edge, so the sheet runs as today, bit-identically. The seam is recorded
in the ledger with its number as a PR3b item.
(4) The RCS tutorial (`examples/tutorials/rcs_scattering.py`: PEC sphere, default absorbing faces, plane
wave) solves an array, the arrangement PR3a refuses. It is converted to `closed_box=True` in this PR and
its snapshot regenerated; it is the second fixture whose values change (after the S0 TF/SF cell), with
before/after in the PR.
(5) Not a feature rewrite, so not refused: a low-level `rfx.run(..., periodic=..., tfsf=...)` call in
which the caller passes the periodic flags; a wall on the PROPAGATION axis of a TF/SF source (as on
main). A traced material override that cannot be judged at trace time keeps today's behaviour, with a
finding that the invariance was not judged.
(6) The frozen design format gains no field. Export of a default full-aperture guide writes the resolved
declaration (PEC transverse faces, absorber on the port axis); any other default-boundary model exports
exactly as on main; a document written by main imports and runs as it did on main.

## Addendum 5d (2026-10-08, lead decision L:0f6340 after measurement) — the default full-aperture guide on the graded mesh is refused

Addendum 5c (1) kept today's behaviour on the paths where main keeps real absorbers on the transverse faces of a
full-aperture guide (non-uniform, distributed_nu) and asked for one measurement. The measurement is in: with the
boundary not passed, the graded path solves a structure without side walls. Against the uniform path on the
two-port contract fixture, |S11| is 14.5-19.4 dB lower on 16 of 20 bins and |S21| is above 0 dB on 13 of 20 bins
(max +0.93 dB). That is a non-physical result handed over silently, so 5c (1) is replaced for these paths:

1. Non-uniform and distributed_nu: a full-aperture waveguide port with the boundary NOT passed is refused at
   dispatch (run, forward, the waveguide S-matrix calculator, with and without preflight). The message names the
   transverse faces and the way through: declare them, e.g. `boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}` for a
   guide along x. This is a stopgap against a silently wrong result; the structural fix (one boundary model realized
   the same way on every path) stays the first item of PR3b, cause issue tracker 1221.
2. An explicitly declared transverse absorber on these paths keeps today's behaviour (the user asked for an open
   cross-section and gets it). An explicitly declared wall keeps today's behaviour.
3. Measured basis for the way through (rfx-archive `rfx/records/20261008-s1-pr3a-guide-default-uniform-vs-graded/`):
   with the side walls declared, the graded path equals the uniform path where both realize the same cross-section
   (0.937 mm cell, 44 nodes on both: |S21| within 0.001 dB, |S11| within 0.03 dB, phase within 0.01 deg, 20 bins).
   At cells where a 40 mm width is not a whole number of cells the uniform builder realizes one more node than the
   graded builder (41.22 vs 39.35 mm at 1.874 mm), and the two differ by that width: |S21| within 0.18 dB, |S11|
   within 0.8 dB off the null bin, phase up to 5.8 deg near cutoff. That one-node difference is a PR3b item, not
   part of this refusal.
4. Judge: one contract test pins both sides on a guide whose cross-section is a whole number of cells on both
   paths: default on the graded path is refused before any field step; explicit walls on the graded path agree with
   the uniform default guide on |S11|, |S21| (dB) and S21 phase at every bin, with a bar derived from the measured
   0.03 dB / 0.01 deg (bar: 0.1 dB and 0.1 deg; a deep-null bin below -40 dB is excluded from the |S11| comparison
   by name). Mutation: the refusal removed on the graded path turns the first half red.
5. What this assumes about each path, checked: uniform, distributed_v2, subgridded realize depth 0 and are governed
   by Addendum 5 (default -> walls with one warning; explicit absorber refused) — build outputs in the PR3a report
   (`BASE_PATHS`): pads (4,4,0,0,0,0). Non-uniform and distributed_nu realize real pads (4 on all faces) — same
   report. ADI: not reachable (constructor refuses the absorber). Not checked by the leader: whether distributed_nu
   reaches the same dispatch point as the single-device graded path for the default-provenance test; the implementer
   confirms by reading and by a build, and reports.

## Addendum 5e (2026-10-09, leader, after the verification review of PR3a) — the plane-wave rule applies only where the wrap is installed; the finding names every wrapped face

Addendum 5 asked for the plane-wave (TF/SF) refusal "on every path". The premise was not checked per path, the same
error Addendum 5c (1) corrected for the guide. Checked now by reading main: the graded-mesh stepper installs no
periodic boundary at all (`rfx/runners/nonuniform.py:1300-1303`: the assembler passes `periodic=(False, False,
False)` unconditionally; `rfx/sources/tfsf.py::tfsf_boundary_flags` says "not used by the non-uniform runner"), and
the reviewer reproduced it (transverse faces ABSORBER, realized depth 4, no periodic stencil). So on the non-uniform
and distributed_nu paths nothing is rewritten, a finite scatterer is not turned into an array, and the remedy the
refusal names ("declare y='periodic'") does not exist there.

1. Non-uniform and distributed_nu: the plane-wave boundary rule of Addendum 5 does not apply. A plane wave with a
   finite or an invariant model keeps main's behaviour: no refusal from this rule, no `tfsf_transverse_periodic`
   finding. Existing refusals of that path (oblique incidence, incidence along z, forward "uniform only") are
   untouched. What a two-plane plane-wave slab between transverse absorbers solves on the graded mesh is not
   characterized here (support matrix: experimental); it is a PR3b item under cause issue 1221, not a PR3a change.
2. Uniform path, oblique incidence (Bloch method): the wrapped transverse axis along which the wavevector is tilted
   keeps main's operator in PR3a (the field is not invariant there by construction; judging a structure along it
   needs the Bloch-period statement and belongs to PR3b). It is no longer silent: the `tfsf_transverse_periodic`
   finding names every face that is solved periodic, including the tilt axis's, and says for the tilt axis that the
   structure was not judged for invariance there. The other transverse axis is judged and refused as at normal
   incidence.
3. Judges added with this addendum: (a) graded mesh, plane wave, finite box and invariant slab: admitted, same realized
   grid and pads as main, no finding of this rule (mutation: the uniform rule applied on the graded path → red);
   (b) uniform, oblique incidence: a box finite along the non-tilt axis is refused; a box finite only along the tilt
   axis is admitted and the finding names the tilt-axis faces with the not-judged wording (mutation: oblique treated
   as "no axis to judge" → red; mutation: tilt-axis faces dropped from the finding → red); (c) graded guide
   (Addendum 5d): a port whose ranges are given explicitly and cover the whole aperture is refused like the
   ranges-omitted port; a port covering part of the aperture is admitted on the graded mesh with the boundary not
   passed (mutations: explicit-range port not refused → red; partial-aperture port refused → red).
4. Per-path statement, corrected: a waveguide port on the subgridded path ends in main's existing NotImplementedError
   on both trees, so "default → walls" is not reachable there; distributed_v2 reaches the uniform rule through the
   single-device fallback.
