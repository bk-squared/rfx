# One realized boundary for every kernel — pre-declaration

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
