# Thin-sheet realization: from cell rasterization toward a plane boundary condition

> **SUPERSEDED IN PART by #931 (the lattice ownership contract), 2026-09-07.**
> Kept as dated history and deliberately NOT rewritten.
> This note's error family — a one-cell PEC slab presenting only its LOWER
> node plane (#706), a live normal edge sampling whatever the geometry left in
> the cell (#702), the sheet-cavity thickness error — was the diagnosis that
> #931 acted on, and the direction it proposed (a sheet as a plane BC that
> owns no cell) is what shipped. The *fixes* it lists as current are gone:
> `two_plane` is removed, the #702 resample is deleted (a sheet has no own
> cell to re-sample), and the one-cell slab now realizes both faces because it
> is a volume. Read it as the problem statement, not as a description of the
> tree.
> The current rule is
> `docs/design_notes/20260906_plan_realign_lattice_ownership.md` §1, and for
> users `docs/public/guide/materials-geometry.mdx` ("How conductors land on
> the lattice").

Status: DIRECTION (PI-endorsed 2026-08-25). Not a commitment; a recorded target
for the next architecture review. Triggers to revisit: the in-plane
discretization verdict of the current crossval campaign, and an external
design review (Codex) of this note's open questions.

## The error family this addresses

One family, met repeatedly whenever a conductor is thinner than a cell on a
graded rectilinear mesh — every member is a public issue:

- a one-cell PEC slab presents only its LOWER node plane as an electrical
  wall; its normal-E edge stays live, so displacement current tunnels through
  the "metal" (#706 — fixed by the opt-in two-plane realization);
- the live edge samples whatever permittivity the geometry left in that cell
  — vacuum, if the dielectric stack abuts the sheet faces (#702 — fixed by
  node+d/2 resampling);
- sheet faces sit off-lattice unless the mesh is built for them; congruent
  conductors then rasterize to different cell counts (#703 preflight family);
- an adjacent-sheet cavity reads up to tens of percent long in electrical
  thickness because each sheet's cell carries dielectric that physically is
  copper (quantified by the sheet-cavity preflight check).

The current stack of fixes (#702 resample + #706 two-plane + #703 checks +
mesh-side face registration) makes cell rasterization HONEST, but each fix
patches one symptom of the same mismatch: the model wants a surface, the
lattice stores volumes.

## Prior art (verified 2026-08-25)

- **openEMS/CSXCAD**: thin conductors are first-class 2D objects. Ideal metal
  = `CSPropMetal` on a 2D primitive lying ON a mesh plane (the mesher is
  REQUIRED to put lines on metal faces/edges — `AddEdges2Grid`, thirds rule
  at in-plane edges). Finite-conductivity foil = `CSPropConductingSheet`
  (thickness, conductivity — a sheet-impedance/skin model on the plane), so
  17 um copper is a boundary condition, not a rasterized volume. The whole
  error family above is dissolved at model-definition time.
- **Meep**: the opposite dissolution — no graded mesh at all; subpixel
  smoothing is dielectric-only ("only ... the instantaneous part of eps and
  mu"; dispersive/PEC excluded), metals staircase at uniform resolution.
  Right for photonics, not for multilayer boards.

## The direction

Promote "sheet on a mesh plane" to the DEFAULT thin-conductor realization:

1. **Mesh contract**: the mesh builder guarantees a node plane at each thin
   conductor's registered plane (mid-plane or face — one convention, stated).
   This is already the working practice of the current crossval mesh builder;
   it becomes a guarantee instead of a discipline.
2. **One sheet operator**: tangential-E treatment applied AT that plane, with
   surface impedance Zs as the parameter — Zs = 0 recovers the PEC sheet,
   finite Zs recovers the lossy foil. rfx already carries the second half
   (#677 sheet-impedance operator, threaded through the MSL lane in #679);
   the direction is to let it own the PEC case too, replacing per-cell
   pec_mask realization for sub-cell conductors.
3. **In-plane edges**: edge-aware line placement (openEMS thirds rule) in the
   mesh builder for the NU lane, instead of ever-finer uniform in-plane cells.

## Open questions (for the design review, before any implementation)

- AD: gradients of geometry parameters currently flow through the rasterized
  masks; a plane-BC realization moves them into (plane position, Zs, edge
  positions). What stays differentiable, and through which parameters?
- vmap/batch sweeps over geometry: plane positions become traced quantities —
  does the realization stay shape-stable?
- Junctions: via barrels meeting sheets, sheet-sheet overlaps, lumped-element
  terminations at sheet edges (the termination-edge enumeration family).
- Stacks with sub-cell separations (sheet planes closer than one cell).
- CPML and eigenmode-witness interactions; migration path (two_plane stays
  the opt-in volumetric realization; plane-BC lands as a third, then flips
  default only on evidence).

## Why record this now

The measured cost of the status quo is no longer hypothetical: the crossval
campaign spent multiple GPU runs and several review cycles discovering,
one symptom at a time, what a plane-BC model would have excluded by
construction. The PI's read: this direction is plausibly the largest
remaining trial-and-error reducer for board-class work.

## Design review outcome (Codex, 2026-08-25)

An independent review (Codex CLI over this repo + issues) endorsed the
direction with one architectural sharpening and several concrete checks:

- **Build it as oriented surface topology, not mask reinterpretation**: a
  `SheetPlaneSpec` (static normal axis, plane slot, footprint, impedance
  model, junction IDs) with an explicit junction graph. Deriving the edge
  set from unioned cell masks would preserve the root cause under a new
  name — every incident in the error family above was an
  orientation/ownership ambiguity of the mask representation.
- **Keep three realizations**: plane sheet (default for sub-cell planar
  metal) / volumetric (barrels, plating, finite thickness) / conformal
  later. `two_plane` remains the compatibility realization.
- **Semantics to state first**: the current operator mixes a penetrable-film
  contract (folded sigma preserves DC film Rs) with a Leontovich opaque-
  boundary Rs input — filed as #711 with the analytic infinite-sheet R/T/A
  case as the pin.
- **CPML**: the sheet update replaces, not composes with, CPML-corrected
  edges; acceptable only because preflight P1.9 refuses geometry in the
  absorber — documentation ask folded into #711.
- **AD/vmap**: fixed-topology parameterization (plane slots with
  softplus-spaced coordinates; Zs via softplus; never differentiate across
  the finite-Zs-vs-PEC branch). Batch sweeps bucket by topology. The
  current nonuniform position lookup (argmin + float) is a known cliff.
- **Junctions are the hard part**: via-barrel-to-sheet contact needs an
  explicit conductor graph; lumped terminations must own enumerated gap
  edges (the termination-edge family) rather than thawing sheet-owned ones.
- Reviewer pushback worth keeping: before scoping this as greenfield, check
  whether the graded-mesh builder's existing plane/face registration can
  serve as the plane registry.

On the separate loaded-Q question the reviewer ranked **port/feed
representation first** (loaded Q is external-Q-sensitive; the locus is a
port observable), radiation loading second — and proposed the decisive
first measurement: a source-off, port-free ring-down with a per-channel
energy budget (Q_radiation / Q_conductor / Q_dielectric / Q_port from
outward flux + per-mechanism dissipation vs stored energy). That
experiment separates "resonator too lossless" from "feed undercoupled"
in one run and is the planned follow-up to the current pattern run.
