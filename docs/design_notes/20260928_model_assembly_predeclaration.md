# Pre-declaration: one model assembly and one set of conventions for every path (2026-09-28)

Status: pre-declaration by the NU leader. This is step 3 of the order the PI approved on 2026-09-27; step 2 (lane admission, #1355) merged on 2026-09-28 as bcf7920d. Phase P0 is frozen by this note (§3), and its scope is §2.6. Each later phase that moves a result brings its own before/after to the PI (§2.4).

## 1. What a user gets today, and why

Since #1355, every path refuses a declared input it does not carry. That closes the silent drops. It does not touch the second failure class: a path carries an input, but it turns the declaration into kernel arrays with its own code, so one physical rule reaches only the paths whose code calls it. Measured examples:

- **A dielectric's faces sit in different places on different paths.** #1213 gives each E edge the mean ε and σ of the four cells around it. The single-device lanes use that rule. The multi-device lane still takes the ε of the one cell that owns the edge, and an εr 4 block differs from single-device `run()` by 1.7 % of the probe peak, while vacuum agrees to 7e-5 (#1303). Declaring conformal walls switches every dielectric in the model back to the per-cell rule, which moves `run()` by a third of the peak (#1306).
- **The same port drive has different units.** The uniform lane adds `Cb·w/d_par` per cell and the graded lane `Cb·w/dV`, so absolute fields differ by 1/dx², about 4e6 at 0.5 mm (#1266). At the time of this predeclaration (before #1442), with `amplitude_kind=None`, the Yee lanes added `Cb·w` (their own deprecation warning says so), while ADI adds `w` (#1355 implementer, read only).
- **A boundary sits in a different place.** A magnetic wall sits half a cell inside its face, and a periodic axis is one cell too long. Both hold on the uniform lane itself (#1221).
- **The same feature lands on different nodes.** Points and conductors snapped with different tie rules (#1295, #1342; the PI fixed one rule, the lower node, on 2026-09-28).
- **The same sweep covers different times.** `vmap_material_sweep` runs every swept value for one step count at its own dt (#1293).
- **A dielectric lands on the wrong cells.** The NU smoothing fallback places some shapes on uniform-mesh cells (#1298).

The leader's code reading (READ-ONLY, not yet measured) adds three more:
- ADI uses one per-cell ε for Ex, Ey and Ez (`rfx/adi.py:807`), not the four-cell mean.
- The distributed graded path uses the four-cell mean for its source drive but the owning cell for its E update (`rfx/runners/distributed_nu.py:1853-1914`), so there are two rules inside one path.
- The subgridded lane uses the four-cell mean inside, but the per-cell ε in its coarse/fine interface terms (`rfx/subgridding/jit_runner.py:852-858`).

Why the table missed the ADI case: a `carries` cell asks whether declaring the input moves the record. A wrong interface rule still moves the record, so "moved" cannot see it. Only a comparison against the right answer can.

## 2. Decisions

1. **The reference is physics, not the uniform lane.** The uniform lane has its own departures (#1221), so it cannot be the yardstick. Each convention is fixed against an analytic case, and every path is then held to that convention. The conventions:

   | quantity | convention | physical check |
   |---|---|---|
   | ε, σ at an E edge | the four-cell mean (#1213); on a graded mesh the weighting is open (§4) | slab reflection and cavity resonance vs analytic |
   | μ at an H edge | same rule, dual | same, magnetic slab |
   | source and port drive | the `amplitude_kind` contract (#571): 'current' = amperes, 'field' = E increment, one unit on every path | delivered power vs analytic for a short dipole |
   | point and conductor snap | nearest node, a tie goes to the lower node (PI, 2026-09-28) | co-location of a port and its conductor |
   | boundary face | the face sits on the declared plane | cavity resonance vs analytic for PEC, PMC and periodic walls |
   | step count | set by physical time, not by count | equal-time records across dt |

2. **One assembler, many consumers.** A single module builds a *realized model* from the declaration and a grid. The realized model is:
   - the per-component ε/σ/μ arrays;
   - the pole parameters per component;
   - the PEC edge masks, sheets and wires;
   - each source and port's stamp with its unit factor;
   - the boundary face descriptors.

   Every path consumes it:
   - the multi-device lanes shard it;
   - ADI reads the per-component arrays;
   - the subgridded lane builds one on each grid with the same function;
   - the calculators take the same object.

   No path computes a coefficient from the declaration itself.
3. **The gate compares realized arrays, not fields.** Two time-stepping schemes (Yee and ADI, coarse and subgridded) give different fields for the same model, so field parity needs a tolerance per scheme. The arrays a kernel consumes do not: ε at an edge is either the convention's value or it is not. For every carried physics input, a contract cell dumps each path's realized arrays for the cell's model and compares them with the assembler's arrays on that path's grid, exactly or to float rounding.
4. **Measure first, then move results.** Phase P0 changes no result:
   - it adds the dump hooks and the realized-array table;
   - it records every departure as a strict xfail, naming its issue;
   - it measures the three READ-ONLY cases above.

   Each later PR moves one convention on the paths that depart from it. It shows before/after on the physical check and on the user-facing numbers it moves, as #1341 did with the patch cross-validation. It asks the PI before merging when a published or gallery number moves.
5. **Scope freeze per PR.** A departure found during a PR becomes a table cell (xfail against its issue), not new PR work.
6. **P0 scope.**
   - Quantities:
     - the E-edge ε and σ, as the per-component arrays each lane's E update uses;
     - the H-edge μ;
     - the drive scale each lane applies to a declared soft source, lumped port and wire port, for each `amplitude_kind` including `None`.
   - Lanes: the nine time-stepping lanes of the #1338 table.
   - Reference: the assembler's arrays, built on the lane's own grid. For P0 these are the single-device uniform lane's functions, `component_e_materials` and the `amplitude_kind` contract (`rfx/api/_source_semantics.py`), applied to that grid. A departure from the reference is recorded, not fixed. The reference is not presumed correct where §4 leaves a question open.
   - Not in P0: dispersive pole arrays, PEC masks (the #931 realized-geometry records already cover them), boundaries (#1221) and the calculators (#1362). Dielectric placement of the cell array (#1298) and defects inside the shared helper are also not checked by P0; these belong to the assembler phase (§2.2).

## 3. Judges for P0 (frozen)

- **J1, P0 moves nothing.** The locks (`tests/locks/`), the two A/B bit-identity gates and the #1338 cells file are unchanged, and so are all existing tests. A dump hook reads the arrays a lane already computes. It computes nothing new on the path, and when it is off it adds no work.
- **J2, every departure is in the table.** Every case in §1 that falls within P0's quantities (§2.6) ends up as one of two things: a passing realized-array cell, or a strict xfail naming its issue. The three READ-ONLY cases are measured, and each becomes a pass or a named xfail. A departure that fits no existing issue goes to one umbrella issue for this note, not to a new issue each.
- **J3, the gate can fail.** Revive a known departure with the lane's own helper calls kept, and the realized-array cell must go red. Two examples: make the single-device uniform lane take the owning cell's ε, or have the graded lane drive a port in the uniform lane's unit. Give at least three such mutations, with the tests that went red.
- **J4, the dump is the kernel's input.** For each lane, show once that the dumped arrays are the arrays the kernel consumes. Perturbing a dumped array and feeding it back must move the lane's field record, and returning it unchanged must reproduce the record bit for bit. A dump that the kernel does not read cannot fail.

Physical checks against analytic cases (the convention table's right-hand column) belong to the phase that moves each convention, not to P0.

## 4. Open questions for the PI

- **The four-cell mean on a graded mesh.** The four cells around an edge have different widths, and today they are averaged equally. The width-weighted mean is the consistent generalization and equals today's rule on a constant axis. It would move graded results only where a dielectric face sits next to a size change. P0 measures how much before anyone decides.
- **Boundaries are not an open question here.** #1221 is a defect to fix; the PI said so on 2026-09-28. Its design is already on main (#1220, `docs/design_notes/20260923_boundary_model_predeclaration.md`), and it is the absorber lane's campaign B1–B6. #1355 covered most of B1.5. This note does not re-decide boundaries. The boundary row of the convention table points to that campaign.
- **Order after P0:** one proposal is interface ε (#1303, #1306, ADI), then drive units (#1266 and amplitude_kind), then #1293 and #1298. Boundaries run in parallel in the absorber lane (#1221). #1342 (the snap) is already separate.

## 5. Out of scope

- The calculators' own coverage (#1362). They adopt the realized model when they are classified.
- Performance and the uniform-kernel retirement rule.
- New physics.
