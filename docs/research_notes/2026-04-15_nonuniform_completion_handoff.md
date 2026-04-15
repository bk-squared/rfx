# NonUniformGrid Completion — handoff and plan — 2026-04-15

## Why this note exists

The SBP-SAT modified-Yee subgrid effort (branch `subgrid-modified-yee-singlegrid`, frozen at commit `7758c9d`) validated the interior physics but was blocked by a structural PML–SBP-stencil incompatibility. See companion note `2026-04-15_sbp_sat_cpml_research_crosscheck.md` for the full negative-result record.

After freezing that branch, a 5-probe parallel investigation (scientist, codex, critic, and four gap-probe agents) re-evaluated memory-efficient FDTD alternatives from scratch, with JAX-native advantage preservation as a hard constraint. Hybrid FDTD–FEM was excluded (tracked separately in `rfx-adv`).

This note records the findings and the resulting work plan.

## Evaluation of alternatives

Candidates surveyed: Bérenger Huygens subgridding (HSG, TAP 2006), Chilton–Lee provably-stable FE-projection subgridding (TAP 2007), Okoniewski overlapping subgrid (OS-FDTD, TAP 1997), Xiao–Liu enlarged-cell interpolation at PML (AWPL 2007), high-order FDTD(2,4) multi-resolution, JAX-speculative vmap multi-resolution, and completion of the existing NonUniformGrid graded mesh.

Rejected and why:
- **OS-FDTD** — author-confirmed late-time instability; recapitulates the SBP-SAT failure mode class (boundary-interpolation coupling into PML).
- **FDTD(2,4)** — wide-stencil + PML = same structural coupling risk; speculative for rfx JAX architecture.
- **Xiao–Liu** — a correctness patch for nonuniform + CPML interface, not a standalone memory technique. Retained as a potential refinement of the NonUniformGrid path.
- **vmap multi-resolution** — no CEM production track record; `jax.vmap` does not solve the coupling problem.
- **Bérenger HSG** — the author explicitly states in TAP 2006 that the scheme is not naturally stable. The "Switched HSG" patch (Hartley et al., TAP 2022) is empirical, not a formal proof. Useful only if paired with the switching mechanism, which is non-trivial.

Retained:
- **Lead — NonUniformGrid completion.** Existing `rfx/nonuniform.py`, `rfx/runners/nonuniform.py`, `rfx/core/yee.py` nonuniform kernels, and the `_build_nonuniform_grid` path in `rfx/simulation.py` cover the graded-mesh lane on a single uniform-update architecture. CPML already duck-types per-axis cell sizes. The lane is real but partially blocked; unblocking it is a finite 2.5–3 person-week scope.
- **Backup — Chilton–Lee TAP 2007.** Provably energy-conserving and charge-conserving FE-projection subgridding with arbitrary odd refinement ratios and material traversal. Citation: R. A. Chilton and R. Lee, "Conservative and provably stable FDTD subgridding," IEEE Trans. Antennas Propag. 55(9), 2537–2548, 2007, DOI 10.1109/TAP.2007.904092. Full dissertation openly available via OhioLINK ETD. This is distinct from the Bérenger HSG surface-injection scheme. Reserved for the case where disjoint fine regions become necessary (multi-element arrays, via grids, multi-scale PCB).

Dispute resolution (critic verdict, 2026-04-15): the initial scientist claim that NonUniformGrid is "production-ready, full JAX jit/gradient/shard_map" overstated the state. Codex was right that the lane is blocked in several load-bearing ways. Specifically:
1. `rfx/simulation.py:448-450` rejects `boundary='upml'` under nonuniform profiles.
2. `rfx/simulation.py:501-503` rejects ADI under nonuniform profiles.
3. `rfx/runners/distributed_v2.py` has zero nonuniform awareness — multi-GPU lane is uniform-only.
4. `tests/test_nonuniform_api.py:113-179` hard-rejects NTFF, DFT plane probe, TFSF, waveguide ports, and lumped RLC on the nonuniform path.
5. No gradient-through-nonuniform test exists.

## Gap probes — empirical findings

**Gradient.** `jax.grad` through the nonuniform lane works with respect to source amplitude (AD vs finite-difference, 0.11% error at `n_steps=80`, grid 18×18×38). It breaks with respect to `dz_profile` because `make_nonuniform_grid` calls `np.asarray` and `float(np.min(dz_full))` on the traced input before any JAX operation touches it — a Python-level host boundary. Gradient-through-geometry requires refactoring `make_nonuniform_grid` to stay inside the trace. Source/material gradients are safe today; geometry gradients are a separate future scope.

**UPML unblock.** The blocker is `_sigma_profile_1d` computing `d = n_layers * dx` with a single scalar. The fix is to accept a per-cell width array and compute cumulative physical positions for grading. Estimated 3–5 hours. Landmines: `_get_axis_cell_sizes` currently returns boundary scalars, not slices; `cb_ex/dx` at line 213 conflates stencil-spacing and PML-grading `dx` and must be disentangled; uniform x/y fast-path must be preserved for existing CPML tests.

**Distributed + nonuniform.** Tractable with restrictions: shard along x only, grading ratio ≤5:1 globally (shared `dt`, no cross-device sub-cycling), x-axis CPML cells uniform (already true via `make_nonuniform_grid` padding), TFSF remains single-device. This is exactly the constraint set used by Meep, Lumerical FDTD Solutions, and XFdtd. Estimated 7 days: ~3 days for per-axis `inv_dx` slab threading through `update_h_nu`/`update_e_nu` distributed kernels, ~1.5 days for CPML init fix (axis-aware cell size instead of scalar `grid.dx`), ~0.5 day for `run_distributed` duck-typing guard, ~2 days for 2-GPU wave-crossing + CPML reflection graded-mesh tests.

**Cell-count reduction, measured by geometry class.** Analytic cell counting on three representative workloads, anisotropic per-axis minimum resolution with 1.3× grading ratio and 8-layer CPML:

| Geometry | Uniform cells | Graded cells | Ratio |
|---|---|---|---|
| Patch antenna (FR4 0.5 mm substrate + 37.5 mm air, 2.4 GHz) | 8.0 M | 0.43 M | 18.5× |
| Microstrip CPW (0.2 mm strip, 50 mm domain, 10 GHz) | 134 M | 1.26 M | 106× (memory) |
| Cavity with 0.1 mm PEC slot (20 mm cavity, 10 GHz) | 1.6 M | 0.49 M | 3.1× |

For planar thin-substrate structures, the 5–10× plan-note claim is a lower bound. For fine features that span the full domain on any axis, 2–5× is the realistic ceiling. A single-number headline of "5–10×" is defensible across the workload mix but undersells the antenna/microstrip class.

**Huygens citation integrity.** Both the scientist and codex reports conflated Bérenger's Huygens subgridding (TAP 2006, author-acknowledged unstable) with Chilton–Lee's provably-stable FE-projection subgridding (TAP 2007). Neither of the IEEE TAP 56(8) 2008 citations used earlier in this repo holds. For the backup candidate, the correct reference is TAP 55(9) 2007.

## Work plan — new branch

Branch: `nonuniform-completion` (forked from current `main`, not from the frozen SBP-SAT branch).

Ordered by cost-risk, lowest first. Each item closes with an explicit verification step. All work preserves the `subgrid-modified-yee-singlegrid` branch as a frozen negative-result artefact; it is not merged.

1. **UPML + nonuniform coupling.** Remove `rfx/simulation.py:448-450` guard. Rewrite `_sigma_profile_1d` in `rfx/boundaries/upml.py` to accept per-cell width array. Thread full PML cell-width slice through `_axis_sigma_E_H`. Disentangle stencil `dx` from PML-grading `dx` in `cb_ex/dx`. Preserve uniform x/y scalar fast-path. Verify: reflection diagnostic (existing UPML plane-wave test pattern) on a graded z-profile reaches the same -20 dB+ absorption as the uniform UPML baseline. Estimated: 3–5 h work + 1 day test hardening.

2. **Gradient-through-source regression test.** Add `tests/test_nonuniform_gradient.py` with two cases: (a) `jax.grad` w.r.t. source amplitude (expected: works, FD agreement <1%); (b) `jax.grad` w.r.t. `dz_profile` (expected: `TracerArrayConversionError`, marked `xfail` with pointer to the refactor scope). Purpose: lock in the source-side differentiability that was just empirically confirmed, surface the geometry-gradient gap as a known scope item instead of a silent hole. Estimated: 1 day.

3. **Distributed nonuniform kernels.** Implement `_update_h_local_nu` / `_update_e_local_nu` in `rfx/runners/distributed.py` (or a new `distributed_nu.py`) accepting `inv_dx`, `inv_dy`, `inv_dz` slabs. Update `rfx/runners/distributed_v2.py` shard_map `in_specs` to shard `inv_dx` along `P("x")` alongside the field slab. Fix `_init_cpml_distributed` to call `_get_axis_cell_sizes` per face. Add duck-typing guard in `run_distributed` to accept `NonUniformGrid`. Verify: 2-GPU wave-crossing with 3:1 x-axis grading matches single-GPU nonuniform reference to eps-level agreement; 2-GPU CPML reflection stays within 1 dB of single-GPU at the same layer count. Estimated: 7 days.

4. **Capability coverage scoping.** For NTFF / DFT plane probe / TFSF / waveguide ports / lumped RLC on the nonuniform lane, decide per-feature: unblock now, unblock later, or permanently scope-cut. Document the decision matrix. Do not unblock NTFF before the distributed lane is working (the two interact through the radiation-box sampling that depends on cell-size arrays). Estimated: 0.5 day decision + variable implementation.

5. **Optional — differentiable geometry.** Refactor `make_nonuniform_grid` to stay inside the JAX trace (replace `np.asarray` / `float(np.min)` with `jnp` equivalents; make CPML coefficient arrays JAX-traceable). Enables `jax.grad` w.r.t. `dz_profile`. Only worth doing if a shape-optimization workload is planned; not on the critical path for memory efficiency. Estimated: +1 week.

Exit criteria for the branch to be merge-ready: steps 1–3 complete and green on CI; step 4 has a written decision matrix; step 5 is either done or explicitly deferred with a tracking issue.

## Do-not-repeat
- Do not claim NonUniformGrid is "production-ready" without checking `rfx/simulation.py:440-510` guards and `tests/test_nonuniform_api.py` capability list.
- Do not cite "Chilton & Sarris TAP 56(8) 2008" for any stability result. The subgridding stability proof is Chilton & Lee TAP 55(9) 2007. TAP 56(8) 2008 is the Lobatto p-refinement paper.
- Do not cite Bérenger TAP 2006 as a stable Huygens subgrid — the author explicitly does not claim stability.
- Do not put CPML or UPML on the interior interface of any two-block subgrid scheme (lesson from SBP-SAT frozen result).
- Do not treat the existing `docs/research_notes/2026-04-03_nonuniform_mesh_plan.md` cell-reduction numbers as a ceiling — they are a reasonable single-number summary but understate the thin-substrate class.

## Files of record
- Frozen negative result: commit `7758c9d` on `subgrid-modified-yee-singlegrid`; note `docs/research_notes/2026-04-15_sbp_sat_cpml_research_crosscheck.md`.
- This note: `docs/research_notes/2026-04-15_nonuniform_completion_handoff.md`.
- Existing nonuniform surface: `rfx/nonuniform.py`, `rfx/runners/nonuniform.py`, `rfx/core/yee.py` (`update_h_nu`, `update_e_nu`), `rfx/boundaries/cpml.py` (`_get_axis_cell_sizes`), `rfx/simulation.py` (`_build_nonuniform_grid`, guards at 448-450 and 501-503).
- Existing tests: `tests/test_nonuniform_api.py`, `tests/test_nonuniform_convergence.py`, `tests/test_nonuniform_xy.py`.
- Earlier plan: `docs/research_notes/2026-04-03_nonuniform_mesh_plan.md`; integration handoff: `docs/research_notes/2026-04-03_nonuniform_integration_handoff.md`.
