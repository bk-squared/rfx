# Priorities after v1.8 — rasterization views, preflight and CST lessons (plan, 2026-09-05)

> **SUPERSEDED IN PART by #931 (the lattice ownership contract), 2026-09-07.**
> Kept as dated history and deliberately NOT rewritten.
> Item 1 of this plan ("rasterization views") became "define and enforce the ownership contract" in #931; visualization reads that contract
> instead of deriving it again. The one-plane / two_plane A/B verdict here compares two realizations, neither of which exists in 2.0:
> conductors are declared as volume, sheet or wire.
> The current rule is
> `docs/design_notes/20260906_plan_realign_lattice_ownership.md` §1, and for
> users `docs/public/guide/materials-geometry.mdx` ("How conductors land on
> the lattice").

After v1.8 closes, prioritize (1) mesh rasterization visualization, (2) simulation preflight, and (3) lessons from CST
cross-validation before #819/#888/v1.9. Except for #868, all defects found in this work were in
the **measurement setup** (record windows, absorber placement, witnesses), rather than physics.
These should have been visible before spending GPU time.

This is a plan. Its numbers come from measurements in this investigation; the code is not implemented yet.

## 0. Existing capabilities (feature-discovery rule: grep first)

- `rfx/visualize.py::plot_rasterized_slice(sim, axis, position)` draws the cells used by the solve.
  It overlays conductors on ε_r with `pcolormesh` at actual cell boundaries. `show_declared=True` overlays declared geometry.
- `rfx/visualize.py::plot_stack_profile(sim, axis, at)` shows declared intervals beside realized cells along a column.
- `rfx/api/_preflight.py` has 52 `_validate_*` checks. Rasterization checks include `_graded_box_rasterization`,
  `_congruent_rasterization_parity`, `_thin_conductor_graded_node`, and the #729 declared-versus-realized z-width notice.
- `rfx/geometry/rasterize_grid.py` performs rasterization itself.

Thus (1) does not mean creating plotting functions; those exist. **What is missing is a single view of the geometry implicated
in this campaign's defects: port planes, reference planes, absorber pads and the time axis (record window versus round trip)**,
produced automatically at preflight.

## 1. Rasterization views — four gaps

These properties became visible only after manual calculations in this campaign. Each could have appeared in a single
preflight plot.

| Gap | How it was found | What the plot must show |
|---|---|---|
| Mirror symmetry of port/reference/probe indices | Static audit script: E-plane index sum is `nx` (`nx−1` for covariance), the basis of F1 | Axial section with port/reference/probe/TFSF planes and indices; annotate left/right sums |
| Pad thickness relative to the domain and the band λ_g used to size it | Reading `cpml_layers` and `lam_g` from JSON | Pad scale in λ_g(f_low); title states the sizing f |
| **Whether the record window contains the far round trip (T/τ)** | Manual ordering of 13 cases, all of #894 | Time bars for source duration, internal transit, far round trip τ and record T. **Red when T < τ.** Use v_g at the lowest band bin |
| Margin for the lowest bin in its own box | R3 converged only in the R2 box | The same time bars for both lowest and highest band bins |

Implementation: `show_ports=True` for `plot_rasterized_slice` (planes and index sums), plus a new `plot_time_budget(sim)`
for the time bars above. Both only draw: preflight computes the quantities and the plots read them.

## 2. Preflight extensions — four checks from these measurements

Retain `feedback_preflight_input_fidelity_only`: preflight reports declared versus realized quantities **in input units**,
without predicting result effects. All four checks below use input units.

1. **`_validate_cfg_record_vs_far_boundary`** — per port, τ_far = 2·(port→far outer wall)/v_g(f_min),
   T = record length. Warn at T/τ < 3, with an error-level warning at < 1. Report T, τ, their ratio and the `num_periods` needed for T/τ ≥ 3.
   This captures the mechanism of #894. Measurements required 3 for a/18, 5 for a/36,
   and at least 8 for a/72. Keep the threshold at 3 regardless of rung, with a note that finer grids need more.
2. **`_validate_cfg_layout_from_band_low_edge`** — notify when domain/pad margins are sized using λ_g at the band's lowest frequency:
   "This layout serves the f_min bin worst; size below f_min." Measurements:
   R3 failed to converge in its own box through T/τ=16, versus +3 % in the next lower band's box.
3. **`_validate_cfg_settling_witness_present`** — if `sim.run()` results have no settling witness (#885),
   report the absence of a truncation guard for NTFF/field-DFT/probe runs. CST lesson 1. Also state that the witness cannot detect
   an underflowed record (#869) or a round trip outside the window (#882).
4. **`_validate_cfg_port_index_mirror_covariance`** — move the static audit in §1 into preflight. The E-plane sum
   `nx` is a known shipped defect, so record it rather than raising an error. This check catches the #868/F1 class without FDTD.

One **post-solve** check: **S21 phase residual against −βL**. This observable carries discretization independently of absorber, record and precision
(order 2.00 over seven bands), answering whether ports explain an unexpected reflection number. Include it as a port
S-parameter result field and a result-banner line, outside preflight.

## 3. CST cross-validation lessons — three instruments, two physics questions

Keep the order in `20260903_cst_crossval_state.md` §5: instrument repairs → discriminating tests → compute.

**Instruments first**
- **#885**: a settling witness in `sim.run()`, the same item as §2-3. Add fields to `Result`; extend what the S-parameter
  path already computes to field-DFT/NTFF/probe paths. A 3-hour field A/B became
  inconclusive because of NaN.
- **NaN guard**: "settling > −40 means INCONCLUSIVE" misfired on a NaN comparison. When a witness is absent,
  report "missing" instead of NaN.
- **Degenerate excess-occupancy metric**: the denominator is 0 when U_ON ≈ U_OFF. Use a metric without layer cancellation.
- Remove the hardcoded plot-title "settling −178 dB" and read it from data.

**Physics after instrumentation**
- **Additive pattern-error floor**: a broad coherent component at −8.5~−17.5 dB relative to the peak appears only
  at low-level angles. Vary absorber thickness: a falling floor implicates absorber reflection; an unchanged floor implicates the field itself.
  **Apply lesson #894**: that absorber is 0.075 λ (0.8 mm) at 28 GHz; before varying thickness, check whether the NTFF record
  window contains the far round trip (T/τ). What looked like a thin-absorber problem in the waveguide was a record-window problem.
  The §2-1 checker supplies half of this discrimination without another solve.
- **one-plane confirmed**: S11 correlation +0.83~0.91 versus +0.28~0.41. Resonance position and mean agree, but depth differs by 12.5 dB,
  indicating Q/loss. This establishes the realization, not reproduced matching. Next: loss A/B.

**Distributed compute** comes after instruments stop producing NaN. Additional arms should be
**sharded rather than submitted serially** (the `waveguide_vi_envelope_shard.py` pattern).

## 4. Sequence and size

| # | Item | Size | Dependency |
|---|---|---|---|
| 1 | Close the v1.8 waveguide campaign (x64 decision + 0-derivative skip) | PR 1 | Precision and zero-derivative policy |
| 2 | Preflight §2-1 (T/τ) + §2-4 (mirror audit): calculation only, no plots | PR 1 | — |
| 3 | `plot_time_budget` + `show_ports`: draw the calculations in §2 | PR 1 | 2 |
| 4 | #885 + NaN guard + occupancy metric (three CST instruments) | PR 1 | — |
| 5 | Preflight §2-2, §2-3 + S21 phase-residual banner | PR 1 | 4 |
| 6 | CST discriminating test (T/τ check → absorber thickness), sharded submission | Measurement 1 | 2, 4 |
| 7 | one-plane loss A/B | Measurement 1 | 6 |

2 and 4 are independent and can run in parallel; 3 and 5 follow respectively. Each PR follows these rules: no measurement
without predeclaration; checkers report only input units; plots draw the calculated quantities without recomputing them.

## 5. Falsifier this plan must answer

The claim is that these checks would have exposed this campaign's defects before GPU execution. Verification: implement §2-1 and
§2-4, then **run them on the original sweep case list (K=3.0, interior rule)** and check for the T/τ ≈ 1.4 warning and
the E-plane sum `nx` record. If they do not appear, the checks are wrong.
