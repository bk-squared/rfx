# T2 → owner of `scripts/diagnostics/probe_fed_msl_openems_referee.py`

The #498 openEMS referee redraws, in openEMS, the board that
`tests/unit/sparams/test_mixed_port_sparam.py::_base_sim` **realizes** — not
the one it declares. That is the right discipline: an external comparator that
draws the declared board is comparing two different structures and calling the
difference physics. The cost of it is that `RFX_REALIZED_RECORD` is a COPY of a
measurement, and this branch moved the measurement.

Nothing went red when it did. The gate that would have caught it is now in
`tests/unit/sparams/test_probe_fed_msl_referee_contract.py::test_referee_record_still_describes_the_fixture_it_names`
— build-time, no openEMS, no solve. It is `xfail(strict=True)` until this file
is updated; when it is, the test goes green and the marker comes off.

## What moved, measured

The fixture is on-lattice now (`dx = h_sub/3`, so the laminate face is a node
line) and its 35 µm foil is declared as a **sheet** — a zero-thickness Box on
that face, realized as one wall plane with the normal Ez live. Before, it was a
one-cell PEC Box at `dx = 80 µm` on a mesh that bisects the laminate face.

| quantity | record (pre-#931) | live (post-#931) |
|---|---|---|
| `dx` | 80.000 µm | 84.667 µm (= h_sub/3) |
| grid shape | (117, 55, 19) | (112, 53, 18) |
| pads | x 8/8, y 8/8, z 0/8 | unchanged |
| realized `h_sub` | 4 cells = 320 µm (**+26 %**) | 3 cells = 254 µm (**exact**) |
| conductor plane `k` | 4 | 3 |
| trace, edge-span reading | 6 cells = 480 µm | 6 cells = 508 µm |
| trace, node-span reading | 7 cells = 560 µm | 7 cells = 592.67 µm |
| `rfx_node_index(0.0)` | 8 | 8 |
| `rfx_node_index(2.00 mm)` | 33 | 32 |
| `rfx_node_index(5.50 mm)` | 77 | 73 |
| `rfx_node_index(8.00 mm − ε)` | 108 | 102 |
| trace y-centre | node 27 = 1.52 mm | node 26 = 1.524 mm |

The ±1-cell ambiguity in the trace width (edge span vs node span) is the same
one the record already carries as a declared systematic; both numbers moved,
so both entries need replacing, not one.

## The anchor term in budget B

The record states, up front and correctly, that rfx anchors its MSL port to the
Hammerstad–Jensen Z0 of the **declared** board, 47.89479996289313 Ω
(W = 600 µm, h = 254 µm, ε_r = 3.66), while the board it actually solves has a
different Z0, and that a |S22| comparison inside budget B inherits the
difference. That paragraph stays; its numbers change:

| board | W, h | Z0 | ε_eff |
|---|---|---|---|
| declared (the anchor) | 600, 254 µm | 47.895 Ω | 2.86939 |
| realized, pre-#931 low | 480, 320 µm | 62.652 Ω | 2.77333 |
| realized, pre-#931 high | 560, 320 µm | 57.463 Ω | 2.80448 |
| realized, post-#931 low | 508, 254 µm | 53.106 Ω | 2.83269 |
| realized, post-#931 high | 592.67, 254 µm | 48.271 Ω | 2.86662 |

(computed with `rfx.sources.msl_eigenmode.hammerstad_jensen_z0_eps_eff`.)

So the realized-vs-anchor systematic goes from **+20 % … +31 %** to
**+0.8 % … +10.9 %**, because the realized substrate stopped being a quarter
too thick. Still REPORTED, NEVER GATED; it still must not replace the analytic
HJ anchor anywhere in shipped code (predeclaration §10). The point is only that
it moved and the referee has not been told.

## The edits

1. `RFX_REALIZED_RECORD["declared"]["dx_m"]`: `80e-6` → `254e-6 / 3.0`. Write it
   as the expression, not the decimal — it is a derived quantity (`h_sub/n`), and
   a decimal is the next copy to rot.
2. `RFX_REALIZED_RECORD["realized"]`: the grid shape, `h_sub_m`, the conductor
   plane index, both trace-width readings and the `y_c` snap, from the table
   above.
3. `rfx_node_index`'s `dx_m` default: `80e-6` → the same expression, and the
   worked examples in its docstring re-derived (0 → 8, 2.00 mm → 32,
   5.50 mm → 73, 8.00 mm − ε → 102). The function is `pad + round(x/dx)` and is
   correct; only its default and its quoted examples are stale.
4. The module docstring's "REALIZED …" block and the budget-B anchor paragraph,
   from the two tables above.
5. Add one line to the record saying the conductor is a **sheet** — one wall
   plane, normal Ez live, owning no cell. The openEMS side already draws the
   trace as a zero-thickness `AddBox` on the metal property, so no geometry
   change is needed there; the record just has to say which object class rfx is
   realizing, because "one plane" is now a declaration and not an artefact of
   the raster rule.

Not done here: the referee's own Stage-1 reproduce-gate legs are an
external-solver run (openEMS is not installed on this pod) and no number in
this file's shipped tests depends on the values above — the contract tests are
arithmetic and structure only. Nothing needs re-solving to make the edits; the
Stage-1/Stage-2 legs need re-running before the referee is next used as a
comparator, and that is an openEMS job, not a VESSL rfx job.

---

## ATTEMPTED AND BACKED OUT 2026-09-07 (phase 2b ingest) — the edit list above is not sufficient

Every number in the tables above was re-measured on the branch and every one
checks out (`dx = 84.667 µm`, shape `(112, 53, 18)`, one wall plane at `k = 3`,
`h_sub` realized 254 µm exactly, trace Ey edge span y 23..28 = 6 cells =
508.00 µm, Ex node span y 23..29 = 7 nodes = 592.67 µm, node indices
0 → 8, 1.44 mm → 25, 2.00 mm → 32, 2.80 mm → 41, 3.60 mm → 51, 4.72 mm → 64,
5.50 mm → 73, 8.00 mm − ε → 102, HJ Z0 53.106 / 48.271 Ω against the 47.895 Ω
anchor). The five edits were applied and then reverted, because applying them
turned 1 red test into 13 and the cause is not a typo:

**The referee's planes of record go off-lattice at `h_sub/3`.** Its own
`plane_on_grid` self-check asserts that every plane it places — 1.44, 1.76,
2.00, 2.24, 2.80, 3.60, 4.08, 4.40, 4.72 mm — is an exact multiple of rfx's
`dx`. All nine are exact multiples of 80 µm and NONE is a multiple of
84.667 µm. The whole Stage-2 comparator is built on those coordinates: the
lumped feed plane, the MSL port's start plane, and the `MeasPlaneShift`
stencil that has to land on-grid at dx = 50 µm for the de-embedding to be a
measured no-op.

So the referee needs its Stage-2 mesh **re-planned**, not its record re-typed:
either the plane list moves onto the new lattice (which changes what the
comparator measures and needs the Stage-1/Stage-2 legs re-run against the new
board), or the fixture's `dx` choice is revisited for this comparator. Both are
the referee owner's call plus an openEMS run, and openEMS is not installed on
this pod.

Also found while doing it, and folded into the edit list above for whoever
takes it: the two width keys were named the wrong way round.
`w_trace_node_span_m` held `trace_y_hi − trace_y_lo` (the span between the
extreme node coordinates = the Ey EDGE count × dx, 480 µm pre-#931), and
`w_trace_cell_span_m` held the NODE count × dx (560 µm). The gate test
`test_referee_record_still_describes_the_fixture_it_names` asks for
`w_trace_edge_span_m`, so the rename is part of the edit, and the script's own
`realized_w_is_node_span` self-check must be renamed with it.

The gate test therefore keeps its `xfail(strict=True)`, with the blocker above
named in the marker.
