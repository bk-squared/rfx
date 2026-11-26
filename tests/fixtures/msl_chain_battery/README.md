# msl_chain_battery — what is in `fixture.json`

One artifact and one pre-declaration
(`docs/design_notes/msl_chain_battery_predeclaration.md`, with its addenda). The
reduced v2.0 battery for the microstrip family, run against
`Simulation.compute_msl_s_matrix` with `mode="laplace"` ports and the RAW S that
`enforce_passivity=False` returns, referred to the feed planes.

The board is solved as drawn. At each cell size (100, 50 and 25 µm) the x and y
cells come from `rfx.mesh_edges.edge_aware_profiles` (non-uniform, graded) under
`snap="strict"`, and the driver asserts, before any step, that the trace, the
stub's width and its open end are solved within 1 nm of the drawing, that the
stub joins the trace over its whole width, and that the port planes and probe
ladders are the declared ones. The thru is built on the notch's cells.

The measurement is `scripts/diagnostics/msl_chain_battery_measure.py`: one stage
per cluster job, each writing its own JSON, and `--assemble` joining them into
this file. Every solve record has the same drive and record length (40 source
periods); the assembler refuses records that differ. Each entry's `provenance`
names the commit and run it was measured at.

The only test that reads this file is the 100 µm drift lock,
`tests/locks/test_msl_chain_battery_drift.py`, which solves that mesh again and
holds it to the stored S. No CI test judges the ladder, the 50 and 25 µm
entries or the 25 µm column power. The tolerances are the v2.0 bar in
`docs/design_notes/chain_closure_contract.md`, carried in the fixture's `bar`
block.

## Top-level keys

| key | what it holds |
|---|---|
| `board` | the declared board, in metres: substrate, trace, stub, margins, band |
| `bar` | the v2.0 numbers each comparison is judged against |
| `battery_drive` | the drive and record length every solve shares, and where each was chosen from |
| `solves` | one entry per (device, cell size): complex S, the curves, the realized geometry the driver asserted before solving, the verbatim preflight text, every warning, the per-bin settling and reliability (`settling_db`, `reliable`, `sigma_max_excess`, `beta_railed`), wall time and peak memory |
| `ladder` | the three cell sizes side by side: notch frequency, its successive differences and their ratio, dB distance to the finest rung inside and outside the notch core, the thru's reflection floor, max column power, median `Re(Z0)`, and per-rung booleans against the bar |
| `identity` | the no-op `eps_override` call against the plain call at 100 µm |
| `adfd` | reverse-mode AD against a float64-loss central finite difference at 100 µm, with the comparator's ULP span recorded before its verdict |
| `plane` | two runs at 50 µm differing only in `n_probe_offset`: their magnitudes, and the rotation of `angle(S11)` and `angle(S21)` against 0 (S is at the feed planes, which the probe ladder does not move), with the probe-plane value `2 beta Delta` beside it as context and `|S21|` at each largest rotation's bin |
| `pilot` | the 20-period drive comparison the battery's drive was chosen from |
| `referee_context` | committed openEMS / Palace / rfx notch records on a DIFFERENT board (254 µm against this battery's 300 µm substrate). Context; nothing is compared against them |

## Analytic notch references

Each notch entry carries the quarter-wave closed form at the declared size
(`f_notch_analytic_hz`), at the realized size (`f_notch_analytic_realized_hz`),
at the solved sheet size (`f_notch_analytic_solved_hz`), and with the stub
length taken from the trace's centreline (`f_notch_analytic_centreline_hz`). On
these boards the solved size is the drawn size, so the first three agree. None
includes the open-end extension, so none is the physical notch; they place the
notch and do not grade it.

## No verdict sentence lives here

The fixture holds measurements and the pre-declared thresholds beside them,
plus a boolean per comparison. What the numbers mean is written where a person
signs it, not in the artifact.
