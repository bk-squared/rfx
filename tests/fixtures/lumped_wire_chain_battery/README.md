# lumped_wire_chain_battery — what is in this directory

One artifact (`fixture.json`), one replay test
(`tests/oracle/test_lumped_wire_chain_battery.py`), one pre-declaration
(`docs/design_notes/lumped_wire_chain_battery_predeclaration.md`). The reduced
v2.0 battery for the lumped and wire port family, run against the
differentiable S11 of `Simulation.forward(port_s11_freqs=...)`. The S matrix of
`run(compute_s_params=True)` is a numpy post-process and is not in the v2.0
chain; the driver never calls it.

The measurement is
`scripts/diagnostics/lumped_wire_chain_battery_measure.py`; the job
specifications that ran it are in `scripts/vessl_lumped_wire_chain_battery/`.
The tolerances are not in this directory: they are the v2.0 bar in
`docs/design_notes/chain_closure_contract.md`, carried in the fixture's `bar`
block so the replay test reads them from the artifact rather than from a
literal.

## The fixture is a line, not a board

A parallel-plate channel one cell wide between PEC plates and magnetic side
walls carries an exact TEM wave, so the port's input impedance is the closed
form of a terminated line and the referee is arithmetic rather than another
solver. The plate gap is one cell for the lumped port and four for the wire
port, and the width is one cell, so `Zc = eta0 * h / w` is `eta0` and `4 eta0`
respectively and does not move with the cell size — only the line's length in
cells does. The port's reference impedance is that same `Zc`, which collapses
the closed form to `S11 = Gamma_L * exp(-2 j beta L)`: the magnitude is
`|Gamma_L|` at every bin, and all the geometry lives in the phase.

## Top-level keys

| key | what it holds |
|---|---|
| `channel` | the declared channel: length, gap and width in cells, `Zc` per port kind, the band, the rungs, the loads |
| `bar` | the v2.0 numbers the replay test compares against, plus this family's passivity bound and the matched control's floor |
| `deviations` | every place the realized fixture differs from the pre-declaration, with what rfx said and what the difference costs |
| `solves` | one entry per (port kind, DUT, cell size): complex S11, the curves, the grid the driver asserted before solving, the port specification the solve returned, the verbatim preflight text, every warning, the record-doubling witness, wall time and peak memory |
| `ladder` | the three cell sizes side by side per (port kind, DUT): dB distance to the finest rung, phase-crossing distance, the matched floor, successive differences and their ratio, and the coarsest rung that sits inside the bar |
| `identity` | the no-op `eps_override` call against the plain call, per port kind |
| `adfd` | reverse-mode AD against a float64-loss central finite difference, in the load resistance and in a permittivity scale. One entry per (port kind, leg, cell size), with `predeclared_rung` marking the one the pre-declaration named. Each case records the comparator's ULP span before its verdict, the closed form's own derivative, the three pairwise distances, and where on the curve the derivative was taken |
| `pilot` | the record-length and drive ladder the battery's `num_periods` and drive were chosen from |
| `port_kind_ab` | one cell of one line declared two ways, in front of three loads whose reflection is an exact number. Lifted from the record `scripts/diagnostics/lumped_port_known_load_line.py` writes, not recomputed here |
| `not_in_chain_observations` | what `run(compute_s_params=True)` returns on the same builds. Not in the v2.0 chain, carries no bar and no pass/fail |
| `degenerate_objective_evidence` | the legs whose objective has no derivative, and what the ULP-span floor did on them |
| `openems_context` | whether a recorded openEMS lumped comparison was readable, and what it is |

## Three analytic references per solve, on purpose

Every solve carries the same closed form evaluated three ways:
`analytic_realized_length` (the line length the lattice actually built),
`analytic_declared_length` (the nominal 30 mm) and `analytic_lattice_beta` (the
realized length with the Yee lattice's own phase constant rather than the
continuum one). They are recorded together because a short is pinned on an E
node and an open on an H half-node, so the two cannot both land an integer
number of cells from the port, and because the lattice's numerical dispersion at
the coarsest rung is of the same size as the difference. The replay test reads
the realized-length reference, which is the one the `deviations` block names.

## Both port kinds are here, and one of them is red

The battery runs the lumped port and the wire port through the same assertions
on the same channel. The wire leg passes them; the lumped leg does not, and its
records are here because that is the measurement, not despite it. Nothing about
the lumped leg is loosened, xfailed or skipped: the replay test asserts the
pre-declared bar on both and reports what it finds.

`port_kind_ab` is the sharpest form of it — one cell, three loads whose
reflection is an exact number, and the port declared two ways with `extent=dx`
the only difference. Its producer is
`scripts/diagnostics/lumped_port_known_load_line.py`, which takes no arguments
and rewrites its own JSON, so a change to the lumped lane can be measured
against that record rather than against a description of it.

## How the gradient legs are read

Criterion 3a is carried by AD against a float64 finite difference, asserted at
5 % on every non-degenerate leg at every cell size; the closed-form derivative
is a third witness of a different object — it differentiates the continuum line
where AD and FD differentiate the lattice one — so it is held only to shrinking
with the mesh at a first-order rate, not to a bar. Band-mean `|S11|^2` on the
short-terminated line is identically 1 and therefore has no derivative to
compare, so those three records are marked `degenerate_on_this_dut` and carry no
comparison at all. Both decisions, and the numbers behind them, are in the
addendum to
[`docs/design_notes/lumped_wire_chain_battery_predeclaration.md`](../../../docs/design_notes/lumped_wire_chain_battery_predeclaration.md).

## Every AD leg carries two closed forms

One on the length that was DRAWN and one on the line's own electrical length,
fitted from that rung's short-circuit phase. The fit is a least-squares
straight line through `unwrap(angle(S11))` against frequency, unweighted, over
the whole band, with `L = -slope * c / (4 pi sqrt(eps_r))`; its residual ships
beside it, because the lattice's dispersion is not linear in frequency and a
straight line is an assumption about the record. Which of the two lengths a
derivative should be compared against is a question this artifact does not
settle, so it stores both and settles nothing.

## Two things the artifact says about its own instruments

The `adfd` block carries `what_the_ulp_span_does_not_say`. The span is
`|f_plus - f_minus|` in ULPs of the loss, so it answers whether the two LOSS
values are resolved from each other, not whether the DERIVATIVE is. An objective
whose true derivative is zero gives two losses millions of ULPs apart whose
difference is round-off, and the span passes it. Read each case's `closed_form`
gradient and the loss beside it before reading its `rel_err`.

Every reflecting solve carries `phase.angle_slope_rad_per_hz`. The declared
phase test is the crossing frequencies, and those cannot see a conjugated S:
`angle = phi - 2 beta L` and `angle = phi + 2 beta L` cross multiples of pi at
the same frequencies. The slope can. It is recorded as a fact with no threshold
— a second criterion is the PI's to declare, not the driver's to invent.

## The recorded openEMS comparison is context

`lumped_openems_pec_box_sweep_comparison.json` is a three-case openEMS
comparison of rfx's lumped port on a two-port 50 ohm PEC box at 0.8–1.8 GHz,
produced by `scripts/diagnostics/build_lumped_openems_sweep_comparison.py`. It
is a different fixture from this battery's parallel-plate line and nothing here
is compared against it. It is committed because the pre-declaration names it as
context for stage 3d and because until now it existed only in an untracked
runtime directory. Its `_committed_as` block records where it came from, the
two path edits made on commit, and what it does not carry — no rfx commit, no
openEMS version, and none of the S arrays, whose `.npz` files were never
committed.

## No verdict sentence lives here

The fixture holds measurements and the pre-declared thresholds beside them,
plus a boolean per comparison. What the numbers mean is written where a person
signs it, not in the artifact.
