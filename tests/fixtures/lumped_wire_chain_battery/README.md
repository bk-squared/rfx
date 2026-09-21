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
| `adfd` | reverse-mode AD against a float64-loss central finite difference, in the load resistance and in a permittivity scale, with the comparator's ULP span recorded before its verdict and the closed form's own derivative beside both |
| `pilot` | the record-length and drive ladder the battery's `num_periods` and drive were chosen from |
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
