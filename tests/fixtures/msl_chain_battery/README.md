# msl_chain_battery — what is in `fixture.json`

One artifact, one replay test (`tests/oracle/test_msl_chain_battery.py`), one
pre-declaration (`docs/design_notes/msl_chain_battery_predeclaration.md`). The
reduced v2.0 battery for the microstrip family, run against
`Simulation.compute_msl_s_matrix` on a uniform mesh with `mode="laplace"` ports
and the RAW S that `enforce_passivity=False` returns.

The measurement is `scripts/diagnostics/msl_chain_battery_measure.py`; the job
specifications that ran it are in `scripts/vessl_msl_chain_battery/`. The
tolerances are not in this directory: they are the v2.0 bar in
`docs/design_notes/chain_closure_contract.md`, carried in the fixture's `bar`
block so the replay test reads them from the artifact rather than from a literal.

## Top-level keys

| key | what it holds |
|---|---|
| `board` | the declared board, in metres: substrate, trace, stub, margins, band |
| `bar` | the v2.0 numbers the replay test compares against |
| `solves` | one entry per (device, cell size): complex S, the curves, the realized geometry the driver asserted before solving, the verbatim preflight text, every warning, the witnesses (`settling_db`, `reliable`, `sigma_max_excess`, `beta_railed`), wall time and peak memory |
| `ladder` | the three cell sizes side by side: notch frequency, its successive differences and their ratio, dB distance to the finest rung inside and outside the notch core, max column power, median `Re(Z0)` |
| `identity` | the no-op `eps_override` call against the plain call at the coarse rung |
| `adfd` | reverse-mode AD against a float64-loss central finite difference, with the comparator's ULP span recorded before its verdict |
| `plane` | two runs differing only in `n_probe_offset`, their magnitudes and the rotation against `2 beta Delta` |
| `pilot` | the record-length and drive ladder the battery's `num_periods` was chosen from |
| `referee_context` | the committed openEMS / Palace / rfx notch records, on a DIFFERENT board (254 um against this battery's 300 um). Context; nothing is compared against them |

## Three analytic notch references, on purpose

Each notch entry carries `f_notch_analytic_hz` (the declared 600 um trace and
12 mm stub), `f_notch_analytic_realized_hz` (the realized electrical width and
the realized centreline stub length) and `f_notch_analytic_solved_hz` (the
solved sheet size: a PEC sheet edge is solved 0.35 cell beyond its last node,
so the trace is `W + 0.7 dx` and the stub `L_stub + 0.35 dx`). They are
recorded together because which one a quasi-TEM closed form should take is a
measured question this battery does not settle. The replay test reads the
declared one, which is the reference the pre-declaration names.

## No verdict sentence lives here

The fixture holds measurements and the pre-declared thresholds beside them,
plus a boolean per comparison. What the numbers mean is written where a person
signs it, not in the artifact.
