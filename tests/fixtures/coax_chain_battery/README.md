# coax_chain_battery — what is in `fixture.json`

One artifact, one replay test (`tests/oracle/test_coax_chain_battery.py`), one
pre-declaration (`docs/design_notes/coax_chain_battery_predeclaration.md`). The
reduced v2.0 battery for the coaxial family, run against
`Simulation.compute_coaxial_two_port` (dielectric bead, thru) and
`Simulation.compute_coaxial_line_reflection` (short, open, 25 and 100 ohm
loads), both through the `eps_scale` design channel.

The measurement is `scripts/diagnostics/coax_chain_battery_measure.py`; the job
specifications that ran it are in `scripts/vessl_coax_chain_battery/`. The
tolerances are not in this directory: they are the v2.0 bar in
`docs/design_notes/chain_closure_contract.md`, carried in the fixture's `bar`
block so the replay test reads them from the artifact rather than from a
literal.

## Top-level keys

| key | what it holds |
|---|---|
| `line` | the radii, the PTFE fill, the analytic `Z_TEM`, both domains, the bead, the three rungs |
| `bar` | the v2.0 numbers the replay test compares against, plus the deep-null level |
| `solves` | one entry per (DUT, rung): complex S, the curves, the realized line the driver asserted before solving, the layout cross-check, the verbatim preflight text, every warning, the residual witnesses, wall time and peak memory |
| `record_length_invariance` | the claims rung run again with the record doubled — the contract's substitute where the lane emits no energy witness |
| `ladder` | the three rungs side by side: reflection-zero frequency, successive differences and their ratio, dB distance to the finest rung inside and outside the zero's core, max column power |
| `identity` | the no-op `eps_scale` call against the untraced call, and the bead in two containers |
| `adfd` | reverse-mode AD against a float64-loss central finite difference, per lane, with the comparator's ULP span recorded before its verdict |
| `plane` | two runs differing only in the DUT's axial position, their magnitudes and the rotation against `2 beta Delta` |
| `pilot` | the record-length ladder, the drive check and the bead-mask control the battery's settings were chosen from |

## The rungs are named by annulus cells, not by a cell size

`dx = (outer_radius - pin_radius) / rung`, so `rung` 4, 6 and 9 realize exactly
4, 6 and 9 cells across the annulus — the dimension the mesh has to resolve, and
the one the extractor's own `status` keys off (`under_resolved` below ~3.5
cells). The committed two-point witness sits at 3.79 and 5.68 cells; this ladder
brackets it with a ratio of 1.5.

## Three places the API decided, not the pre-declaration

- **The line is PTFE-filled**, not air: `stamp_coaxial_line` stamps
  `PTFE_EPS_R`. The analytic referee therefore uses the fill's own `beta`. The
  bead's `eps_scale = 4` still makes the section's impedance exactly `Z_TEM / 2`
  and its phase constant exactly `2 beta_line`, so `Gamma = -1/3` either way.
- **`compute_coaxial_two_port` has no `reference_plane_axial_index_offset`** —
  that keyword belongs to the deprecated `compute_coaxial_s_matrix`. Its
  reference planes ARE its feed planes and are not settable, so the plane stage
  translates the DUT instead: the grid, the probes, the drive and the step count
  are byte-identical between the two arms, and each port's electrical distance
  to the bead changes by one `Delta` in opposite directions.
- **The ring-down witness is skipped on the `eps_scale` path** (it needs a
  concrete time series) and the one-port result carries no `settling_db` field
  at all. Every record says so in its `settling` block, and the claims rung is
  run a second time with the record doubled so the contract's one admissible
  substitute is measured rather than assumed.

## The rungs do not realize one line

`stamp_coaxial_line` makes the outer conductor ONE CELL thick, so its inner
radius is `b - dx` and moves with the mesh: the dielectric annulus runs to
1700 um at 4 annulus cells and to 1897 um at 9, and the characteristic
impedance that implies goes 40.74 -> 43.53 -> 45.29 ohm against the declared
48.59. The pin rasterizes to 10, 21 and 52 cells and its realized radius is not
monotone in dx. `ladder.<dut>.realized_cross_section` carries all of it per
rung, because a ladder whose rungs refine the geometry as well as the mesh is
not the convergence study the contract's dx-ladder guard asks for.

## Two witnesses of the same line, in `line_witnesses`

The resistive loads measure the characteristic impedance
(`z0_numerical_ohm = R (1 - Gamma) / (1 + Gamma)`); the matrix-pencil fit
measures the phase constant, and `eps_eff = (beta c / omega)^2` is the
permittivity a uniform TEM line would need to propagate at it. On a uniform
lossless line both follow from one `L'` and `C'`, so they are not independent
witnesses of each other, and the block records them side by side rather than
reconciling them.

## No verdict sentence lives here

The fixture holds measurements and the pre-declared thresholds beside them, plus
a boolean per comparison. What the numbers mean is written where a person signs
it, not in the artifact.
