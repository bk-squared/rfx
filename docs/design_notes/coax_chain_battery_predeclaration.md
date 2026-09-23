# Coaxial chain battery — pre-declaration (reduced v2.0 form)

Status: written BEFORE the first battery solve. Contract:
[`chain_closure_contract.md`](chain_closure_contract.md), section "The v2.0 battery for
lumped/wire, MSL and coax". The verdict numbers are that section's, with the PI's 2026-09-21
ruling on deep nulls (below); this note fixes the line, the rungs, the DUTs, the referee and what
each stage records.

Lanes under test: `Simulation.compute_coaxial_two_port(...)` and
`Simulation.compute_coaxial_line_reflection(...)`, differentiable channel `eps_scale`. Both
construct the complete line from the registered `add_coaxial_port(...)`; they refuse registered
geometry, so every DUT here is an `eps_scale` ARRAY (a concrete one for the measured S, a traced
one for the gradient). `compute_coaxial_s_matrix` is deprecated and outside the battery.

## Line

The repository's existing two-port fixture line, unchanged (the one
`tests/unit/sparams/test_coax_two_port_smatrix.py::_sim` builds): domain 8 x 8 x 60 mm, one
`add_coaxial_port((4 mm, 4 mm, 20 mm), face="top", pin_length=5 mm)`, CPML, air-filled line. The
radii, the analytic `Z_TEM` and the analytic `beta = omega / c` are read from the port the API
builds and stored; nothing is re-declared here that the API owns.

Band: 4 - 12 GHz, 81 points (the band the committed one-port envelope covers). Drive: the
fixture's own pulse unless a pilot shows unreliable bins at a band edge, in which case the pulse is
widened and the pilot recorded (as in the microstrip battery).

## Rungs

Three cell sizes chosen so the annulus (outer minus inner radius) holds about 4, 6 and 9 cells —
a ratio-1.5 ladder, because the annulus is what the mesh has to resolve and the committed two-point
witness already sits at 3.79 and 5.68 cells. The finest is the claims rung. The realized annulus
cell count, the realized radii and the realized bead length in cells are asserted and stored per
rung. The recommended cell size the support matrix will state is "annulus cells >= N", N being
the coarsest rung from which every compared quantity stays inside the bar against the finest.

## DUTs

Two-port:
1. **Dielectric bead** (reflecting): `eps_scale = 4.0` over the full cross-section for an axial
   length of 6 mm centred between the two probe arrays, 1.0 elsewhere. Analytic referee: a lossless
   TEM section of impedance `Z_TEM / 2` and electrical length `2 beta L` between two `Z_TEM`
   lines, `S11 = Gamma (1 - e^{-2j theta}) / (1 - Gamma^2 e^{-2j theta})`, `S21` likewise, with
   `Gamma = -1/3`. It has reflection zeros in band (at `theta = n pi`): those are deep nulls and
   fall under the ruling below.
2. **Thru** (control): `eps_scale = None`.

One-port (exists already, re-measured on the same three rungs): PEC short, open, 25 ohm and
100 ohm resistive loads.

## Deep nulls (PI ruling, 2026-09-21)

A quantity that is by construction near zero is not compared in dB from rung to rung. The thru's
|S11| is held to an upper bound, -20 dB at every bin of every rung, and recorded as the line's own
reflection floor. The bead's |S11| is compared within 2 dB only OUTSIDE the cores of its reflection
zeros (bins where the analytic |S11| is below -20 dB); inside a core the verdict is the zero's
FREQUENCY against 1 %, the depth is recorded without a comparison.

## Stages and what each records

| stage | rungs | DUTs | records |
|---|---|---|---|
| S — solve | all | bead, thru; one-port short / open / 25 / 100 ohm | complex S(f) (or Gamma), `settling_db`, `cond_a`, `recurrence_residual`, `fit_residual`, annulus cells, measured `beta` and `Z0`, preflight text, warnings, wall time, peak memory |
| 2 — physics gates | finest (others reported) | bead | max column power <= 1.02; `max_f |S21 - S12| / max|S| <= 0.02`; power closure `1 - |S11|^2 - |S21|^2` as a curve (lossless: reported, bounded by the column-power gate); settling <= -40 dB |
| 1(2) — forward identity | coarsest | bead | complex S with `eps_scale` passed as a traced-compatible `jnp` array equal to the concrete bead, against the concrete call: `rtol=1e-5, atol=1e-7` on COMPLEX S. (The committed identity tests compare magnitudes at 1e-3; this is the contract's form.) |
| 3a — AD vs FD | coarsest | bead (two-port), short (one-port) | theta = the bead's permittivity ratio (two-port) / the mid-line slab's `deps` (one-port). Float32 AD against a float64-loss central FD with the ULP-span assert evaluated first; objectives: band-mean `|S21|^2`, band-mean `|S11|^2`, and `Re(S21 S11*)` at the bin of largest |S11| |
| 3b — plane invariance | middle | bead | two runs differing only in `reference_plane_axial_index_offset` (by 4 cells, both ports): `| |S| - |S|' |`, rotation of `angle(S11)` against `2 beta Delta` and of `angle(S21)` against `beta (Delta_1 + Delta_2)` with the analytic `beta`; both signs stored |
| 3c — ladder | all | bead, thru, one-port loads | |S11|, |S21|, `angle(S21)` curves and the reflection-zero frequencies per rung; successive differences and their ratio; first- and second-order extrapolated limits labelled as arithmetic |
| 3d — referee | finest | bead; one-port loads | the analytic TEM expressions above and `(R - Z0)/(R + Z0)`; the committed Meep and openEMS records are quoted as context |

Falsifiers declared now:

- If the bead's reflection-zero frequencies at the claims rung are more than 1 % from the analytic
  values, or |S21| / |S11| outside the cores more than 2 dB from the analytic curve, the family
  does not close on this battery; the curves are reported and the PI decides.
- If the successive ladder differences do not shrink, no annulus cell count is recommended.
- If the raw column power exceeds 1.02 on a settled record at the claims rung, the family does not
  close.
- If the forward identity fails at the contract's tolerance on complex S, criterion 1(2) is open
  and the measured per-entry difference is reported; the tolerance is not loosened.

## Execution

One VESSL job per (DUT, rung) for stage S; one job each for 1(2), 3a two-port, 3a one-port (the
one-port AD is recorded at +10.8 GB host memory: use the a6000 preset), and 3b. Each job records
its commit with no fallback and aborts on a dirty tree; a failed stage ships its partial JSON. An
assembler joins the stage records into `tests/fixtures/coax_chain_battery/fixture.json`, which one
replay test re-derives every assembled number from.

## Addendum (leader, 2026-09-23, before the battery is re-run on the fixed conductors)

Written after the conductor realization fix (PR 1169, `02ff8849`) and before any battery solve on
it. Corrections to the text above, each a fact from the repository, not a change of bar:

1. **The line is PTFE-filled, not air-filled.** The fixture line carries `eps_r = 2.1`; the
   analytic `beta = omega sqrt(2.1) / c` and `Z_TEM = eta0 ln(b/a) / (2 pi sqrt(2.1))`
   (48.591 ohm for the fixture radii). "air-filled" above is wrong and is superseded here.
2. **Plane invariance (3b) is a DUT translation, not a probe move.** The lane has no probe
   ladder offset for its TEM plane source; the test that stands in is the bead moved by
   `Delta` cells along the axis with the ports fixed, so `angle(S11)` rotates by `2 beta Delta`
   and `angle(S21)` by `beta Delta`, with the analytic beta. Both signs stored.
3. **Settling witness.** Where the lane emits no `settling_db`, the contract's criterion-2
   substitute applies: the record is doubled once and the change in |S| at every bin is stored;
   a change above 0.1 dB anywhere in band means the record is not settled and is re-run longer.
4. **AD (3a) runs on the short 8 x 8 x 12 mm board only**, because the two-port lane has no
   checkpoint segments and the full board's tape does not fit any preset (622 GiB estimated).
   The short board is inside the bar for beta on neither rung (its three-probe span is
   0.17-0.41 rad), so 3a is a gradient check and says nothing about the S of that board;
   this is stated in the record.
5. **Bead mask.** The bead is an `eps_scale` array over the full cross-section between the
   conductors; conductor cells are excluded by the PEC edge masks, not by the scale array.
6. **Ladder reading.** A rung is "inside the bar against the finest" when the bead's
   reflection-zero frequencies are within 1 % and |S21| / |S11| outside the -20 dB cores within
   2 dB of the 9-cell rung. The recommended annulus cell count is the coarsest such rung; the
   conductor fix put 3.789 cells inside for beta and Z0, so the bead is what decides.
7. **Impedance.** The lane's load-derived Z0 is the closed form by construction (see
   `coax_conductor_realization.md`); the battery does not report it as a measurement. The
   thru's |S11| against the continuum `Z_TEM` is the impedance witness (bound
   `|Z/Z_TEM - 1| <= 2 max|S11|`).

## Addendum 2 (2026-09-23, after the first re-run on the fixed conductors)

Appended after the first re-run's records were read. Items 8 and 11 are the leader's; items
9 and 10 are the PI's decisions of 2026-09-23 and are cited in the fixture and the replay test
where they apply.

8. **The bead is four annulus widths, a whole number of cells at every rung (leader).** The
   declared 6 mm rasterized to 17 / 25 / 38 cells, i.e. 6.035 / 5.917 / 5.996 mm at 4 / 6 / 9
   annulus cells. The reflection zero is fixed by the bead's length, so the 6-cell rung's zero
   sat 1.37 % from the 9-cell one while each rung was within 0.06-0.16 % of the closed form on
   its OWN realized length: the ladder compared three beads, not one bead on three meshes. The
   bead is now 4 (b - a) = 5.68 mm, which is 16 / 24 / 36 cells, and the driver asserts that
   cell count per rung before any solve. Every bead stage is re-run with it: the bead solves at
   4 / 6 / 9 annulus cells and the doubled record at 9, the plane test at 6 cells (translation
   4 cells), the identity's bead arm, and the two-port AD leg on its 16 mm board (bead 16
   cells, 5.68 mm). Nothing else is re-run; the other records are carried from `ca6da2b1`,
   whose `rfx/` tree is the same git object as the re-run's.
9. **The battery closes without the open termination (PI, P1).** Every open record stays in
   the fixture as measured, marked `judged: false` with the reason "open end inside the lane's
   closed PEC can (absorbers on z only); its reflection does not settle at 9 annulus cells —
   energy near the outer region's first cutoff grows with record length
   (open_absorber_diagnostic.json); PI 2026-09-23". The passivity and settling checks skip the
   open with that reason; nothing else about the open or the other terminations changes.
10. **Criterion 1(2) is the identity within the traced path (PI, P2).** The bead in a numpy and
    in a jnp container is held to rtol 1e-5 / atol 1e-7 as declared. The untraced call (float64
    NumPy assembly) and the traced call (float32 jnp) are two functions by design; the
    contract's identity clause applies where the traced and untraced call are the same
    function. Their difference on the thru, max |dS| = 6.938e-4, is recorded and pinned as a
    measured envelope at 1.5x (1.041e-3). This supersedes the falsifier line "If the forward
    identity fails at the contract's tolerance on complex S, criterion 1(2) is open" for the
    thru arm only.
11. **A deep quantity is judged by the bound on the ladder (leader).** A ladder quantity below
    -20 dB at every bin, on the finest rung or in its closed form (the thru's |S11| and |S22|),
    is judged by the -20 dB bound at every rung, never by a dB comparison. A quantity deep only
    inside a reflection zero's core (the bead's |S11| and |S22|) is compared in dB outside the
    core; inside it the verdict stays the zero's frequency (PI ruling of 2026-09-21). The
    thru's rung verdict therefore reads on |S21| within 2 dB and on the bound.
