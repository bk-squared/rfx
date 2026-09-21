# Lumped / wire port chain battery — pre-declaration (reduced v2.0 form)

Status: written BEFORE the first battery solve. Contract:
[`chain_closure_contract.md`](chain_closure_contract.md), section "The v2.0 battery for
lumped/wire, MSL and coax", with the PI's 2026-09-21 ruling on deep nulls (below).

Chain scope (contract, per-family table): the differentiable **S11** of
`Simulation.forward(port_s11_freqs=...)`, for a lumped port (`add_port(..., extent=None)`) and a
wire port (`add_port(..., extent=...)`). The S matrix of `run(compute_s_params=True)` is a numpy
post-process and is not in the v2.0 chain. Reciprocity, power closure and reference-plane
invariance are dropped for this family: a one-port at a feed point has no second port and no
propagation plane. What replaces them as the physics gate is passivity of the one-port,
`|S11| <= 1 + 0.02` on a settled record.

## Fixture: a line whose input impedance is known in closed form

A lumped port has no modal reference, so the DUT must be something whose impedance AT THE PORT is
analytic. A parallel-plate line one cell wide, between PEC plates (top and bottom) and magnetic
side walls, carries an exact TEM wave: `Zc = eta0 * h / w / sqrt(eps_r)`, `beta = omega
sqrt(eps_r) / c`. The port spans the plate gap at one end of the line, backed by a magnetic wall
(an open), so the port sees only the line: `Zin = Zc (ZL + j Zc tan(beta L)) / (Zc + j ZL tan(beta
L))`, `S11 = (Zin - Zref) / (Zin + Zref)`.

| quantity | value |
|---|---|
| line length `L` (port cell centre to the termination plane) | 30 mm |
| plate gap `h` | lumped port: 1 cell; wire port: 4 cells (`extent = h`) |
| width `w` | 1 cell, magnetic side walls |
| filling | air; `eps_r = 2.2` for the permittivity-gradient leg |
| port reference `Zref` | the line's own analytic `Zc` at that rung (so a matched load is a match) |
| band | 1 - 10 GHz, 91 points |

Because `h` and `w` are counted in cells, `Zc` is the same at every rung for the lumped port
(`eta0`) and for the wire port (`4 eta0`); only `L` in cells changes. If the solver's one-cell-wide
magnetic-wall channel does not hold the TEM field (it must be checked FIRST on the thru-to-open
case against the closed form), the width is realized as two cells or with periodic side walls and
that is recorded — the referee is the 1-D line either way.

## Terminations (DUTs)

1. **Short**: PEC wall at the far end. `|S11| = 1`, phase from `j Zc tan(beta L)`.
2. **Open**: magnetic wall at the far end. `|S11| = 1`, phase from `-j Zc cot(beta L)`.
3. **Resistive**, `R = Zc/2` and `R = 2 Zc`: a lumped resistor (`add_lumped_rlc`) filling the end
   cross-section, backed by a magnetic wall. `|Gamma_L| = 1/3`, so `|S11| = 1/3` across the band
   with the line's phase.
4. **Matched**, `R = Zc` (control): `S11 = 0` in the ideal line. A deep null — see the ruling.

## Deep nulls (PI ruling, 2026-09-21)

The matched case is held to an upper bound, `|S11| <= -20 dB` at every bin of every rung, and
recorded as the port's own reflection floor (this number is what a user most needs from the
battery: how well a lumped port can be matched at a given cell size). It is not compared in dB
from rung to rung. For short and open, magnitude is compared against 1 within 2 dB (in practice
within the 1.02 passivity bound) and the PHASE curve against the closed form: the frequencies at
which `angle(S11)` crosses 0 and pi are held to 1 %.

## Rungs

`dx = 1.0, 0.5, 0.25 mm` (30 / 60 / 120 cells per wavelength at 10 GHz; `L` = 30 / 60 / 120
cells). The 0.25 mm rung is the claims rung. The recommended cell size the support matrix will
state is cells per wavelength at the highest frequency of interest, the coarsest rung from which
the compared quantities stay inside the bar against the finest — for the lumped and the wire port
separately.

## Stages and what each records

| stage | rungs | DUTs | records |
|---|---|---|---|
| S — solve | all | all five, lumped and wire | complex S11(f) from `forward(port_s11_freqs=...)`, the settling witness (or, where the lane emits none, the record-length-doubling substitute of the contract's criterion 2), realized `L`, `h`, `w` in cells and metres, the analytic `Zc` used as `Zref`, preflight text, warnings, wall time |
| 2 — passivity | claims rung (others reported) | short, open, resistive | `max_f |S11| <= 1.02` |
| 1(1)/(2) — trace and forward identity | coarsest | resistive `2 Zc` | `value_and_grad` finite; S11 under a no-op traced `eps_override` equals the untraced call to `rtol=1e-5, atol=1e-7` on complex S11 |
| 3a — AD vs FD | coarsest | resistive (theta = `R`), and the `eps_r = 2.2` line with a short (theta = a scale on the filling's permittivity) | float32 AD against a float64-loss central FD with the ULP-span assert first; objectives band-mean `|S11|^2` and `Re(S11)` at mid-band. The closed form gives the derivative too (`d|S11|^2/dR`, `d/d eps_r`): stored beside AD and FD as a third witness |
| 3c — ladder | all | all | |S11| and `angle(S11)` curves, the phase-crossing frequencies, the matched floor per rung; successive differences and their ratio; first- and second-order extrapolated limits labelled as arithmetic |
| 3d — referee | claims rung | all | the closed forms above. The repository's recorded openEMS PEC-box magnitude comparison is committed as a fixture in the same PR and quoted as context, not used for a verdict |

Falsifiers declared now:

- If the thru-to-open phase at the claims rung is more than 1 % in frequency from the closed form,
  the fixture is not the 1-D line it claims to be and nothing else is read until that is explained.
- If `|S11|` exceeds 1.02 on a settled record at the claims rung, the family does not close.
- If the resistive cases' `|S11|` is more than 2 dB from 1/3 anywhere in band at the claims rung,
  the family does not close on this battery; the curve is reported and the PI decides.
- If AD, FD and the closed-form derivative disagree by more than 5 % pairwise where the FD is
  interpretable, criterion 3a is open.

## Execution

Small problems (a 30 mm one-cell-wide line): one VESSL job per (port kind, DUT, rung) for stage S —
thirty jobs — plus one each for identity and the two AD legs. Each records its commit with no
fallback and aborts on a dirty tree; a failed stage ships its partial JSON. An assembler joins the
records into `tests/fixtures/lumped_wire_chain_battery/fixture.json`, which one replay test
re-derives every assembled number from.
