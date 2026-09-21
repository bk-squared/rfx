# Microstrip chain battery — pre-declaration (reduced v2.0 form)

Status: written BEFORE the first battery solve. Contract:
[`chain_closure_contract.md`](chain_closure_contract.md), section "The v2.0 battery for
lumped/wire, MSL and coax". The verdict numbers are that section's and are not re-declared here;
this note fixes the board, the rungs, the DUTs, the referee and what each stage records.

Lane under test: `Simulation.compute_msl_s_matrix(...)`, uniform mesh, `mode="laplace"` ports,
RAW S (`enforce_passivity=False`, the default).

## Board

One board at every rung. Every patterned dimension is a whole number of cells at the coarsest
rung, so no rung rounds a dimension:

| quantity | value |
|---|---|
| substrate | `eps_r = 3.66`, lossless, `h = 300 um` |
| trace | zero-thickness PEC sheet, `W = 600 um` |
| ground | PEC wall at the bottom of the domain (`pec_faces`), so the return is not a finite sheet |
| line length between port planes | 20 mm |
| stub (notch DUT only) | open-circuit, `W_stub = 600 um`, `L_stub = 12 mm`, at the line's midpoint |
| air above the substrate | 1.5 mm |
| lateral margin beyond the stub's open end / the trace | 3 mm |
| absorber | CPML, 8 layers, on x, y and +z |
| band | 1 - 7 GHz, 121 points |

What the grid realizes is ASSERTED per rung before any solve (substrate cells under the strip,
trace width in cells, stub length in cells, port-plane positions in metres) and the realized
values are stored. A PEC sheet is solved about 0.35 cell wider at each free edge
(`rfx/mesh_edges.py`); the trace is therefore solved as `W + 0.7 dx` and the stub as
`L_stub + 0.35 dx`. That is not corrected here: how fast it converges is one of the things the
ladder is for.

## Rungs

`dx = 100, 50, 25 um` (3 / 6 / 12 cells under the strip; 6 / 12 / 24 across it). The 25 um rung is
the claims rung; verdicts are read there, the other two are the trend. The recommended cell size
the support matrix will state is the coarsest rung from which the notch frequency stays within 1 %
and |S21|, |S11| (outside the notch's -20 dB core) within 2 dB of the 25 um rung.

## DUTs

1. **Open-stub notch** (reflecting). Expected: one notch in band near
   `c / (4 L_stub sqrt(eps_eff))`, `eps_eff` from `hammerstad_jensen_z0_eps_eff(W, h, eps_r)`.
2. **Thru** (control): the same line without the stub. It shows the gates are not vacuous and
   gives the line's own |S11| floor.

## Stages and what each records

| stage | rungs | DUTs | records |
|---|---|---|---|
| S — solve | all | both | complex S(f), `sigma_max_excess`, `settling_db`, `reliable`, `Z0`, `beta`, preflight text, realized geometry, wall time, peak memory |
| 2 — physics gates | 25 um (others reported) | notch | max column power, `max_f |S21 - S12| / max|S|`, power closure `1 - sum_i |S_i1|^2` (lossless board: the residual is what the absorber and radiation take; REPORTED with its curve, gated only as "<= 1.02 column power") ; settling <= -40 dB |
| 1(2) — forward identity | 100 um | notch | complex S of a no-op `eps_override` against the plain call |
| 3a — AD vs FD | 100 um | notch | `d/d(eps_r scale)` of band-mean `|S21|^2` over 3 - 5 GHz and of `Re(S21 S11*)` at the bin nearest the notch; float32 AD against a float64-loss central FD, ULP-span assert first |
| 3b — plane invariance | 50 um | notch | two runs with the probe ladder moved by `Delta = 10` cells at both ports (`n_probe_offset`): `| |S| - |S|' |`, the rotation of `angle(S11)` against `2 beta Delta` and of `angle(S21)` against `beta (Delta_1 + Delta_2)`, with beta the run's own fitted value |
| 3c — ladder | all | both | notch frequency, -10 dB width, |S21| and |S11| curves, `Re(Z0)` median, successive differences and their ratio |
| 3d — referee | 25 um | notch | notch frequency against the analytic quarter-wave value; the committed openEMS / Palace notch records are for a different board (h = 254 um) and are quoted as context, not compared bin by bin |

Falsifiers declared now:

- If the notch frequency's successive differences do not shrink (ratio >= 1), the ladder is not
  interpretable and no cell size is recommended.
- If the 25 um notch is more than 1 % from the analytic value, the family does not close on this
  battery; the curve and the realized stub length are reported and the PI decides.
- If the raw column power exceeds 1.02 at the claims rung on a settled record, the family does not
  close; `enforce_passivity=True` is not used to pass it.
- If plane invariance fails in magnitude by more than 2 dB anywhere outside the notch core, the
  reference plane is not what the result says it is.

## Execution

One VESSL `gpu-rtx4090` job per (DUT, rung) for stage S — six jobs — plus one job each for
1(2)+3a and for 3b. Each job records its commit with no fallback and aborts on a dirty tree. A
stage writes its own JSON; an assembler joins them into
`tests/fixtures/msl_chain_battery/fixture.json`, which one replay test reads. `num_periods` is
chosen per rung from the settling witness of a short pilot at 100 um and written into the record;
a record above -40 dB is re-run longer, not interpreted.
