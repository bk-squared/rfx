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

## Addendum 2026-09-21 (leader; written after the 100 and 50 um rungs, before the 25 um rung was read)

Three things the note above left open were settled while the battery ran. None moves a verdict
number.

1. **Record length and drive**, from the pilot the last paragraph asks for: 20 source periods (10
   reads -37 dB on the settling witness, 20 reads -68 dB), and a drive of `f0 = 4 GHz`,
   `bandwidth = 1.0` instead of the shipped `f0 = 3.5 GHz`, `bandwidth = 0.8`, because the shipped
   pulse is about 39 dB below its own peak at 7 GHz and leaves 6.4 - 7.0 GHz unreliable on both
   ports. Both pilots are in the fixture.
2. **Deep nulls (PI ruling, 2026-09-21).** A quantity that is near zero by construction is not
   compared in dB from rung to rung. (a) The thru's |S11| is held to an upper bound, -20 dB at every
   bin of every rung, and recorded as the line's own reflection floor. (b) The 2 dB comparison
   applies to the thru's |S21| and to the notch's |S21| and |S11| OUTSIDE the notch core — the bins
   where the finest rung's |S21| is at or below -20 dB. (c) Inside the core the verdict is the notch
   FREQUENCY against 1 %; the depth is recorded per rung with no comparison. The depth read -42.5 dB
   at 100 um and -33.8 dB at 50 um purely from where the 50 MHz bin fell.
3. **The AD stage runs on a 48 GB card.** On the 24 GB card the reverse-mode checkpoint stack asked
   for 14.08 GiB in one allocation while the allocator's own summary showed its pool a little over
   half full (the log prints a usage bar, not a byte count). Nothing was reduced to make it
   fit; the numbers go in the record and, as a statement of cost, in the support matrix.

