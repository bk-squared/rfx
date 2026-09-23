# Physics witness records

A 24 mm PEC cube was excited by an Ez differentiated-Gaussian point source; the TM110 closed-form frequency is 8.832720 GHz. The parallel-plate line has PEC plates, PMC side faces, and resistive terminations with closed-form |S11| values 1/3, 0, 1/3. The measured frequencies, port magnitudes, and CPML-box probe values are tabulated below.

Preflight output is preserved verbatim in each linked `.log`, including the empty-report banner. Each JSON contains the realized grid, pads, timestep, record length, source cutoff step, field-energy record summary, and run id. NPZ files retain the full probe, source, and per-step energy arrays. Energy is the recorded discrete 0.5 sum(epsilon0 |E|² + mu0 |H|²) dx³, with absorber pads excluded. No claim of mesh convergence, passivity, or settling is assigned here.

## W1 — closed PEC cube

The requested two spacings and four entry points are listed against the same analytic value. Source (11,12,10) mm, probe (14,9,13) mm, Ez; source f0 = 8.832720 GHz, fractional bandwidth 0.8. Ring-down analysis is the repository `rfx.harminv.harminv`, with the complete returned mode list in each JSON; the table selects the nearest mode to TM110.

| Entry | dx (mm) | Realized nodes | f (GHz) | Analytic (GHz) | End/post-source peak energy (dB) | Evidence |
|---|---:|---|---:|---:|---:|---|
| run | 1 | [25, 25, 25] | 8.830531861 | 8.832720000 | -0.2381947761382165 | [witness/W1__run__1.json](witness/W1__run__1.json) |
| forward | 1 | [25, 25, 25] | 8.830531862 | 8.832720000 | -0.23819438442365787 | [witness/W1__forward__1.json](witness/W1__forward__1.json) |
| sweep | 1 | [25, 25, 25] | 8.830531749 | 8.832720000 | -0.23819088284488665 | [witness/W1__sweep__1.json](witness/W1__sweep__1.json) |
| nonuniform | 1 | [25, 25, 25] | 8.830531749 | 8.832720000 | -0.23818982103822037 | [witness/W1__nonuniform__1_nu_measured.json](witness/W1__nonuniform__1_nu_measured.json) |
| run | 0.5 | [49, 49, 49] | 8.832173129 | 8.832720000 | -0.14672914475347648 | [witness/W1__run__0p5.json](witness/W1__run__0p5.json) |
| forward | 0.5 | [49, 49, 49] | 8.832173129 | 8.832720000 | -0.14672915653439067 | [witness/W1__forward__0p5.json](witness/W1__forward__0p5.json) |
| sweep | 0.5 | [49, 49, 49] | 8.832173014 | 8.832720000 | -0.14672739521761125 | [witness/W1__sweep__0p5.json](witness/W1__sweep__0p5.json) |
| nonuniform | 0.5 | [49, 49, 49] | 8.832173015 | 8.832720000 | -0.14672739521761125 | [witness/W1__nonuniform__0p5_nu_measured.json](witness/W1__nonuniform__0p5_nu_measured.json) |

## W2 — requested PMC cavity

**NOT MEASURED.** Issue #1164 body and comments do not specify the cavity dimensions, full face assignment, source, or probe. An asynchronous request for that configuration was made; no supplied configuration is present. The stated 7.495 / 16.759 GHz numbers do not uniquely specify a cavity.

The wall-operator measurement is separate: `operator_measurement.json` records tangential H zeroing at low index 0 and high index N−2 on each of the six faces. At dx = 1 mm the recorded positions are +0.5 mm from the low node plane and −0.5 mm from the high node plane. No cavity frequency is inferred from that operator measurement.

## W3 — parallel-plate known-load line

Reference fixture: `reference_known_load_line.py`, retrieved from PR #1162 at `89326838939bf9573b11352ad832688f5548d4f6`; the file is absent at the measured source revision. The solver is always `798ec64e`. Domain 4 x 1 x 1 mm; dx = 1 mm; PMC x/y, PEC z; Zc = 376.730313668 ohm; load at x = 3 mm. The two requested port positions are x = 1 mm and x = 0; both retain y = z = 0 from the committed fixture. There are 2,100 steps in every arm.

| Entry / port | x (mm) | R/Zc | Closed-form magnitude | S11 magnitude at 1 / 2.5 / 5 / 7.5 / 10 GHz | Evidence |
|---|---:|---:|---:|---|---|
| run / lumped | 1 | 0.5 | 0.33333333 | 0.99999994, 1, 1, 0.99999994, 0.99999994 | [witness/W3__run__lumped__0p5__1.json](witness/W3__run__lumped__0p5__1.json) |
| run / lumped | 0 | 0.5 | 0.33333333 | 0, 0, 0, 0, 0 | [witness/W3__run__lumped__0p5__0.json](witness/W3__run__lumped__0p5__0.json) |
| run / lumped | 1 | 1 | 0 | 0.99999994, 1, 1, 0.99999994, 0.99999994 | [witness/W3__run__lumped__1__1.json](witness/W3__run__lumped__1__1.json) |
| run / lumped | 0 | 1 | 0 | 0, 0, 0, 0, 0 | [witness/W3__run__lumped__1__0.json](witness/W3__run__lumped__1__0.json) |
| run / lumped | 1 | 2 | 0.33333333 | 0.99999994, 1, 1, 0.99999994, 0.99999994 | [witness/W3__run__lumped__2__1.json](witness/W3__run__lumped__2__1.json) |
| run / lumped | 0 | 2 | 0.33333333 | 0, 0, 0, 0, 0 | [witness/W3__run__lumped__2__0.json](witness/W3__run__lumped__2__0.json) |
| run / wire | 1 | 0.5 | 0.33333333 | 1.0000076, 0.99999696, 0.99998736, 1.0000175, 1.0000212 | [witness/W3__run__wire__0p5__1.json](witness/W3__run__wire__0p5__1.json) |
| run / wire | 0 | 0.5 | 0.33333333 | 1, 1, 1, 1, 1 | [witness/W3__run__wire__0p5__0.json](witness/W3__run__wire__0p5__0.json) |
| run / wire | 1 | 1 | 0 | 1.0000076, 0.99999696, 0.99998736, 1.0000175, 1.0000212 | [witness/W3__run__wire__1__1.json](witness/W3__run__wire__1__1.json) |
| run / wire | 0 | 1 | 0 | 1, 1, 1, 1, 1 | [witness/W3__run__wire__1__0.json](witness/W3__run__wire__1__0.json) |
| run / wire | 1 | 2 | 0.33333333 | 1.0000076, 0.99999696, 0.99998736, 1.0000175, 1.0000212 | [witness/W3__run__wire__2__1.json](witness/W3__run__wire__2__1.json) |
| run / wire | 0 | 2 | 0.33333333 | 1, 1, 1, 1, 1 | [witness/W3__run__wire__2__0.json](witness/W3__run__wire__2__0.json) |
| forward / lumped | 1 | 0.5 | 0.33333333 | 1, 1, 1, 1, 0.99999994 | [witness/W3__forward__lumped__0p5__1.json](witness/W3__forward__lumped__0p5__1.json) |
| forward / lumped | 0 | 0.5 | 0.33333333 | 0, 0, 0, 0, 0 | [witness/W3__forward__lumped__0p5__0.json](witness/W3__forward__lumped__0p5__0.json) |
| forward / lumped | 1 | 1 | 0 | 1, 1, 1, 1, 0.99999994 | [witness/W3__forward__lumped__1__1.json](witness/W3__forward__lumped__1__1.json) |
| forward / lumped | 0 | 1 | 0 | 0, 0, 0, 0, 0 | [witness/W3__forward__lumped__1__0.json](witness/W3__forward__lumped__1__0.json) |
| forward / lumped | 1 | 2 | 0.33333333 | 1, 1, 1, 1, 0.99999994 | [witness/W3__forward__lumped__2__1.json](witness/W3__forward__lumped__2__1.json) |
| forward / lumped | 0 | 2 | 0.33333333 | 0, 0, 0, 0, 0 | [witness/W3__forward__lumped__2__0.json](witness/W3__forward__lumped__2__0.json) |
| forward / wire | 1 | 0.5 | 0.33333333 | 1.0000076, 0.99999696, 0.99998736, 1.0000175, 1.0000212 | [witness/W3__forward__wire__0p5__1.json](witness/W3__forward__wire__0p5__1.json) |
| forward / wire | 0 | 0.5 | 0.33333333 | 1, 1, 1, 1, 1 | [witness/W3__forward__wire__0p5__0.json](witness/W3__forward__wire__0p5__0.json) |
| forward / wire | 1 | 1 | 0 | 1.0000076, 0.99999696, 0.99998736, 1.0000175, 1.0000212 | [witness/W3__forward__wire__1__1.json](witness/W3__forward__wire__1__1.json) |
| forward / wire | 0 | 1 | 0 | 1, 1, 1, 1, 1 | [witness/W3__forward__wire__1__0.json](witness/W3__forward__wire__1__0.json) |
| forward / wire | 1 | 2 | 0.33333333 | 1.0000076, 0.99999696, 0.99998736, 1.0000175, 1.0000212 | [witness/W3__forward__wire__2__1.json](witness/W3__forward__wire__2__1.json) |
| forward / wire | 0 | 2 | 0.33333333 | 1, 1, 1, 1, 1 | [witness/W3__forward__wire__2__0.json](witness/W3__forward__wire__2__0.json) |

## W4 — all-CPML point-source box

The declared 24 x 24 x 24 mm vacuum box has eight CPML layers on every face and 41 x 41 x 41 stored nodes. The Ez source is at (12,12,12) mm and the probe at (18,14,13) mm; source f0 = 7 GHz, bandwidth 1.0; dx = 1 mm, 4,096 steps. The third arm returns the input state from `rfx.simulation.apply_pec`; absorber updates are retained.

| Entry | Final Ez (V/m) | End/post-source peak energy (dB) | Evidence |
|---|---:|---:|---|
| run | -8.74824536368e-07 | -16.531873 | [witness/W4__run.json](witness/W4__run.json) |
| forward | -8.70895860317e-07 | -16.536028 | [witness/W4__forward.json](witness/W4__forward.json) |
| run-no-backing | -8.70895860317e-07 | -16.536028 | [witness/W4__run-no-backing.json](witness/W4__run-no-backing.json) |

| Subtraction | Final ΔEz (V/m) | Maximum magnitude of ΔEz (V/m) |
|---|---:|---:|
| run minus forward | -3.92867605115e-09 | 3.0267983675e-08 |
| run minus run-no-backing | -3.92867605115e-09 | 3.0267983675e-08 |
| forward minus run-no-backing | 0 | 0 |

Open-box energy criterion from workspace guidance: end/post-source peak below −40 dB. W4 values above are above that threshold; workspace label: `truncation-suspect`.

## Per-arm evidence and energy

| Arm JSON | Run id | Preflight log | Post-source energy peak / final (J), per captured scan |
|---|---|---|---|
| [witness/W1__run__1.json](witness/W1__run__1.json) | 369367263562 | [witness/W1__run__1.log](witness/W1__run__1.log) | 6.54425e-22 / 6.19498e-22 |
| [witness/W1__forward__1.json](witness/W1__forward__1.json) | 369367263562 | [witness/W1__forward__1.log](witness/W1__forward__1.log) | 6.54425e-22 / 6.19498e-22 |
| [witness/W1__sweep__1.json](witness/W1__sweep__1.json) | 369367263562 | [witness/W1__sweep__1.log](witness/W1__sweep__1.log) | 6.54425e-22 / 6.19499e-22 |
| [witness/W1__nonuniform__1_nu_measured.json](witness/W1__nonuniform__1_nu_measured.json) | 369367263567 | [witness/W1__nonuniform__1_nu_measured.log](witness/W1__nonuniform__1_nu_measured.log) | 6.54425e-22 / 6.19499e-22 |
| [witness/W1__run__0p5.json](witness/W1__run__0p5.json) | 369367263562 | [witness/W1__run__0p5.log](witness/W1__run__0p5.log) | 3.99719e-23 / 3.8644e-23 |
| [witness/W1__forward__0p5.json](witness/W1__forward__0p5.json) | 369367263562 | [witness/W1__forward__0p5.log](witness/W1__forward__0p5.log) | 3.99719e-23 / 3.8644e-23 |
| [witness/W1__sweep__0p5.json](witness/W1__sweep__0p5.json) | 369367263562 | [witness/W1__sweep__0p5.log](witness/W1__sweep__0p5.log) | 3.99719e-23 / 3.8644e-23 |
| [witness/W1__nonuniform__0p5_nu_measured.json](witness/W1__nonuniform__0p5_nu_measured.json) | 369367263567 | [witness/W1__nonuniform__0p5_nu_measured.log](witness/W1__nonuniform__0p5_nu_measured.log) | 3.99719e-23 / 3.8644e-23 |
| [witness/W3__run__lumped__0p5__1.json](witness/W3__run__lumped__0p5__1.json) | 369367263563 | [witness/W3__run__lumped__0p5__1.log](witness/W3__run__lumped__0p5__1.log) | 1.99532e-22 / 1.99532e-22; 1.99532e-22 / 1.99532e-22 |
| [witness/W3__run__lumped__0p5__0.json](witness/W3__run__lumped__0p5__0.json) | 369367263563 | [witness/W3__run__lumped__0p5__0.log](witness/W3__run__lumped__0p5__0.log) | 7.7467e-27 / 0; 7.7467e-27 / 0 |
| [witness/W3__run__lumped__1__1.json](witness/W3__run__lumped__1__1.json) | 369367263563 | [witness/W3__run__lumped__1__1.log](witness/W3__run__lumped__1__1.log) | 1.99532e-22 / 1.99532e-22; 1.99532e-22 / 1.99532e-22 |
| [witness/W3__run__lumped__1__0.json](witness/W3__run__lumped__1__0.json) | 369367263563 | [witness/W3__run__lumped__1__0.log](witness/W3__run__lumped__1__0.log) | 7.7467e-27 / 0; 7.7467e-27 / 0 |
| [witness/W3__run__lumped__2__1.json](witness/W3__run__lumped__2__1.json) | 369367263563 | [witness/W3__run__lumped__2__1.log](witness/W3__run__lumped__2__1.log) | 1.99532e-22 / 1.99532e-22; 1.99532e-22 / 1.99532e-22 |
| [witness/W3__run__lumped__2__0.json](witness/W3__run__lumped__2__0.json) | 369367263563 | [witness/W3__run__lumped__2__0.log](witness/W3__run__lumped__2__0.log) | 7.7467e-27 / 0; 7.7467e-27 / 0 |
| [witness/W3__run__wire__0p5__1.json](witness/W3__run__wire__0p5__1.json) | 369367263563 | [witness/W3__run__wire__0p5__1.log](witness/W3__run__wire__0p5__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__run__wire__0p5__0.json](witness/W3__run__wire__0p5__0.json) | 369367263563 | [witness/W3__run__wire__0p5__0.log](witness/W3__run__wire__0p5__0.log) | 7.7467e-27 / 0 |
| [witness/W3__run__wire__1__1.json](witness/W3__run__wire__1__1.json) | 369367263563 | [witness/W3__run__wire__1__1.log](witness/W3__run__wire__1__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__run__wire__1__0.json](witness/W3__run__wire__1__0.json) | 369367263563 | [witness/W3__run__wire__1__0.log](witness/W3__run__wire__1__0.log) | 7.7467e-27 / 0 |
| [witness/W3__run__wire__2__1.json](witness/W3__run__wire__2__1.json) | 369367263563 | [witness/W3__run__wire__2__1.log](witness/W3__run__wire__2__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__run__wire__2__0.json](witness/W3__run__wire__2__0.json) | 369367263563 | [witness/W3__run__wire__2__0.log](witness/W3__run__wire__2__0.log) | 7.7467e-27 / 0 |
| [witness/W3__forward__lumped__0p5__1.json](witness/W3__forward__lumped__0p5__1.json) | 369367263563 | [witness/W3__forward__lumped__0p5__1.log](witness/W3__forward__lumped__0p5__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__forward__lumped__0p5__0.json](witness/W3__forward__lumped__0p5__0.json) | 369367263563 | [witness/W3__forward__lumped__0p5__0.log](witness/W3__forward__lumped__0p5__0.log) | 7.7467e-27 / 0 |
| [witness/W3__forward__lumped__1__1.json](witness/W3__forward__lumped__1__1.json) | 369367263563 | [witness/W3__forward__lumped__1__1.log](witness/W3__forward__lumped__1__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__forward__lumped__1__0.json](witness/W3__forward__lumped__1__0.json) | 369367263563 | [witness/W3__forward__lumped__1__0.log](witness/W3__forward__lumped__1__0.log) | 7.7467e-27 / 0 |
| [witness/W3__forward__lumped__2__1.json](witness/W3__forward__lumped__2__1.json) | 369367263563 | [witness/W3__forward__lumped__2__1.log](witness/W3__forward__lumped__2__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__forward__lumped__2__0.json](witness/W3__forward__lumped__2__0.json) | 369367263563 | [witness/W3__forward__lumped__2__0.log](witness/W3__forward__lumped__2__0.log) | 7.7467e-27 / 0 |
| [witness/W3__forward__wire__0p5__1.json](witness/W3__forward__wire__0p5__1.json) | 369367263563 | [witness/W3__forward__wire__0p5__1.log](witness/W3__forward__wire__0p5__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__forward__wire__0p5__0.json](witness/W3__forward__wire__0p5__0.json) | 369367263563 | [witness/W3__forward__wire__0p5__0.log](witness/W3__forward__wire__0p5__0.log) | 7.7467e-27 / 0 |
| [witness/W3__forward__wire__1__1.json](witness/W3__forward__wire__1__1.json) | 369367263563 | [witness/W3__forward__wire__1__1.log](witness/W3__forward__wire__1__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__forward__wire__1__0.json](witness/W3__forward__wire__1__0.json) | 369367263563 | [witness/W3__forward__wire__1__0.log](witness/W3__forward__wire__1__0.log) | 7.7467e-27 / 0 |
| [witness/W3__forward__wire__2__1.json](witness/W3__forward__wire__2__1.json) | 369367263563 | [witness/W3__forward__wire__2__1.log](witness/W3__forward__wire__2__1.log) | 1.99532e-22 / 1.99532e-22 |
| [witness/W3__forward__wire__2__0.json](witness/W3__forward__wire__2__0.json) | 369367263563 | [witness/W3__forward__wire__2__0.log](witness/W3__forward__wire__2__0.log) | 7.7467e-27 / 0 |
| [witness/W4__run.json](witness/W4__run.json) | 369367263564 | [witness/W4__run.log](witness/W4__run.log) | 4.4393e-25 / 9.86569e-27 |
| [witness/W4__forward.json](witness/W4__forward.json) | 369367263564 | [witness/W4__forward.log](witness/W4__forward.log) | 4.43921e-25 / 9.85606e-27 |
| [witness/W4__run-no-backing.json](witness/W4__run-no-backing.json) | 369367263564 | [witness/W4__run-no-backing.log](witness/W4__run-no-backing.log) | 4.43921e-25 / 9.85606e-27 |

Conclusion: leader fills.
