# Restore the retained #931 chain-battery checks

The raw VESSL 369367259427 measurement is unchanged. The ingest had consumed
the report-first artifact without doing the predeclared pin step, then treated
null pins as reports. The accompanying `enforcement_931_realized_pec_forward2_run369367259427.json`
is an audited replay overlay tied to the raw file by SHA-256. Replay loads its
committed pins; it never derives a new gate from the measurement under test.

All 28 retained checks have the necessary measurements in this artifact.
The existing `pin_fixture` derivation recovers every pin, and all 28 numeric
pins equal the historical v18 pins. No gate is widened. None of the 28 owes a
new measurement. Schema-4 missing pins now yield `owed`; historical report-first
schemas retain their original verdicts.

The raw artifact alone now replays as 102 pass, 48 report_only, and 29 owed
(these 28 pin records plus the omitted falsifier). The live fixture loader
applies the audited overlay before replay; the measurement driver's assembly
summary also prints each owed key, so an unadjudicated ingest cannot present
its debt merely as zero failures.

| Check | Stored measurement | Restored pin | Disposition |
|---|---:|---:|---|
| `gradient_invariance\|pec_short\|false\|eps:s11_complex` | 1.41279895492e-05 | 0.001 | restored from artifact |
| `gradient_invariance\|pec_short\|false\|sigma:s11_mag2` | 7.41778901523e-08 | 0.001 | restored from artifact |
| `gradient_invariance\|pec_short\|flux\|eps:s11_complex` | 2.98005959464e-06 | 0.001 | restored from artifact |
| `gradient_invariance\|pec_short\|flux\|sigma:s11_mag2` | 7.42567622106e-08 | 0.001 | restored from artifact |
| `gradient_invariance\|slab\|false\|eps:s11_mag2` | 0 | 0.001 | restored from artifact |
| `gradient_invariance\|slab\|false\|eps:s21_complex` | 3.58110569348e-06 | 0.001 | restored from artifact |
| `gradient_invariance\|slab\|false\|eps:s21_mag2` | 0 | 0.001 | restored from artifact |
| `gradient_invariance\|slab\|flux\|eps:s11_mag2` | 9.46182555516e-08 | 0.001 | restored from artifact |
| `gradient_invariance\|slab\|flux\|eps:s21_complex` | 2.17098731622e-06 | 0.001 | restored from artifact |
| `gradient_invariance\|slab\|flux\|eps:s21_mag2` | 0 | 0.001 | restored from artifact |
| `ladder_monotone\|pec_short_s11_mag\|false` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|pec_short_s11_mag\|flux` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|pec_short_s11_phase_deg\|false` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|pec_short_s11_phase_deg\|flux` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|slab_s11_mag\|false` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|slab_s11_mag\|flux` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|slab_s21_mag\|false` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|slab_s21_mag\|flux` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|slab_s21_phase_deg\|false` | 1 | 0.66 | restored from artifact |
| `ladder_monotone\|slab_s21_phase_deg\|flux` | 1 | 0.66 | restored from artifact |
| `ladder_richardson\|pec_short_s11_phase_deg\|false` | 0.99499792896 | 1.5 | restored from artifact |
| `ladder_richardson\|pec_short_s11_phase_deg\|flux` | 0.883943477809 | 1.4 | restored from artifact |
| `ladder_richardson\|slab_s11_mag\|false` | 0.0175890853716 | 0.03 | restored from artifact |
| `ladder_richardson\|slab_s11_mag\|flux` | 0.0181118015877 | 0.03 | restored from artifact |
| `ladder_richardson\|slab_s21_mag\|false` | 0.0131126894933 | 0.02 | restored from artifact |
| `ladder_richardson\|slab_s21_mag\|flux` | 0.0116885352433 | 0.02 | restored from artifact |
| `ladder_richardson\|slab_s21_phase_deg\|false` | 2.0512656222 | 3.1 | restored from artifact |
| `ladder_richardson\|slab_s21_phase_deg\|flux` | 1.91263629326 | 2.9 | restored from artifact |

Gradient measurements are relative changes; Richardson measurements are the
mid-fine oracle residual (magnitude or degrees as named); monotonicity is the
fraction of bins. Gradient upper bounds use the shared envelope at quantum
1000, Richardson upper bounds use quantum 100 for magnitude and 10 for phase,
and monotonicity lower bounds use quantum 100, all with the unchanged shared
envelope multiplier. The generation step was `G.pin_fixture(deepcopy(raw))`;
regression tests independently rederive and compare both the overlay and the
historical pins.

## The restored falsifier

`cheap_refute_flip_shift_sign` exists under the realized PEC operator. It
requires no new field solve: reference-plane translation acts on the solved
modal amplitudes. For `S_ij = b_i/a_j`, its translation is `S_ij * B_i/A_j`,
where `A_i, B_i = _shift_modal_waves(1, 1, beta_i, shift_i, step_sign_i)`.
The test calls the production modal shift function on the measured cell S
matrices, using each stored cutoff and shift. Reversing both port signs is
the same post-solve mutation as the measurement driver’s `_FlippedShift`.

The unmutated control reproduces all four stored shifted matrices with
maximum complex difference below 2.4e-7 (within the existing live tolerance).
The reversed-sign case preserves magnitudes and yields per-entry worst
Yee rotation residuals from 116.776811 to 177.732561 degrees, so every
measurable entry rejects the wrong sign under the unchanged 10 degree
discriminator. This is explicitly a recovered post-solve falsifier, not a
claim that the omitted VESSL refute stage was run. A positive-sign control
fails the falsifier, and deleting its record reports `owed`.

## Census and regression checks

The enforced census is 179 = 185 historical minus the six authorized
lossless PEC magnitude-leg removals. It contains 131 pass, 48 report_only,
zero fail, and zero owed. The set comparison names those six removals; a
count alone cannot authorize removing another discriminator. The 48 reports
are the original off-claims-rung physics/referee and power-closure reports.

`test_waveguide_chain_battery_v18_close.py` adds 28 separate bad-measurement
mutations, 28 separate missing-pin checks, a pin recovery/no-widening check,
and the production shift-sign falsifier with its positive control. The ingest
test compares the complete retained key set and the frozen verdict record.
