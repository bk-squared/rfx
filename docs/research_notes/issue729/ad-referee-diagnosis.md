# AD consumer: verify the precision of the evaluated fields

2026-09-11. This is an unresolved AD/FD comparison at the initial checkpoint,
with a confirmed defect in the old reference configuration. No gradient
tolerance or resolving-power floor has been changed.

## Measurements before correcting the reference

Run 369367260429, candidate b036dbc7, passed all 24 projection tests and the
unchanged x/y rotation gate (complex S difference 3.947e-6, every drive
settled below -100 dB). It then failed the legacy MSL AD gate:

- f32 AD: 8.871184e-5;
- nominally f64 FD: 1.097956e-3;
- relative difference 0.9192 against the unchanged 0.03 gate;
- reported final-loss resolving power 1.98e10 ULP.

The driver stopped before NU. Raw returned matrices and the complete log
are retained in `gpu-final-consumers-369367260429/`.

Run 369367260430 performed two sequential controls using the same original
fixture, objective, record, finite-difference step and gate:

| Arm | AD | FD | Relative difference |
|---|---:|---:|---:|
| Current main e32e386d, default precision | 1.536663e-3 | 1.546729e-3 | 0.0065 |
| Candidate b036dbc7, highest ambient matmul precision | 8.871184e-5 | 1.097956e-3 | 0.9192 |

The highest-precision control reproduces the candidate's values. Ambient
matmul precision is therefore not the explanation for this AD comparison.
The main pass does not establish the reference's evaluation accuracy on
the much weaker candidate gradient.

## Confirmed configuration defect

The old test entered `enable_x64()` and supplied float64 eps_override, but
its builder did not request `Simulation(precision="float64")`. The API's
default is float32. `_resolve_field_dtype()` therefore returned None, and
`rfx.simulation._build_step_setup` mapped that to float32 initial E/H state.
Yee and CPML updates cast fields back to that dtype each step. The final
DFT/extractor/loss becoming float64 did not undo that rounding.

Consequently, the printed ULP count measures the final loss's storage
resolution, not a bound on the noise of the internal field evaluation.
The same defect was present in `scripts/msl_ad_fd_f64_referee.py`.

The test and diagnostic now use `_build_msl_f64_referee()`, which requests
float64 fields as well as the scoped x64 arithmetic. A small always-on
test intercepts the real core `init_state`, creates its actual six E/H
arrays, records their dtypes and stops before a scan. Both arms enable
x64 and provide eps64: the default builder produces float32 E/H, whereas
the reference builder produces float64 E/H. This verifies the computational
carrier, rather than assuming the dtype of the final scalar proves it.

The full corrected reference still needs to run. This configuration finding
alone does not establish that every part of the 91.9% discrepancy is fixed.

## Separate geometry debt

The legacy AD fixture declares h_sub=254 um at dx=80 um but explicitly
retains its historical trace sheet at 320 um. The source span correction
and this compensation interact. Redraw onto an aligned physical board is
needed for a matching-geometry acceptance fixture, but changing geometry
before identifying the reference defect could hide the failing comparison.
The reference correction is therefore measured on the existing geometry
first. No production policy for mismatched declared/actual port heights
is inferred from this test; that PI question remains pending.
