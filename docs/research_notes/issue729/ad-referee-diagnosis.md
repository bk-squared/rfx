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

Run 369367260431 verified actual float64 E/H state on every FD drive. The
comparison still failed: g_fd=1.095574e-3, versus the same f32 AD value.
Thus the configuration defect was real, but did not explain this discrepancy.

Run 369367260432 performed only a full-f64 AD solve and reused the matching
retained FD matrices. It measured g_ad64=9.101203512968051e-5, still far
from the h=1e-3 finite difference. Changing arithmetic precision alone is
not the fix for the finite-step comparison.

## Finite-step error and the port observation

Run 369367260434 retained the same legacy fixture and f64 fields, reusing
the f64 AD value above, and predeclared three smaller FD steps:

| h | FD derivative | Relative difference from f64 AD |
|---|---:|---:|
| 1e-4 | 5.446927175967176e-4 | 83.29% |
| 1e-5 | 9.630481412337132e-5 | 5.50% |
| 1e-6 | 9.107337106684099e-5 | 0.06735% |

The two fine steps approach the AD derivative. The old h=1e-3 evaluates
outside the local linear regime of this weak objective; its failure is
not evidence of an incorrect FDTD derivative. The f32 AD value differs
from the finest FD by about 2.6%, also inside the unchanged 3% gate.

This does not justify merely reducing h to make the old fixture pass.
Its returned `reliable` mask is false at the highest frequency for both
ports, while its objective averages that bin anyway. The old implicit
GaussianPulse(f0=F_MAX/2) falls below the existing 10%-of-median excitation
screen there. A setup-only JVP/FD diagnostic also found the material and
source-waveform derivatives consistent (epsilon 1.1e-13 relative, source
waveforms 3.8e-9 relative, sigma derivatives both exactly zero). No
source-parameter derivative omission was identified by that check.

## Separate geometry debt

The legacy AD fixture declares h_sub=254 um at dx=80 um but explicitly
retains its historical trace sheet at 320 um. The source span correction
and this compensation interact. Redraw onto an aligned physical board is
needed for a matching-geometry acceptance fixture, but changing geometry
before identifying the reference defect could hide the failing comparison.
The reference correction is therefore measured on the existing geometry
first. No production policy for mismatched declared/actual port heights
is inferred from this test; that PI question remains pending.

The repaired acceptance fixture now places the zero-thickness trace at
254 um on dx=254/3 um, and explicitly uses GaussianPulse(f0=F_MAX) to
excite the entire existing frequency band. Always-on wall and sampled
excitation checks passed before any solve. The live AD test now requires
all of its fixed objective bins to be reliable and both drives settled
below -40 dB before computing a gradient; it does not adaptively discard
bins. The original 3% gradient gate and h=1e-3 remain unchanged at this
checkpoint. A new live measurement is required to qualify the repaired
fixture; the legacy measurements above cannot qualify it.

## Repaired fixture measurement

Run 369367260436 on b4b0893e passed the repaired AD test without changing
h=1e-3, the eight frequencies, 20 periods, or the 3% derivative limit:

- g_ad32=1.122180e-3, g_fd64=1.121532e-3; relative difference about 0.058%.
- Every fixed objective bin passed the wave-split low-signal screen.
- Forward drive settling was -128.26 / -124.82 dB.
- The incident-wave system's per-bin condition number ranged 1.031–1.052.

Raw E-plane voltage/current records, incident/reflected waves, returned S,
and actual field initialization dtypes are retained. The run then passed
the unchanged NU patch gate (280 periods, 81 frequencies): max|S11|=0.9839,
in-band minimum|S11|=0.9112, and the original edge-fed resonance/liveness
checks passed with no truncation advisory. This NU case uses uniform-valued
profiles to exercise the NU runner; it does not validate arbitrary grading.

These results qualify the repaired AD comparison and existing consumer
gates. They do not establish absolute RF calibration of every MSL port or
close #726/#715. General production validation of declared versus actual
port heights and the full CI checks still remain for #729.
