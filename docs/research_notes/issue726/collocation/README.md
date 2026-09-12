# MSL current at the voltage reference plane

This record covers the current-plane correction and its bounded validation.
The parent directory's RF records and manifest describe earlier source
revisions and remain unchanged.

The voltage reader integrates Ez at propagation-axis E node `i`. Both
transverse H components at array index `i` live at the next cell centre.
The old reader therefore combined voltage and current separated by half
a cell on a uniform mesh. The existing temporal half-step correction does
not remove that spatial separation.

Keep the voltage reference plane. Read H at cell centres `i-1` and `i`,
then interpolate each component before applying the existing temporal
phase correction and transverse Ampere contour. With distances `d_left`
and `d_right` from the E node, the weights are
`(d_right, d_left) / (d_left + d_right)`. Uniform cells give equal weights;
graded cells generally do not. Port direction changes the contour sign,
not which physical H nodes bracket the E node. Registration coordinates
still select array indices and are recorded separately from physical H
coordinates.

This operation reproduces an affine H field exactly and preserves both
traveling directions. For a matched single mode on a uniform mesh,
the stagger-only apparent reflection is first order in cell width;
the centered average leaves a second-order error. This is a discretization
improvement, not an exact full-wave port calibration. In particular, the
old spatial offset alone need not produce excess reflected power at a
perfect reflector. The earlier observed power excess remains a separate
measurement to diagnose.

The standalone MSL driver uses the same stencil on run and differentiable
forward paths, for x/y directions and uniform/nonuniform grids. The mixed
driver and the diagnostic plane-probe helper retain their existing
uniform, x-directed support envelope. No missing H sample is synthesized
from a historical one-sided capture.

The separate `probe_clearance` result describes the existing downstream
bounding-box layout rule on actual sampled E nodes: `satisfied`,
`insufficient`, or `unavailable`. It uses the resolved auto ladder and
preserves a negative gap when a ladder has already crossed a candidate
reflector. It is not an accuracy certificate. `reliable` retains its
existing relative low-signal threshold and coverage policy.

Optional raw MSL dumps now retain both time-corrected side currents as
well as the interpolated current, each port's stencil, and the analytic
reference impedances actually used by the wave solve. This allows an
independent `B A^-1` comparison of both extraction choices on identical
FDTD fields, without using passivity projection to hide the difference.

Local validation includes 143 integrated cases, two mixed affine-field
falsifiers, 39 final stencil/preflight checks, and five updated structural/AD
tests. These overlapping groups check wiring and arithmetic. Historical
one-sided captures remain intact; the f64/f32 assembly checks now use complete
manufactured H records and an independently planted two-port S matrix.

Pinned GPU run [369367260604](gpu-369367260604/manifest.json), source
`ec3bedfe2fc1a8c8be12ebd2b7054562a5fcf096`, completed four sequential cases:

- x/y thru equivalence passed: max complex S difference `3.626e-6`.
- Actual material AD/FD passed: `g_AD=1.120153e-3`, `g_FD=1.119666e-3`,
  relative error about `0.000435` versus the unchanged `0.03` bound.
  The FD reference evolved float64 fields; its resolving power was
  `2.02e10` ULP versus the existing `1e4` floor.
- The existing NU patch gate passed. Its propagation cells are uniform;
  this is not evidence for unequal interpolation weights on real fields.
- The aligned coupon passed the existing raw physical screens and settled
  below `-102 dB`, but its old-algorithm golden lock failed at max absolute
  difference `0.002851`. Same-field old-current reconstruction plus the same
  passivity projection returns within `1.784e-5` of that golden. Thus this is
  attributed to the authorized current-plane correction; the old tolerance
  remains the acceptance bound. The qualified replacement and retained
  historical golden are described below.

On the x-directed thru, changing only the current extraction on identical
fields moves complex S by at most `0.005674`. Maximum raw column power moves
from `1.000690` to `1.000682`; spatial interpolation alone does not resolve
that excess. The report preserves raw values and makes no calibration claim.
The full provider log and all records were backed up and checked before
the terminal run was deleted.

Run [369367260605](gpu-369367260605/manifest.json) completed the fixed-source
CV06b pair and one explicit graded-propagation diagnostic sequentially.
The [pair comparison](cv06b-current-impact.json) fixes the original control's
`3.77125 GHz` notch bin:

| Observation | Old current S11 (dB) | Centered current S11 (dB) |
|---|---:|---:|
| Control | +0.026895 | +0.026934 |
| Near reflector | +0.018065 | +0.018081 |

The near-minus-control difference is `-0.008854 dB` after interpolation
(`-0.008830 dB` before). For both arms all saved voltage phasors and the
same-index side current are bit-identical to the earlier record. An
independent old-current wave solve reproduces the old S within `2.3e-7`.
This isolates the extraction change; it does not identify an absolute
full-wave error. Both records settle below `-118 dB`, while their existing
relative low-signal mask still flags the notch. The producer retains
`not_read`, and the raw power excess persists: maxima `1.006490` (control)
and `1.007856` (near). No passivity projection was used.

[Build-only clearance reconstruction](measured-pair-clearance.json) reports
control `(satisfied, satisfied)` and near `(insufficient, satisfied)` on
these same inputs. Distances use registered conductor bounds, as the
diagnosis documents, and are distinct from the fixture's realized PEC
front certificate. Neither status changes the low-signal mask.

The [graded case](graded-current-impact.json) changes four x cells around
each first voltage plane to `55/45/55/45 um`, keeping the board drawing and
voltage planes fixed. Actual H weights are `.55/.45`; all existing coupon
physical screens pass and both drives settle below `-102 dB`. Its
unprojected current-plane S change is at most `0.003122`. This is one
specific grading diagnostic, not broader in-plane grading qualification.
Its NumPy-generated frequency vector differs by 1--2 float32 ULP from the
JAX-generated base coupon; it is not a bitwise same-frequency mesh
comparison. The separate confirmation/refinement freezes the actual base
frequency vector.

Run [369367260608](gpu-369367260608/manifest.json) completed independent
confirmation and 2x refinement sequentially on the same pinned source.
Both pass the existing physical screens. Confirmation is bit-identical
to base604 in raw and projected S. The [offline adjudication](coupon-qualification.json)
reads pre-projection S directly from the dumps, verifies unchanged actual
voltage reference planes and frequencies, and measures a maximum raw-complex
mesh difference of `0.007077362` against the unchanged `.02` budget.
The fine-grid [independent wave solve](refinement-current-impact.json)
matches the saved raw S within `1.4e-7`.

The base604 projected S was transferred to the active coupon golden after
these checks, with `rtol=.005 / atol=.002` retained. Its previous values and
complete provenance remain in [pre-collocation-golden/](pre-collocation-golden/manifest.json).
The active manifest records the new source, records, hashes, diagnostics and
unchanged qualification criteria. Full provider logs and artifacts were
checked before deleting terminal runs605/608. CI and final review remain
separate merge requirements; these results do not certify arbitrary ports
or fix the separately observed raw power excess.

The residual [raw passivity readout](remaining-passivity.json) also checks
the largest singular value, not only individual drive-column power. These
two-port records share the same positive real reference impedance, so
`sigma_max(S)^2` is the maximum power gain over coherent input combinations.
CV06b gives `1.011293` at the control and `1.013087` near the reflector;
the maximum individual coefficient is `1.003106`. These are consistency
violations of the extracted S, not bounds on absolute RF error and not
proof that the field update itself produces energy. The user requested
continued investigation of the residual power/amplitude defect under #726.
That issue remains open after the current-plane correction is integrated.
