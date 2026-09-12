# MSL current at the voltage reference plane

Status: implementation and validation in progress. The parent directory's
RF records and manifest describe earlier source revisions and remain unchanged.

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

Validation so far: 143 local manufactured-field, geometric, primitive,
signal-mask and wave-solve cases pass. These check wiring, arithmetic and
AD propagation, not RF accuracy. Existing regression tests, real FDTD/AD
consumer gates and before/after RF interpretation remain in progress.
