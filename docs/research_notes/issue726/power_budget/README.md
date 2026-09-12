# MSL power accounting after power-wave normalization

PR #987 normalizes unequal-reference MSL S in a consistent power metric.
This record addresses the separate equal-reference CV06b residual. No
calibration factor, passivity projection, or relaxed RF gate is applied.

Run 369367260633 uses source `d3f049d806ebea58778ee774cbf8794dd8ca39d7`
(the exact reviewed #987 head; merged main is `87927064`). It retains the
605 control's sources, loads, DUT, grid, 161 frequencies and 100-period
record. Two drives execute sequentially. Extra passive DFT observers record
six x stations and the boundary of a box enclosing the DUT, at five frozen
frequency bins. Every H component is saved on both real neighbouring
normal planes; the bottom PEC boundary needs no fabricated H sample.

[Recording impact](recording-impact.json) finds the new full raw V, I and S
**bit-identical** to the earlier 605 control. Each drive evolves float32
fields for 117998 steps and saves complex64 DFT arrays. Settling is
`[-122.15785575, -120.00605474] dB` at the existing witnesses. The
[manifest](gpu-369367260633/manifest.json) records all files and hashes;
the complete provider log and artifacts were verified before deleting the
completed run. The source archive remains at the manifest's NFS location.

## Independent face accounting

[The NumPy reader](audit-field-power.py) applies the existing `+omega*dt/2`
H temporal correction, averages the two normal H samples, then integrates
each Yee cross-product on its own transverse lattice. For an x-normal
node-bounded face, Ey*conj(Hz) uses whole y cells and half weights on the
two endpoint z nodes; Ez*conj(Hy) uses the complementary rule. It retains
complex cross-drive terms before taking the Hermitian part. The three-axis
affine-surface and complex coherent-excitation self-test passes.

The resulting drive Gram G satisfies `c.H @ G @ c = outward flux` for a
complex drive combination c. For comparison it is transformed by
`A^-H G A^-1`, using the original V/I incident-wave matrix A. This is an
invertible congruence and preserves inertia; its normalization is **not an
independent calibration of incident electromagnetic power**. Raw DFT
products are not labelled watts. In the JSON, individual face Grams point
in the positive coordinate direction; outward signs enter the closed sum.

[The full readout](field-power-audit.json) gives closed-face normalized
eigenvalue magnitudes at most `2.9141e-5` across the five bins. The native
two-port V/I inward Gram has eigenvalues down to `-0.01129285`. Native V/I
reconstructed from the full face records agrees with production within
`1.1e-7` relative error. The initial analysis mistakenly followed the
voltage helper's stale minus-sign docstring; independent review caught
this before interpretation. Production uses positive sum(Ez), paired with
negative contour current for the positive-going port. Neither numerical
sign was changed.

At `3.653125120 GHz`, the coherent combination with the largest apparent
V/I power excess gives the following [decomposition](coherent-flow-decomposition.json)
in the original V/I input basis:

| Contribution | Signed value |
|---|---:|
| Apparent two-port outgoing minus incoming power | +0.01129285 |
| Outward flux through both x faces | +0.01221128 |
| Outward flux through the remaining faces | -0.01222960 |
| Closed-face outward sum | -0.00001832 |

Thus the excess seen by the two-port representation is largely accompanied
by inward electromagnetic flux through its unrepresented lateral faces.
The data do not support attributing the 1.13% apparent gain to field energy
creation. Finite-window endpoint terms and numerical cancellation are not
bounded globally here; small closed-face residuals are not exact zero.
The origin of lateral inflow (source, finite leads, or boundary response)
is still unresolved.

The two drives' transverse E profiles cease to be proportional near the
notch. This is a diagnostic, with the small transmitted-drive norms retained
alongside it; no noise-qualified modal verdict follows from coherence alone.
High coherence also cannot certify a single mode or accurate V/I power.

## Bounded follow-up

[Actual periodic Yee evolution](tem-dft-clock.json) of both traveling
directions and a standing wave confirms the temporal correction's sign:
maximum relative discrepancy is `2.03e-15` in float64 and
[`8.28e-7` in float32](tem-dft-clock-f32.json). These are integer-period
vacuum tests, not certifications of arbitrary finite DFT windows or MSL RF.

The original fixture comment says its trace continues through CPML, but
its realized trace has no longitudinal PEC edges in either x pad. An
[explicit continuation build](explicit-lead-build.json) extends the trace
from the first to last allocated x node while retaining the exact DUT-box
PEC masks and source declarations. This is a build-only candidate for the
next causal comparison; it has no measured power result yet. It does not
authorize silently extending arbitrary user geometry.

Reproduce from the repository root:

    python docs/research_notes/issue726/power_budget/audit-field-power.py --self-test
    python docs/research_notes/issue726/power_budget/audit-field-power.py --root docs/research_notes/issue726/power_budget/gpu-369367260633/artifacts --out /tmp/new-power-audit.json

#726 remains open. This record identifies missing power channels in the
measurement and a testable lead-continuation defect, not a completed port
calibration or a broad support promotion.
