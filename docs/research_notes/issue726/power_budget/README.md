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
PEC masks and source declarations. This predeclared build candidate was
subsequently measured in647, reported below. It does not authorize silently
extending arbitrary user geometry.

Reproduce from the repository root:

    python docs/research_notes/issue726/power_budget/audit-field-power.py --self-test
    python docs/research_notes/issue726/power_budget/audit-field-power.py --root docs/research_notes/issue726/power_budget/gpu-369367260633/artifacts --out /tmp/new-power-audit.json

#726 remains open. This record identifies unrepresented power channels in
the measurement. The continuation comparison below did not establish a
complete power fix or a broader calibrated support range.

An [independent raw-array referee](independent-field-budget.json) reproduces
the flux budget by directly superposing complex fields and recovering the
Gram by polarization, without using the reader above. At 3.653125120 GHz
the upper z face contributes -0.01284067; both y faces together contribute
+0.00061107 in the same original V/I input basis. The [next recorder](record_field_power.py)
accepts an explicit lead-continuation switch and checks unchanged materials,
source declarations, and PEC across the complete source-to-source interval
before stepping. Its [build receipt](explicit-lead-record-plan.json)
records the exact changed geometry.


## Continued-lead result and subsequent code audit

Run [369367260647](gpu-369367260647/manifest.json) completed the single
predeclared continuation comparison on main87927064, whose Git tree is
identical to633's d3f049d8. Software and precision were checked against633
before stepping. The [provenance note](continued-lead-provenance-note.json)
corrects inherited scope labels without modifying the raw producer plan.

The [comparison](continued-lead-comparison.json) is not grounds to adopt
continuation as the power fix. At the original worst-notch coherent drive,
apparent excess decreases from0.01129285 to0.00389477 in the same original
input basis. Across the full 161-bin S sweep, however, maximum coherent gain
increases from 1.01129281 to 1.01953449 near 6.803125GHz. The existing settling
screen passes(-65.28/-64.16dB versus its unchanged-40dB threshold), but is
much weaker than633. The selected-bin closed-face Gram residual reaches
0.00237806 in the continued run's own V/I input basis; there is no global
finite-window bound. Full raw records/logs were hash-checked before deleting
the completed provider. No continuation change has been applied to the
production fixture.

Code review then followed actual source/loading construction into the
scalar-conductivity electric update and post-update source addition. It
found no negative dissipation coefficient or duplicate load in this case.
The [electric-substep audit](source-work-audit.json) calls real loading,
source construction and update_e with prescribed fields; midpoint source
work equals electric storage change plus full ohmic loss to 2.9e-16 relative.
This is a local algebra check, not a Maxwell eigenmode or RF benchmark.

For a supplied Ez profile e and N=sum(volume*e^2), the code chooses
sigma_port=1/(R*N), adds source force e*u, and damps all three E components
with scalar sigma. Its power-conjugate variables are
V_proj=sum(volume*e*Ez)/N and I_source=N*u; source work uses midpoint E.
The waveform u is not itself a calibrated Thevenin voltage: U=R*N*u.
For fields beyond the supplied Ez profile, load loss includes additional
nonnegative terms and is not generally V_centre^2/R. These facts alone do
not explain S gain; drive-column scaling cancels from B A^-1.

[Actual support inspection](source-support-audit.json) finds 72 Laplace
source/load cells per port.32 cells in8 lateral fringe columns do not end
on the trace; they carry11.58988% of the supplied profile's weighted e^2
norm. This is not an RF power fraction or an automatic invalidity verdict
for an impressed source. All 72 cells are interior, so direct source
injection into CPML is refuted for this fixture. Every profile-norm term is
included by actual source/load loops; there is no normalization truncation.

The subsequent667 field observation follows this
source-work contract and distinguishes a source-model power port
from the first-probe-plane V/I observable. A passive source-model result
must not be substituted for the intended MSL S or used as its calibration.


The [source-work recorder](record_source_work.py) adds only source-cell
observers to the unchanged633 model:144 ordinary Ez point probes plus six
cropped source E DFTs. A [real runner-entry interception](source-work-wiring.json)
checks all point indices, actual added sigma and 72 active SourceSpecs before
any field step. Actual source increments agree with independently reconstructed
Cb*e*u within 2.22e-7 relative peak error; no exact-float32 claim is made.
The [midpoint diagnostic](source_work_observable.py) passes independent
prescribed passive and active response tests with unequal references and
orthogonal field components. A separate review verified time levels, shunt
power decomposition and absence of a fitted normalization or passivity clip.
Its response remains a distinct source-model observable, not a replacement
for the intended MSL S. Streaming E DFTs are not silently equated to the
independent midpoint clock.


## Source-work observation result

Run [369367260667](gpu-369367260667/manifest.json) completed both unchanged
model drives. The [impact check](source-observer-impact.json) finds all
existing MSL raw V/I/S shape, dtype and bytes identical to633. The
[readout](source-work-readout.json) and [independent reconstruction](independent-source-work.json)
agree on the source-work S to 4.44e-14 maximum absolute difference.

| Sampled 161-bin response | Maximum coherent gain | Maximum complex reciprocity difference |
|---|---:|---:|
| Source-work network |0.9051584041|3.2742e-7|
| Existing first-plane MSL |1.0112928110|5.1330e-4|

The source-work matrix therefore lies within the passive bound for these
two fixed channels and sampled frequencies. It is a different observable,
with more than the intended first-plane DUT in its network, and cannot be
substituted for that MSL matrix. It does not exonerate every CPML mode or
identify a unique origin of the lateral inflow. See the [method note](source-work-method.md)
for settling, clocks, reference construction and remaining uncertainty.
The full provider log and all artifacts were hash-checked before deleting
the completed run. The next design-scope choice is with the user; public
port behavior and existing autodiff remain unchanged.


A [same-Yee mode residual evaluator](yee-mode-residual.py) supplies a common
read-only check for future mode candidates. It takes one actual real Yee
update of the real/imaginary quadratures on three propagation slices and
checks only the central slice. Known vacuum/PEC plate modes, wrong-stagger
falsifiers and timestep-scaling checks are [retained](yee-mode-residual.json).
This is not an eigensolver or source change. CPML candidates are explicitly
refused until harmonic auxiliary states are provided; zero memory would
confuse startup transients with mode error. It does not change port AD.

The [calculation contract audit](code_contract_audit.md) also distinguishes
the fixed launch shape from the material-dependent electric source increment.
Always-on source-work tests cover the actual load/source/electric substep and
its JIT material gradient in both float32 and float64. Corrected source
docstrings now state the implemented conductivity and forcing equations;
no source amplitude, termination or production numerical operation changed.

The first PR988 CI run exposed a test-only context-manager incompatibility:
the repository's newer-JAX fallback did not accept the Boolean precision
argument. The test now uses the same public-JAX/experimental fallback import
as the existing power-normalization tests. All 16 cases pass with local
JAX0.6.2 and with Python3.11/JAX0.10.2, the JAX version resolved by the failing
CI job. The [compatibility record](jax-compatibility.json) retains the failure
and validation scope; no numerical assertion or production operation changed.
