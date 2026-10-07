# S0 path × feature equivalence

Generated from `_ADMITTED_ON` / `ADMITS` and checked against path disposition.
Every equivalence builder is constructed for both lanes during generation;
each lane's admission detector must report the named row active. Refusal cells
exercise the admission gate, and in-scope pairs also enter the public runner
with a first-scan tripwire. Refusal queries and admission-message construction
also run under a scan tripwire; the no-step assertion is outside strict-xfail
classification. Internal assembly errors are retained as findings, not accepted
as feature-naming refusals. Distributed TFSF/waveguide cells test the explicit
lane admission gate: their public API intentionally falls back to one device,
so forcing that public dispatcher is not a refusal test.

The base is an asymmetric CPML box, dielectric block, soft source and two
probes. `dx = 1/1024 m`; box lengths are `(10.3, 7.2, 6.4) * dx`, x-low/high
absorbers are 1/2 cells, block corners are `(3.2, 2.1, 1.3) * dx` and
`(6.4, 4.3, 3.2) * dx`. A driven port, waveguide or TFSF row supplies its own
excitation instead of the base soft source. `add_source()` internally occupies
a zero-impedance `_PortEntry`; including that entry with these features would
be a builder defect. Passive terminations retain the soft source. MSL adds its
required ground and trace sheets. The `_solver` row uses ADI with PEC boundaries:
ADI refuses absorbing boundaries at construction, so CPML would be a builder
conflict. This row only exercises refusal; it produces no lossless field record.
TFSF uses margin 1 so its boundary and adjacent cells are vacuum. Waveguide
walls are explicit PEC; transverse modal dimensions are exactly `8 * dx` and
`7 * dx` on both paths. Its x length remains fractional and the block remains
off-grid. This avoids comparing different declared modal apertures on the same
rounded grid.

Uniform/NU use identical node coordinates and constant profiles; graded pairs
share profiles with two interior widths `0.9 * dx`, `1.1 * dx`. Uniform keeps
its computed dt and NU pins that exact value. Every cell checks exact dt first.
Geometry equality covers realized nodes, cell sizes, ranges, bounds, masks,
and materials. Lane labels, declared fields and declaration-derived residuals
are excluded. The declared-length record semantics are documented once in
FINDINGS.md. Kernel material views come from `_realized.capture`; E-side
lumped terms are read at E, not from H's intentionally smaller container.

`comparison.py` owns the bars: 9 float32 ULP at the field peak per step;
`1e-4` of that peak for accumulated vector fields. Scalar quantities, objective
and gradient keep their own peak. Forward
cells use JVP of a design-permittivity multiplier, with squared probe samples
as the objective. Current moments use their supported bounded design box.
Named E components share the largest E magnitude on either path; H components
share a separate H peak by default. Flux tangential components follow the same
rule. The caller opts in by passing the scene's impedance range, for example
`_tree(..., paired_impedance=wave_impedance_range(10))` for nonmagnetic media
with relative permittivity from 1 to 10. The helper returns
`(Z0 / sqrt(eps_r_max), Z0)` using `Z0 = sqrt(MU_0 / EPS_0)` from the solver's
constants (approximately 376.730313668 ohm). With `(z_low, z_high)`, the peaks
are `max(E, z_low * H)` and `max(H, E / z_high)`, in SI units. A propagating
wave with impedance inside those bounds retains both own peaks. A single
positive finite number Z uses Z in both directions; using vacuum Z alone
for an eps_r = 10 wave raises the E peak by sqrt(10). There is no default
vacuum impedance. Bounds must be positive, finite, ordered numbers, not bools.
An absent partner or a nonfinite converted partner retains the own peak.
NTFF faces are split into E/H slots with peaks across the six faces, and use
the same option. The matrix runner does not opt any cell in.

In the prompting case, the step bar is 9 ULP of E/Z0, approximately
7.9e-7 of E/Z0: 22% of H's own peak when H sits 109 dB below E/Z0.
The accumulated bar is 1e-4 of E/Z0, or 28 times H's own peak there.
This option is for bit-level trace checks, not for judging H accuracy.
Different observers and different physical quantities (e.g. V and I) are not
merged.
Observer configuration and geometry (`OBSERVER_METADATA`,
including frequencies, indices, windows and area weights) remain exact.
The named `NTFF_KAHAN_RESIDUALS` list excludes `c_x_lo`, `c_x_hi`,
`c_y_lo`, `c_y_hi`, `c_z_lo`, `c_z_hi`: these are internal Kahan carries, never
read by the far-field transform (`rfx/farfield.py:127`).
Missing shared channels fail. Live canaries detect a disabled `compare`,
`_comparison`, or `_tree`, including traversal through a dict containing a list.
Known numeric discrepancies retain a same-bar fingerprint so a further change
cannot hide under xfail. Unknown failures are never accepted by an xfail.

The worker subprocess selects CPU and two host devices before importing JAX;
it is bounded and reaped by the fixture. No test enables x64. Identical solves
are cached within a run. DFT, flux and wire use 12 and 36 steps; others use 12.
The PR selection includes refusal cells on the main-path lanes only, the cheapest measured 12-step
strict-xfail cell per finding cause, and the cheapest measured passing 12-step
candidate per admission attribute family. A cell may witness several causes.
`PR_FINDING_CHOICES` declares the cause witnesses; generation rejects an
uncovered cause or a stale/non-12-step witness. Costs use uncached solve times
(or elapsed execution time when execution fails before both solves finish).
Main-path refusal lanes are `run_uniform`, `fwd_uniform`, `run_nonuniform`,
`fwd_nonuniform`, `run_distributed`, and `fwd_distributed_nu`. ADI and subgridded
refusals run weekly unless chosen as a strict finding witness. All remaining cause
witnesses remain in the PR subset.
The PR wall-time budget is 240 seconds, judged by the VESSL gate contract step
and PR CI shard on comparable load. Shared-Mac wall time is not a budget gate.
The remaining equivalence cells run weekly. With no environment variable the
matrix runs this PR subset, including when a gate overrides pytest's slow filter.
Only `RFX_S0_FULL=1` enables the full matrix; the weekly slow job sets it.

The six base families `_freq_max`, `_domain`, `_dx`, `_boundary`, `_cpml_layers`,
and `_probes` share one cached base solve per lane/grid/dt/record length via
`BASE_ROWS` (also shared with its material/source/profile aliases).
`_adi_cfl_factor` is a no-op row on the Yee lanes: its self-comparison is
expected and does not provide ADI coverage.

Run PR tests: `python -m pytest tests/contracts/path_equivalence -q`.
Run full tests: `RFX_S0_FULL=1 python -m pytest tests/contracts/path_equivalence -q -o addopts="" -m "not gpu"`.
The fixture writes raw measurements in its pytest temporary directory.
`findings.json` stores each fingerprint and witness variant once in `causes`,
alongside implementation file:line references. Optional cell/record scopes on
a fingerprint preserve which failures each cell accepts. `cells` maps records
to cause IDs, with a witness-variant reference only when it differs from the
default. The loader expands this into the existing strict-finding checks.
Measurement records, per-cell timings and DFT step dumps are in untracked
`.s0-work/`, for the leader to move to rfx-archive. See FINDINGS.md and MUTATIONS.md
for short review summaries. Conclusions: in the S0 PR body (leader)
