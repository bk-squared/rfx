# S0 path × feature equivalence

Generated from `_ADMITTED_ON` / `ADMITS` and checked against path disposition.
Every equivalence builder is constructed for both lanes during generation;
each lane's admission detector must report the named row active. Refusal cells
exercise the admission gate, and in-scope pairs also enter the public runner
with a first-scan tripwire. Distributed TFSF/waveguide cells test the explicit
lane admission gate: their public API intentionally falls back to one device,
so forcing that public dispatcher is not a refusal test.

The base is an asymmetric CPML box, dielectric block, soft source and two
probes. `dx = 1/1024 m`; box lengths are `(10.3, 7.2, 6.4) * dx`, x-low/high
absorbers are 1/2 cells, block corners are `(3.2, 2.1, 1.3) * dx` and
`(6.4, 4.3, 3.2) * dx`. A driven port, waveguide or TFSF row supplies its own
excitation instead of the base soft source. `add_source()` internally occupies
a zero-impedance `_PortEntry`; including that entry with these features would
be a builder defect. Passive terminations retain the soft source. MSL adds its
required ground and trace sheets.
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

`comparison.py` owns the bars: 9 float32 ULP at the larger peak per step;
`1e-4` of that peak for accumulated records, objective and gradient. Forward
cells use JVP of a design-permittivity multiplier, with squared probe samples
as the objective. Current moments use their supported bounded design box.
Missing shared channels fail. A comparison canary detects a disabled helper.
Known numeric discrepancies retain a same-bar fingerprint so a further change
cannot hide under xfail. Unknown failures are never accepted by an xfail.

The worker subprocess selects CPU and two host devices before importing JAX;
it is bounded and reaped by the fixture. No test enables x64. Identical solves
are cached within a run. DFT, flux and wire use 12 and 36 steps; others use 12.
The PR selection uses the cheapest measured passing 12-step candidate per
admission attribute family, plus all refusal cells. Cost is the sum of the two
uncached solve times (retained when a solve is reused), not a cache-hit cell
timer. Expected failures and remaining equivalence cells run weekly. The
explicit weekly-only xfail rule excludes `_mode`, `_dft_planes`, and
`_flux_monitors` from PR equivalence selection: these families have no fully
passing cell. All their cells still run in the full weekly matrix.
The no-code PR contract job sets `RFX_S0_PR_SUBSET=1` to retain this selection
when overriding pytest's normal slow filter.

Run PR tests: `python -m pytest tests/contracts/path_equivalence -q`.
Run full tests: `python -m pytest tests/contracts/path_equivalence -q -o addopts="" -m "not gpu"`.
The fixture writes raw measurements in its pytest temporary directory.
`findings.json` is a reviewed test manifest with one cause key per finding.
Measurement records, per-cell timings and DFT step dumps are in untracked
`.s0-work/`, for the leader to move to rfx-archive. See FINDINGS.md and MUTATIONS.md
for short review summaries. Issue numbers and conclusions: 리더가 채움.
