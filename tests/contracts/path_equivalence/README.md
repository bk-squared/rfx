# S0 path × feature equivalence

This matrix is generated from `_ADMITTED_ON` and `ADMITS`, checked against the
path-disposition table. Every requested admitted pair has an explicit builder;
a new row without one fails generation. ADI and subgridded cases exercise only
refusal gates. Other refusal cells also invoke the real entry point with a
first-kernel-scan tripwire. Selector inputs are routed explicitly to the lane
whose refusal is being checked, after the real dispatch validation.

The base is a dielectric block in an asymmetric CPML box, with one field soft
source and two probes off the symmetry planes. `dx = 1/1024 m`; declared box
lengths are `(10.3, 7.2, 6.4) * dx`. The x-low/high absorber depths are 1/2
cells. The block corners are `(3.2, 2.1, 1.3) * dx` and
`(6.4, 4.3, 3.2) * dx`. A feature overrides the setting it exercises. The wire
builder uses a passive wire termination, retaining the base soft source as the
only drive. The MSL builder adds its required ground and trace sheets. A constructor's rejection of a feature/base combination is a
finding; the source is not removed to make that combination run.

Uniform-valued NU profiles use the uniform grid's cell counts, including its
round-up of fractional domain lengths. Graded pairs use identical profiles
with two adjacent interior widths of `0.9 * dx` and `1.1 * dx`. The uniform
lane keeps its computed `dt`; NU receives that exact value. The graded pair
uses its first lane's computed `dt` on both sides. The realized and returned
`dt`, nodes, complete geometry record, and consumed material arrays are checked
exactly. The geometry lane tag is retained, so its literal difference is also
visible in the findings.

`comparison.py` owns both numerical bars: 9 float32 ULP at the larger array
peak for per-step records, and `1e-4` of that peak for accumulated records.
Missing records and missing channels fail. Port time samples are compared
channel by channel; raw port DFTs are compared by physical channel, rather than
by their lane-specific metadata container type. The complete geometry record
already checks the realized port edges.

Material arrays are observed through `_realized.capture` at the H consumption
site, including the eps/sigma arrays in its actual `MaterialArrays`. This also
works on dispersive paths which have no electric-material capture hook.
Distributed owned rows are reconstructed from the electric hook's ownership
metadata and matching H material records. Only the optional source-normalization
capture is disabled: production source specifications and operations are
unchanged. Forward cells use a scalar design-permittivity multiplier and JVP
with tangent 1, giving its exact scalar derivative along with the objective.
The current-moment row uses the supported bounded design-box override inside its monitor slab; other forward rows use the whole-grid override. The objective is the sum of squared probe samples. No test enables JAX x64.

All solves run in a bounded subprocess which sets
`XLA_FLAGS=--xla_force_host_platform_device_count=2` before importing JAX and
selects CPU. Identical declarations are cached within that worker; the timing
table reports the actual ordered cell times, including cache hits. Float32
solver records are materialized before timing ends. Record lengths are 12 and
36 steps for DFT planes, flux, and wire rows; other cells use 12 steps.

A cell is equivalent only when **all** its record checks pass. Separate checks
for realized prerequisites, probes, port samples/DFTs, observers, objective,
and gradient retain diagnostics even when another record is already a finding.
Known failed checks are `xfail(strict=True, raises=KnownFinding)` with an
`S0 finding:` reason. An unrecognized failure raises a different exception and
cannot be absorbed by an existing xfail. For an already numerically failing
record, its measured relative discrepancy is a fingerprint: a change larger
than the same governance bar also fails review. This does not change the
cross-path bar or turn the known discrepancy into a pass. It makes m2 detectable
when the unmutated DFT cell already fails. Fixing the failed check gives strict
XPASS. A live comparison canary makes m1 fail the matrix itself.

The full matrix exceeded three minutes in the recorded CPU measurement and is
placed in the weekly lane. `generation.pr_subset` deterministically chooses
one equivalence cell per attribute family (the first element of the admission
row), every refusal cell, and every cell containing a strict expected failure.
Every record check of each selected cell runs in the PR subset. Remaining cells
have `slow` and `s0_weekly` markers. The weekly workflow includes them through
its existing marker expression. The no-code PR contract job uses
`RFX_S0_PR_SUBSET=1` because that job intentionally overrides the normal `slow`
filter; its other contract tests retain their original selection.

Run the normal PR subset with:

```sh
python -m pytest tests/contracts/path_equivalence -q
```

Run the complete matrix with:

```sh
python -m pytest tests/contracts/path_equivalence -q -o addopts="" -m "not gpu"
```

The worker writes `measurements.json` and `stderr.txt` in pytest's temporary
`s0-worker` directory. `findings.json` is the reviewed expected-failure manifest,
not an automatically updated snapshot. `FINDINGS.md`, `TIMINGS.md`,
`evidence.json`, and `MUTATIONS.md` contain this run's evidence. Issue numbers are
left for the leader. No GPU customer, remote action, or production fix belongs
to this S0 change.

Conclusions: 리더가 채움.
