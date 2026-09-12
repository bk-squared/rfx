# cv06b falsifier input and diagnostic checkpoint

Base main: `0662c3b96d82ca65d7bd041b94c4ab984230e00d` (2026-09-11).
**Issue #953 remains open.** The original checkpoint fixed input geometry and
a misleading preflight interpretation. The subsequent reporting repair below
implements the user's G1-scope decision. Fresh RF qualification remains
unfinished; no field solve was run for either packet.

## Observed input and correction

The original `W_STUB=5*DX` declaration has off-node faces. Under #931's closed
sheet footprint it contains five nodes, hence four geometric intervals:
254.0 um rather than the declared 317.5 um. Its physical centre is also
31.75 um (half a cell) away from the baseline centre. This follows the
specified sheet rule; the diagnostic's on-lattice drawing claim was wrong.

The corrected arm takes actual node coordinates from the completed baseline
grid, after port/boundary registration. It selects endpoints five intervals
apart with the same physical centre and passes their coordinates to the new
builder. Six longitudinal-current node columns then span 317.5 um. The trace,
normal plane and stub length stay fixed. The separate one-cell-length arm
retains the baseline width and centre and shortens its realized stub by one DX.
An incompatible centre/interval parity is refused rather than shifting it.

All three conductor geometries are checked before the first field solve.
Full output records retain their measured geometry. The existing committed
2026-09-07 narrow-arm spectrum still belongs to the older four-interval arm;
it was preserved and is not a measurement of the corrected input.

`production-inputs.json` is a build-only certificate of the current producer.
Run `input_certificate.py` from this checkout with its `rfx` on PYTHONPATH to
inspect a new build. The certificate disables both field-solve entry points.
The `before-*` files and original probe sources preserve the earlier main
inspection (the case source was identical at 9369e0ae and 0662c3b9).

## Preflight interpretation

The generic preflight did warn about the narrow arm's off-node faces. However,
its nearest-node residual was 13.75 um while the actual width loss was 63.5 um.
The warning's claim that extent error is bounded by that residual was false.

Synthetic production-assembly controls demonstrate the same distinction:
a five-mm sheet loses one mm while its maximum nearest-node residual is
0.2 mm; a volume's extent grows by 0.6 mm with maximum face residual 0.3 mm.
The warning threshold/severity and observed residuals are unchanged. Its text
now distinguishes alignment from actual extent and avoids predicting a
frequency change without a mode/dimension model. Only seven live example
snapshot message fields changed; all other semantic values were checked equal.
Frozen measurement artifacts containing the older warning remain untouched.

## Reference and history still needing adjudication

The historical narrow-arm frequency comparison uses the main-line width in
its quarter-wave reference. Holding the recorded notch fixed at
3.915833569 GHz, substituting that arm's own row-pitch width changes the
arithmetic discrepancy from 6.4388% to 3.7506%. This does not establish physical
accuracy: the row-pitch electrical-width convention itself is unresolved,
and these are retained pre-#729 fields, not a new measurement. No frequency
gate or tolerance was changed by that original checkpoint. The later scope
decision is recorded below; this calculation remains historical evidence.

The original pre-#931 run was 369367257702; its provider log records a staging
copy and failed git revision lookup. Import commit 4bee84f3 is therefore not
an established producer SHA. A separate retained run 369367259002 identifies
d990e18c, reports a 322.6-second S-matrix execution, and has identical baseline
and narrow sweep arrays. It is a documented reference path, not a complete
source-manifest proof. Post run 369367259191 matches the imported fae08d10
artifacts, but its `commit.txt` is empty. These provenance gaps remain explicit.

The immutable d990 PEC code uses `cell_mask & (backward | forward)` and leaves
normal Ez live on one z layer. The present volume incidence rule shorts those
normal edges and both tangential faces. A current-main volume-versus-sheet
experiment would therefore not reproduce the historical A/B. The old ground
was already the domain `z_lo=PEC` boundary; the Box-to-sheet entries were trace
and stub. Current main also includes #729's source-span correction. Historical
frequency-shift attribution must respect these distinctions.

## Checks

- 15 cv06b build/model tests passed, including ignored bounds, centre movement,
  and parity conflict rejected before any solve.
- 46 preflight/rasterization tests passed.
- 206 example fidelity/safety tests passed.
- Restoring the old main advisory in memory makes both extent-counterexample
  tests fail; source files remain unchanged by that control.
- Changed diagnostic/core/test files pass Ruff, and the legacy case passes
  the repository's CI lint selection.

Hashes and commands are in `manifest.json`. This packet supports the input
and message repairs, not completion of #953 or a new RF accuracy claim.

## Approved frequency-reporting scope

The user selected baseline-only frequency accuracy, with the narrow arm
remaining a G2-fail/depth-pass control. The branch was rebased onto main
`87927064` before implementing that decision. Baseline G1 retains its 4%
limit and its existing main-line row-pitch/declared-extension reference.
Neither the baseline tolerance nor G2's ideal r=1 bandwidth target changes.

Perturbed arms no longer compute or persist a G1 boolean. Their frequency
comparison is under `frequency_diagnostic`, explicitly excluded from an
accuracy verdict. The one-cell arm retains the baseline width/length
conventions: its changed length still supplies the existing shift prediction
used by the visibility criterion. That reference is not merely display data.

The narrow arm uses its own realized node-span width and extension length in
a stated continuous quarter-wave approximation. Metadata records width,
length convention, substrate height, permittivity and effective permittivity.
Open-end and tee-junction corrections are absent. A geometric width is not a
certification of discrete electrical width. The old main-line reference and
its deviation are retained as a separately named diagnostic comparison.

New reports use schema version 2. Historical spectra and schema-1 summaries
remain unchanged; the producer refuses any existing per-arm or summary output
before building or solving. Use a new output directory. Structural
tests exercise the actual solve/report/evaluation wiring with only the field
result stubbed; these do not qualify the corrected five-interval RF response
or uniquely attribute the historical frequency shift.

The reporting checkpoint passed 58 focused tests. A fresh build certificate
matches all three earlier approved input records exactly. Commands, source
hashes and remaining limits are in [reporting-checkpoint.json](reporting-checkpoint.json),
with the current geometry in [reporting-inputs.json](reporting-inputs.json).
