# cv06b falsifier input and diagnostic checkpoint

Base main: `0662c3b96d82ca65d7bd041b94c4ab984230e00d` (2026-09-11).
**Current-main repair and controlled RF qualification are complete on this
branch; integration is pending.** The initial checkpoint fixed input geometry
and a misleading preflight interpretation. Later sections retain the reporting
decision, the669 control defect and the671 qualification. The earlier build-only
packets are historical and do not themselves claim an RF result.

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

## Sequential RF measurement declaration

`record_falsifiers.py` invokes the current producer on its three already
validated simulation instances. It preserves 100 frequencies and 20 periods
for every arm and returns each original MSL result unchanged. It adds the
ordinary raw V/I dump request and copies complex S, preprojection S when
present, references, fit/rail flags and settling metadata. No bins are removed
and no new passivity clipping is applied.

The [build declaration](rf-predeclaration.json) records dimensions and reference
conventions before fields run. The canonical `settling_verdict` is used without
changing its inclusive -40 dB limit; absent/nonfinite witnesses cannot qualify
an RF result. The producer's own verdict and this settling screen are recorded
separately. A successful producer exit alone is not an accuracy certificate.

`recorder_smoke_test.py` checks observation-only behavior with a synthetic
backend. Its [summary](recorder-smoke-summary.json) is explicitly synthetic:
array bytes are retained, an exact -40 dB witness passes, absent witnesses
are rejected, gains above one remain visible, and a nonfinite raw bin makes
the aggregate gain unavailable rather than being filtered out. These checks
do not measure the physical response of any of the three geometries.

## Run669 result and the remaining comparison-control gap

[Run369367260669](gpu-369367260669/manifest.json) completed all three arms
on source7011f78b. Its complete provider log, 20 retained files and source
archive were hash-verified before deleting the terminal provider run.
No port algorithm changed, and no additional campaign ran in parallel.

An [independent NumPy referee](independent-rf-669.py) reconstructs power-wave S
directly from measured V/I and recorded references using an explicit 2x2
inverse. It imports neither the production replay nor its spectral estimator.
The [report](independent-rf-669.json) reproduces the producer's spectral metrics
and matches saved raw S within 2.41e-7 maximum absolute difference. The raw
dump and saved result's preprojection S are byte-identical. Both returned
and reconstructed raw S retain the approved outcomes:

| Quantity | Returned S | Reconstructed raw S |
|---|---:|---:|
| Baseline G1 reference deviation | 2.166151% | 2.167659% |
| Narrow bandwidth ratio, lower gate limit 0.80 | 0.725836 | 0.723685 |
| Narrow sampled notch depth, witness limit -10 dB | -40.3864 dB | -40.0414 dB |
| One-cell refined frequency shift | 0.824843% | 0.822069% |

The existing one-cell prediction is 0.531982%; both shifts exceed its
unchanged half-prediction visibility threshold. All beta rail counts are
zero. The worst settling witness is -92.6437 dB, within the canonical screen.
Raw coherent power gain nevertheless remains 1.01126–1.01140: this is not
a power-calibrated MSL result. Notch bins carry `reliable=False` at both ports;
the bandwidth crossing brackets carry `True`. Retain those low-signal flags
without interpreting them as either a global S failure or an error bound.

The retained metadata exposes an additional control problem. The one-cell
arm has ny=279 instead of the baseline's280 because the builder sizes its
domain from stub length. The narrow arm automatically resolves probe0 to
indices100/451 instead of99/452 (offset61 instead of60). Thus669 verifies
the production builder's three outputs, but cannot attribute every difference
solely to the intended metal perturbation. It also cannot uniquely explain
the older frequency shift. The next repair must hold the baseline grid,
materials and resolved port/probe plan fixed across the controlled arms,
check that invariant before solving, and qualify that comparison before
declaring #953 complete. The669 raw data remain unchanged.

## Fixed-environment comparison repair

The [controlled input declaration](controlled-predeclaration.json) retains
the669 metal shapes and baseline G1/reference policy. All three arms now use
the baseline domain **and substrate carrier**, the same assembled background
arrays, and the baseline's resolved probe choices registered explicitly.
The common grid is553×280×37; offsets60, spacing16 and five probes give
probe0 indices99/452. `eps_r_sub=None` remains unchanged so the source still
infers the registered float32 substrate value. Auto placement is disabled on
perturbed arms and the resolved physical E/H plan is compared again.

The real engine entry receives an additional observation-only check during
qualification. [consumed_plan.py](consumed_plan.py) fingerprints actual
load-inclusive material arrays, SourceSpec waveform bytes, point/DFT indices,
crops, frequencies and dtypes, grid/operator settings and PEC outside the
baseline stub node box. It also records the full varying PEC hashes. Both
endpoints of a permitted tangential edge must lie in that box; normal Ez
changes are never admitted for the zero-thickness sheet. Each perturbed drive
must match its baseline drive before the original runner advances fields.

The scoped checker refuses unsupported physics rather than omitting it. It
does not compare geometry-dependent reflector gaps or the narrow diagnostic
reference as though they were fixed inputs. Nominal and consumed-plan
falsifiers catch domain/substrate divergence, retained auto flags, hidden
material changes, waveform/crop/frequency changes, CPML changes and PEC changes
outside the allowed stub. The combined79 tests pass. A first-drive dry build
of all three full-size arms also passed the real runner-input comparison,
with no field advancement. These checks preceded the controlled671 result below.

## Controlled qualification: run671

[Run369367260671](gpu-369367260671/manifest.json) completed all three arms
with the fixed environment. All six per-drive consumed input dictionaries and
their recomputed hashes match the corresponding baseline drive. Full PEC
changes are retained and restricted to the intended tangential stub edges.
Baseline frequencies, raw V/I, both current-side records and preprojection S
are byte-identical to669. This also checks that the added observation-only
audit preserved the original baseline computation.

The [independent reader](independent-rf-671.py) reconstructs S from V/I without
the production replay or estimator. Its [report](independent-rf-671.json)
preserves the approved predicates on returned, saved raw and reconstructed raw
S. On reconstructed raw S, baseline G1 is2.167659% (<4%); narrow bandwidth
ratio is0.723670 (<0.80, the intended failure), with sampled depth-40.0379dB
(<-10dB, the intended pass). The one-cell refined shift is0.821846%, above
the unchanged half-prediction visibility bar. The raw S reconstruction differs
from the saved matrix by at most3.20e-7. The source and recorder were unchanged
after staging; all21 files, the source archive and the complete provider log
were hash-verified before deleting the terminal run.

All beta rail counts are zero; the worst settling witness is-92.4678dB.
Notch low-signal flags remain false and bandwidth crossing flags remain true.
Raw coherent gain still reaches1.01140. These records qualify the stated
falsifier behavior on this fixture, not a globally calibrated MSL port or an
absolute error bound. The separate power problem remains tracked by#726.

The [historical assessment](history-assessment.md) confirms the old recorded
narrow feature moved by229.224MHz on the same stored frequency axis. Its
unique cause cannot be identified from the retained, incompletely controlled
producer pair. Claims that the unused G1 supplied additional accuracy coverage,
that modern finite-thickness removal reproduced the old operator, or that a
width convention uniquely explained the movement are retired. Original
spectra, summaries and the migration's falsified predictions remain intact.
No extra modern volume/sheet run can supply the missing historical provenance.

The final controls passed79 focused tests on both local JAX0.6.2 and
Python3.11/JAX0.10.2, with the full-size dry setup and recorder falsifiers
retained above. The controlled671 result supplies the missing RF qualification.
