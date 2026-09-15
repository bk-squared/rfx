# Repair the physical premise before comparing historical outputs

Historical repair/handoff record. Live follow-up and superseding dispositions:
[T10 fixture qualification](T10-fixture-live-qualification.md).

Source base: `949e1845b84b26826d4e5309363bea4c2ffa5920`;
worktree `rfx-931-fastlane-reds`, branch `feat/931-fixture-repair`.
The PI explicitly authorized fixture repair, keeping #931 and the corrected
MSL board. No other worktree was read or modified. No VESSL run was launched.

The common class is **a historical assertion whose declared physical problem
and actual measurement do not agree**. Ownership changes exposed both geometry
that the old rule had accidentally absorbed and instruments/comparisons that
were already invalid. The repair order is drawing, realized mesh, observable,
qualification, and only then a numerical snapshot. Restoring old ownership or
surrounding a new answer with a wider interval would not repair this class.

R1 basis: the normative #931 note requires both dimensions and placement on
nodes; `T7-fixture-purpose-and-engineering-review.md` says “Comparing a corrected
live board against an old-board output no longer tests same-problem equivalence.”
`T7-slow-interaction-stop.md` records the former authorization stop, superseded
by this PI decision. The observed far-field and short confounders in
`slow-consumer-evidence/` motivate the controls below. Repo-local private memory
files remain absent; no other checkout was opened to retrieve them.

## Repairs and retained gates

| Fixture | Fixed physical drawing and measurement | Mesh and acceptance |
|---|---|---|
| V173A composition successor | 28 x 36 mm nominal 35 um foil, represented by an explicit PEC sheet at the 1.5 mm, eps_r=4.3 substrate interface; 50 x 50 x 20 mm computational domain, PEC ground at z=0, laminate continues into lateral absorber. Source (25,18,0.5) mm and probe (25,18,1) mm measure live Ez gap edges. Explicit differentiated-Gaussian current source, 1 A amplitude convention. | dx=0.5 mm gives three substrate cells and exact patch/laminate faces. 6000 steps preserve the old physical record duration. Eight CPML cells now mean 4 mm, versus the old 8 mm. The scalar drift tolerances remain frequency rtol=1e-6 and spectrum atol=0.1 dB. |
| MSL | Keep authorized h=254 um, W=600 um, launch-plane separation=10 mm and lossless eps_r=3.66. Domain 14 x 3.4 x 1.978 mm, trace edges y=1.4/2.0 mm, ground z=0, sheet z=254 um; ports x=2/12 mm. Five physical measurement planes x=5,6,7,8,9 mm, reversed for the opposite drive; S is referenced to x=5/9 mm (4 mm span). | dx=dy=50 um. Substrate dz=(50,50,38.5,38.5,38.5,38.5) um, six cells summing to 254 um. Full dz is each cell of ([100]+14*[77]+8*[100]) um bisected. Sixteen 50 um CPML cells preserve 0.8 mm thickness; 16 uniform interior z-hi cells and 600 um lateral clearance after the CPML buffer. Refinement bisects all cells without moving faces, probes or boundaries. Original complex rtol=.005/atol=.002 unchanged; original golden unchanged pending qualification. |
| Graded far-field | Same 1.5 mm PEC cube at 10.25--11.75 mm, same current source (11,11,9.5) mm and NTFF faces 4.5/17.5 mm in 22 mm cube. Both lanes now share conductor cells, source local volume, minimum spacing, dt and 600-step window. | Uniform 250 um control; graded x/y 250 um central 7--15 mm and exterior, 312.5 um shoulders 4.5--7 and15--17.5 mm. All cube/source/NTFF faces on nodes. Unchanged discrepancy<5% and local error<scalar error/2. |
| Coarse short | Correct label: 40 x 20 mm rectangular guide, length 120 mm. Buildable 2 mm closing plate at x=84--86 mm, preserving today's coarse realized body. Sources 10/110 mm, references 16/104 mm, probes 30/90 mm stay on their source's connected side of the short. | dx=2 mm coarse and 1 mm fine divide all faces and measurement coordinates. Unchanged coarse (2.25,3] interval and fine <=2.25 bound. Warning check now specifically requires the column-power advisory; reciprocity advice cannot satisfy it. |

The microstrip and patch footprints are ordinary etched dimensions, with copper
idealized by PEC sheets. These fixtures do not claim fabricated connector,
finite-copper loss/roughness, actual FR4 variability or finite-board accuracy.
The solid cube and closing plate have independently specified physical thickness.
No geometry dimension is changed as a function of mesh refinement.

## V173A: replace the unsupported physical claim

This revises the review's proposal by choosing its explicitly allowed synthetic
composition descendant, **not** the proposed finite 80 x 80 x 30 mm domain and
0.25 mm antenna mesh. The latter is a separate antenna qualification campaign.
The original V173A was never a calibrated reflection experiment. Its replacement
calls the FFT quantity `probe_spectrum_min_db`; a source-amplitude falsifier
shows exactly 20log10(2)=6.0206 dB change on doubling the same trace.

The initial repaired-board extraction found no Q>=5 pole. Automatic decimation
was suspected, but restoring the original non-decimated instrument also failed.
The first harness raised before dumping the trace, a diagnostic persistence
defect fixed before repeating that capture. The saved trace then falsified the
instrument hypothesis: fitting the **driven** full record finds only low-Q
pulse poles; fitting the free ring-down exposes a persistent mode. No further
geometry or source sweep was run. The final instrument starts at t0+6tau,
1.7904931097838228 ns (sample 1879), bounding the source by 12exp(-36) A, and
subtracts the static mean before the original non-decimated Q>=5 fit.
The 1000-sample exploratory cut is recorded but is not the selected instrument.

The successor remains an amplitude-ranked dominant probe mode, not a spatial
mode identity or a Q/antenna matching assertion. Its ring-down tail remains
0.7684 of its peak; neither convergence nor physical Q is established. Existing
`tests/locks/test_patch_edgefed_resonance_harminv.py` supplies mode-labelled
patch coverage and `tests/crossval/test_patch_canonical_farfield_e4.py` supplies
mode-labelled far-field/reference regression coverage. Those are coverage
pointers, not new results from this task. No test was deleted or xfailed.
The old JSON and its numbers remain archived, with its false interpretation
explicitly marked retired.

## MSL qualification is separate from the historical replay

The unchanged 80 um replay binaries still run on their original builder. The
new fully aligned live E2E builder is separate from replay and synthetic AD.
An initial anisotropic-constant profile failed the existing boundary-cell
contract during assembly; a subsequent coarse proposal exposed insufficient
clearance/runway in preflight. Its owned preliminary run was terminated after
checking PID/cwd/command identity; its log is retained. These were fixture
construction defects, not numerical results to fit. The final preflight has
only the lossless-Q notice, irrelevant to this transmission coupon.

Before the final solve, qualification budgets were declared: all ten 0.5--5 GHz
bins finite/reliable and well-conditioned, each drive settled below -40 dB,
physical-band 3--4.5 GHz raw reflection<.15, transmission(.90,1.05), maximum raw
singular value<=1.05, reciprocity<.01, drawing-model Z0/beta within10% and
transmission electrical-length phase within5 degrees. The new model budgets
are conservative screening choices, not measured accuracy envelopes. The
matched-line limit and six deliberately wrong observables exercise them.

The API forms waves with its drawing-based line Z0, despite 50 ohm launch
metadata (`rfx/api/_sparams.py`, Z0_HJ wave assembly). The comparator therefore
uses the matched-line response exp(-j beta L), correcting the review's assumed
50 ohm ABCD reference. Source review also found that returned S uses the
first measurement planes, not the launches: L=4 mm between x=5 and 9 mm.
The initial 10 mm oracle was corrected without changing the 5 degree gate.
Running-solve outputs using that initial comparator are retained and checked
offline against the corrected physical reference, with provenance explicit. Its approximate quasi-static beta shares a model family
with production; agreement is not an independent full-wave certification.
Raw/projected S, conditioning, reliability, settling and fitted Z0/beta are all
recorded. The intended 4.5 GHz endpoint is represented as 4,500,000,256 Hz by the
float32 frequency vector. A one-ULP endpoint guard includes it in the physical
checks; this strengthens the checked subset without changing any tolerance.

The completed v1 capture passed every corrected numerical screen except
reliability: both ports flag 4.5 and 5 GHz. Its source pulse had implicit
f0=2.5 GHz / bandwidth=.8. The differentiated-Gaussian spectrum predicts
exactly those two weak bins, at 7.25% and 2.46% of its band median, below the
API's unchanged 10% low-signal screen. Raw S remains smooth, nearly reciprocal
and near unity in transmission; this does not override the reliability mask.

The final **v2 fixture explicitly uses f0=5 GHz / bandwidth=.8**, giving
35.48% or more of the source's band-median amplitude at every declared bin.
A sampled-waveform test also verifies a predeclared 25% excitation margin and
rejects the old pulse at exactly the two observed bins. This repairs a source
that did not adequately cover the measurement band; no reliability threshold
is lowered. Final v2 preflight and source-spectrum checks pass, but **no v2
FDTD capture or physical qualification is claimed**. The preserved v1 fields
must not be relabelled as results of v2. The prepared VESSL jobs are the live
qualification of the final source.

The v1 capture differs from all 40 historical complex entries, max|delta S| =
0.20305470527038097. It settles at -103.19/-103.41 dB, cond(A)<=1.34, maximum
raw singular value 1.00012418, and corrected phase error<=0.40341 degrees.
See the full matrices and source evidence in `fixture-repair-evidence/`.
**No MSL snapshot is refreshed** without passing physical checks, a fixed-board
refinement and an independently named confirmation. External full-band accuracy
remains outside this evidence.

## Results, selection and launch handoff

Only the V173A composition successor gets new numerical pins:
1,984,479,720.2163157 Hz and 190.48059426681422 dB for the honestly named
FFT statistic. The saved-field reanalysis and independent fresh-FDTD test
confirm them at the unchanged 1e-6/0.1 dB tolerances. Its proximity to the
old 1.995 GHz number is not a recovery or accuracy claim. Far-field now reads
0.3590896% against the unchanged 5% gate; the local-cell discriminator also
passes. The short remains a finding at 1.0444802 against unchanged (2.25,3].

Pytest census (one controller at a time, at most 4 workers): initial V173A
checks 2 passed; fast contract selection 139 passed; full 158-case selection
155 passed /3 failed; final focused follow-up 18 passed. No skips, xfails or
errors. These are 317 completed reports, 314 passes and 3 failures across
versions, not 317 distinct tests. The full selection predates final MSL source
and comparator repairs. Its V173A build-only collector failure was fixed and
passed the follow-up. The short interval stays red. Final v2 MSL FDTD is
unverified; the v1 failure is not presented as a run of v2.

The independently solved v1 MSL raw matrix was array-identical to the named
capture; independently solved V173A frequency/spectrum deltas were exactly 0.
Final numerical outcomes and exact pytest census are recorded in
`fixture-repair-evidence/selection-summary.json`. Detailed full-trace notes and
visually inspected figures in that directory distinguish observations from
accuracy claims. The original far-field and short acceptance limits remain in
executable assertions. A repaired fixture that still misses is a finding.

Six CPU 8/32GiB VESSL jobs are prepared in `scripts/vessl_931/fixture-repair-*.yaml`.
Use `scripts/vessl_931/fixture-repair-jobs.md` to render the final full commit SHA
and submit with recorded run IDs. They collect evidence and never write goldens.
The independently scheduled MSL base/long/refined records and far-field 1200-step record test unresolved
window/mesh sensitivity; the short job deliberately preserves the coarse red.
No VESSL execution, GPU comparison or complete repository suite is claimed.

R3: consistent with the quoted review and current PI authorization; protected
production files unchanged. One final physical configuration per fixture, with
identified construction/instrument defects recorded before repeats. Falsifiers
include wrong-observable rejection, exact realized geometry, frozen physical
probe planes, and the unchanged failing coarse-short interval. There was no
parameter search for a green scalar gate.
