# Adjudicate the repaired fixtures on pinned live runs

Continuation of [T9](T9-fixture-repair.md), under the PI's 2026-09-09 full
VESSL authority. Work is confined to `rfx-931-fastlane-reds`, branch
`feat/931-fixture-repair`. All physics jobs clone
`b36fc46cdf21d1c57f221e6a057654bcad60bae2` to unique node-local directories.
The #931 conductor contract and `rfx/` are unchanged. No gate is widened,
no test is deleted or xfailed.

The exact prelaunch predictions and collection commands are in
[the evidence directory](fixture-repair-live-evidence/). Empty pytest addopts,
explicit thread timeouts, and exact selectors collected 4/12/11/11/3/16 cases.
Six isolated CPU jobs were submitted together. A seventh, capture-only MSL
base replication supplies the existing independently named confirmation
requirement. The prepared top-level CPU/memory fields were silently dropped by VESSL.
A resource-only probe (369367259621) showed that this cluster also drops the
[documented custom `requests` schema](https://docs.vessl.ai/reference/yaml/run-yaml), with unlimited CPU/memory cgroups.
The three completed physics lanes are retained as numerical evidence; the
four unfinished MSL jobs were terminated after logs/identity checks and
replaced using explicit CPU-only presets. `base-pod` is verified by completed
probe 369367259623 at 8 CPU / 32 GiB in both `run read` and cgroup limits.
The refinement preset `cpu-32-mem-64` supplies 32 CPU / 64 GiB; its separate
probe is 369367259624. All maintained YAMLs now assert and record these limits
before starting any simulation. No GPU is requested.

## Launch and collection

The initial six runs failed before simulation at `git clone`, rc128: trusting
only the checkout did not trust its linked Git administrative directory.
The corrected job specifications add that exact administrative path to the
pod's `safe.directory` list. No source or numerical parameter changed for
this retry. `vessl_submit.sh` now copies the YAML into its owned temporary
directory and invokes VESSL there, so calling it from a git worktree works.
A mocked CLI check exercised plain cwd, copied YAML and submitter run-ID
recording without creating a run. The submitter wait is now60 minutes;
probe624's original5-minute wait expired during image pull, so the submitter
recovered its known ID from the original create response after the unique
output directory appeared. That recovery is recorded separately.

The base pytest confirms qualification then fails the old golden at maximum
difference0.20305846,40/40 entries. xdist with `-s` dropped its diagnostic
stdout; the independent captures retain all matrices. The maintained runner
now uses `--capture=tee-sys` and `junit_logging=all`, verified with a synthetic
passing test whose stdout is present in console and JUnit. This changes
logging only. All actual job IDs are recorded by the
submitter in their artifact directories; original JSONs remain unmodified.

| Lane | Initial clone failure | Corrected run | Result |
|---|---|---|---|
| V173A | 369367259608 | 369367259614 | capture rc0; 4 passed |
| MSL base | 369367259605 | 369367259636 | capture qualifies;11passed/1old-golden failure (612 terminated for allocation repair) |
| MSL long | 369367259607 | 369367259637 | capture qualifies, temporal comparison passes;11passed (615 terminated for allocation repair) |
| MSL refinement | 369367259610 | 369367259648 | pending (617 terminated for allocation repair) |
| MSL confirmation | none | 369367259638 | capture rc0, qualifies and exactly reproduces base (616 terminated for allocation repair) |
| Far-field | 369367259609 | 369367259613 | both captures rc0; 3 passed |
| Short | 369367259606 | 369367259618 | captures rc0; 15 passed, 1 failed |

Initial clone-failure logs are retained as provenance evidence. The interrupted
local confirmation submitter before the corrected wave created no VESSL run,
as verified against the run list. It is not counted as a launch.

## Far-field: temporal qualification closed at the stated norm

The prelaunch temporal screen required <1% change in each mesh's absolute
integrated power and full combined complex angular field L2. There is no
phase/amplitude alignment or normalization before the complex comparison.
Here "absolute" means the unnormalized integral under the fixed source
convention, not a new calibrated-watt claim. The 1200-step record is the
reference. Same SHA, geometry, source waveform
prefix and angular axes are verified, and power is recomputed from the saved
complex fields rather than accepted from a scalar report.

| Quantity | Uniform | Graded |
|---|---:|---:|
| Power change, 600 to 1200 | 0.0232341% | 0.0150934% |
| Combined complex field L2 change | 0.495627% | 0.494430% |
| Power-normalized pattern L2 change | 0.6934% | 0.6898% |

The original uniform/graded discrepancy is 0.359089% at 600 and 0.367258%
at 1200, both below the unchanged 5% gate and less than half the scalar-cell
integration error (4.52319% and 4.54444%). The full test module passes.
The named figure `fixture-repair-live-evidence/farfield-temporal-power-angular.png`
shows overlapping angular lobes and the residual complex field change.

This closes finite-window sensitivity for this fixture's integrated-power and
combined-field observable. It does not claim arbitrary mesh convergence or
separate cross-polarization accuracy: weak E_phi, only about 1.6–1.8% of the
combined field norm, changes 7.684%/6.821% relative to itself. Maximum
peak-normalized intensity pointwise changes are 2.043%/1.841%; neither those
nor a polarization-specific gate was substituted for the declared norm.
A common 0.1-rad phase rotation leaves power unchanged but fails the offline
complex criterion; a corrupted source prefix is rejected.

## Short: causal interpretation closed; original assertion remains red

Run 369367259618 measures maximum column powers 1.0444794 coarse and
1.0494230 fine (recomputed in complex128; original float32 scalars retained).
The coarse value fails the unchanged `(2.25,3]`; the fine control passes
`<=2.25`. All sixteen selected tests execute; the sole red is
`test_soft_advisory_real_coarse_pec_short_witness`.

The interval encoded an invalid outgoing-power sum. With the old right
source/reference/probe at 90/84/70 mm, receiving-port measurements were on
the left side of the plate and interpreted a second observation of the left
cavity as transmitted power. Restoring only historical node sampling recovered
2.5279038, versus historical sigma damping's 2.5279102. At 6 GHz the former
sum combined |S11|=1.1338613 with spurious |S21|=1.1145682.

The controlled current-owner historical/repaired pair has bit-identical S11
at every frequency and identical left-drive/left-port V/I histories. Only the
invalid remote measurement changed. The new named run independently reproduces
two observable reflections, exactly zero cross-plate S12/S21 and zero
opposite-side V/I traces. Cross-host reproduction differences are explicitly
recorded separately from the exact controlled identity. See
[the full causal note](fixture-repair-evidence/short-live-adjudication.md) and
`fixture-repair-live-evidence/short-adjudication.json` for per-bin decompositions,
source fingerprints, connected port planes and trace comparisons.

No product change is indicated by this gate miss. The historical live advisory
assertion remains visible and red under the PI's unchanged constraints;
synthetic helper tests retain the warning interval and hard-limit coverage.
This is not absolute reflection qualification: the coarse record fails -40 dB
settling, and both absorbers are shallower than their documented discipline.
The fine control changes band, duration and physical absorber thickness, so
it is not a fixed-configuration mesh convergence witness.

## V173A: independent composition confirmation closed

Run 369367259614 returns dominant probe mode 1,984,479,720.2162623 Hz and
probe spectrum minimum 190.48059426681422 dB. The frequency differs from the
committed pin by about 0.0000534 Hz, and the spectrum delta is zero; all four
tests pass at the existing tolerances. All 6000 raw samples and dt are
array-identical to the archived repaired capture; the named fresh ringdown/FFT
figure is `fixture-repair-live-evidence/v173a/v173a-live-ringdown-spectrum.png`.
Its captured ringdown remains 0.768401
of the peak, so this confirms the composition lock, not physical Q, antenna
matching, or mode-labelled mesh convergence.

## MSL qualification

Base run 369367259636 and independent confirmation369367259638 pass every
physical screen and reproduce all 40 raw/projected complex S entries exactly.
Both drives are reliable at all 10 bins and settle at -103.77146/-104.00676 dB;
condition number is at most 1.339612, raw singular value at most 1.00012875,
and physical-band electrical-length phase error at most 0.366582 degrees.
The source repair prediction is supported: compared with the preserved v1
capture on the same board, maximum raw complex change is 2.57616e-5 while
both drives' previously unreliable 4.5 / 5 GHz bins now pass. No v1 unreliable
result is relabelled as qualified v2 evidence.

The 24-period record369367259637 also qualifies. Versus12 periods, maximum
raw S change is2.45756e-7 and projected change5.33120e-7, passing the unchanged
.005/.002 budget with maximum budget ratios9.49575e-5 and1.50574e-4.
Settling improves to -116.66416/-116.76717 dB. Temporal convergence is closed
for the stated fixed fixture and measured band.

Refinement is pending. The authorized254um/600um board and original golden
remain unchanged until that final predeclared mesh check passes.
