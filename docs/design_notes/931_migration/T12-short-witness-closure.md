# Retire the short as an obligatory advisory trigger

Decision, 2026-09-09. This supersedes only T11's short disposition. The PI
explicitly authorized changing its acceptance contract while preserving #931,
production code, and the separate MSL item.

## Property and derivation

The old witness protects **warning-policy integrity**: callers must see an
advisory for extracted outgoing power in the policy's soft range, with no
exception or mutation of the result; excessive/nonfinite results retain hard
handling. It does not protect a law requiring a passive short to create power.
For unit incident wave a_j=1, outgoing power is P_j=sum_i |S_ij|^2.
The published policy (CHANGELOG.md, "waveguide S-matrix soft over-unity
advisory", introduced at e7fb2ba774064cdcc4024df82578d5b5b07c91cb) is:

| Path | Silent column-power gate | Warn-only | Hard diagnostic (raise if strict) |
|---|---|---|---|
| normalize=False | P <= 2.25 | 2.25 < P <= 1+2 | P > 1+2 |
| normalize=True or flux | P <= 1+0.10 | empty soft interval | P > 1+0.10 |

Other diagnostics may independently warn. The tests distinguish the column
power advisory from reciprocity and per-entry amplitude warnings.

**Verdict:** `(2.25, 3]` remains the existing software advisory policy. It was
never derived as a physical short prediction. The 2.25 floor was chosen with
margin over a measured ~2.0 envelope, and the ~2.51 coarse realization was
chosen to exercise it. That realization double-counted one cavity as two
outgoing ports. Requiring a valid short to stay in this range fitted an invalid
measurement. We retire that physical requirement, without shifting or widening
any interval and without choosing a new numerical pin.

The replacement public-API test injects S at the numerical extractor boundary,
then exercises real assembly, normalization dispatch, result construction and
the diagnostic epilogue. Unit incident power and real outgoing entries give
P=1, 2.25, 2.5, 3, 3.25; exact float64 sums are checked before use. This covers
both loose-policy boundaries (open lower and closed upper), both sides,
all three normalization modes, strict/non-strict handling, and exact preservation of S. The helper tests
also check the adjacent floating-point values around |S|=1.5. These are policy
inputs derived from the boundary, not fitted FDTD outputs.

The repaired live fixture now protects **PEC isolation with observable drives**.
A full-cross-section PEC plate gives two uncoupled regions of the Maxwell update.
Zero initial fields and no source in the opposite region imply its V/I histories
are exactly zero, hence S12=S21=0 at every bin. Both driven a/b spectra and all
four local V/I histories must be finite and nonzero: an unobserved drive cannot
pass by returning an all-zero matrix. Existing compiled-plane geometry checks
protect both walls and the source/reference/probe connected regions. These
criteria require no fitted amplitude tolerance, settling or mesh extrapolation.
Coarse and fine are independent topology checks, not a mesh-convergence pair.

Reflection-accuracy coverage remains the separate
`tests/oracle/test_waveguide_port_validation_battery.py::test_pec_short_s11_magnitude`
(min >=0.99, max <1.03, mean within 0.02 of unity). Its committed record names
VESSL 369367259278, module 9 passed. That is historical coverage, not a fresh
run here and not qualification of this coarse fixture.

## Evidence and prelaunch audit

Relevant project memory directories are absent in this worktree; no other
worktree was consulted. The existing durable design notes supply the evidence:

- T11: "Changing that acceptance contract or starting a new absolute-reflection
  study is a separate decision". This decision is now explicitly authorized;
  it does not claim the old qualification succeeded.
- `fixture-repair-evidence/short-live-adjudication.md`: "Both transmission terms
  and all opposite-side traces are zero, as required for the closed
  full-cross-section plate." The new topology property is consistent with this
  independent observation and #931's full PEC wall ownership.
- #931 normative design: "a Box drawn ... on node planes realizes tangential
  walls at BOTH" faces. No product operator or alternate realization is changed.

Prior named run 369367259618 at b36fc46cdf21d1c57f221e6a057654bcad60bae2:
**15 passed, 1 failed**, coarse maximum **1.044479250907898**, fine
**1.0494229793548584**. Historical node sampling under current edge ownership
restored **2.527903795**, versus **2.527910233** with historical sigma damping;
at 6 GHz |S11|=1.133861328 and spurious |S21|=1.114568210. The controlled repair
preserved left S11 and its V/I traces bit-for-bit while removing false remote
power. Full committed captures and inspection figure are linked from T11.

Prelaunch checks: local policy/geometry **55 passed, 0 failed**; exact coarse
and fine node selections rehearsed separately with `-o addopts=` and timeout
settings. Cheap falsifiers inject a missing guard, wrong tolerance and a soft
advisory promoted to exception: all must be rejected by the public API gate.
R2: zero new physics attempts before this campaign; one predeclared topology
qualification per existing resolution. No parameter search or repeat of the
old causal interventions. Falsifier: any nonzero cross-region history/S entry,
missing driven wave, or wrong public diagnostic outcome fails qualification.

VESSL: use maintained base-pod/remilab-c0 from fixture-repair-short.yaml;
resource reference 369367259623 verifies 8 CPU/32 GiB. Status snapshots and
lab-status are unavailable here, so the maintained-definition fallback applies.
The job verifies cgroup minima, CPU backend, exact SHA and local import. Each
job gets a unique node-local clone and an independent artifact pointer. The
submitter copies YAML outside git and records the run ID. One pytest per pod,
policy -n 4; each physics job -n 0; all use thread timeouts. Prior short run
369367259618 logs are backed up and retained as claims-bearing evidence; other
campaigns and the running MSL refinement are outside this task's cleanup scope.

## Live result

**Closed by retirement and passing replacement coverage.** All three named
runs executed pinned source **db4ee4a1797e2109f1fe8f70b14f93740fbd249b** and
completed successfully. All resource records verify 8 CPU / 32 GiB, CPU JAX,
and no product changes. Submitter run IDs, exact submitted YAMLs, commit files,
return codes, complete logs/JUnit and machine-readable outcomes are archived
in [short-closure-evidence](short-closure-evidence/summary.json).

| Run name | VESSL ID | Passed | Failed | Errors | Skipped |
|---|---|---:|---:|---:|---:|
| rfx-931-short-closure-policy | 369367259732 | 55 | 0 | 0 | 0 |
| rfx-931-short-closure-coarse | 369367259733 | 1 | 0 | 0 | 0 |
| rfx-931-short-closure-fine | 369367259734 | 1 | 0 | 0 | 0 |
| Total, newly launched | | **57** | **0** | **0** | **0** |

No failed/terminated launches or retries. Local policy/geometry separately
passed 55 tests, with 0 failures; collect-only selected 55/1/1. The three
injected policy faults were all rejected; those were diagnostic calls, not
additional pytest cases. This is the selected short closure scope, not a full
repository-suite result. The retired node is removed and replaced, not xfailed,
skipped, or hidden behind a marker filter.

| Diagnostic (not an acceptance pin) | Coarse 369367259733 | Fine 369367259734 |
|---|---:|---:|
| Max column power, original float32 arithmetic | 1.0444785356521606 | 1.0494229793548584 |
| Left/right settling dB | -25.4948909801 / -49.5955696612 | -47.4758522315 / -103.1566072296 |
| Samples per V/I record | 1312 | 2998 |
| Cross-region nonzero samples, all eight records combined | 0 | 0 |
| Both off-diagonal S entries, every frequency | exactly 0 | exactly 0 |
| Both driven a/b spectra and eight local V/I records | finite, nonzero | finite, nonzero |

Every complex S bin, driven a/b magnitudes and trace peak is saved in
[coarse.json](short-closure-evidence/coarse/coarse.json) and
[fine.json](short-closure-evidence/fine/fine.json). The corresponding
`*-records.npz` files retain all 16 V/I records per lane. The
[inspection figure](short-closure-evidence/short-closure.png) was rendered and
visually inspected: zero cross-region traces, nonzero driven pulses, coarse
late left-drive lobe and the 4 GHz deficit are all visible. Full complex bins
were inspected separately, including phase. Plot envelopes normalize each
record to its own driven peak; they are not the energy-settling diagnostic.

For explicit per-bin review, left/right reflection magnitudes are:

| GHz, coarse | abs(S11) | abs(S22) | GHz, fine | abs(S11) | abs(S22) |
|---:|---:|---:|---:|---:|---:|
| 4.0 | 0.55135090 | 1.01479124 | 5.0 | 0.93018902 | 0.99981368 |
| 4.4 | 1.02199734 | 1.00075651 | 5.4 | 0.99654694 | 0.99966238 |
| 4.8 | 1.00598729 | 0.99971339 | 5.8 | 0.99964555 | 0.99954825 |
| 5.2 | 0.99963299 | 0.99906418 | 6.2 | 1.00016184 | 0.99947482 |
| 5.6 | 0.99564621 | 0.99842106 | 6.6 | 1.00131600 | 0.99942868 |
| 6.0 | 0.94419927 | 0.99301034 | 7.0 | 1.02441349 | 0.99949123 |

The coarse maximum differs from 369367259618 by -7.15e-7; no cross-host bitwise
identity is claimed. No golden, numerical pin, fixture dimensions, solver
setting, production threshold or rfx/ file changed in this closure. This is a
retirement and property replacement, not a re-pin to 1.0445. The commit records
both named runs and their diagnostic values nevertheless.

**Unsettled and explicitly outside the new gate:** absolute reflection/phase
accuracy and fixed-configuration convergence of this particular short remain
unqualified. Coarse settling fails the existing -40 dB screen and both cases
retain absorber-depth warnings. Neither the near-unity maximum nor fine
settling establishes whole-band accuracy (see 0.93019 at fine 5 GHz). The
separate existing short-magnitude oracle named above covers reflection
accuracy within its own declared envelope. Production comments and changelog
about the historical ~2.0 envelope remain historical policy rationale, not a
newly endorsed physical explanation. No product defect was established or
product change needed. The MSL golden, test and running refinement were left
untouched; its acceptance condition in T11 is unchanged.

Pre-final-commit audit: the result agrees with the declared topology and
warning-policy falsifiers; R2-attempts=1 per resolution, both closing, no
iteration. Independent read-only review found no blocking issue. Exact SHA
and JUnit counts are verified, the figure and full records inspected, and the
final diff is checked for an empty rfx/ and MSL scope.
