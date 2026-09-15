# Fixture repair handover: qualification incomplete

**MSL update, 2026-09-09:** [T13](T13-msl-repin-closure.md) records the landed
refinement, passing four-run adjudication and offline base re-pin. The MSL
waiting/no-re-pin statements below are the historical handover snapshot.

**Short-only update, 2026-09-09:** [T12](T12-short-witness-closure.md)
closes the short by retiring its invalid physical advisory trigger and
qualifying replacement warning-policy and isolation tests. The short status
below is the historical handover snapshot. The MSL disposition and pickup
condition are unchanged.

Disk-only handover, 2026-09-09, from `feat/931-fixture-repair` at
`b36fc46cdf21d1c57f221e6a057654bcad60bae2` plus the previous session's
uncommitted work. No VESSL query, launch, restart, solve, or wait was performed
for this handover. The user's last service census was **20 launched: 7
completed, 8 failed, 4 terminated, 1 running**. Local artifacts reconcile with
that census; it is not a refreshed service-status claim.

**1. Fixture dispositions**

Evidence paths below are relative to this directory. `fixture-repair-live-evidence/`
contains archived completed outputs; the original live outputs remain under
`output/931-fixture-repair-vessl/` at the worktree root.

| Fixture | Disposition | Observed numbers and unchanged gates |
|---|---|---|
| V173A composition successor | **Closed for the declared composition lock.** Named run `369367259614` independently confirms the pin already committed at b36fc46c. | Dominant probe mode **1,984,479,720.2162623 Hz**, against pin **1,984,479,720.2163157 Hz**, at `rtol=1e-6`; probe spectrum minimum **190.48059426681422 dB**, equal to the pin, at `atol=0.1 dB`. **4 passed**. All **6000** samples and dt are array-identical to the archived repaired capture. Ringdown tail/peak **0.768401026725769**: no physical-Q, antenna-match, or mode-labelled mesh-convergence qualification is claimed. |
| Graded far-field envelope | **Closed for the declared 600/1200-step temporal and original envelope checks.** Run `369367259613`. | Uniform/graded discrepancy **0.35908912493589026%** at 600 and **0.36725781636394684%** at 1200, both `<5%` and `<` half the scalar-cell error (**4.523190069984444%**, **4.544436606181994%**). Uniform/graded power drift **0.023234056844992048% / 0.015093364826270435%**; combined complex angular L2 drift **0.495627207405906% / 0.4944295549243557%**, each `<1%`, without phase or amplitude alignment. **3 passed**. This is not general mesh or independent polarization convergence: relative E_phi drift is **7.68402587434826% / 6.821400252483222%**. |
| MSL historical base, fixed 254 um / 600 um board | **Waiting on refinement `369367259648`; no re-pin.** Base `369367259636`, confirmation `369367259638`, long `369367259637` have landed. | Base and confirmation qualify with **0** maximum raw/projected complex difference across all **40** entries. Base settling **-103.77145626460994 / -104.00676059515744 dB**, max condition **1.3396112277690317**, max raw singular value **1.0001287460327148**, max physical-band phase error **0.3665817688040584 degrees**. Base/long max raw drift **2.457562461863554e-7**, projected **5.331201499700045e-7**, passing `rtol=.005, atol=.002`; long settling **-116.66415868348415 / -116.76717302647188 dB**. Base pytest: **11 passed, 1 failed**, old golden **40/40** mismatches, max delta **0.2030584600560962**. Long: **11 passed**. The run that would settle the remaining mesh check has not landed. |
| Coarse PEC-short witness | **Still open as an acceptance finding. Causal interpretation is closed, but no qualified replacement witness exists.** Run `369367259618`. | Coarse max column power **1.044479250907898** in capture/test, **1.0444794476841999** recomputed in complex128, fails unchanged `2.25 < power <= 3`. Fine **1.0494229793548584** captured, **1.0494230043994222** recomputed, passes `<=2.25`. **15 passed, 1 failed**. Repaired cross-plate **S12=S21=0**. Coarse settling **-25.494890697081768 / -49.59556966120925 dB** fails the `<=-40 dB` screen for one drive; fine **-47.475852231488204 / -103.15660722963592 dB** passes. Both absorbers are below their documented depth discipline, and fine changes band, duration and physical absorber thickness, so it cannot close mesh convergence. |

The short's old receiving measurement sampled the source-side cavity as if it
were transmitted power. Restoring historical node sampling alone gives
**2.5279038**, versus historical sigma damping **2.5279102**; at 6 GHz the
former combined **|S11|=1.1338613** with spurious **|S21|=1.1145682**. The
controlled historical/repaired pair retains bit-identical S11 and left-port
histories while correcting the invalid remote measurement. This supports the
measurement-error explanation, not a new valid advisory interval or a product
regression claim. The MSL refinement will not settle this separate short finding.

Direct numerical sources:

- [V173A capture](fixture-repair-live-evidence/v173a/capture.json) and [trace comparison](fixture-repair-live-evidence/v173a/v173a-live-trace-inspection.json).
- [Far-field adjudication](fixture-repair-live-evidence/farfield-temporal-adjudication.json), with original complex fields in `farfield/{600,1200}/farfield-attribution.npz`.
- [MSL base](fixture-repair-live-evidence/msl-base/base.json), [base screen](fixture-repair-live-evidence/msl-base-screen.json), [confirmation comparison](fixture-repair-live-evidence/msl-confirmation-comparison.json), and [temporal comparison](fixture-repair-live-evidence/msl-temporal-comparison.json).
- [Short adjudication](fixture-repair-live-evidence/short-adjudication.json) and [causal note](fixture-repair-evidence/short-live-adjudication.md).
- Each completed pytest lane retains `summary.json`, return-code files and compressed `pytest.log.gz` / `junit.xml.gz`. These saved five lanes report **44 passes and 2 failures** across **46** reports, not an all-green suite or a new test run by this handover.

**2. Working tree and commit disposition**

The inherited changes form a coherent **incomplete evidence checkpoint**, not
a completed qualification or merge-ready all-green branch. They do not make
the numerical branch worse than b36fc46c: `rfx/`, all tests, all goldens and
all acceptance tolerances have no working-tree changes. The two numerical reds
are the retained historical assertions; no test was deleted or xfailed.

The 18 top-level status entries comprise:

- Ten tracked modifications: the T9 pointer to T10; six maintained
  `scripts/vessl_931/fixture-repair-*.yaml` definitions; their jobs guide;
  `run_fixture_repair_evidence.py`; and `scripts/vessl_submit.sh`.
- T10's live qualification note, the short causal note, and the archived live
  evidence directory containing named captures, traces, JSON adjudications,
  logs, JUnit, launch/resource provenance and submitted YAML revisions.
- Five offline diagnostic/plot scripts: `adjudicate_931_farfield_temporal.py`,
  `adjudicate_931_msl_live.py`, `adjudicate_931_short_repair.py`,
  `inspect_931_v173a_capture.py`, and `plot_931_short_adjudication.py`.

Reusable changes repair resource selection and checks, Git worktree submission
and clone trust, the submitter's run-ID wait (5 to 60 minutes), and pytest
stdout/JUnit preservation (`--capture=tee-sys`, `junit_logging=all`). Actual
physics runs cloned b36fc46c: the later runner logging change is not retroactive.
The four-capture MSL adjudicator is prepared but has not run with a final
refinement capture. This is an explicit validation gap, not a passing result.

The handover adds only this document. Commit the inherited work plus this
handover with qualification explicitly incomplete. Include the three existing
figures explicitly because `**/*.png` is ignored by Git:

- `fixture-repair-live-evidence/farfield-temporal-power-angular.png`
- `fixture-repair-live-evidence/v173a/v173a-live-ringdown-spectrum.png`
- `fixture-repair-evidence/short-live-369367259618.png`

The raw output directory, original prepared jobs, and transcript stay on disk;
they are not newly generated results. Inspection checks for this handover:
The inherited tracked diff passes `git diff --check`; shell syntax for the
submitter and AST parsing of the six changed/new Python scripts pass. After
staging the raw evidence, `git diff --cached --check` reports trailing whitespace
at line 10 of each `resource-read-369367259623.txt` and
`resource-read-369367259624.txt`: these are preserved CLI output, including
ANSI formatting, not implementation changes. No simulation or pytest was
rerun. Saved smoke outputs document the prior session's submitter/logging checks.

**3. All 20 runs and why they ended**

| Status at supplied census | Exact run IDs | Meaning |
|---|---|---|
| Failed: initial clone, rc128 | `369367259605` MSL base; `369367259606` short; `369367259607` MSL long; `369367259608` V173A; `369367259609` far-field; `369367259610` MSL refine | Six job-definition failures before simulation. Trusting the checkout did not trust its linked Git administrative directory. Retry adds that exact directory to `safe.directory`; physics SHA and numerical parameters stay fixed. |
| Failed: real assertion | `369367259618` short | Captures both rc0; pytest rc1, **15 passed / 1 failed**, `test_soft_advisory_real_coarse_pec_short_witness`, **1.044479250907898** below **2.25**. This is a retained physical-measurement/acceptance finding. |
| Failed: real assertion | `369367259636` MSL base | Capture rc0 and physical qualification passes; pytest rc1, **11 passed / 1 failed**, `test_compute_msl_s_matrix_end_to_end_matches_historical_base`, **40/40** old-golden mismatches, max **0.20305846**. Conditional re-pin candidate, pending refinement; not a failed physical screen. |
| Terminated | `369367259612` MSL base; `369367259615` MSL long; `369367259616` independent MSL confirmation (service title inherited base); `369367259617` MSL refine | Four unfinished physics runs stopped after identity/log backup because the resource fields had been discarded. No completed qualification failure is recorded for these attempts. Replacements are **636 / 637 / 638 / 648**, respectively. |
| Completed physics | `369367259613` far-field; `369367259614` V173A; `369367259637` MSL long; `369367259638` MSL confirmation | Results above. Confirmation is capture-only; no separate pytest count. |
| Completed resource probes | `369367259621` custom requests; `369367259623` base preset; `369367259624` refinement preset | **621** completed as a probe but falsified custom requests: unlimited cgroups. **623** verified **8 CPU / 32 GiB**, `cpu.max=800000 100000`, `memory.max=34359738368`. **624** verified **32 CPU / 64 GiB**, `cpu.max=3200000 100000`, `memory.max=68719476736`. No FDTD in these three probes. |
| Running / result absent | `369367259648` MSL refine | User census start **02:09 UTC**; local runner start **02:10:46.560475 UTC**. Refinement **2**, **12 periods**, command timeout **86400 s**, job cap **129600 s**. No `refine.json`, `refine.rc`, or `job-exit.json` at handover inspection. Do not restart or wait for it. |

Provenance: [T10 run table](T10-fixture-live-qualification.md),
[prelaunch audit](fixture-repair-live-evidence/prelaunch-audit.txt),
`fixture-repair-live-evidence/submission*.json`, `launch-failures/*.log.gz`,
`vessl-logs/*.log.gz`, and the per-lane return codes. The separately interrupted
local confirmation submitter created no VESSL run and is not a 21st launch.
Probe 624's original five-minute local submitter wait expired during image
pull; its known ID was recovered from the original create response and recorded
in `resource624-submitter-recovery.json`. It was not a failed VESSL run.

The final working allocation definition is CPU-only `base-pod` for ordinary
lanes and `cpu-32-mem-64` for refinement, with recorded numeric cgroup limits
and fail-fast minimum checks, exact administrative-directory trust, and unique
temporary clones of the pinned SHA. The archived executed definitions are
`fixture-repair-live-evidence/submitted-yamls/`: V173A/far-field/short `*-r2.yaml`
predate the preset guard, while MSL base/long/confirmation/refine `*-r3.yaml`
use verified presets. The completed r2 physics evidence is retained with that
resource-provenance limitation; it was not retroactively run under presets.

**Stale preparation artifacts must not drive a relaunch.**
`output/931-fixture-repair-jobs-b36fc46c/manifest.json` still says
`not_launched: true`; its YAMLs are the original definitions with discarded
resource fields and incomplete Git trust. They are historical preparation,
not the final definitions or current census. Maintained
`scripts/vessl_931/fixture-repair-*.yaml` contain the fixes. The jobs guide's
old statement that admission is unverified is superseded by probes 623/624;
its launch recipe is not the next action for this already-launched campaign.

**4. Exact pickup after run 369367259648 lands**

The next action is to read that existing run's output when it becomes
available, without launching another job. The exact directory is:

```text
output/931-fixture-repair-vessl/issue931-fixture-repair-msl-refine-20260909T021013Z-b36fc46cdf21-106/
```

Read `run_id.txt` (**369367259648**), `commit.txt` (full **b36fc46c** SHA above),
`resources.json`, `refine.json`, `refine.rc`, `summary.json`, `pytest.rc`, and
`job-exit.json`; preserve logs and JUnit alongside the capture. A running
`summary.json` or preflight log is not a numerical result. If the JSON never
lands, or the job times out/fails before a valid capture, MSL remains open:
**the run that would settle this has not landed**. Record the actual exit
reason; do not infer a mesh result or automatically restart it.

Once the complete capture exists, use the existing offline adjudicator with
the saved base, long and independent confirmation, keeping source SHA pinned
to **b36fc46cdf21d1c57f221e6a057654bcad60bae2**, even after this handover commit.
The following is a future read/reanalysis command, not executed by the handover:

```sh
E=docs/design_notes/931_migration/fixture-repair-live-evidence
R=output/931-fixture-repair-vessl
D=$R/issue931-fixture-repair-msl-refine-20260909T021013Z-b36fc46cdf21-106
PYTHONPATH=$PWD python scripts/diagnostics/adjudicate_931_msl_live.py \
  --base "$E/msl-base/base.json" \
  --long "$E/msl-long/long.json" \
  --confirm "$E/msl-confirm/confirm.json" \
  --refine "$D/refine.json" \
  --output "$E/msl-final-adjudication.json"
```

The script defaults to the exact source SHA above and refuses existing output
JSON/figure paths. Archive the landed refinement directory under the evidence
directory before final publication, including its ID and commit witnesses.
The decisive new number is
`comparisons.base_refine_raw.maximum_abs_delta`: **max over all 40 complex
entries of abs(base.S_raw - refine.S_raw) must be <=0.02** (`rtol=0`).
It must not compare either mesh to the historical golden or compare only
magnitudes/projected values to decide mesh qualification.

The full decision also requires:

- Exact fixture/source/CPU/dtype provenance, distinct four run IDs/names,
  matching ten frequencies, fixed board/reference planes and source, and
  correctly bisected mesh with unchanged physical geometry.
- Every role's physical qualification failures empty: finite raw/projected S,
  multi-drive assembly, all ten bins reliable, both drives settled `<=-40 dB`,
  condition `<=1000`; in the declared 3--4.5 GHz band, no railed beta,
  reflection `<.15`, transmission `(.90,1.05)`, raw singular value `<=1.05`,
  complex reciprocity `<.01`, Z0/beta model errors `<10%`, phase error `<5 degrees`.
- Base/confirmation projected S and base/long raw **and** projected S satisfy
  `abs(base-other) <= .002 + .005*abs(other)` elementwise. Those saved checks
  already pass; the final adjudicator cross-checks them with all four inputs.

**Re-pin:** only if the final report says `eligible_for_repin: true`, its
`failures` is empty, and the landed job has no unresolved failure. Use the
**base run 369367259636 `S_projected` complex64 array**, shape `(2,2,10)`, as
the candidate for `tests/fixtures/msl_s_matrix_golden.npy`, not the refined or
long result. Record the base value array, exact SHA, all named reports/run IDs
and comparison evidence in `tests/fixtures/msl_s_matrix_golden.json`, and write
the reason in `test_compute_msl_s_matrix_end_to_end_matches_historical_base`'s
docstring in `tests/unit/autodiff/test_msl_sparam_ad.py`. Preserve
`rtol=5e-3, atol=2e-3`; preserve the 80 um replay fixtures and `rfx/`.
Verify the saved base and independent confirmation against the new pin offline.
The existing `capture_msl_e2e_golden.py --write-golden` entrypoint **executes a
new solve before writing**; it is not an offline import command. Do not invoke
it merely to consume the existing result. There is no existing offline golden
writer in this checkpoint; implementing/reviewing that transfer is future work
after qualification, not work performed by this handover.

**Finding:** if raw base/refine drift exceeds **0.02**, any physical screen
fails, provenance is inconsistent, or another landed failure is unresolved,
leave the old golden intact and report the precise failing bins/entries and
observed values. Do not widen gates or adjust the board to force a pin.

For the short, the immediate pickup is to retain the red and its causal note;
there is no pending run that supplies a qualified replacement advisory witness.
Changing that acceptance contract or starting a new absolute-reflection study
is a separate decision, not a re-pin supported by these artifacts. V173A and
the stated far-field temporal checks require no additional run for this handover.
