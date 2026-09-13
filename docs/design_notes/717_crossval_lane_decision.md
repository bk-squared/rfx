# Which lane runs the crossval CPU tier and the external-solver legs (#717 item 2)

Decision note. No code change here — issue #717 item 2 asks a question only the PI can
answer, so this file is the inventory and the options, not a lane.

Inventory taken at `5c6a9816` (2026-09-13). Every claim below is a file:line or a command
you can re-run; where a number is measured, the command and the machine are given with it.

Line numbers into `.github/workflows/validation.yml` are read at `94b56afe`, this branch's
own change to that file, not at `5c6a9816` — `94b56afe` inserted comment lines and pushed the
later refs down. Everything else is at `5c6a9816`. Each `validation.yml` reference below also
quotes the string it points at, so grep for the quoted text if the file moves again.

---

## 1. What the four gpu-marked crossval test files do today

The files, and where the `gpu` mark lives:

| file | mark | tests collected |
|---|---|---|
| `tests/crossval/test_crossval_comprehensive.py` | `:22` | 11 |
| `tests/crossval/test_meep_crossval.py` | `:17` | 3 |
| `tests/crossval/test_meep_crossval_dielectric_cavity.py` | `:14` | 3 |
| `tests/crossval/test_openems_crossval.py` | `:32` | 1 |

Counts from `python -m pytest --collect-only -q -o addopts="" <file>`, 18 in total.

They are excluded twice over from every CPU lane — once by the `gpu` marker, once by an
explicit `--ignore`:

| lane | selection | the four `--ignore`s |
|---|---|---|
| `.github/workflows/pr-tests.yml` fast shards | default addopts (`pyproject.toml:78`) | `:129-132` |
| `.github/workflows/validation.yml` slow shards | `-m "not gpu and not highmem"` | `:134-137` |
| `.github/workflows/regen-durations.yml` fast / slow | default / `not gpu and not highmem` | `:33-36`, `:66-69` |
| `scripts/vessl_validation_lane_a6000.yaml` weekly GPU | `-m gpu` (`:44`) and `-m "highmem and not gpu"` (`:40`) | `:37` (`$IGN`) |
| `scripts/vessl_gpu_suite.yaml` on-demand GPU | `-m gpu` (`:46`) | **none** |

**Correction to the issue text.** They do not execute *nowhere*. `scripts/vessl_gpu_suite.yaml`
selects `-m gpu` and applies no `--ignore`, so it is the one lane that collects them. It is a
manual harness — "run per release tag + after GPU-touching merges" (`:2`) — and it installs
only `scipy`, `h5py`, `matplotlib`, `pytest` (`:24`), so its meep and openEMS legs skip there
as well. The accurate statement is: **no recurring lane runs them, and no lane anywhere has
ever run their external legs.** The one recurring GPU lane (`vessl_validation_lane_a6000.yaml`)
does select `-m gpu` and then excludes them by hand at `:37`.

### What is actually inside those 18 tests

Measured on this pod, 2026-09-13, 32 vCPU under load average 5.7, no GPU, `meep` installed
but ABI-broken, `openEMS` absent:

```
python -m pytest -p no:cacheprovider -o addopts="" -v -ra --durations=0 \
  tests/crossval/test_crossval_comprehensive.py \
  tests/crossval/test_meep_crossval.py \
  tests/crossval/test_meep_crossval_dielectric_cavity.py \
  tests/crossval/test_openems_crossval.py
→ 11 failed, 5 passed, 2 skipped, 5 warnings in 459.15s (0:07:39)
```

The 5 that pass need no external solver at all:

| test | wall |
|---|---|
| `test_crossval_comprehensive.py::TestPECCavity::test_rfx_vs_analytical` | 400.69 s |
| `test_crossval_comprehensive.py::TestLumpedPortCavity::test_rfx_lumped_port_s11` | 31.61 s |
| `test_crossval_comprehensive.py::TestLumpedPortCavity::test_rfx_lumped_port_resonance_via_probe` | 13.04 s |
| `test_crossval_comprehensive.py::TestWaveguideCutoff::test_rfx_waveguide_cutoff` | 10.57 s |
| `test_meep_crossval.py::test_apply_pec_is_still_the_domain_face_surface_these_loops_use` | < 0.005 s |

The other 13 are the external legs. #717 records these four rfx-only gates at "~4 min CPU"
measured in the week of 2026-08-27; here they are 455.9 s ≈ 7.6 min, dominated by the single
PEC-cavity run. Both numbers are CPU-lane numbers on different loaded machines; neither is a
GitHub-runner number and neither is a GPU number.

Preflight is part of the result, so the two lumped-port tests' advisory, verbatim:

> `[run] preflight found 1 advisory issue(s) - pass skip_preflight=True to suppress:`
> `- Single-cell port at (0.016666666666666666, 0.013333333333333334, 0.01) (ez) sits inside dielectric 'dielectric' (eps_r=2.20) with no adjacent PEC along the z-axis. A floating single-cell port inside substrate does not couple to patch-antenna TM modes. Pass extent=<substrate_height> to create a WirePort spanning ground → patch plane (issue #71).`

Those two tests pass while preflight says their port is not the port they meant. Whoever
adopts them into a lane should read that before calling them coverage.

### The failure mode a lane must not walk into

The 11 failures above are not physics. `meep` is *installed* in this pod and raises
`ImportError: numpy.core.multiarray failed to import` (a meep built against numpy 1.x under
numpy 2.x). Under pytest 9, `pytest.importorskip` no longer absorbs that:

```
# /tmp probe, pytest 9.1.1
test_missing_module_skips  -> SKIPPED  (ModuleNotFoundError)
test_broken_module_skips   -> FAILED   (plain ImportError propagates)
```

So the guard in those files turns a broken solver install into a **red lane**, not a skip.
`scripts/run_crossval_cpu.py` already has the equivalent class for the *scripts*
(`ENV_BROKEN_REF_MARKERS`, `:136-141`, classified `ENV-SKIP` at `:179-180`); the pytest files
have no such guard. A lane that installs pymeep and gets the numpy pin wrong goes red for an
environment reason and looks like a physics failure. On a runner with no meep at all the same
tests skip cleanly — which is what the two openEMS entries did here.

---

## 2. What the manifest's `cpu-runner` tier claims, and who executes it

- 17 of 21 cases carry `cpu-runner` in `execution_tiers`
  (`python -c "import json; m=json.load(open('validation/crossval/manifest.json')); print(sum('cpu-runner' in c['execution_tiers'] for c in m['cases']))"`).
  #717 says 16/21; the manifest has moved since (cv05 retired 2026-09-10, issues #959/#965).
- `scripts/run_crossval_cpu.py:106-112` builds `CPU_SUBSET` from each case's
  `cpu_runner.order`, and `:59` gives every script a 900 s timeout.
- **Nothing calls it.** `grep -rn run_crossval_cpu .github/ scripts/` matches one line: the
  script's own usage docstring, `scripts/run_crossval_cpu.py:35`. Repo-wide
  (`grep -rn run_crossval_cpu . --exclude-dir=.git`) the other references are documentation —
  `README.md:117`, `docs/public/guide/benchmarks.mdx:203`,
  `docs/guides/reference_lane_contract.md:93` — plus an existence check in
  `tests/contracts/test_crossval_manifest_contract.py:18,451` and a docstring mention in
  `tests/crossval/test_crossval_gate_logic.py:11`. None of them executes it.

That is the whole of #741's remaining item: the largest declared tier has no executor.

It is narrower than it sounds, because two other things already cover most of it.

**8 of the 17 already run weekly.** `validation.yml`'s `crossval-external` job reads
`scheduled_external_order` straight out of the manifest (`:270-295`) and runs 01, 02, 03, 04,
09, 10, 22, 23 against a real conda-forge Meep (`:247`). Those are the Meep-dependent cases,
and they run there *better* than a bare CPU runner could — with the reference present instead
of ENV-SKIPped.

**9 have a `cpu-runner` tier and no recurring executor**: 07, 11, 14, 15, 16, 17, 18, 20, 21.
Of these, 11 has an on-demand VESSL lane (`scripts/vessl_crossval_external.yaml`), and 07,
15, 20, 21 need openEMS and are also tagged `external-manual`. The four that need no external
solver at all and carry `cpu-runner` as their **only** tier are **14, 16, 17, 18**.

**Their gates are still pinned.** Not running a script is not the same as not checking its
numbers. Every one of the nine has at least one unmarked test in `tests/crossval/` that runs
in the fast lane on every PR — `test_crossval_cv14_cavity_gates.py`,
`test_crossval_cv15_wall_planes.py`, `test_rcs_mie_ka_sweep_gates.py`,
`test_rcs_dielectric_sphere_mie_gates.py`, `test_wr90_iris_modematch_gates.py`,
`test_sheen_lpf_palace_referee_gates.py`, `test_cv11_realized_short.py`,
`test_msl_phase_referee_header.py`, `test_coax_two_port_referee_header.py`. (For 14 and 15 the
manifest's own `gate_paths` lists only the script; those two test files exist and are simply
not registered there — a separate small inaccuracy in the manifest.)

So what the missing lane actually costs is: **"the script still runs end to end and still
reproduces its committed artifact"**, not the gate numbers. That is worth having — it is how
a silent breakage in a script's own plumbing gets caught — but it is not the difference
between checked and unchecked physics.

---

## 3. The three candidate lanes

### (a) A GitHub CPU-runner job on the weekly schedule

**What**: one more job in `validation.yml`: checkout, `pip install -e ".[dev]"`,
`PYTHONPATH=. python scripts/run_crossval_cpu.py`.

**Setup cost**: about twelve lines of YAML. No new image, no solver, no secret, no lab GPU
seat, no conda. The runner's exit contract already exists and is contract-tested.

**Minutes**: *not measured*. The bound is 17 scripts × 900 s = 255 min worst case against a
360 min job ceiling. The only per-case CPU figures in the repo are cv02 (~1 min quiet, up to
~4.3 min under contention — `run_crossval_cpu.py:53-58`) and the retired cv05 (~6 min). One
`workflow_dispatch` run gives the real number; guessing it is how shard 3 of the slow-tests
job hit 95 min against a 90 min ceiling on the 08-31 run (`validation.yml:70`).

**What it buys**: the four cases whose *script* has no other executor (14, 16, 17, 18) plus the
rfx side of 07, 15, 20, 21 — their gate tests already run in the fast lane, as §2 says.
**What it does not buy**: on a runner without Meep, cases 01-04/22/23 land in
`SELF-CHECK-ONLY` or `ENV-SKIP` — coverage `crossval-external` already provides properly, so
this job would partly duplicate it at lower quality.

**Risk**: a 17-script serial job on a hosted runner is the same shape as the job that has been
timing out. Measure before putting it on the cron.

### (b) The weekly VESSL a6000 lane with meep and openEMS installed

**What**: drop `$IGN` from `scripts/vessl_validation_lane_a6000.yaml` — the `IGN=` definition
at `:37` **and its two expansions at `:40` and `:44`**, see §4 item 1 for why all three go
together — and add the solvers so the external legs run for real.

**Setup cost**: this is the expensive option, and the reason is a dependency conflict rather
than effort.

- The lane's image is `nvcr.io/nvidia/jax:24.10-py3` (`:7`).
- Meep has no working pip wheel; conda-forge `pymeep` only, pinned `numpy>=1.26,<2`
  (`validation.yml:256`), while rfx/JAX want numpy ≥ 2. `run_crossval_cpu.py:136-141` exists
  precisely because that ABI split is a recurring fact. pymeep therefore needs its own conda
  prefix inside the pod, and the tests would have to run under it.
- openEMS is a source build. The repo already amortizes it into
  `ghcr.io/bk-squared/rfx-openems` (`docker/openems-lane/Dockerfile`, built by
  `.github/workflows/build-openems-image.yml`), but that Dockerfile's own BOUNDARY comment
  (`:17-25`) says it is the solver environment only and must not gain rfx or jax, and its base
  is `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (`:30`), not the jax image this lane needs.

So (b) means either a third image carrying jax-GPU **and** openEMS **and** a pymeep prefix, or
a per-run conda solve on a lab GPU seat.

**Minutes**: on top of the lane's current 7200 s highmem + 13800 s gpu ceilings (`:40`, `:44`),
the four files' own runtime (7.6 min on CPU here, GPU time unmeasured) plus a pymeep conda
solve every Monday unless a new image is built. It also holds one of the two lab GPU seats
longer each week.

### (c) On-demand VESSL only, with the coverage claim corrected

**What**: no new lane. Leave the four files where they are, say in the manifest and the
runbook that their external legs are manual-only, and give the existing on-demand lanes a
written cadence.

**Setup cost**: near zero. `scripts/vessl_crossval_external.yaml` already ships openEMS in its
image and already runs cv11; `scripts/vessl_gpu_suite.yaml` already collects the four files.

**Minutes**: zero recurring.

**What it buys**: honesty, and nothing else. The four files keep not running weekly.

---

## 4. Recommendation

**(a), scoped, with one line of (b) taken and the rest of (b) declined.** In this order:

1. **Drop `$IGN` from `scripts/vessl_validation_lane_a6000.yaml` — the `IGN=` definition at
   `:37` AND both of its expansions, at `:40` (the `highmem` selection) and `:44` (the `gpu`
   selection).** All three, not the definition alone: that run block is `set -eu` (`:19`), so
   deleting `:37` and leaving `$IGN` expanded aborts the lane at its first pytest call.
   Reproduce the abort with `sh -c 'set -eu; echo "using: $IGN"'` → `sh: 1: IGN: parameter not
   set`, exit 2. Do not touch the four CPU jobs, where the `gpu` marker already deselects these
   files and the `--ignore`s are belt-and-braces. That lane already selects `-m gpu`, so
   removing the variable makes all 18 tests run weekly on the a6000 at **no new environment
   cost**: the 5 rfx-only gates execute, and the 13 external legs report as skips instead of
   being invisible. The image has no meep at all, so they hit `ModuleNotFoundError` and skip
   cleanly — the pytest-9 trap in §1 only fires if someone installs a solver with the wrong
   numpy pin. Budget the 7.6 min CPU figure as the upper reference until a GPU number exists.
2. **Measure `scripts/run_crossval_cpu.py` once by `workflow_dispatch`** before it goes near
   the cron. If the wall time fits the schedule, add it as a weekly job and let it be the
   `cpu-runner` tier's executor. If it does not, narrow the job to 14, 16, 17, 18 — the four
   with no external dependency and no other recurring executor for their script — and say so
   in the manifest.
3. **Do not build (b) beyond step 1.** Its only unique coverage is "rfx-vs-Meep and
   rfx-vs-openEMS *inside pytest*", and the repo already gets rfx-vs-Meep weekly from
   `crossval-external` with a real Meep, and rfx-vs-openEMS on demand from
   `vessl_crossval_external.yaml` with a prebuilt openEMS image. Paying for a numpy-split
   conda environment on a weekly GPU seat to duplicate both is the worst cost-per-coverage of
   the three options.

The reason in one sentence: the cheap half of the gap — rfx-only gates that never run on a
schedule — closes by deleting one variable from one lane file, and the expensive half —
external solvers inside pytest — is already covered outside pytest by two lanes that work.

---

## 5. What this note does not decide

- **Splitting the rfx-only gates out of the gpu-marked files** (#717 item 2a). It is a code
  change with a marker policy behind it, and it only matters if step 1 above is rejected.
- **Whether cv14/cv15's `gate_paths` should list their existing test files.** Small manifest
  accuracy fix, separate from the lane question.

---

## 6. Resolved after the first draft (review round, 2026-09-13)

- **The stale sentence in `tests/data/v173a_pre_t7_phase2_baseline.json`** (#717 item 1's
  third bullet) is corrected, in this PR. The first draft deferred it on the grounds that
  `grep -rn v173a_pre_t7_phase2 --include=*.py .` finds no reader. That grep was scoped to
  `*.py`; repo-wide it also hits `docs/design_notes/issue802_807_rasterization_predeclaration.md`
  and `docs/design_notes/931_migration/T7-fixture-purpose-and-engineering-review.md`, the
  second of which quotes the fixture's `_note` directly. Orphanhood was the wrong question
  anyway: the sentence is false whether or not anything reads it. The correction went in as a
  sibling key, `_correction_717_lane_signal`, which is the convention the file already uses —
  the existing `_retired_by_931_fixture_repair` key corrects other claims in the same `_note`
  the same way, and rewriting archived prose in place would break the T7 note's quotation of
  it.

## Evidence

| claim | command |
|---|---|
| per-file collect counts | `python -m pytest --collect-only -q -o addopts="" <file>` |
| 5 pass / 13 need a solver, 459.15 s | the four-file pytest command in §1 |
| importorskip fails on a broken import under pytest 9.1.1 | two-test probe in §1 |
| 17/21 cases declare `cpu-runner` | the one-liner in §2 |
| nothing calls the CPU runner | `grep -rn run_crossval_cpu . --exclude-dir=.git` |
| which cases the weekly external lane runs | `validation.yml:279-304` reading `scheduled_external_order` |
| `$IGN` unset aborts an `set -eu` run block | `sh -c 'set -eu; echo "using: $IGN"'` → exit 2 |
| the fixture note has non-`*.py` readers | `grep -rn v173a_pre_t7_phase2 . --exclude-dir=.git` |
