# cv05 leaves the scheduled crossval cases (issues #959, #965)

**Correction (2026-09-10, later the same day):** the first version of this note said
`05_patch_antenna` was *removed* from `validation/crossval/manifest.json`'s `cases` array. That was tried and
reverted within the hour: `test_manifest_covers_every_crossval_script_exactly_once` requires
every non-underscore `.py` file under `validation/crossval/` to have exactly one manifest case,
and the script stays (see "What changed" below for why), so deleting the case broke that
contract test along with two others. The case is **retained**, with its scheduled tiers
replaced by `external-manual` and its `cpu_runner` entry replaced by an `excluded_reason` —
functionally the same outcome (nothing schedules it), reached the correct way. Read "What
changed" for the corrected mechanism and what else that surfaced.

An independent audit asked which of the 21 crossval cases should exist at all. cv05 ranked
first for removal on a manifest-only signal: 20 of 21 cases list their own crossval script in
`gate_paths`; cv05 does not — its four `gate_paths` entries are all test files. This note
records what was actually checked before acting on that signal, what changed, and — because
the signal turned out to be incomplete in an instructive way — what did not change and why.

## The manifest-only signal pointed at the right case for an incomplete reason

`gate_paths` for `05_patch_antenna` lists `tests/locks/test_patch_edgefed_resonance_harminv.py`,
`tests/locks/test_patch_edgefed_s11_passivity.py`, `tests/oracle/test_patch_cavity_eps_oracle.py`
and `tests/crossval/test_patch_canonical_farfield_e4.py`. Read individually, none of the four
runs cv05's own declared board:

- the two `test_patch_edgefed_*` files measure a different geometry, `DX = 0.197 mm` exactly,
  "Board S", a 44x51-cell patch — their own docstrings say so;
- `tests/crossval/test_patch_canonical_farfield_e4.py` measures openEMS's canonical thirds-rule tutorial
  recipe — a 32x40 mm patch, eps_r = 3.38, 1.524 mm substrate, 60x60 mm ground — not cv05's
  29.5x38 mm FR4 board;
- `tests/oracle/test_patch_cavity_eps_oracle.py` measures an idealized closed PEC cavity, 24x24x12 mm,
  eps_r = 4.0 — not patch-shaped at all.

All four are real, useful regression locks or oracles on their OWN boards. None of them checks
cv05's specific probe-fed 29.5x38 mm FR4 patch with the NU-graded z-mesh.

**What actually checks cv05's own board is `tests/crossval/test_cv05_realized_sheet_planes.py`
— and it is not in `gate_paths` either.** It runs `validation/crossval/05_patch_antenna.py` via subprocess in
`RFX_CV05_BUILD_ONLY=1` mode, which exits (`SystemExit(0)`, around line 686) after building the
full NU-graded z-mesh and running `assert_realized_sheets` on both builds, strictly before any
`openEMS`/`CSXCAD` import (those start at line 758+, reached only in full-run mode). So the
graded-z construction is exercised, the sheet-plane realization is checked, and the known
28x37 mm-from-29.5x38 mm footprint gap (issue #959) is pinned as a stated fact, all without
touching openEMS.

**So `gate_paths` was wrong in both directions for this case**: it names four tests that check
other boards, and omits the one test that checks this one. That is a manifest-accuracy defect
in its own right, not just a reason cv05 happened to be findable. Whether the same pattern
exists for other cases is not evaluated here; the field's accuracy is worth a repo-wide check
before it is trusted as a signal again.

## What changed

`validation/crossval/05_patch_antenna.py` stays at its current path, unmodified — five real
functional dependents (below) hard-code that path or its declared constant names, and a repo
contract, `test_manifest_covers_every_crossval_script_exactly_once`, requires every
non-underscore `.py` file under `validation/crossval/` to correspond to exactly one manifest
case. Deleting the case while keeping the file is therefore not a legal state; it was tried
first and reverted (see the correction notice above).

**What actually retires cv05 from automated scheduling:** the case entry stays in
`validation/crossval/manifest.json`'s `cases` array, with `execution_tiers` changed from `["cpu-runner",
"vessl-external"]` to `["external-manual"]`, and `cpu_runner` changed from `{"order": 3}` to
`{"excluded_reason": "..."}` (same shape cv06b/cv19/cv24 already use for CPU-infeasible cases —
not a new pattern). `scripts/run_crossval_cpu.py` builds `CPU_SUBSET` from `cpu_order` values
only, so a case with `excluded_reason` instead never enters it:
```
$ python3 -c "import sys; sys.path.insert(0,'scripts'); import run_crossval_cpu as m; \
  print('05_patch_antenna.py in CPU_SUBSET:', 'validation/crossval/05_patch_antenna.py' in m.CPU_SUBSET); \
  print('05_patch_antenna.py in EXCLUDED:', '05_patch_antenna.py' in m.EXCLUDED)"
05_patch_antenna.py in CPU_SUBSET: False
05_patch_antenna.py in EXCLUDED: True
```
`public_group` ("B") and `role` ("diagnostic-reporter") are unchanged, matching cv07/cv15/cv20/
cv21's existing `external-manual` diagnostic-reporter pattern — this is a well-trodden manifest
shape, not an invented one. `docs/public/guide/benchmarks.mdx` needed no edit as a result:
`test_public_validation_docs_match_manifest` reads `role`, and it did not change.

**Consequence: `cpu_runner.order` needed renumbering, contradicting earlier advice not to.**
Orders were `[1,2,4,5,...,18]` (17 values, 3 missing) when the case still held slot 3.
`test_manifest_entries_are_self_consistent_and_grounded` asserts `sorted(cpu_orders) ==
list(range(1, len(cpu_orders) + 1))` — no gaps, ever — which `scripts/run_crossval_cpu.py`'s own runtime
sort tolerates but this separate contract test does not. Advice earlier in this thread said the
gap was harmless and not to renumber; that was checked against the runner's behavior, not this
test, and does not hold once cv05's slot is vacated by `excluded_reason` rather than filled by
another case. Renumbered orders 4-18 down to 3-17 (each case's id-to-relative-order mapping
otherwise unchanged — verified below), 17 sed replacements done through a collision-proof
two-pass placeholder substitution, not by hand.

**The `vessl-external` openEMS tier is a separate lever from the manifest, and editing the
manifest case (whether by deletion or by the `excluded_reason` correction above) does not touch
it — the first version of this note said the manifest edit alone would free it, and that was
wrong.**
`scripts/vessl_crossval_external.yaml` does not enumerate from the manifest; it hardcodes its
two scripts (`grep -nE 'timeout [0-9]+ .*validation/crossval/.*\.py' scripts/vessl_crossval_external.yaml`
found exactly two matches, one each for cv05 and cv11). Removing the manifest case would have
left this yaml still running cv05 for up to 7200 s on every submission. Fixed by editing the
yaml to drop the cv05 invocation and its verdict-block references (`cv05_rc` etc.), leaving
cv11's block untouched; the description no longer names cv05. This file is git-tracked (unlike
several other `vessl_*.yaml` scratch files in `scripts/`, which are gitignored and never made it
into version control) — `git ls-files scripts/vessl_crossval_external.yaml` returns it, so the
fix is a normal, reviewable commit, not a local-only edit.

**That grep also answered a question worth recording: the external tier's only two occupants
are cv05 and cv11 — both on the audit's MOVE list.** If cv11 also leaves crossval, this lane
retires entirely rather than shrinking to one case; worth knowing before cv11's disposition is
decided.

`validation/crossval/05_patch_antenna.py` itself is **unchanged and stays at its current
path**, exactly as this note originally intended even though the manifest mechanism for
achieving that changed. Five things in this repo depend on that exact path or its declared
constant names, and none of them care whether the manifest schedules the case or excludes it —
they read the script file directly, not the manifest:

1. **`tests/crossval/test_cv05_realized_sheet_planes.py`** — the actual gate on cv05's board
   (see above). Runs the script by subprocess path.
2. **`scripts/diagnostics/build_cv05_ringdown_spectra.py`** — execs the script in a fresh
   globals dict at five patch lengths (`RFX_CV05_PATCH_L_MM`), catches the `SystemExit` the
   script raises when openEMS is absent, and reads the rfx-only ring-down results (PART 1) out
   of the script's globals. Regenerates
   `tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json`, which the GATED
   `tests/crossval/test_patch_mode_identification.py` suite depends on. This path needs openEMS
   to be ABSENT, not present, so retiring the openEMS leg does not affect it — deleting or
   relocating the script would.
3. **`tests/crossval/test_patch_mode_identification.py::test_cv05_constants_in_this_file_are_the_scripts_own`**
   — text-scans the script's source for `C0`, `eps_r`, `h_sub`, `L`, `W`, `HARMINV_F_LO`/`_HI`
   to keep a hand-copied dict in the fixture-builder's caller in sync with the script's own
   declared geometry.
4. **`tests/_example_fidelity_lib.py`** — a registry entry (`"module_level_solve"`) checked by
   the example-fidelity contract suite.
5. **`tests/contracts/test_two_plane_gone_from_docs_and_scripts.py`** and
   **`tests/crossval/test_lattice_ownership_crossval_witness.py`** — allowlist entries in two
   banned-keyword contract scans, both keyed on this exact path.

None of these five require cv05 to be a scheduled crossval case. All five require the file to
keep existing with its current Part-1 (pre-openEMS) behavior and constant names.

**"Five dependents" is an undercount of everything that names this path — say so plainly, so
the next person greps once, not twice.** Beyond the five functional dependents above, ten more
files reference `validation/crossval/05_patch_antenna.py`, none of them scheduled, all
consistent with keeping the file where it is:

- `scripts/vessl_crossval05_openems.yaml`, `scripts/vessl_crossval05_mesh_sweep.yaml`,
  `scripts/vessl_cv05_openems_and_ringdown_fixture.yaml`,
  `scripts/physics_gate_port_external_wire.yaml` — manual/on-demand diagnostic yamls, run by
  hand, not by any scheduler.
- `scripts/archive/vessl_issue31_crossval_regression.yaml`,
  `scripts/archive/vessl_v150_gpu_validation.yaml`, `scripts/archive/vessl_issue31_patch_viz.yaml`,
  `scripts/archive/20260715_cv05_325_crossval/vessl_cv05_325.yaml` — archived, historical.
- `scripts/_gallery_v3_patch_figs.py`, `scripts/archive/issue48_uniform_patch_ffrp.py`,
  `scripts/diagnostics/build_coaxial_line_openems_broad_comparison.py`,
  `examples/tutorials/nonuniform_patch_demo.py` — prose/comment mentions only ("matches",
  "mirrors", "identical to"), no import or subprocess.
- `tests/oracle/test_patch_cavity_eps_oracle.py` — a docstring lineage note (`gate_paths`
  delegate patch-accuracy evidence "HERE"), not a functional read of the script.
- `tests/unit/nonuniform/test_patch_uniform_fine_substrate.py` — explicitly does NOT read the
  script ("not an importable function, so this test HAND-COPIES" the geometry, a named,
  self-acknowledged drift risk independent of this disposition).

## What did not change, and why

`docs/design_notes` gained a link, not a rewrite. In particular:

- The three other `gate_paths` tests (`tests/locks/test_patch_edgefed_resonance_harminv.py`,
  `tests/locks/test_patch_edgefed_s11_passivity.py`, `tests/oracle/test_patch_cavity_eps_oracle.py`) never depended on
  cv05 being a scheduled case — they are independent boards. No change.
- `tests/crossval/test_patch_canonical_farfield_e4.py` is likewise independent (different geometry). No
  change.
- The two committed `_05_patch_results/cv05_run_openems_*.json` legs and their logs are
  retained as historical record (README added in the same directory, same convention as
  `validation/crossval/_15_patch_results/README.md` and
  `validation/crossval/_06b_msl_notch_results/README.md`).

## Issue #959 (the 28x37 mm vs 29.5x38 mm geometry mismatch) is not fixed here

Disposition was settled before touching #959's geometry, on purpose: fixing a comparator that
is about to stop being scheduled would have been wasted work if the disposition had gone the
other way. Checked for a live consumer of the mismatched number outside the retiring leg and
found none — `tests/crossval/test_patch_canonical_farfield_e4.py` uses an unrelated geometry, and
`tests/crossval/test_patch_mode_identification.py`'s fixture pipeline uses the rfx-only Part 1 engine with no
openEMS comparison, so it is blind to #959 by construction, not by luck. #959 stays open,
re-scoped: the comparison it names is retired, not fixed.

## Do any of the deselected "slow" tests touch cv05's board?

No. Of the 34 tests across the four `gate_paths` files plus `tests/crossval/test_cv05_realized_sheet_planes.py`,
23 are collected by default and 11 are marked slow and deselected. All 11 are in
`tests/locks/test_patch_edgefed_resonance_harminv.py` (4), `tests/locks/test_patch_edgefed_s11_passivity.py` (1),
`tests/oracle/test_patch_cavity_eps_oracle.py` (1) and `tests/crossval/test_patch_canonical_farfield_e4.py` (5) — the four
files already established above as measuring OTHER boards, not cv05's. Zero of
`tests/crossval/test_cv05_realized_sheet_planes.py`'s 11 tests are marked slow; all ran in the 23-test pass.
So the deselection is a real gap in coverage of those other boards' physics, but not a gap in
coverage of cv05's board specifically.

## cv15's claim_scope

`validation/crossval/manifest.json`'s cv15 `claim_scope` said patch accuracy "remains
DELEGATED to id 05 and the committed tests that case names ... which stay authoritative." The
tests stay authoritative and unchanged; "id 05" no longer names a scheduled case. Updated the
clause to delegate to the named tests directly and note that cv05's file is retained,
unscheduled — see the diff for the exact wording; the surrounding paragraph is otherwise
unchanged.
