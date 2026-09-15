# Close the four verification blockers at 877ea558

Work is confined to `rfx-931-fastlane-reds`, branch `feat/931-verify-blockers`,
based on `877ea558e2154701ce457fa0b380c682f9c8e5a9`. Every Python invocation
sets `PYTHONPATH=$PWD` and `JAX_PLATFORMS=cpu`; the installed path package points
elsewhere, so its default import is unsuitable. No other worktree is changed
and no VESSL job is launched. Pytest invocations are serialized with `-n 4`
and a shell `timeout 1200`.

## Cause and correction

P1-1 and P2-3 share a declaration/resolution phase error. They cannot use one
identical guard: registration asks what the caller declared while construction
is incomplete; execution asks whether the completed resolved model supports
the declared solver. They now use the existing mesh abstraction at those
respective boundaries instead of adding exceptions to individual runners.

The constructor's ADI/profile checks could only see explicit profiles.
Adding an FR4 slab later caused automatic mesh resolution to supply a z
profile. `_dispatch_plan` returned a non-uniform Yee lane before it reached
the ADI lane; both `run(skip_preflight=True)` and `forward(skip_preflight=True)`
could complete without honoring `solver="adi"`. The shared dispatch now calls
`_require_uniform_mesh("solver='adi'")` before any lane selection. Automatic
NU ADI models are refused explicitly even without an interior conductor.
Supported uniform models still execute the actual ADI solver. The existing
realized-edge interior-PEC guard is unchanged, as is automatic mesh resolution.

Both DFT-plane and flux registration read `_domain`, which resolves against
the geometry currently registered. One shared coordinate validator now reads
`_declared_mesh` and validates axis and bounds without planning a grid.
On the 10 GHz witness with declared domain `(0.03, 0.03, 0.2)`, a low 1.6 mm
FR4 slab and another beginning at z=0.17 m produce these intermediate views:

| Geometry registered | Declared z extent (m) | Resolved z extent (m) | z=0.15 registration |
|---|---:|---:|---|
| none | 0.2 | 0.2 | accepted |
| low slab | 0.2 | 0.07654811450000004 | accepted |
| both slabs | 0.2 | 0.24654811450000003 | accepted |

z=0.21 is refused at every stage, including after the resolved mesh expands.
The inferred mesh is still used for execution; registration does not freeze
or overwrite it.

P1-4 is the corresponding lifetime error: a preview of a partially built
automatic model is not a stable lattice for geometry derived from that preview.
The PEC-short fixture first chose 2.1414 mm, placed its faces on those nodes,
then adding that body selected 1 mm. The final drawing was no longer on-node.

The explicit construction boundary is now `grid = sim.freeze_mesh()`. It
captures resolved spacing, profiles and extent (including the actual empty-model
default spacing), without overwriting caller declarations. Later geometry or
material additions cannot remesh it; they still face the existing #931 refusals.
Profiles and extent are copied so caller-owned arrays/lists cannot mutate the
frozen lattice. Mesh-input reassignment raises; to remesh, construct a new
Simulation. Ordinary `_build_grid()`/preflight previews remain provisional and
continue to invalidate after edits. This preserves the auto-mesh parity fix.

The short fixture explicitly freezes before deriving both faces and guide-wall
coverage. The normative contract and public guide/API now state this workflow.
The final fixture grid has `dx = 0.0021413747 m`, with faces at `0.085654988 m`
and `0.0899377374 m`: exactly cells 40 and 42 after adding the body.
NU placement must use actual node coordinates; register boundaries and ports
before retaining array indices because padding can change. Domain node positions
remain stable. The existing strict short magnitude gate stays at 0.99.

## Fixture enforcement

[The companion audit](verify_blockers_battery_enforcement.md) lists **each of
the 28 checks**, its measurement, restored pin and disposition. All are
restored from the artifact and exactly equal the historical pins. None owes
a new measurement. The raw artifact is byte-identical; a SHA-256-bound
committed overlay supplies replay metadata, never a newly fitted replay gate.

`cheap_refute_flip_shift_sign` is restored using the actual post-solve modal
translation operator on the stored S matrices. Its unmutated control agrees
with the stored shifted matrices; reversing the signs fails the unchanged
rotation gate on every measurable entry. This recovers the falsifier without
claiming that the missing VESSL refute stage ran. The census is exactly the
historical key set minus the six authorized lossless-leg removals: **179
verdicts, 131 pass, 48 original report-only rows, zero fail and zero owed**.
Schema-4 missing pins or a missing falsifier now yield `owed`.

## Regression coverage

`tests/contracts/test_declared_solver_and_monitor_domain.py`: **36 cases**.
Twelve exercise both real entry points with automatic NU meshes, with and
without preview, and with no PEC, a sheet or a volume. The exception assertion
contains only the execution call, preventing a later `_build_grid()` refusal
from being mistaken for an execution refusal. Four supported uniform cases
execute and spy on the actual ADI solver; two cover distributed dispatch.
Twelve monitor cases cover both APIs at each geometry stage with/without
preview and forbid resolution inside registration. Six cover declared bounds
on all axes, both endpoints, and invalid axes.

`tests/oracle/test_waveguide_chain_battery_v18_close.py`: **59 added cases**.
Every restored check has an independent bad-measurement mutation that must
fail and a missing-pin mutation that must report owed. Additional tests check
the pin derivation against both the artifact and historical thresholds and
exercise the production sign-flip falsifier with its unmutated control, and
assert the raw artifact reports all 29 owed items. The
existing ingest test now locks the retained key set, not only its size.

`tests/contracts/test_frozen_mesh_construction.py`: **7 added cases**. The
actual short fixture's faces must be integer node coordinates and span exactly
SHORT_CELLS, with realized wall coverage checked under automatic and explicit
spacing. Both real execution entry points must preserve a lattice frozen before
the first body. Other checks refuse a sub-cell body, preserve automatic NU
profiles/extent after added geometry, and protect against mutation of caller
profiles/domain and reassignment of mesh inputs. The full oracle short test also
runs with its unchanged 0.99 gate.

## Verification record

Current verification logs/XML are under `output/verify-blockers/`.
`pytest-timeout` is unavailable; shell `timeout 1200` enforces the limit.
All invocations use `PYTHONPATH=$PWD JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1` and at most `-n 4`. The earlier attempt left a running
`full-default` pytest at this same checkout. Its timeout parent and process
group were verified and stopped before new pytest execution; its partial output
is not evidence for this patch.

The initial three-file regression selection passed **318 tests** in 10.01 s.
A subsequent review tightened frozen extent ownership against mutable caller
sequences; final verification of that change is recorded below.

The blast radius is the entire repository default lane (`tests`, marker
expression `not gpu and not slow and not slow_physics`), partitioned with
`--splits 4 --group N` and executed sequentially. The common selection was:

```sh
PYTHONPATH=$PWD JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 1200 python -m pytest tests -n 4 -q --splits 4 --group N \
  --tb=short --durations=15 --junitxml=output/verify-blockers/blast-N.xml
```

For partitions 3, 4 and the final partition-1 repeat, the local diagnostic
plugin `-p output.verify_blockers_progress` additionally wrote per-test
start/setup/call/teardown events under `output/verify-blockers/`. It observes
reports without changing verdicts or execution. Partition 1 was repeated after
the final frozen-snapshot ownership and public-facade changes; the original
logs are retained as `blast-1-initial.*`.

| Final partition | Passed | Runtime skipped | Existing xfailed | Command exit |
|---|---:|---:|---:|---|
| 1 | 2124 | 7 | 1 | 0, 218.20 s |
| 2 | 1905 | 3 | 17 | 124 at shutdown, complete XML |
| 3 | 1977 | 15 | 4 | 0, 957.04 s |
| 4 | 1170 | 0 | 9 | 124 at finalization, complete per-test events |
| **Unique test total** | **7176** | **25** | **31** | **0 failed/error test reports** |

One additional collection skip is `tests.unit.api.test_visualize_3d` because
`plotly` is absent. It appears in each partition's collection and is counted
once here. Thus the ordinary partition-1/2/3 summaries show 8/4/16 skipped.
Existing runtime skips include the deprecated RIS workflow, platform-sensitive
knife-edge cases, and the subprocess-only RSS helper. No skip or xfail marker
was added or altered.

Fresh full collection (`python -m pytest tests --collect-only -q`, same CPU
environment and `timeout 1200`) selects **7,232 of 7,633 tests**, with **401
deselected** by the unchanged default markers. The audit compares that complete
node-ID set against partitions 1–3 XML and partition 4's per-test records:
**7,232 selected = 7,232 completed, zero missing, zero extra, zero duplicate**.
`output/verify-blockers/coverage-audit.json` contains the exact census. This is
a full default-lane run; no proximity filter or additional exclusion was used.

The final regression invocation selected these four paths, with the same CPU
environment, `timeout 1200`, `-n 4 -q --tb=short`:

```text
tests/contracts/test_frozen_mesh_construction.py
tests/contracts/test_declared_solver_and_monitor_domain.py
tests/oracle/test_waveguide_chain_battery_v18_close.py
tests/oracle/test_waveguide_port_validation_battery.py::test_pec_short_s11_magnitude
```

Result: **319 passed**, zero skipped/xfail/fail, in 13.57 s. It includes all
**102 added cases** (36 declaration/solver + 7 mesh construction + 59 battery)
and the original PEC-short oracle on the final public facade.

Partition 2 completed **1,905 passed, 4 skipped, 17 existing xfailed** in
1198.67 s and wrote complete XML with zero failures/errors. The shell timeout
expired during process shutdown (exit 124); this is recorded rather than
called a clean command exit. Its 1,925 selected test IDs are checked against
the 1,925 test cases in XML (plus one collection skip). In particular,
`test_pec_short_s11_magnitude` passed in 8.440 s with its unchanged gate.
Subsequent partitions log individual start/report events as well as XML.

Partition 4 likewise completed all 1,179 test calls and teardowns before the
1200 s deadline, but timed out during finalization before XML/summary output.
Every per-test setup/call/teardown report is preserved in
`blast-4.events.jsonl`; the full-collection audit verifies their completeness.
Both timed commands returned 124, so neither is described as a clean command
exit. No failed/error test report occurred, and the final regression command
returned 0. GPU, slow and slow_physics selections remain outside this run.

The public API inventory gained only `freeze_mesh`. The API-reference surface
check and public site-map check pass. pdoc was installed into an isolated
environment under `output/verify-blockers/`, and the generated reference both
passes `check_api_reference.py --html-dir ...` and includes `freeze_mesh`.
The public method delegates to the shared mesh implementation; making the
facade explicit also prevents pdoc from hiding a method inherited from a
private mixin. This final facade is covered again in the final regressions.

Repository lint (`ruff check rfx/ tests/ validation/ --select E,F,W --ignore
E501,F401,E741,E731,E701,E702,E402`) reports two pre-existing F841 errors:
`validation/research/convergence_floor/fixture.py:170` and
`validation/research/multiband_nu/w4r_port_supraconvergence.py:124` both assign
unused `dzf`. Both lines are identical at `877ea558`; these unrelated files
are unchanged. The same lint gate on all `rfx/`, all `tests/`, and the changed
measurement driver passes. `git diff --check` passes.

## Constraints and audit

Local `docs/agent-memory/` is absent; no sibling checkout supplied substitute
memory. The external memory registry was consulted only for the prior RFX
source-evidence review standard, not as evidence about this branch.
The normative conductor note's section 1.7 says the realized edge
owner is the only geometry-to-edge source, and the ADI guard note says
"neither dropping metal nor silently reducing dt is an acceptable remedy."
These fixes are consistent with both: no conductor realization or solver
equation changes. The existing fixture ingest note requires preserving raw
bytes and the unexplained slab coarse/flux drift; both remain intact.

The slab coarse/flux difference from the historical fixture remains
unexplained as previously recorded; this patch makes no new claim about it.
The raw artifact's `provenance.run_id` remains `UNSET-see-log-filename`, as in
the original ingest; no provenance field is silently rewritten. GPU and
high-memory VESSL lane results are outside this CPU verification. The local
interpreter is JAX 0.6.2 on CPU, and `rfx.__file__` resolves to this checkout.
