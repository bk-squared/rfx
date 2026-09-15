# Re-pin the qualified repaired MSL board

Decision, 2026-09-09. This closes T11's MSL pickup condition using the landed
four-run evidence. No VESSL run was launched or restarted for this transfer.
The short disposition remains the separate closure in T12.

The existing four-run adjudicator reports `eligible_for_repin: true` and
`failures: []`. It rechecks source hashes at the exact capture SHA,
`b36fc46cdf21d1c57f221e6a057654bcad60bae2`, CPU/complex64 configuration,
four distinct names and IDs, the fixed board/source/reference planes, exact
mesh bisection, and the unchanged physical and convergence screens.

| Role | VESSL ID | Archived report |
|---|---|---|
| Base, 12 periods | 369367259636 | [base.json](fixture-repair-live-evidence/msl-base/base.json) |
| Independent confirmation | 369367259638 | [confirm.json](fixture-repair-live-evidence/msl-confirm/confirm.json) |
| Long, 24 periods | 369367259637 | [long.json](fixture-repair-live-evidence/msl-long/long.json) |
| Refinement, 2x / 12 periods | 369367259648 | [refine.json](fixture-repair-live-evidence/msl-refine/refine.json) |

The refinement archive includes source/run witnesses, resources, return codes,
summary, complete compressed logs and JUnit. Its runner/refine/pytest return
codes are zero and JUnit records 11 passed, 0 failed/errors/skipped. These are
landed-run counts, distinct from the local post-pin verification below.

| Comparison, all 40 complex entries | Maximum absolute delta | Mean absolute delta | Existing budget |
|---|---:|---:|---|
| Base/refine raw | 0.007206651301290537 | 0.0030316962211016643 | <=0.02, rtol=0 |
| Base/refine projected (additional evidence) | 0.007206838874606767 | 0.0030314413405571096 | No replacement for the raw gate |
| Base/confirmation projected | 0 | 0 | atol=.002, rtol=.005 |
| Base/long raw | 2.457562461863554e-7 | See per-entry report | atol=.002, rtol=.005 |
| Base/long projected | 5.331201499700045e-7 | See per-entry report | atol=.002, rtol=.005 |

Every role's physical qualification failures are empty. All bins are reliable;
no beta rail is present. Base drive settling is -103.77145626460994 and
-104.00676059515744 dB; refine is -103.37339248568554 and
-103.56844140913047 dB. The old-golden maximum gap is
0.2030584600560962 for base and 0.19956575038187566 for refine.

This is convergence within the predeclared absolute complex mesh budget for
the repaired board. Refinement does not remove the old-golden discrepancy.
The source and geometry explain why: historical replay retains uniform 80 um
cells and a trace plane realized at 320 um, while the final live coupon uses
50 um x/y cells, substrate z cells (50,50,38.5,38.5,38.5,38.5) um and a PEC
sheet at the actual 254 um interface. Scalar 254/3 um belongs to an intermediate
fixture/helper, not these final captures. The new pin must describe the board
actually run. This does not assert zero discretization error or asymptotic order.

The [full adjudication](fixture-repair-live-evidence/msl-final-adjudication.json)
retains all per-bin complex traces, model witnesses and per-entry budgets.
The [inspection figure](fixture-repair-live-evidence/msl-final-adjudication.png)
was rendered and inspected: transmission stays near unity with phase tracking
the drawing model; reflection is small and decreases under refinement, while
confirmation and long curves overlap base. This supports the qualified
mesh/temporal interpretation rather than relying only on the headline delta.

## Transfer and values

Added `scripts/diagnostics/import_931_msl_golden.py` as an explicitly offline
writer. It verifies the adjudicator/input fingerprints, recomputes the unchanged
comparisons, checks the landed refinement completion, and verifies the old pin
before writing. No call to `capture_msl_e2e_golden.py --write-golden` was made.
The candidate is **base S_projected**, losslessly cast to complex64 with shape
(2,2,10). Reloading the saved pin gives exact equality to all 40 base entries
and passes all 40 independent confirmation entries at rtol=.005, atol=.002.

`tests/fixtures/msl_s_matrix_golden.json` contains the exact base value pairs,
frequencies, SHA, source hashes, four run names/IDs, named report fingerprints,
full adjudication, mesh comparison, old-pin comparison, completion evidence,
and offline verification. The `.npy` SHA256 is
`4776a6fd73d725206339c2310ffe4983d0522d551668fc666b44ca4a422d4d88`.

The table displays the newly pinned values to nine decimals; the JSON preserves
all exact float32 values and actual float32-rounded frequencies.

| Nominal GHz | S11 | S12 | S21 | S22 |
|---:|---|---|---|---|
| 0.5 | -0.000148714-0.001634501j | +0.997521579-0.070190713j | +0.997522235-0.070195414j | -0.000086827-0.001639247j |
| 1 | -0.000582695-0.003202289j | +0.990126848-0.140050784j | +0.990127146-0.140050143j | -0.000336873-0.003237003j |
| 1.5 | -0.001292944-0.004656300j | +0.977837801-0.209219545j | +0.977837622-0.209224120j | -0.000746174-0.004775152j |
| 2 | -0.002253026-0.005956233j | +0.960712373-0.277369887j | +0.960712552-0.277372628j | -0.001308034-0.006231772j |
| 2.5 | -0.003434688-0.007069722j | +0.938847184-0.344152510j | +0.938845336-0.344156086j | -0.001994053-0.007598192j |
| 3 | -0.004794240-0.007951498j | +0.912337184-0.409241915j | +0.912337422-0.409242094j | -0.002806150-0.008846242j |
| 3.5 | -0.006295800-0.008575082j | +0.881310284-0.472322106j | +0.881310403-0.472321689j | -0.003723571-0.009961635j |
| 4 | -0.007908940-0.008918196j | +0.845916927-0.533072650j | +0.845919907-0.533068061j | -0.004724942-0.010942355j |
| 4.5 | -0.009569377-0.008959919j | +0.806336761-0.591190338j | +0.806347370-0.591185868j | -0.005773496-0.011761438j |
| 5 | -0.011213183-0.008703768j | +0.762772918-0.646374822j | +0.762772679-0.646394312j | -0.006866343-0.012436256j |

## Verification and scope

Commands run from this worktree (the adjudicator and writer are offline):

```bash
JAX_PLATFORMS=cpu python scripts/diagnostics/adjudicate_931_msl_live.py \
  --base docs/design_notes/931_migration/fixture-repair-live-evidence/msl-base/base.json \
  --confirm docs/design_notes/931_migration/fixture-repair-live-evidence/msl-confirm/confirm.json \
  --long docs/design_notes/931_migration/fixture-repair-live-evidence/msl-long/long.json \
  --refine docs/design_notes/931_migration/fixture-repair-live-evidence/msl-refine/refine.json \
  --output docs/design_notes/931_migration/fixture-repair-live-evidence/msl-final-adjudication.json
JAX_PLATFORMS=cpu python scripts/diagnostics/import_931_msl_golden.py \
  --adjudication docs/design_notes/931_migration/fixture-repair-live-evidence/msl-final-adjudication.json
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -o addopts= -n 4 -v --capture=tee-sys -rA \
  --junitxml=output/931-msl-repin-verification/junit.xml \
  tests/unit/autodiff/test_msl_sparam_ad.py
python -m ruff check scripts/diagnostics/import_931_msl_golden.py
git diff --check
```

Fresh local verification: **16 passed, 0 failed, 0 errors, 0 skipped**, with
23 warnings, in **4890.85 s (81m30s)**. This includes the slow end-to-end
FDTD lock and all replay, AD, geometry and qualification cases in the module.
The live raw qualification failures are empty. Its maximum projected delta
from the new pin is **5.331201499700045e-7** across all 40 entries, at the
unchanged rtol=.005, atol=.002. The full live trace was inspected and agrees
with the archived base curve within that delta.
The 23 warnings include the preserved historical replay/synthetic AD null,
beta-rail and impedance advisories, the declared lossless sheet/dielectric,
and the live passivity projection (raw maximum singular value
1.0001287460327148, below the unchanged 1.05 qualification ceiling).
The historical/synthetic warnings do not describe the qualified live board.

[Local verification summary](fixture-repair-live-evidence/msl-repin-verification/summary.json),
[provenance](fixture-repair-live-evidence/msl-repin-verification/provenance.json),
[complete pytest log](fixture-repair-live-evidence/msl-repin-verification/pytest.log.gz),
and [JUnit](fixture-repair-live-evidence/msl-repin-verification/junit.xml.gz)
are archived alongside the four-run evidence. JAX 0.6.2, NumPy 2.2.6, CPU,
x64 disabled; one pytest invocation with at most four workers.
The historical base lane's 11 passes / 1 old-pin failure remains unchanged in
its original archive; this fresh live pass resolves that drift-lock red.
No MSL re-pin item remains open. The full repository suite was not run.

Four offline writer checks passed: the qualified transfer reproduces the pin;
an ineligible adjudication, changed input fingerprint, and widened recorded
mesh gate each fail before writing. Ruff and `git diff --check` pass.
The test module's executable AST is unchanged after removing docstrings;
only its explanation/comments changed. No tolerance, marker or assertion was
changed. `rfx/` and both historical 80 um replay binaries remain unchanged
relative to starting commit 294038ca646a4e841b208bea8c9a6522b73de615.

No repo-local durable-memory directory exists in this worktree; no other
worktree was consulted. This decision is consistent with T11: "Use the base
run 369367259636 S_projected complex64 array" and "Preserve rtol=5e-3,
atol=2e-3". The landed four-run pass supplies the required new evidence.
R2: one local live verification, no parameter search or repeated failed
physics intervention. Cheap falsifiers are saved-base/pin inequality, a
rejected negative-control acceptance, or failure of the unchanged live gate.
