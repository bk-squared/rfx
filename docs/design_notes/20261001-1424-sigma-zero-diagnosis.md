# F2 #1424 sigma-at-zero raw-coefficient measurement

Interpretation: **lead fills**.

Step 1 used the lossless point fixture, float64, 2,400 steps, CPU only.
The three Ca and three Cb arrays each have shape (4, 4, 4). Both differentiated
functions receive those identical resolved edge arrays directly. The test-only
hook captures the production context before its scan; no production code changed.
Autodiff differentiates the ordinary production core with eight checkpointed
segments. F2 differentiates `design_adjoint_scan` with its custom VJP.

The reported max relative difference is max absolute difference over all three
components divided by peak absolute autodiff gradient over all three components,
separately for Ca and Cb. Component-relative values use each component's own peak.

| Gradient | Max absolute difference | Autodiff peak | Max relative difference |
|---|---:|---:|---:|
| gCa | 0.00017021145529196589 | 0.0043979031557148003 | 0.038702865721539761 |
| gCb | 3.8818913361549079e-12 | 0.0083774265615138999 | 4.633751555624982e-10 |

| Gradient | x relative | y relative | z relative |
|---|---:|---:|---:|
| gCa | 0.017807351064151256 | 0.020621159645917604 | 0.09286820550686804 |
| gCb | 4.360396606422256e-10 | 1.965395876124624e-10 | 4.633751555624982e-10 |

| Objective | Value |
|---|---:|
| Autodiff | 0.016414876930560818 |
| F2 | 0.01641487693056077 |

The diagnostic's 1e-5 gradient assertion failed on gCa; the objective-value
assertion passed. Pytest: 1 failed in 6.95 s. `git diff --check` passed.
The failing diagnostic is retained as the measured witness.

## File:line evidence

- `tests/unit/autodiff/test_reciprocity_adjoint.py:303`: test-only context capture.
- `tests/unit/autodiff/test_reciprocity_adjoint.py:314`: resolved arrays selected.
- `tests/unit/autodiff/test_reciprocity_adjoint.py:317`: identical coefficient inputs installed.
- `tests/unit/autodiff/test_reciprocity_adjoint.py:320`: F2 scan.
- `tests/unit/autodiff/test_reciprocity_adjoint.py:322`: ordinary production core.
- `tests/unit/autodiff/test_reciprocity_adjoint.py:329`: checkpointed autodiff scan.
- `tests/unit/autodiff/test_reciprocity_adjoint.py:334`: direct coefficient differentiation.
- `tests/unit/autodiff/test_reciprocity_adjoint.py:337`: error computation.
- `rfx/adjoint.py:204`: coefficient-gradient overlap expression.

## Stop and remaining requested outputs

Stopped after step 1 under the user's instruction: “If (1) shows gCa wrong,
STOP and report”. Classification of the discrepancy: **lead fills**.
Step 2 traced-zero/traced-0.2 branch comparison: not executed after this stop.
Step 3 chain fix: none. Full G1 after fix: not run; no fix was made.
Lead decision: **lead fills**.

Raw log: `/private/tmp/f2-1424-followup/step1.log`.
Command (from this worktree):

```sh
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" /Users/byungkwankim/Documents/rfx/.venv/bin/python -m pytest tests/unit/autodiff/test_reciprocity_adjoint.py::test_g1_raw_coefficients_lossless_point -q -s
```

Starting HEAD: `655dafe46a57e23ff57e8e9df98282abb78e8280`.
Step 1 commit: this document and the diagnostic are committed together; the
hash is recorded in `/private/tmp/f2-1424-followup/report.md` after committing.
Steps 2/3 commits: none (stop condition).

`git rev-parse --git-common-dir` output:

```text
/Users/byungkwankim/Documents/rfx/.git
```

Writes were confined to this worktree, `/private/tmp`, and the authorized git
metadata. No push, PR, comments, VESSL, or GPU execution.
