# Non-uniform directional autodiff (AD-Q / AD6) — pre-declaration

**Status:** pre-declaration; this commit precedes code and measurement.
Every numeric window below is FROZEN. One attempt per arm; a fired
window is recorded, never widened. Results are appended below.
**Class:** independent central-FD directional derivative of the existing
NU realized-cell-vector loss, including the cells controlling dt.
**Tree:** rfx-ad-quant, feat/nu-ad-directional, base origin/main
fa3929136159e0a644783b56012b21c1bce21340. Date: 2026-09-13 (KST).
**Provenance:** import printed and verified as
`/Users/byungkwankim/Documents/rfx-ad-quant/rfx/__init__.py`.
Every JSON records rfx_file and git_sha; execution is CPU, float32,
using PYTHONPATH=/Users/byungkwankim/Documents/rfx-ad-quant and
/Users/byungkwankim/Documents/rfx/.venv/bin/python.

## 0. Existing evidence (read-only verification before measurement)

The committed W7 JSON gives:

| axis | ties | attempt 1 worst | attempt 2 worst resolved | reverse vs forward worst | max all JVP | max non-tied JVP |
|---|---:|---:|---:|---:|---:|---:|
| x | 8 | 0.0043092729 | 0.0058954637 | 0.0083603605 | 991.527954 | 10.044477 |
| y | 8 | 0.5224044348 | 0.0358454787 | 0.0000191379 | 585.089722 | 0.779206 |
| z | 8 | 0.9923850266 | 0.0478644736 | 0.0000308796 | 946.350403 | 0.312869 |

Attempt 1 fires on y/z; attempt 2 does not. These records and the strict
xfail remain unchanged. AD-vs-AD shares the min convention and cannot
validate that convention independently. Non-tied per-cell checks do not
cover the tied gradient mass.

Loss0 = 0.13063086569309235, ulp = 1.4901161193847656e-8.
For relative h the absolute cell displacement is h*d[k]; the gradient
quantum is ulp/(2*h*d[k]), not ulp/(2*h). Attempt 1's worst y cell k=10
has d=0.0009606982646 m, quantum 0.0077553805 and 7 quanta;
z k=35 has d=0.0009935915172 m, quantum 0.0074986355 and 3 quanta.
The 5%-of-JVP-max thresholds are 0.0389603 and 0.0156435, only about
5.0 and 2.1 such quanta. The new ladder will independently check the
resolution explanation; these stored values are not a new attempt.

## 1. Fixtures and directions (FROZEN)

Reuse exactly AD1's A1 MB s=2 dz vector and committed AD1_EPS_PATTERN
(material fixed by node index), and AD3's A3 x/y/z vectors and loss.
Both use AD_N_STEPS=120, PEC, existing source/probe/waveform, float32.
No rfx behavior changes. Material stays attached to node index.

Flatten A3 in x,y,z order; AD1 contains only z. Tied masks are exact
float32 equality to each axis minimum (assert the expected 8/8/8 and 2).
Per fixture test the following, in this order:

- Indicator e_tied for each axis, then joint e_tied (joint omitted for
  AD1 because it is identical to its sole axis direction).
- All-ones for each axis, then joint all-ones (same omission for AD1).
  This checks the chain-rule identity g dot 1 = dL(d+t*1)/dt.
- k=8 random Euclidean unit vectors, numpy default_rng seed=20260913,
  reset per fixture; draw standard normals in flattened order, replace
  entries on each tied set by their arithmetic mean, then normalize.

Indicators and ones retain their literal amplitudes, not unit normalization.
For every direction v, displacement is delta*v with
`delta = h*min(d)/max(abs(v))`, h in (0.001, 0.002, 0.004).
Use host float64 base builder vectors, cast perturbed inputs to float32
as the existing instrument does. Report v, delta, plus/minus losses,
central FD and quantum ulp(loss0)/(2*delta) at all three steps.

Directions varying inside a tied set are excluded explicitly: the two
one-sided derivatives of min disagree there. They are a subgradient
question, not silently passing directions. For one tied cell, in the
one-sided derivative limit, JAX's convention is
`g_ad = FD+ + (FD- - FD+)/n_tied`. A per-cell optimizer breaking the tie
must use the appropriate one-sided slope; the equal split need not
predict its actual finite update. Constant-on-tie layer variables avoid
this ambiguity. This tests realized cells, not differentiation of the
host-side band builder or moving material interfaces.

## 2. Reference, gates and coverage (FROZEN)

Let D1,D2,D4 denote FD at the three steps. Reference
R12=(4*D1-D2)/3; independent step check R24=(4*D2-D4)/3.
Conservative one-ulp-per-loss-difference resolution is
qR=(4*q1+q2)/3; report abs(R12)/qR float32 quanta.
Below **50 quanta**: INCONCLUSIVE, never pass/fail.
Richardson reference check requires both abs(R12-D1)/abs(R12) and
abs(R12-R24)/abs(R12) <= **0.05**; otherwise INCONCLUSIVE
(truncation/reference-limited). This is a consistency check, not a
claim to demonstrate asymptotic order in float32.
For every resolved, consistent reference, require
abs(g dot v-R12)/abs(R12) <= **0.15** and identical nonzero signs.
Nonfinite loss/gradient/reference is FIRED. No adaptive steps or reruns.

Coverage is **||P_span(V_pass) g||_2 / ||g||_2**, per fixture, not the
sum of overlapping directional projections and not the norm of the
set of touched cells. Compute orthonormal basis by SVD of the passing
vectors as columns, rank cutoff 1e-12 times largest singular value.
Report rank, full norm, projected norm, residual norm, coverage, and
also tied-gradient norm fraction and the maximum possible projection
onto the entire constant-on-tie subspace. No coverage threshold is
assumed: a small fraction is a result and limits the usable claim.

## 3. Attempt-1 resolution ladder (FROZEN)

Within AD6, independently recompute per-cell central FD on every
non-tied y/z cell at relative h=(0.001,0.003,0.01,0.03), exactly the
existing attempt-1 rule: dominant >5% of largest non-tied abs(FD).
Report all FD arrays, cellwise quanta and errors, dominant indices,
worst error and its cell/quantum/quanta, resolved/unresolved counts at
50 quanta and worst resolved error. Compare h=0.001 to the committed
attempt-1 worst errors at relative 1e-6; a moved pin is a STOP, never
adjusted. Resolution explanation is supported if the original firing
reproduces on below-floor cells and the h=0.03 dominant set clears the
15% sign gate; otherwise state which numerical evidence refutes it.
Large-step agreement alone does not prove absence of every AD defect.

## 4. Tree gates and reporting

Before and after: full tests/unit tests/contracts with -q -o addopts=
-m "not gpu and not slow and not slow_physics" -p no:cacheprovider;
also explicitly tests/unit/autodiff and the W7 replay file, same options.
Ruff rfx/ tests/ validation/ --select E,F,W
--ignore E501,F401,E741,E731,E701,E702,E402. Counts and failure sets
reported. Existing pins and strict xfail untouched. Commit code before
measurement for clean provenance. No push or PR.

## Results

Pending the single declared attempt.
