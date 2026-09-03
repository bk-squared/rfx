# cv19 FDFD unitarity witness: what 1.4655e-09 actually is, and what the gate should be

Issue #884. Diagnosis only — **no test or fixture change is proposed for merge here.**
This note records the measurements, the verdict, and a proposal with its falsifier so
the change can be reviewed on the evidence rather than on the red mark.

Branch: `agent/issue-884-diagnosis`.

---

## 0. The headline, first

The cv19 FDFD comparator **contains no JAX**. `validation/crossval/comparators/fdfd_hplane.py`
imports `numpy`, `scipy.sparse` and `scipy.sparse.linalg` and nothing else ("Zero rfx
dependency by design: numpy and scipy only", module docstring line 67). The solve is
`scipy.sparse.linalg.spsolve` — a **direct** sparse LU (SuperLU), `complex128`, no
iteration, no tolerance, no XLA, no `jax.config.x64` on the path at all.

So issue #884's second hypothesis — "the newer XLA changes the solve itself (different
reduction or fused path in the linear solve)" — has no mechanism. There is no XLA in
this solve to change. The first hypothesis is the correct one, and this note shows it
is not merely plausible but **measured**: the committed `1.4655e-09` is a roundoff
realization of an ill-conditioned direct factorization, not a property of the method.
The method's own unitarity is **3.06e-14**.

---

## 1. Reproduction

`self_test(a=22.86e-3, f=11.0 GHz, base_cells=90, r=1, apertures=[40,26,24,26,40],
cavities=[56,62,62,56], t=8, margin=45)` — exactly the call in the test.

| build | `empty_s11` | `unitarity` |
|---|---|---|
| **committed** (`fixture.json`, `rfx.json`) = CI py3.10 / jax 0.6.2 / Linux x86-64 | `4.998689747642886e-14` | `1.4655321400880439e-09` |
| CI py3.11 / jax ≥0.7 / Linux x86-64 (from #884) | (not reported; test named `unitarity`) | `1.5806898456816043e-08` |
| local `~/Documents/rfx/.venv` — py3.11, **jax 0.10.2**, numpy 2.4.6, scipy 1.17.1, arm64/Accelerate | `4.66988620937557925e-14` | `2.57592525088057300e-09` |
| local `venv062` — py3.11, **jax 0.6.2**, numpy 2.2.6, scipy 1.15.3, arm64/Accelerate | `4.66988620937557925e-14` | `2.57592525088057300e-09` |

Also identical on both local venvs, to the last bit: `empty_s21 - 1 = +3.77475828372553224e-15`,
`|S11| = 2.31011821637078313e-01`, `|S21| = 9.72950944724267131e-01`, `nx=90, nz=366,
unknowns=32663, h=2.54e-4, metal_nodes_z=9`.

**Which local number matches CI's 3.10 value: neither.** The two local venvs are
**bit-identical to each other** and both differ from the committed/CI-3.10 number by
0.2449 decades. Applying the separation rule stated in the assignment:

* two local venvs agree ⇒ **the JAX/numpy/scipy version effect on this quantity is exactly zero** —
  which is what a numpy/scipy-only code path predicts;
* both local venvs disagree with CI-3.10 ⇒ **platform** (arm64/Accelerate vs x86-64/OpenBLAS).

The CI-3.10 → CI-3.11 gap is on one platform, so it *is* a library-build effect there —
but the libraries that can move it are numpy and scipy (the wheels' bundled SuperLU and
BLAS), **not** jax. The 3.11 lane installs unpinned, so it resolves newer numpy/scipy as
a side effect of the newer Python; #883's stated purpose (surfacing JAX-version effects)
is not what fired here.

The test **passes on both local venvs** (`1 passed in 1.45s` / `1.49s`):
decades = 0.2449 < 1.0. The CI 3.11 failure reproduces arithmetically —
`|log10(1.4655e-9) − log10(1.5807e-8)| = 1.0328513161589736`, the exact number in the
traceback — but the *condition* does not reproduce on this platform.

---

## 2. What the residual measures, and what the solver's criterion is

### 2.1 The quantity

`self_test` (fdfd_hplane.py:184) computes, on the **loaded** structure:

```
u = | |S11|^2 + |S21|^2 - 1 |
```

with `S11 = <phi_1, E[:,0]> h - 1` and `S21 = <phi_1, E[:,-1]> h exp(+gamma_1 nz h)`,
two length-89 inner products of the TE_10 modal profile against the first and last
node columns of the solved field. It is **not** a linear-algebra residual. It is the
lossless-two-port power identity — a *physics* identity that the discrete formulation
either satisfies or does not.

### 2.2 The solve, and its "tolerance"

```
E = spl.spsolve(A.tocsc(), rhs)          # fdfd_hplane.py:161
```

* **Direct**, not iterative: SuperLU sparse LU with partial pivoting.
* **dtype `complex128`** (float64 real and imaginary parts) throughout — matrix, rhs and
  solution. Confirmed by inspection at runtime. `float32` never appears; `jax.config`
  `x64` is irrelevant because jax is not imported.
* **There is no convergence tolerance.** A direct factorization has no iteration count
  and no stopping criterion. The only tolerances in the file are the *gate* thresholds
  `empty_tol=1e-12` and `unitary_tol=1e-6` (fdfd_hplane.py:169) — acceptance thresholds
  on the answer, not criteria the solver drives itself to.

Measured on the committed configuration (identical on both venvs):

| quantity | value |
|---|---|
| unknowns N | 32663 |
| nnz | 167221 |
| ‖A‖∞ | 1.239471e+08 |
| ‖x‖∞ | 4.796599e+01 |
| ‖Ax−b‖∞ | 4.191043e-06 |
| **normwise backward error** ‖Ax−b‖∞/(‖A‖∞‖x‖∞) | **7.049402e-16 ≈ 3.2 eps** |
| est. cond₁(A) | **9.85e+11** (0.10.2 run) / 1.01e+12 (0.6.2 run) |
| cond₁(A)·eps | **2.19e-04** |

The factorization is **backward stable to 3 ulp**. It is also solving a system with
condition number ~10¹², so the a-priori forward-error scale is 2.2e-04 — *seven decades
above* the observed unitarity residual. That is the whole story in one line: **a strict
derived bound from the tolerance is useless here** (2.2e-04 ≫ the 1e-6 gate), and
correspondingly the observed 1e-9 is not "at the solver's tolerance" — the solver has no
tolerance, and its honest error bar is enormously wider than the number recorded.

### 2.3 The decisive measurement: the method's unitarity is 3.06e-14

Iterative refinement on the same LU factor, with the residual accumulated exactly
(`math.fsum` per row), converges the solution and with it the unitarity:

| | unitarity |
|---|---|
| raw `spsolve` (the committed path) | 2.57592525088057300e-09 |
| + 1 refinement step | 2.42028619368284126e-14 |
| + 2 | 1.87405646556726424e-13 |
| + 3 | 2.23154827949656465e-14 |
| exact-residual refine ×3 | **3.06421554796543205e-14** |

Bit-identical on both venvs. **The discrete formulation is unitary to ~3e-14, i.e. to
machine precision.** Five decades of the committed `1.47e-09` are LU roundoff amplified
by cond(A) ≈ 10¹². The witness was measuring the factorization's luck, not the physics.

### 2.4 Two independent confirmations that it is roundoff, on one build

**(a) Fill-reducing ordering.** `permc_spec` selects a column permutation. All four
choices solve the *same* system and are mathematically identical. On a single build:

| `permc_spec` | unitarity | ‖Δx‖∞/‖x‖∞ vs COLAMD |
|---|---|---|
| COLAMD (scipy default — the committed path) | 2.57592525088057300e-09 | 0 |
| MMD_AT_PLUS_A | 5.94043947366174052e-09 | 5.64e-08 |
| MMD_ATA | 1.78573380527069503e-08 | 5.87e-08 |
| NATURAL | 1.96889057280102975e-07 | 2.56e-07 |

**Band: [2.58e-09, 1.97e-07] — 1.883 decades wide, within one process, one build, one
platform.** The CI 3.10→3.11 gap the test calls a failure is 1.033 decades: *smaller
than the spread a free solver choice produces on a single machine.* Both the committed
`1.4655e-09` (0.24 decades below the band's floor) and CI-3.11's `1.5807e-08` (inside
the band) are ordinary members of this distribution.

**(b) One ulp in the DtN block moves it.** `Q = (phi.T * exp(-gt h)) @ phi * h`
(fdfd_hplane.py:138) is the only dense GEMM on the path — an 89×89 complex matmul that
goes to BLAS. Recomputing the identical product with numpy's own loop
(`einsum(..., optimize=False)`, and independently a row-wise sum) differs from the BLAS
result by `1.11e-16` absolute, `3.04e-16` relative — **one ulp on one entry**:

| Q built by | unitarity | \|S11\| |
|---|---|---|
| BLAS zgemm (as shipped) | 2.57592525088057300e-09 | 0.231011821637 |
| numpy einsum loop | 3.05603553574229636e-09 | 0.231011816963 |
| row-wise sum | 3.05603553574229636e-09 | 0.231011816963 |

Thread count (`OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`) does not change it here,
but the mechanism is exhibited: a one-ulp difference in the assembled matrix — exactly
what a different BLAS or a different scipy wheel gives you — moves the witness. The
number is a function of the *build*, not of the discretization.

### 2.5 The empty-guide witness is the same class of quantity

`empty_s11` over the same four orderings: 4.66988620937557925e-14 (COLAMD),
7.15497180242101622e-14, 1.29046719130623963e-13, 3.37108313563769565e-13 (NATURAL) —
**0.859 decades of spread**, against a 1.0-decade gate whose current margin (committed
vs local) is 0.0295 decades. cond₁(A_empty) ≈ 4.01e+04, four decades better conditioned
than the loaded problem, which is why it has not fired yet. **It is the same bug, one
build away from firing.** Any fix should treat both witnesses.

---

## 3. Verdict: which reading holds

> "polished evidence" (the committed number was tuned/lucky on one build), or
> "the solve changed"?

**Neither, precisely — and the closer of the two is "polished/lucky on one build."**

* "The solve changed" is **ruled out with a mechanism, not by absence of evidence**:
  the solve contains no JAX/XLA, and the two local venvs spanning jax 0.6.2 → 0.10.2 and
  numpy 2.2.6 → 2.4.6 and scipy 1.15.3 → 1.17.1 produce **bit-identical** output. There
  is nothing for a newer XLA to have changed. (No HLO diff is presented because no HLO
  exists for this code path; that is the finding, not an omission.)
* "Polished" in the pejorative sense — someone tuned the number — is **not** supported
  either. `1.4655321400880439e-09` is a faithful, honestly recorded reading of the
  generation machine.
* What actually happened: **the witness records a quantity that has no build-independent
  value.** Its legitimate spread is 1.88 decades from solver-internal choices alone
  (§2.4a) and it moves on one ulp of the assembled matrix (§2.4b). Recording it to 17
  significant figures and gating a re-run to one decade is a category error: it commits
  a sample from a distribution as if it were a constant. The committed value was *lucky*
  in the narrow sense that it sat 0.24 decades below the band floor, which bought the
  gate its slack until a build finally sampled 10.8× higher.

**Consequence for the number itself: `1.47e-09` was never a property of the method.**
The method's unitarity is `3.06e-14` (§2.3). Per the assignment's own decision rule —
"if the residual is orders of magnitude below the solver's own tolerance, then it was
never a property of the method and the witness should be a bound derived from the
tolerance" — the residual is **not** below any solver tolerance (there is none) but it
is **five decades above the method's true unitarity and seven decades below the
solve's honest forward-error bar of cond·eps = 2.19e-04**. It sits in neither place a
recorded measurement can be meaningful. It must be replaced.

---

## 4. Proposal

Replace the single recorded scalar with **three assertions, none of which records a
roundoff realization**. Notation: `u_raw` = unitarity of `spsolve`'s answer;
`u_ref` = unitarity after iterative refinement on the *same* LU factor,
defined deterministically as `min` over refinement steps 1 and 2 (the minimum before
refinement stagnates at the residual-evaluation floor).

Cost: `splu` + 2 triangular solves = **+4 ms on a 0.12 s solve (3%)**. No second
factorization is needed.

### U1 — the method's unitarity, gated by a *derived* bound

```
assert u_ref < 1e-11
```

**Derivation.** `S11` and `S21` are length-(nx−1) = 89 inner products of O(1) terms, so
their own evaluation floor is ≈ √89·eps·|S| ≈ 2.1e-15; `u` inherits
2(|S11|+|S21|) ≈ 2.41 times that, ≈ 5e-15, and the refined solve adds the
residual-evaluation floor of the same order. `1e-11` is **~2000× that analytic floor**
and **~150× the worst value measured** over 4 orderings × 2 refinement steps × 2 venvs
(worst single step 3.92e-13; worst by the min rule 6.93e-14).

Measured `u_ref` (min rule), identical on **both venvs**: COLAMD 2.42e-14,
MMD_AT_PLUS_A 3.60e-14, MMD_ATA 6.93e-14, NATURAL 2.26e-14 — **0.486 decades of total
spread**, versus 1.883 for `u_raw`. This is the quantity that is a property of the
method; it is version- and platform-independent because the refinement removes the
factorization's roundoff realization from it.

### U2 — the diagnosis itself, as a same-run relative check

```
assert u_raw / u_ref > 100
```

This asserts *the finding of this note*: the sweep-visible residual is conditioning
noise, not method error. If the discretization ever stops being unitary, `u_ref` rises,
the ratio collapses, and this note's premise is flagged as expired. Measured ratios:
1.06e+05 (COLAMD) … 8.69e+06 (NATURAL); worst case over the whole ensemble,
`min(u_raw)/max(u_ref)` = **3.72e+04, a 372× margin** on the threshold (6.57e+03, still
66×, if `u_ref` is taken as the worst single refinement step rather than the min rule).
Secondary
check — it is the weakest of the three, since no hard lower bound on `u_raw` is
derivable; the threshold is set two decades under the worst observation.

### U3 — anti-polishing, one-sided, floor derived in-run

The existing decade test exists because "an independently designed mutation set
`empty_s11` to 1e-16 and `unitarity` to 1e-12 and the whole suite stayed green". That
job must survive. It does not need a *two-sided* comparison — only a floor:

```
assert committed_unitarity > 1000 * u_ref          # a recorded value must look like a
assert committed_unitarity < unitary_tol           # realization, not a machine-eps number
```

The committed value must be a *plausible LU realization* of a cond≈10¹² solve. Observed
realizations sit 4.8–6.9 decades above `u_ref`; the floor sits at 3 decades, i.e. **1.8
decades below the lowest realization ever observed** (1.4655e-09, the committed one) and
**1.4 decades above the polishing attack**. Measured: floor = 1000 × 2.4203e-14 =
**2.4203e-11**; committed 1.4655e-09 clears it by 60×.

The two-sided decade comparison on `unitarity` is **deleted**. §2.4 shows it compares
two samples from a 1.88-decade distribution against a 1.0-decade window; it has no
discriminating power to lose. `empty_s11` should get the same treatment (§2.5) — its
current margin is 0.0295 decades against an 0.859-decade intrinsic spread.

The fixture should record `unitarity_lu_realization` (renamed, with a note that it is
ordering-, BLAS- and platform-dependent with a measured 1.88-decade spread) and
`unitarity_refined`, and gate only the latter.

### (a) The proposal passes on both venvs

All three, both venvs, bit-identical:

| | jax 0.10.2 / np 2.4.6 / sp 1.17.1 | jax 0.6.2 / np 2.2.6 / sp 1.15.3 |
|---|---|---|
| `u_raw` | 2.5759e-09 | 2.5759e-09 |
| `u_ref` | 2.4203e-14 | 2.4203e-14 |
| U1 `u_ref < 1e-11` | PASS (413× margin) | PASS (413×) |
| U2 `ratio > 100` | PASS (1.06e+05) | PASS (1.06e+05) |
| U3 `committed > 2.4203e-11` | PASS (60×) | PASS (60×) |

And it passes on the CI-3.11 number that currently fails: `1.5807e-08 > 2.4203e-11`. ✔

### (b) It still fails on deliberate defects

Falsifiers run on both venvs; results bit-identical. `u_ref` is the min-rule value.

| defect | `u_raw` | `u_ref` | U1 | verdict |
|---|---|---|---|---|
| *(reference)* as committed | 2.5759e-09 | 2.4203e-14 | PASS | — |
| ε_r = 1 + 1e-15j (loss) | 9.0687e-09 | 1.4100e-13 | PASS | below detection |
| **ε_r = 1 + 1e-13j (loss)** | 2.5640e-09 | **1.6693e-11** | **FAIL** | **caught** |
| ε_r = 1 + 1e-12j | 1.9128e-08 | 1.6793e-10 | FAIL | caught |
| ε_r = 1 + 1e-11j | 2.7988e-09 | 1.6787e-09 | FAIL | caught (U2 too: ratio 1.67) |
| ε_r = 1 + 1e-09j | 1.6758e-07 | 1.6788e-07 | FAIL | caught |
| ε_r = 1 + 1e-06j | 1.6790e-04 | 1.6789e-04 | FAIL | caught |
| missing `/h` in `discrete_gamma` (the historical bug) | 8.6597e-15 | 2.2204e-16 | pass | caught by U2 (ratio 39.0) **and** by `empty_s11` = 1.000000 |

**Detection power, stated as a number.** The proposed U1 catches a lossy permittivity
from **Im(ε_r) ≳ 1e-13**. The gate it replaces (`u_raw < 1e-6`) is blind until
Im(ε_r) ≈ 1e-6 gives `u_raw = 1.68e-04`; at 1e-9j `u_raw` is 1.68e-07, still passing.
**The proposal is roughly four decades more sensitive to the defect unitarity exists to
catch.** That is the argument that it is not a widening.

**The honest negative result — the geometry off-by-one.** Perturbing the first aperture
from 40 to 42 cells:

| | `u_raw` | `u_ref` | \|S11\| |
|---|---|---|---|
| reference | 2.5759e-09 | 2.4203e-14 | 0.231012 |
| aperture 40→42 | 3.2754e-08 | 3.6082e-14 | 0.164100 |

**No form of the unitarity witness catches it, and none should**: the perturbed filter
is a *different but still perfectly lossless* structure, so |S11|²+|S21|²=1 continues to
hold exactly. The same is true of a real (lossless) permittivity perturbation — at
ε_r = 1.01, |S11| moves 0.2310 → 0.1819 while `u_ref` stays 4.91e-14. Anyone reading
the committed 1.47e-09 as evidence that the *geometry* is right has misread it.

What does catch the off-by-one is the **other half of the same test**, the live r=2
spot-frequency anchor at index 65 (11.05 GHz):

| | committed r=2 `s11[65]` | live | \|Δ\| | gate |
|---|---|---|---|---|
| reference | 0.1532810000 | 0.1532810941 | 9.41e-08 | `abs=1e-5` ✔ |
| aperture 40→42 | 0.1532810000 | 0.1628341537 | **9.55e-03** | **955× over — CAUGHT** |

That anchor is the geometry gate; the unitarity witness is the losslessness gate; the
empty-guide witness is the port-transparency gate. The three are not interchangeable,
and the current test's decade comparison was doing none of the three.

### (c) It is not a widening

The one number that moves *upward* is the abandoned two-sided comparison, which is
deleted rather than loosened — and it is replaced by a bound (`u_ref < 1e-11`) that is
**3.6 decades tighter than the quantity it replaces** and, as measured above, ~4 decades
more sensitive to real loss. `1.58e-08` is not accommodated by widening a threshold to
reach it; it is reclassified as what it is — an unremarkable member of a 1.88-decade
roundoff distribution — and then not gated as an equality at all.

---

## 5. Bearing on PR #883

**This red is pre-existing and unrelated to #883's changes.**

* #883 touches `.github/workflows/pr-tests.yml`, `rfx/boundaries/cpml.py`,
  `scripts/diagnostics/probe_fed_msl_openems_referee.py` and four `tests/unit/...`
  files. It touches **neither** `validation/crossval/comparators/fdfd_hplane.py`,
  **nor** `tests/crossval/test_wr90_iris_filter_gates.py`, **nor**
  `tests/fixtures/wr90_iris_filter/fixture.json`.
* The failing solve imports only numpy and scipy, so nothing in #883's CPML, stencil,
  DFT or mesh changes can reach it.
* The failure is a property of the *matrix* the 3.11 lane's numpy/scipy wheels assemble
  and factor. #883 did not create it; it **exposed** it, by adding a lane that installs a
  different resolved dependency set. The latent defect — a roundoff realization committed
  as a constant — has been in the fixture since cv19 was generated.

**Recommendation.** #883's own six fixes are verified and orthogonal; blocking it on a
pre-existing latent defect it merely revealed is the wrong trade, since the 3.11 lane is
exactly the instrument that found it. Merge #883 with the 3.11 job present, and either
(a) land the §4 witness first as a small dependent PR, or (b) mark this one test
`xfail(strict=False)` on 3.11 with a link to #884 for the days between. **Do not
re-record `1.4655321400880439e-09` as `1.5806898456816043e-08`** — that swaps one
sample of a 1.88-decade distribution for another and re-arms the same failure on the
next wheel.

---

## Appendix: reproduction

Both venvs, py3.11.2, arm64 macOS (Darwin 25.5.0), BLAS = Accelerate:

* `~/Documents/rfx/.venv/bin/python` — jax 0.10.2, numpy 2.4.6, scipy 1.17.1
* `…/scratchpad/venv062/bin/python` — jax 0.6.2, numpy 2.2.6, scipy 1.15.3

```
python -m pytest -q -p no:cacheprovider \
  "tests/crossval/test_wr90_iris_filter_gates.py::test_fdfd_solver_gates_run_live_and_committed_curves_are_reproducible"
# -> 1 passed on BOTH venvs (decades = 0.2449 < 1.0)
```

Measurement scripts (conditioning, ordering ensemble, refinement, BLAS-ulp probe and the
falsifier sweep) were run out of tree and are not committed; every number in this note is
reproducible from the recipe above plus the ordering/refinement definitions in §2.3–2.4
and §4.

*Caveat on scope:* all local measurements are arm64 macOS. The claim that the CI
3.10→3.11 move is a numpy/scipy-build effect rests on (i) the proof that no JAX code is
on the path and (ii) the demonstrated one-ulp sensitivity of the witness, not on a Linux
A/B. The proposal is designed to be insensitive to that distinction, so confirming it on
Linux is a nice-to-have rather than a precondition.
