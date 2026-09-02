# Issue #811 — dz-only waveguide S-matrix dispatch: pre-declaration

Date: 2026-09-01 (KST). Tree: worktree at `150bffb4` (origin/main tip).
Written BEFORE any falsifier arm ran and BEFORE the fix was applied.

## Defect, verified live on this tree

`rfx/api/_sparams.py:2144` gates the non-uniform waveguide S-matrix lane on

```python
if self._dx_profile is not None or self._dy_profile is not None:
```

`dz_profile` is missing, so a simulation whose only profile is `dz_profile`
is silently solved on the uniform grid built from the scalar `dx`, while
`preflight()` builds the non-uniform grid and describes the graded mesh the
solve never uses. Verified empirically on this tree before any change: a
dz-only WR-90 two-port sim with `Simulation._compute_waveguide_s_matrix_nu`
monkeypatched to raise a sentinel completed
`compute_waveguide_s_matrix(n_steps=1, normalize="flux")` without the
sentinel firing — the NU lane was never entered.

The same two-profile gate is mirrored (with the byte-identical fence
message, enforced by
`tests/unit/sparams/test_sparameter_support_contract.py::test_waveguide_nu_fence_message_parity_sparams_vs_preflight`)
in `rfx/api/_preflight.py:2595`
(`_validate_waveguide_sparameter_request_for_preflight`).

### Class scan (fix the class, not the instance)

Every other multi-profile dispatch predicate in the package was read with
context on this tree. All of them test `dz_profile` too:

- `rfx/api/_sparams.py` siblings at 2966, 4242, 5126, 5472, 5955, 6583
  (msl / mixed / coax family lanes) — dz first operand, all correct.
- `rfx/api/_execute.py` 205, 2347 (`_dispatch_plan` for `run()`/`forward()`),
  `rfx/api/_compile.py` 523, `rfx/optimize.py` 359, 719,
  `rfx/visualize.py` 482-483, 786-787,
  `rfx/api/_preflight.py` 1197, 2188-2189, 2452, 2524, 2633, 3046, 3786, 4354
  — dz present in every condition.
- `rfx/api/__init__.py:606` checks dx/dy only, deliberately: it is the ADI
  guard, and ADI rejects `dz_profile` separately one line above its dx/dy
  check (`solver='adi'` raises on nonuniform dz; ZCZ is the scheme name,
  not a z-graded lane — corrected in the review round). Per-axis by
  design, not a member of this defect class.

The defect class is exactly the two sites above. Reachability of the sibling
gates was probed empirically on this tree: minimal dz-only fixtures raise the
documented errors for `compute_coaxial_s_matrix` (NotImplementedError,
"uniform Yee lane only"), `compute_coaxial_line_reflection` /
`compute_coaxial_two_port` (ValueError, "dx_profile, dy_profile, and
dz_profile are not supported"), and `compute_mixed_s_matrix`
(NotImplementedError, "supports the uniform mesh only").

## R1 memory citations

1. `docs/agent-memory/rfx-known-issues.md:23` (primary checkout):
   "🔴 #811 compute_waveguide_s_matrix ignores dz_profile — graded-z
   declared, uniform solved (silent wrong answer)". This work is the fix for
   that active ledger row — consistent.
2. `docs/agent-memory/rfx-known-issues.md:1010`: "a uniform-valued
   `dz_profile` tests the NU plumbing, never the NU metrics". Consistent:
   every falsifier below that claims physics sensitivity uses genuinely
   different graded z meshes; the one uniform-valued profile (the dy shim in
   F2) is declared a PLUMBING witness only.
3. Workspace memory `feedback_generalize_dont_debug_per_example`: "class
   defect ⇒ default/architecture/detector, never fixture-by-fixture".
   Consistent: the fix lands at the dispatch gate plus its preflight mirror,
   and a source-level class lock (AST scan over the dispatch family) plus a
   contract row per public S-matrix entry point guard the class, not the one
   fixture.
4. `CLAUDE.md` (rfx) "No silent gate loosening": no committed gate or
   tolerance is touched by this work; F3 below is an order-of-magnitude
   plausibility window on a new measurement, not a gate move.
5. `docs/agent-memory/index.md` / known-issues comparator rule: no external
   solver is involved here; both sides of every comparison are rfx runs on
   one geometry, so the comparator is `np.array_equal` / `max|dS|` on
   identical-shape arrays.

## Planned change (declared before implementation)

1. `rfx/api/_sparams.py`: add `self._dz_profile is not None` to the
   waveguide NU dispatch gate (house style of the six siblings); extend the
   fence message to name dz; extend the method docstring, the historical
   dispatch comment, and the NU helper's normalize=False message.
2. `rfx/api/_preflight.py`: identical gate change and BYTE-IDENTICAL fence
   message change in the preflight mirror.
3. Docs that describe the old behaviour: `docs/guides/support_matrix.md`
   (waveguide NU row), `docs/guides/sparameter_support_matrix.md`
   ("Nonuniform transverse mesh" block), and
   `docs/guides/sparameter_support_matrix.json` (waveguide `known_limits`).
   Each keeps a history line pointing at #811 and states that dz-graded
   ACCURACY evidence is still pending (#810) — dispatch is fixed, the
   observable is not thereby validated.
4. New `tests/unit/nonuniform/test_dz_only_dispatch_contract.py`: a dz-only contract row for
   every public `compute_*` S-matrix entry point plus `run()`/`forward()`
   lane selection, a no-FDTD dispatch witness (sentinel monkeypatch), a
   fail-loud check for the default `normalize=False` on dz-only, the AST
   class lock, and a slow_physics falsifier asserting two genuinely
   different z meshes change the answer. Existing
   `test_sparameter_support_contract.py` NU-waveguide fixtures get
   parametrized over dy-only AND dz-only profiles so the fence-message
   parity binds on dz too. No module-level x64 anywhere.

Behaviour change to be documented loudly: dz-only +
`compute_waveguide_s_matrix` with the DEFAULT `normalize=False` flips from
silently-wrong uniform numbers to `NotImplementedError` (the NU lane's
existing scope); dz-only + `normalize=True`/`'flux'` single-mode now runs
the NU solve, so those numbers MOVE — that movement is the fix.

## Falsifiers (declared before any arm ran)

Instrument: `scripts/diagnostics/wr90_dz_dispatch_falsifier.py` (committed
with this note, before any run). One WR-90 two-port geometry
(a=22.86 mm, b=10.16 mm, domain x=0.10 m, dx=1 mm, CPML x / PEC y,z,
cpml_layers=20, eps_r=2.2 slab from x=0.045 to 0.055 m spanning the full
cross-section, TE10 ports at x=0.015/+x and 0.085/−x, reference planes
0.020/0.080, freqs = linspace(8.2, 12.4 GHz, 9),
`compute_waveguide_s_matrix(num_periods=20, normalize="flux")`).

Arms (identical geometry; only the mesh differs; profiles sum EXACTLY to
b=10.16 mm; adjacent ratios ≤ 1.4):

- `U` — no profiles (uniform lane; equals what every pre-fix dz-only run
  actually got).
- `A` — `dz_profile` = 10×0.40 mm + 3×0.52 mm + 2×0.70 mm + 4×0.80 mm
  (19 cells, fine→coarse).
- `B` — reversed(A) (same cells, mirrored placement).
- `C` — 6×0.80 mm + 4×0.62 mm + 4×0.72 mm (14 cells; min cell 0.62 mm, so
  its dt differs from A/B's by ≈1.55×).
- `A_shim` — A plus uniform-valued `dy_profile = full(23, 1.0 mm)` (equal to
  the dy the NU lane synthesizes itself). PLUMBING witness only (memory
  citation 2).

### F1 — dz-only arms must stop being bit-identical

Post-fix, every pair among {A, B, C} must satisfy
`np.array_equal(S_i, S_j) == False`. Reported per pair: `max|dS|` over all
(receiver, driver, bin) entries AND the per-bin `|dS11|` table (9 bins).
Baseline expectation, declared now: on the PRE-fix tree, A, B, C (and U) are
bit-identical to one another — running the same instrument before the fix
must reproduce the defect signature (exit code 2). This is the falsifier's
resolving-power check.

### F2 — shim agreement (plumbing witness ONLY)

Post-fix `max|S(A) − S(A_shim)|` expected exactly 0.0; PASS tolerance 1e-6
(a few float32 ulps of headroom). This witnesses that an explicit
uniform-valued dy equals the synthesized dy on the same lane — plumbing, not
metric accuracy (memory citation 2).

### F3 — the fixed dz-only answer must move off the uniform answer plausibly

`max_bin |S11(A) − S11(U)|` must land in the order-of-magnitude window
[1e-5, 1e-1]. Context, not a gate: the issue measured 1.1486e-3 for the
dispatch flip on ITS meshes (finer, different grading; measure:
max abs difference of complex S11 over its band); our arms are coarser and
differently graded, so only the order of magnitude is declared. A value
below 1e-5 would mean dz still is not reaching the solve (F1 would also
fire); above 1e-1 would mean the NU solve is broken, not merely dispatched.
Per-bin `|dS11|` reported verbatim either way.

Additional provenance witness: post-fix arm U must be bit-identical to the
PRE-fix baseline's arm A (the uniform lane is untouched by this diff; the
old dz-only "answer" WAS the uniform answer). Checked across the two JSONs.

### F4 — nothing else moves

Only the enumerated stale guards/docs change and no committed pinned value
moves. Commands (declared):

```
pytest tests/unit/nonuniform/test_dz_only_dispatch_contract.py tests/unit/sparams/test_sparameter_support_contract.py \
       tests/contracts/test_support_matrix_parity.py tests/contracts/test_evidence_citation_pointers.py -q
pytest tests -k "waveguide and (sparam or s_matrix or dispatch or nonuniform)" -q
pytest tests/unit/nonuniform/test_dz_only_dispatch_contract.py -m slow_physics -q
ruff check rfx/ tests/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402
```

### Falsifier run commands (declared)

```
# pre-fix baseline (defect signature expected, exit 2):
python scripts/diagnostics/wr90_dz_dispatch_falsifier.py \
    --arms U,A,B,A_shim \
    --out scripts/diagnostics/wr90_dz_dispatch_falsifier_prefix_baseline.json

# post-fix battery (all falsifiers expected PASS, exit 0):
python scripts/diagnostics/wr90_dz_dispatch_falsifier.py \
    --arms U,A,B,C,A_shim \
    --out scripts/diagnostics/wr90_dz_dispatch_falsifier_results.json
```

Scope statement: the numbers these runs produce are dispatch evidence, not
validated S-parameters. Run length (num_periods=20) is common to every
compared arm, so truncation is common-mode; no ring-down witness is claimed
or needed for a bit-identity/non-identity verdict, and none of these numbers
may be quoted as physics. The fence is explicit in the committed JSONs
themselves: every arm's `compute_warnings` carries the common-mode
passivity self-check, quoted verbatim — "compute_waveguide_s_matrix:
extracted S-matrix failed a passivity/finiteness self-check —
passivity_violation: max column power 1.36422 exceeds limit 1.1 at
driven port 0, frequency index 0" (graded arms: 1.35884 on A/B/A_shim, 1.35905 on C) — the
truncation-artifact class the adjacent settling warning names, so any S
value read out of these runs carries its own do-not-quote marker. dz-graded waveguide S accuracy remains an OPEN
item (#810).

## Locked-value audit (bucket a)

Swept on this tree: no committed test gate, fixture, snapshot, or validation
result was produced by `compute_waveguide_s_matrix` on a dz-only simulation.
`grep -rl compute_waveguide_s_matrix tests/ | xargs grep -l dz_profile`
returns six files, each verified benign:

- `tests/unit/geometry/test_stage2_dual_path.py` — its cwsm gate runs on a uniform sim;
  its dz-only sim expects a raise from `run()`.
- `tests/unit/sparams/test_waveguide_nu_nontrivial.py` — grades dx; the manual
  `sim._dz_profile` there feeds a dispatch-bypassing internals helper.
- `tests/_example_fidelity_lib.py` — input-side fidelity snapshots; no
  variant combines dz_profile with a waveguide cwsm script.
- `tests/unit/sparams/test_sparameter_support_contract.py` — NU waveguide fixture is
  dy-only; its dz fixtures are MSL/TFSF (dz-aware dispatches).
- `tests/unit/preflight/test_preflight_absorber.py` — cwsm named in a docstring only.
- `tests/unit/materials/test_sheet_impedance.py` — dz-only fixtures are fence/raise
  tests; the cwsm fences use the uniform `_wr90` fixture.

So bucket (a) is EMPTY: the dispatch fix moves no committed pinned value.
The values that DO move are out-of-repo dz-only user results (uniform-mesh
artifacts, to be re-produced on the mesh actually requested) and the
unpushed #810 working tree's arm results (regenerated by whoever lands #810,
per that note's own instruction).

## R2 status

Attempt count for this mechanism (dz dispatch fix): 1. No prior attempt
exists in memory or the ledger; the falsifiers above are pre-declared with
numeric windows, so the attempt is closing by construction (it ends at a
declared PASS or a declared STOP).

## Results

### Import-provenance correction (before any accepted run)

The first baseline execution imported the editable-INSTALLED rfx (the
primary checkout) instead of this tree: `python script.py` puts the
script's own directory on `sys.path`, not the cwd, so the stale editable
install shadowed the checkout silently (the known
stale-editable-install trap). The instrument now records `rfx.__file__`
in its JSON, and both declared runs below were (re-)executed with
`PYTHONPATH` pinned to this tree; both JSONs record
`rfx_module_file = .../wf_7a4bcc28-1ea-4/rfx/__init__.py`.

### Pre-fix baseline (declared defect-signature run)

Working tree: commit `50e38371` with the two fixed files restored to
their pre-fix state (`git checkout 77158f8f -- rfx/api/_sparams.py
rfx/api/_preflight.py`), then re-restored from HEAD after the run.
Output: `scripts/diagnostics/wr90_dz_dispatch_falsifier_prefix_baseline.json`.

```
arm U       nz=10 dt=1.906575e-12 s  |S11| = [0.9285838 0.5331533 0.31759596 0.21109244 0.14664528 0.08948956 0.04622725 0.1506129 0.29077476]
arm A       nz=19 dt=1.149708e-12 s  |S11| = identical to U, bit for bit
arm B       nz=19 dt=1.149708e-12 s  |S11| = identical to U, bit for bit
arm A_shim  nz=19 dt=1.149708e-12 s  |S11| = [0.9240932 0.5317502 0.31734684 0.21096453 0.14590815 0.08813652 0.04777444 0.15393807 0.29540628]

F1 A vs B: bit_identical=True  max|dS|=0.000000e+00   per-bin |dS11| all 0.0
F2 A vs A_shim: max|dS|=9.922925e-03  (the dispatch-flip magnitude on these meshes)
F3 A vs U: max|dS11|=0.000000e+00  max|dS|=0.000000e+00
VERDICT: dz-only arms bit-identical -- dispatch defect signature (#811) present  (exit 2)
```

The dz arms' dt differs from the uniform arm's by 1.66x, and preflight
described the graded mesh ("REALIZED guide ... declared geometry
(non-uniform mesh)") while the solve returned bit-identical uniform
numbers — the issue's evidence, reproduced from this tree's own code.

### Post-fix battery (all falsifiers)

Working tree: commit `50e38371` with the fix (`1b5799bd`) in place — the
`rfx/` code is byte-identical through the later test/docs commits.
Output: `scripts/diagnostics/wr90_dz_dispatch_falsifier_results.json`.
Per-bin |S11(left,left)| across the nine 8.2–12.4 GHz bins:

```
arm U       nz=10 dt=1.906575e-12 s  [0.9285838  0.5331533  0.31759596 0.21109244 0.14664528 0.08948956 0.04622725 0.1506129  0.29077476]
arm A       nz=19 dt=1.149708e-12 s  [0.9240932  0.5317502  0.31734684 0.21096453 0.14590815 0.08813652 0.04777444 0.15393807 0.29540628]
arm B       nz=19 dt=1.149708e-12 s  [0.9240932  0.53175026 0.31734666 0.21096452 0.14590847 0.08813567 0.04777575 0.15393846 0.2954064 ]
arm C       nz=14 dt=1.539454e-12 s  [0.92546934 0.5321789  0.317404   0.2110142  0.14629401 0.0888041  0.04684966 0.15236984 0.29239026]
arm A_shim  nz=19 dt=1.149708e-12 s  [0.9240932  0.5317502  0.31734684 0.21096453 0.14590815 0.08813652 0.04777444 0.15393807 0.29540628]

F1 A vs B: bit_identical=False max|dS|=6.303530e-03
   per-bin |dS11|: [0.00306204 0.0042529  0.00384773 0.00082517 0.00238995 0.00630353 0.0018739  0.00053462 0.0040504 ]
F1 A vs C: bit_identical=False max|dS|=5.926809e-03
   per-bin |dS11|: [0.00183645 0.00206113 0.00145947 0.00061904 0.00042017 0.00141354 0.0009829  0.00162074 0.00309862]
F1 B vs C: bit_identical=False max|dS|=6.897247e-03
   per-bin |dS11|: [0.00449613 0.00227924 0.00239044 0.00021407 0.00225993 0.0051258  0.00178225 0.00182926 0.00561847]
F2 A vs A_shim: max|dS|=0.000000e+00 (tolerance 1e-6) -> PASS
F3 A vs U: max|dS11|=5.686986e-03 (window [1e-5, 1e-1]) max|dS|=9.922925e-03 -> PASS
   per-bin |dS11|: [0.00566571 0.00168449 0.00045048 0.000841   0.0015705  0.00208749 0.00161001 0.00357124 0.00568699]
VERDICT: all evaluated falsifiers PASS  (exit 0)
```

F1 PASS: every dz-only pair differs (bit-identity gone). F2 PASS at
exactly 0.0: the explicit uniform-valued dy equals the synthesized dy,
bit for bit (plumbing witness, as declared). F3 PASS: 5.687e-3 — same
order class as the issue's 1.1486e-3 (our meshes are coarser and
differently graded, hence a few-x larger).

Cross-JSON provenance witnesses (first one declared, second one an
additional measured observation):

```
postfix-U vs prefix-A:      bit_identical=True  max|dS|=0.0
postfix-A vs prefix-A_shim: bit_identical=True  max|dS|=0.0
```

The first proves the old dz-only "answer" WAS the uniform-grid answer
(the uniform lane is untouched by this diff). The second proves the
fixed dz-only path lands on exactly the solve the dy shim used to force.

Preflight context for these arms, quoted (identical family across arms;
full verbatim text in both JSONs' `preflight_issues`): dielectric 'slab'
resolution advisory "16.3 cells per λ_eff (eps_r=2.20, freq_max=12.4GHz,
dx=1mm). Need ≥20 cells/λ_eff"; "all dielectric(s) ['slab'] are
perfectly lossless in an open (CPML) domain" (harmless — no Q measured);
both ports' "max measurement frequency 12.400 GHz exceeds 0.90 ×
fc_next=11.803 GHz ... Evanescent TE20 contamination may exceed 1 %".
These bound the arms' absolute fidelity, are common-mode across every
comparison above, and do not affect a bit-identity / non-identity
verdict. As declared: none of these S values may be quoted as physics;
dz-graded accuracy evidence remains open under #810.

### F4 — suite results

- `pytest tests/unit/nonuniform/test_dz_only_dispatch_contract.py tests/unit/sparams/test_sparameter_support_contract.py tests/contracts/test_support_matrix_parity.py tests/contracts/test_evidence_citation_pointers.py -q`
  → **76 passed, 6 skipped, 1 deselected** (the deselected one is the
  slow_physics falsifier).
- `pytest tests/unit/nonuniform/test_dz_only_dispatch_contract.py -m slow_physics -q`
  → **1 passed, 11 deselected** (two genuinely different graded z meshes
  change the answer; per-bin dump printed by the test).
- `ruff check rfx/ tests/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402`
  → All checks passed.
- Targeted suite `pytest tests -k "waveguide and (sparam or s_matrix or dispatch or nonuniform)" -q`
  → **52 passed, 1 skipped, 4804 deselected in 119.25s** (exit 0; every
  warning in the log is pre-existing advisory chatter from other tests'
  own fixtures).

### Locked values moved

None. Bucket (a) was empty as declared (no committed pin was produced by
`compute_waveguide_s_matrix` on a dz-only simulation), and the F4 suites
confirm no committed gate moved. The values that move are out-of-repo
dz-only user results — uniform-grid artifacts of #811, to be re-produced
on the mesh actually requested — and the unpushed #810 working tree's
arm results (regenerated by whoever lands #810, per that note's own
instruction; its uniform-dy shim is no longer needed).
