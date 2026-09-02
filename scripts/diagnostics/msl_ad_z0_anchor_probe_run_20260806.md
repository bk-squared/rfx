# MSL AD gradient channel-attribution probe — run log (issue #560)

Tracked evidence for `scripts/diagnostics/msl_ad_z0_anchor_probe.py`, run locally
(CPU, float32, this worktree) against the exact fixture
`tests/unit/autodiff/test_msl_ad_fd_converged.py::_build_msl_sim` uses at its own
`_NUM_PERIODS=20`, `_N_FREQS=8`. Two process invocations produced this
result — the second re-confirms determinism for anchor B after the first
combined run was killed by a background-task duration limit before it
could print anchor B's second repeat or write its own JSON (everything
printed up to that point is intact and is reproduced below verbatim; the
kill was an infrastructure limit, not a script error — every step it did
reach printed a clean, expected result).

## Provenance (both runs)

```
rfx        : rfx/__init__.py (this worktree checkout)
git SHA    : 3e63a8ad1552f3c2a48cd2b6ed4ee24466ada02f
jax        : 0.6.2   devices: [CpuDevice(id=0)]
dtype      : float32 (AD, as shipped)   platform: CPU
fixture    : num_periods=20 n_freqs=8 (gate defaults: 20, 8)
grid       : (142, 54, 19)  n_steps=26226  checkpoint_segments=141
```

Identical to the committed gate fixture
(`scripts/diagnostics/msl_ad_band_mean_owner_measurement/owner_runs_20260804.md`),
so `g_a` below is directly comparable to the committed owner-platform number.

## Run 1 — combined (`--repeats 2`), killed mid-way through anchor B's 2nd repeat

Raw log: this run's full stdout is not separately committed (it is
reproduced verbatim below); the process was killed by the harness's
background-task duration limit while re-running anchor B's `run 1`
preflight, after both anchor-A repeats and anchor-B's `run 0` had already
printed cleanly.

```
--- preliminary forward (production, unpatched, alpha=1.0) ---
  wall-time: 370.6s
  assembly path: 'multi_drive_solve' (expected 'multi_drive_solve' -- see script header)
  forward |S| range: [0.0080, 1.0000]
  port 0: fitted z0 (band-mean, real) = 83.1861 ohm (band-mean imag = 2.0800 ohm)
  port 1: fitted z0 (band-mean, real) = 83.2578 ohm (band-mean imag = 1.8005 ohm)

--- anchor A: production (frozen analytic Hammerstad-Jensen z0_hj) ---
  run 0: loss = 0.99787515   g_ad = 1.602236e-03   (1033.3s)
  run 1: loss = 0.99787515   g_ad = 1.602236e-03   (1031.5s)

--- anchor B: frozen fitted z0 (measured at alpha=1.0, held constant) ---
  run 0: loss = 1.00077176   g_ad = 6.885110e-05   (995.4s)
  run 1: [KILLED before printing -- background-task duration limit]
```

Anchor A's own determinism is therefore already established from this run
alone: `run 0` and `run 1` are bit-identical (`1.602236e-03` both times).
`g_a = 1.602236e-03` (CPU, this session) matches the committed owner-
platform GPU number `1.602933e-03`
(`scripts/diagnostics/msl_ad_band_mean_owner_measurement/owner_runs_20260804.md`)
to within 0.04% relative — consistent with the CPU-vs-GPU float32 agreement
seen elsewhere in this repo (issue #477: `g_ad` CPU `-2.1104e-01` vs GPU
`-2.1105e-01`), and confirms this run reproduces the gate's own fixture
correctly. `assembly path = 'multi_drive_solve'` confirms the injection
point documented in the script's header (`compute_msl_s_matrix`'s
issue-#507 multi-drive `S = B @ A^-1` solve is the path that actually
reaches the objective on this 2-port fixture — the `a_fwd_d`/`b_ref_d`
block issue #560's text names is a vestigial intermediate that gets
unconditionally overwritten).

## Run 2 — anchor B only (`--only b --z0-fit-ohm 83.1861 83.2578`), re-confirming determinism

Skips the preliminary forward and anchor A (both already established in
Run 1); uses the SAME `z0_fit_per_port` values Run 1 measured internally
(supplied on the command line, hence rounded to 4 decimal places — see
"Anchor-value precision note" below). Ran to completion cleanly, both
repeats:

```
--- preliminary forward SKIPPED (--z0-fit-ohm given) ---
  using precomputed z0_fit_per_port = [83.1861, 83.2578]

--- anchor B: frozen fitted z0 (measured at alpha=1.0, held constant) ---
  run 0: loss = 1.00077271   g_ad = 6.884444e-05   (1105.5s)
  run 1: loss = 1.00077271   g_ad = 6.884444e-05   (1215.7s)

[determinism] anchor B exact match across 2 runs: True
Partial run ('b') complete. Wrote anchor_b_reconfirm.json
```

`run 0` and `run 1` of THIS invocation are bit-identical
(`6.884444e-05` both times) — anchor B's determinism (same code, same
input -> same output) is therefore independently confirmed, closing the
gap Run 1's kill left open.

### Anchor-value precision note

Run 2's `g_b = 6.884444e-05` differs from Run 1's `g_b = 6.885110e-05` by
`9.67e-05` relative (~0.01%). This is NOT evidence of non-determinism in
the FDTD+AD pipeline (which Run 1's anchor-A pair and Run 2's own repeat
both independently prove is bit-exact for a fixed input) — it is the
EXPECTED consequence of feeding Run 2 the `z0_fit_per_port` values through
a command-line float parse that only carried the 4 decimal places printed
by Run 1 (`83.1861`, `83.2578`), rather than Run 1's full double-precision
internal values. A ~1e-6 relative rounding on the anchor input producing a
~1e-4 relative shift in `g_b` (roughly 100x amplification) is itself
informative and consistent with mechanism 2: `g_b` is what remains after
removing the dominant frozen-reference-gap channel, so it is a smaller,
more sensitive residual than `g_a` — exactly the shape a near-cancelling
quantity takes (the same class of behavior issue #527 documented for the
retired `sum_ij|S_ij|**2` objective's near-cancelling residue, here
appearing in a different quantity for an unrelated, non-alarming reason).

## Combined result

```
g_a (production, frozen analytic z0_hj)          = 1.602236e-03  (Run 1, deterministic 2/2)
g_b (frozen fitted z0, full-precision anchor)     = 6.885110e-05  (Run 1, run 0 -- HEADLINE, un-repeated: Run 1's own run 1 was killed before printing)
g_b (frozen fitted z0, CLI-rounded anchor)        = 6.884444e-05  (Run 2, deterministic 2/2 -- the bit-exact-confirmed value)

ratio (headline:    g_a / g_b[Run 1, un-repeated]) = 23.271
ratio (cross-check: g_a / g_b[Run 2, 2/2 exact])   = 23.273
same sign: both positive
loss_a = 0.99787515 (<= 1, no passivity flag)
loss_b = 1.00077176 / 1.00077271 (both anchor-B runs, > 1 -- see "Passivity note" below)
```

### Primary criterion — issue #560's own qualitative wording, applied literally

Issue #560 does NOT state a numeric collapse threshold anywhere in its
body (checked: zero occurrences of "5x" or any ratio). Its actual
criterion, quoted verbatim: *"If `|g_ad|` collapses (drops toward the
FD-unresolvable floor, i.e. the reference-plane mismatch was supplying
most of the sensitivity) [...] If `|g_ad|` stays close to the current
`1.602933e-03`"*. Operationalizing "FD-unresolvable floor" with this
repo's own established standard (issue #527: `test_msl_ad_fd_converged.py`'s
`_fd_ulp_span`, and `test_comparator_floor_rejects_the_f32_reference_that_caused_527`,
which measured the RETIRED objective's f32 comparator at 4.449 ULP and
declared it untrustworthy):

```
first-order Taylor estimate of anchor B's FD signal at the gate's h=1e-3
(2h|g_b|, NOT an actual FD re-run -- verified via _fd_ulp_span reused
directly, see scripts/diagnostics/msl_ad_z0_anchor_probe.py's RESULT
section for the runtime code this reproduces):

  g_b headline (6.885110e-05):   1.1551 ULP of a float32 loss near 1.0008
  g_b reconfirmed (6.884444e-05): 1.1550 ULP
```

Both are well below the 4.449-ULP mark #527 declared unresolvable — by
this repo's own established standard, `g_b` has literally "dropped toward
the FD-unresolvable floor" in the issue's own words. **This alone settles
the issue on its own terms**, without needing any numeric ratio.

### Secondary criterion — this PR's own pre-declared threshold (NOT #560's)

`scripts/diagnostics/msl_ad_z0_anchor_probe.py`'s docstring pre-declares
a 5x/2x operational threshold BEFORE running the probe, as ITS OWN
choice — an earlier version of that docstring (and of this run log,
the PR body, and the GitHub issue comment) wrongly attributed "5x" to
issue #560 as a verbatim quote. It is not there; this has been corrected
throughout. Using the script's own pre-declared threshold: ratio ~23.3x
is 4.6x past the self-declared 5x collapse bar — consistent with, and
reinforcing, the primary criterion above.

### Passivity note (F3)

`loss_a` = 0.99787515 (a passivity-consistent value: band-mean `|S21|**2`
<= 1). Both anchor-B runs read `loss_b` > 1 (1.00077176 and 1.00077271) —
a passivity violation for a passive thru, which must be attributed, not
left silent. Attribution: `compute_msl_s_matrix` applies NO passivity
projection on the `eps_override` channel by design (so `jax.grad` and a
finite-difference reference see the identical raw function — see that
method's "EXEMPTION" docstring paragraph, which documents a measured
`sigma_max` of 1.18 on a coarse thru even in PRODUCTION with the frozen
analytic anchor). This does not threaten the ratio-based channel
attribution above (`g_a` and `g_b` both come from the same unprojected raw
`S` channel, so the comparison is apples to apples). It IS, however, the
strongest argument that the fitted anchor is not self-evidently "more
correct" than the frozen analytic one: swapping to it pushes THIS
extraction further over the passivity bound (implied `Γ_spur` from the
~74% z0 gap, 47.89 Ω analytic vs ~83.19 Ω fitted, is ≈0.27) — see the
"Production-anchor design question" note below.

### Production-anchor design question (F2) — NOT decided by this probe

This probe measures which channel dominates the AD *gradient* under a
hypothetical alternate anchor. It does **not** decide whether
`compute_msl_s_matrix` **should** anchor its *production* wave split on
the fitted `z0` instead of the frozen analytic `z0_hj`. That is a
separate, undecided design question. The passivity note above is exactly
why that question needs its own analysis rather than being inferred from
this result: the ~74% gap between the analytic anchor (47.89 Ω) and the
fitted one (~83.19 Ω) invites the reading "so switch production to the
fitted anchor," but the fitted anchor is what pushes `loss_b` over the
passivity bound here — not obviously the "more correct" choice on its
own. This PR does not open or resolve that design question; it is flagged
here so it is not silently inferred from the channel-attribution result.

**VERDICT: COLLAPSE** (issue #560's own qualitative criterion, satisfied
directly via the ULP argument above; also ~23.3x past this PR's own
pre-declared 5x threshold, 4.6x margin — not a borderline call either
way). The frozen-reference normalization gap (mechanism 2) is the
dominant channel behind `d(band-mean|S21|**2)/d(alpha)` on this fixture.
See `scripts/diagnostics/msl_ad_z0_anchor_probe.py`'s header for the full
decision-rule pre-declaration and for what this does and does not change
about `test_msl_ad_fd_converged_tight`'s validity as an AD-vs-FD
comparator (unaffected either way — see "WHAT THIS PROBE DOES NOT CHANGE").

## R3 self-audit

1. Contradicted by a known memory/feedback entry? None found after grep
   (`project_issue527_f32_comparator.md` and `feedback_gate_can_bind_artifact.md`
   are the nearest neighbors; both concern a DIFFERENT question — comparator
   resolving power / gate validity, not gradient mechanism attribution — and
   neither is contradicted by this result).
2. R2 trip? No — this is issue #560's own single pre-scoped decisive probe,
   not a repeated attempt at a previously falsified mechanism hypothesis.
3. Falsifier: the probe is its own falsifier per its pre-declaration — run
   both anchors twice and require bit-exact agreement before trusting the
   ratio. Anchor A: 2/2 exact (Run 1). Anchor B: 2/2 exact (Run 2, and
   cross-checked against Run 1's independently-computed value to 4
   significant figures under a deliberately perturbed anchor input).
