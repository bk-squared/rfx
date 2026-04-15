# Issue #31 — Memory-Efficient Inverse Design: Results

Date: 2026-04-15
Branch: `nu-memory-efficient` (off main @ eab9a6b, commit 72efa1d)

---

## TL;DR

- **Phase A** (`jax.checkpoint(step_fn)` inside scan body) **does not solve the problem** for FDTD inverse design. XLA still stacks the full scan carry on the backward tape (`n_steps × carry_size`), which OOMs at 608 k cells × 10 000 steps (184 GB projected) on a 24 GB RTX 4090.
- **Phase B (segmented scan-of-scan)** does. Wrapping a segment-level outer scan in `jax.checkpoint` forces XLA to remat the inner scan during backward. Memory becomes O(`sqrt(n_steps) × carry_size`).
- **Phase C (`emit_time_series=False`)** removes the per-step probe tape; marginal on patch-sized jobs but meaningful when probe count × n_steps gets large.
- **Phase B drop originally proposed in the plan (bf16 fields) is not needed**. The A+C+segmented stack leaves ample headroom at 1.9 M cells × 10 000 steps (~11.5 GB peak on a 24 GB GPU).

Same-geometry frontier on RTX 4090 (from VESSL job 369367233490):

| checkpoint_every | 608 k cells × n_steps=10 000 |
| ---------------- | ---------------------------- |
| 0 (Phase A only) | **OOM** (184 GB projected)   |
| 50               | 4.82 GB                      |
| 100              | 2.45 GB                      |
| 200              | 1.26 GB                      |
| 500              | 0.59 GB                      |
| 1000             | 0.33 GB                      |

Ratio at `checkpoint_every=1000`: **~560× smaller** than the Phase A OOM cap.

---

## What was delivered

1. `checkpoint: bool` kwarg wired through `run_nonuniform` → `run_nonuniform_path` → `_forward_nonuniform_from_materials` → `forward()` (commit `c449cbb`). Kept for API symmetry; documented as ineffective on its own for FDTD carry.
2. `emit_time_series: bool` kwarg through the same four layers (commit `26e07fd`). Uniform path rejects `False` with `NotImplementedError`. Objectives in `rfx.optimize_objectives` that consume `result.time_series` raise a clear error if called on an empty series.
3. `checkpoint_every: int | None` — segmented scan-of-scan (commit `85217c6`). Outer scan wrapped in `jax.checkpoint` forces XLA to remat the inner scan. Non-divisible `n_steps` are padded with zero source and the trailing slice discarded.

Test pins (all pass on CPU locally, 35/35 on GPU via VESSL `369367233498`):

- `tests/test_nonuniform_checkpoint.py` — ckpt vs plain forward + grad bit-match.
- `tests/test_nonuniform_emit_ts.py` — emit flag empties time series; NTFF remains bit-identical; time-domain objectives raise a clear ValueError.
- `tests/test_nonuniform_segmented.py` — segmented forward and grad bit-match plain across three (n_steps, chunk) combinations.

---

## Evidence chain

### Memory
- **VESSL 369367233486** — first big-smoke. 608 k × 10 000 with `checkpoint=True` OOM’d at 184 GB. XLA log: `Can't reduce memory use below -137.14GiB by rematerialization`. This invalidated the plan’s §1 estimator (`ad_ckpt ≈ 4 × forward`).
- **VESSL 369367233488** — Phase A scaling sweep. Only 207 k cells × n_steps=2 000 (13.4 GB) fit; every other cell/step combo tried OOM’d.
- **VESSL 369367233490** — segmented scan validation. All six `checkpoint_every` values at 608 k cells × 10 000 steps passed; 1.22 M cells × 10 000 passed at 4.7 GB.

### Physics
- **VESSL 369367233498** — NU regression on the nu-memory-efficient branch. 35 passed, 2 xfailed. Patch TM010 harminv: 2.4621 GHz vs analytic 2.4235 GHz → 1.59 % error (matches the pre-branch pin).
- **VESSL 369367233509** — stable-FDTD physics smoke (2.4 GHz FR4 patch). Three scales, all finite and all monotonic across 4-6 grad steps:
  - 603 k cells × 2 000 steps, chunk=64: loss 6.61 × 10¹⁶ → 6.03 × 10¹⁶ (−8.7 %), peak 2.56 GB.
  - 603 k cells × 10 000 steps, chunk=100: loss 3.83 × 10¹⁷ → 3.64 × 10¹⁷, peak 5.84 GB.
  - 1.91 M cells × 5 000 steps, chunk=100: loss 1.90 × 10¹⁴ → 1.86 × 10¹⁴, peak **11.5 GB** on a 24 GB budget.

---

## Findings vs plan

| Plan claim | Reality |
| ---------- | ------- |
| Phase A alone delivers 188× (plan §1.1) | No — the estimator formula was wrong for FDTD because it ignored scan-carry tape. Phase A alone is useful only as API symmetry. |
| Phase B (bf16 fields) needed for big 3D (plan §1.3) | No — segmented scan already fits a 1.9 M-cell × 10 k-step case in 11.5 GB. bf16 would add numerical risk (CPML, ADE recursion) with negligible marginal headroom. Plan §3 kill-switch effectively triggered on ROI grounds. |
| checkpoint_every segmented remat is "out of scope" (plan §5.2) | Promoted to scope because Phase A alone failed the §7 gate. Landed as the primary memory win. |

## What was not done (deferred to a follow-up issue)

- `n_warmup + n_optimize` stop-grad (plan §5.2 item 2).
- `stop_gradient` on non-design cells (plan §5.2 item 4).
- Progressive multi-resolution (plan §5.2 item 5).
- Streaming NTFF multi-freq (plan §5.2 item 6).
- Distributed + NU + checkpoint (plan §5.1).
- `estimate_ad_memory` formula update. The hardcoded `ad_ckpt ≈ 4 × forward` is wrong for the NU path after this branch; parameterising it requires deciding what memory number to return for `checkpoint_every=None` vs a concrete value, and is better left to a dedicated refactor.

## Artifacts on branch

- Code: `rfx/nonuniform.py`, `rfx/runners/nonuniform.py`, `rfx/api.py`, `rfx/optimize_objectives.py`.
- Tests: `tests/test_nonuniform_checkpoint.py`, `tests/test_nonuniform_emit_ts.py`, `tests/test_nonuniform_segmented.py`.
- Docs: this note + `docs/agent-memory/nu_known_limits.md` (pre-existing NU sentinels).
- Scripts: `scripts/issue31_ad_memory_sweep.py` (local estimator sweep), `scripts/issue31_big_smoke.py` (NU patch-antenna smoke with memory readout).
- VESSL YAMLs: `scripts/vessl_issue31_big_smoke.yaml`, `scripts/vessl_issue31_segmented.yaml`, `scripts/vessl_issue31_crossval_regression.yaml`, `scripts/vessl_issue31_physics.yaml`, `scripts/vessl_issue31_smoke_sweep.yaml`.

## Next steps

1. PR `nu-memory-efficient` → `main`.
2. Close Issue #31 with a link to this note.
3. Open a follow-up issue for the deferred items above, tagging the `estimate_ad_memory` refactor as prereq.
