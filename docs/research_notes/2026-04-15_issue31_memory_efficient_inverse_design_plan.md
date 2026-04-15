# Issue #31 — Memory-Efficient Inverse Design on NU Mesh (Plan)

Date: 2026-04-15
Branch: `nu-memory-efficient` (off main @ eab9a6b, post-PR #34)
Scope: planning doc only. No code in this commit.

---

## 0. TL;DR

Baseline (patch, 2.4 GHz FR4, 553k cells, n_steps=2000) from
`docs/research_notes/2026-04-15_nu_feature_completion_and_crossval.md`:

| Path                   | fwd      | ckpt-AD  | full-AD   |
| ---------------------- | -------- | -------- | --------- |
| NU plain (fp32)        | 40.5 MB  |  162 MB  |  30.5 GB  |
| Uniform fine (fp32)    | 1324 MB  | 5297 MB  |   996 GB  |

Projected ceilings after this plan, same baseline:

| Path                                | fwd     | AD (reverse)          | vs NU plain full-AD |
| ----------------------------------- | ------- | --------------------- | ------------------- |
| NU + A (remat only, fp32)           | 40.5 MB | ~162 MB               | 188x smaller        |
| NU + A + B (remat + bf16 fields)    | ~27 MB  | ~108 MB               | 282x smaller        |
| NU + A + B + C (add windowing)      | ~27 MB  | ~100 MB               | 305x smaller        |

Headline number: **~0.1 GB** reverse-mode AD on the patch baseline after
full stack, vs 30.5 GB today. The 30x savings vs NU-plain-ckpt-AD come
mostly from A (already projected by `estimate_ad_memory` at
`rfx/api.py:2622` — `ad_ckpt_bytes = 4 * forward_bytes`). B and C are
incremental refinements that matter for larger 3D jobs where field and
time-series buffers dominate.

**Recommended order: A → C → B.** A is a 3-line plumbing change that
unlocks the 188x improvement; C is UX-visible but numerically trivial;
B is the risky one (precision) and should land last with the most
regression coverage.

---

## 1. Goal numbers (projected before coding)

Formula (from `estimate_ad_memory`, `rfx/api.py:2585-2633`):
```
forward_bytes = cells * (6 field + 6 mat) * bytes_per_cell + ~15% CPML
ad_ckpt       = 4 * forward_bytes + ntff_bytes
ad_full       = n_steps * (cells * 6 * bytes_per_cell) + ntff + forward
```

Patch baseline: cells = 553k, n_steps = 2000, no NTFF in baseline run.

### 1.1 Phase A alone (checkpoint, still fp32)

`estimate_ad_memory` already computes the checkpointed number assuming
remat; the NU path just isn't wired to it. Plumbing `checkpoint=True`
into the scan converts measured AD memory from 30.5 GB → 162 MB on the
patch. Ratio **NU+A vs NU plain full-AD = 188x**.

### 1.2 Phase A + B (checkpoint + bf16 fields)

Drop field bytes 4 → 2. Material arrays stay fp32 (they are swept by
the optimizer and must retain range). CPML psi stays fp32 (narrow
dynamic range is unsafe; see §3). DFT/NTFF accumulators stay
complex64. So `forward_bytes` drops from `cells*12*4 = 26.5 MB` to
`cells*(6*2 + 6*4) = cells*36 = 19.9 MB`. ckpt-AD ≈ 4x that = **~80
MB**. Adding psi/debye overhead back at fp32 the realistic estimate is
**~100-110 MB** for ckpt-AD.

### 1.3 Phase A + B + C (add windowing)

Time-series today is `(n_steps, n_probes, 3)` inside scan outputs.
Even with a single probe, n_steps=2000 × 3 × 4B = 24 KB — tiny on the
patch baseline. Windowing wins are geometry-dependent; they become
meaningful when probes are plane DFTs or when n_steps reaches
20k-100k. **The C payoff on the patch is ~1-2 MB**, but on a WR-90
inverse design with n_steps = 30000 and 3 time-domain probes it is
hundreds of MB.

### 1.4 Summary

On the patch baseline, A does essentially all the heavy lifting.
B and C matter for larger 3D inverse-design jobs where field buffers
and time series dominate — so this plan's success criterion should
include at least one "big geometry" re-measurement (§7) rather than
declaring victory on the patch alone.

---

## 2. Phase A — checkpointed NU scan

### 2.1 Current state (verified, not assumed)

- `rfx/api.py:3270-3307` `_forward_nonuniform_from_materials` accepts
  `checkpoint: bool = True` but **throws it away** (`del checkpoint`,
  line 3289). The comment at 3283-3286 explicitly flags this as TODO.
- `rfx/runners/nonuniform.py:188-620` `run_nonuniform_path` does **not**
  take a `checkpoint` kwarg. It calls `run_nonuniform(...)` at line 519.
- `rfx/nonuniform.py:474-495` `run_nonuniform` also has no `checkpoint`
  kwarg. The scan is at line 834; `step_fn` at line 653.
- Uniform path precedent: `rfx/simulation.py:927`
  `body = jax.checkpoint(step_fn) if checkpoint else step_fn` — this
  is the exact pattern to replicate.

### 2.2 Change surface

Three-level kwarg threading:
1. `rfx/nonuniform.py:474` — add `checkpoint: bool = False` to
   `run_nonuniform`; at line 834 replace
   `jax.lax.scan(step_fn, ...)` with
   `jax.lax.scan(jax.checkpoint(step_fn) if checkpoint else step_fn, ...)`.
2. `rfx/runners/nonuniform.py:188` — add `checkpoint: bool = False` to
   `run_nonuniform_path`; forward it on line 519 `run_nonuniform(...)`.
3. `rfx/api.py:3270` — drop the `del checkpoint` and forward the kwarg
   into `run_nonuniform_path`. Default stays `True` (matches uniform).

Estimated LOC: ~8 non-blank. No invariant risk.

### 2.3 Scope warning

The NU `step_fn` (line 653) closes over `materials`, `debye`, `lorentz`,
`rlc_metas`, `pec_mask`, `dft_planes`, etc. Under `jax.checkpoint`,
these are recomputed on backward. Verify that none of them are
Python-level closures that would break under rematerialization — a
quick eyeball of `step_fn` body during execution is required before
the test pin lands.

### 2.4 Test plan

- Keep green: `tests/test_nonuniform_forward_grad.py`,
  `tests/test_nonuniform_gradient.py` (AD-vs-FD pin).
- Add: `test_nu_checkpoint_memory_pin` — call
  `sim.estimate_ad_memory(n_steps=2000)`, assert
  `ad_checkpointed_gb < 0.5` and `ad_full_gb > 10` for the patch case,
  so a regression that silently re-enables full tape is caught.
- Add: `test_nu_forward_ckpt_matches_plain` — compare
  `sim.forward(checkpoint=True)` and `sim.forward(checkpoint=False)`
  returns bit-identical time-series within 1e-6.

---

## 3. Phase B — mixed precision

### 3.1 fp16 vs bf16

**Pick bf16.** Rationale:
- fp16: 5-bit exponent, range ~±6e4. FDTD ringdown fields can span 8+
  orders of magnitude during source decay and long transients; fp16
  underflow is the dominant failure mode for long-time FDTD.
- bf16: 8-bit exponent (same as fp32), 7-bit mantissa. Mantissa
  precision loss is the tradeoff, but for FDTD update coefficients of
  form `(eps - σdt/2)/(eps + σdt/2)` the mantissa pressure is mild.
- Nvidia Ampere+ (A100, RTX 30/40, H100) have hardware bf16 tensor
  cores; no throughput penalty.
- Precedent: `rfx/runners/distributed_v2.py` already handles dtype
  casting for pmap; `rfx/nonuniform.py:552` `field_dtype` is the
  existing hook (currently fp32).

### 3.2 Cast points (where bf16, where fp32)

Keep **bf16**:
- E, H field arrays (scan carry `field`).
- Update coefficients derived from material at cast time.

Keep **fp32**:
- `materials.eps_r`, `materials.sigma`, `materials.mu_r` (optimization
  variables; need full range for gradient updates).
- CPML psi state (narrow dynamic range + recursive accumulation =
  catastrophic for low precision).
- Debye/Lorentz auxiliary state (ADE recursion, same reason).
- RLC auxiliary state.

Keep **complex64** (not complex32 / bf16):
- All DFT accumulators (`jnp.zeros(..., dtype=jnp.complex64)` at
  `rfx/nonuniform.py:625-627`, plane DFT, NTFF DFT).
- Rationale: DFTs are the *output* of the forward pass and feed the
  loss; any precision loss here destroys gradients.

Keep **fp32**:
- `src_waveforms` (line 642), time-series probe outputs, the loss
  scalar. Cast field → fp32 at probe tap points before accumulating.

### 3.3 Risk items (numerical)

1. **CPML stability at bf16 fields.** Even with fp32 psi, field values
   feeding psi updates are bf16. Under large σ the multiplier can
   denormalize.  Mitigation: sweep a 1-antenna case at bf16 vs fp32
   and check return loss deviation.
2. **Debye/Lorentz polarization recursion** (`init_debye`,
   `init_lorentz`, line 650-ish of `nonuniform.py`). Polarization
   current `J` accumulates; bf16 would drift. Keep the aux state and
   the J-update itself in fp32.
3. **Soft-source casting.** Already handled at `nonuniform.py:786` /
   `1289` — source value cast to `field.dtype`. Verify this stays
   correct under bf16.
4. **Lumped-port sigma update.** If mixed-precision sigma is not
   explicitly routed, the port RLC time constant may shift. Verify
   `setup_rlc_materials` stays fp32.
5. **Gradient accumulation.** `jax.grad` output is fp32 by default
   when the loss is fp32; verify no implicit downcast.

### 3.4 Test plan

- Dipole far-field regression: `examples/crossval/01_*` equivalent in
  tests, 1% relative error vs fp32 baseline at 3 frequencies.
- Patch resonance f0: within 0.1% of fp32 baseline (stricter than
  far-field because it's a linear frequency pin).
- CPML reflection: re-run CPML sweep on one geometry (the CPML-heavy
  case from known-issues); max reflectivity stays within 3 dB of fp32.
- All 730+ existing tests still pass at the *default* precision (fp32)
  — bf16 must be opt-in (`Simulation(precision="mixed")`).

### 3.5 Scope warning

If §3.3 item 1 (CPML stability) fails the 3 dB pin, bf16 is not
viable without also upcasting field→fp32 at CPML cell faces, which
undoes the savings in the PML region. This is a **kill switch** for
Phase B; if triggered, deliver A+C and defer B to a dedicated
research note with sweep evidence.

---

## 4. Phase C — temporal windowing

### 4.1 What `forward()` emits today (verified)

`ForwardResult` (see `rfx/api.py` near 3257): `time_series, ntff_data,
ntff_box, grid, s_params, freqs`. NTFF and s_params are already DFT
accumulators — frequency-domain. Only `time_series` is per-step.

In the NU path `rfx/runners/nonuniform.py:609`:
`time_series=r["time_series"]`. This is a dense `(n_steps, n_probes,
3)` array scanned out of `jax.lax.scan` at `nonuniform.py:834`.

### 4.2 Change surface

Add a `time_series: bool = True` kwarg at three levels (mirroring §2):
1. `rfx/nonuniform.py` `run_nonuniform`: when False, scan returns
   `final` only (discard the second `jax.lax.scan` output by emitting
   an empty `xs`), and `r["time_series"]` becomes an empty array with
   shape `(0, n_probes, 3)`.
2. `rfx/runners/nonuniform.py` `run_nonuniform_path`: thread through.
3. `rfx/api.py` `forward()` and `_forward_nonuniform_from_materials`:
   expose as `emit_time_series` (less ambiguous than `time_series` as
   a bool when the field has the same name).

Memory win on patch: tiny (KB). Memory win on WR-90 n_steps=30000 with
4 probes: 30k × 4 × 3 × 4B ≈ 1.4 MB — still small compared to fields,
but the AD-tape footprint of the per-step write (held in reverse-mode
without checkpoint) is much larger than the forward storage.

### 4.3 Edge cases

- `minimize_reflected_energy` in `rfx/optimize_objectives.py:374` is a
  **time-domain** proxy — it needs the time series. The objective
  wrappers (line 99 and 250 reference it as the fallback when S-params
  unavailable) must raise a clear error if `emit_time_series=False`.
- Waveguide-port DFTs (`nonuniform.py:625-627`) already skip the time
  series; they accumulate in complex64 during scan. Confirm no test
  reads `time_series` to derive waveguide S11/S21 when a waveguide
  port is defined (spot check `tests/test_waveguide_*`).
- NTFF similarly uses scan-internal DFT; independent of time series.

### 4.4 Test plan

- `test_nu_forward_notimeseries` — call forward with
  `emit_time_series=False`, assert `result.time_series.shape[0] == 0`
  and that NTFF and waveguide S-params are bit-identical to the
  `True` case.
- `test_minimize_reflected_energy_requires_timeseries` — expect
  explicit ValueError with message referencing the option name.

---

## 5. Order of operations and out-of-scope

### 5.1 Order: A → C → B

- **A first**: 3-line change, ~30x headline win, zero numerical risk.
  Unblocks larger inverse-design jobs today on consumer GPUs.
- **C second**: API addition only, no kernel changes, easy to test.
  Lands the UX improvement while B is being validated.
- **B last**: The only phase that can fail a numerical gate. Land it
  behind an opt-in `precision="mixed"` flag so fp32 remains the
  default and the regression surface is small.

### 5.2 Out of scope (defer)

- Distributed + NU + checkpoint (needs separate branch; issue-31 says
  distributed is its own problem and `distributed_v2.py` has pmap
  carry constraints that interact with remat).
- TPU support (bf16 story is different; no current user).
- INT8 / NF4 quantization (materials-side; different plan).
- `checkpoint_every=N` segmented remat (issue-31 §1). Start with
  whole-scan remat; segmented is a follow-up once A lands.
- `n_warmup` + `n_optimize` stop-grad (issue-31 §2). Nice to have but
  orthogonal to A/B/C and has its own correctness story.
- `stop_gradient` on non-design cells (issue-31 §4). Requires a
  design-region mask API; separate plan.
- Progressive multi-resolution (issue-31 §5). Orchestrator-level, not
  runner-level.
- Streaming NTFF multi-freq (issue-31 §6). Touches NTFF carry shape;
  separate plan.

---

## 6. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
| - | ---- | ---------- | ------ | ---------- |
| 1 | `jax.checkpoint` around NU `step_fn` re-traces unhashable closures (dispersion masks, rlc_metas) | Medium | Medium — breaks A entirely | Pre-survey: eyeball `step_fn` closure contents before threading; if problematic, refactor closure captures into explicit scan carry before landing A |
| 2 | bf16 field breaks CPML at high σ, measured as >3 dB reflectivity regression | Medium | High — kills B | Sweep before code: run one CPML geometry at bf16 in a scratch notebook with existing flags and confirm pin before merging B |
| 3 | bf16 field silently degrades Debye/Lorentz ringing, visible only at long n_steps | Low | High — correctness bug | Keep ADE aux state fp32; add long-time Debye regression test to the B gate |
| 4 | `emit_time_series=False` silently breaks `minimize_reflected_energy` during an optimize() loop | Medium | Medium — bad UX | Raise at objective construction time if probe is time-domain and flag is False |
| 5 | `estimate_ad_memory` becomes stale once remat+bf16 land (it hardcodes bytes_per_cell=4 at `api.py:2591`) | High | Low — estimator drift | Parameterize `bytes_per_cell` from sim precision before Phase B lands; update test_nu_checkpoint_memory_pin accordingly |
| 6 | AD-vs-FD pin (`test_nonuniform_forward_grad`) tolerance becomes too tight under bf16 | Medium | Medium — false regressions | Pin uses fp32 default; B-specific tests use relaxed tolerance documented in the test docstring |
| 7 | Patch baseline doesn't stress C; shipping C without a large-geometry measurement overstates the win | High | Low — doc hygiene | Add one n_steps=30k probe-heavy case to the §7 verification list |

---

## 7. Verification gates

Must pass before merge to main.

- **Unit + AD pins** — all existing tests green, plus the three new pins
  from §2.4, §3.4, §4.4.
- **Memory pin via estimator** — `sim.estimate_ad_memory(2000)` on the
  patch baseline returns `ad_checkpointed_gb < 0.5` after A, and
  `forward_gb` drops by ~1/3 after B (bytes_per_cell=2 path).
- **OpenEMS crossval regression** — re-run VESSL 369367233458 equivalent
  (rfx vs OpenEMS GPU patch) on the `nu-memory-efficient` branch after
  each phase. Must reproduce the 0.99% agreement pin from
  `docs/research_notes/2026-04-15_nu_feature_completion_and_crossval.md`.
  A-only, A+C, and A+C+B(bf16 default-off) each get their own run.
- **One "big" inverse design smoke** — WR-90 or horn at n_steps >= 10k,
  confirm optimize() runs end-to-end on a single 24 GB GPU without OOM
  (currently OOMs at ~n_steps=500 per issue-31 §Impact). No accuracy
  pin — just OOM-free and loss decreases for 20 iterations.
- **Preflight** — `sim.preflight()` 12 checks still pass on all of
  `examples/crossval/01..07`.
- **Lint** — `ruff check` clean.

---

## 8. Deliverables per phase

- **A**: code diff ≤ 20 LOC across 3 files, 2 new tests, updated NU
  forward docstring, short entry appended to
  `docs/agent-memory/index.md`.
- **C**: code diff ≤ 40 LOC, 2 new tests, new example snippet in
  `examples/` showing `emit_time_series=False` with NTFF.
- **B**: `Simulation(precision="mixed")` flag, internal dtype routing,
  full test suite at default fp32 still green, new bf16 regression
  test, research note documenting the CPML sweep result.

---

## 9. Open questions for executor session

- [ ] Does NU `step_fn` (`rfx/nonuniform.py:653`) contain any
      side-effecting Python closures that would break under
      `jax.checkpoint`? (Must verify before Phase A.)
- [ ] Does `distributed_v2.py` already have a bf16 cast convention we
      should reuse verbatim, or is its path orthogonal enough that B
      should define its own?
- [ ] Should `precision="mixed"` be a single flag, or a richer
      `PrecisionPolicy(fields="bf16", psi="fp32", dft="complex64")`
      struct?  (Prefer single flag for now; policy struct is a
      refactor after B stabilizes.)

(These also belong in `.omc/plans/open-questions.md` per the planner
protocol; added separately.)
