# cv06b — post-#931 recompute (lattice ownership contract)

Branch `feat/931-crossval-b`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XB-crossval-b`.

## Why every artifact in this directory is stale

The main line and the open stub are now declared as SHEETS (a zero-thickness
Box on the substrate-top node plane) instead of one-cell PEC Boxes, which the
contract would realize as a 63.5 µm filled metal slab with walls at BOTH 254
and 317.5 µm. What that changes on the board, counted build-only (see
`realized_metal()` in the case script):

| quantity | pre-#931 | post-#931 |
|---|---|---|
| PEC volume cells | trace+stub as a node mask | 0 (two sheets) |
| realized wall planes in z | one plane, k=4 | one plane, k=4 (unchanged) |
| geometric realized width (both lines) | 635.0 µm | 571.5 µm |
| electrical width, n_rows·dx (the HJ input) | 635.0 µm | 635.0 µm |
| stub open end | 13652.5 µm | 13589.0 µm |
| quarter-wave length from the line centre | 12350.75 µm | 12287.25 µm (−0.514 %) |

So the solve changes: the stub is one cell shorter. Nothing here can be
translated; it must be re-solved.

## The run

```
vessl run create -f scripts/vessl_931_xb/post-cv06b.yaml      # from a non-git cwd
```

* **VESSL run id 369367259191**, name `rfx-931-post-cv06b`, preset
  `gpu-rtx4090`, cluster `remilab-c0`, image `nvcr.io/nvidia/jax:24.10-py3`.
* Pre-change baseline to compare against: run **369367259002**
  (`rfx-931-base-cv06b`, origin/main d990e18c), outputs under
  `/root/workspace/claude-workspace/rfx/runs/issue931-baseline-cv06b-<ts>/`.
* Outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv06b-<ts>/`
  (the pointer file `issue931-post-cv06b.latest` in the runs dir names it).
* Steps and expected runtime, from the baseline: estimator replay (CPU-bound
  replay of a saved sweep, ~2 min), the case script (**~330 s** solve on one
  RTX 4090; a CPU run of this mesh was abandoned at 2 h 52 m), the three build
  falsifiers (**3 × ~330 s**). Whole job ~25–35 min.

`vessl run list` is slow and is not used; read the run by id or read the
`.latest` pointer.

## Fixture keys that change

* `_06b_msl_notch_results/cv06b_build_falsifiers_summary.json` — every value
  under `criterion_A_baseline` (`err_pct`, `bw_ratio`, `witness_bins`,
  `notch_depth_db`, `f_notch_refined_hz`, `f_notch_bin_hz`,
  `f_notch_analytic_hz`, `sub_bin_shift_bins`, `z0_median_ohm`, `solve_s`) and
  under the criterion-B arms (`stub_1cell`, `stub_narrow`);
* `_06b_msl_notch_results/cv06b_falsifier_{baseline,stub_1cell,stub_narrow}.json`
  — full sweeps;
* `_06b_msl_notch_results/cv06b_baseline_run.log` / `.exit`;
* a new `_06b_notch_uniform_logs/<ts>_run.log` (the committed
  `20260827T131217Z_run.log` is the pre-#931 record and stays);
* `tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json`;
* `validation/crossval/manifest.json` `cases[06b].claim_scope` — NOT owned by
  this branch's crossval agent; the replacement text is in
  `docs/design_notes/931_migration/xb-manifest.md`.

## Pre-declared, before the run (design note §5)

Reported verbatim against this list, pass or fail. Full derivation in the
case script's "#931 LATTICE OWNERSHIP" docstring section.

1. **G4 Z0 median stays 46.48 ± 1.0 Ω.** This is the falsifier for the width
   convention: ~46.5 Ω says the electrical width of an n-row strip is n·dx and
   the analytic reference was right to stay at 3.678954 GHz; ~49.4 Ω says the
   geometric span is the electrical width, and then `_realized_trace_width`,
   EPS_EFF, F_NOTCH_AN and G1's shunt-T term all re-derive on 571.5 µm.
2. Measured notch rises ≈0.51 % (the stub's one cell): 3.6255 → 3.644 GHz
   (refined vertex), ±0.3 pp.
3. `err_pct` falls 1.4530 → ≈0.95 %. A RISE above 1.45 % falsifies (2).
4. G2 `bw_ratio` stays 0.9684 ± 0.05 — r = 1 is preserved by construction
   (both lines realize 10 rows), and `assert_realized_metal` now checks it.
5. The half-grid witness (0.3175 bin) and the notch depth (−43.3 dB) are not
   predicted to move.
6. No gate window moves. `NOTCH_FREQ_TOL_PCT` stays 4.0.

## For the ingest phase

The build falsifier summary's `all_ok` is `false` in the committed state
(`stub_1cell` is not sub-bin-visible: 0.145 % measured against 0.532 %
predicted). That is the pre-existing recorded state, not something this
change introduces — do not read a `false` as a #931 regression without
comparing the two `stub_1cell` blocks.

## MEASURED — run 369367259191, reported verbatim against the list above

Artifacts in
`/root/workspace/claude-workspace/rfx/runs/issue931-post-cv06b-20260907T124851Z/`;
all three steps exited 0. Everything in this directory, plus
`tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json` and
`_06b_notch_uniform_logs/20260907T124851Z_sheets_931_vessl369367259191_run.log`,
is that run's own output, copied, not edited.

| pre-declaration | predicted | measured | verdict |
|---|---|---|---|
| 1. G4 Z0 median | 46.48 ± 1.0 Ω | **48.192 Ω** | **FALSIFIED** |
| 2. notch rises one stub cell | 3.6255 → 3.644 GHz (+0.51 %, ±0.3 pp) | 3.7586 GHz (**+3.67 %**) | **FALSIFIED** |
| 3. err_pct falls | 1.4530 → ≈0.95 % | **2.1649 %** | **FALSIFIED** (a rise, as (2) predicted would falsify it) |
| 4. G2 bw_ratio | 0.9684 ± 0.05 | 0.99913 | held |
| 5. half-grid witness, notch depth | do not move (0.3175 bin, −43.3 dB) | 0.4469 bin, −39.44 dB | both moved |
| 6. no gate window moves | — | none moved | held |

All four gates still PASS on their own windows: G1 2.16 % (< 4.0 %),
G2 0.9991 ∈ (0.80, 1.20), G3 0.4469 bin (< 1.0), G4 48.2 Ω ∈ (40, 65).
`NOTCH_FREQ_TOL_PCT` is still 4.0 and nothing was widened.

### What (1) actually settles, and what it does not

G4 was the pre-declared falsifier for the width convention, with two
candidate answers: ~46.5 Ω would say the electrical width of an n-row strip
is `n·dx` = 635.0 µm, ~49.4 Ω would say it is the geometric node span
`(n−1)·dx` = 571.5 µm. The board measures **48.19 Ω, between the two** —
4.35 % above HJ(635.0) = 46.18 Ω and 2.43 % below HJ(571.5) = 49.39 Ω.

So the falsifier fired against BOTH readings. The pre-#931 argument for
`n·dx` — that this case's own Re(Z0) 46.48 Ω matched HJ(635) to 0.65 % —
was made on a board whose metal really was a 635 µm-wide one-cell Box. On
the zero-thickness sheet the same 10 node rows carry a line that behaves
like a strip somewhere between the node span and `n·dx`, and closer to the
node span. `_realized_trace_width` still returns `n·dx`, so `F_NOTCH_AN`
= 3.678954 GHz is built on a width this run says is ~4 % too wide, which is
most of why `err_pct` rose to 2.16 % instead of falling to 0.95 %.

**Nothing is re-derived from this.** One measurement of one board at one
dx cannot fix an effective-width convention, and fitting the convention to
it would be tuning the model to the datum it is supposed to be checked
against. What the next agent needs is a second point: the same board at a
different dx (the committed dx = 80 µm log is a pre-#931 Box board and
cannot be spliced in), or an independent Z0 witness. Until then the case
ships `n·dx`, the two named quantities stay named, and this paragraph is
the reason.

### Falsifier lane: criterion B now closes

`stub_1cell` went `visible: false → true` — a true shift of 0.5320 % is now
seen as `refined_delta_pct` 0.8228 (it was 0.1447), and `bin_argmin_delta_pct`
0.0 → 1.6949. `verdict.all_ok` therefore goes `false → true`. The
"For the ingest phase" caveat above — do not read the committed `false` as a
#931 regression — is spent: the arm passes now.

`stub_narrow` is the deliberately-broken arm. It still fires G2
(`G2_fired: true`) and now fires G1 as well (err_pct 0.208 → 6.439), i.e.
the falsifier arm got MORE sensitive, and `depth_witness_still_passes`
stays true — the depth gate is still blind to it, which is the point of
the arm.

### Two things this run hands on

* **The estimator fixture moved only at ~1 ulp** (six values, last decimal
  place). The estimator lane replays a saved sweep and is board-independent
  by construction; the RECOMPUTE listed it as a key that changes, and
  measuring that it does not is the useful answer. The run's own file is
  committed anyway, so the fixture provably post-dates the re-solve.
* **The collector-less `_assemble_materials` PreflightWarning fires on this
  case's production solve path** (`PEC sheets/wires were classified but the
  caller passed no pec_sheets/pec_wires collector`). That is the open core
  item — `_warn_uncollected_pec` not yet a raise, nine callers — not a
  cv06b defect. Recorded here because this log is where it is visible.

### STALE AFTER THIS RUN — for the Docs group, not done here

The shipped board's headline numbers are now **2.16 % / −39.4 dB / 48.2 Ω**,
not 1.40 / 43.3 / 46.5. Still pointing at the pre-#931 record:

* `tests/crossval/test_msl_notch_public_carriers.py` — `RUN_LOG` is still
  `20260827T131217Z_run.log`, and `test_carrier_quotes_the_current_headline`
  pins the literals `"1.40"`, `"43.3"`, `"46.5"`;
* the five carriers that test checks: `validation/README.md`,
  `docs/guides/sparameter_support_matrix.{md,json}`,
  `docs/agent/port-selection.mdx`, `docs/public/guide/benchmarks.mdx`.

Repointing `RUN_LOG` at
`20260907T124851Z_sheets_931_vessl369367259191_run.log` and moving those
literals is one coupled edit across a group boundary — the carrier docs are
the Docs group's. The old log stays committed either way: it is the
pre-#931 provenance record and the `dx=80 µm` comparison in
`test_realized_board_is_measured_not_assumed` reads two pre-#931 logs and
must keep doing so.
