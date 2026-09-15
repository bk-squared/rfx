# cv07 — post-#931 recompute (lattice ownership contract)

Branch `feat/931-crossval-b`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XB-crossval-b`.

## Why the rfx leg is stale (and the openEMS leg is not)

The input feed, the wide patch and the output feed are now SHEETS (equal z
corners at `z = H_SUB`) instead of one-cell PEC Boxes, which the contract
realizes as a 200 µm filled metal slab with walls at BOTH 800 and 1000 µm.
Measured build-only (`realized_metal()`; the case prints this table into every
run):

| quantity | pre-#931 | post-#931 |
|---|---|---|
| PEC volume cells | 2814 | 0 (three sheets) |
| realized wall planes in z | k=4 and k=5 (800 + 1000 µm) | k=4 only (800 µm) |
| feed width (both feeds) | 2400.0 µm | 2200.0 µm |
| patch transverse | 20400.0 µm | 20200.0 µm |
| patch propagation | 2600.0 µm | 2400.0 µm |

The openEMS leg already draws zero-thickness metal on the declared board and
does not move; `_07_sheen_results/openems.json` stays as committed.

## The run

```
vessl run create -f scripts/vessl_931_xb/post-cv07.yaml        # from a non-git cwd
```

* **VESSL run id 369367259192**, name `rfx-931-post-cv07`, preset
  `gpu-rtx4090`, cluster `remilab-c0`, image
  `ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8`, `JAX_PLATFORMS=cpu` (the
  committed leg is a CPU leg; the GPU preset is for the queue, not the maths).
* Pre-change baseline to compare against: run **369367259003**
  (`rfx-931-base-cv07`). Its D4 witness read 0.0075 against the committed
  0.0145 at the same bin (17.378 GHz) — environment drift on a witness, so
  the post-change comparison must be made against THAT run, not against the
  committed number.
* Outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv07-<ts>/`.
* Steps: `07_sheen_lpf.py rfx` (**~260 s** solve, committed `runtime_s` 258.76)
  then `07_sheen_lpf.py compare`. Whole job ~10 min.

## Fixture keys that change

* `_07_sheen_results/rfx.json` — `freqs_hz`, `s11_mag`, `s21_mag`, `re_z0`,
  `energy_sum`, `settling_db`, `passivity_correction`, `preflight`,
  `fidelity_report`, `rfx_provenance`, `recipe_evidence`, `runtime_s`;
* the `COMMITTED` dict in `07_sheen_lpf.py` — every `rfx_*` entry
  (`rfx_null_ghz`, `rfx_passband_mean_s21`, `rfx_max_column_power`,
  `rfx_settling_db`, `rfx_worst_corr`, `rfx_worst_corr_ghz`,
  `rfx_corr_bins_*`) and `doublet_ghz["rfx"]`, `corner_ghz["rfx"]`. The
  openEMS entries do not move;
* `tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json` —
  `referee.fdtd_doublet_ghz.rfx`, `structure_distance_pct.rfx`,
  `referee_sides_with`; regenerate with
  `python scripts/diagnostics/build_sheen_lpf_palace_referee.py` AFTER the
  leg, or gate C4b (which cross-checks COMMITTED against this fixture) fails;
* `tests/fixtures/cv07_estimator_regate/cv07_estimator_falsifiers.json`;
* `validation/crossval/manifest.json` `cases[07].claim_scope` — replacement
  text in `docs/design_notes/931_migration/xb-manifest.md`.

## Pre-declared, before the run

1. Every rfx-leg number moves; the direction is not predicted, because the
   board changed in two ways at once (metal thickness 200 µm → 0, and every
   in-plane extent one cell smaller). What IS predicted: the openEMS-side
   numbers do not move at all, and `structure_distance_pct.rfx` is
   re-derived from the new declared-vs-realized table rather than translated.
2. **No window is widened.** C4's ±0.50 % and C6's ±0.25 % derive from the
   cell size ("one dx=200 µm cell on the 20.320 mm patch = 0.984 %"), and the
   cell size did not change. If a doublet moves more than its window, that is
   a finding to report, not a re-pin — and `referee_sides_with` flipping is a
   finding, not a re-pin either.
3. The build gate `assert_realized_metal` passes before the solve: three
   sheets, one plane, that plane = the realized laminate face, zero PEC
   volume cells, and the two 50-Ω feeds realizing the same 12 node rows.

## Two decisions recorded here, both refusals with measurements

* **The mesh stays dx = 200 µm.** `dx = H_SUB/4 = 198.5 µm` (the design
  note's on-lattice redraw) realizes the two nominally identical 50-Ω feeds
  12 and 13 node rows wide — 2183.5 vs 2382.0 µm, HJ Z0 54.22 vs 51.19 Ω —
  because their centres sit at different sub-cell offsets. The build gate
  refuses that mesh and `tests/crossval/test_cv07_realized_metal.py` pins the
  refusal. Finer aligned meshes (H_SUB/5, H_SUB/6) restore the symmetry but
  change what `n_probe_offset = 30 CELLS` means physically.
* **The declared-vs-realized port mismatch is measured, not closed.** Each
  port spans 13 node rows while its feed metal realizes 12. Passing the
  contract's realized extent (2200 µm) fixes the rows but sets the wave
  split's reference to HJ(2200, 800) = 54.22 Ω on a line this case measures
  at 50.30 Ω. Left open for §1.9 core work; printed by the gate and pinned by
  a test.
