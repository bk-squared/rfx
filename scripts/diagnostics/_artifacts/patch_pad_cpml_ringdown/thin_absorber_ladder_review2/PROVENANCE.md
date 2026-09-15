# Thin-absorber ladder — MEASURED BY THE INDEPENDENT REVIEWER, imported here

These artifacts are **not** this lane's measurements. They were produced by the independent
review of `1a383283` / PR #1077 and copied here unchanged so the #801 narrowing has its
evidence in the repo rather than only on NFS.

* Source (read-only): `/root/workspace/claude-workspace/rfx/runs/801r-review2-cpml-ladder-20260915/`
* Imported: 2026-09-15 KST, by the #801 diagnosis lane, on branch
  `diag/801-patch-ringdown-padding`.
* Not re-run. The leader's instruction was explicit: do not re-measure what is already
  measured. The `.npz` probe series stay at the source path (≈1.2 MB); the `.json` files
  copied here carry the per-probe rates, the settling values and the decimated envelope
  traces, which is what the ladder table is built from.
* `801r_score_with_gate_metrics.py` scores those series with PR #1077's **own** metric
  functions, which is why the ladder and the gate are directly comparable.
* `801r_check_gate_fixture.py` is the reviewer's fixture check.
* `801r_pr1077_green_cpu.log` is the reviewer's CPU reproduction of the PR's green arm.

Every arm below is 150 periods, `+10h` lateral pad unless stated, on **current main** —
i.e. with the #931 edge rule in place, no mutation.

| arm | absorber | physical | settling dB | worst rate /step | verdict |
|---|---|---|---|---|---|
| n = 3, cpml 6 | 6 cells | 1.574 mm | **0.00** | **+8.53e-4** | GROWS |
| n = 2, cpml 4 | 4 cells | 1.574 mm | **0.00** | **+2.73e-3** | GROWS (269x) |
| n = 2, cpml 8 | 8 cells | 3.148 mm | -44.46 | -3.58e-4 | settles |
| n = 2, cpml 16 | 16 cells | 6.296 mm | -47.00 | -3.88e-4 | settles |
| n = 2, cpml 4, **pad 0** | 4 cells | 1.574 mm | -43.71 | -3.40e-4 | settles |
| n = 4, cpml 8 | 8 cells | 1.574 mm | -43.37 | -1.93e-4 | settles (the PR #1077 gate point) |

Readings this fixes:

* **Not under-resolution.** n = 2 with 8 or 16 layers settles at the same dx that grows with
  4. The resolution is held; only the layer count moves.
* **Not physical thickness either.** n = 3 / 6 layers and n = 4 / 8 layers are both 1.574 mm
  and disagree, while n = 2 / 4 layers (1.574 mm) grows and n = 2 / 8 (3.148 mm) settles.
  The variable that tracks the verdict is the **layer COUNT**, with the lateral padding as a
  co-factor: the same 4-layer absorber settles with `pad = 0`.
* **Not #1070.** The vacuum absorber pad is present in growing and settling arms alike.
* The block-max envelope on a growing arm decays to about **-67.4 dB** and then climbs
  monotonically — an eigenvalue of the update operator, not a transient that has not finished.
* `n = 3 / cpml 6` is one of **#801's own recorded arms**, so this is the original report
  still reproducing on current main, not a new configuration.
* Preflight says nothing about `cpml_layers` on any of these.
