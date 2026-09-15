# cv16 recompute — issue #931 (group X-C): CONTROL ONLY, no fixture regeneration

Branch `feat/931-crossval-C`.

## Verdict: cv16 is fenced, and the fixture must NOT be regenerated

cv16's sphere is a high-sigma MATERIAL FILL — `rasterize(Sphere, eps_r 1.0,
sigma 1e7)` straight onto `MaterialArrays`, never `Simulation.add(...,
material='pec')`. Design note §1.8 fences that model out of the ownership
contract: a sigma fill is a lossy volume model, keeps NODE sampling, and does
not move. So no cv16 number changes and the fixture stays as committed.

Two reasons the fixture must not be regenerated even opportunistically:

1. it still carries the pre-`a_eff` deltas, and cv16's own docstring records
   what regenerating does — `env_fine` becomes 2.731 dB, the DERIVED assert in
   `tests/crossval/test_rcs_mie_ka_sweep_gates.py` would demand a 4.1 dB fine
   gate while the deliberately redundant literal demands 4.0. That is a gate
   LOOSENING, and the decision on record is to keep 4.0;
2. the run costs ~30 min and buys nothing about #931.

## The control run (what IS submitted)

| field | value |
|---|---|
| VESSL run id | `369367259153` (submitted 2026-09-07, remilab/byungkwan) |
| yaml | `scripts/vessl_931_post_cv16.yaml` (`vessl run create -f`) |
| preset | `gpu-rtx4090`, cluster `remilab-c0`, `JAX_PLATFORMS=cpu` |
| command | `python -u validation/crossval/16_pec_sphere_mie_ka_sweep.py` |
| expected runtime | 5-10 min (the five GATED points only; **no** `--write-fixture`) |
| writes | nothing — without `--write-fixture` the script only gates and prints |
| outputs | `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv16-<UTC>/` |

Accept only if every gated `delta_db` and `a_eff_over_a` reproduces the
committed fixture row and the log carries the new `[CONDUCTOR MODEL]` line at
each point. Any movement means the fence leaked and the contract reached a
material fill.

## What the control is checking, and what it would cost if it failed

`assert_conductor_model()` (new, build-time, called from `run_point`) refuses
unless the sigma fill still equals `Sphere.mask(grid)`, and prints the
centre-sampled PEC-volume cell set beside it. Measured at cv16's own operating
points:

| bin | node-sampled (the model cv16 gates) | centre-sampled PEC volume | cells differing |
|---|---|---|---|
| coarse ka = 0.50 | N = 1082, a_eff/a = 0.988032 | N = 1123, a_eff/a = 1.000357 | 259 |
| fine ka = 2.00 | N = 9264, a_eff/a = 0.998319 | N = 9339, a_eff/a = 1.001006 | 957 |

If cv16 is ever brought under the contract, a_eff moves ~1.2 % at ka = 0.5, the
Mie reference leg moves with it (both legs share one `pi_a2`), and the whole
fixture — 15-point curve x 3 domains, the 7-point clearance scan, the
truncation witness — plus both gate constants must be regenerated from a
re-solve, ~30 min CPU. That is a PI decision. The design note is not internally
consistent on it: §5 lists "cv16 recomputed under centre sampling", §1.8 fences
the sigma fill, and the X-C brief settles it as FENCED. This branch follows the
brief and makes the disagreement legible rather than resolving it silently.
