# cv17 recompute — issue #931 (group X-C): CONTROL ONLY

Branch `feat/931-crossval-C`.

## Verdict: cv17 has no conductor, so nothing may move

The lattice ownership contract changes PEC VOLUME sampling from node to cell
CENTRES and leaves dielectric sampling untouched (design note §1.1). cv17 is
dielectric-only, so every number must be bit-identical across the change. It is
the design note's §5 "dielectric-only cases" falsifier, applied to the case
where the raster IS the claim (the BINARY rasterize path with no sub-cell
interface averaging).

## The control run

| field | value |
|---|---|
| VESSL run id | `369367259154` (submitted 2026-09-07, remilab/byungkwan) |
| yaml | `scripts/vessl_931_post_cv17.yaml` (`vessl run create -f`) |
| preset | `gpu-rtx4090`, cluster `remilab-c0`, `JAX_PLATFORMS=cpu` |
| command | `python -u validation/crossval/17_dielectric_sphere_mie.py` |
| expected runtime | 5-10 min (gated bins only; **no** `--write-fixture`) |
| writes | nothing — without `--write-fixture` the script only gates and prints |
| outputs | `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv17-<UTC>/` |

Accept only if every gated `delta_db` and the material gate (`eps_realized`
within 0.5 % of 2.56, exactly two distinct eps values) reproduce the committed
fixture rows EXACTLY. "Close" is a failure here: the claim is bit-identity, and
a dielectric number that moved means the conductor contract leaked into the
material path.

## No artifact is regenerated

`_17_dielectric_results/rfx.json`, `material_blind_window.json`,
`cv17_permittivity_island.json` and `cv17_gated_A.log` all stay as committed.

## Static counterpart already in the tree

`tests/crossval/test_rcs_dielectric_sphere_mie_gates.py::test_cv17_dielectric_raster_is_untouched_by_the_ownership_contract`
asserts the same thing without a solve, at every gated bin: two distinct eps
values with the non-background one at the declared 2.56, and an occupied-cell
count equal to the NODE-sampled shape mask (1082 at ka = 0.5 — the same sphere
on the same mesh as cv16). The VESSL run is the physics leg of that claim; the
test is the cheap one that runs in CI.
