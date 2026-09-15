# E — examples: recompute record (#931 lattice ownership contract)

Branch `feat/931-examples`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-E-examples`.
All runs submitted at commit `7e0de966ad736606d3c6dbea2f0bd6d9b670d9de`
(the second of this group's three commits; the third adds only these notes).

Machine rule: nothing longer than ~1 minute is solved on the shared pod. Every
re-solve is one VESSL run per case, preset `gpu-rtx4090` with
`JAX_PLATFORMS=cpu` inside the job (these fixtures are CPU-produced).
YAMLs live in the worktree at `.vessl931/` (untracked) and were submitted from
a non-git directory — `vessl run create` walks up for a git repo and dies on a
worktree's `.git` FILE with
`NotADirectoryError: .../.git/HEAD`, so submit from outside the checkout.

| case | run id | script | expected runtime |
|---|---|---|---|
| ex-patch-demo | **369367259175** | `examples/tutorials/patch_antenna_demo.py` | ~22 min CPU (pre-change measurement 1345 s) |
| ex-nu-patch-demo | **369367259177** | `examples/tutorials/nonuniform_patch_demo.py` | ~19 min CPU (pre-change measurement 1169 s) |
| ex-tutorials | **369367259181** | the 12 tutorial / quickstart scripts end to end plus `rfx run examples/config/microstrip_thru.yaml` | ~30-40 min total |

Submit command used, per case:

```
cd <a directory that is NOT inside a git checkout>
vessl run create -f <case>.yaml        # retried 3x on Bad Gateway
```

Outputs land in `/root/workspace/claude-workspace/rfx/runs/issue931-post-<case>-<ts>/`,
with the latest path echoed to `issue931-post-<case>.latest`. Each job harvests
every file the run wrote into its staged checkout under `produced/`.

## What each run is for

**ex-patch-demo / ex-nu-patch-demo** — the two patch tutorials record measured
numbers in their docstrings and printed output (runtime, settling dB, the
harminv mode list, the deviation against openEMS 2.4221 GHz). Those numbers
are still the PRE-migration ones in the committed files: the migration removed
a vacuum cell from inside the realized cavity (patch demo solved a 1.905 mm
board against a declared 1.524 mm; the NU demo 2.0 mm against 1.5 mm), so
every recorded frequency is expected to move DOWN. The ingest phase replaces
the recorded numbers from these logs with the pre-change values quoted beside
them.

Pre-change baselines for the same two cases were submitted by the baseline
agent from `scratchpad/vessl_baseline/{ex-patch-demo,ex-nu-patch-demo}.yaml`
(ids in `scratchpad/vessl_baseline/submitted_ids.txt`); compare against those,
not against the docstrings.

Declared falsifier for both, from design note §5 (cv15 arm), already asserted
at build time and measured on this pod without solving:

* `patch_antenna_demo`: sheets at z-node 29 / 33, realized wall planes
  `[29, 33]`, node-to-node cavity 1.524 mm = declared `SUB_THICK`.
* `nonuniform_patch_demo`: sheets at z-node 33 / 39, realized wall planes
  `[33, 39]`, node-to-node cavity 1.500 mm = declared `h_sub`.

If a re-solve comes back with the resonance UNMOVED, the diagnosis is wrong:
the cavity changed by 25 % (patch) and 33 % (NU) in vacuum-cell terms, so the
frequency must move.

**ex-tutorials** — the zero-advisory gates and the migrated example scripts,
run where the box is not shared. Three of them
(`rcs_scattering`, `run_control_and_fields`, `materials_and_dispersion`) fail
`tests/contracts/test_tutorial_examples.py` on this pod for a reason that is
NOT the migration: that test runs each tutorial through `subprocess` with a
60 s timeout, and on this loaded pod `ports_and_sparams_101` alone takes 101 s
BEFORE the migration and 110 s after (measured, both). Two of the three
failures are on files this group never touched. The timeout was left alone;
this run is the evidence that the scripts themselves are healthy.

Checked directly on the pod, without VESSL, before submitting:

| script | result |
|---|---|
| `ports_and_sparams_101.py` | rc 0; realized z wall planes `[8, 12]` = 1.25 / 2.25 mm, strip-to-ground 1.00 mm; generic-port S11 bit-identical to pre-change; `All checks passed` 4 -> 3 |
| `slab_rt_flux_monitor.py` | rc 0; material arrays bit-identical to the nudged version (290 cells) |
| `rcs_scattering.py` | rc 0; PEC volume 1067 cells / sigma fill 1045 / analytic 1079.8; backscatter unchanged (the sigma path is fenced, §1.8) |
| `resonance_harminv.py` | rc 0; TE101 7.505940 GHz, error 0.005706 % — boundary-face PEC control, unchanged |
| `run_control_and_fields.py` | rc 0; `All checks passed`, until-decay 281 samples — control, unchanged |
| `cad_mesh_import_demo.py` | rc 0; plate realizes 2 cells, walls at z = 14 / 15 / 16 mm |
| `examples/config/microstrip_thru.yaml` | builds; sheets at z-node 12 / 14 = 2.0 / 3.0 mm |

## Not recomputed here, on purpose

* `tests/data/example_fidelity_snapshot.json` — the ingest phase regenerates it
  with `JAX_ENABLE_X64=0 python scripts/capture_example_fidelity_snapshot.py`
  AFTER the preflight and crossval groups land, because the snapshot covers
  validation scripts too. See `E-examples-example_fidelity_snapshot.md`.
* `docs/public/gallery/assets/**` and the committed `sparams.json` / `.s1p`
  from `scripts/_gallery_v3_patch_figs.py` and
  `scripts/precompute_gallery_artifacts.py` — outside this group's ownership.
  The gallery script's mesh changed from 1 mm to 0.5 mm cells (it had to: its
  foils were half-cell PEC Boxes, which the contract refuses), so its assets
  cost ~10x per case now.
* `scripts/patch_edgefed_s11_validation.py` — its own long-window run
  (`num_periods=200`) and the locks in
  `tests/locks/test_patch_edgefed_s11_passivity.py` /
  `test_patch_edgefed_resonance_harminv.py`. The board moved onto the lattice
  (dx 0.197 mm -> h_sub/4 = 0.19675 mm) and the metal moved onto the board
  faces, so both locks will move. Owned by whoever owns `tests/locks/`.

## One thing the merge agent should NOT "fix"

The core branch gained `tests/_realized_geometry.py` (`realized(sim)`,
`assert_wall_planes(...)`) after this group branched, and its docstring says a
second hand-rolled check is the drift the single-owner rule exists to stop.
That is right for tests. It is not available to the seven build-time checks
this group added, because they live in **shipped examples and top-level
scripts**, and a shipped tutorial must not import from `tests/`.

Those checks are not a second rule: each one calls
`rfx.boundaries.pec.realized_pec_edge_masks` and `realized_wall_planes` — the
single owner — over the arrays the assembly hands the stepper. What they
duplicate is four lines of assembly plumbing, and only because
`Simulation` has no public accessor for its realized edge set yet. The right
resolution is to give `Simulation` that accessor and have BOTH the tests
helper and the examples call it; until then, do not rewrite the examples to
import a test module.

Where the hand-rolled checks live:

* `examples/tutorials/patch_antenna_demo.py::realized_z_wall_planes`
* `examples/tutorials/nonuniform_patch_demo.py` (inline, after the sources)
* `examples/tutorials/ports_and_sparams_101.py::build_microstrip_ports`
* `examples/tutorials/slab_rt_flux_monitor.py::build_sim` (fidelity report,
  dielectric extent — not an edge check)
* `scripts/_gallery_v3_patch_figs.py::_assert_realized_stack`
* `scripts/precompute_gallery_artifacts.py::_assert_realized_planes`
  (covers both the NU port model and the uniform animation model)
* `scripts/patch_edgefed_s11_validation.py` and
  `scripts/msl_flux_ratio_dof.py` (inline, before preflight)

---

# Phase 2b (ingest) addendum — what the runs said, 2026-09-07

## Runs ingested

| case | run | verdict |
|---|---|---|
| ex-patch-demo | 369367259175 | ingested (commit `97d64765`) |
| ex-nu-patch-demo | 369367259177 | UNDER-SETTLED at 120 periods; superseded |
| ex-nu-patch-demo-settled | **369367259280** | ingested (commits `7d2686f8`, `1805f0e5`) |
| ex-tutorials | 369367259181 | all 13 steps rc 0; no recorded number in the tutorial prose moved |
| ex-gallery (GPU) | 369367259281 | **HUNG** before the script started — do not wait on it |
| ex-gallery-cpu | **369367259303** | rc 0; the measurement below |
| ex-gallery-patch-v3 | **369367259309** | submitted 20:18 UTC, ~1.5–2 h — the served patch bundle |

The baseline runs the post runs are read against: 369367259020 (patch demo),
369367259021 (NU demo). Both reproduce the numbers the committed docstrings
carried, which is what licenses replacing those numbers with the post runs'.

## The gallery is two pipelines, and only one of them is the served one

`scripts/precompute_gallery_artifacts.py` is NOT what the public pages read.
The committed bundle is the union of two generations, reconciled by a third
script:

* `precompute_gallery_artifacts.py` -> `sparams.{png,json,s1p,s2p}`,
  `smith.png`, `geometry.png`, `fields.gif`, `manifest.json`;
* `scripts/_gallery_v3_*_figs.py` -> `field_anim.gif`, `autodiff.png`,
  `gradient.json`, `s11_db.png`, `field_resonance.png`, `validation.png`,
  `field_xt.png`, `rt_overlay.png`, `field_te10.png`;
* `scripts/reconcile_gallery_manifests.py` -> rebuilds `assets[]` from disk and
  renames `fields.gif` -> `field_anim.gif`, which is the name
  `docs/public/gallery/patch_antenna.mdx` actually references.

So running the precompute script alone produces 7 files per case against the
11–12 committed, under partly different names. Copying that output over the
committed tree would half-replace each bundle and leave the served v3 assets
showing the old board. It was not done. The correct invocation is
precompute -> v3 figs -> reconcile, which is what 369367259309 runs.

## Which cases the contract actually moved (measured, run 369367259303)

`--case all` on the migrated tree, S-parameters compared against the committed
`sparams.json` element by element:

| case | max abs S diff | reading |
|---|---|---|
| `multilayer_fresnel` | 6.418e-06 | unchanged |
| `waveguide_wr90` | 6.167e-08 | unchanged |
| `patch_antenna` | **1.357** | changed completely |

Only the patch case declares thin conductors and only the patch case takes a
wire port, so this is the expected division and it is now measured rather than
assumed. `multilayer_fresnel` has no conductor at all and `waveguide_wr90` uses
waveguide ports and boundary-face PEC, neither of which §1.3 touches.

Consequence for the ingest: **do not regenerate the fresnel and waveguide
bundles.** Their assets would change only in timestamp and sha256, which is
churn that costs a reviewer real attention for no physics.

The patch move, per port: committed |S11| spans [0.2449, 0.4262] over
1.5–3.5 GHz, produced [0.7397, 0.9469]. The direction is the physical one — a
patch on FR4 fed by one port reflects most of what it is given away from the
match point, and an |S11| that never exceeds 0.43 anywhere in the band means
the old port was absorbing 80–94 % everywhere. The committed manifest already
says this case is `validation.passed = null`, tier `unqualified`, metric
"not reported: stored port setup fails current preflight", so neither the old
curve nor the new one is a validated result. The new one is not offered as one.
What changed under the contract: the foils stopped owning cells, so the port's
declared extent now spans laminate rather than laminate plus two vacuum cells,
and R8's half-open wire-port fix (`6d66ac65`) stopped the port driving one cell
past its declared end. Both act on the same port.

## Left for whoever picks this up

1. Ingest 369367259309: copy `produced/docs/public/gallery/assets/patch_antenna/**`
   into the worktree, commit with the run id and the sweep's chosen inset. The
   run re-runs the inset sweep ON the 0.5 mm mesh rather than reusing
   `MATCHED_INSET = 3.0e-3`, which was chosen on the 1 mm mesh; if the sweep
   prints no inset the run aborts rather than run `final` on a stale constant.
   If the swept inset differs from 3.0 mm, `MATCHED_INSET` in
   `scripts/_gallery_v3_patch_figs.py` must be updated to it in the same commit,
   or the next `final` silently reverts the bundle.
2. `docs/public/gallery/patch_antenna.mdx` quotes numbers from the old bundle;
   it is the docs owner's, and it cannot be corrected until (1) lands.
3. `tests/data/example_fidelity_snapshot.json` — still last, still the merge
   agent's, per `E-examples-example_fidelity_snapshot.md`.
