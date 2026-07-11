# crossval — research lane: material calibration / inverse parameter estimation

## Self-classification (issues #300 / #301)

- **Role: research-only.** This directory is **NOT claims-bearing**. Nothing
  here is validation evidence for rfx; membership in this directory implies
  **no** validation status whatsoever. Claims-bearing cross-validation lives in
  `examples/crossval/` (registry-gated post-#300) and in the committed
  physics-gate artifacts.
- **Scope:** *material calibration / inverse parameter estimation* research
  (issue #273): multi-solver patch cross-runs whose end use is anchoring the
  differentiable calibration pipeline (`calibrate_material_s11`, recovered
  eps_r/tan-delta) against independent solvers and, eventually, a fabricated
  board (`DESIGN_DOC_patch_student.md`). Deliberately **not** called bare
  "calibration": in `docs/guides/sparameter_support_matrix.md` that word means
  the caveated S-matrix **port-calibration** convention, which is a different
  thing.

## Why this path (placement rationale)

`scripts/crossval/` exists nowhere on `origin/main` and would invent a new
top-level lane class; `examples/crossval/` is claims-bearing territory being
cleaned under issues #300/#301 (post-#300 every script there must appear in a
registry). This branch already established `scripts/research/calibration/`
(`demo_shape_only_flaw.py`, committed), so the lane's home is
`scripts/research/calibration/crossval/`.

## Contents

- `verdicts.json` — the falsification ledger: one row per killed/kept
  hypothesis from the 2026-07 campaigns, with numbers and artifact paths.
  Read this first.
- `evidence/` — small (<100 KB) copies of the key artifacts backing
  `verdicts.json` rows. They live here because `out*/` is gitignored;
  everything else regenerates from the scripts.
- `DESIGN_DOC_patch_student.md` — student-facing fabrication / CST / VNA
  design doc for the X-band inset patch (the measurement anchor).
- `palace_patch/` — Palace (FEM) third-solver mesh generator + configs +
  README (retraction + final Level-1/Level-2 tables recorded there).
- `matched_patch_geometry.json` — geometry spec emitted by the openEMS inset
  sweep.

## Script inventory (#300 pre-conformance)

Execution tiers: `gpu` = rfx FDTD, sized for a GPU (CPU runs but is slow);
`openems` = needs a local openEMS install; `meep` = needs the dedicated Meep
venv (serial build, `/tmp/meepenv`); `palace` = needs the Palace binary
(`mpirun -n N palace <config>`); `cpu` = plain numpy/matplotlib.

| script | role | oracle | tier | exit semantics |
|---|---|---|---|---|
| `rfx_patch_xband_canonical_frame.py` | **current** rfx producer: X-band inset patch in the canonical z_lo-PEC frame (`--mode s11 / harminv / shielded-harminv`) | openEMS + Palace legs, Balanis 9.21 GHz | gpu | 0 after writing JSON (producer, no gate) |
| `openems_patch_inset_xband.py` | openEMS replica of the X-band inset patch (+ `--shielded` Level-1 box; prints passivity PASS/FAIL, 1.05 per-bin slack per `tests/test_msl_port_integration.py:411`) | rfx + Palace legs | openems | 0 after writing JSON (producer, no gate) |
| `palace_patch/mesh_patch.py` | Gmsh mesh generator for the Palace legs | — | cpu (gmsh) | 0 after writing `.msh` |
| `palace_patch/patch_s11.json`, `patch_eigen*.json` | Palace driven / eigenmode configs | — | palace | Palace exit status |
| `rfx_patch_inset_xband.py` | rfx X-band inset sweep — **SUPERSEDED-BY** `rfx_patch_xband_canonical_frame.py` (retired interior-ground frame; kept as history) | — (history) | gpu | 0 after writing JSON |
| `rfx_patch_inset_msl.py` | rfx canonical-substrate (2.1 GHz) inset sweep — **SUPERSEDED-BY** `rfx_patch_xband_canonical_frame.py` (same retired frame) | — (history) | gpu | 0 after writing JSON |
| `rfx_inset_harminv.py` | port-independent ring-down of the inset geometry (ast-imports `rfx_patch_inset_xband.py`, same dir) | Level-1 resonance vs openEMS/Palace | gpu | 0; prints modes (diagnostic) |
| `rfx_inset_drive_probe.py` | port-driven interior-probe diagnostic (does feed energy reach TM010?) | — (diagnostic) | gpu | 0; prints spectra (diagnostic) |
| `meep_patch_s11.py` | Meep third-solver producer, 2.4 GHz FR4 patch ring-down (Harminv-primary; env-tunable) | openEMS 2.487 / rfx 2.553 / Balanis 2.424 GHz window | meep | 0 after writing JSON; window verdict printed (see `verdicts.json`: both Meep rows are KILL) |
| `openems_patch_s11.py` | openEMS producer, 2.4 GHz canonical patch | rfx convergence sweep | openems | 0 after writing JSON |
| `openems_patch_inset_sweep.py` | openEMS inset-depth matching sweep, canonical substrate | — | openems | 0 after writing JSON |
| `openems_patch_qwt.py` | openEMS quarter-wave-transformer matched patch | — | openems | 0 after writing JSON |
| `rfx_patch_s11_convergence.py` | rfx mesh-convergence series, 2.4 GHz patch | openEMS reference | gpu | 0 after writing JSON |
| `rfx_patch_canonical.py` | rfx canonical Rogers-plate patch (~2.1 GHz) | openEMS canonical tutorial | gpu | 0 after writing JSON |
| `compare_patch_s11.py` | aggregator/overlay plot: rfx sweeps vs openEMS reference | — (consumer) | cpu | 0 after writing PNG |

**Exit-code convention** (for anything promoted OUT of this lane, per
`scripts/run_crossval_cpu.py`): `0` = pass including external cross-check,
`1` = a numeric self-check gate failed, `2` = external reference or optional
dependency missing = **inconclusive, not a pass**. The producers above predate
that convention and exit 0 after writing artifacts; promotion to
`examples/crossval/` requires adopting 0/1/2 plus a registry entry (#300).

## Documented limits (do not un-learn these)

1. **The patch |S11| dip frequency is mesh-limited and an unstable argmin**;
   it must never be a gate (`tests/test_issue80_patch_s11_regression.py`).
   This lane gates on **recovered material parameters / band-integrated
   curves only**. Dip depth is even weaker: not comparable across solvers
   while loss models differ (see `palace_patch/README.md`).
2. **Matched-thru floor:** if the MSL matched-thru |S11| floor is cited
   anywhere, the honest figure is **~0.16–0.22** — the clean-mesh
   staircase-Z0 floor from the mesh-convergence study (#183,
   `scripts/diagnostics/msl_thru_mesh_convergence.py`). The oft-quoted
   `0.118` was a danger-zone artifact; do **not** cite 0.118 as the floor.

## Snapshot note

`/root/workspace/lab-shared/rfx-patch-crossval/` carries a student-facing snapshot of the
comparison data (JSONs, Palace CSV, plots). **The repo copy here is
canonical**; the lab-shared copy is a convenience export and may lag.
