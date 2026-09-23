# Records behind the boundary-model pre-declaration

The design is `docs/design_notes/20260923_boundary_model_predeclaration.md`. This directory holds the
measurements it cites, made 2026-09-22/23 on `remilab-c0` (rtx4090) and on the lab pod's CPU from source
trees exported with `git archive` (commits named in each report), and the briefs they were made from.

| directory | question | read first |
|---|---|---|
| `B0/` | which walls, absorbers and wraps each entry point applies for each declaration (at 798ec64e, before PR #1205); the PEC cube, the PMC parallel-plate line, the absorber backing | `REPORT.md`, `MATRIX.md`, `WITNESS.md`, `INVENTORY.md` |
| `B0b/` | does a Floquet port at a scan angle solve that angle | `REPORT.md`, `TABLES.md` |
| `B0c/` | where a magnetic wall is realized: the 24 × 20 × 4 mm cavity at three dx on CPU, and the GPU relaunch | `REPORT.md`, `gpu_relaunch/DETAILS.md` |

Field dumps (`*.npz`), job logs and the measurement drivers stayed on the lab share
(`bk-workspace/.boundary-model/`), as did the three design reviews. Absolute paths in these records are
the lab pod's.
