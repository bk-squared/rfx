# Review witnesses for the boundary-model pre-declaration

The design note (`docs/design_notes/20260923_boundary_model_predeclaration.md`) quotes numbers that
the three independent reviews measured, not the B0 records. This directory holds the scripts that
produced them and the output they printed, so each number has a file in the repository.

| directory | review | what it measured |
|---|---|---|
| `opus_review1/` | first design review (Opus, 2026-09-22) | the one-cell PMC line (`w3_repro.py`), the dead face-node plane (`pmc_dead_plane.py`), the image-wall prototype (`pmc_image.py`), the periodic ring period (`periodic_period.py`: 11.970 GHz for a declared 24 mm ring at dx = 1 mm), the `RISUnitCell` grid (`ris_grid.py`: 10.5 mm), the GPU fast path on CPU (`fastpath_cpu.py`) |
| `opus_review2/` | second review of the leader's recommendations and the recheck of the final note (Opus, 2026-09-23) | 1-D eigenvalue maps of rfx's kernels under each termination (`wall_eigs*.py`: the free termination at L + 1.5 dx, the image wall at second order on uniform, graded and dielectric face nodes, the (2,4) ribbon), the Bloch phase in the smoothing and Debye branches (`bloch_branch*.py`), the periodic index map (`d2_*.py`), the image on the one-cell line (`w3_image.py`) |
| `codex_review/` | Codex review (2026-09-23) | the mirrored-domain checks (`checks.py`, `checks.json`: 0 V/m for vacuum and εr = 4, −0.0550 / −0.0563 / −0.0550 V/m on the Debye / Lorentz / mixed E updates), the RIS normalization, the magnetic cavity variants (`cavity_check.py`, `cavity.json`) |

How these were kept. The two Opus reviews ran in session scratchpads under `/tmp`, which a pod
restart on 2026-09-23 wiped. Their scripts were rebuilt from the reviewers' session transcripts by
replaying every file write, edit and in-place `sed` in order; `RUN_OUTPUTS.md` and `LOG_OUTPUTS.md`
are the commands the reviewers ran and the output they saw, copied from the same transcripts. The
paths inside those commands point at the lost scratchpads. Nothing here was re-run after the
rebuild. Codex's files are copied as it wrote them (`checks.json` records the rfx commit it ran on). All runs
are CPU.
