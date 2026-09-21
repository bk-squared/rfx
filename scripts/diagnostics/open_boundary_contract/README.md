# Records behind the open-boundary contract

The rule is `docs/design_notes/20260921_open_boundary_contract.md`. This directory holds what it was
measured from: result tables and JSON, the drivers and job files that produced them, and the written
briefs the runs were made from (`BRIEF_*.md`). Field dumps (`*.npz`) are not here; they stayed on the
lab share. Measured 2026-09-20/21 on `remilab-c0` (rtx4090, float32), source trees exported with
`git archive` (`PROVENANCE.txt`), VESSL run ids in `run_id.txt` and in each subdirectory's own.

| directory | question | read first |
|---|---|---|
| top level | which patch ring-down arms grow, on which tree and grid sizing | `result_*.json`, `run_arms.py` |
| `dumps/` | where the growing field lives | `TABLE.md`, `*/leader_profiles.txt` |
| `cont/` | does continuing the ground plane through the absorber remove the growth | `TABLE.md`, `variant_note.txt` |
| `ports/` | which committed structures have a conductor ending at an absorber | `CENSUS.md` |
| `msl/`, `msl_inset/` | what continuing a microstrip strip into the absorber does; is the outer wall the cause | `EDGES.md`, `TABLE_display.md`, `msl_inset/TABLE.md` |
| `alpha/` | does the absorber's frequency-shift term set that damage | `TABLE.md`, `readback.md` |

`MANIFEST.txt` lists every file of the original bundle with its sha256. Three of them are not tracked
here, on purpose:

- `dumps/n2_pad10_main/leader_z.png`, `dumps/n2_pad10_main1012/leader_z.png`,
  `cont/n2_pad10_cpml4_a/slices.png`: the repository tracks figures under `docs/design_notes/figures/`
  only. The first and the third are there as `open_boundary_growing_wave_on_absorber_faces.png` and
  `open_boundary_ground_continued.png`.
- `alpha/readback_report.py`: an abandoned draft that does not parse. `alpha/readback_report_v2.py` is
  the one that ran (`alpha/REPORT.md`, "readback_report.py AST validation"); the repository requires
  every tracked Python file to parse.

The records name two fixtures by the labels they had when the runs were made: "cv06b" is the MSL notch
filter (`validation/crossval/06b_msl_notch_filter_uniform.py`), "cv20" the MSL phase-referee line
(`validation/crossval/20_msl_phase_referee.py`). The `ports/` before/after pair used plumbing smoke
fixtures whose |S11| is numerical noise and carries no information; `msl/` replaced it.

Absolute paths in these records (`/root/workspace/…`) are the lab pod's. The rule files some briefs
name there are private and are not part of this repository; nothing from them is quoted here.
