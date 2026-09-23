# Brief for Codex — copy the small measurement records into the design-note worktree

Mechanical file work only. No interpretation, no edits to any record's content, no git commands at all
(no add, no commit, no push, no checkout). If anything is unexpected, stop and write down what happened.

## Command safety (hard rules)
- One state-changing action per command. Never `rm`, never `mv`, never overwrite an existing file.
  Copies only (`cp -n`), plus `mkdir` for new directories.
- Source (READ ONLY): `/root/workspace/bk-workspace/.801-measure/`.
- Destination: `/root/workspace/bk-workspace/rfx-wt-obc/scripts/diagnostics/open_boundary_contract/`
  (create it with `mkdir`; if it already exists, STOP). Touch nothing else in that worktree.

## What to copy (keep each file's relative path under the destination)
Small text records ONLY. No `.npz`, no `.png` except the two named below, no `__pycache__`, no source
trees (`src-*`), no `mpl_config`, no Codex run logs (`codex_*_run.log`, `codex_*_last_message.md`), no
VESSL log files (`vessl_*.log`), nothing larger than 150 kB.
- top level: `PROVENANCE.txt`, `run_id.txt`, `run_arms.py`, `dump_fields.py`, `reduce_dumps.py`,
  `leader_replot.py`, `leader_dump_n2_1012.py`, every `vessl_*.yaml`, every `result_*.json`,
  every `BRIEF_*.md`
- `dumps/`: `TABLE.md`; for each label directory its `meta.json`, `summary.json`, `leader_profiles.txt`;
  and exactly two figures: `dumps/n2_pad10_main/leader_z.png`, `dumps/n2_pad10_main1012/leader_z.png`
- `cont/`: `REPORT.md`, `TABLE.md`, `variant_note.txt`, `build_check.txt`, `run_id.txt`; for each arm
  directory its `result.json` and `summary.json`; plus `cont/n2_pad10_cpml4_a/slices.png`
- `ports/`: `REPORT.md`, `CENSUS.md`, `TABLE.md`, `run_id.txt`, `verification.json`, the `*.py` drivers,
  the two `vessl_ports_*.yaml`
- `msl/`: `REPORT.md`, `EDGES.md`, `TABLE_display.md`, `run_id.txt`, `verification.json`, the `*.py`
  drivers, the two yaml files
- `msl_inset/`: `REPORT.md`, `EDGES_inset1.md`, `TABLE.md`, `run_id.txt`, the `*.py` drivers, yaml files
- `alpha/`: `REPORT.md`, `TABLE.md`, `readback.md`, `run_id.txt`, `driver.py`, other `*.py`, yaml files
If a named file does not exist, do not substitute another; list it as missing.

## Then write two new files in the destination
- `MANIFEST.txt`: one line per copied file: sha256, size in bytes, relative path; sorted by path; and a
  final line with the file count and the total size.
- `SKIPPED.txt`: every file under the source that matched a "what to copy" pattern but was skipped, with
  the reason (too large / missing / excluded type).

## Verify and report (print to stdout; also save as `COPY_REPORT.txt` in the destination)
- `find <destination> -type f | wc -l`, `du -sh <destination>`, the ten largest files with sizes;
- for every copied file, confirm sha256(source) == sha256(destination) and print the count that match
  and the count that do not;
- confirm no `.npz`, `__pycache__`, `src-*` or file over 150 kB is present in the destination.
