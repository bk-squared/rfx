All task paths are relative to `/root/workspace/bk-workspace/.801-measure/B/`. Local Python is `/root/workspace/bk-workspace/rfx/.venv/bin/python -B`; GPU jobs use their image Python with `-B`. Every solve ran on VESSL.

| Command / action | Last line or result |
| --- | --- |
| Read full brief, Addenda 1/2, DESIGN.md, REPORT.md, REPORT_2.md and inherited command/report rules | Read before implementation; source export retained |
| Hash source export and original reports/drivers into baseline_3.json | BASELINE 2749 files |
| mkdir jobs_3 raw_3 results_3 failed_instrument | exit 0 |
| mv TABLE_plane.md CROSSINGS_plane.csv SUMMARY_2_plane.json WITNESSES_2_plane.csv failed_instrument/ | exit 0; separate command |
| mv results/plane failed_instrument/results_plane | exit 0; separate command |
| python -B oracle_3.py exact --scale 1 --layers 8 --dry | BUILD_ONLY; smoke_oracle_3.log |
| python -B diagnostic_3.py msl --dry | DIAGNOSTIC_FINAL build-only msl; smoke_diagnostic_msl_3.log |
| python -B diagnostic_3.py waveguide --dry | DIAGNOSTIC_FINAL build-only waveguide; smoke_diagnostic_waveguide_3.log |
| python -B oracle_3.py frequency --f0-ghz 1 --scale 1 --layers 8 --dry | BUILD_ONLY; smoke_frequency_1_3.log |
| Compare generated s=1/N=8 numerical method AST to committed method | EXACT_ORACLE_AST_IDENTICAL; CPU_FDTD_steps=0 |
| Compare stock CPML arrays to alpha-scale-1 arrays, without time stepping | SCALE_1_CPML_ARRAYS_BIT_IDENTICAL_TO_STOCK; 30 arrays; FDTD_steps=0 |
| python -B prepare_jobs_3.py | Six YAML blocks passed sh -n; no Git calls or heredocs; prepare_jobs_3.log |
| vessl run list | pre_submit_runs_3.txt; prior eight B runs retained as cited reference records |
| vessl run create -f vessl_3_oracle.yaml | URL ending in 369367263428; exactly one submission; jobs_3/oracle/submit.log |
| Write submitter-known ID to jobs_3/oracle/run_id.txt | 369367263428; separate command after submission |
| vessl run read 369367263428 | completed; jobs_3/oracle/read_final.txt |
| vessl run logs 369367263428 --tail 100000 | [12:51:17.953705] Workload status changed to completed |
| vessl run create -f vessl_3_msl.yaml | URL ending in 369367263429; exactly one submission; jobs_3/msl/submit.log |
| Write submitter-known ID to jobs_3/msl/run_id.txt | 369367263429; separate command after submission |
| vessl run read 369367263429 | completed; jobs_3/msl/read_final.txt |
| vessl run logs 369367263429 --tail 100000 | [12:51:02.728081] Workload status changed to completed |
| vessl run create -f vessl_3_waveguide.yaml | URL ending in 369367263430; exactly one submission; jobs_3/waveguide/submit.log |
| Write submitter-known ID to jobs_3/waveguide/run_id.txt | 369367263430; separate command after submission |
| vessl run read 369367263430 | completed; jobs_3/waveguide/read_final.txt |
| vessl run logs 369367263430 --tail 100000 | [12:52:59.731402] Workload status changed to completed |
| vessl run create -f vessl_3_frequency_1.yaml | URL ending in 369367263447; exactly one submission; jobs_3/frequency_1/submit.log |
| Write submitter-known ID to jobs_3/frequency_1/run_id.txt | 369367263447; separate command after submission |
| vessl run read 369367263447 | completed; jobs_3/frequency_1/read_final.txt |
| vessl run logs 369367263447 --tail 100000 | [12:58:08.715292] Workload status changed to completed |
| vessl run create -f vessl_3_frequency_2.yaml | URL ending in 369367263449; exactly one submission; jobs_3/frequency_2/submit.log |
| Write submitter-known ID to jobs_3/frequency_2/run_id.txt | 369367263449; separate command after submission |
| vessl run read 369367263449 | completed; jobs_3/frequency_2/read_final.txt |
| vessl run logs 369367263449 --tail 100000 | [12:53:56.734742] Workload status changed to completed |
| vessl run create -f vessl_3_frequency_4.yaml | URL ending in 369367263450; exactly one submission; jobs_3/frequency_4/submit.log |
| Write submitter-known ID to jobs_3/frequency_4/run_id.txt | 369367263450; separate command after submission |
| vessl run read 369367263450 | completed; jobs_3/frequency_4/read_final.txt |
| vessl run logs 369367263450 --tail 100000 | [13:08:14.755140] Workload status changed to completed |
| Replay retained WR-90 matrix through its library shared passivity guard | PASSIVITY_WARNING_3.txt and passivity_replay_3.json; zero FDTD solves |
| Compare retained variant time traces with two extraction formulas | variant_trace_replay_3.json; zero FDTD solves |
| Compare original/resized-reference and point/5x5-source records | COMPARATOR_COMPARISONS_3.json; identical CPML probe samples for the two reference sizes |
| python -B field_maps_3.py | FIELD_MAP raw_3/diagnostic_waveguide/fields_end_00.png; field_maps_3.log; four final maps |
| python -B reduce_3.py | REDUCTION_COMPLETE: oracle tables, final-field table, repeat comparisons |
| Initial python -B verify_3.py with exact local NumPy logarithm equality | AssertionError; one replay differed by 7.105427357601002e-15 dB; log_replay_precision_3.json preserves all values |
| Replay the same saved peak amplitudes with math.log10, keeping exact equality | All 19 stored dB values match exactly; no tolerance or GPU result changed |
| python -B verify_3.py | VERIFIED: 2744 source files unchanged; 19 oracle measurements, 3 STOPs, 4 field dumps, 42 GPU solves, 98815 steps; CPU FDTD steps 0. |

The initial oracle/MSL/WR-90 provider-capacity queue is retained in the full logs and initial/snapshot status files. The three frequency jobs were submitted only after all nine exact-oracle cases completed. The 0.5 GHz case has no submission; its three explicit STOP records state the assumed duration, sizing calculation and options. No job was relaunched.

`write_pending_report_3.py` was prepared during the queue but **never executed**: the oracle started before that report was written. The final report is generated by `write_report_3.py` from completed measurements and the explicit 0.5 GHz STOPs.

No CPU FDTD solve, repository test suite, Git command, commit, push, PR, GitHub comment, worktree operation or VESSL run deletion occurred. Original and new evidence-bearing runs were retained under the reference-run exception. All Python imports disabled bytecode writes to the source export.
