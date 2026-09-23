All Python commands used `/root/workspace/bk-workspace/rfx/.venv/bin/python -B`; paths below are relative to `B/` unless written in full.

| Command | Last line / recorded result |
|---|---|
| `test ! -e B/src` | exit 0 before creation |
| `mkdir /root/workspace/bk-workspace/.801-measure/B/src` | exit 0; no output |
| `git -C /root/workspace/bk-workspace/rfx archive --format=tar origin/main` piped to `tar -x --no-same-owner -C /root/workspace/bk-workspace/.801-measure/B/src` | exit 0; no output |
| `git -C /root/workspace/bk-workspace/rfx rev-parse origin/main > B/src/PROVENANCE.txt` | exit 0; file contains `e7f7e02704fd46ea7e21f127b19fc81cb66d6148` |
| Copy prior drivers, YAMLs, tables and results into `failed_launch/` | `Failed-launch drivers, specifications, tables, and 112 records copied; jobs/ retained in place.` |
| `python -B prepare_jobs.py` | `vessl_plane.yaml: sh -n passed; no heredocs; one structure` (four passed) |
| AST and exact shell-block checks | `4 POSIX job blocks passed; no Git calls or heredocs; REPORT.md unchanged.` |
| `python -B verify_job_sources.py` | all five driver SHA-256 checks passed; job copies in `jobs_2/*/source_verification.txt` |
| `env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B sweep_driver.py plane --scale 0 --layers 4 --dry` | `ARM_FINAL` status `build-only`, wall 1.898917565 s; [full last line](smoke_export_plane.log) |
| `vessl run list` | [pre_submit_runs_2.txt](pre_submit_runs_2.txt); prior four failed runs retained as cited records |
| `vessl run create -f vessl_msl.yaml` | `Check your Run at: https://app.vessl.ai/remilab/runs/byungkwan/369367263409`; one submission |
| Write submitter-known run ID to `jobs_2/msl/run_id.txt` | `369367263409` |
| `vessl run read 369367263409` | status `completed`; [terminal read](jobs_2/msl/read_final.txt) |
| `vessl run logs 369367263409 --tail 100000` | `[12:11:41.717574] Workload status changed to completed`; [full available log](jobs_2/msl/provider.log) |
| `vessl run create -f vessl_waveguide.yaml` | `Check your Run at: https://app.vessl.ai/remilab/runs/byungkwan/369367263410`; one submission |
| Write submitter-known run ID to `jobs_2/waveguide/run_id.txt` | `369367263410` |
| `vessl run read 369367263410` | status `completed`; [terminal read](jobs_2/waveguide/read_final.txt) |
| `vessl run logs 369367263410 --tail 100000` | `[11:16:41.660104] Workload status changed to completed`; [full available log](jobs_2/waveguide/provider.log) |
| `vessl run create -f vessl_patch.yaml` | `Check your Run at: https://app.vessl.ai/remilab/runs/byungkwan/369367263411`; one submission |
| Write submitter-known run ID to `jobs_2/patch/run_id.txt` | `369367263411` |
| `vessl run read 369367263411` | status `completed`; [terminal read](jobs_2/patch/read_final.txt) |
| `vessl run logs 369367263411 --tail 100000` | `[11:18:20.705368] Workload status changed to completed`; [full available log](jobs_2/patch/provider.log) |
| `vessl run create -f vessl_plane.yaml` | `Check your Run at: https://app.vessl.ai/remilab/runs/byungkwan/369367263412`; one submission |
| Write submitter-known run ID to `jobs_2/plane/run_id.txt` | `369367263412` |
| `vessl run read 369367263412` | status `completed`; [terminal read](jobs_2/plane/read_final.txt) |
| `vessl run logs 369367263412 --tail 100000` | `[11:13:17.721205] Workload status changed to completed`; [full available log](jobs_2/plane/provider.log) |
| `python -B reduce_2.py` | `REDUCTION_COMPLETE 112 arms; 196 sweep solves; 3962714 sweep timesteps` |
| `python -B write_report_2.py` | `REPORT_2 written; prior REPORT.md unchanged` |
| `python -B verify_2.py` | see [verification_2.json](verification_2.json) |
| `test ! -e /root/workspace/bk-workspace/rfx-wt-B-sweep` | exit 0; no worktree recreated |
