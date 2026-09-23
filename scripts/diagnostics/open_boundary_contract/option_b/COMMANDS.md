Commands ran with Python `/root/workspace/bk-workspace/rfx/.venv/bin/python`; worktree `rfx-wt-B-sweep`; output root `.801-measure/B`.

| Command | Last line / result |
|---|---|
| `git -C /root/workspace/bk-workspace/rfx worktree add --detach /root/workspace/bk-workspace/rfx-wt-B-sweep origin/main` | `HEAD is now at e7f7e027 A conductor that reaches an absorbing boundary continues through the absorber; a lossless grounded patch no longer gains energy (#801) (#1178)` |
| `python -B sweep_driver.py msl --scale 0 --layers 4 --dry` | `build-only`; wall 90.481 s; [exact last line](smoke_msl_0_4.log) |
| `python -B sweep_driver.py patch --scale 1 --layers 6 --dry` | `build-only`; wall 6.508 s; [exact last line](smoke_patch_1_6.log) |
| `python -B sweep_driver.py plane --scale 0 --layers 4 --dry` | `build-only`; wall 1.841 s; [exact last line](smoke_plane_0_4.log) |
| `python -B sweep_driver.py waveguide --scale 0.3 --layers 8 --dry` | `build-only`; wall 7.569 s; [exact last line](smoke_waveguide_03_8.log) |
| `python -B sweep_driver.py waveguide --scale 1 --layers 8 --dry` | `STOP`; wall 8.108 s; [exact last line](smoke_waveguide_1_8.log) |
| AST parse of five driver/staging files | `prepare_jobs.py: AST parse passed` |
| `python -B prepare_jobs.py` | `vessl_plane.yaml: sh -n passed; no heredocs; one structure` (all four passed) |
| `vessl run list` | [pre_submit_runs.txt](pre_submit_runs.txt); no `rfx-801-B-sweep-` runs before submission |
| `vessl run create -f vessl_msl.yaml` | `Check your Run at: https://app.vessl.ai/remilab/runs/byungkwan/369367263402` |
| `vessl run read 369367263402` | status `failed`; [full read](jobs/msl/read_final.txt) |
| `vessl run logs 369367263402 --tail 100000` | `[11:00:41.586800] Workload status changed to failed`; [full log](jobs/msl/provider.log) |
| `vessl run create -f vessl_waveguide.yaml` | `Check your Run at: https://app.vessl.ai/remilab/runs/byungkwan/369367263403` |
| `vessl run read 369367263403` | status `failed`; [full read](jobs/waveguide/read_final.txt) |
| `vessl run logs 369367263403 --tail 100000` | `[11:00:39.024154] Workload status changed to failed`; [full log](jobs/waveguide/provider.log) |
| `vessl run create -f vessl_patch.yaml` | `Check your Run at: https://app.vessl.ai/remilab/runs/byungkwan/369367263404` |
| `vessl run read 369367263404` | status `failed`; [full read](jobs/patch/read_final.txt) |
| `vessl run logs 369367263404 --tail 100000` | `[11:01:32.738925] Workload status changed to failed`; [full log](jobs/patch/provider.log) |
| `vessl run create -f vessl_plane.yaml` | `Check your Run at: https://app.vessl.ai/remilab/runs/byungkwan/369367263405` |
| `vessl run read 369367263405` | status `failed`; [full read](jobs/plane/read_final.txt) |
| `vessl run logs 369367263405 --tail 100000` | `[11:00:47.723727] Workload status changed to failed`; [full log](jobs/plane/provider.log) |
| `git status --porcelain=v1 --untracked-files=all` | empty; [saved output](worktree_status_before_removal.txt) |
| `git log --oneline origin/main..` | empty; [commit list](commit_list.txt) |
| `git -C /root/workspace/bk-workspace/rfx worktree remove /root/workspace/bk-workspace/rfx-wt-B-sweep` | exit 0; no output |
| `test ! -e /root/workspace/bk-workspace/rfx-wt-B-sweep` | exit 0; no output |
| Artifact verification | 112 STOP JSON records, four nonempty provider logs, zero solves; report links exist; [verification.json](verification.json) |
