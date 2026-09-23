Read-only review execution record, 2026-09-23.

All generated files are under `.boundary-model/review_codex/`. No repository files, branches, index, worktrees, GitHub objects, or remote state were changed by this review. No GPU or VESSL calls were made. Python bytecode was disabled; temporary files, matplotlib configuration, and XDG caches were redirected here.

The standing checkout HEAD and origin/main both initially resolved to `c2dbf922bc9230cd6fa4054a7b6d84d118768bde`. The initial and final read-only status both contained only the existing untracked `cst/` and `docs/plans/`; tracked diff was empty. During the review another session advanced origin/main to `94fc5015c220c6c330580eb17d733c7a9d563887`. HEAD remained c2dbf922. I read that intervening diff: the only product file changed is `rfx/runners/distributed_v2.py`, passing material/absorber arrays as JIT arguments for all device topologies. All executed checks used c2dbf922; no checkout or fetch was performed.

Final successful commands, run with cwd equal to this directory:

```bash
env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/root/workspace/bk-workspace/.boundary-model/review_codex/cache TMPDIR=/root/workspace/bk-workspace/.boundary-model/review_codex MPLCONFIGDIR=/root/workspace/bk-workspace/.boundary-model/review_codex/mpl PYTHONPATH=/root/workspace/bk-workspace/rfx /root/workspace/bk-workspace/rfx/.venv/bin/python -B checks.py > checks.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/root/workspace/bk-workspace/.boundary-model/review_codex/cache TMPDIR=/root/workspace/bk-workspace/.boundary-model/review_codex MPLCONFIGDIR=/root/workspace/bk-workspace/.boundary-model/review_codex/mpl PYTHONPATH=/root/workspace/bk-workspace/rfx /root/workspace/bk-workspace/rfx/.venv/bin/python -B cavity_check.py > cavity.log 2>&1
```

Both exited 0. Final process CPU usage: checks.py 6.281 s; cavity_check.py 8.755 s. The first checks.py attempt had an unused BoundarySpec import from the wrong package; that scratch import was removed. An initial successful version of checks.py (6.163 CPU s) was rerun to add the positive dispersive-consumer control and use a canonical high ghost for its negative control. All checks stayed far below the ten CPU-minute limit.

`checks.py` compares the image operator against a mirrored periodic domain, exercises traced fields with static ghost rules, reproduces the current static-Bloch vmap refusal, checks period/index arithmetic, and exercises the RIS fallback with synthetic input. These are local numerical/code diagnostics, not measured S parameters.

`cavity_check.py` uses the public Simulation API and the B0c cavity/source/probe, with temporary process-local function substitutions only. It records the unchanged current wall, two image meshes, a dropped-image defect with only the legitimate y/z electric walls, and restored x electric walls while image calls remain active. The full probe records, timesteps, frequencies and preflight stdout are retained in NPZ, JSON and cavity.log. No production image implementation or new repository test was created.
