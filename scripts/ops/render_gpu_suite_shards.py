"""Render one VESSL job YAML per shard of the gpu-marked test suite.

The 29 gpu-marked test files (see ``gpu_suite_shards.json``) had never run in
any CI lane: every CPU lane deselects ``gpu`` and the single-GPU
``scripts/vessl_gpu_suite.yaml`` harness was submitted by hand and took ~3.8 h
serial. This renders K jobs that each run one line-balanced shard on its own
RTX 4090 against a checkout staged under
``claude-workspace/rfx/checkouts/<slug>/`` (rsync'd from the Mac by
``weekly_gpu_suite.sh``), writing a JUnit XML and the full pytest log to
``claude-workspace/rfx/runs/gpu-suite/<stamp>/shard-<i>/``.

Usage::

    python scripts/ops/render_gpu_suite_shards.py --slug gpu-suite-abc1234 \
        --stamp 20260902T120000Z --sha abc1234 --out /tmp/yamls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARDS = HERE / "gpu_suite_shards.json"

TEMPLATE = """name: rfx-gpu-suite-weekly-{stamp}-shard{i}
description: "Weekly gpu-marked pytest suite, shard {i}/{k} of {nfiles} files (line-balanced; these files run in no CI lane). main @ {sha}. Rendered by scripts/ops/render_gpu_suite_shards.py, submitted by scripts/ops/weekly_gpu_suite.sh from the lab Mac."
tags: [rfx, gpu-suite, weekly, shard{i}]
resources:
  cluster: remilab-c0
  preset: gpu-rtx4090
image: nvcr.io/nvidia/jax:24.10-py3
env:
  PYTHONUNBUFFERED: "1"
  XLA_PYTHON_CLIENT_PREALLOCATE: "false"
  HDF5_USE_FILE_LOCKING: "FALSE"
  LANG: "C.UTF-8"
  MPLBACKEND: "Agg"
mount:
  /root/workspace/: volume://remilab-fs/personal-workspaces/
run: |-
  set -eu
  ROOT=/root/work/{slug}
  mkdir -p "$ROOT"
  cp -r /root/workspace/claude-workspace/rfx/checkouts/{slug}/. "$ROOT/"
  cd "$ROOT"
  OUT=/root/workspace/claude-workspace/rfx/runs/gpu-suite/{stamp}/shard-{i}
  mkdir -p "$OUT"
  echo "shard {i}/{k}  main@{sha}" | tee "$OUT/meta.txt"
  python -m pip install -q "scipy>=1.11" "h5py>=3.8" "matplotlib>=3.7" "pytest>=7"
  export PYTHONPATH="$ROOT"
  python -c "import jax, rfx; print('probe ok | jax', jax.__version__, '| devices', jax.devices())" | tee -a "$OUT/meta.txt"
  set +e
  timeout 10800 python -m pytest -o addopts="" -m gpu -q -ra -p no:cacheprovider \\
      --junitxml "$OUT/junit.xml" \\
      {files} \\
      > "$OUT/pytest.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$OUT/rc"
  tail -n 40 "$OUT/pytest.log" || true
  echo "shard{i}_rc=$rc"
  exit 0
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = json.loads(SHARDS.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = spec["n_shards"]
    for i, files in enumerate(spec["shards"]):
        text = TEMPLATE.format(stamp=args.stamp, i=i, k=k, nfiles=len(files), sha=args.sha,
                               slug=args.slug, files=" ".join(files))
        (out / f"gpu_suite_shard{i}.yaml").write_text(text)
        print(out / f"gpu_suite_shard{i}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
