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
  # Python 3.11 + JAX 0.10.2 in a uv venv (PI 2026-09-25); see scripts/vessl_validation_lane_a6000.yaml.
  # Everything below runs "$PY", never the image's python 3.10 (JAX 0.4.33, #1252).
  python -m pip install -q "uv==0.12.19"
  uv venv -q --python 3.11.16 /tmp/venv-py311
  PY=/tmp/venv-py311/bin/python
  uv pip install -q --python "$PY" --exclude-newer 2026-09-27T13:37:21Z "jax[cuda12]==0.10.2" "numpy==2.4.6" "scipy==1.17.1" "h5py==3.16.0" "matplotlib==3.11.2" "ml_dtypes==0.6.0" "pyyaml==6.0.3" "optax==0.2.8" "pillow==12.3.0" "pytest==9.1.1" "pytest-split==0.11.0"
  # Assert what was realized, on its own line: a pipe under `set -eu` would hide a failure.
  "$PY" -c "import jax, sys; assert sys.version_info[:2] == (3, 11), sys.version; assert jax.__version__ == '0.10.2', jax.__version__; assert jax.default_backend() == 'gpu', jax.default_backend()"
  uv pip freeze --python "$PY" > "$OUT/pip_freeze.txt"
  export PYTHONPATH="$ROOT"
  "$PY" -c "import jax, rfx; print('probe ok | jax', jax.__version__, '| devices', jax.devices())" | tee -a "$OUT/meta.txt"
  set +e
  timeout 10800 "$PY" -m pytest -o addopts="" -m gpu -q -ra -p no:cacheprovider \\
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
    tests_root = HERE.parents[1] / "tests"

    def resolve(name: str) -> str:
        hits = sorted(tests_root.rglob(name))
        if len(hits) != 1:
            raise SystemExit(f"{name}: expected exactly one match under tests/, found {hits}")
        return str(hits[0].relative_to(HERE.parents[1]))

    for i, names in enumerate(spec["shards"]):
        files = [resolve(n) for n in names]
        text = TEMPLATE.format(stamp=args.stamp, i=i, k=k, nfiles=len(files), sha=args.sha,
                               slug=args.slug, files=" ".join(files))
        (out / f"gpu_suite_shard{i}.yaml").write_text(text)
        print(out / f"gpu_suite_shard{i}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
