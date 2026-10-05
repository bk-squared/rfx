#!/bin/bash
set -euo pipefail
root=/root/workspace/claude-workspace/rfx/runs/s31
scratch=/private/tmp/s31
mkdir -p "$scratch" "$root/results"
export TMPDIR="$scratch" PIP_CACHE_DIR="$scratch/pip-cache"
export UV_CACHE_DIR="$scratch/uv-cache" UV_PYTHON_INSTALL_DIR="$scratch/python"
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$scratch/mpl"
python -m pip install -q --target "$scratch/bootstrap" uv==0.12.19
UV="$scratch/bootstrap/bin/uv"
"$UV" venv -q --python 3.11.16 "$scratch/venv"
PY="$scratch/venv/bin/python"
"$UV" pip install -q --python "$PY" --exclude-newer 2026-09-27T13:37:21Z 'jax[cuda12]==0.10.2' 'numpy==2.4.6' 'scipy==1.17.1' 'h5py==3.16.0' 'matplotlib==3.11.2' 'ml_dtypes==0.6.0' 'pyyaml==6.0.3' 'optax==0.2.8'
"$UV" pip freeze --python "$PY" > "$root/results/pip-freeze.txt"
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv > "$root/results/gpu.csv"
for revision in before after; do
  mkdir -p "$scratch/$revision"
  tar -xzf "$root/$revision.tar.gz" -C "$scratch/$revision"
  for board in msl wire; do
    echo "START $revision $board $(date -u +%FT%TZ)"
    PYTHONPATH="$scratch/$revision" "$PY" "$root/measure_sources.py" \
      --board "$board" --inputs "$root/inputs" \
      --out "$root/results/$revision-$board.json" \
      > "$root/results/$revision-$board.log" 2>&1
    echo "DONE $revision $board $(date -u +%FT%TZ)"
  done
done
echo 'S31 COMPLETE'
