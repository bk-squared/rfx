#!/usr/bin/env bash
# Submit one GPU lane.
#   bash validation/vessl/submit.sh j0|j1|j2 [--no-upload]
# The source is the WORKING TREE (the backend is uncommitted), so the tarball
# is rebuilt and re-uploaded to the VESSL-managed volume on every submission
# and the job pulls it in with the YAML's import: section (see
# build_gpu_lanes.py "WHY THIS FILE EXISTS": mount: of that volume is
# rejected, and there is no way to write the NFS from the Mac).
# ONE GPU job at a time on this shared cluster -- the running list is printed
# first, and a job of mine must never be left behind (see wait.sh).
set -eu
LANE=${1:?usage: submit.sh j0|j1|j2 [--no-upload]}
NOUP=${2:-}
export PATH="$HOME/.local/bin:$PATH"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VOL=volume://vessl-storage/rfx-fdfd-gpu-v-20260915
cd "$REPO"
.venv/bin/python validation/vessl/build_gpu_lanes.py --tarball
if [ "$NOUP" != "--no-upload" ]; then
  echo "=== uploading /tmp/rfx-src.tgz -> $VOL"
  vessl storage copy-file /tmp/rfx-src.tgz "$VOL" 2>&1 | grep -v "PythonDeprecationWarning\|warnings.warn" || true
  vessl storage list-files "$VOL" | tail -4
fi
echo "=== running / pending jobs on the cluster:"
vessl run list 2>/dev/null | awk '$4=="running"||$4=="pending"||$4=="initializing"{print}' || true
echo "=== submitting $LANE"
vessl run create -f "validation/vessl/gpu_${LANE}.yaml"
