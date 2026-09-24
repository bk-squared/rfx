#!/bin/sh
# Shared worker entry point. One invocation and one JAX process per worker.
set -eu
: "${RFX_STAMP:?}" "${RFX_EXPECTED_SHA:?}" "${RFX_KIND:?}" "${RFX_NX:?}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE LANG=C.UTF-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg JAX_PLATFORMS=cuda
export MPLCONFIGDIR=/tmp/rfx-multinode-matplotlib
RFX_RUNS_ROOT=${RFX_RUNS_ROOT:-/root/workspace/claude-workspace/rfx/runs}
case "$RFX_KIND" in
  oracle|local) rank=0; world=1; coordinator=unused ;;
  two)
    rank=${RANK:?}; world=${WORLD_SIZE:?}; port=${MASTER_PORT:?}
    [ "$world" -eq 2 ]
    base=${HOSTNAME%-${rank}-*}
    [ "$base" != "$HOSTNAME" ]
    coordinator=${base}-experiment-tcp-0:${port}
    ;;
  *) echo "Invalid RFX_KIND: $RFX_KIND"; exit 1 ;;
esac
job=${RFX_KIND}-${RFX_NX}${RFX_JOB_SUFFIX:-}
echo "multinode worker start $(date -u +%Y-%m-%dT%H:%M:%SZ) rank=$rank world=$world host=$HOSTNAME coordinator=$coordinator"
out=$RFX_RUNS_ROOT/multinode-$RFX_STAMP/$job
mkdir -p "$out"
work=$(mktemp -d /tmp/rfx-multinode.XXXXXX)
collect() {
  rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$rc" > "$out/job.rank$rank.rc"
  python "$RFX_BOOTSTRAP/summarize_multinode_job.py" --output "$out" \
    --rank "$rank" --process-count "$world" --exit-code "$rc" --sha "$RFX_EXPECTED_SHA"
  summary_rc=$?
  if [ "$RFX_KIND" = two ]; then
    # Experiment output is provider-backed and shared; the legacy CLI has no
    # NFS mount flag. Preserve evidence even if /root/workspace is pod-local.
    dest=/output/multinode-$RFX_STAMP/$job
    mkdir -p "$dest"
    for file in "$out"/*; do
      [ -f "$file" ] || continue
      case "$file" in */summary.json) continue ;; esac
      cp "$file" "$dest/" || summary_rc=1
    done
    python "$RFX_BOOTSTRAP/summarize_multinode_job.py" --output "$dest" \
      --rank "$rank" --process-count "$world" --exit-code "$rc" --sha "$RFX_EXPECTED_SHA" || summary_rc=1
  fi
  chmod -R a+rX "$out" 2>/dev/null || true
  [ "$rc" -ne 0 ] || rc=$summary_rc
  exit "$rc"
}
trap collect EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# A FIFO preserves the command's return code under sh while streaming via tee.
mkfifo "$work/logpipe"
tee "$out/job.rank$rank.log" < "$work/logpipe" &
tee_pid=$!
run_worker() {
  command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
  source_repo=${RFX_SOURCE_ARCHIVE:-${RFX_SRC:-/root/workspace/claude-workspace/rfx/mirrors/multinode-probe.git}}
  git config --global --add safe.directory "$source_repo"
  ref=${RFX_REF:-green}
  resolved=$(git -C "$source_repo" rev-parse "$ref^{commit}")
  [ "$resolved" = "$RFX_EXPECTED_SHA" ] || { echo "$ref differs from pinned SHA"; return 1; }
  git clone -q --no-hardlinks "$source_repo" "$work/src"
  git -C "$work/src" checkout -q --detach "$RFX_EXPECTED_SHA"
  sha=$(git -C "$work/src" rev-parse HEAD)
  printf '%s\n' "$sha" > "$out/commit.rank$rank.txt"
  printf '%s\n' "$resolved" > "$out/tooling-commit.rank$rank.txt"
  tree=$(git -C "$work/src" rev-parse HEAD:rfx)
  baseline=$(git -C "$work/src" rev-parse "$RFX_EXPECTED_SHA":rfx)
  [ "$tree" = "$baseline" ] || { echo "rfx tree differs from the pinned commit"; return 1; }
  printf '%s\n' "$tree" > "$out/rfx-tree.rank$rank.txt"
  for file in distributed_multinode_probe.py compare_multinode_probe.py; do
    git -C "$source_repo" show "$resolved:scripts/diagnostics/$file" > "$work/src/scripts/diagnostics/$file"
  done
  cd "$work/src"
  export PYTHONPATH="$work/src"
  export RFX_TOOLING_SHA="$resolved"
  # Preserve the image's JAX/jaxlib; do not pip-install the project or upgrade JAX.
  python -m pip install -q 'numpy<2' 'scipy>=1.11,<1.15' 'h5py>=3.8,<4' 'matplotlib>=3.7,<3.10' 'pyyaml>=6,<7'
  python -m pip freeze > "$out/pip-freeze.rank$rank.txt"
  nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total --format=csv > "$out/gpu.rank$rank.csv"
  printf 'hostname=%s\nrank=%s\nworld=%s\ncoordinator=%s\nmaster_addr=%s\n' \
    "$HOSTNAME" "$rank" "$world" "$coordinator" "${MASTER_ADDR:-}" > "$out/topology.rank$rank.txt"
  # DMI product UUID identifies the host when readable; pod hostname alone is
  # not a node identity. Keep missing node evidence explicit.
  if [ -r /sys/class/dmi/id/product_uuid ]; then
    cat /sys/class/dmi/id/product_uuid > "$out/node-product-uuid.rank$rank.txt"
  else
    printf 'unavailable\n' > "$out/node-product-uuid.rank$rank.txt"
  fi
  [ "$world" -eq 1 ] || getent hosts "${coordinator%:*}"
  run_probe "$job"
  [ -n "${RFX_REF_B:-}" ] || return 0
  # A/B on the same nodes: a second solver ref, same probe arguments, tag "<job>-b".
  # Node pairs differ by ~2x in cross-node cost, so before and after must share them.
  resolved_b=$(git -C "$source_repo" rev-parse "$RFX_REF_B^{commit}")
  [ "$resolved_b" = "${RFX_EXPECTED_SHA_B:?}" ] || { echo "$RFX_REF_B differs from pinned SHA"; return 1; }
  git clone -q --no-hardlinks "$source_repo" "$work/src_b"
  git -C "$work/src_b" checkout -q --detach "$RFX_EXPECTED_SHA_B"
  git -C "$work/src_b" rev-parse HEAD > "$out/commit-b.rank$rank.txt"
  git -C "$work/src_b" rev-parse HEAD:rfx > "$out/rfx-tree-b.rank$rank.txt"
  for file in distributed_multinode_probe.py compare_multinode_probe.py; do
    git -C "$source_repo" show "$resolved_b:scripts/diagnostics/$file" > "$work/src_b/scripts/diagnostics/$file"
  done
  cd "$work/src_b"
  export PYTHONPATH="$work/src_b" RFX_TOOLING_SHA="$resolved_b"
  # Rank 0 hosts the coordinator: let it leave the first run and start the second
  # before rank 1 connects, so rank 1 never reaches the first run's coordinator.
  [ "$rank" -eq 0 ] || sleep 30
  run_probe "$job-b"
}
run_probe() {
  set -- --nx-per-rank "$RFX_NX" --ny "${RFX_NY:-116}" --nz "${RFX_NZ:-116}" \
    --steps "${RFX_STEPS:-200}" --repeats "${RFX_REPEATS:-3}" --model "${RFX_MODEL:-vacuum}" \
    --process-count "$world" --process-id "$rank" --local-device-id 0 --output "$out" --tag "$1"
  # local: one process drives every GPU the preset gives it (e.g. gpu-a6000-2).
  [ "$RFX_KIND" = local ] && set -- "$@" --local-devices
  set -- "$@" --lane "${RFX_LANE:-run}" --checkpoint-every "${RFX_CHECKPOINT_EVERY:-0}"
  [ "${RFX_GRAD:-0}" = 1 ] && set -- "$@" --grad
  [ "$world" -eq 1 ] || set -- "$@" --coordinator-address "$coordinator"
  timeout --signal=TERM --kill-after=30s 1800s python scripts/diagnostics/distributed_multinode_probe.py "$@"
}
# Run a separate shell subshell with errexit enabled; do not place it in an
# if/|| conditional (which would disable errexit inside run_worker).
set +e
(set -e; run_worker) > "$work/logpipe" 2>&1
rc=$?
wait "$tee_pid"
tee_rc=$?
set -e
[ "$tee_rc" -eq 0 ] || exit "$tee_rc"
exit "$rc"
