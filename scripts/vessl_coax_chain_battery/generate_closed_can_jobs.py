#!/usr/bin/env python3
"""Write one VESSL job specification per arm of the closed-can test
(``scripts/diagnostics/coax_open_closed_can_arms.py``), plus one smoke job.

Same job body as the battery's own campaign (``generate_jobs.py``: pinned
commit, clean-tree guard, node-local copy, EXIT-trap collection, JAX 0.6.2
installed before the first import), on the VRAM-band preset ``gpu-8gb``: the
GPU model does not enter any number here, and a 9-annulus-cell line needs
about 1 GiB of device memory. A job may run more than one command (the O0 job
also computes W1); each command's exit code is logged and the next still runs.

    python scripts/vessl_coax_chain_battery/generate_closed_can_jobs.py \\
        --sha <pushed HEAD> --src <run tree path> --out <directory>

then submit each file with ``sh scripts/vessl_submit.sh <yaml> <prefix>``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnostics"))

from generate_jobs import CLUSTER, IMAGE, RUNS  # noqa: E402
from coax_open_closed_can_arms import ARMS, DRIVER  # noqa: E402

PRESET = "gpu-8gb"
TIMEOUT_S = 20000

TEMPLATE = """name: {name}
description: "{description}"
tags: [rfx, coax-chain-battery, closed-can, {tag}]
resources:
  cluster: {cluster}
  preset: {preset}
image: {image}
env:
  DEBIAN_FRONTEND: noninteractive
  PYTHONUNBUFFERED: "1"
  OMP_NUM_THREADS: "4"
  HDF5_USE_FILE_LOCKING: "FALSE"
  LANG: "C.UTF-8"
  MPLBACKEND: "Agg"
  RFX_SHA: "{sha}"
mount:
  /root/workspace/: volume://remilab-fs/personal-workspaces/
run: |-
  set -eu
  PY=python
  command -v "$PY" >/dev/null 2>&1 || PY=/opt/conda/bin/python
  command -v git >/dev/null 2>&1 || {{ apt-get update -qq && apt-get install -y -qq git; }}
  SRC={src}
  git config --global --add safe.directory "$SRC"
  git config --global --add safe.directory "$(git -C "$SRC" rev-parse --absolute-git-dir)"
  SHA=$(git -C "$SRC" rev-parse HEAD)
  test "$SHA" = "$RFX_SHA" || {{ echo "FATAL: worktree moved: $SHA"; exit 3; }}
  test -z "$(git -C "$SRC" status --porcelain)" || {{ echo "FATAL: dirty worktree"; exit 3; }}
  RUNS={runs}
  OUT=$RUNS/{prefix}-$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$OUT"
  echo "$SHA" > "$OUT/commit.txt"
  echo "$OUT" > "$RUNS/{prefix}.latest"
  WORK=/root/work/{prefix}
  rm -rf "$WORK"; mkdir -p "$WORK"; cp -a "$SRC/." "$WORK/"; cd "$WORK"
  mkdir -p "$WORK/out"
  trap 'rc=$?; echo "$rc" > "$OUT/job.rc"; cp -a "$WORK/out/." "$OUT/" 2>/dev/null || true; chmod -R a+rX "$OUT" 2>/dev/null || true' EXIT
  "$PY" -m pip install -q "jax[cuda12]==0.6.2" "numpy>=2" "scipy>=1.11" "h5py>=3.8" "matplotlib>=3.7" pytest pytest-split optax
  export PYTHONPATH="$WORK"
  git config --global --add safe.directory "*"
  WSHA=$(git -C "$WORK" rev-parse HEAD)
  test "$WSHA" = "$RFX_SHA" || {{ echo "FATAL: the work copy resolves $WSHA"; exit 3; }}
  "$PY" -c "import rfx, os, jax; print('rfx from', os.path.dirname(rfx.__file__)); print('jax', jax.__version__, jax.devices())"
  "$PY" -c "import jax, sys; sys.exit(0 if jax.default_backend() == 'gpu' else 4)" || {{ echo "FATAL: jax has no GPU backend on this node"; exit 4; }}
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
{commands}
  echo COAX_CLOSED_CAN_JOB_DONE
"""

COMMAND = ('  {{ timeout {timeout} "$PY" {driver} {cli} --out "$WORK/out" --run-id "{prefix}" '
           '2>&1; echo "rc=$?"; }} | tee -a "$OUT/run.log" | tail -80')


def jobs() -> list[dict]:
    out = [dict(
        key="smoke",
        clis=["--smoke --arm ALL", "--smoke --w1",
              '--smoke --assemble --artifact-out "$WORK/out/smoke_open_closed_can_arms.json"'],
        description=("Closed-can test, smoke: every arm, W1 and the assembler at a one- and "
                     "two-unit record, to exercise the code paths before the campaign."),
    )]
    for arm, spec in ARMS.items():
        clis = [f"--arm {arm}"]
        if arm == "O0":
            clis.append("--w1")
        units = " and ".join(f"{u:g}" for u in spec["units"])
        out.append(dict(
            key=arm, clis=clis,
            description=(f"Closed-can test, arm {arm} at 9 annulus cells, {units} record "
                         f"units: {spec['what']}." + (" Also W1." if arm == "O0" else "")),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True, help="the pushed commit every job must see")
    ap.add_argument("--src", required=True, help="the run tree the jobs read")
    ap.add_argument("--out", required=True, help="directory to write the YAMLs into")
    args = ap.parse_args()

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for job in jobs():
        key = f"closed-can-{job['key']}"
        prefix = f"coax-{key}"
        commands = "\n".join(COMMAND.format(timeout=TIMEOUT_S, driver=DRIVER, cli=cli,
                                            prefix=prefix) for cli in job["clis"])
        text = TEMPLATE.format(
            name=f"rfx-{prefix}", description=job["description"].replace('"', "'"),
            tag=key, cluster=CLUSTER, preset=PRESET, image=IMAGE, sha=args.sha,
            src=args.src, runs=RUNS, prefix=prefix, commands=commands)
        path = dest / f"{key}.yaml"
        path.write_text(text)
        written.append((path, prefix))
    for path, prefix in written:
        print(f"sh scripts/vessl_submit.sh {path} {prefix}")
    print(f"\n{len(written)} job specifications in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
