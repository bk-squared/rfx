#!/usr/bin/env python3
"""Write one VESSL job specification per (arm, rung, record) of the coaxial
open-end absorber test (``scripts/diagnostics/coax_open_absorber_diagnostic.py``).

Same job body as the battery's own campaign (``generate_jobs.py``: pinned
commit, clean-tree guard, node-local copy, EXIT-trap collection), on the CPU
preset: the CUDA jax wheel is swapped for the CPU one, ``JAX_PLATFORMS=cpu`` is
set, and the time limit is set per job; nothing else changes.

    python scripts/vessl_coax_chain_battery/generate_open_absorber_jobs.py \
        --sha <pushed HEAD> --src <worktree path> --out <directory>

then submit each file with ``sh scripts/vessl_submit.sh <yaml> <prefix>``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnostics"))

from generate_jobs import CLUSTER, IMAGE, RUNS, TEMPLATE  # noqa: E402
# The pre-declared matrix and the driver path, from the diagnostic itself.
from coax_open_absorber_diagnostic import CAMPAIGN, DRIVER  # noqa: E402

PRESET = "cpu-32-mem-64"
GPU_WHEEL = '"jax[cuda12]==0.6.2"'
CPU_WHEEL = '"jax==0.6.2"'
BATTERY_TIMEOUT = "timeout 20000 "
ENV_ANCHOR = '  MPLBACKEND: "Agg"\n'
CPU_ENV = ENV_ANCHOR + '  JAX_PLATFORMS: "cpu"\n'


def cpu_template(timeout_s: int) -> str:
    for needle in (GPU_WHEEL, BATTERY_TIMEOUT, ENV_ANCHOR):
        if TEMPLATE.count(needle) != 1:
            raise RuntimeError(f"the battery template no longer holds {needle!r} exactly once")
    return (TEMPLATE.replace(GPU_WHEEL, CPU_WHEEL)
            .replace(BATTERY_TIMEOUT, f"timeout {int(timeout_s)} ")
            .replace(ENV_ANCHOR, CPU_ENV))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True, help="the pushed commit every job must see")
    ap.add_argument("--src", required=True, help="the worktree the jobs read")
    ap.add_argument("--out", required=True, help="directory to write the YAMLs into")
    args = ap.parse_args()

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for arm, rung, units in CAMPAIGN:
        key = f"open-absorber-{arm}-r{rung}-u{units:g}"
        prefix = f"coax-{key}"
        timeout_s = 40000 if units >= 48.0 else 20000
        text = cpu_template(timeout_s).format(
            name=f"rfx-{prefix}",
            description=(f"Coaxial open end, arm {arm} at {rung} annulus cells, "
                         f"{units:g} record units: pre-declared absorber test."),
            tag=key, cluster=CLUSTER, preset=PRESET, image=IMAGE, sha=args.sha,
            src=args.src, runs=RUNS, prefix=prefix,
            command=(f'{DRIVER} --arm {arm} --rung {rung} --record-units {units:g} '
                     f'--out "$WORK/out" --run-id "{prefix}"'))
        path = dest / f"{key}.yaml"
        path.write_text(text)
        written.append((path, prefix))
    for path, prefix in written:
        print(f"sh scripts/vessl_submit.sh {path} {prefix}")
    print(f"\n{len(written)} job specifications in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
