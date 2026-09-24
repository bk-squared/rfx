#!/usr/bin/env python3
"""Blast-radius and oracle jobs for the coax conductor-realization change.

Every group runs TWICE — once against ``origin/main`` and once against the fix —
so the report's table has a before and an after from the same command on the
same cluster, not a before quoted from memory. The oracle exists only on the fix
branch; its "before" is the mutation, which puts the sigma realization back with
every helper call left in place.

    python scripts/vessl_coax_conductor_fix/generate_jobs.py \
        --sha <pushed HEAD> --src <fix worktree> \
        --before-sha <origin/main sha> --before-src <reference worktree> \
        --out <directory>
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vessl_coax_chain_battery"))

IMAGE = "ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8"
CLUSTER = "remilab-c0"
RUNS = "/root/workspace/claude-workspace/rfx/runs"

TEMPLATE = """name: {name}
description: "{description}"
tags: [rfx, coax-conductor-fix, {tag}]
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
  XLA_PYTHON_CLIENT_MEM_FRACTION: "0.92"
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
  "$PY" -c "import rfx, os; print('rfx from', os.path.dirname(rfx.__file__))"
  {{ timeout 20000 {cmd} 2>&1; echo "rc=$?"; }} | tee "$OUT/run.log" | tail -120
  echo COAX_FIX_JOB_DONE
"""

PYTEST = ('"$PY" -m pytest -p no:cacheprovider -o addopts="" --no-header -q -rf')

# (key, preset, pytest target and marks, description). Each runs on both trees
# unless it is marked fix-only.
GROUPS = [
    ("unit-fast", "cpu-32-mem-64",
     f'{PYTEST} tests/unit/ports/test_coaxial_port.py '
     'tests/unit/ports/test_port_preflight.py '
     'tests/unit/preflight/test_preflight_advisory_emission_contract.py '
     'tests/unit/sparams/test_compute_s_matrix_dispatch.py '
     'tests/unit/sparams/test_settling_witness.py '
     'tests/unit/sparams/test_sparam_passivity_guard.py '
     'tests/unit/sparams/test_sparameter_support_contract.py '
     'tests/locks/test_sparams_split_bit_identity.py '
     'tests/locks/test_preflight_split_snapshot.py',
     "coax unit tests that need no FDTD, plus the preflight and lock snapshots"),
    ("slow-lanes", "gpu-rtx4090",
     f'{PYTEST} -m slow_physics tests/unit/sparams/test_coax_two_port_smatrix.py '
     'tests/unit/sparams/test_coaxial_line_reflection.py',
     "the two-port and one-port lanes' own slow_physics gates"),
    ("transition", "gpu-rtx4090",
     f'{PYTEST} tests/unit/sparams/test_coax_msl_transition.py '
     'tests/unit/sparams/test_coax_msl_transition_ladder_dump.py '
     'tests/unit/sparams/test_coax_msl_transition_wave_roles.py '
     'tests/unit/sparams/test_coax_msl_ladder_witnesses.py',
     "the coax-to-MSL transition, which shares the stamping helper"),
    ("autodiff", "gpu-a6000-1",
     f'{PYTEST} -m "slow_physics or highmem" '
     'tests/unit/autodiff/test_coax_two_port_ad.py '
     'tests/unit/autodiff/test_coax_end_to_end_ad.py '
     'tests/unit/autodiff/test_ad_surface_contract.py',
     "the AD gates; the one-port leg is the memory-heavy one"),
    ("crossval", "cpu-32-mem-64",
     f'{PYTEST} tests/crossval/test_coax_broad_e4_comparison_gates.py '
     'tests/crossval/test_coax_broad_e5_envelope_gates.py '
     'tests/crossval/test_coax_two_port_referee_header.py',
     "the committed coax envelope replays and the referee header"),
    ("contracts", "cpu-32-mem-64",
     f'{PYTEST} tests/contracts/test_lattice_ownership_contract.py '
     'tests/contracts/test_example_fidelity_contract.py '
     'tests/contracts/test_support_matrix_parity.py '
     'tests/contracts/test_physics_gate_reporting.py',
     "the ownership contract, the example snapshot and the support-matrix parity"),
]

FIX_ONLY = [
    ("oracle", "gpu-rtx4090",
     f'{PYTEST} -m slow_physics tests/oracle/test_coax_conductor_realization.py '
     'tests/unit/sparams/test_coax_conductor_geometry.py',
     "the new oracle (AFTER) and the no-solve geometry test"),
    ("mutation", "gpu-rtx4090",
     '"$PY" scripts/diagnostics/coax_conductor_mutation.py --out "$WORK/out" '
     '--run-id mutation',
     "the (b) falsifier: the sigma realization put back, every helper call left"),
    # The geometry test carries no marker, so the oracle job's ``-m
    # slow_physics`` deselected all 8 of its cases. It needs its own run.
    ("geometry", "cpu-32-mem-64",
     f'{PYTEST} tests/unit/sparams/test_coax_conductor_geometry.py',
     "the no-solve geometry test, which the oracle job's marker deselected"),
]

# The oracle's four comparisons as a mesh ladder. Its gate asserts on ONE mesh,
# which the v2 accuracy bar does not allow, and its board realizes 3.789
# annulus cells. These rungs are the ones NOT already measured: the diagnostic
# recorded the long board at 4, 6 and 9 cells under the same realization
# (design note, arm 4), so only the gate's own rung on the long board and the
# gate board's own refinement are missing. ``thru_long`` at rung 4 is carried
# as a cross-check that the shipped code reproduces the wrapper's arm-4 number.
LADDER = [
    # (board, rung, label, load_ohm, preset)
    ("thru_long", 3.789288121451007, "thrulong-r379", None, "gpu-rtx4090"),
    ("thru_long", 4.0, "thrulong-r4", None, "gpu-rtx4090"),
    ("thru_gate", 3.789288121451007, "thrugate-r379", None, "gpu-rtx4090"),
    ("thru_gate", 6.0, "thrugate-r6", None, "gpu-rtx4090"),
    ("thru_gate", 9.0, "thrugate-r9", None, "gpu-rtx4090"),
    ("load_gate", 6.0, "loadgate25-r6", 25.0, "gpu-rtx4090"),
    ("load_gate", 9.0, "loadgate25-r9", 25.0, "gpu-rtx4090"),
    ("load_gate", 6.0, "loadgate100-r6", 100.0, "gpu-rtx4090"),
    ("load_gate", 9.0, "loadgate100-r9", 100.0, "gpu-rtx4090"),
]


def ladder_groups():
    for board, rung, label, load, preset in LADDER:
        cmd = ('"$PY" scripts/diagnostics/coax_conductor_oracle_ladder.py '
               f'--board {board} --rung {rung!r} --out "$WORK/out" '
               f'--run-id {label}')
        if load is not None:
            cmd += f" --load-ohm {load!r}"
        yield (f"ladder-{label}", preset, cmd,
               f"the oracle's comparisons on {board} at {rung:g} annulus cells")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--before-sha", required=True)
    ap.add_argument("--before-src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None,
                    help="emit only the jobs whose name contains this text; "
                         "the collision guard still runs over ALL names")
    args = ap.parse_args()

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    cases = [("after", args.sha, args.src,
              GROUPS + FIX_ONLY + list(ladder_groups())),
             ("before", args.before_sha, args.before_src, GROUPS)]
    all_names = []
    for side, sha, src, groups in cases:
        for key, preset, cmd, desc in groups:
            name = f"coaxfix-{side}-{key}"
            all_names.append(name)
            if args.only and args.only not in name:
                continue
            text = TEMPLATE.format(
                name=f"rfx-{name}", description=f"{desc} ({side}).", tag=name,
                cluster=CLUSTER, preset=preset, image=IMAGE, sha=sha, src=src,
                runs=RUNS, prefix=name, cmd=cmd)
            path = dest / f"{name}.yaml"
            path.write_text(text)
            written.append((path, name))

    # Over ALL names, not just the emitted ones: a run directory is found by
    # ``-name "<prefix>*"``, so a name that prefixes another collides whether or
    # not this invocation happened to emit both.
    bad = [(a, b) for a, b in itertools.permutations(all_names, 2)
           if b.startswith(a)]
    if bad:
        raise SystemExit(f"run prefixes collide: {bad}")
    for path, prefix in written:
        print(f"sh scripts/vessl_submit.sh {path} {prefix}")
    print(f"\n{len(written)} job specifications in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
