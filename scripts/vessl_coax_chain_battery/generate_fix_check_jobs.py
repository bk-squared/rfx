#!/usr/bin/env python3
"""Write the CPU job specifications that test issue 1218's fix.

Four jobs, one stage each, on the CPU preset (pinned commit, clean-tree guard,
node-local copy, EXIT-trap collection, the CPU JAX 0.6.2 wheel):

* ``open-settles-as-fixed`` — ``tests/oracle/test_coax_open_end_settles.py`` as
  shipped, pinned to four cores so its wall time is the one the weekly lane's
  runner sees;
* ``open-settles-forced-z`` — the same test with both coax lanes forced back to
  ``cpml_axes="z"`` inside the runner call, every helper call kept: the
  mutation that revives the closed can. The test has to go red;
* ``coax-fast`` / ``coax-slow`` — every coax test file (and the contracts on
  the fast one), in the default selection and in the slow and slow_physics
  one. The job image's git (2.25) has no ``git init -b``, which the
  worktree-pruning, changelog-fragment and two CI-workflow contract tests use
  to build their scratch repositories; those are left to the pull request's
  own CI;
* ``drift-lock`` — the coax battery's drift lock on its own, pinned to four
  cores like the open-end check, so its wall time is the weekly runner's.

    python scripts/vessl_coax_chain_battery/generate_fix_check_jobs.py \\
        --sha <commit> --src <run tree> --out <directory>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_jobs import CLUSTER, IMAGE, RUNS  # noqa: E402

PRESET = "cpu-32-mem-64"

TEMPLATE = """name: {name}
description: "{description}"
tags: [rfx, coax-chain-battery, issue-1218-fix, {tag}]
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
  JAX_PLATFORMS: "cpu"
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
  WORK=/root/work/{prefix}
  rm -rf "$WORK"; mkdir -p "$WORK"; cp -a "$SRC/." "$WORK/"; cd "$WORK"
  trap 'rc=$?; echo "$rc" > "$OUT/job.rc"; chmod -R a+rX "$OUT" 2>/dev/null || true' EXIT
  "$PY" -m pip uninstall -y -q jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null || true
  "$PY" -m pip install -q "jax[cpu]==0.6.2" "numpy>=2" "scipy>=1.11" "h5py>=3.8" "matplotlib>=3.7" pytest pytest-xdist pytest-timeout pytest-split optax
  export PYTHONPATH="$WORK"
  git config --global --add safe.directory "*"
  WSHA=$(git -C "$WORK" rev-parse HEAD)
  test "$WSHA" = "$RFX_SHA" || {{ echo "FATAL: the work copy resolves $WSHA"; exit 3; }}
  "$PY" -c "import rfx, os, jax; print('rfx from', os.path.dirname(rfx.__file__)); print('jax', jax.__version__, jax.devices())"
{mutation}  {{ timeout 20000 {taskset}"$PY" -m pytest -p no:cacheprovider -o addopts="" --timeout=1800 --timeout-method=thread -rfE --durations=0 --junitxml="$OUT/junit.xml" {pytest_args} 2>&1; echo "rc=$?"; }} | tee "$OUT/pytest.log" | tail -120
  echo COAX_FIX_CHECK_DONE
"""

# Both lanes' runner calls read ``cpml_axes=cpml_axes,``; the mutation writes
# "z" there and leaves the refusal helper and every other call in place.
MUTATION = ('''  "$PY" -c "import pathlib; p = pathlib.Path('rfx/sparams/coax.py'); '''
            '''s = p.read_text(); n = s.count('cpml_axes=cpml_axes,'); assert n == 2, n; '''
            '''p.write_text(s.replace('cpml_axes=cpml_axes,', 'cpml_axes=\\"z\\",')); '''
            '''print('forced cpml_axes=z in', n, 'runner calls')"\n'''
            '''  grep -n 'cpml_axes="z",' rfx/sparams/coax.py\n''')

COAX_FILES = (
    "tests/unit/sparams/test_coax_lanes_absorb_on_every_axis.py "
    "tests/unit/sparams/test_coax_conductor_geometry.py "
    "tests/unit/sparams/test_coax_two_port_smatrix.py "
    "tests/unit/sparams/test_coaxial_line_reflection.py "
    "tests/unit/sparams/test_coax_msl_transition.py "
    "tests/unit/sparams/test_settling_witness.py "
    "tests/unit/sparams/test_compute_s_matrix_dispatch.py "
    "tests/unit/sparams/test_sparam_passivity_guard.py "
    "tests/unit/preflight/test_removed_coaxial_s_matrix_lane.py "
    "tests/unit/preflight/test_preflight_advisory_emission_contract.py "
    "tests/unit/nonuniform/test_dz_only_dispatch_contract.py "
    "tests/unit/autodiff/test_coax_two_port_ad.py "
    "tests/unit/autodiff/test_coax_end_to_end_ad.py "
    "tests/unit/autodiff/test_ad_surface_contract.py "
    "tests/oracle/test_coax_conductor_realization.py "
    "tests/oracle/test_coax_chain_battery.py "
    "tests/oracle/test_coax_open_end_settles.py "
    "tests/locks/test_coax_chain_battery_drift.py "
    "tests/crossval/test_coax_broad_e5_envelope_gates.py "
    "tests/crossval/test_coax_broad_e4_comparison_gates.py"
)

# Contract tests that build scratch repositories with ``git init -b``, which the
# job image's git 2.25 does not have; the pull request's CI runs them.
NEEDS_NEWER_GIT = (
    "--ignore=tests/contracts/test_prune_worktrees.py "
    "--ignore=tests/contracts/test_changelog_fragments.py "
    "--deselect tests/contracts/test_ci_workflows_contract.py::"
    "test_a_rename_out_of_the_package_is_seen_as_a_code_change "
    "--deselect tests/contracts/test_ci_workflows_contract.py::"
    "test_a_path_containing_a_quote_survives_the_pipe"
)

JOBS = (
    dict(key="open-settles-as-fixed", taskset="taskset -c 0-3 ", mutation="",
         pytest_args='-m slow_physics -s tests/oracle/test_coax_open_end_settles.py',
         description=("Issue 1218 fix: the coax open end on the cheapest board where the "
                      "shipped lane fails, as shipped, on four pinned cores.")),
    dict(key="open-settles-forced-z", taskset="taskset -c 0-3 ", mutation=MUTATION,
         pytest_args='-m slow_physics -s tests/oracle/test_coax_open_end_settles.py',
         description=("Issue 1218 fix, mutation: both coax lanes forced back to "
                      "cpml_axes=z in the runner call, helper calls kept; the open-end "
                      "check has to go red.")),
    dict(key="coax-fast", taskset="", mutation="",
         pytest_args=(f'-n 8 -m "not gpu and not slow and not slow_physics" {COAX_FILES} '
                      f'tests/contracts {NEEDS_NEWER_GIT}'),
         description="Issue 1218 fix: every coax test file and the contracts, default selection."),
    dict(key="coax-slow", taskset="", mutation="",
         pytest_args=f'-n 8 -m "(slow or slow_physics) and not gpu and not highmem" {COAX_FILES}',
         description="Issue 1218 fix: every coax test file, slow and slow_physics selection."),
    dict(key="drift-lock", taskset="taskset -c 0-3 ", mutation="",
         pytest_args='-m slow_physics -s tests/locks/test_coax_chain_battery_drift.py',
         description=("Issue 1218 fix: the coax battery's drift lock against the re-measured "
                      "records, on four pinned cores.")),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        key = f"fix-check-{job['key']}"
        prefix = f"coax-{key}"
        text = TEMPLATE.format(
            name=f"rfx-{prefix}", description=job["description"], tag=key,
            cluster=CLUSTER, preset=PRESET, image=IMAGE, sha=args.sha, src=args.src,
            runs=RUNS, prefix=prefix, mutation=job["mutation"], taskset=job["taskset"],
            pytest_args=job["pytest_args"])
        path = dest / f"{key}.yaml"
        path.write_text(text)
        print(f"sh scripts/vessl_submit.sh {path} {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
