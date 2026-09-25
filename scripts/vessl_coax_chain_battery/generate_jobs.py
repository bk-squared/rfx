#!/usr/bin/env python3
"""Write one VESSL job specification per (DUT, rung, stage) of the coax battery.

One job per measurement, submitted together; the scheduler queues them. Every
job pins the commit it must run (``RFX_SHA``, no fallback), refuses a worktree
that moved or is dirty, copies the tree to node-local storage, writes its stage
JSON there and copies it back to the run directory from an EXIT trap, so a job
that dies still leaves what it had.

Regenerate the campaign with::

    python scripts/vessl_coax_chain_battery/generate_jobs.py \
        --sha <pushed HEAD> --src <worktree path> --out <directory>

then submit each file with ``sh scripts/vessl_submit.sh <yaml> <prefix>``.
The committed copies carry the sha the campaign actually ran at.
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMAGE = "ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8"
CLUSTER = "remilab-c0"
DRIVER = "scripts/diagnostics/coax_chain_battery_measure.py"
RUNS = "/root/workspace/claude-workspace/rfx/runs"

RUNGS = (4, 6, 9)
CLAIMS_RUNG = 9
TWO_PORT_DUTS = ("bead", "thru")
ONE_PORT_DUTS = ("short", "open", "r25", "r100")
DUTS = TWO_PORT_DUTS + ONE_PORT_DUTS

# The GPU jobs' preset: a VRAM band, chosen at submission from the live
# occupancy (gpu-8gb, or gpu-24gb when gpu-8gb is full and gpu-24gb has free
# nodes). Every one of these jobs fits 8 GB; none depends on the GPU model. The
# two AD stages keep the 48 GB model preset: their reverse-mode tape does not
# fit 24 GB. The replay and lint jobs are CPU jobs.
GPU_BANDS = ("gpu-8gb", "gpu-24gb")
GPU_BAND = GPU_BANDS[0]

RECORD_UNITS = 12.0
DOUBLE_RECORD_UNITS = 24.0
# The record length that puts the coarsest rung's two-port run at exactly the
# 6000 steps the committed gate uses (measured with the driver's own
# record_steps at rung 4: 13.6 -> 5900, 13.7 -> 6000).
UNITS_FOR_6000_STEPS = 13.7

TEMPLATE = """name: {name}
description: "{description}"
tags: [rfx, coax-chain-battery, {tag}]
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
  "$PY" -c "import rfx, os; print('rfx from', os.path.dirname(rfx.__file__))"
  {{ timeout 20000 "$PY" {command} 2>&1; echo "rc=$?"; }} | tee "$OUT/run.log" | tail -80
  echo COAX_BATTERY_STAGE_DONE
"""


def jobs() -> list[dict]:
    out: list[dict] = []
    out.append(dict(
        key="pilot", preset=GPU_BAND,
        cli=f"--stage pilot --rung 4 --record-units {RECORD_UNITS}",
        description=("Coax chain battery pilot: record-length ladder, the wider drive, "
                     "the bead-mask control and the AD boards' own settling, at the "
                     "coarsest rung."),
    ))
    for dut in DUTS:
        for rung in RUNGS:
            out.append(dict(
                key=f"solve-{dut}-r{rung}", preset=GPU_BAND,
                cli=(f"--stage solve --dut {dut} --rung {rung} "
                     f"--record-units {RECORD_UNITS}"),
                description=(f"Coax chain battery: {dut} at {rung} annulus cells, "
                             f"{RECORD_UNITS:g} record units."),
            ))
    for dut in DUTS:
        out.append(dict(
            # "double" before the rung, not after: vessl_submit.sh finds a run
            # directory by `-name "<prefix>*"`, so a prefix that EXTENDS another
            # one would have the shorter job record the longer job's id.
            key=f"solve-{dut}-double-r{CLAIMS_RUNG}", preset=GPU_BAND,
            cli=(f"--stage solve --dut {dut} --rung {CLAIMS_RUNG} "
                 f"--record-units {DOUBLE_RECORD_UNITS} --tag double"),
            description=(f"Coax chain battery: {dut} at the claims rung with the record "
                         f"doubled — the contract's substitute for the energy witness "
                         f"this lane does not emit on the eps_scale path."),
        ))
    out.append(dict(
        key="identity", preset=GPU_BAND,
        cli=f"--stage identity --rung 4 --record-units {RECORD_UNITS}",
        description=("Coax chain battery: forward identity, the untraced numpy path "
                     "against the no-op jnp path and the bead in two containers."),
    ))
    # The bead arm alone: re-run when the bead became a whole number of cells
    # (the pre-declaration's second addendum, item 8); the thru arm has no bead.
    # Keyed "bead-identity", not "identity-bead": vessl_submit.sh matches run
    # directories by prefix, and "identity" would match the longer key's.
    out.append(dict(
        key="bead-identity", preset=GPU_BAND,
        cli=f"--stage identity --rung 4 --record-units {RECORD_UNITS} --identity-arms bead",
        description=("Coax chain battery: forward identity, the bead arm alone (the same "
                     "bead in a numpy and a jnp container)."),
    ))
    out.append(dict(
        key="adfd-twoport", preset="gpu-a6000-1",
        cli="--stage adfd-twoport",
        description=("Coax chain battery: reverse-mode AD against a float64-loss central "
                     "finite difference on the two-port bead. 48 GB preset — this lane "
                     "has no checkpoint_segments, so the tape is O(n_steps)."),
    ))
    out.append(dict(
        key="adfd-oneport", preset="gpu-a6000-1",
        cli="--stage adfd-oneport",
        description=("Coax chain battery: the same AD/FD comparison on the one-port "
                     "short. 48 GB preset."),
    ))
    # Two controls that separate the record length from the cell size, and
    # this tree from the committed gate. Both were hand-written YAMLs on the
    # first round; they are generated here so the campaign regenerates whole.
    out.append(dict(
        key="control-steps6000", preset=GPU_BAND,
        cli=(f"--stage solve --dut thru --rung {RUNGS[0]} "
             f"--record-units {UNITS_FOR_6000_STEPS} --tag steps6000"),
        description=("Control: the thru at the coarsest rung with n_steps = 6000, the "
                     "step count the committed two-port gate uses, so the record length "
                     "can be told apart from the cell size."),
    ))
    out.append(dict(
        key="control-committed", preset=GPU_BAND, pytest=True,
        cli=('-m pytest -p no:cacheprovider -o addopts="" -m slow_physics -s '
             "tests/unit/sparams/test_coax_two_port_smatrix.py "
             "-k test_matched_through_line_transmits_reciprocally -q"),
        description=("Control: the repository's own committed thru gate on this tree, "
                     "unchanged, so the battery's thru can be read beside it."),
    ))
    out.append(dict(
        key="replay", preset="cpu-32-mem-64", pytest=True,
        cli=('-m pytest -p no:cacheprovider -o addopts="" '
             "tests/oracle/test_coax_chain_battery.py "
             "tests/contracts/test_evidence_numeric_provenance.py -q"),
        description=("The assembled fixture replayed: every stored number re-derived "
                     "from the stored S and compared against the bar, plus the evidence "
                     "provenance contract. Arithmetic only, no FDTD."),
    ))
    out.append(dict(
        key="ruff", preset="cpu-32-mem-64", pytest=True, extra_pip="ruff",
        cli=("-m ruff check rfx/ tests/ validation/ scripts/ci/ scripts/dev/ "
             "scripts/changelog/ scripts/diagnostics/coax_chain_battery_measure.py "
             "scripts/diagnostics/coax_open_absorber_diagnostic.py "
             "scripts/vessl_coax_chain_battery/ "
             "--select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402"),
        description=("ruff on the CI lint scope (scripts/ci/lint.sh) plus this "
                     "battery's driver, diagnostic and job generators."),
    ))
    out.append(dict(
        key="plane", preset=GPU_BAND,
        cli=f"--stage plane --rung 6 --record-units {RECORD_UNITS}",
        description=("Coax chain battery: reference-plane invariance — the bead "
                     "translated 4 cells along the line, the grid untouched."),
    ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True, help="the pushed commit every job must see")
    ap.add_argument("--src", required=True, help="the worktree the jobs read")
    ap.add_argument("--out", required=True, help="directory to write the YAMLs into")
    ap.add_argument("--only", default=None,
                    help="comma-separated job keys to write; every job when omitted")
    ap.add_argument("--gpu-band", default=GPU_BANDS[0], choices=GPU_BANDS,
                    help="the VRAM-band preset for the GPU jobs, read off the live "
                         "occupancy right before submission")
    args = ap.parse_args()
    global GPU_BAND
    GPU_BAND = args.gpu_band
    only = None if args.only is None else set(args.only.split(","))
    known = {job["key"] for job in jobs()}
    if only is not None and not only <= known:
        ap.error(f"unknown job key(s): {sorted(only - known)}")

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for job in jobs():
        if only is not None and job["key"] not in only:
            continue
        prefix = f"coax-battery-{job['key']}"
        text = TEMPLATE.format(
            name=f"rfx-{prefix}", description=job["description"], tag=job["key"],
            cluster=CLUSTER, preset=job["preset"], image=IMAGE, sha=args.sha,
            src=args.src, runs=RUNS, prefix=prefix,
            command=(job["cli"] if job.get("pytest") else
                     f'{DRIVER} {job["cli"]} --out "$WORK/out" --run-id "{prefix}"'))
        path = dest / f"{job['key']}.yaml"
        if job.get("extra_pip"):
            # One more package on this job's install line only (ruff for the
            # lint job); the anchor is the template's last package.
            anchor = " pytest pytest-split optax\n"
            if text.count(anchor) != 1:
                raise RuntimeError("the job template's install line moved")
            text = text.replace(anchor, f" pytest pytest-split optax {job['extra_pip']}\n")
        path.write_text(text)
        written.append((path, prefix))
    for path, prefix in written:
        print(f"sh scripts/vessl_submit.sh {path} {prefix}")
    print(f"\n{len(written)} job specifications in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
