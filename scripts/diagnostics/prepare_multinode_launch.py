#!/usr/bin/env python3
"""Render three oracle Run YAMLs and exact two-worker Experiment CLI commands.

This only writes local specifications; it never submits a job. The lead must
commit the tooling and push it to the bare mirror's green ref first. Experiments
receive a self-contained upload of that mirror because CLI 0.1.188 has no NFS
mount option. Workers trap-copy outputs to provider-backed /output for download.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distributed_multinode_probe import timing_plan_error  # noqa: E402


def command(args):
    return shlex.join([str(a) for a in args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tooling-sha", required=True)
    parser.add_argument("--mirror", type=Path, required=True,
                        help="Bare mirror as visible on the submitter (Mac or NFS-mounted pod)")
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="runs directory as visible on the submitter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--oracle", action="append", metavar="PRESET:NX",
                        help="one-GPU plain-path job (repeatable); default gpu-rtx4090:200/400/800")
    parser.add_argument("--local", action="append", metavar="PRESET:NX_PER_DEVICE",
                        help="one process driving every GPU of a multi-GPU preset (repeatable)")
    parser.add_argument("--model", choices=("vacuum", "loaded"), default="vacuum")
    parser.add_argument("--ref", default="green",
                        help="mirror ref the jobs pin (distinct refs let queued jobs of different commits coexist)")
    parser.add_argument("--lane", choices=("run", "forward"), default="run",
                        help="forward: time forward(distributed=True) with an x-sharded eps design")
    parser.add_argument("--grad", action="store_true", help="forward lane: also time jax.grad")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="forward lane: segmented remat length (0 = none); every timed step "
                             "count must be a multiple of it larger than it")
    parser.add_argument("--steps-short", type=int, default=0,
                        help="both lanes: each repeat also times this many steps first; the paired "
                             "difference gives the per-step cost without compilation (0 = off)")
    parser.add_argument("--pip-jax", default="",
                        help="pip requirement installed over the image's JAX, e.g. 'jax[cuda12]==0.6.2' "
                             "(default: keep the image's JAX)")
    parser.add_argument("--ref-b", help="second solver ref run after --ref in the same job (A/B on the same nodes)")
    parser.add_argument("--tooling-sha-b", help="full commit SHA that --ref-b must resolve to")
    parser.add_argument("--two", action="append", type=int, metavar="NX_PER_RANK",
                        help="two-worker job (repeatable); default 100/200/400")
    parser.add_argument("--two-preset", default="gpu-rtx4090")
    parser.add_argument("--ny", type=int, default=116)
    parser.add_argument("--nz", type=int, default=116)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mem-fraction", help="XLA_PYTHON_CLIENT_MEM_FRACTION for every job")
    parser.add_argument("--job-suffix", default="",
                        help="appended to every job name (e.g. -cap); oracle names also get the preset")
    args = parser.parse_args()
    explicit_oracles = args.oracle is not None
    oracles = []
    for item in (args.oracle or ["gpu-rtx4090:200", "gpu-rtx4090:400", "gpu-rtx4090:800"]):
        preset, _, nx_text = item.rpartition(":")
        if not preset or not nx_text.isdigit():
            parser.error(f"--oracle must be PRESET:NX, got {item!r}")
        oracles.append((preset, int(nx_text)))
    slabs = args.two or [100, 200, 400]
    if args.job_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.job_suffix):
        parser.error("job-suffix must contain only letters, digits, underscores, hyphens")
    shape_env = {"RFX_NY": str(args.ny), "RFX_NZ": str(args.nz), "RFX_STEPS": str(args.steps),
                 "RFX_REPEATS": str(args.repeats), "RFX_MODEL": args.model, "RFX_REF": args.ref}
    if args.mem_fraction:
        shape_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.mem_fraction
    if args.grad and args.lane != "forward":
        parser.error("--grad needs --lane forward")
    # The probe refuses these at argument parsing too; refusing here keeps a
    # launch from reaching the queue only to stop on every worker.
    problem = timing_plan_error(args.lane, args.steps, args.steps_short, args.checkpoint_every)
    if problem:
        parser.error(problem)
    shape_env.update(RFX_LANE=args.lane, RFX_GRAD="1" if args.grad else "0",
                     RFX_CHECKPOINT_EVERY=str(args.checkpoint_every),
                     RFX_STEPS_SHORT=str(args.steps_short))
    if args.pip_jax:
        shape_env["RFX_PIP_JAX"] = args.pip_jax
    if bool(args.ref_b) != bool(args.tooling_sha_b):
        parser.error("--ref-b and --tooling-sha-b go together")
    if args.ref_b:
        if not re.fullmatch(r"[0-9a-f]{40}", args.tooling_sha_b):
            parser.error("tooling-sha-b must be a full 40-character commit SHA")
        shape_env.update(RFX_REF_B=args.ref_b, RFX_EXPECTED_SHA_B=args.tooling_sha_b)
    if not re.fullmatch(r"[0-9a-f]{40}", args.tooling_sha):
        parser.error("tooling-sha must be a full 40-character commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.stamp):
        parser.error("stamp must contain only letters, digits, underscores, hyphens")
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifacts = args.artifact_root.resolve() / f"multinode-{args.stamp}"
    template = yaml.safe_load((root / "scripts/vessl_multinode_oracle.yaml").read_text())
    yaml.add_representer(str, lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None))
    # Directories made from the Mac are owned by its user; the pods reach the
    # share as nobody, so open them or the oracle's own mkdir is refused.
    oracle_commands = ["#!/bin/sh", "set -eu", command(["mkdir", "-p", artifacts]),
                       command(["chmod", "777", artifacts])]
    locals_ = []
    for item in (args.local or []):
        preset, _, nx_text = item.rpartition(":")
        if not preset or not nx_text.isdigit():
            parser.error(f"--local must be PRESET:NX_PER_DEVICE, got {item!r}")
        locals_.append(("local", preset, int(nx_text)))
    one_process_jobs = ([] if args.local and args.oracle is None else
                        [("oracle", preset, nx) for preset, nx in oracles]) + locals_
    pids = []
    for index, (kind, preset, nx) in enumerate(one_process_jobs):
        short = preset.removeprefix("gpu-")
        suffix = args.job_suffix + (f"-{short}" if explicit_oracles or kind == "local" else "")
        job = f"{kind}-{nx}{suffix}"
        template["name"] = f"rfx-multinode-{job}"
        template["resources"]["preset"] = preset
        template["env"].update(RFX_STAMP=args.stamp, RFX_NX=str(nx), RFX_EXPECTED_SHA=args.tooling_sha,
                               RFX_KIND=kind, RFX_JOB_SUFFIX=suffix, **shape_env)
        path = out / f"{job}.yaml"
        path.write_text(yaml.dump(template, sort_keys=False))
        oracle_commands.append(command(["sh", root / "scripts/vessl_submit.sh", path,
                                        job, artifacts]) + " &")
        oracle_commands.append(f"p{index}=$!")
        pids.append(f'"$p{index}"')
    oracle_commands.extend(["rc=0", f"for pid in {' '.join(pids)}; do",
                            '  wait "$pid" || rc=1', "done", 'exit "$rc"'])
    (out / "submit-oracles.sh").write_text("\n".join(oracle_commands) + "\n")

    payload = out / "payload"
    preparation = ["#!/bin/sh", "set -eu",
                   command(["test", "!", "-e", payload]), command(["mkdir", "-p", payload]),
                   command(["git", "clone", "--mirror", "--no-hardlinks", args.mirror.resolve(),
                            payload / "source.git"]),
                   'test "$(' + command(["git", "-C", payload / "source.git", "rev-parse", f"{args.ref}^{{commit}}"])
                   + ')" = ' + shlex.quote(args.tooling_sha)]
    if args.ref_b:
        preparation.append('test "$(' + command(["git", "-C", payload / "source.git", "rev-parse",
                                                 f"{args.ref_b}^{{commit}}"]) + ')" = '
                           + shlex.quote(args.tooling_sha_b))
    for name in ("multinode_probe_job.sh", "summarize_multinode_job.py"):
        preparation.append(command(["git", "-C", payload / "source.git", "show",
                                    f"{args.tooling_sha}:scripts/diagnostics/{name}"])
                           + " > " + shlex.quote(str(payload / name)))
    preparation.append(command(["git", "-C", payload / "source.git", "cat-file", "-e", f"{args.tooling_sha}^{{commit}}"]))
    preparation.append("find " + shlex.quote(str(payload)) + " -name '._*' -delete")
    preparation.append("test \"$(find " + shlex.quote(str(payload)) + " -name '._*' | wc -l | tr -d ' ')\" = 0")
    (out / "prepare-upload.sh").write_text("\n".join(preparation) + "\n")
    for slab in slabs:
        job = f"two-{slab}{args.job_suffix}"
        env = {"RFX_STAMP": args.stamp, "RFX_EXPECTED_SHA": args.tooling_sha,
               "RFX_KIND": "two", "RFX_NX": str(slab), "RFX_JOB_SUFFIX": args.job_suffix, **shape_env,
               "RFX_SOURCE_ARCHIVE": "/input/mn/source.git", "RFX_BOOTSTRAP": "/input/mn",
               "RFX_RUNS_ROOT": "/root/workspace/claude-workspace/rfx/runs",
               "PYTHONUNBUFFERED": "1", "HDF5_USE_FILE_LOCKING": "FALSE", "LANG": "C.UTF-8",
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_PLATFORMS": "cuda"}
        cli = ["vessl", "experiment", "create", "--project", "byungkwan", "--cluster", "remilab-c0",
               "--resource", args.two_preset, "--image-url", "nvcr.io/nvidia/jax:24.10-py3",
               "--worker-count", "2", "--framework-type", "pytorch", "--message", f"multinode-{args.stamp}-{job}",
               "--upload-local-file", f"{payload}:/input/mn", "--no-use-vesslignore",
               "--working-dir", "/input/mn", "--output-dir", "/output",
               "--command", "sh /input/mn/multinode_probe_job.sh"]
        for key, value in env.items():
            # The experiment CLI splits each hyperparameter on every "=" and aborts
            # on a value that holds one ("jax[cuda12]==0.6.2"); refuse it here.
            if "=" in str(value):
                parser.error(f"{key}={value}: an experiment hyperparameter value cannot contain '=' "
                             "(write a pip pin as 'jax[cuda12]>0.6.1,<0.6.3')")
            cli.extend(["--hyperparameter", f"{key}={value}"])
        log = out / f"{job}.submit.log"
        receipt = artifacts / job
        parse = ("import pathlib,re,sys; t=pathlib.Path(sys.argv[1]).read_text(); "
                 "m=re.search(r\"Created '(\\d+)'\",t); "
                 "assert m, 'Experiment number missing from submit log'; "
                 "pathlib.Path(sys.argv[2]).write_text(m[1]+'\\n'); print(m[1])")
        lines = ["#!/bin/sh", "set -eu", command(["cd", out]), command(["mkdir", "-p", receipt]),
                 command(["chmod", "777", artifacts, receipt]),
                 "set +e", command(cli) + " > " + shlex.quote(str(log)) + " 2>&1", "rc=$?", "set -e",
                 command(["cat", log]), '[ "$rc" -eq 0 ] || exit "$rc"',
                 command(["python3", "-c", parse, log, receipt / "experiment_number.txt"])]
        (out / f"submit-{job}.sh").write_text("\n".join(lines) + "\n")
        number_file = shlex.quote(str(receipt / "experiment_number.txt"))
        download = ["#!/bin/sh", "set -eu", command(["cd", out]), f'number=$(cat {number_file})',
                    f'vessl experiment read "$number" --project byungkwan > {shlex.quote(str(receipt / "provider-status.txt"))}',
                    f'vessl experiment logs "$number" --project byungkwan --tail 100000 > {shlex.quote(str(receipt / "provider.log"))}',
                    'vessl experiment download-output "$number" --project byungkwan --worker-number 0 '
                    + command(["--path", out / f"download-{job}"]),
                    'src=$(find ' + shlex.quote(str(out / f"download-{job}")) + ' -type d -path '
                    + shlex.quote(f"*/multinode-{args.stamp}/{job}") + ' | head -1)',
                    '[ -n "$src" ] || { echo "downloaded output has no ' + job + ' directory"; exit 1; }',
                    'find "$src" -name "._*" -delete',
                    'cp -R "$src/." ' + shlex.quote(str(receipt)),
                    'find ' + shlex.quote(str(receipt)) + ' -name "._*" -delete',
                    command(["python3", root / "scripts/diagnostics/summarize_multinode_job.py", "--output", receipt,
                             "--process-count", "2", "--sha", args.tooling_sha, "--aggregate-only"])]
        (out / f"collect-{job}.sh").write_text("\n".join(download) + "\n")
    print(f"Prepared {out}; artifact root {artifacts}. No jobs submitted.")


if __name__ == "__main__":
    main()
