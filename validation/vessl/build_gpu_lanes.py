"""Generate the VESSL lane YAMLs for the GPU sparse-direct backend (J0/J1/J2).

    .venv/bin/python validation/vessl/build_gpu_lanes.py            # write the YAMLs
    .venv/bin/python validation/vessl/build_gpu_lanes.py --tarball  # + the source tarball

Then, once per source change:

    bash validation/vessl/submit.sh j0        # upload the tarball + vessl run create
    bash validation/vessl/wait.sh <run-id>    # poll, tail, terminate on stall
    bash validation/vessl/harvest.sh          # copy the job JSONs off the NFS

WHY THIS FILE EXISTS
--------------------
Two hard constraints of this cluster, both verified:

1. A quoted heredoc inside a VESSL ``run:`` block kills the job at parse
   time. The lane's measurement code therefore travels as base64 on one
   line (``echo <b64> | base64 -d > ...``), wrapped in ``set +x`` /
   ``set -x`` so the log stays readable -- and it is generated from a real
   ``.py`` file in this directory, which ruff and mypy see, instead of a
   string literal inside a YAML.
2. There is NO working file-upload path to this cluster, so the source
   travels INSIDE the YAML. Measured, in this order:

   * ``vessl storage copy-file <local> volume://remilab-fs/...`` fails with
     "InvalidParameters (400): unsupported storage type cluster-nfs" -- the
     NFS storage does not support the file API -- and the same NFS is
     mounted READ-ONLY on the Mac, so it cannot be written from either end.
   * Uploading to the VESSL-managed storage DOES work
     (``vessl storage copy-file /tmp/rfx-src.tgz
     volume://vessl-storage/<volume>``), and a run can then pull it in with
     an ``import:`` section -- but NOT with ``mount:``: both
     ``mount: {/root/upload/: volume://vessl-storage/<volume>}`` and the
     same with a trailing slash are rejected at submission with
     "InvalidParameters (400): Invalid parameters", while
     ``import: {/root/upload/: volume://vessl-storage/<volume>}`` is
     accepted. So ``import:`` is how the source gets in, and the NFS
     ``mount:`` is how the artifacts get out.
   * NOTE on reading the queue: "0/14 nodes are available: pod has unbound
     immediate PersistentVolumeClaims" appears for a few seconds on EVERY
     run here, including the lanes that work (it is in the other session's
     running job's log too, six seconds before it was assigned). It is the
     volume being provisioned, not a rejection. Do not terminate on it.
   * ``--payload`` is the fallback if ``import:`` ever stops working: a
     tar.xz of the working tree's Python, base64 in the ``run:`` block in
     60 kB chunks appended line by line (one 1.4 MB ``echo`` argument would
     risk MAX_ARG_STRLEN; ``echo`` is a shell builtin so the chunks cost
     nothing), unpacked with Python's own ``tarfile``. It makes the YAML
     1.4 MB, which may exceed what the run spec accepts -- it is untested,
     and it is the reason the default path is the upload.

Artifacts go to ``/root/workspace/claude-workspace/rfx/runs/fdfd-gpu-<lane>-<utc>/``
(the NFS mount), which is readable on the Mac at
``/Users/bk-squared/nfs-remilab/personal-workspaces/claude-workspace/rfx/runs/``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

# The known-good image on this cluster (read: the NFS lane
# byungkwan-workspace/research/rfx/scripts/vessl_931_post_cv14.yaml).
# pytorch/pytorch:2.1.0-cuda12.1 base: conda Python 3.10, pip works, PyPI
# reachable, NO apt/dnf mirrors. The task's alternative if this one is
# rejected is pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime (Python 3.11).
IMAGE = "ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8"
IMAGE_ALT = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime"

CLUSTER, PRESET = "remilab-c0", "gpu-rtx4090"
NFS = "volume://remilab-fs/personal-workspaces/"
# the VESSL-managed volume the source tarball is uploaded to (created once
# with ``vessl storage create-volume``); pulled in with ``import:``
SRC_VOLUME = "rfx-fdfd-gpu-v-20260915"
SRC_IMPORT = f"volume://vessl-storage/{SRC_VOLUME}"
TARBALL = "rfx-src.tgz"

# Python pins. The image's conda Python is 3.10, and jax 0.11.1 / scipy
# 1.18.1 -- the Mac's versions, which produced the CPU levels -- both
# require >= 3.12, so they CANNOT be installed here. The lane pins the
# newest pair the image's interpreter accepts and makes the version
# difference a measured non-issue instead of an assumption: J0 recomputes
# the W/3 level and compares L_dut with the CPU study's recorded
# 3.005576903387643e-10 H (key ``gates.L_dut_vs_cpu_study``).
# shapely is not optional here: rfx.fdfd.gds.fill_fractions needs it for the
# area-exact rasterisation every spiral level is built with (measured: the
# J0 run of 20260915T053936Z died in build_level without it).
PINS = 'jax[cpu]==0.6.2 "numpy<2.3" "scipy>=1.14" shapely pytest'
GPU_PINS = '"nvmath-python[cu12]" cupy-cuda12x'

RUNS = "/root/workspace/claude-workspace/rfx/runs"


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


PAYLOAD_MARK = "  # PAYLOAD: 60 kB base64 chunks of the source tar.xz, injected by --payload"
CHUNK = 60000


def payload_lines(with_payload: bool) -> str:
    """The shell that materialises the source tree inside the job.

    Without ``--payload`` this is one marker comment, so the YAML kept in
    the repository is 17 kB of readable shell instead of 1.4 MB of base64;
    ``submit.sh`` always regenerates with the payload.
    """
    head = ('  echo "=== unpacking the embedded source payload ==="\n'
            '  set +x\n'
            '  : > /tmp/rfx-src.b64\n')
    tail = ('  set -x\n'
            '  base64 -d < /tmp/rfx-src.b64 > /tmp/rfx-src.txz\n'
            '  ls -la /tmp/rfx-src.txz\n'
            '  "$PY" -c "import tarfile; tarfile.open(\'/tmp/rfx-src.txz\')'
            '.extractall(\'$WORK\')"\n'
            '  rm -f /tmp/rfx-src.b64 /tmp/rfx-src.txz\n')
    if not with_payload:
        return (f'  echo "=== source: the uploaded tarball, imported at '
                f'/root/upload/{TARBALL} ==="\n'
                f'  ls -la /root/upload/ || true\n'
                f'  test -f /root/upload/{TARBALL} || {{ echo "ERROR: import did not '
                f'deliver /root/upload/{TARBALL}"; exit 3; }}\n'
                f'  tar xzf /root/upload/{TARBALL} -C "$WORK"\n')
    blob = b64(payload_tarball())
    chunks = [blob[i:i + CHUNK] for i in range(0, len(blob), CHUNK)]
    body = "".join(f"  echo {c} >> /tmp/rfx-src.b64\n" for c in chunks)
    return head + body + tail


def upload_tarball() -> pathlib.Path:
    """The tar.gz that goes to the VESSL volume and into the job.

    The WORKING TREE, not ``git archive HEAD``: the backend and the lanes
    are uncommitted (the branch is shared with two other agents), so an
    archive of HEAD would ship a tree without ``rfx/fdfd/_cudss.py``.
    """
    path = pathlib.Path("/tmp") / TARBALL
    subprocess.run(
        ["tar", "czf", str(path), "--exclude=__pycache__", "--exclude=*.pyc",
         "--exclude=.mypy_cache", "--exclude=.pytest_cache",
         "rfx", "tests", "conftest.py", "pyproject.toml",
         "validation/fdfd", "validation/vessl",
         # the Greenhouse referee and the hplane comparator live here:
         # spiral_convergence.load_referee() and tests/unit/fdfd/test_fdfd_hplane.py
         # both import them BY PATH, so leaving the directory out costs 3
         # test failures and J2's referee block (measured in the J0 run of
         # 20260915T053936Z, which is why it is here)
         "validation/crossval/comparators"],
        cwd=REPO, check=True)
    print(f"tarball {path} {path.stat().st_size / 1e6:.2f} MB "
          f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")
    return path


def payload_tarball() -> pathlib.Path:
    """tar.xz of every ``.py`` of the WORKING TREE the lanes import.

    Not ``git archive HEAD``: the backend and the lanes are uncommitted
    (the branch is shared with two other agents), so an archive of HEAD
    would ship a tree without ``rfx/fdfd/_cudss.py``. Only Python, the
    two test files the lanes run, the spiral study and its JSON, and this
    directory -- 1.1 MB compressed, against 7.3 MB for the whole tree.
    """
    path = pathlib.Path("/tmp") / "rfx-src.txz"
    listing = pathlib.Path("/tmp/rfx-payload.list")
    files = sorted(str(p.relative_to(REPO)) for p in REPO.glob("rfx/**/*.py")
                   if "__pycache__" not in p.parts)
    files += sorted(str(p.relative_to(REPO))
                    for p in (REPO / "validation/crossval/comparators").glob("*.py"))
    files += ["rfx/py.typed", "conftest.py", "pyproject.toml",
              "tests/__init__.py", "tests/_x64_compat.py",
              "tests/unit/fdfd/test_fdfd_linear_solve.py", "tests/unit/fdfd/test_fdfd_hplane.py",
              "validation/fdfd/spiral_convergence.py",
              "validation/fdfd/spiral_convergence.json"]
    for optional in ("validation/fdfd/rfic_spiral.py",
                     "validation/fdfd/rfic_spiral.json"):
        if (REPO / optional).exists():
            files.append(optional)
    files += sorted(str(p.relative_to(REPO)) for p in (HERE).iterdir()
                    if p.suffix in (".py", ".sh"))
    listing.write_text("\n".join(files) + "\n")
    subprocess.run(["tar", "cJf", str(path), "-T", str(listing)],
                   cwd=REPO, check=True)
    print(f"payload {path} {path.stat().st_size / 1e6:.2f} MB, {len(files)} files")
    return path


def shell(lane: str, script: pathlib.Path, pytest_block: str, timeout_s: int,
          with_payload: bool = False) -> str:
    """The ``run:`` block. No heredocs (they kill the job at parse time), and
    POSIX shell only -- this image's /bin/sh is dash: no ``PIPESTATUS``, no
    ``[[ ]]``, no arrays. The lane's exit code travels through a file
    written inside the subshell so the output can still be ``tee``d live."""
    code = b64(script)
    digest = hashlib.sha256(script.read_bytes()).hexdigest()[:16]
    return f"""  set -eu
  echo "=== rfx fdfd GPU lane {lane} ({script.name}, sha256 {digest}) ==="
  date -u
  nvidia-smi || echo "nvidia-smi FAILED"
  PY=python
  command -v "$PY" >/dev/null 2>&1 || PY=/opt/conda/bin/python
  "$PY" -V
  OUT={RUNS}/fdfd-gpu-{lane}-$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$OUT"
  echo "$OUT" > {RUNS}/fdfd-gpu-{lane}.latest
  trap 'chmod -R a+rX "$OUT" 2>/dev/null || true' EXIT
  echo "artifacts: $OUT"
  export RFX_OUT="$OUT"
  WORK=/root/work/rfx-fdfd-gpu-{lane}
  rm -rf "$WORK"; mkdir -p "$WORK"
{payload_lines(with_payload)}  cd "$WORK"
  test -f "$WORK/rfx/fdfd/_cudss.py" || {{ echo "ERROR: payload did not unpack"; ls -la "$WORK"; exit 3; }}
  echo "=== pip install (PyPI at job start) ==="
  "$PY" -m pip install -q {PINS} 2>&1 | tail -5
  set +e
  "$PY" -m pip install -q {GPU_PINS} 2>&1 | tail -10
  gpu_pip=$?
  set -e
  echo "gpu_pip_rc=$gpu_pip" | tee "$OUT/pip.rc"
  "$PY" -m pip freeze > "$OUT/pip-freeze.txt" 2>/dev/null || true
  export PYTHONPATH="$WORK"
  export RFX_REPO_ROOT="$WORK"
  export MPLBACKEND=Agg
  "$PY" -c "import jax, rfx, rfx.fdfd.linear_solve as ls; print('probe ok | jax', jax.__version__, '| jax backend', jax.default_backend(), '| default_backend', ls.get_default_backend())"
  "$PY" -c "from rfx.fdfd import _cudss; import json; print('backends:', json.dumps(_cudss.availability(), indent=1))" | tee "$OUT/backends.txt"
{pytest_block}  echo "=== lane {lane} ==="
  set +x
  echo {code} | base64 -d > "$WORK/validation/vessl/_lane_embedded.py"
  set -x
  cmp -s "$WORK/validation/vessl/_lane_embedded.py" "$WORK/validation/vessl/{script.name}" && echo "embedded lane == repo lane" || echo "NOTE: embedded lane differs from the tarball copy"
  set +e
  ( timeout {timeout_s} "$PY" -u "$WORK/validation/vessl/_lane_embedded.py"; echo $? > "$OUT/{lane}.rc" ) 2>&1 | tee "$OUT/{lane}.log"
  set -e
  rc=$(cat "$OUT/{lane}.rc" 2>/dev/null || echo 99)
  ls -la "$OUT"
  chmod -R a+rX "$OUT" 2>/dev/null || true
  echo "LANE={lane} RC=$rc"
  exit "$rc"
"""


PYTEST_J0 = """  echo "=== pytest: linear_solve + hplane, backend forced BOTH ways ==="
  set +e
  RFX_FDFD_BACKEND=superlu timeout 1800 "$PY" -m pytest tests/unit/fdfd/test_fdfd_linear_solve.py tests/unit/fdfd/test_fdfd_hplane.py -q -rs -p no:cacheprovider > "$OUT/pytest-superlu.log" 2>&1
  echo "$?" > "$OUT/pytest-superlu.rc"
  tail -n 15 "$OUT/pytest-superlu.log"
  RFX_FDFD_BACKEND=cudss timeout 1800 "$PY" -m pytest tests/unit/fdfd/test_fdfd_linear_solve.py tests/unit/fdfd/test_fdfd_hplane.py -q -rs -p no:cacheprovider > "$OUT/pytest-cudss.log" 2>&1
  echo "$?" > "$OUT/pytest-cudss.rc"
  tail -n 15 "$OUT/pytest-cudss.log"
  set -e
  echo "pytest rcs: superlu=$(cat "$OUT/pytest-superlu.rc") cudss=$(cat "$OUT/pytest-cudss.rc")"
"""


def yaml_for(lane: str, script: str, description: str, env: dict[str, str],
             pytest_block: str, timeout_s: int, image: str = IMAGE,
             with_payload: bool = False) -> str:
    base_env = {
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        # cuDSS reorders on the host; the node has ~32 cores and the
        # threading layer in the nvidia-cudss wheel is picked up by
        # rfx.fdfd._cudss.threading_lib()
        "OMP_NUM_THREADS": "16",
        "LANG": "C.UTF-8",
        # JAX stays on the CPU on purpose: the assembly is cheap and the
        # solve is the cost, and the GPU is for the factorisation. Measuring
        # jax[cuda12] too is explicitly optional and not worth the pip risk
        # on this image.
        "JAX_PLATFORMS": "cpu",
    }
    base_env.update(env)
    env_yaml = "\n".join(f'  {k}: "{v}"' for k, v in base_env.items())
    import_yaml = "" if with_payload else f"import:\n  /root/upload/: {SRC_IMPORT}\n"
    body = shell(lane, HERE / script, pytest_block, timeout_s, with_payload)
    return f"""name: rfx-fdfd-gpu-{lane}
description: "{description}"
tags: [rfx, fdfd, gpu, cudss, {lane}]
resources:
  cluster: {CLUSTER}
  preset: {PRESET}
image: {image}
env:
{env_yaml}
mount:
  /root/workspace/: {NFS}
{import_yaml}run: |-
{body}"""


LANES: dict[str, dict] = {
    "j0": dict(
        script="lane_j0_probe.py",
        description=(
            "J0 probe: cuDSS/cuSOLVER availability, the linear_solve and hplane pytest "
            "suites with RFX_FDFD_BACKEND forced both ways, _cudss.selftest, gate G1 on "
            "the real W/3 spiral DUT fixture (N=108898: residuals, solution difference, "
            "factor/solve wall times, device memory, LU nnz), cuDSS plan-only memory "
            "estimates at W/3-W/4-W/6, and value_and_grad of L_dut at W/3 with both "
            "backends (equality 1e-8, plus the cuDSS gradient against the CPU study's "
            "recorded FD4). Submit: bash validation/vessl/submit.sh j0"),
        env={"RFX_FDFD_BACKEND": "superlu"},
        pytest_block=PYTEST_J0,
        timeout_s=2400,
    ),
    "j1": dict(
        script="lane_j1_scale.py",
        description=(
            "J1 scale: the spiral levels the Mac cannot reach -- W/4 (N=172747) and W/6 "
            "(N=352536) of the SAME fixture as the CPU study, value_and_grad of L_dut "
            "through cuDSS, written as level blocks in spiral_convergence.json's "
            "structure plus the referee, and the extended 5-level Richardson. "
            "Submit: bash validation/vessl/submit.sh j1"),
        env={"RFX_FDFD_BACKEND": "cudss",
             "RFX_LANE_DIVS": "4 6",
             "RFX_LANE_GRAD": "1",
             "RFX_LANE_FD_DIV": "4",
             "RFX_LANE_FD_PARAM": "2",
             # 1, not the default 4: cuDSS has no transposed solve, so a
             # bigger LU cache saves the adjoint no factorisation and only
             # raises the number of factors resident on the 24 GB device
             # (see "Device memory" in rfx/fdfd/_cudss.py)
             "RFX_LANE_FACTOR_CACHE": "1",
             "RFX_LANE_DEADLINE_S": "5400",
             # J0 measured 1.38 / 2.43 / 5.68 GB of permanent device memory
             # per cuDSS factor at W/3 / W/4 / W/6, so W/6 fits on a 24 GB
             # card with room to spare and hybrid memory mode is not needed
             "RFX_FDFD_CUDSS_HYBRID": "0",
             "RFX_FDFD_CUDSS_FREE_FORWARD": "1",
             # the PEC fixture (gate V1b) is affordable now: J0 measured a
             # cuDSS factorisation of the W/3 DUT operator at 0.95 s against
             # SuperLU's 139.23 s
             "RFX_LANE_PEC": "1",
             "RFX_LANE_IR_DIV": "3",
             "RFX_LANE_IR_SWEEP": "0 1 2 4",
             "RFX_LANE_IR_PHYSICS": "0 2 4"},
        pytest_block="",
        timeout_s=6600,
    ),
    "j2": dict(
        script="lane_j2_paper.py",
        description=(
            "J2 paper geometry (only after J1): the 2.4 GHz LC-VCO paper's 3-turn square "
            "spiral (r_out 218 um, W 30 um, S 14 um, TopMetal2 3 um) on study H's own "
            "SmallStack conventions (validation/fdfd/rfic_spiral.py), L at 100 MHz "
            "uniform-current against the bridge-corrected referee, plus cuDSS plan-only "
            "memory estimates for the W/3 grid study H built but could not solve "
            "(N=509580). Submit: bash validation/vessl/submit.sh j2"),
        env={"RFX_FDFD_BACKEND": "cudss",
             # W/1 is study H's own level (N = 85785, 2 cells across W) and
             # W/3 is the one it could only BUILD (N = 509580, 6 cells).
             # Both are affordable now: J0 measured a cuDSS factorisation of
             # the W/3 spiral operator (N = 108898) at 0.95 s and a whole
             # three-fixture value_and_grad at 6.9 s against SuperLU's 432.9 s.
             "RFX_LANE_DIVS": "1 3",
             # planned FIRST, so the device-memory estimate for N = 509580 is
             # on record before the solve that has to fit in 24 GB
             "RFX_LANE_PLAN_DIVS": "3",
             "RFX_LANE_FACTOR_CACHE": "1",
             "RFX_LANE_DEADLINE_S": "5400",
             "RFX_FDFD_CUDSS_FREE_FORWARD": "1",
             # the one bounded measurement of the cuSOLVER fallback at
             # N ~ 1e5 on a real system (one right-hand-side column)
             "RFX_LANE_CUSOLVER_DIV": "1"},
        pytest_block="",
        timeout_s=6600,
    ),
}


def build(alt_image: bool = False, with_payload: bool = False,
          out_dir: pathlib.Path | None = None,
          lanes: tuple[str, ...] = ()) -> list[pathlib.Path]:
    out = []
    dest = out_dir or HERE
    for lane, cfg in LANES.items():
        if lanes and lane not in lanes:
            continue
        text = yaml_for(lane, image=IMAGE_ALT if alt_image else IMAGE,
                        with_payload=with_payload, **cfg)
        path = dest / f"gpu_{lane}{'_alt' if alt_image else ''}.yaml"
        path.write_text(text)
        out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--payload", action="store_true",
                    help="embed the source tar.xz (submit-ready; 1.4 MB per YAML)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="write the YAMLs here instead of validation/vessl/")
    ap.add_argument("--lane", action="append", default=[],
                    help="only this lane (repeatable)")
    ap.add_argument("--alt-image", action="store_true",
                    help=f"generate against {IMAGE_ALT} instead")
    ap.add_argument("--tarball", action="store_true",
                    help="also build /tmp/rfx-src.tgz for the upload")
    args = ap.parse_args(argv)
    if args.tarball:
        upload_tarball()
    for p in build(args.alt_image, args.payload, args.out, tuple(args.lane)):
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
