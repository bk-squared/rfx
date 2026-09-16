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
# study P (validation/fdfd/invariant_ladder.py) uploads to ITS OWN volume and
# under its own tarball name, so that neither track can ever import the
# other's (older) source tree by mistake
P_VOLUME = "rfx-fdfd-gpu-p-20260915"
P_TARBALL = "rfx-src-p.tgz"
PRESET_A6000 = "gpu-a6000-1"      # 48 GB, remilab-c0: the finest level only
# study R (validation/fdfd/rfic_spiral.py + rfic_design.py): its own volume and
# tarball name, for the same reason
R_VOLUME = "rfx-fdfd-gpu-r-20260916"
R_TARBALL = "rfx-src-r.tgz"
# study P's hybrid-memory lane (P3, lane_p3_hybrid.py): its own volume again, so
# it can never import study P's or study R's older tree
# (the rerun that re-measured the plans with the shipped constructor-only
# limit gets its own volume again, so no job can import the older tree)
P3_VOLUME = "rfx-fdfd-gpu-p3b-20260916"
P3_TARBALL = "rfx-src-p3.tgz"

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


def payload_lines(with_payload: bool, tarball: str = TARBALL) -> str:
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
                f'/root/upload/{tarball} ==="\n'
                f'  ls -la /root/upload/ || true\n'
                f'  test -f /root/upload/{tarball} || {{ echo "ERROR: import did not '
                f'deliver /root/upload/{tarball}"; exit 3; }}\n'
                f'  tar xzf /root/upload/{tarball} -C "$WORK"\n')
    blob = b64(payload_tarball())
    chunks = [blob[i:i + CHUNK] for i in range(0, len(blob), CHUNK)]
    body = "".join(f"  echo {c} >> /tmp/rfx-src.b64\n" for c in chunks)
    return head + body + tail


def upload_tarball(name: str = TARBALL) -> pathlib.Path:
    """The tar.gz that goes to the VESSL volume and into the job.

    The WORKING TREE, not ``git archive HEAD``: the backend and the lanes
    are uncommitted (the branch is shared with two other agents), so an
    archive of HEAD would ship a tree without ``rfx/fdfd/_cudss.py``.
    """
    path = pathlib.Path("/tmp") / name
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
          with_payload: bool = False, tarball: str = TARBALL) -> str:
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
{payload_lines(with_payload, tarball)}  cd "$WORK"
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
             with_payload: bool = False, volume: str = SRC_VOLUME,
             tarball: str = TARBALL, preset: str = PRESET) -> str:
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
    src_import = f"volume://vessl-storage/{volume}"
    import_yaml = "" if with_payload else f"import:\n  /root/upload/: {src_import}\n"
    body = shell(lane, HERE / script, pytest_block, timeout_s, with_payload, tarball)
    return f"""name: rfx-fdfd-gpu-{lane}
description: "{description}"
tags: [rfx, fdfd, gpu, cudss, {lane}]
resources:
  cluster: {CLUSTER}
  preset: {preset}
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
    # ---- study P: the level-invariant fixture (validation/fdfd/invariant_ladder.py)
    "p0": dict(
        script="lane_p0_probe.py",
        description=(
            "P0 probe of study P (level-invariant spiral fixture): cuDSS plan-only memory "
            "estimates of the jointly refined levels, the level-2 wall ladder (margins 10/20/"
            "40/80 W), gate P6 (old fixture with walls / short standard / vertical grid made "
            "physical one at a time, W/1 and W/2), the uniform-current sigma plateau at "
            "level 3 plus the 10 MHz RC twin, and value_and_grad at levels 1-2. "
            "Submit: bash validation/vessl/submit.sh p0"),
        env={"RFX_FDFD_BACKEND": "cudss",
             "RFX_P_BLOCKS": "plans walls p6 plateau ladder bigplans",
             "RFX_P_PLANS": "10:2:3 20:2:3 10:2:4 20:2:4 10:1:4",
             "RFX_P_WALLS": "10 20 40 80",
             "RFX_P_P6_LEVELS": "1 2",
             "RFX_P_PLATEAU_LEVEL": "3",
             # the plateau is a RELATIVE sigma sweep; the 10 W walls (N = 581430
             # at level 3) keep it on the 24 GB card whatever margin is chosen
             "RFX_P_PLATEAU_WALL": "10",
             "RFX_P_LEVELS": "1 2",
             "RFX_P_BIGPLANS": "10:2:5 10:1:5 10:1:6",
             "RFX_P_FACTOR_CACHE": "1",
             "RFX_P_DEADLINE_S": "4200",
             "RFX_FDFD_CUDSS_HYBRID": "0",
             "RFX_FDFD_CUDSS_FREE_FORWARD": "1"},
        pytest_block="",
        timeout_s=4800,
        volume=P_VOLUME, tarball=P_TARBALL,
    ),
    "p1": dict(
        script="lane_p1_ladder.py",
        description=(
            "P1 ladder of study P on the 24 GB card: the level-invariant spiral fixture "
            "(10 W walls, physical post and port gap) jointly refined, levels m = 1, 2, 3 "
            "(metal_cells 1 at level 1): cuDSS factor probe + value_and_grad of L_dut per "
            "level, FD4 of dL/dwidth at m = 3, the thru probe at m = 1-3, and the "
            "metal_cells-2 family's m = 3 as a cross-check. "
            "Submit: bash validation/vessl/submit.sh p1"),
        env={"RFX_FDFD_BACKEND": "cudss",
             "RFX_P_LANE_NAME": "P1",
             "RFX_P_JSON": "p1_ladder.json",
             "RFX_P_WALL": "10",
             # levels 1 and 2 and the thru probe at 1-3 landed in the first P1 run
             # (369367261233); its level 3 died in value_and_grad with cuDSS
             # ALLOC_FAILED because a factor cache of 1 keeps the last fixture's
             # A factor alive while its A^T is factorised (two factors at once).
             # A cache of 0 builds each factor for one solve and frees it: peak
             # one factor, and nothing is lost -- cuDSS refactors A^T anyway
             "RFX_P_LEVELS": "3",
             "RFX_P_XLEVELS": "2:3",
             "RFX_P_FD_LEVEL": "3",
             "RFX_P_THRU_LEVELS": "",
             "RFX_P_FACTOR_CACHE": "0",
             "RFX_P_DEADLINE_S": "4200",
             "RFX_FDFD_CUDSS_HYBRID": "0",
             "RFX_FDFD_CUDSS_FREE_FORWARD": "1"},
        pytest_block="",
        timeout_s=4800,
        volume=P_VOLUME, tarball=P_TARBALL,
    ),
    "p1b": dict(
        script="lane_p1_ladder.py",
        description=(
            "P1b, the finest level of study P on the 48 GB card: level m = 4 of the "
            "level-invariant spiral fixture (N = 1095964; P0's cuDSS plan: 36.61 GB permanent), "
            "factor probe + value_and_grad of L_dut, FD4 of dL/dwidth through cuDSS, the "
            "thru probe at m = 4. Submit: bash validation/vessl/submit.sh p1b"),
        env={"RFX_FDFD_BACKEND": "cudss",
             "RFX_P_LANE_NAME": "P1b",
             "RFX_P_JSON": "p1b_ladder.json",
             "RFX_P_WALL": "10",
             "RFX_P_LEVELS": "4",
             "RFX_P_FD_LEVEL": "4",
             "RFX_P_THRU_LEVELS": "4",
             "RFX_P_FACTOR_CACHE": "0",
             "RFX_P_DEADLINE_S": "4200",
             "RFX_FDFD_CUDSS_HYBRID": "0",
             "RFX_FDFD_CUDSS_FREE_FORWARD": "1"},
        pytest_block="",
        timeout_s=4800,
        volume=P_VOLUME, tarball=P_TARBALL, preset=PRESET_A6000,
    ),
    "p1c": dict(
        script="lane_p1_ladder.py",
        description=(
            "P1c, study P's stretch level on the 48 GB card: level m = 5 of the level-invariant "
            "spiral fixture (N = 2128175; P0's cuDSS plan: 91.87 GB permanent, 16.78 GB hybrid "
            "minimum) as a three-fixture FORWARD solve in cuDSS hybrid (host-backed) memory "
            "mode -- L_dut only, for a 4th point in the Richardson series. "
            "Submit: bash validation/vessl/submit.sh p1c"),
        env={"RFX_FDFD_BACKEND": "cudss",
             "RFX_P_LANE_NAME": "P1c",
             "RFX_P_JSON": "p1c_ladder.json",
             "RFX_P_WALL": "10",
             "RFX_P_LEVELS": "5",
             "RFX_P_GRAD": "0",
             "RFX_P_FACTOR_PROBE": "0",
             "RFX_P_TWIN": "0",
             "RFX_P_FD_LEVEL": "",
             "RFX_P_THRU_LEVELS": "",
             "RFX_P_HYBRID_LEVELS": "5",
             "RFX_P_FACTOR_CACHE": "0",
             "RFX_P_DEADLINE_S": "4500",
             "RFX_FDFD_CUDSS_HYBRID": "0",
             "RFX_FDFD_CUDSS_FREE_FORWARD": "1"},
        pytest_block="",
        timeout_s=4900,
        volume=P_VOLUME, tarball=P_TARBALL, preset=PRESET_A6000,
    ),
    "p2": dict(
        script="lane_p2_paper.py",
        description=(
            "P2, study P's secondary: the paper-geometry square spiral (3 turns, r_out 218 um, "
            "W 30 um, S 14 um, M2 3 um, rfic_spiral.py's stack, vacuum) on the level-invariant "
            "fixture with a valid uniform-current protocol (10 MHz, 2e6 S/m: delta = 3.75 W): "
            "cuDSS plans, level-1 wall check, the sigma plateau at level 2 (4 cells across W) "
            "plus the 1 MHz RC twin, forward L at levels 1-2. "
            "Submit: bash validation/vessl/submit.sh p2"),
        env={"RFX_FDFD_BACKEND": "cudss",
             "RFX_P_BLOCKS": "plans walls plateau levels",
             "RFX_P_PAPER_PLANS": "2 3",
             "RFX_P_PAPER_PLATEAU": "2",
             "RFX_P_PAPER_LEVELS": "1 2",
             "RFX_P_FACTOR_CACHE": "1",
             "RFX_P_DEADLINE_S": "4200",
             "RFX_FDFD_CUDSS_HYBRID": "0",
             "RFX_FDFD_CUDSS_FREE_FORWARD": "1"},
        pytest_block="",
        timeout_s=4800,
        volume=P_VOLUME, tarball=P_TARBALL,
    ),
}

PYTEST_P3 = """  echo "=== pytest: the hybrid-memory option, on the GPU ==="
  set +e
  RFX_FDFD_BACKEND=cudss timeout 900 "$PY" -m pytest tests/unit/fdfd/test_fdfd_linear_solve.py -q -rs -p no:cacheprovider -k "hybrid" > "$OUT/pytest-hybrid.log" 2>&1
  echo "$?" > "$OUT/pytest-hybrid.rc"
  tail -n 15 "$OUT/pytest-hybrid.log"
  set -e
  echo "pytest rc: hybrid=$(cat "$OUT/pytest-hybrid.rc")"
"""


LANES["p3"] = dict(
    script="lane_p3_hybrid.py",
    description=(
        "P3, study P's hybrid-memory lane on the 48 GB card: cuDSS hybrid (host+device) "
        "memory mode with an EXPLICIT device-memory limit (RFX_FDFD_CUDSS_HYBRID_LIMIT, "
        "which the m = 5 attempt of run 369367261252 ran WITHOUT). Gate H1 at level 3 "
        "(11.37 GB factor by cuDSS's in-core plan): hybrid with a 6 GiB limit -- below the "
        "factor -- against the in-memory run, L and the gradient to 1e-9, each with its "
        "in-core-vs-in-core CONTROL (the solution vector's and the value_and_grad's), the "
        "slowdown and the sampled device peak; then the container's host-memory limit "
        "(cgroup, not /proc/meminfo) against what a level-5 factor (91.87 GB) needs in host "
        "memory, and the level-5 forward + value_and_grad only if it fits. The plan "
        "estimates carry hybrid_limit_applied=constructor: the limit reaches cuDSS at "
        "DirectSolver construction and nowhere else. "
        "Submit: bash validation/vessl/submit.sh p3"),
    env={"RFX_FDFD_BACKEND": "cudss",
         "RFX_P_LANE_NAME": "P3",
         "RFX_P_JSON": "p3_ladder.json",
         "RFX_P_WALL": "10",
         "RFX_P3_BLOCKS": "host h1 m5",
         "RFX_P3_H1_LEVEL": "3",
         # 6 GiB is 53 % of the 11.37 GB the level-3 factor occupies in device
         # memory (cuDSS's own in-core plan; the 12.65 GB the P1 lane reported
         # is device memory IN USE, factor + operands + pool), and above
         # cuDSS's hybrid minimum for this size (2.73 GB), so the factor
         # CANNOT be resident: hybrid mode has to stream it from the host
         "RFX_P3_H1_LIMIT": "6GiB",
         "RFX_P3_M5_LEVEL": "5",
         # 40 GiB of the A6000's 47.54 GB, leaving room for the right-hand
         # sides, the CSR copies, cupy's pool and the context
         "RFX_P3_M5_LIMIT": "40GiB",
         "RFX_P3_HOST_MARGIN_GB": "4",
         # both controls on: the same computation in-core TWICE, for the
         # solution vector of the probe and for the value_and_grad
         "RFX_P3_PROBE_CONTROL": "1",
         "RFX_P3_CONTROL": "1",
         "RFX_P_DEADLINE_S": "5400",
         # the lane switches hybrid mode on per block; these are the defaults
         "RFX_FDFD_CUDSS_HYBRID": "0",
         "RFX_FDFD_CUDSS_FREE_FORWARD": "1"},
    pytest_block=PYTEST_P3,
    timeout_s=5700,
    volume=P3_VOLUME, tarball=P3_TARBALL, preset=PRESET_A6000)


# ---- study R: the paper-scale RFIC inductor (validation/fdfd/rfic_spiral.py,
#      validation/fdfd/rfic_design.py), one lane script, blocks by env
R_ENV = {"RFX_FDFD_BACKEND": "cudss",
         # cuDSS has no transposed solve: a cache of 0 keeps ONE factor resident
         # (study P's ALLOC_FAILED at a cache of 1 is why)
         "RFX_R_FACTOR_CACHE": "0",
         "RFX_FDFD_CUDSS_HYBRID": "0",
         "RFX_FDFD_CUDSS_FREE_FORWARD": "1"}
LANES.update({
    "r0": dict(
        script="lane_r_rfic.py",
        description=(
            "R0 probe of study R (paper-scale square spiral, level-invariant fixture, graded "
            "190 um silicon, Leontovich 3.05e7 S/m, 2 S/m silicon, 2.45 GHz): cuDSS plans of "
            "the candidate resolved / finer levels, level-1 substrate-thickness sensitivity "
            "(120/190/300 um) + uniform-vs-graded silicon, forward solves at levels 1-2. "
            "Submit: bash validation/vessl/submit.sh r0"),
        env=dict(R_ENV, RFX_R_LANE="R0", RFX_R_JSON="r0_probe.json",
                 RFX_R_BLOCKS="plans tsi1 levels",
                 RFX_R_PLANS="190:1.5:2 190:2:2 190:0:2 120:1.5:2 190:1.5:3",
                 RFX_R_LEVELS="1 2", RFX_R_DEADLINE_S="3000"),
        pytest_block="", timeout_s=3600, volume=R_VOLUME, tarball=R_TARBALL),
    "r1": dict(
        script="lane_r_rfic.py",
        description=(
            "R1, study R parts A/B on the 24 GB card (graded silicon at ratio 2): level-1 "
            "thickness sensitivity, levels 1-2 (2 / 4 cells across W) at 2.45 GHz, L_diff/Q_diff "
            "gradients (jax.vjp through cuDSS) + FD4 in r_out and width at level 2, the "
            "0.5-8 GHz sweep, the resolved level at 120 um, cuDSS plans of levels 2-3. "
            "Submit: bash validation/vessl/submit.sh r1"),
        env=dict(R_ENV, RFX_R_LANE="R1", RFX_R_JSON="r1_study.json",
                 RFX_R_BLOCKS="tsi1 levels grad fd sweep tsi2 plans",
                 RFX_R_PLANS="190:2:2 190:2:3", RFX_R_LEVELS="1 2",
                 RFX_R_DEADLINE_S="4700"),
        pytest_block="", timeout_s=5100, volume=R_VOLUME, tarball=R_TARBALL),
    "r2": dict(
        script="lane_r_rfic.py",
        description=(
            "R2, study R part C on the 24 GB card: L-BFGS-B with jax.value_and_grad through "
            "cuDSS to L_diff = 4.000 nH (1 %) maximising Q_diff over (r_out, spacing, width) "
            "at the resolved level, every evaluation and iterate logged. "
            "Submit: bash validation/vessl/submit.sh r2"),
        env=dict(R_ENV, RFX_R_LANE="R2", RFX_R_JSON="r2_design.json",
                 RFX_R_BLOCKS="design", RFX_R_QREF="0", RFX_R_FDOPT="r_out width",
                 RFX_R_DEADLINE_S="4700"),
        pytest_block="", timeout_s=5100, volume=R_VOLUME, tarball=R_TARBALL),
    "r3": dict(
        script="lane_r_rfic.py",
        description=(
            "R3, study R part C on the 24 GB card: AD gradients + FD4 (r_out, width) at the "
            "design optimum, then the 3x3x3 baseline sweep of the design box (27 forward "
            "solves). "
            "Submit: bash validation/vessl/submit.sh r3"),
        env=dict(R_ENV, RFX_R_LANE="R3", RFX_R_JSON="r3_fdopt.json",
                 RFX_R_BLOCKS="fdopt csweep",
                 RFX_R_THETA_OPT="0.00020017717250242226 1.18316461480054e-05 2.2e-05", RFX_R_FDOPT="r_out width", RFX_R_DEADLINE_S="4700"),
        pytest_block="", timeout_s=5100, volume=R_VOLUME, tarball=R_TARBALL),
    "r4": dict(
        script="lane_r_rfic.py",
        description=(
            "R4, study R part C on the 48 GB card (the finest grids only): the design "
            "optimum one level finer -- in-plane (6 cells across W, z at level 2, walls 3 W) "
            "and vertically (z at level 3, 4 cells across W, walls 10 W) -- with their "
            "same-box references and the level-1/2 checks that the level shift does not "
            "depend on the box and that the two directions add. The full level 3 is a "
            "106.71 GB factor and does not fit this preset (48 GB device, 32 GiB host). "
            "Submit: bash validation/vessl/submit.sh r4"),
        env=dict(R_ENV, RFX_R_LANE="R4", RFX_R_JSON="r4_fine.json", RFX_R_BLOCKS="fine",
                 RFX_R_THETA_OPT="0.00020017717250242226 1.18316461480054e-05 2.2e-05",
                 RFX_R_FINE_CASES=" ".join([
                     "opt_m3z2_w3:3:2:3:opt", "opt_m2_w3:2:0:3:opt",
                     "opt_m2z3_w10:2:3:10:opt", "opt_m1_w10:1:0:10:opt",
                     "nom_m1_w3:1:0:3:nom", "nom_m2_w3:2:0:3:nom",
                     "nom_m1z2_w10:1:2:10:nom", "nom_m2z1_w10:2:1:10:nom",
                     "nom_m2z3_w10:2:3:10:nom", "nom_m3z2_w3:3:2:3:nom"]),
                 RFX_R_INCORE_MAX_GB="44", RFX_R_DEADLINE_S="4700"),
        pytest_block="", timeout_s=5100, volume=R_VOLUME, tarball=R_TARBALL,
        preset=PRESET_A6000),
})


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
    ap.add_argument("--tarball-for", default=None,
                    help="build the upload tarball of this lane (its own name)")
    ap.add_argument("--upload-target", default=None,
                    help="print '<tarball path> <volume>' of this lane and exit")
    args = ap.parse_args(argv)
    if args.upload_target:
        cfg = LANES[args.upload_target]
        print(f"/tmp/{cfg.get('tarball', TARBALL)} {cfg.get('volume', SRC_VOLUME)}")
        return 0
    if args.tarball:
        upload_tarball()
    if args.tarball_for:
        upload_tarball(LANES[args.tarball_for].get("tarball", TARBALL))
    for p in build(args.alt_image, args.payload, args.out, tuple(args.lane)):
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
