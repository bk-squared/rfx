"""The maintained GPU jobs run JAX 0.10.2 in a Python 3.11 venv, never the VESSL image's python.

PI decision 2026-09-25 ("move if no problems"): GPU work moves from JAX 0.6.2 on the image's
Python 3.10 to a uv venv with Python 3.11 and JAX 0.10.2 (which needs Python >= 3.11). On VESSL
369367265580 (A6000) that venv gave the same outcomes as 0.6.2 on the lane's gpu selection.
The image these jobs use, nvcr.io/nvidia/jax:24.10-py3, ships Python 3.10 with JAX 0.4.33,
whose XLA aborts while compiling forward(distributed=True) on
CHECK_LE(common_utilization, producer_output_utilization) in its fusion cost model (#1252;
the check is gone from JAX 0.4.36). A job that loses the install, or calls the image's `python`
where it meant the venv's, runs 0.4.33 with no message. So each job creates the venv, installs
the pinned set on one line (resolved no later than that run's install), asserts the Python and
JAX it got on its own line before anything else imports JAX, records the venv's freeze, and runs
everything through "$PY", pytest as `"$PY" -m pytest`; this file checks those lines in each
job's run block.
"""

import importlib.util
import re
import string
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
UV = 'python -m pip install -q "uv==0.12.19"'
VENV = "uv venv -q --python 3.11.16 /tmp/venv-py311"
PY = "PY=/tmp/venv-py311/bin/python"
# The pins of the environment that ran VESSL 369367265580, resolved no later than its install
# (its unpinned dependencies, the CUDA wheels among them, then come out as they did there).
PIN = ('uv pip install -q --python "$PY" --exclude-newer 2026-09-27T13:37:21Z "jax[cuda12]==0.10.2" '
       '"numpy==2.4.6" "scipy==1.17.1" "h5py==3.16.0" "matplotlib==3.11.2" "ml_dtypes==0.6.0" '
       '"pyyaml==6.0.3" "optax==0.2.8" "pillow==12.3.0" "pytest==9.1.1" "pytest-split==0.11.0"')
# A job may add pins after the shared set, only these and only in that job. The per-release GPU
# suite ends a stalled test with pytest-timeout's thread method (#1376: run 369367265464 hung 4.9 h).
EXTRA_PINS = {"scripts/vessl_gpu_suite.yaml": ' "pytest-timeout==2.4.0"'}
CHECK = ('"$PY" -c "import jax, sys; assert sys.version_info[:2] == (3, 11), sys.version; '
         "assert jax.__version__ == '0.10.2', jax.__version__; "
         "assert jax.default_backend() == 'gpu', jax.default_backend()\"")
FREEZE = re.compile(r'uv pip freeze --python "\$PY" > "\$(OUT|out)/pip_freeze\.txt"')
# A shell word boundary before a command name: start of line, blank, separator, quote or '='.
_B = r"""(?:^|(?<=[\s;{(|&"'=]))"""
PY_ASSIGN = re.compile(_B + r"(?:export\s+)?PY=")
IMAGE_PYTHON = re.compile(_B + r"(?:/[^\s\"']*/)?python[0-9.]*(?=[\s\"';)]|$)")
PYTEST = re.compile(_B + r"(?:/[^\s\"']*/)?pytest(?=[\s\"';)]|$)")
JOBS = ["scripts/vessl_gpu_suite.yaml",                 # per release and after GPU-touching merges
        "scripts/vessl_validation_lane_a6000.yaml",     # weekly A6000 lane (validation.yml cron)
        "scripts/vessl_crossval_ladder.yaml"]           # monthly crossval-ladder lane (crossval-ladder.yml cron)


def _job_text(path):
    """The job as VESSL gets it: a YAML file, or the TEMPLATE a renderer script formats."""
    if path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("renderer", ROOT / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fields = {name for _, name, _, _ in string.Formatter().parse(module.TEMPLATE) if name}
        return module.TEMPLATE.format(**dict.fromkeys(fields, "x"))
    return (ROOT / path).read_text(encoding="utf-8")


def _commands(text):
    """Uncommented, stripped lines: a commented-out or chained install does not count."""
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


@pytest.mark.parametrize("path", JOBS)
def test_gpu_job_runs_jax_0102_in_a_python_311_venv(path):
    text = _job_text(path)
    assert "nvcr.io/nvidia/jax:24.10-py3" in text, "image changed: re-decide the venv and its pins"
    lines = _commands(yaml.safe_load(text)["run"])
    pin = PIN + EXTRA_PINS.get(path, "")
    for line, what in [(UV, "install uv"), (VENV, "create the Python 3.11 venv"),
                       (pin, "install the pinned set"), (CHECK, "assert Python and JAX (no pipe)")]:
        assert lines.count(line) == 1, f"{path}: {what} once, on its own line"
    assigns = [line for line in lines for _ in PY_ASSIGN.finditer(line)]
    assert assigns == [PY], f"{path}: PY is assigned once, to the venv's python: {assigns}"
    assert sum(bool(FREEZE.fullmatch(line)) for line in lines) == 1, f"{path}: record the venv's freeze"
    first_import = next(i for i, line in enumerate(lines) if "import jax" in line)
    order = [lines.index(line) for line in (UV, VENV, PY, pin, CHECK)]
    assert order == sorted(order) and order[-1] == first_import, f"{path}: order is venv, install, check, use"
    others = [line for line in lines if re.search(r"pip install.*\bjax", line) and line != pin]
    assert not others, f"{path}: a second JAX install could replace 0.10.2: {others}"
    image_python = [line for line in lines if IMAGE_PYTHON.search(line) and line not in (UV, PY)]
    assert not image_python, f"{path}: the image's python 3.10 runs JAX 0.4.33; use \"$PY\": {image_python}"
    runs = [(line, m) for line in lines for m in PYTEST.finditer(line)]
    assert runs, f"{path}: no pytest run found"
    stray = [line for line, m in runs if not line[:m.start()].endswith('"$PY" -m ')]
    assert not stray, f"{path}: run pytest as \"$PY\" -m pytest: {stray}"


@pytest.mark.parametrize("pip_jax", [None, "", "jax[cuda12]>0.10.1,<0.10.3"])
def test_multinode_launcher_refuses_python_310_jobs(tmp_path, pip_jax):
    output = tmp_path / "rendered"
    args = [sys.executable, str(ROOT / "scripts/diagnostics/prepare_multinode_launch.py"),
            "--tooling-sha", "a" * 40, "--mirror", str(tmp_path / "mirror"),
            "--artifact-root", str(tmp_path / "artifacts"), "--output", str(output)]
    if pip_jax is not None:
        args.extend(["--pip-jax", pip_jax])
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    for message in ("#1429", "Python 3.10", "Python >= 3.11", "JAX >= 0.10.2",
                    "scripts/vessl_gpu_suite.yaml", "uv venv", "$PY"):
        assert message in result.stderr
    assert not output.exists(), "refuse before writing any launch artifacts"
