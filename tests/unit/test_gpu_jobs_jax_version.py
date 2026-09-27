"""The maintained GPU jobs run JAX 0.10.2 in a Python 3.11 venv, never the VESSL image's python.

PI decision 2026-09-25 ("move if no problems"): GPU work moves from JAX 0.6.2 on the image's
Python 3.10 to a uv venv with Python 3.11 and JAX 0.10.2 (which needs Python >= 3.11). On VESSL
369367265580 (A6000) that venv gave the same outcomes as 0.6.2 on the lane's gpu selection.
The image these jobs use, nvcr.io/nvidia/jax:24.10-py3, ships Python 3.10 with JAX 0.4.33,
whose XLA aborts while compiling forward(distributed=True) on
CHECK_LE(common_utilization, producer_output_utilization) in its fusion cost model (#1252;
the check is gone from JAX 0.4.36). A job that loses the install, or calls the image's `python`
where it meant the venv's, runs 0.4.33 with no message. So each job creates the venv, installs
the pinned set on one line, asserts the Python and JAX it got on its own line before anything
else imports JAX, and runs everything through "$PY"; this file checks those lines are there.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UV = 'python -m pip install -q "uv==0.12.19"'
VENV = "uv venv -q --python 3.11.16 /tmp/venv-py311"
PY = "PY=/tmp/venv-py311/bin/python"
# The pins of the environment that ran VESSL 369367265580.
PIN = ('uv pip install -q --python "$PY" "jax[cuda12]==0.10.2" "numpy==2.4.6" "scipy==1.17.1" '
       '"h5py==3.16.0" "matplotlib==3.11.2" "ml_dtypes==0.6.0" "pyyaml==6.0.3" "optax==0.2.8" '
       '"pillow==12.3.0" "pytest==9.1.1" "pytest-split==0.11.0"')
CHECK = ('"$PY" -c "import jax, sys; assert sys.version_info[:2] == (3, 11), sys.version; '
         "assert jax.__version__ == '0.10.2', jax.__version__; "
         "assert jax.default_backend() == 'gpu', jax.default_backend()\"")
JOBS = ["scripts/vessl_gpu_suite.yaml",                 # per release and after GPU-touching merges
        "scripts/ops/render_gpu_suite_shards.py",       # weekly sharded suite (template)
        "scripts/vessl_validation_lane_a6000.yaml",     # weekly A6000 lane (validation.yml cron)
        "scripts/vessl_crossval_ladder.yaml"]           # monthly crossval-ladder lane (crossval-ladder.yml cron)


def _job_text(path):
    """The job as VESSL gets it: a YAML file, or the TEMPLATE a renderer script formats."""
    if path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("renderer", ROOT / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TEMPLATE
    return (ROOT / path).read_text(encoding="utf-8")


def _commands(text):
    """Uncommented, stripped lines: a commented-out or chained install does not count."""
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


@pytest.mark.parametrize("path", JOBS)
def test_gpu_job_runs_jax_0102_in_a_python_311_venv(path):
    text = _job_text(path)
    assert "nvcr.io/nvidia/jax:24.10-py3" in text, "image changed: re-decide the venv and its pins"
    lines = _commands(text)
    for line, what in [(UV, "install uv"), (VENV, "create the Python 3.11 venv"), (PY, "name its python"),
                       (PIN, "install the pinned set"), (CHECK, "assert Python and JAX (no pipe)")]:
        assert lines.count(line) == 1, f"{path}: {what} once, on its own line"
    first_import = next(i for i, line in enumerate(lines) if "import jax" in line)
    order = [lines.index(line) for line in (UV, VENV, PY, PIN, CHECK)]
    assert order == sorted(order) and order[-1] == first_import, f"{path}: order is venv, install, check, use"
    others = [line for line in lines if re.search(r"pip install.*\bjax", line) and line != PIN]
    assert not others, f"{path}: a second JAX install could replace 0.10.2: {others}"
    image_python = [line for line in lines if re.search(r"(^|[\s;{(|&])python3?\s", line) and line != UV]
    assert not image_python, f"{path}: the image's python 3.10 runs JAX 0.4.33; use \"$PY\": {image_python}"


def test_multinode_launcher_defaults_to_jax_062():
    """The multi-node probe still installs over the image's python 3.10, where 0.10.2 cannot go."""
    requirements = pytest.importorskip("packaging.requirements")
    source = (ROOT / "scripts/diagnostics/prepare_multinode_launch.py").read_text(encoding="utf-8")
    match = re.search(r'"--pip-jax", default="([^"]*)"', source)
    assert match, "the launcher's --pip-jax default is not a literal string"
    requirement = requirements.Requirement(match[1])
    assert requirement.name == "jax" and "cuda12" in requirement.extras
    releases = ["0.4.33", "0.4.35", "0.4.36", "0.6.1", "0.6.2", "0.7.0"]
    assert list(requirement.specifier.filter(releases)) == ["0.6.2"], match[1]
    # The VESSL experiment CLI splits each hyperparameter on every '='.
    assert "=" not in match[1]
