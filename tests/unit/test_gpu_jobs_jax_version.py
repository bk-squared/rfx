"""The maintained GPU jobs install JAX 0.6.2 over the VESSL image's own JAX, and check it.

PI decision 2026-09-24: GPU work runs the JAX that CI's Python 3.10 lane resolves, 0.6.2.
The image these jobs use, nvcr.io/nvidia/jax:24.10-py3, ships JAX 0.4.33, whose XLA aborts
while compiling forward(distributed=True) on
CHECK_LE(common_utilization, producer_output_utilization) in its fusion cost model (#1252;
the check is gone from JAX 0.4.36). A job that loses the install falls back to 0.4.33 with
no message, so each job installs 0.6.2 on its own line and then asserts the version it got,
on its own line, before anything else imports JAX; this file checks both lines are there.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIN = 'python -m pip install -q "jax[cuda12]==0.6.2"'
CHECK = "python -c \"import jax; assert jax.__version__ == '0.6.2', jax.__version__\""
JOBS = ["scripts/vessl_gpu_suite.yaml",                 # per release and after GPU-touching merges
        "scripts/ops/render_gpu_suite_shards.py",       # weekly sharded suite (template)
        "scripts/vessl_validation_lane_a6000.yaml"]     # weekly A6000 lane (validation.yml cron)


def _commands(text):
    """Uncommented, stripped lines: a commented-out or chained install does not count."""
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


@pytest.mark.parametrize("path", JOBS)
def test_gpu_job_installs_and_asserts_jax_062(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/jax:24.10-py3" in text, "image changed: re-decide whether the pin is needed"
    lines = _commands(text)
    assert lines.count(PIN) == 1, f"{path}: install JAX 0.6.2 once, on its own line"
    assert lines.count(CHECK) == 1, f"{path}: assert the installed version on its own line (no pipe)"
    first_import = next(i for i, line in enumerate(lines) if "import jax" in line)
    assert lines.index(PIN) < lines.index(CHECK) == first_import, f"{path}: order is install, check, use"
    others = [line for line in lines if re.search(r"pip install.*\bjax", line) and line != PIN]
    assert not others, f"{path}: a second JAX install could replace 0.6.2: {others}"


def test_multinode_launcher_defaults_to_jax_062():
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
