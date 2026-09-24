"""The maintained GPU jobs install JAX 0.6.2 over the VESSL image's own JAX.

PI decision 2026-09-24: GPU work runs the JAX that CI's Python 3.10 lane resolves, 0.6.2.
The image these jobs use, nvcr.io/nvidia/jax:24.10-py3, ships JAX 0.4.33, whose XLA aborts
while compiling forward(distributed=True) on
CHECK_LE(common_utilization, producer_output_utilization) in its fusion cost model (#1252;
the check is gone from JAX 0.4.36). A job that loses the install line falls back to 0.4.33
with no message, so the line is checked here.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIN = 'python -m pip install -q "jax[cuda12]==0.6.2"'


@pytest.mark.parametrize("path", ["scripts/vessl_gpu_suite.yaml",
                                  "scripts/ops/render_gpu_suite_shards.py"])
def test_gpu_suite_installs_jax_062_before_importing_it(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/jax:24.10-py3" in text, "image changed: re-decide whether the pin is needed"
    assert text.count(PIN) == 1, f"{path} must install JAX 0.6.2 exactly once"
    assert text.index(PIN) < text.index("import jax"), f"{path} imports JAX before installing 0.6.2"


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
