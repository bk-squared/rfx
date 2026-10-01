"""Pin the #1429 floor across metadata, runtime, docs tooling and CI."""

import os
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
import pytest
import yaml

from rfx import _runtime_floor as floor
from rfx import diagnostics

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_floor_contract():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert floor.MIN_PYTHON == (3, 11)
    assert floor.MIN_JAX == "0.10.2"
    assert config["project"]["requires-python"] == ">=3.11"
    assert config["tool"]["mypy"]["python_version"] == "3.11"
    deps = {r.name: r for r in map(Requirement, config["project"]["dependencies"])}
    docs = {
        r.name: r for r in map(Requirement, (
            line for line in (ROOT / "scripts/requirements-public-docs.txt").read_text().splitlines()
            if line and not line.startswith("#")
        ))
    }
    workflow = yaml.safe_load((ROOT / ".github/workflows/docs-consistency.yml").read_text())
    install = next(step["run"] for step in workflow["jobs"]["docs-consistency"]["steps"]
                   if step["name"] == "Install package with dev dependencies")
    for name in ("jax", "jaxlib"):
        assert str(deps[name].specifier) == f">={floor.MIN_JAX}"
        assert str(docs[name].specifier) == f"=={floor.MIN_JAX}"
        assert f'"{name}=={floor.MIN_JAX}"' in install


@pytest.mark.parametrize("python,jax,jaxlib", [
    ((3, 10), "0.10.2", "0.10.2"),
    ((3, 11), "0.6.2", "0.10.2"),
    ((3, 11), "0.10.2", "0.6.2"),
    ((3, 11), "0.10.2.dev20260101", "0.10.2"),
    ((3, 11), "0.10.2rc1", "0.10.2"),
    ((3, 11), "unknown", "0.10.2"),
])
def test_check_refuses_unsupported_runtime(python, jax, jaxlib):
    with pytest.raises(ImportError) as caught:
        floor.check(python, jax, jaxlib)
    message = str(caught.value)
    for expected in ("#1429", ".".join(map(str, python)), jax, jaxlib,
                     "Python >= 3.11", "jax >= 0.10.2", "jaxlib >= 0.10.2",
                     "venv", "jax==0.10.2 jaxlib==0.10.2",
                     "scripts/vessl_gpu_suite.yaml", "rfx 1.8.x"):
        assert expected in message


@pytest.mark.parametrize("version", ["0.10.2", "0.10.2+cuda12", "0.10.2.post1",
                                    "0.10.3.dev20260101", "0.11.0", "1.0.0"])
def test_check_accepts_supported_runtime(version):
    floor.check((3, 11), version, version)


@pytest.mark.parametrize("package", ["jax", "jaxlib"])
def test_import_refuses_old_jax_before_loading_rfx_modules(package):
    code = f'''
import sys
import jax
import jaxlib
{package}.__version__ = "0.6.2"
try:
    import rfx
except ImportError as exc:
    assert "#1429" in str(exc), str(exc)
    assert "{package} 0.6.2" in str(exc), str(exc)
    assert "rfx.grid" not in sys.modules
else:
    raise AssertionError("import rfx accepted an unsupported runtime")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT), "JAX_PLATFORMS": "cpu"},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("package", ["jax", "jaxlib"])
def test_diagnostics_reports_old_jax_as_failure(monkeypatch, package):
    monkeypatch.setattr(sys.modules[package], "__version__", "0.6.2")
    report = diagnostics._Report()
    diagnostics._check_jax(report)
    assert report.critical_failures == 1
    assert "[FAIL] jax / jaxlib" in str(report)
    assert "#1429" in str(report)
