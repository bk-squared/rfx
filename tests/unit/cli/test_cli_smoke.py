"""Fast CPU contracts for the installed CLI entry points and their artifacts."""

import importlib
import json
from pathlib import Path
import time
import tomllib

import h5py
import jax
import numpy as np
import pytest

from rfx import diagnostics
from rfx.cli import main


@pytest.fixture
def tiny_config(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text("""\
frequency: {freq_max: 10000000000.0}
domain: [0.008, 0.008, 0.008]
dx: 0.002
boundary: pec
sources:
  - type: source
    position: [0.004, 0.004, 0.004]
    component: ez
    amplitude_kind: field
    waveform: {type: gaussian_pulse, f0: 5000000000.0, bandwidth: 0.8}
probes:
  - position: [0.004, 0.004, 0.004]
    component: ez
execution: {n_steps: 4, compute_s_params: false}
""", encoding="utf-8")
    return path


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "{run,plot,export-geometry}" in capsys.readouterr().out


def test_export_geometry_writes_declared_shapes(tiny_config, tmp_path):
    with tiny_config.open("a", encoding="utf-8") as config:
        config.write("""\
materials:
  dielectric: {eps_r: 2.0}
geometry:
  - shape: box
    bounds: [[0.002, 0.002, 0.002], [0.004, 0.004, 0.004]]
    material: dielectric
""")
    output = tmp_path / "geometry.json"
    with jax.default_device(jax.devices("cpu")[0]):
        assert main(["export-geometry", str(tiny_config), "--output", str(output)]) == 0
    geometry = json.loads(output.read_text(encoding="utf-8"))
    assert geometry["geometry"] == [{
        "type": "Box",
        "material": "dielectric",
        "bbox": [[0.002, 0.002, 0.002], [0.004, 0.004, 0.004]],
    }]


def test_run_writes_probe_on_cpu(tiny_config, tmp_path):
    output = tmp_path / "result.h5"
    started = time.perf_counter()
    with jax.default_device(jax.devices("cpu")[0]):
        assert main(["run", str(tiny_config), "--output", str(output)]) == 0
    assert time.perf_counter() - started < 30.0
    assert output.is_file()
    with h5py.File(output, "r") as result:
        # Four cells per axis have five bounding nodes with PEC walls.
        assert tuple(result["input"].attrs["grid_shape"]) == (5, 5, 5)
        probe = result["output/time_series"][:]
    assert probe.shape == (4, 1)
    assert np.isfinite(probe).all()
    assert np.any(probe != 0)


def test_diagnostics_exits_successfully(capsys):
    # Diagnostics has no flags: argv is ignored, so exercise its real default.
    with jax.default_device(jax.devices("cpu")[0]):
        assert diagnostics.main([]) == 0
    assert "Summary: All critical checks passed." in capsys.readouterr().out


@pytest.mark.source_scan
def test_project_scripts_resolve_to_callables():
    project = Path(__file__).resolve().parents[3] / "pyproject.toml"
    with project.open("rb") as config:
        scripts = tomllib.load(config)["project"]["scripts"]
    assert scripts == {
        "rfx": "rfx.cli:main",
        "rfx-dashboard": "rfx.dashboard.cli:main",
        "rfx-diagnose": "rfx.diagnostics:main",
    }
    for target in scripts.values():
        module, name = target.split(":")
        assert callable(getattr(importlib.import_module(module), name)), target
