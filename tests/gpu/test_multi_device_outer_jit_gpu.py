"""Tracker 1598's real CUDA entry points; each case gets a fresh process."""
import os
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.multi_gpu


@pytest.fixture(scope="module")
def cuda_devices():
    devices = [d for d in jax.devices() if d.platform == "gpu" and
               ("cuda" in str(d).lower() or "nvidia" in d.device_kind.lower())]
    if len(devices) < 2:
        pytest.skip("needs two or more CUDA GPUs in one node; fewer than two are visible")
    return devices


def worker(case, tmp_path, *, count=2, flags=None):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / f"{case}.npy"
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    if flags is not None:
        env["XLA_FLAGS"] = flags
    result = subprocess.run([sys.executable, str(root / "tests/gpu/_outer_jit_worker.py"),
                             case, "--devices", str(count), "--out", str(out)],
                            cwd=root, env=env, text=True, capture_output=True, timeout=1500)
    print(result.stdout, end="")
    print(result.stderr, end="")
    assert result.returncode == 0, result.stdout + result.stderr
    if "NOT_REACHABLE:" in result.stdout:
        pytest.skip(result.stdout.split("NOT_REACHABLE:", 1)[1].strip())
    return out


@pytest.fixture(scope="module")
def plain_gradient(cuda_devices, tmp_path_factory):
    return np.load(worker("grad", tmp_path_factory.mktemp("plain_gradient")))


@pytest.mark.parametrize("case", ["forward", "grad"])
def test_G_a_plain(cuda_devices, tmp_path, case, request):
    if case == "grad":
        assert np.isfinite(request.getfixturevalue("plain_gradient")).all()
    else:
        worker(case, tmp_path)


@pytest.mark.parametrize("case", ["jit_forward", "jit_grad"])
def test_G_b_refused(cuda_devices, tmp_path, case):
    worker(case, tmp_path)


def within_cross_trace_bar(baseline, compiled, ulps=9):
    peak = np.float32(np.abs(baseline).max())
    difference = np.abs(baseline.astype(np.float64) - compiled.astype(np.float64)).max()
    return bool(difference <= ulps * float(np.spacing(peak))), float(difference / np.spacing(peak))


def test_G_c_flag_gradient_within_cross_trace_bar(cuda_devices, tmp_path, plain_gradient):
    baseline = plain_gradient
    compiled = np.load(worker("off_grad", tmp_path,
                              flags="--xla_gpu_enable_command_buffer="))
    assert baseline.dtype == compiled.dtype and baseline.shape == compiled.shape
    # The enclosing jit compiles a different program than the plain call; float32 rounding then
    # differs (inferred from #1905: 1 ULP of the peak at 4 cells, plain runs repeat bit-equal).
    # Cross-trace rule, rfx CLAUDE.md: the per-step 9 ULP is chosen on purpose, tighter than the
    # 1e-4 of the peak the rule allows for this record-summed gradient.
    ok, ulps = within_cross_trace_bar(baseline, compiled)
    assert ok, f"jitted gradient differs from the plain one by {ulps:.2f} ULP of the peak"


@pytest.mark.parametrize("case", ["checkpoint", "vmap", "uniform_run",
                                  "jit_checkpoint", "jit_vmap", "jit_uniform_run"])
def test_G_d_entries(cuda_devices, tmp_path, case):
    worker(case, tmp_path)


@pytest.mark.parametrize("case", ["grad", "jit_grad"])
def test_G_e_three_devices(cuda_devices, tmp_path, case):
    if len(cuda_devices) < 3:
        pytest.skip("needs three CUDA GPUs in one node; only two are visible")
    worker(case, tmp_path, count=3)
