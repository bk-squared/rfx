"""The root conftest gives the suite two CPU devices whatever XLA_FLAGS it inherits.

nvcr.io/nvidia/jax presets XLA_FLAGS with GPU scheduler flags. The conftest
used ``os.environ.setdefault``, so under that image the device-count flag was
never set: one device, and 39 two-device tests of one fast-suite shard failed
(2026-10-14). Each row loads the real conftest in a fresh process with a preset
XLA_FLAGS and counts the devices JAX then sees.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]

PROBE = (
    "import os, runpy, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "runpy.run_path(os.path.join(sys.argv[1], 'conftest.py'))\n"
    "import jax\n"
    "print(len(jax.devices()))\n"
    "print(os.environ['XLA_FLAGS'])\n"
)


@pytest.mark.parametrize(
    ("preset", "devices", "kept"),
    [
        (None, 2, ""),
        ("", 2, ""),
        ("--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false", 2,
         "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false"),
        ("--xla_force_host_platform_device_count=3", 3, "--xla_force_host_platform_device_count=3"),
    ],
    ids=["unset", "empty", "image-preset", "caller-count"],
)
def test_conftest_device_count_survives_preset_flags(preset, devices, kept):
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["JAX_PLATFORMS"] = "cpu"
    if preset is not None:
        env["XLA_FLAGS"] = preset
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(REPO)],
        env=env, cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    count, flags = out.stdout.strip().splitlines()[-2:]
    assert int(count) == devices
    assert flags.startswith(kept)
