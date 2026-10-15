"""Zero-depth CPML retains its electric backing in the distributed runner."""
import os
from pathlib import Path
import subprocess
import sys


def test_zero_depth_absorber_probe_matches_single_device(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS="cpu", JAX_ENABLE_X64="false",
               XLA_FLAGS="--xla_force_host_platform_device_count=2", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-m", "tests._zero_depth_wall_witness", str(tmp_path / "wall.json")],
                   cwd=root, env=env, check=True, capture_output=True, text=True, timeout=180)
