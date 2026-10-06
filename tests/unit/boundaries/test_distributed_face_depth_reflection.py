"""An asymmetric absorber must survive each two-device builder in a PEC guide."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_asymmetric_face_reflection_on_two_devices(tmp_path):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "reflections.json"
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS="cpu", JAX_ENABLE_X64="false",
               XLA_FLAGS="--xla_force_host_platform_device_count=2",
               PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-m", "tests._distributed_absorber_witness", str(output)],
                   cwd=root, env=env, check=True, capture_output=True, text=True, timeout=180)
    returns = {key: np.array(value) for key, value in json.loads(output.read_text())["reflection_db"].items()}
    assert_reflections(returns)


def assert_reflections(returns):
    reference = returns["uniform_one"]
    amplitudes = 10 ** (reference / 20)
    margin = reference[0] - reference[1]
    assert np.all(margin > 35), ("single-device depth advantage", margin)
    # Measured float32 |delta R| floor: 7.075e-8 across all four paths;
    # uniform/graded two-vs-one floors are 5.022e-8 / 6.586e-8. A 2e-7
    # absolute bar resolves the ~-100 dB deep echo without dividing by it.
    for lane, values in returns.items():
        np.testing.assert_allclose(10 ** (values / 20), amplitudes, atol=2e-7, rtol=0,
                                   err_msg=f"{lane}: per-face plane-wave reflection")
        assert np.all(values[0] - values[1] >= margin - 0.25), (
            f"{lane}: 16-layer face lost single-device depth advantage", values)
