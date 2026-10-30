"""Invalid point-probe directions fail before building or modifying a simulation."""
import subprocess
import sys
from pathlib import Path

import pytest

from rfx.probes.msl_wave_decomp import register_msl_wave_probes


def test_invalid_direction_raises_value_error():
    with pytest.raises(ValueError, match=r"^direction must be '\+x' or '-x', got '-y'$"):
        register_msl_wave_probes(
            None, feed_x=0.0, direction='-y', y_centre=0.0,
            z_ez=0.0, z_hy=0.0)


def test_invalid_direction_is_rejected_with_python_optimization():
    result = subprocess.run(
        [sys.executable, '-O', '-c', """
from rfx.probes.msl_wave_decomp import register_msl_wave_probes
try:
    register_msl_wave_probes(None, feed_x=0.0, direction='-y',
                             y_centre=0.0, z_ez=0.0, z_hy=0.0)
except ValueError as error:
    if str(error) != "direction must be '+x' or '-x', got '-y'":
        raise
else:
    raise SystemExit('invalid direction was accepted')
"""], capture_output=True, text=True, check=False,
        cwd=Path(__file__).resolve().parents[3])
    assert result.returncode == 0, result.stdout + result.stderr
