"""Physics pin: a settled short line's voltage node cannot fail settling."""

import numpy as np
import pytest

from tests.unit.sparams.test_coaxial_line_settling import _run


@pytest.mark.slow_physics
def test_short_line_voltage_null_at_1650_steps():
    """The 1650-step choice leaves settling margin below -40 dB because settling measured earlier was -41.5 dB at 1500 steps versus -59.1 dB at 1650 steps."""
    result = _run(1650, termination='short')
    assert result.settling_db <= -40., result.settling_db
    assert result.settling_witness['status'] == 'pass'
    target = float(np.asarray(result.freqs)[3])
    assert np.isclose(target, 5.5e9, rtol=1e-7, atol=0.)
    row = next(row for row in result.settling_witness['dropped_per_bin']
               if row['freq_hz'] == target)
    assert 'coax/plane10/V' in row['channels'], row
    assert row['count'] == len(row['channels'])
