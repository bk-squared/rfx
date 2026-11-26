"""Physics pin: a settled short line's voltage node cannot fail settling."""

import numpy as np
import pytest

from tests.unit.sparams.test_coaxial_line_settling import _run


@pytest.mark.slow_physics
def test_short_line_voltage_null_on_a_judged_record():
    """The shortest echo-bound-accepted record is 1,668 steps.

    Historically, settling was -41.5 dB at 1,500 steps and -59.1 dB at
    1,650 steps; both records are too short for the current echo bound.
    """
    result = _run(1668, termination='short')
    assert result.settling_db <= -40., result.settling_db
    assert result.settling_witness['status'] == 'pass'
    target = float(np.asarray(result.freqs)[3])
    assert np.isclose(target, 5.5e9, rtol=1e-7, atol=0.)
    row = next(row for row in result.settling_witness['dropped_per_bin']
               if row['freq_hz'] == target)
    assert 'coax/plane10/V' in row['channels'], row
    assert row['count'] == len(row['channels'])
