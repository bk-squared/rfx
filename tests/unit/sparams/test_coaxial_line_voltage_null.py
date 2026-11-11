"""Physics pin: a settled short line's voltage node cannot fail settling."""

import numpy as np
import pytest

from tests.unit.sparams.test_coaxial_line_settling import _run


@pytest.mark.slow_physics
def test_short_line_voltage_null_at_1650_steps():
    # 1650 steps: the matched feed sheet keeps the line's permittivity, so the
    # short line rings down 15.9 dB per round trip (was 18.3); 1650 restores the
    # settling precondition's ~6 dB margin to -40 dB (record 20261107-coax-oneport-floor).
    result = _run(1650, termination='short')
    assert result.settling_db <= -40., result.settling_db
    assert result.settling_witness['status'] == 'pass'
    # The requested 5.5 GHz bin can be rounded by the fixture's JAX float32.
    target = float(np.asarray(result.freqs)[3])
    assert np.isclose(target, 5.5e9, rtol=1e-7, atol=0.)
    row = next(row for row in result.settling_witness['dropped_per_bin']
               if row['freq_hz'] == target)
    assert 'coax/plane10/V' in row['channels'], row
    assert row['count'] == len(row['channels'])
