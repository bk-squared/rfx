"""CSV gain uses the full-sphere IEEE/realized normalization (#1369)."""
import numpy as np
import pytest

from rfx.antenna import antenna_gain_dB, _total_radiated_power
from rfx.farfield import FarFieldResult
from rfx.io import export_radiation_pattern


@pytest.mark.parametrize("power_kind", ["ieee", "scalar", "per_frequency"])
def test_short_dipole_export_gain(tmp_path, power_kind):
    theta = np.linspace(0, np.pi, 181)
    phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    amplitude = np.array([1.0, 3.0])[:, None, None]
    electric = amplitude * np.sin(theta)[None, :, None] * np.ones((1, 1, len(phi)), complex)
    ff = FarFieldResult(electric, np.zeros_like(electric), theta, phi, np.array([3e9, 5e9]))
    power = _total_radiated_power(ff)
    input_power = {"ieee": None, "scalar": 2 * power[1], "per_frequency": 2 * power}[power_kind]
    path = tmp_path / "dipole.csv"
    export_radiation_pattern(path, ff, freq_idx=1, input_power=input_power)
    csv = np.genfromtxt(path, delimiter=",", names=True)
    expected = antenna_gain_dB(ff, input_power=input_power)[1].ravel()
    # %.6e CSV formatting has <= 5e-7 relative rounding error.
    np.testing.assert_allclose(csv["gain_dBi"], expected, rtol=5e-7, atol=5e-7)
    peak = np.max(csv["gain_dBi"])
    analytic = 10 * np.log10(1.5 if input_power is None else 0.75)
    # For sin^3(theta), endpoint first derivatives vanish: the trapezoid
    # power integral has relative error h^4/80 + O(h^6), ~1.2e-9 here.
    # In dB that is < 1e-8; CSV rounding dominates, so allow 1e-6 dB.
    assert abs(peak - analytic) < 1e-6
    print(f"\n{power_kind}: exported_peak_dBi={peak:.9f}, analytic_dBi={analytic:.9f}")
