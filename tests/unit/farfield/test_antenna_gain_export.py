"""CSV gain uses the full-sphere IEEE/realized normalization (#1369)."""
import numpy as np
import pytest

from rfx.antenna import (
    ETA_0, antenna_efficiency, antenna_gain, antenna_gain_dB,
    _total_radiated_power,
)
from rfx.farfield import FarFieldResult, directivity
from rfx.io import export_radiation_pattern


def _dipole(theta, phi):
    electric = np.sin(theta)[None, :, None] * np.ones((1, 1, len(phi)), complex)
    return FarFieldResult(electric, np.zeros_like(electric), theta, phi, np.array([3e9]))


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


_PARTIAL_GRIDS = [
    pytest.param(np.linspace(0, np.pi, 181), np.array([0.0, np.pi / 2]), id="E-H-cuts"),
    pytest.param(np.linspace(0, np.pi / 2, 91),
                 np.linspace(0, 2 * np.pi, 24, endpoint=False), id="upper-hemisphere"),
    pytest.param(np.linspace(0, np.pi, 181), np.linspace(0, np.pi, 181), id="half-phi"),
]


@pytest.mark.parametrize("theta,phi", _PARTIAL_GRIDS)
@pytest.mark.parametrize("metric", ["power", "gain", "gain_dB", "efficiency", "directivity", "export"])
def test_partial_sphere_refused(tmp_path, theta, phi, metric):
    # These cuts omit radiated power. The old export reported 4.7712,
    # 4.7147 and 4.7472 dBi respectively for the same 1.7609 dBi dipole.
    ff = _dipole(theta, phi)
    calls = {
        "power": lambda: _total_radiated_power(ff),
        "gain": lambda: antenna_gain(ff),
        "gain_dB": lambda: antenna_gain_dB(ff),
        # Supplying input power cannot supply the missing P_rad numerator.
        "efficiency": lambda: antenna_efficiency(ff, input_power=1.0),
        "directivity": lambda: directivity(ff),
        "export": lambda: export_radiation_pattern(tmp_path / "partial.csv", ff),
    }
    with pytest.raises(ValueError, match="full-sphere coverage") as exc:
        calls[metric]()
    message = str(exc.value)
    assert "theta must span [0, pi]" in message and "half-step" in message
    assert "phi must span a full turn" in message and "exactly one sample" in message
    assert "Realized gain with input_power needs no full-sphere coverage" in message
    assert not (tmp_path / "partial.csv").exists()


@pytest.mark.parametrize("phi", [np.array([0.0]), np.linspace(0, 2 * np.pi, 24, endpoint=False)],
                         ids=["axisymmetric-single-cut", "full-sphere"])
def test_short_dipole_absolute_gain(tmp_path, phi):
    ff = _dipole(np.linspace(0, np.pi, 181), phi)
    analytic = 10 * np.log10(1.5)
    assert abs(antenna_gain_dB(ff).max() - analytic) < 1e-8
    assert abs(directivity(ff)[0] - analytic) < 1e-8
    path = tmp_path / "dipole.csv"
    export_radiation_pattern(path, ff)
    csv = np.genfromtxt(path, delimiter=",", names=True)
    assert abs(csv["gain_dBi"].max() - analytic) < 1e-6


@pytest.mark.parametrize("theta,phi", _PARTIAL_GRIDS)
@pytest.mark.parametrize("per_frequency", [False, True])
def test_partial_sphere_realized_gain(tmp_path, monkeypatch, theta, phi, per_frequency):
    # The analytic full-sphere dipole power is 4*pi/(3*eta_0), regardless
    # of which directions were sampled. Twice that input power gives 0.75
    # peak realized gain (-1.249387 dBi), with no angular integration.
    ff = _dipole(theta, phi)
    input_power = 8 * np.pi / (3 * ETA_0)
    if per_frequency:
        input_power = np.array([input_power])

    def integral_is_forbidden(*args, **kwargs):
        pytest.fail("realized gain must not call the radiated-power integral")

    monkeypatch.setattr("rfx.antenna._total_radiated_power", integral_is_forbidden)
    gain = antenna_gain(ff, input_power=input_power)
    expected = np.broadcast_to(0.75 * np.sin(theta)[None, :, None] ** 2, gain.shape)
    np.testing.assert_allclose(gain, expected, rtol=1e-14)
    assert abs(antenna_gain_dB(ff, input_power=input_power).max() - 10 * np.log10(0.75)) < 1e-14
    path = tmp_path / "realized.csv"
    export_radiation_pattern(path, ff, input_power=input_power)
    csv = np.genfromtxt(path, delimiter=",", names=True)
    np.testing.assert_allclose(csv["gain_dBi"],
                               10 * np.log10(np.maximum(expected.ravel(), 1e-30)),
                               rtol=5e-7, atol=5e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("grid", ["closed", "open", "midpoint", "nonuniform", "single"])
def test_valid_sphere_keeps_legacy_bits(dtype, grid):
    # Preserve even the historical double-weighted seam on a closed phi
    # grid. Changing that quadrature would be a separate numerical change.
    theta = np.linspace(0, np.pi, 37)
    phi = np.linspace(0, 2 * np.pi, 24, endpoint=grid == "closed")
    if grid == "midpoint":
        theta = (np.arange(37) + 0.5) * np.pi / 37
        phi += 0.31  # A full turn need not start at zero.
    elif grid == "nonuniform":
        theta = np.pi * np.linspace(0, 1, 37) ** 1.3
        phi = 2 * np.pi * np.linspace(0, 1, 24) ** 1.2
    elif grid == "single":
        phi = np.array([0.7])
    theta, phi = theta.astype(dtype), phi.astype(dtype)
    rng = np.random.default_rng(1369)
    shape = (2, len(theta), len(phi))
    et = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64 if dtype == np.float32 else np.complex128)
    ep = et * (0.3 + 0.7j)
    ff = FarFieldResult(et, ep, theta, phi, np.array([3e9, 5e9]))
    power = np.abs(et) ** 2 + np.abs(ep) ** 2

    def old_integral(u):
        integrand = u * np.sin(theta)[None, :, None]
        dth = np.gradient(theta)
        dph = np.gradient(phi) if len(phi) > 1 else np.array([2 * np.pi])
        return np.sum(integrand * dth[None, :, None] * dph[None, None, :], axis=(1, 2))

    intensity = power / (2.0 * ETA_0)
    expected_power = old_integral(intensity)
    expected_gain = 4.0 * np.pi * intensity / expected_power[:, None, None]
    expected_directivity = 10 * np.log10(4 * np.pi * np.max(power, axis=(1, 2)) / old_integral(power))
    np.testing.assert_array_equal(_total_radiated_power(ff), expected_power)
    np.testing.assert_array_equal(antenna_gain(ff), expected_gain)
    np.testing.assert_array_equal(antenna_gain_dB(ff), 10 * np.log10(expected_gain))
    np.testing.assert_array_equal(directivity(ff), expected_directivity)
    np.testing.assert_array_equal(antenna_efficiency(ff, 0.25), expected_power / 0.25)


@pytest.mark.parametrize("theta,phi", [
    pytest.param(np.linspace(0.01, np.pi - 0.01, 181), np.array([0.0]), id="polar-gaps"),
    pytest.param(np.array([np.pi / 2]), np.array([0.0]), id="one-theta"),
    pytest.param(np.linspace(0, np.pi, 37),
                 np.arange(23) * 2 * np.pi / 24, id="two-missing-phi-steps"),
])
def test_sphere_coverage_step_limits(theta, phi):
    with pytest.raises(ValueError, match="full-sphere coverage"):
        _total_radiated_power(_dipole(theta, phi))
