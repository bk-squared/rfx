"""Coaxial TEM plane offsets, higher-mode decay and reflection fits."""
import jax
import numpy as np
import pytest

from rfx.errors import RfxRefusal, RfxUnsupported
from rfx.measurement.coax_records import coaxial_line_reflection_from_plane_voltages
from rfx.model import coax_planes


SCENE = dict(inner_radius=1e-3, outer_radius=6e-3, eps_r=2.1, dz=1e-3,
             freq_max=4.5e9, freq_min=4.5e8, kappa=0.54, tolerance=1e-2)
SMA = dict(inner_radius=0.635e-3, outer_radius=2.055e-3, eps_r=2.1, dz=2.5e-4,
           freq_max=18e9, freq_min=1.8e9, kappa=0.54, tolerance=1e-2)
START_FALLBACK = dict(SCENE, dz=5e-4, freq_min=1.5e9, kappa=1.09, tolerance=1e-3)
CLEARANCE_FALLBACK = dict(SMA, dz=6.25e-5, kappa=0.27)
DEFAULT_ROWS = [
    pytest.param(SCENE, 1e-3, (33, 4, 17, 17, 17, 6, 105, False, False, 157), id="scene-1mm"),
    pytest.param(SCENE, 5e-4, (65, 8, 32, 34, 34, 12, 204, False, False, 313), id="scene-0.5mm"),
    pytest.param(SCENE, 2.5e-4, (129, 16, 63, 68, 68, 24, 402, False, False, 625), id="scene-0.25mm"),
    pytest.param(SMA, 2.5e-4, (49, 4, 18, 26, 26, 9, 133, False, False, 300), id="sma-0.25mm"),
    pytest.param(SMA, 1.25e-4, (98, 8, 35, 51, 51, 17, 259, False, False, 599), id="sma-0.125mm"),
    pytest.param(SMA, 6.25e-5, (195, 16, 68, 102, 102, 33, 511, False, False, 1198), id="sma-0.0625mm"),
]
DECLARED_ROWS = [
    pytest.param(SCENE, 1e-3, 1.5e9, (25, 4, 10, 11, 11, 6, 91, False, False, 157), id="scene-1mm"),
    pytest.param(SCENE, 5e-4, 1.5e9, (49, 8, 19, 21, 21, 12, 175, False, False, 313), id="scene-0.5mm"),
    pytest.param(SCENE, 2.5e-4, 1.5e9, (97, 16, 37, 41, 41, 24, 343, False, False, 625), id="scene-0.25mm"),
    pytest.param(SMA, 2.5e-4, 1e9, (55, 4, 23, 31, 31, 9, 144, False, False, 300), id="sma-0.25mm"),
    pytest.param(SMA, 1.25e-4, 1e9, (110, 8, 45, 62, 62, 17, 282, False, False, 599), id="sma-0.125mm"),
    pytest.param(SMA, 6.25e-5, 1e9, (219, 16, 89, 124, 124, 33, 557, False, False, 1198), id="sma-0.0625mm"),
]


@pytest.fixture(autouse=True)
def clear_cache():
    coax_planes._coax_plane_rule.cache_clear()
    yield
    coax_planes._coax_plane_rule.cache_clear()


@pytest.mark.parametrize("line,dz,expected", DEFAULT_ROWS)
def test_default_band_rows(line, dz, expected):
    result = coax_planes.coax_plane_rule(**dict(line, dz=dz))
    assert result[:10] == expected


@pytest.mark.parametrize("line,dz,freq_min,expected", DECLARED_ROWS)
def test_declared_band_rows(line, dz, freq_min, expected):
    result = coax_planes.coax_plane_rule(**dict(line, dz=dz, freq_min=freq_min))
    assert result[:10] == expected


def test_start_fallback_pin():
    result = coax_planes.coax_plane_rule(**START_FALLBACK)
    assert result[:10] == (56, 16, 24, 27, 27, 12, 276, True, False, 313)


def test_clearance_fallback_row():
    result = coax_planes.coax_plane_rule(**CLEARANCE_FALLBACK)
    assert result[:10] == (219, 8, 89, 118, 118, 33, 463, False, True, 1198)


@pytest.mark.parametrize("line,kc1,kc2,fc1", [
    (SCENE, 290.42, 507.43, 9562164522.670092),
    (SMA, 764.41, 1440.12, 25168563662.759033),
])
def test_cutoff_wavenumbers(line, kc1, kc2, fc1):
    result = coax_planes.coax_plane_rule(**line)
    assert abs(result.cutoff_wavenumber_te11 - kc1) <= 0.01
    assert abs(result.cutoff_wavenumber_te21 - kc2) <= 0.01
    assert result.cutoff_frequency_te11 == pytest.approx(fc1, rel=1e-12)


def test_eps_r_scaling():
    result = coax_planes.coax_plane_rule(**dict(
        SCENE, eps_r=4*2.1, freq_max=2.25e9, freq_min=2.25e8))
    assert result[:10] == (33, 4, 17, 17, 17, 6, 105, False, False, 157)
    assert result.cutoff_frequency_te11 == pytest.approx(4781082261.335046, rel=1e-12)


@pytest.mark.parametrize("line,expected", [
    pytest.param(SCENE, (1, 1), id="scene"),
    pytest.param(START_FALLBACK, (11, 1), id="start"),
    pytest.param(CLEARANCE_FALLBACK, (1, 13), id="clearance"),
])
def test_all_bin_evaluation_counts(monkeypatch, line, expected):
    counts = [0, 0]
    start_error = coax_planes._start_error
    source_error = coax_planes._source_error

    def start(z, beta, alpha_te11, alpha_te21):
        counts[0] += len(beta) == 41
        return start_error(z, beta, alpha_te11, alpha_te21)

    def source(z, beta, alpha_te11, alpha_te21, source_position):
        counts[1] += len(beta) == 41
        return source_error(z, beta, alpha_te11, alpha_te21, source_position)

    monkeypatch.setattr(coax_planes, "_start_error", start)
    monkeypatch.setattr(coax_planes, "_source_error", source)
    coax_planes.coax_plane_rule(**line)
    assert tuple(counts) == expected


@pytest.mark.parametrize("line,n", [(SCENE, 33), (START_FALLBACK, 55)])
def test_one_bin_error_is_bin_zero(line, n, capsys):
    inputs = coax_planes._line_inputs(
        line["inner_radius"], line["outer_radius"], line["eps_r"], line["dz"],
        line["freq_max"], line["freq_min"], line["kappa"])
    one, every = coax_planes._start_errors(inputs)
    difference = abs(one(n) - every(n)[0])
    with capsys.disabled():
        print(f"bin-zero n={n}: largest difference {difference:.17g}")
    assert difference <= 1e-9


def test_fit_matches_public_fit_two_wave(capsys):
    z = 1e-3*(10 + 4*np.arange(12))
    beta = 120
    Gamma = 0.3*np.exp(0.7j)
    V = np.exp(1j*beta*z) + Gamma*np.exp(-1j*beta*z)
    fitted = coax_planes._fit_reflection(z, V[None, :])[0]
    public = coaxial_line_reflection_from_plane_voltages(
        z, V, reference_plane_m=0.0).reflection
    differences = abs(fitted - Gamma), abs(public - Gamma)
    with capsys.disabled():
        print(f"two-wave: differences {differences}; largest {max(differences):.17g}")
    assert differences[0] <= 1e-9
    assert differences[1] <= 1e-9


def test_fit_matches_public_fit_with_higher_modes(capsys):
    z = 1e-3*(10 + 4*np.arange(12))
    beta = 120
    Gamma = 0.3*np.exp(0.7j)
    V = (np.exp(1j*beta*z) + Gamma*np.exp(-1j*beta*z)
         + 1.0*np.exp(1.1j)*np.exp(-250.0*z)
         + 1.0*np.exp(2.3j)*np.exp(-0.9313*480.0*z))
    fitted = coax_planes._fit_reflection(z, V[None, :])[0]
    public = coaxial_line_reflection_from_plane_voltages(
        z, V, reference_plane_m=0.0).reflection
    difference = abs(fitted - public)
    with capsys.disabled():
        print(f"higher modes: largest difference {difference:.17g}")
    assert difference <= 1e-9


def test_refuses_traced_input():
    with pytest.raises(RfxRefusal) as caught:
        jax.jit(lambda e: coax_planes.coax_plane_rule(**dict(SCENE, eps_r=e)))(2.1)
    assert type(caught.value) is RfxRefusal
    assert str(caught.value).startswith("coaxial line: eps_r must be a concrete real number")


@pytest.mark.parametrize("name", ["inner_radius", "outer_radius", "eps_r", "dz", "freq_max", "kappa"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_refuses_nonpositive_inputs(name, value):
    with pytest.raises(RfxRefusal) as caught:
        coax_planes.coax_plane_rule(**dict(SCENE, **{name: value}))
    assert type(caught.value) is RfxRefusal
    assert str(caught.value).startswith(f"coaxial line: {name} must be a finite positive number")


def test_refuses_outer_not_above_inner():
    with pytest.raises(RfxRefusal) as caught:
        coax_planes.coax_plane_rule(**dict(SCENE, inner_radius=6e-3, outer_radius=6e-3))
    assert type(caught.value) is RfxRefusal
    assert str(caught.value).startswith("coaxial line: outer_radius must exceed inner_radius")
    assert "6 mm" in str(caught.value)


@pytest.mark.parametrize("value", [0.0, -1.0, 4.5e9, 9e9, np.nan])
def test_refuses_freq_min_outside_band(value):
    with pytest.raises(RfxRefusal) as caught:
        coax_planes.coax_plane_rule(**dict(SCENE, freq_min=value))
    assert type(caught.value) is RfxRefusal
    message = str(caught.value)
    assert message.startswith("coaxial line: freq_min must satisfy")
    assert f"freq_min {value:.6g} Hz" in message
    assert "freq_max 4.5e+09 Hz" in message


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5, np.nan])
def test_refuses_tolerance_outside_unit_interval(value):
    with pytest.raises(RfxRefusal) as caught:
        coax_planes.coax_plane_rule(**dict(SCENE, tolerance=value))
    assert type(caught.value) is RfxRefusal
    assert str(caught.value).startswith("coaxial line: tolerance must satisfy")
    assert repr(value) in str(caught.value)


def test_refuses_radius_ratio():
    with pytest.raises(RfxUnsupported) as caught:
        coax_planes.coax_plane_rule(**dict(SCENE, inner_radius=1e-4, outer_radius=2e-2))
    assert type(caught.value) is RfxUnsupported
    assert str(caught.value).startswith("coaxial line: the outer to inner radius ratio 200")
    assert "100" in str(caught.value)


def test_refuses_freq_max_at_cutoff():
    with pytest.raises(RfxUnsupported) as caught:
        coax_planes.coax_plane_rule(**dict(SCENE, freq_max=10e9, freq_min=1e9))
    assert type(caught.value) is RfxUnsupported
    assert str(caught.value).startswith("coaxial line: freq_max 10 GHz is at or above the TE11 cutoff 9.56216 GHz")


def test_refuses_phase_step_at_pi():
    with pytest.raises(RfxRefusal) as caught:
        coax_planes.coax_plane_rule(**dict(SCENE, kappa=4.0))
    assert type(caught.value) is RfxRefusal
    assert str(caught.value).startswith("coaxial line: the phase step between planes at freq_max is")
    assert "at or above pi" in str(caught.value)


def test_refuses_start_search_bound(monkeypatch):
    monkeypatch.setattr(coax_planes, "_start_error", lambda z, beta, *args: np.ones(len(beta)))
    with pytest.raises(RfxUnsupported) as caught:
        coax_planes.coax_plane_rule(**SCENE)
    assert type(caught.value) is RfxUnsupported
    assert str(caught.value).startswith("coaxial line: no first-plane offset up to 157 cells")
    assert "0.008" in str(caught.value)


def test_refuses_source_search_bound(monkeypatch):
    monkeypatch.setattr(coax_planes, "_source_error", lambda z, beta, *args: np.ones(len(beta)))
    with pytest.raises(RfxUnsupported) as caught:
        coax_planes.coax_plane_rule(**SCENE)
    assert type(caught.value) is RfxUnsupported
    assert str(caught.value).startswith("coaxial line: no source clearance up to 157 cells")
    assert "0.001" in str(caught.value)


def test_refuses_closed_form_clearance_bound(monkeypatch):
    monkeypatch.setattr(coax_planes, "CLEARANCE_SCAN_FACTOR", 0)
    with pytest.raises(RfxUnsupported) as caught:
        coax_planes.coax_plane_rule(**SCENE)
    assert type(caught.value) is RfxUnsupported
    assert str(caught.value).startswith("coaxial line: the closed-form source clearance exceeds its scan bound of 0 cells")


def test_refuses_cutoff_scan_without_root(monkeypatch):
    monkeypatch.setattr(coax_planes, "CUTOFF_SCAN", (0.01, 0.02, 1e-3))
    with pytest.raises(RfxUnsupported) as caught:
        coax_planes.coax_plane_rule(**SCENE)
    assert type(caught.value) is RfxUnsupported
    assert str(caught.value).startswith("coaxial line: the cutoff scan from")
    assert "holds no TE11 root" in str(caught.value)
    assert "ratio 6" in str(caught.value)


def test_refuses_zero_spacing():
    with pytest.raises(RfxRefusal) as caught:
        coax_planes.coax_plane_rule(**dict(SCENE, kappa=1e-15))
    assert type(caught.value) is RfxRefusal
    assert str(caught.value).startswith("coaxial line: kappa 1e-15")
    assert "plane spacing of 0 cells" in str(caught.value)


@pytest.mark.parametrize("name,search", [
    ("_start_error", "first-plane offset"),
    ("_source_error", "source clearance"),
])
def test_refuses_nonfinite_fit_error(monkeypatch, name, search):
    monkeypatch.setattr(coax_planes, name, lambda z, beta, *args: np.full(len(beta), np.nan))
    with pytest.raises(RfxUnsupported) as caught:
        coax_planes.coax_plane_rule(**SCENE)
    assert type(caught.value) is RfxUnsupported
    assert str(caught.value).startswith("coaxial line: the plane fit error is not finite")
    assert search in str(caught.value)
