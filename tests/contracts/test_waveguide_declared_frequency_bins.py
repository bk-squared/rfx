"""Waveguide declarations retain their bins on every run and forward lane."""

import numpy as np
import pytest

from rfx import Simulation


FMAX = 15e9
PATHS = ("uniform-run", "uniform-forward", "graded-run", "graded-forward")
SHARED_PATHS = PATHS[1:]


def _scene(path, declarations):
    profile = {"dz_profile": np.full(8, .002)} if path.startswith("graded") else {}
    sim = Simulation(freq_max=FMAX, domain=(.016,) * 3, dx=.002,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"},
                     cpml_layers=2, **profile)
    for name, declaration in declarations:
        sim.add_waveguide_port(.004, direction="+x", f0=12e9,
                              ref_offset=1, probe_offset=2, mode_profile="analytic",
                              name=name, **declaration)
    return sim


def _solve(sim, path):
    options = dict(n_steps=8, skip_preflight=True)
    if path.endswith("forward"):
        return sim.forward(**options, checkpoint=False)
    return sim.run(**options)


def _assert_bins(result, name, expected, path):
    bins = np.asarray(result.waveguide_sparams[name].freqs)
    assert bins.shape == expected.shape
    np.testing.assert_allclose(bins, expected, rtol=2 * np.finfo(np.float32).eps, atol=0)
    if path != "uniform-run":
        assert result.waveguide_ports[name].freqs.dtype == np.dtype("float32")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("count", [7, 50], ids=["n_freqs", "default"])
def test_declared_bins(path, count):
    declaration = {"n_freqs": count} if count == 7 else {}
    result = _solve(_scene(path, [("input", declaration)]), path)
    expected = np.linspace(FMAX / 10, FMAX, count, dtype=np.float32)
    _assert_bins(result, "input", expected, path)


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_run_forward_same_bins(mesh):
    declaration = [("input", {"n_freqs": 7})]
    run = _solve(_scene(mesh + "-run", declaration), mesh + "-run")
    forward = _solve(_scene(mesh + "-forward", declaration), mesh + "-forward")
    expected = np.linspace(FMAX / 10, FMAX, 7)
    _assert_bins(run, "input", expected, mesh + "-run")
    _assert_bins(forward, "input", expected, mesh + "-forward")
    np.testing.assert_allclose(run.waveguide_sparams["input"].freqs,
                               forward.waveguide_sparams["input"].freqs,
                               rtol=2 * np.finfo(np.float32).eps, atol=0)


@pytest.mark.parametrize("path", SHARED_PATHS)
@pytest.mark.parametrize("difference", ["values", "shape", "n_freqs"])
def test_different_bins_refused_before_fields(path, difference, monkeypatch):
    import rfx.stepping.uniform_setup as uniform_setup
    import rfx.stepping.graded_setup as graded_setup

    def unexpected_fields(*args, **kwargs):
        pytest.fail("fields were built before differing port bins were refused")

    monkeypatch.setattr(uniform_setup, "init_state", unexpected_fields)
    monkeypatch.setattr(graded_setup, "init_state", unexpected_fields)
    first = np.array([10e9, 12e9, 14e9])
    second = np.array([10e9, 13e9, 14e9]) if difference == "values" else np.array([11e9, 13e9])
    declarations = [("input", {"freqs": first}), ("output", {"freqs": second})]
    if difference == "n_freqs":
        first, second = np.linspace(FMAX / 10, FMAX, 7), np.linspace(FMAX / 10, FMAX, 5)
        declarations = [("input", {"n_freqs": 7}), ("output", {"n_freqs": 5})]
    with pytest.raises(ValueError) as caught:
        _solve(_scene(path, declarations), path)
    message = str(caught.value)
    assert path.replace("-", " ") + "()" in message
    for name, bins in (("input", first), ("output", second)):
        bins = bins.astype(np.float32)
        assert f"port '{name}' has {bins.size} bins" in message
        assert f"first={bins[0]:.9g} Hz, last={bins[-1]:.9g} Hz" in message
    assert "same freqs on every port" in message
    assert "compute_waveguide_s_matrix()" in message


@pytest.mark.parametrize("path", SHARED_PATHS)
def test_equal_explicit_bins_on_both_ports(path):
    expected = np.array([10e9, 12e9, 14e9])
    declarations = [(name, {"freqs": expected}) for name in ("input", "output")]
    result = _solve(_scene(path, declarations), path)
    for name, _ in declarations:
        _assert_bins(result, name, expected, path)


def test_graded_calculator_s_uses_declared_bins():
    sim = Simulation(freq_max=FMAX, domain=(.032, .016, .016), dx=.002,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"},
                     cpml_layers=4, dz_profile=np.full(8, .002))
    for name, position, direction in (("input", .008, "+x"), ("output", .024, "-x")):
        sim.add_waveguide_port(position, direction=direction, f0=12e9,
                              ref_offset=1, probe_offset=2, mode_profile="analytic",
                              name=name, n_freqs=7)
    result = sim.compute_waveguide_s_matrix(n_steps=30, normalize=True, skip_preflight=True)
    expected = np.linspace(FMAX / 10, FMAX, 7)
    assert result.s_params.shape == (2, 2, 7)
    assert result.freqs.shape == expected.shape
    np.testing.assert_allclose(result.freqs, expected,
                               rtol=2 * np.finfo(np.float32).eps, atol=0)
