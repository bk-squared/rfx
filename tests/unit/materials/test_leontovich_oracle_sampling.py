"""The O3 recorder/model handoff preserves actual Yee sampling coordinates.

Only the field backend and analytic model calls are mocked. The committed
_run_guide extraction and _model_fits handoff execute unchanged; no FDTD or
transverse eigenvalue solve runs in this test.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.oracle import test_leontovich_alpha_oracle as oracle


@pytest.mark.parametrize("ez_cell,expected_ez_z", [(4, .00225), (8, .00425)])
def test_recorded_component_coordinates_reach_model_fit_and_prediction(
    monkeypatch, ez_cell, expected_ez_z,
):
    # Nonzero padding detects forgotten coordinate offsets. The returned Ez
    # registration deliberately differs from the builder's nominal z=3 mm.
    grid = SimpleNamespace(pad_x_lo=7, pad_y_lo=2, pad_z_lo=3,
                           shape=(390, 9, 18))
    nx, ny, nz = grid.shape
    x_nodes = (np.arange(nx) - 7) * .0005
    y_nodes = (np.arange(ny) - 2) * .0005
    z_h = (np.arange(nz) - 3 + .5) * .0005
    frequencies = tuple(oracle.O3_FREQS)

    def electric(fi, x, y):
        return (fi + 1) * (1 + 1j * y / .001) * np.exp((-.8 - 17j) * x)

    def magnetic(fi, x, z):
        # Vary both physical axes and quadratures. A constant plane would
        # fail to discriminate a shifted/incorrectly cropped observation.
        return (fi + 1) * (2 + x / .0005 + 1j * (5 + z / .0005))

    ez = np.stack([electric(fi, x_nodes[:, None], y_nodes[None, :])
                   for fi in range(len(frequencies))])
    hy = np.stack([magnetic(fi, x_nodes[:, None] + .00025, z_h[None, :])
                   for fi in range(len(frequencies))])
    # Twelve physical Hy z centres span the 6 mm stack. The bounding E
    # node's Hy slot (and the padding after it) is outside that stack.
    hy[:, :, :3] = 1e30 + 2e30j
    hy[:, :, 15:] = -3e30 + 4e30j
    result = SimpleNamespace(
        dft_planes={
            "midplane": SimpleNamespace(accumulator=ez, index=3 + ez_cell),
            "yhy": SimpleNamespace(accumulator=hy),
        },
        time_series=np.exp(-np.arange(37) / 8)[:, None],
    )
    builds, runs = [], []

    class FakeSimulation:
        def _build_grid(self):
            return grid

        def run(self, **kwargs):
            runs.append(kwargs)
            return result

    def build(*args, **kwargs):
        builds.append((args, kwargs))
        return FakeSimulation()

    monkeypatch.setattr(oracle, "_build_guide", build)
    # The oracle owns module-level caches for expensive runs/fits. A prior
    # oracle test must not make this handoff check skip its fit/predict calls.
    monkeypatch.setattr(oracle, "_cache", {})
    out = oracle._run_guide(freqs=frequencies, n_steps=37)
    assert builds == [((oracle.SIGMA_BULK,), {
        "f0_mode": True, "thickness": oracle.THICKNESS, "freqs": frequencies,
    })]
    assert runs == [{"n_steps": 37, "compute_s_params": False}]

    # Independent hand geometry, including both endpoints of the 25–125 mm
    # fit span. Ez uses x nodes; Hy is half a cell to their right.
    expected_ex = np.linspace(.025, .125, 201)
    expected_hx = np.linspace(.02525, .12525, 201)
    expected_hz = np.linspace(.00025, .00575, 12)
    np.testing.assert_allclose(out["xs"], expected_ex, rtol=0, atol=1e-16)
    np.testing.assert_allclose(out["hy_xs"], expected_hx, rtol=0, atol=1e-16)
    np.testing.assert_allclose(out["z_nodes"], expected_hz, rtol=0, atol=1e-17)
    assert out["ez_z"] == pytest.approx(expected_ez_z, abs=1e-17)
    assert out["hy_plane"].shape == (len(frequencies), 201, 12)
    for fi in range(len(frequencies)):
        np.testing.assert_allclose(
            out["hy_plane"][fi], magnetic(fi, expected_hx[:, None], expected_hz[None, :]),
            rtol=1e-14, atol=1e-12)
        np.testing.assert_allclose(out["profile"][fi], np.abs(electric(fi, expected_ex, .001)),
                                   rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(out["alpha"], .8, rtol=0, atol=1e-12)

    fit_calls, prediction_calls = [], []

    def fit(xs, zs, measured_hy, f, b, g, rs, eta):
        fi = len(fit_calls)
        np.testing.assert_allclose(xs, expected_hx, rtol=0, atol=1e-16)
        np.testing.assert_allclose(zs, expected_hz, rtol=0, atol=1e-17)
        np.testing.assert_array_equal(measured_hy, out["hy_plane"][fi])
        assert (f, b, g, rs, eta) == (
            frequencies[fi], oracle.B_PLATE, oracle.G_STUB, oracle.RS0, oracle.ETA_0)
        fitted = {"alpha_model": .1 + fi, "x_reference": float(xs[0]), "fi": fi}
        fit_calls.append(fitted)
        return fitted

    def predict(fitted, xs, zs, f, b, g, rs, eta):
        fi = len(prediction_calls)
        assert fitted is fit_calls[fi]
        assert fitted["x_reference"] == pytest.approx(expected_hx[0], abs=1e-16)
        np.testing.assert_allclose(xs, expected_ex, rtol=0, atol=1e-16)
        np.testing.assert_allclose(zs, [expected_ez_z], rtol=0, atol=1e-17)
        assert (f, b, g, rs, eta) == (
            frequencies[fi], oracle.B_PLATE, oracle.G_STUB, oracle.RS0, oracle.ETA_0)
        prediction_calls.append(fitted)
        # Let the real alpha-fit consumer process an unfitted Ez prediction
        # with a slope distinct from the mocked Hy model's alpha.
        return np.exp(-(2 + fi) * np.asarray(xs))[:, None] * (1 + .2j)

    monkeypatch.setattr(oracle._trm, "fit_hy_field", fit)
    monkeypatch.setattr(oracle._trm, "predict_ez_from_hy_fit", predict)
    fits = oracle._model_fits(out)
    assert len(fit_calls) == len(prediction_calls) == len(frequencies)
    assert fits is oracle._cache["model_fits"]
    np.testing.assert_allclose([item["alpha_model_ez"] for item in fits],
                               2 + np.arange(len(frequencies)), rtol=0, atol=1e-12)
