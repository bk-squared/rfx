from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/showcase"))
import _lens_filter_common as common
import design_lens as lens


def test_lens_band(monkeypatch, tmp_path):
    expected = np.array([9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9,
                         10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8,
                         10.9, 11.0]) * 1e9
    design = np.array([9.5, 10.0, 10.5]) * 1e9
    bins = common.band_frequencies("9,11,0.1", lens.FREQS)
    np.testing.assert_allclose(bins, expected, atol=1e3, rtol=0)
    np.testing.assert_array_equal(bins[[5, 10, 15]], design)
    np.testing.assert_array_equal(lens.FREQS, design)
    for text in ("9,11,0", "9,11,nan", "11,9,.1", "9,11,.3", "9,11,.2", "9,11"):
        with pytest.raises(ValueError):
            common.band_frequencies(text, design)
    for stage in ("trial", "main", "field", "frame", "cost"):
        with pytest.raises(SystemExit) as exc:
            common.main(lens, ["--stage", stage, "--out", str(tmp_path), "--boundary", "nearest",
                               "--band", "9,11,.1"])
        assert exc.value.code == 2
    import rfx.measurement.farfield as farfield
    import rfx.sparams._tail_witness as tail
    from rfx.farfield import init_ntff_data, make_ntff_box
    with jax.default_device(jax.devices("cpu")[0]):
        model = lens.with_objective("fan").Model(boundary="nearest", freqs=bins)
        box = make_ntff_box(model.grid, *model.sim._ntff)
        np.testing.assert_allclose(np.asarray(box.freqs), expected, atol=1e3, rtol=0)
        data = init_ntff_data(box)
        assert data.x_lo.shape[0] == 21
        assert data.y_lo.shape[0] == 21
        assert data.z_lo.shape[0] == 21
        calls = []
        saved = SimpleNamespace(ntff_data=data, time_series=np.zeros((1, 4)),
                                settling_source_end_index=0)
        def forward(**kwargs):
            calls.append(kwargs)
            return saved
        def far_field(ntff, declared, grid, theta, phi):
            assert ntff.x_lo.shape[0] == 21
            shape = (len(declared.freqs), len(theta), len(phi))
            return SimpleNamespace(E_theta=jnp.ones(shape) * 1e-14, E_phi=jnp.zeros(shape))
        witness = namedtuple("Witness", "status share_per_bin")
        def settling(channels, dt, end, freqs, **kwargs):
            np.testing.assert_allclose(freqs, expected, atol=1e3, rtol=0)
            return witness("pass", [0.] * 21)
        monkeypatch.setattr(model.sim, "forward", forward)
        monkeypatch.setattr(farfield, "compute_far_field_jax", far_field)
        monkeypatch.setattr(tail, "tail_share_witness", settling)
        response, row = model.evaluate(lens.starts()["primary"], 1)
        assert len(calls) == 1
        assert response.shape[0] == 21
        assert response.dtype == np.float64
        for key in ("strip_min_dbi", "strip_max_dbi", "narrow_hpbw_deg", "outside_max_dbi"):
            assert np.asarray(row[key]).shape == (21,)
            assert np.asarray(row[key]).dtype == np.float64
        bore = lens.summary(response, freqs=bins)
        assert bore["boresight_dbi"].shape == (21,)
        assert bore["aperture_bound_dbi"].shape == (21,)
        np.testing.assert_array_equal(model.freqs[model.design_indices], design)
