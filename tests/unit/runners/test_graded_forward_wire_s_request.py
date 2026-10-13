"""Graded forward requests wire S explicitly and shares run's admission."""

import numpy as np
import pytest

from tests.contracts.run_forward_census import base_scene, build_p_wire


def test_plain_source_forward_only_requests_s_on_opt_in():
    options = dict(n_steps=24, skip_preflight=True)
    expected = build_p_wire('graded').run(compute_s_params=False, **options)
    actual = build_p_wire('graded').forward(checkpoint=False, **options)

    assert actual.s_params is None
    assert actual.freqs is None
    assert actual.wire_port_sparams is None
    np.testing.assert_array_equal(actual.time_series, expected.time_series)
    assert np.asarray(actual.time_series).tobytes() == np.asarray(expected.time_series).tobytes()

    with pytest.raises(NotImplementedError) as run_error:
        build_p_wire('graded').run(**options)
    bins = np.linspace(2e9, 20e9, 50)
    with pytest.raises(NotImplementedError) as forward_error:
        build_p_wire('graded').forward(port_s11_freqs=bins, **options)
    assert str(forward_error.value) == str(run_error.value)


def test_requested_wire_s_equals_run_default_without_plain_source():
    def scene():
        return base_scene('graded', source=False).add_port(
            (.008, .012, .003), 'ez', impedance=50., extent=.003)

    options = dict(n_steps=24, skip_preflight=True)
    expected = scene().run(**options)
    np.testing.assert_array_equal(expected.freqs, np.linspace(2e9, 20e9, 50).astype(np.float32))
    actual = scene().forward(port_s11_freqs=expected.freqs, checkpoint=False, **options)

    assert expected.s_params.shape == (1, 1, 50)
    np.testing.assert_array_equal(actual.s_params, expected.s_params)
    assert np.asarray(actual.s_params).tobytes() == np.asarray(expected.s_params).tobytes()
