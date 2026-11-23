"""A vacuum TEM pulse measures each absorber face's reflection at 2–30 GHz.
For (layers, kappa) = (4, 1), measured -16.6 dB and defect -9.9 dB give midpoint -13 dB; for (8, 1), measured -49.8 dB gives +6 dB rounded up to -40 dB; for (16, 1), measured -96.3 dB gives +6 dB rounded up to -90 dB; for (8, 3), measured -33.1 dB gives +6 dB rounded up to -25 dB.
"""

import jax
import numpy as np
import pytest

from tests._absorber_witness import (
    absorber_sensitivity,
    oblique_plane_wave_reflection,
    plane_wave_reflection,
    tfsf_auxiliary_reflection,
)


FREQUENCIES = (2e9, 10e9, 30e9)
BOUNDS_DB = {
    (4, 1): -13,
    (8, 1): -40,
    (16, 1): -90,
    (8, 3): -25,
}


@pytest.fixture(scope="module")
def reflections():
    with jax.default_device(jax.devices("cpu")[0]):
        return {
            (layers, kappa, face): plane_wave_reflection(layers, kappa, face, FREQUENCIES)
            for layers, kappa in BOUNDS_DB for face in ("lo", "hi")
        }


@pytest.mark.parametrize("layers,kappa", BOUNDS_DB)
@pytest.mark.parametrize("face", ("lo", "hi"))
def test_reflection_below_depth_bound(reflections, layers, kappa, face):
    measured = reflections[layers, kappa, face]
    assert np.all(np.isfinite(measured))
    assert np.all(measured < BOUNDS_DB[layers, kappa]), (
        f"{layers} layers, kappa={kappa}, {face}: R at 2/10/30 GHz = {measured} dB"
    )


@pytest.mark.parametrize("layers,kappa", BOUNDS_DB)
def test_faces_agree_within_one_db(reflections, layers, kappa):
    difference = np.abs(reflections[layers, kappa, "lo"] - reflections[layers, kappa, "hi"])
    assert np.all(difference <= 1), f"lo/hi magnitude difference = {difference} dB"


@pytest.mark.parametrize("face", ("lo", "hi"))
def test_reflection_strictly_improves_with_depth(reflections, face):
    for shallow, deep in ((4, 8), (8, 16)):
        assert np.all(reflections[deep, 1, face] < reflections[shallow, 1, face])


def test_absorber_sensitivity_of_complex_record():
    """The deeper record is the denominator, including unchanged and zero entries."""
    calls = []
    records = {8: np.array([3 + 4j, 2j, 0, 1]), 16: np.array([3, 2j, 0, 0])}

    def measure(layers):
        calls.append(layers)
        return records[layers]

    np.testing.assert_allclose(absorber_sensitivity(measure, 8), [4 / 3, 0, 0, np.inf])
    assert calls == [8, 16]


def test_oblique_reflection_at_45_degrees():
    """Measured -69.7 dB gives +6 dB rounded up to -60 dB."""
    with jax.default_device(jax.devices("cpu")[0]):
        measured = oblique_plane_wave_reflection()
    assert measured < -60, f"45 degree TMz return = {measured} dB"


@pytest.mark.parametrize("layers,measured_db", [
    (20, -27.1),
    (200, -84.0),
])
@pytest.mark.parametrize("face", ("lo", "hi"))
def test_tfsf_auxiliary_reflection(layers, measured_db, face):
    """At 20 layers, matched -112.2 dB gives +6 dB rounded up to -105 dB; at 200 layers, matched -114.9 dB gives +6 dB rounded up to -105 dB."""
    with jax.default_device(jax.devices("cpu")[0]):
        measured = tfsf_auxiliary_reflection(layers, face, FREQUENCIES)
    assert np.all(measured < -105), (
        f"{layers} auxiliary layers, {face}: R = {measured} dB; "
        f"#1234 baseline worst {measured_db} dB"
    )
