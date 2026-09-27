"""A vacuum TEM pulse measures each absorber face's reflection at 2–30 GHz.

ADI has no periodic transverse option for this plane-wave measurement; leave it to #1234.
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
    # Measured -16.6 dB, defect -9.9 dB: midpoint -13 dB; #1012 fixed this absorber; before it, 8 layers reflected −15 dB.
    (4, 1): -13,
    # Measured -49.8 dB, defect -15.3 dB: +6 dB rounded up to -40 dB; #1012 fixed this absorber; before it, 8 layers reflected −15 dB.
    (8, 1): -40,
    # Measured -96.3 dB, defect -22.6 dB: +6 dB rounded up to -90 dB; #1012 fixed this absorber; before it, 8 layers reflected −15 dB.
    (16, 1): -90,
    # Measured -33.1 dB, defect -12.8 dB: +6 dB rounded up to -25 dB; #1012 fixed this absorber; before it, 8 layers reflected −15 dB.
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
    # An independently specified pair checks complex differences and the
    # deeper record as denominator, including unchanged and zero entries.
    calls = []
    records = {8: np.array([3 + 4j, 2j, 0, 1]), 16: np.array([3, 2j, 0, 0])}

    def measure(layers):
        calls.append(layers)
        return records[layers]

    np.testing.assert_allclose(absorber_sensitivity(measure, 8), [4 / 3, 0, 0, np.inf])
    assert calls == [8, 16]


def test_oblique_reflection_at_45_degrees():
    with jax.default_device(jax.devices("cpu")[0]):
        measured = oblique_plane_wave_reflection()
    # Measured -69.7 dB, defect -15.9 dB: +6 dB rounded up to -60 dB; #1012 fixed this absorber; before it, 8 layers reflected −15 dB.
    assert measured < -60, f"45 degree TMz return = {measured} dB"


@pytest.mark.parametrize("layers,measured_db", [
    # Matched -112.2 dB, shipped defect -27.1 dB: +6 dB rounded up to -105 dB; main reference: #1012 fixed this absorber; before it, 8 layers reflected −15 dB.
    (20, -27.1),
    # Matched -114.9 dB, shipped defect -84.0 dB: +6 dB rounded up to -105 dB; main reference: #1012 fixed this absorber; before it, 8 layers reflected −15 dB.
    (200, -84.0),
])
@pytest.mark.parametrize("face", ("lo", "hi"))
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#1234: auxiliary H profile uses E nodes; 20/200 cells return -27/-84 dB at 2 GHz")
def test_tfsf_auxiliary_reflection(layers, measured_db, face):
    with jax.default_device(jax.devices("cpu")[0]):
        measured = tfsf_auxiliary_reflection(layers, face, FREQUENCIES)
    assert np.all(measured < -105), (
        f"{layers} auxiliary layers, {face}: R = {measured} dB; "
        f"#1234 baseline worst {measured_db} dB"
    )
