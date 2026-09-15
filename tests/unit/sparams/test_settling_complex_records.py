"""Recorded energy includes both quadratures, independently of global phase."""
import numpy as np
import pytest

from rfx.sources.waveguide_port import settling_db_from_named_records


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pure_imaginary_ringdown_has_the_same_coverage_as_real(dtype):
    record = np.exp(-np.arange(400) / 20.).astype(dtype)
    real = settling_db_from_named_records([("record", record)])
    imaginary = settling_db_from_named_records([("record", 1j * record)])
    assert np.isfinite(imaginary)
    assert imaginary == real


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distinct_quadrature_decays_follow_their_total_power(dtype):
    time = np.arange(400)
    first = np.exp(-time / 20.).astype(dtype)
    second = (.5 * np.exp(-time / 50.)).astype(dtype)
    record = first + 1j * second
    power = first.astype(np.float64)**2 + second.astype(np.float64)**2
    expected = 10 * np.log10(power[-40:].mean() / power.max())
    actual = settling_db_from_named_records([("record", record)])
    rotated = settling_db_from_named_records([("record", 1j * record)])
    assert actual == pytest.approx(expected, abs=1e-12)
    assert rotated == actual


def test_complex64_coverage_uses_the_same_storage_floor():
    good = np.full(100, np.complex64(3e-36j))
    quiet = np.full(100, np.complex64(1e-37j))
    value, detail = settling_db_from_named_records(
        [("good", good), ("quiet", quiet)], return_detail=True)
    assert value == pytest.approx(0.)
    assert detail["skipped_records"] == ["quiet"]
    assert detail["n_witnessed"] == 1


@pytest.mark.parametrize("scale", [1., 1e200, 1e-200])
def test_finite_amplitude_units_do_not_change_the_power_ratio(scale):
    record = np.exp(-np.arange(100) / 8.) * (1. + .5j)
    baseline = settling_db_from_named_records([("record", record)])
    changed = settling_db_from_named_records([("record", record * scale)])
    assert np.isfinite(changed)
    assert changed == pytest.approx(baseline, abs=1e-12)


@pytest.mark.parametrize("constant", [1e200, complex(1e308, 1e308)])
def test_large_constant_record_cannot_hide_behind_a_decaying_record(constant):
    decayed = np.exp(-np.arange(100) / 8.)
    loud = np.full(100, constant)
    assert np.isfinite(loud).all()
    value, detail = settling_db_from_named_records(
        [("decayed", decayed), ("constant", loud)], return_detail=True)
    assert value == 0.0
    assert detail["per_record_db"]["constant"] == 0.0
    assert detail["n_witnessed"] == 2 and not detail["skipped_records"]


def test_complex_storage_floor_reads_both_components_together():
    floor = np.finfo(np.float32).tiny * 100.
    above = np.full(100, .75 * floor * (1. + 1j), dtype=np.complex64)
    below = np.full(100, .5 * floor * (1. + 1j), dtype=np.complex64)
    value, detail = settling_db_from_named_records(
        [("above", above), ("below", below)], return_detail=True)
    assert value == 0.0
    assert detail["skipped_records"] == ["below"]
    assert detail["n_witnessed"] == 1
