"""Waveguide reporting planes are shared by the two public entries."""

import numpy as np
import pytest

from rfx import Simulation
from rfx.sources.waveguide_port import extract_waveguide_sparams


DX = 1 / 512


def _port_sim(*, graded=False, crossing=False, preset=None, reference=None, probe=None):
    profile = np.r_[np.full(16, DX), [0.75 * DX], np.full(8, DX / 2),
                    [0.75 * DX], np.full(16, DX)]
    sim = Simulation(
        freq_max=12e9, domain=(float(profile.sum()), 12 * DX, 6 * DX),
        dx=DX, dx_profile=profile if graded else None,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8,
    )
    sim.add_waveguide_port(
        (15 if crossing else 5) * DX, freqs=np.linspace(8.4e9, 11.6e9, 5),
        f0=10e9, ref_offset=3, probe_offset=10, name="left",
        calibration_preset=preset, reference_plane=reference, probe_plane=probe,
    )
    return sim


@pytest.mark.parametrize("preset,reference,probe,label", [
    (None, None, None, "source_to_probe"),
    ("measured", None, None, "measured"),
    ("source_to_probe", None, None, "source_to_probe"),
    (None, 6 * DX, None, "explicit"),
    (None, None, 16 * DX, "explicit"),
])
def test_c3_uniform_matches_low_level(preset, reference, probe, label, record_property):
    sim = _port_sim(preset=preset, reference=reference, probe=probe)
    entry = sim._waveguide_ports[0]
    source = round(entry.x_position / DX) * DX
    measured = source + entry.ref_offset * DX
    measured_probe = source + entry.probe_offset * DX
    desired = reference if reference is not None else (measured if preset == "measured" else source)
    desired_probe = probe if probe is not None else measured_probe
    result = sim.run(n_steps=160, compute_s_params=False)
    actual = result.waveguide_sparams["left"]
    expected = extract_waveguide_sparams(
        result.waveguide_ports["left"], ref_shift=desired - measured,
        probe_shift=desired_probe - measured_probe,
    )
    _record_comparison(record_property, actual, expected)
    np.testing.assert_array_equal(actual.s11, expected[0])
    np.testing.assert_array_equal(actual.s21, expected[1])
    assert actual.reference_plane == desired
    assert actual.probe_plane == desired_probe
    assert actual.calibration_preset == label


def test_c4_graded_matches_low_level(record_property):
    sim = _port_sim(graded=True)
    entry = sim._waveguide_ports[0]
    source = round(entry.x_position / DX) * DX
    measured = source + entry.ref_offset * DX
    result = sim.run(n_steps=160, compute_s_params=False)
    actual = result.waveguide_sparams["left"]
    expected = extract_waveguide_sparams(
        result.waveguide_ports["left"], ref_shift=source - measured,
    )
    _record_comparison(record_property, actual, expected)
    np.testing.assert_array_equal(actual.s11, expected[0])
    np.testing.assert_array_equal(actual.s21, expected[1])
    assert actual.reference_plane == source
    assert actual.probe_plane == source + entry.probe_offset * DX
    assert actual.calibration_preset == "source_to_probe"


def test_c4_graded_crossing_refuses():
    sim = _port_sim(graded=True, crossing=True)
    with pytest.raises(ValueError, match="waveguide port 'left'.*ref_offset.*measured"):
        sim.run(n_steps=16, compute_s_params=False)


def test_c4_measured_crossing_needs_no_shift(record_property):
    sim = _port_sim(graded=True, crossing=True, preset="measured")
    result = sim.run(n_steps=16, compute_s_params=False)
    actual = result.waveguide_sparams["left"]
    expected = extract_waveguide_sparams(result.waveguide_ports["left"])
    _record_comparison(record_property, actual, expected)
    np.testing.assert_array_equal(actual.s11, expected[0])
    np.testing.assert_array_equal(actual.s21, expected[1])
    assert actual.reference_plane == (15 + 1 + 0.75 + 0.5) * DX


def _record_comparison(record, actual, expected):
    record("max_s11", float(np.max(np.abs(actual.s11))))
    record("s11_max_difference", float(np.max(np.abs(actual.s11 - expected[0]))))
    record("s21_max_difference", float(np.max(np.abs(actual.s21 - expected[1]))))
    record("reference_plane", actual.reference_plane)
    record("probe_plane", actual.probe_plane)
