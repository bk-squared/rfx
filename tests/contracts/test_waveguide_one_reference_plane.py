"""Waveguide reporting planes are shared by the two public entries."""

from dataclasses import replace

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources.waveguide_port import extract_waveguide_sparams
from tests._waveguide_chain_battery_fixture import DX_COARSE, NUM_PERIODS, build_simulation


DX = 1 / 512


@pytest.mark.parametrize("preset", [None, "measured"], ids=["C1-default", "C2-measured"])
def test_slab_entries_share_reference_plane(preset, record_property):
    one_port = build_simulation("slab", DX_COARSE)
    two_port = build_simulation("slab", DX_COARSE)
    for sim in (one_port, two_port):
        sim._waveguide_ports = [
            replace(entry, reference_plane=None, calibration_preset=preset)
            for entry in sim._waveguide_ports
        ]
    one_port._waveguide_ports = one_port._waveguide_ports[:1]
    entry = one_port._waveguide_ports[0]
    source = round(entry.x_position / DX_COARSE) * DX_COARSE
    measured = source + entry.ref_offset * DX_COARSE
    expected_plane = measured if preset == "measured" else source
    n_steps = int(one_port._build_grid().num_timesteps(NUM_PERIODS))
    run_result = one_port.run(n_steps=n_steps, compute_s_params=False)
    run_s = run_result.waveguide_sparams["left"]
    matrix = two_port.compute_waveguide_s_matrix(n_steps=n_steps, normalize=False)
    expected_s11, _ = extract_waveguide_sparams(
        run_result.waveguide_ports["left"], ref_shift=expected_plane - measured,
    )
    np.testing.assert_array_equal(run_s.s11, expected_s11)
    difference = float(np.max(np.abs(run_s.s11 - matrix.s_params[0, 0])))
    peak = float(np.max(np.abs(expected_s11)))
    record_property("n_steps", n_steps)
    record_property("max_s11", peak)
    record_property("s11_max_difference", difference)
    record_property("s11_relative_difference", difference / peak)
    record_property("run_reference_plane", float(run_s.reference_plane))
    record_property("matrix_reference_planes", [float(p) for p in matrix.reference_planes])
    assert difference <= 1e-4 * peak
    assert run_s.reference_plane == expected_plane
    assert matrix.reference_planes[0] == expected_plane
    expected_planes = [
        round(port.x_position / DX_COARSE) * DX_COARSE
        + ((1 if port.direction == "+x" else -1) * port.ref_offset * DX_COARSE
           if preset == "measured" else 0)
        for port in two_port._waveguide_ports
    ]
    np.testing.assert_array_equal(matrix.reference_planes, expected_planes)


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


def test_d1b_graded_forward_crossing_refuses_like_run():
    # forward() returns the declared waveguide records and S since #1373 D7, so
    # it resolves the same planes as run() and refuses the same crossing port
    # with the same message (equal requests give equal results).
    messages = {}
    for entry, call in (("run", lambda s: s.run(n_steps=16, compute_s_params=False)),
                        ("forward", lambda s: s.forward(n_steps=16))):
        sim = _port_sim(graded=True, crossing=True)
        with pytest.raises(ValueError, match="waveguide port 'left'.*ref_offset.*measured") as exc:
            call(sim)
        messages[entry] = str(exc.value)
    assert messages["run"] == messages["forward"]


@pytest.mark.parametrize("graded,n_modes,normalize", [
    (False, 2, False), (True, 1, True),
], ids=["multimode", "graded"])
def test_calculator_measured_planes(graded, n_modes, normalize, record_property):
    sim = _port_sim(graded=graded, preset="measured")
    sim._waveguide_ports = [replace(sim._waveguide_ports[0], x_position=5.2 * DX,
                                   n_modes=n_modes)]
    sim.add_waveguide_port(
        sim._domain[0] - 5.25 * DX, direction="-x", name="right",
        freqs=np.linspace(8.4e9, 11.6e9, 5), f0=10e9,
        ref_offset=3, probe_offset=10, n_modes=n_modes, calibration_preset="measured",
    )
    if graded:
        # A reflected wave makes a wrong extraction shift observable after normalization.
        sim.add_material("slab", eps_r=4.0)
        sim.add(Box((18 * DX, 0, 0), (22 * DX, 12 * DX, 6 * DX)), material="slab")
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    nodes = np.asarray([grid.node_of("x", i) for i in range(grid.nx)])
    expected = []
    sources = []
    for entry in sim._waveguide_ports:
        index = int(np.argmin(np.abs(nodes - entry.x_position)))
        sign = 1 if entry.direction == "+x" else -1
        sources.append(float(nodes[index]))
        expected.append(float(nodes[index + sign * entry.ref_offset]))
    result = sim.compute_waveguide_s_matrix(n_steps=160, normalize=normalize)
    record_property("source_planes", sources)
    record_property("expected_measured_planes", expected)
    record_property("reported_reference_planes", [float(p) for p in result.reference_planes])
    np.testing.assert_array_equal(result.reference_planes, expected)
    assert np.all(np.asarray(expected) != np.asarray(sources))
    if graded:
        sim._waveguide_ports = [
            replace(entry, calibration_preset=None, reference_plane=plane)
            for entry, plane in zip(sim._waveguide_ports, expected)
        ]
        explicit = sim.compute_waveguide_s_matrix(n_steps=160, normalize=normalize)
        difference = float(np.max(np.abs(result.s_params - explicit.s_params)))
        peak = float(np.max(np.abs(explicit.s_params)))
        record_property("measured_vs_explicit_max_difference", difference)
        record_property("explicit_max_s", peak)
        record_property("measured_vs_explicit_relative_difference", difference / peak)
        record_property("explicit_max_s11", float(np.max(np.abs(explicit.s_params[0, 0]))))
        assert np.max(np.abs(explicit.s_params[0, 0])) > 0
        np.testing.assert_array_equal(explicit.reference_planes, expected)
        np.testing.assert_array_equal(result.s_params, explicit.s_params)


def test_c4_graded_crossing_refuses():
    sim = _port_sim(graded=True, crossing=True)
    with pytest.raises(ValueError, match="waveguide port 'left'.*ref_offset.*measured") as exc:
        sim.run(n_steps=16, compute_s_params=False)
    assert str(exc.value).count("waveguide port 'left':") == 1


@pytest.mark.parametrize("planes", [{"reference": 20 * DX}, {"probe": 25 * DX}])
def test_explicit_crossing_remedy(planes):
    sim = _port_sim(graded=True, crossing=True, **planes)
    with pytest.raises(ValueError) as exc:
        sim.run(n_steps=16, compute_s_params=False)
    message = str(exc.value)
    assert message.count("waveguide port 'left':") == 1
    assert "set reference_plane inside one uniform zone" in message
    assert "calibration_preset" not in message


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
