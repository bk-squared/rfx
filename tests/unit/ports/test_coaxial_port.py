"""Principled coverage for the coaxial-port abstraction.

These tests exercise the current public/high-level builder plus the low-level
material/source helpers that already define the coaxial-port behavior.
"""

from __future__ import annotations

import pytest

from rfx import CoaxialPort, GaussianPulse, Simulation
from rfx.core.yee import EPS_0, init_materials, init_state
from rfx.grid import Grid
from rfx.sources.coaxial_port import (
    PEC_SIGMA,
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    coaxial_load_reflection,
    coaxial_tem_capacitance_per_m,
    coaxial_tem_characteristic_impedance,
    coaxial_tem_inductance_per_m,
    coaxial_tem_phase_constant,
    coaxial_tem_reference_plane_s11,
    coaxial_tem_reference_plane_vi,
    coaxial_tem_reference_plane_vi_from_cartesian_plane,
    make_coaxial_port_source,
    setup_coaxial_port,
)


class ConstantWaveform:
    def __init__(self, amplitude: float):
        self.amplitude = amplitude

    def __call__(self, t: float) -> float:
        return self.amplitude


FACE_CASES = [
    pytest.param("top", (0.010, 0.010, 0.015), "ez", id="top"),
    pytest.param("bottom", (0.010, 0.010, 0.005), "ez", id="bottom"),
    pytest.param("front", (0.010, 0.005, 0.010), "ey", id="front"),
    pytest.param("back", (0.010, 0.015, 0.010), "ey", id="back"),
    pytest.param("left", (0.005, 0.010, 0.010), "ex", id="left"),
    pytest.param("right", (0.015, 0.010, 0.010), "ex", id="right"),
]


def _make_grid() -> Grid:
    return Grid(
        freq_max=10e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.5e-3,
        cpml_layers=0,
    )


def test_add_coaxial_port_records_defaults_on_simulation():
    sim = Simulation(freq_max=8e9, domain=(0.020, 0.020, 0.020), boundary="cpml")

    returned = sim.add_coaxial_port((0.010, 0.010, 0.015))

    assert returned is sim
    assert len(sim._coaxial_ports) == 1

    port = sim._coaxial_ports[0]
    assert isinstance(port, CoaxialPort)
    assert port.position == (0.010, 0.010, 0.015)
    assert port.face == "top"
    assert port.pin_length == pytest.approx(5e-3)
    assert port.pin_radius == pytest.approx(SMA_PIN_RADIUS)
    assert port.outer_radius == pytest.approx(SMA_OUTER_RADIUS)
    assert port.impedance == pytest.approx(50.0)
    assert isinstance(port.excitation, GaussianPulse)
    assert port.excitation.f0 == pytest.approx(4e9)
    assert port.excitation.bandwidth == pytest.approx(0.8)


def test_add_coaxial_port_preserves_explicit_parameters_and_waveform():
    waveform = ConstantWaveform(1.25)
    sim = Simulation(freq_max=6e9, domain=(0.020, 0.020, 0.020), boundary="pec")

    sim.add_coaxial_port(
        (0.005, 0.010, 0.010),
        face="left",
        pin_length=3.0e-3,
        pin_radius=0.4e-3,
        outer_radius=1.4e-3,
        impedance=75.0,
        waveform=waveform,
    )

    port = sim._coaxial_ports[0]
    assert port.face == "left"
    assert port.pin_length == pytest.approx(3.0e-3)
    assert port.pin_radius == pytest.approx(0.4e-3)
    assert port.outer_radius == pytest.approx(1.4e-3)
    assert port.impedance == pytest.approx(75.0)
    assert port.excitation is waveform


def test_setup_coaxial_port_stamps_pin_dielectric_and_gap_conductivity():
    grid = _make_grid()
    base_materials = init_materials(grid.shape)
    port = CoaxialPort(
        position=(0.010, 0.010, 0.015),
        face="top",
        pin_length=5e-3,
        pin_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        impedance=50.0,
        excitation=ConstantWaveform(0.0),
    )

    stamped = setup_coaxial_port(grid, port, base_materials)

    pin_idx = grid.position_to_index((0.010, 0.010, 0.0125))
    ptfe_idx = grid.position_to_index((0.0110, 0.010, 0.0125))
    shell_idx = grid.position_to_index((0.0120, 0.010, 0.0125))
    free_idx = grid.position_to_index((0.0140, 0.010, 0.0125))
    gap_idx = grid.position_to_index(port.position)

    assert float(stamped.eps_r[pin_idx]) == pytest.approx(1.0)
    assert float(stamped.sigma[pin_idx]) >= PEC_SIGMA

    assert float(stamped.eps_r[ptfe_idx]) == pytest.approx(PTFE_EPS_R)
    assert float(stamped.sigma[ptfe_idx]) == pytest.approx(0.0)

    assert float(stamped.eps_r[shell_idx]) == pytest.approx(1.0)
    assert float(stamped.sigma[shell_idx]) >= PEC_SIGMA

    assert float(stamped.eps_r[free_idx]) == pytest.approx(1.0)
    assert float(stamped.sigma[free_idx]) == pytest.approx(0.0)

    sigma_port = 1.0 / (port.impedance * grid.dx)
    assert float(stamped.sigma[gap_idx]) >= sigma_port


def test_coaxial_tem_analytic_helpers_match_closed_form_identities():
    z0 = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS,
        SMA_OUTER_RADIUS,
        PTFE_EPS_R,
    )
    c_per_m = coaxial_tem_capacitance_per_m(
        SMA_PIN_RADIUS,
        SMA_OUTER_RADIUS,
        PTFE_EPS_R,
    )
    l_per_m = coaxial_tem_inductance_per_m(SMA_PIN_RADIUS, SMA_OUTER_RADIUS)

    assert z0 == pytest.approx((l_per_m / c_per_m) ** 0.5)
    # Standard PTFE-filled SMA dimensions are close to a 50-ohm line.
    assert z0 == pytest.approx(48.6, rel=0.03)


def test_coaxial_tem_phase_and_load_reflection_oracles():
    import numpy as np
    from rfx.grid import C0

    freqs = np.array([1.0e9, 2.0e9, 3.0e9])
    beta = np.asarray(coaxial_tem_phase_constant(freqs, eps_r=PTFE_EPS_R))
    expected_beta = 2.0 * np.pi * freqs * np.sqrt(PTFE_EPS_R) / C0
    np.testing.assert_allclose(beta, expected_beta, rtol=2e-7, atol=0.0)

    z0 = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS,
        SMA_OUTER_RADIUS,
        PTFE_EPS_R,
    )
    gamma_matched = complex(coaxial_load_reflection(z0, z0))
    gamma_short = complex(coaxial_load_reflection(0.0, z0))
    gamma_open = complex(coaxial_load_reflection(np.inf, z0))
    gamma_25 = complex(coaxial_load_reflection(25.0, z0))

    assert gamma_matched == pytest.approx(0.0 + 0.0j, abs=1e-7)
    assert gamma_short == pytest.approx(-1.0 + 0.0j, abs=1e-7)
    assert gamma_open == pytest.approx(1.0 + 0.0j, abs=1e-7)
    assert gamma_25.real < 0.0


def test_coaxial_tem_reference_plane_vi_recovers_synthetic_tem_fields():
    import numpy as np

    inner = SMA_PIN_RADIUS
    outer = SMA_OUTER_RADIUS
    eps_r = PTFE_EPS_R
    z0 = coaxial_tem_characteristic_impedance(inner, outer, eps_r)
    radial_positions = np.linspace(inner, outer, 4097)
    h_radius = 0.5 * (inner + outer)
    phi = np.linspace(0.0, 2.0 * np.pi, 65, endpoint=False)
    gamma = np.asarray([0.0 + 0.0j, -0.35 + 0.2j, 0.25 - 0.1j])
    incident_voltage = np.asarray([1.0 + 0.0j, 0.8 - 0.1j, 0.7 + 0.2j])
    voltage = incident_voltage * (1.0 + gamma)
    current = incident_voltage * (1.0 - gamma) / z0
    e_radial = voltage[:, None] / (
        radial_positions[None, :] * np.log(outer / inner)
    )
    h_phi = np.broadcast_to(
        (current / (2.0 * np.pi * h_radius))[:, None],
        (current.size, phi.size),
    )

    extracted = coaxial_tem_reference_plane_vi(
        radial_positions,
        e_radial,
        h_phi,
        h_sample_radius_m=h_radius,
        inner_radius=inner,
        outer_radius=outer,
        eps_r=eps_r,
    )
    s11 = coaxial_tem_reference_plane_s11(extracted.voltage, extracted.current, z0)

    np.testing.assert_allclose(extracted.voltage, voltage, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(extracted.current, current, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(s11, gamma, rtol=3e-7, atol=3e-7)


def test_coaxial_tem_reference_plane_cartesian_adapter_recovers_tem_fields():
    import numpy as np

    inner = SMA_PIN_RADIUS
    outer = SMA_OUTER_RADIUS
    eps_r = PTFE_EPS_R
    z0 = coaxial_tem_characteristic_impedance(inner, outer, eps_r)
    center_u = 0.0
    center_v = 0.0
    span = 2.35e-3
    u = np.linspace(-span, span, 601)
    v = np.linspace(-span, span, 601)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    rr = np.hypot(uu - center_u, vv - center_v)
    cos_phi = np.divide(uu - center_u, rr, out=np.zeros_like(rr), where=rr > 0.0)
    sin_phi = np.divide(vv - center_v, rr, out=np.zeros_like(rr), where=rr > 0.0)
    nonzero_r = rr > 0.0

    gamma = np.asarray([0.0 + 0.0j, -0.2 + 0.35j])
    incident_voltage = np.asarray([1.0 + 0.0j, 0.7 - 0.15j])
    voltage = incident_voltage * (1.0 + gamma)
    current = incident_voltage * (1.0 - gamma) / z0
    e_radial = np.zeros((gamma.size,) + rr.shape, dtype=np.complex128)
    h_phi = np.zeros_like(e_radial)
    e_radial[:, nonzero_r] = voltage[:, None] / (
        rr[nonzero_r][None, :] * np.log(outer / inner)
    )
    h_phi[:, nonzero_r] = current[:, None] / (2.0 * np.pi * rr[nonzero_r][None, :])
    e_u = e_radial * cos_phi
    e_v = e_radial * sin_phi
    h_u = -h_phi * sin_phi
    h_v = h_phi * cos_phi

    extracted = coaxial_tem_reference_plane_vi_from_cartesian_plane(
        u,
        v,
        e_u,
        e_v,
        h_u,
        h_v,
        center_u_m=center_u,
        center_v_m=center_v,
        inner_radius=inner,
        outer_radius=outer,
        eps_r=eps_r,
        radial_positions_m=np.linspace(inner, outer, 257),
        h_sample_radius_m=0.5 * (inner + outer),
        azimuthal_angles_rad=np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False),
    )
    s11 = coaxial_tem_reference_plane_s11(
        extracted.vi.voltage,
        extracted.vi.current,
        z0,
    )

    np.testing.assert_allclose(extracted.vi.voltage, voltage, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(extracted.vi.current, current, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(s11, gamma, rtol=2e-3, atol=2e-3)


def test_coaxial_tem_reference_plane_vi_validates_sampling_geometry():
    import numpy as np

    radial_positions = np.asarray([SMA_PIN_RADIUS, SMA_OUTER_RADIUS])
    e_radial = np.ones((1, 2))
    h_phi = np.ones((1, 4))

    with pytest.raises(ValueError, match="strictly increasing"):
        coaxial_tem_reference_plane_vi(
            radial_positions[::-1],
            e_radial,
            h_phi,
            h_sample_radius_m=1.0e-3,
            inner_radius=SMA_PIN_RADIUS,
            outer_radius=SMA_OUTER_RADIUS,
        )


@pytest.mark.parametrize(("face", "position", "component"), FACE_CASES)
def test_make_coaxial_port_source_injects_expected_e_field_component(face, position, component):
    grid = _make_grid()
    waveform = ConstantWaveform(2.5)
    port = CoaxialPort(
        position=position,
        face=face,
        pin_length=5e-3,
        pin_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        impedance=50.0,
        excitation=waveform,
    )
    materials = setup_coaxial_port(grid, port, init_materials(grid.shape))
    source = make_coaxial_port_source(grid, port, materials, n_steps=8)
    state = init_state(grid.shape)

    updated = source(state, 0.0)
    gap_idx = grid.position_to_index(position)

    eps = float(materials.eps_r[gap_idx]) * EPS_0
    sigma = float(materials.sigma[gap_idx])
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)
    expected_delta = cb * waveform.amplitude / grid.dx

    for field_name in ("ex", "ey", "ez"):
        field = getattr(updated, field_name)
        observed = float(field[gap_idx])
        if field_name == component:
            assert observed == pytest.approx(expected_delta)
        else:
            assert observed == pytest.approx(0.0)
