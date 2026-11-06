"""Built coax Z: independent electrostatics and public S-reference wiring."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import label

from rfx.api import Simulation
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.sources.coax_impedance import coaxial_built_line_impedance
from rfx.sources.coaxial_port import (
    _coaxial_tem_edge_profile, stamp_coaxial_line,
)
from tests.unit.autodiff.test_coax_two_port_ad import _build_small_two_port_sim


def _section(h, *, small=False):
    # The main fixture has exactly the declared SMA grid and stamped mask.
    scale = 0.25 if small else 1.0
    sim = Simulation(freq_max=10e9, domain=(0.008 * scale, 0.008 * scale,
                                          8 * h if small else 0.060), dx=h)
    sim.add_coaxial_port((0.004 * scale, 0.004 * scale, 4 * h),
                        pin_radius=0.000635 * scale, outer_radius=0.002055 * scale)
    grid = sim._build_grid()
    port = sim._coaxial_ports[0]
    materials, _, _ = sim._build_materials(grid)
    k = grid.pad_z_lo + 3
    _, shell, cells = stamp_coaxial_line(
        grid, materials, center_xy=port.position[:2], z_lo_index=k - 1,
        z_hi_index=k + 1, pin_radius=port.pin_radius,
        outer_radius=port.outer_radius, shell_thickness_m=0.001 * scale)
    ex, ey = _coaxial_tem_edge_profile(grid, port, k, shell, cells)
    return grid, port, k, cells, ex, ey


def _jacobi_energy_impedance(grid, port, k, cells):
    # Read the realized conductor edges, but do not call the source's Laplace
    # solver or any impedance/energy helper. Connected metal nodes are fixed
    # at 1 V (pin) or 0 V (shell); iterate every free potential by Jacobi.
    mx, my, _ = [np.asarray(m)[:, :, k] for m in realized_pec_edge_masks(cells)]
    metal = mx | my
    metal[1:] |= mx[:-1]
    metal[:, 1:] |= my[:, :-1]
    components, _ = label(metal)
    centre = grid.position_to_index(port.position)[:2]
    pin = components == components[centre]
    assert components[centre] != 0
    phi = pin.astype(float)
    free = ~metal
    for _ in range(20000):
        neighbours = np.zeros_like(phi)
        degree = np.zeros_like(phi)
        neighbours[1:] += phi[:-1]
        neighbours[:-1] += phi[1:]
        neighbours[:, 1:] += phi[:, :-1]
        neighbours[:, :-1] += phi[:, 1:]
        degree[1:] += 1
        degree[:-1] += 1
        degree[:, 1:] += 1
        degree[:, :-1] += 1
        updated = phi.copy()
        updated[free] = (neighbours / degree)[free]
        delta = np.max(np.abs(updated - phi))
        phi = updated
        if delta < 1e-12:
            break
    else:
        pytest.fail("independent Jacobi potential did not converge")
    # Stored electrostatic energy per metre; h cancels for square edge areas.
    energy = 0.5 * 8.8541878128e-12 * 2.1 * (
        np.square(np.diff(phi, axis=0)).sum()
        + np.square(np.diff(phi, axis=1)).sum())
    return np.sqrt(2.1) / (299792458.0 * 2 * energy)


def test_sma_built_impedance_against_independent_laplace():
    grid, port, k, cells, ex, ey = _section(0.0004)
    expected = _jacobi_energy_impedance(grid, port, k, cells)
    actual = coaxial_built_line_impedance(grid, ex, ey, 2.1)
    print(f"SMA built Z={actual:.12f}, independent Jacobi Z={expected:.12f}")
    assert actual == pytest.approx(expected, rel=1e-4)
    assert actual == pytest.approx(40.546, abs=1e-3)


def test_small_coax_refinement_approaches_closed_form():
    values = []
    for h in (0.0001, 0.00005, 0.000025):
        grid, _, _, _, ex, ey = _section(h, small=True)
        values.append(coaxial_built_line_impedance(grid, ex, ey, 2.1))
    closed = np.sqrt(4e-7 * np.pi / (8.8541878128e-12 * 2.1)) * np.log(2.055 / 0.635) / (2 * np.pi)
    errors = np.abs(np.asarray(values) - closed)
    print(f"small coax Z={values}; closed={closed:.12f}; errors={errors.tolist()}")
    assert errors[2] < errors[1] < errors[0]


def test_two_port_returned_s11_uses_reported_built_impedance(monkeypatch):
    # Canonical small 8x8x12 mm AD fixture. Like the canonical two-port wiring
    # test, replace time stepping/voltage assembly with a known native line;
    # the real stamps, source solves, reference transform and result execute.
    import rfx.simulation as runner
    import rfx.sources.coaxial_port as source
    import rfx.sparams.coax as lane

    native = np.zeros((2, 2, 3), dtype=complex)
    t = np.exp(-1j * np.array([0.4, 0.8, 1.2]))
    native[0, 1] = native[1, 0] = t

    def run(grid, materials, n_steps, **kw):
        return SimpleNamespace(
            dft_planes=[SimpleNamespace(accumulator=np.zeros(3, complex))
                        for _ in kw["dft_planes"]],
            time_series=np.full((n_steps, len(kw["probes"])), 1e-6), dt=grid.dt)

    def assemble(**kw):
        return native.copy(), np.ones(3), np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.zeros((2, 2, 3), complex)

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(source, "coaxial_line_plane_voltage", lambda *a, **kw: np.zeros(3, complex))
    monkeypatch.setattr(lane, "_assemble_coaxial_two_port_from_voltages", assemble)
    result = _build_small_two_port_sim().compute_coaxial_two_port(
        n_steps=10, freqs=np.array([6e9, 8e9, 10e9]), probe_count=3,
        probe_start_cells=4, probe_spacing_cells=2)
    assert np.all(result.reference_impedances == 50)
    np.testing.assert_allclose(result.line_impedances[0], result.line_impedances[1])
    rho = (50 - result.line_impedances[0]) / (50 + result.line_impedances[0])
    expected = rho * (t * t - 1) / (1 - rho * rho * t * t)
    np.testing.assert_allclose(result.s_params[0, 0], expected, rtol=1e-12, atol=1e-12)


def test_each_port_keeps_its_own_built_impedance():
    from rfx.sparams._port_reference import coax_two_port_line_waves
    native = np.ones((2, 2, 1), complex)
    actual, z = coax_two_port_line_waves(
        native, None, None, (40.0, 45.0), center_xy=None,
        pin_radius=None, outer_radius=None, ladders=((), ()))
    np.testing.assert_array_equal(z, [40, 45])
    assert actual[1, 0, 0] == pytest.approx(np.sqrt(40 / 45))
    assert actual[0, 1, 0] == pytest.approx(np.sqrt(45 / 40))
