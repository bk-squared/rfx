"""Public MSL spatial-sampling wiring with manufactured DFTs, never FDTD.

The field oracle is affine in the physical propagation coordinate. Its
closed contour gives I = (1 - 1/4) * 5U * s(p_E) for every port direction.
Only run/forward are replaced; registration, cropping, V/I extraction and
S assembly execute normally. The material-tape test is a synthetic dataflow
contract, not a Maxwell solution or a material-gradient accuracy claim.
"""
from __future__ import annotations

from types import SimpleNamespace

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.probes.probes import DFTPlaneProbe
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

U = 2.0**-12
EPS_R = 2.0
FREQS = np.array([0.8e9, 1.2e9])
DIRECTIONS = ("+x", "-x", "+y", "-y")


def _build_case(direction, graded):
    prop = 0 if direction[-1] == "x" else 1
    width = 1 - prop
    profiles = [np.full(12, U), np.full(12, U), np.full(8, U)]
    profiles[prop] = np.full(28, U)
    if graded:
        profiles[prop][11:13] = [2 * U, 4 * U]
    prop_nodes = np.r_[0.0, np.cumsum(profiles[prop])]
    domain = tuple(float(p.sum()) for p in profiles)
    grid_kwargs = {f"d{direction[-1]}_profile": profiles[prop]} if graded else {}
    sim = Simulation(freq_max=2e9, domain=domain, dx=U, cpml_layers=2,
                     boundary="cpml", **grid_kwargs)
    sim.add_material("substrate", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (domain[0], domain[1], 2 * U)), material="substrate")
    sim.add_thin_conductor(Box((0, 0, 0), (domain[0], domain[1], 0)))
    lo, hi = [0.0, 0.0, 2 * U], [domain[0], domain[1], 2 * U]
    lo[width], hi[width] = 4 * U, 8 * U
    sim.add_thin_conductor(Box(tuple(lo), tuple(hi)))
    position = [6 * U, 6 * U, 0.0]
    position[prop] = float(prop_nodes[2 if direction[0] == "+" else 22])
    sim.add_msl_port(position=tuple(position), width=4 * U, height=2 * U,
                     direction=direction, impedance=50.0, eps_r_sub=EPS_R,
                     mode="uniform", n_probe_offset=10, n_probe_spacing=2,
                     n_probes=3)
    return sim, sim._build_realized_grid(), profiles


def _hand_coordinates(grid, profiles):
    """Coordinates derived from the declared cells, not the new stencil."""
    nodes, centres = [], []
    for axis, profile in enumerate(profiles):
        name = "xyz"[axis]
        nlo, nhi = getattr(grid, f"pad_{name}_lo"), getattr(grid, f"pad_{name}_hi")
        cells = np.r_[np.full(nlo, profile[0]), profile,
                      np.full(nhi, profile[-1]), profile[-1]]
        node = np.r_[0.0, np.cumsum(cells[:-1])] - nlo * profile[0]
        assert len(node) == grid.shape[axis]
        nodes.append(node)
        centres.append(node + cells / 2)
    return nodes, centres


def _install_manufactured_backends(monkeypatch, sim, grid, profiles, direction, graded):
    nodes, centres = _hand_coordinates(grid, profiles)
    prop = 0 if direction[-1] == "x" else 1
    width = 1 - prop
    h_width = f"h{'xyz'[width]}"
    orientation = 1 if prop == 0 else -1
    sign = 1 if direction[0] == "+" else -1
    calls, snapshots = [], []

    def record(scale):
        planes, registrations = {}, []
        for entry in sim._dft_planes:
            assert entry.axis == direction[-1]
            point = [0.0, 0.0, 0.0]
            point[prop] = entry.coordinate
            indices = (sim._pos_to_nu_index(grid, tuple(point)) if graded
                       else grid.position_to_index(tuple(point)))
            index = indices[prop]
            region = sim._dft_plane_regions[entry.name]
            lo_w, hi_w, lo_z, hi_z = region
            shape = (hi_w - lo_w, hi_z - lo_z)
            if entry.component == "ez":
                sample = nodes[prop][index]
                field = np.ones(shape)  # Existing production convention: V = sum(Ez*dz) = 2U.
                phase = np.ones(len(entry.freqs), dtype=complex)
                field_scale = 1.0
            else:
                sample = centres[prop][index]
                s = 1 + sample / (16 * U)
                if entry.component == h_width:
                    field = np.broadcast_to(centres[2][lo_z:hi_z][None, :] / U, shape)
                else:
                    assert entry.component == "hz"
                    field = np.broadcast_to(
                        0.25 * centres[width][lo_w:hi_w][:, None] / U, shape)
                field = orientation * sign * s * field
                # Invert ONLY temporal staggering here. Spatial coordinates
                # remain the actual H centres, so production must interpolate.
                phase = np.exp(-1j * 2 * np.pi * np.asarray(entry.freqs) * grid.dt / 2)
                field_scale = scale
            accumulator = (jnp.asarray(field, dtype=jnp.complex64)[None, :, :]
                           * jnp.asarray(phase, dtype=jnp.complex64)[:, None, None]
                           * field_scale)
            planes[entry.name] = DFTPlaneProbe(
                accumulator=accumulator, freqs=entry.freqs, component=entry.component,
                axis=prop, index=index, total_steps=1, window="rect", window_alpha=0.25,
                region=region,
            )
            registrations.append(dict(component=entry.component, coordinate=entry.coordinate,
                                      index=index, sample=sample, region=region,
                                      shape=accumulator.shape))
        snapshots.append(registrations)
        return SimpleNamespace(dft_planes=planes)

    def fake_run(**kwargs):
        calls.append("run")
        return record(1.0)

    def fake_forward(*, eps_override, **kwargs):
        calls.append("forward")
        assert eps_override.shape == grid.shape
        # Depend on one interior material cell: a reduction over thousands
        # of identical cells would add unrelated float32 summation error to
        # this scalar-parameter pullback, without testing another H sample.
        cell = tuple(pad + 1 for pad in grid.axis_pads)
        return record(eps_override[cell])

    monkeypatch.setattr(sim, "run", fake_run)
    monkeypatch.setattr(sim, "forward", fake_forward)
    return calls, snapshots


def _assert_h_registrations(records, grid, direction, graded):
    prop = 0 if direction[-1] == "x" else 1
    width = 1 - prop
    pad_p, pad_w, pad_z = grid.axis_pads[prop], grid.axis_pads[width], grid.axis_pads[2]
    h_records = [r for r in records if r["component"].startswith("h")]
    assert len(h_records) == 4
    assert {r["component"] for r in h_records} == {f"h{'xyz'[width]}", "hz"}
    registration = U * np.array([11, 13] if graded else [11, 12])
    samples = U * np.array([12, 15] if graded else [11.5, 12.5])
    for component in (f"h{'xyz'[width]}", "hz"):
        pair = sorted((r for r in h_records if r["component"] == component),
                      key=lambda r: r["sample"])
        np.testing.assert_array_equal([r["index"] for r in pair], [pad_p + 11, pad_p + 12])
        np.testing.assert_array_equal([r["coordinate"] for r in pair], registration)
        np.testing.assert_array_equal([r["sample"] for r in pair], samples)
        for r in pair:
            assert r["region"] == (pad_w + 3, pad_w + 9, pad_z + 1, pad_z + 3)
            assert r["shape"] == (len(FREQS), 6, 2)


def _expected_vi(graded, scale):
    # Five width-node segments and one normal segment around the thin trace.
    # The H_width pair contributes +5U*s, H_z contributes -0.25*5U*s.
    target = (13 if graded else 12) * U
    return 2 * U, scale * 0.75 * 5 * U * (1 + target / (16 * U))


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("route", ["run", "forward"])
def test_public_msl_driver_centres_current_at_voltage_plane(
    monkeypatch, tmp_path, direction, graded, route,
):
    sim, grid, profiles = _build_case(direction, graded)
    calls, snapshots = _install_manufactured_backends(
        monkeypatch, sim, grid, profiles, direction, graded)
    scale = 1.25 if route == "forward" else 1.0
    kwargs = {"eps_override": jnp.full(grid.shape, scale)} if route == "forward" else {}
    dump = tmp_path / "raw.npz"
    result = sim.compute_msl_s_matrix(
        freqs=FREQS, n_steps=1, num_periods=1, enforce_passivity=False,
        raw_3probe_dump_path=str(dump), **kwargs)
    assert calls == [route]
    _assert_h_registrations(snapshots[0], grid, direction, graded)
    voltage, current = _expected_vi(graded, scale)
    with np.load(dump) as raw:
        np.testing.assert_allclose(raw["raw_v"], voltage, rtol=2e-6, atol=1e-10)
        np.testing.assert_allclose(raw["raw_i1"], current, rtol=2e-6, atol=1e-10)
        left_p, right_p = ([12, 15] if graded else [11.5, 12.5])
        for key, position in (("raw_i1_left", left_p), ("raw_i1_same_index", right_p)):
            expected_side = scale * 0.75 * 5 * U * (1 + position / 16)
            np.testing.assert_allclose(raw[key], expected_side, rtol=2e-6, atol=1e-10)
        metadata = json.loads(str(raw["metadata_json"]))
        assert metadata["current_spatial_alignment"] == "linear_bracketing_H_to_E_node"
        np.testing.assert_allclose(metadata["current_plane_stencils"][0]["weights"],
                                   [2 / 3, 1 / 3] if graded else [0.5, 0.5])
    zref = float(hammerstad_jensen_z0_eps_eff(4 * U, 2 * U, EPS_R)[0])
    expected_s = (voltage - zref * current) / (voltage + zref * current)
    assert result.assembly == "multi_drive_solve"
    np.testing.assert_allclose(result.S, expected_s, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_public_forward_keeps_manufactured_material_sensitivity(monkeypatch, direction, graded):
    sim, grid, profiles = _build_case(direction, graded)
    calls, snapshots = _install_manufactured_backends(
        monkeypatch, sim, grid, profiles, direction, graded)

    def objective(scale):
        result = sim.compute_msl_s_matrix(
            freqs=FREQS, n_steps=1, num_periods=1, enforce_passivity=False,
            eps_override=jnp.full(grid.shape, scale))
        return jnp.mean(jnp.real(result.S[0, 0]))

    scale = 1.25
    value, derivative = jax.value_and_grad(objective)(jnp.asarray(scale, dtype=jnp.float32))
    assert calls == ["forward"]
    _assert_h_registrations(snapshots[0], grid, direction, graded)
    voltage, unit_current = _expected_vi(graded, 1.0)
    zi = float(hammerstad_jensen_z0_eps_eff(4 * U, 2 * U, EPS_R)[0]) * unit_current
    expected_s = (voltage - zi * scale) / (voltage + zi * scale)
    expected_derivative = -2 * voltage * zi / (voltage + zi * scale)**2
    assert abs(expected_derivative) > 1e-4
    np.testing.assert_allclose(value, expected_s, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=3e-5, atol=1e-8)
