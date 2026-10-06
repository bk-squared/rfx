"""Port Cb and RLC D0 consume the stored, stamped E-update operands."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core import yee
from rfx.core.yee import EPS_0
from rfx.lumped import edge_update_denominator
from rfx.model.materials import e_update_material_at, with_components
from rfx.sources.port_drive import PortSourceQueue, port_drive_waveform
from types import SimpleNamespace


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("periodic", [(False, False, False), (True, True, True)])
def test_drive_and_denominator_match_stored_update(component, periodic):
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    transverse = (axis + 1) % 3
    shape = (4, 4, 4)
    cell = (0, 0, 0) if periodic[0] else (2, 2, 2)
    slab = (jnp.indices(shape)[transverse] >= 2)
    eps = jnp.where(slab, 4., 1.)
    sigma = jnp.where(slab, .2, .0)
    stamp = jnp.zeros(shape).at[cell].set(3.)
    parts = tuple(stamp if a == axis else None for a in range(3))
    raw = yee.MaterialArrays(eps + stamp, sigma + stamp, jnp.ones(shape),
                             eps_r_lumped=parts, sigma_lumped=parts)
    mats = with_components(raw, None, periodic=periodic)
    grid = SimpleNamespace(dt=1e-12)
    # Independent grid-wide E-update coefficients; never the cell helper.
    cb = yee.e_component_coeffs(mats, grid.dt, periodic)[1][axis][cell]
    wave = port_drive_waveform(grid, cell, component, jnp.ones_like, 3, mats,
                               sigma_port=2., unit_field=4.)
    np.testing.assert_array_equal(wave, jnp.full(3, cb * 2. * 4.))
    e, s = e_update_material_at(mats, cell, component)
    assert float(e) != float(raw.eps_r[cell])
    expected = float(e) * EPS_0 / grid.dt + float(s) / 2.
    assert edge_update_denominator(mats, cell, component, grid.dt, as_float=True) == expected
    # Future rules may differ from today's mean: only the stored operand moves.
    changed = list(mats.components.eps_update)
    changed[axis] = changed[axis].at[cell].multiply(1.7)
    changed = mats._replace(components=replace(mats.components, eps_update=tuple(changed)))
    changed_cb = yee.e_component_coeffs(changed, grid.dt, periodic)[1][axis][cell]
    changed_wave = port_drive_waveform(grid, cell, component, jnp.ones_like, 3, changed,
                                       sigma_port=2., unit_field=4.)
    np.testing.assert_array_equal(changed_wave, jnp.full(3, changed_cb * 2. * 4.))
    assert not np.array_equal(wave, changed_wave)
    assert edge_update_denominator(changed, cell, component, grid.dt) != edge_update_denominator(mats, cell, component, grid.dt)


def test_queue_preserves_source_order_and_supplies_final_materials():
    queue = PortSourceQueue()
    queue.append(("first",))
    seen = []

    def build(label, *, materials):
        seen.append(materials)
        return [(label,), ("second edge",)]

    queue.defer(build, "port")
    queue.append(("last",))
    assert not seen
    final = object()
    assert queue.resolve(final) == [("first",), ("port",), ("second edge",), ("last",)]
    assert seen == [final]


def test_realized_edge_material_gradient_matches_legacy():
    def loss(scale, realized):
        eps = jnp.ones((4, 4, 4)).at[2:].set(4. * scale)
        mats = yee.MaterialArrays(eps, jnp.zeros_like(eps), jnp.ones_like(eps))
        if realized:
            mats = with_components(mats, None, periodic=(False,) * 3)
        wave = port_drive_waveform(SimpleNamespace(dt=1e-12), (2, 2, 2), "ez",
                                  jnp.ones_like, 2, mats, sigma_port=2., unit_field=4.)
        return jnp.sum(wave) + edge_update_denominator(mats, (2, 2, 2), "ez", 1e-12)

    a = jax.value_and_grad(lambda x: loss(x, False))(jnp.float32(1.))
    b = jax.value_and_grad(lambda x: loss(x, True))(jnp.float32(1.))
    np.testing.assert_allclose(a, b, rtol=1e-4)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("kind", ["lumped", "wire", "msl"])
def test_main_paths_build_drives_and_rlc_after_realization(monkeypatch, graded, forward, kind):
    from rfx import Box, Simulation
    from rfx.model import materials as model
    from rfx.sources import port_drive
    from tests.contracts.test_measurement_plan import setup
    import rfx.lumped as lumped
    import rfx.runners.nonuniform as nu_runner

    kwargs = dict(freq_max=8e9, domain=(.012, .011, .012), dx=.001,
                  boundary="cpml", cpml_layers=2)
    if graded:
        kwargs["dx_profile"] = np.array([.001]*4 + [.0009, .0011] + [.001]*6)
    sim = Simulation(**kwargs)
    sim.add_material("slab", eps_r=4.)
    sim.add(Box((.005, .002, .002), (.010, .009, .010)), material="slab")
    if kind == "msl":
        sim.add(Box((.002, .002, .003), (.010, .009, .003)), material="pec")
        sim.add(Box((.002, .004, .005), (.010, .006, .005)), material="pec")
        sim.add_msl_port(position=(.005, .005, .003), width=.002, height=.002,
                         direction="+x", impedance=50., waveform=jnp.ones_like, eps_r_sub=4.)
    else:
        sim.add_port((.005, .005, .005), "ez", waveform=jnp.ones_like,
                     extent=.002 if kind == "wire" else None)
        if kind == "lumped":
            sim.add_port((.005, .005, .005), "ez", impedance=75., waveform=jnp.ones_like)
    for j, topology in enumerate(("series", "parallel")):
        sim.add_lumped_rlc((.005, .006 + .001*j, .005), "ez",
                           R=70., L=2e-9, C=.3e-12, topology=topology)
    original_realize = model.with_components
    original_drive = port_drive.port_drive_waveform
    original_denominator = lumped.edge_update_denominator
    calls = {"realize": 0, "drive": 0, "rlc": 0}

    def realize(*args, **kwargs):
        mats = original_realize(*args, **kwargs)
        calls["realize"] += 1
        # Model a later rule change without changing cells or any consumer.
        c = mats.components
        return mats._replace(components=replace(c, eps_update=tuple(e * 1.3 for e in c.eps_update)))

    def drive(grid, cell, component, excitation, n_steps, materials, **kwargs):
        assert materials.components is not None
        calls["drive"] += 1
        value = original_drive(grid, cell, component, excitation, n_steps, materials, **kwargs)
        axis = {"ex": 0, "ey": 1, "ez": 2}[component]
        cb = yee.e_component_coeffs(materials, grid.dt)[1][axis][tuple(cell)]
        np.testing.assert_array_equal(value, jnp.full(n_steps, cb * kwargs["sigma_port"] * kwargs["unit_field"]))
        return value

    def denominator(materials, cell, component, dt, *args, **kwargs):
        assert materials.components is not None
        calls["rlc"] += 1
        value = original_denominator(materials, cell, component, dt, *args, **kwargs)
        axis = {"ex": 0, "ey": 1, "ez": 2}[component]
        eps = materials.components.eps_update[axis][tuple(cell)]
        sigma = materials.components.sigma_update[axis][tuple(cell)]
        if kwargs.get("as_float"):
            assert value == float(eps) * EPS_0 / dt + float(sigma) / 2.
        else:
            np.testing.assert_array_equal(value, eps * EPS_0 / dt + sigma / 2.)
        return value

    monkeypatch.setattr(model, "with_components", realize)
    monkeypatch.setattr(port_drive, "port_drive_waveform", drive)
    monkeypatch.setattr(nu_runner, "port_drive_waveform", drive)
    monkeypatch.setattr(lumped, "edge_update_denominator", denominator)
    setup(sim, forward=forward)
    assert calls["realize"] == 1
    assert calls["drive"] > 0
    assert calls["rlc"] == 2
