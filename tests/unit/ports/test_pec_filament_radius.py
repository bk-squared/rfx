"""Declared PEC filament radii use the same component materials as wire ports."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import PolylineWire
from rfx.sources.wire_radius import prepare_pec_wire_radii
from tests.unit.ports.test_wire_port_radius import (
    _plates, _impedance, _oracle, RADIUS,
)


def _filament_plates(dx, lane, *, radius=RADIUS, port_radius=RADIUS):
    sim = _plates(dx, lane, radius=port_radius)
    sim._ports[0] = replace(sim._ports[0], extent=dx)
    sim.add(PolylineWire(((.004, .004, dx), (.004, .004, .0015)),
                         radius=radius), material="pec")
    return sim


def _assert_filament_oracle(z):
    np.testing.assert_array_less(np.abs(z-_oracle())/np.abs(_oracle()), .02)
    assert abs(z[1, 1].imag-z[0, 1].imag) <= 1.0


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_one_edge_feed_with_pec_filament_oracle(lane):
    z = np.stack([_impedance(_filament_plates(dx, lane)) for dx in (.5e-3, .25e-3)])
    _assert_filament_oracle(z)


@pytest.mark.parametrize("failure", ["error", "drift"])
def test_filament_oracle_assertions_are_live(failure):
    z = np.stack([_oracle(), _oracle()])
    if failure == "error":
        z += 15j
    else:
        z[0] -= .7j
        z[1] += .7j
    with pytest.raises(AssertionError):
        _assert_filament_oracle(z)


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_filament_radius_changes_fields_with_no_radius_port(lane):
    fields = []
    for radius in (0., RADIUS):
        sim = _filament_plates(.0005, lane, radius=radius, port_radius=None)
        result = sim.run(n_steps=120, skip_preflight=True, compute_s_params=False)
        fields.append(np.asarray(result.time_series))
    assert np.isfinite(fields).all()
    assert not np.array_equal(*fields)
    assert np.max(abs(fields[1]-fields[0])) > 1e-4*np.max(abs(fields[0]))


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
@pytest.mark.parametrize("ratio", [.2001, .3, .499])
def test_filament_refused_band_before_scan(lane, ratio, monkeypatch):
    sim = _filament_plates(.0005, lane, radius=ratio*.0005, port_radius=None)
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped"))
    with pytest.raises(ValueError, match="refine.*volume.*coarsen"):
        sim.run(n_steps=1, skip_preflight=True)


def test_realized_filament_retains_radius_and_stamps_only_once():
    sim = _filament_plates(.0005, "uniform")
    grid = sim._build_grid()
    wires = []
    materials = sim._assemble_materials(grid, pec_wires=wires, pec_sheets=[])[0]
    assert wires[0].radius == RADIUS
    assert materials.mu_r_wire is None
    stamped, prepared = prepare_pec_wire_radii(grid, materials, wires)
    assert stamped.mu_r_wire is not None
    assert prepared[0].radius_stamped
    assert prepare_pec_wire_radii(grid, stamped, prepared)[0] is stamped


@pytest.mark.parametrize("ratio,kind", [(0., "filament"), (.2, "filament"), (.5, "volume"), (1., "volume")])
def test_radius_endpoints_keep_filament_volume_ownership(ratio, kind):
    from rfx.geometry.rasterize_grid import wire_filament_nodes
    nodes = tuple(np.arange(10)*.001 for _ in range(3))
    sizes = tuple(np.full(10, .001) for _ in range(3))
    result = wire_filament_nodes(((.004, .004, .002), (.004, .004, .007)),
                                 ratio*.001, nodes, sizes)
    assert (result is None) == (kind == "volume")


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_filament_forward_material_gradient(lane):
    sim = _filament_plates(.0005, lane)
    shape = sim._build_realized_grid().shape

    def objective(eps):
        r = sim.forward(eps_override=jnp.ones(shape)*eps, n_steps=100,
                        skip_preflight=True)
        return jnp.log(jnp.mean(r.time_series**2))

    value, grad = jax.jit(jax.value_and_grad(objective))(jnp.array(1.4))
    fd = (objective(1.405)-objective(1.395))/.01
    assert np.isfinite(value) and abs(fd) > 1e-3
    np.testing.assert_allclose(grad, fd, rtol=1e-3)


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_filament_without_radius_port_refuses_dispersion(lane, kind, monkeypatch):
    from rfx import Box, DebyePole, LorentzPole
    sim = _filament_plates(.0005, lane, port_radius=None)
    poles = (dict(debye_poles=[DebyePole(delta_eps=1., tau=1e-11)]) if kind == "debye"
             else dict(lorentz_poles=[LorentzPole(omega_0=2*np.pi*20e9,
                                                delta=1e9, kappa=(2*np.pi*20e9)**2)]))
    sim.add_material("dispersive", eps_r=2., **poles)
    sim.add(Box((.001, .001, .0005), (.002, .002, .001)), material="dispersive")
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped"))
    with pytest.raises(NotImplementedError, match="radius"):
        sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("entry", ["adi_run", "adi_forward", "subgridded",
                                   "distributed_v2", "distributed_nu", "upml"])
def test_unsupported_filament_refuses_before_scan(entry, monkeypatch):
    from rfx import Simulation
    from rfx.runners.distributed_v2 import run_distributed
    opts = dict(solver="adi") if entry.startswith("adi") else {}
    if entry == "distributed_nu":
        opts["dx_profile"] = np.full(12, .0005)
    sim = Simulation(freq_max=30e9, domain=(.006,)*3, dx=.0005,
                     boundary="upml" if entry == "upml" else "pec", **opts)
    sim.add(PolylineWire(((.003, .003, .001), (.003, .003, .004)), radius=RADIUS),
            material="pec")
    if entry == "subgridded":
        sim.add_refinement(z_range=(.001, .004), ratio=2)
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped"))
    with pytest.raises((NotImplementedError, ValueError), match="(?i)(radius|wire|PEC)"):
        if entry == "distributed_v2":
            run_distributed(sim, n_steps=1, devices=jax.devices("cpu")[:2])
        elif entry == "distributed_nu":
            sim.forward(n_steps=1, distributed=True, devices=jax.devices("cpu")[:2], skip_preflight=True)
        elif entry == "adi_forward":
            sim.forward(n_steps=1, skip_preflight=True)
        else:
            sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_outer_jit_forward_carries_filament_record_without_radius_port(lane, monkeypatch):
    from rfx.core import yee
    sim = _filament_plates(.0005, lane, port_radius=None)
    observed = []
    owner = yee.component_h_materials

    def check(materials, periodic=(False, False, False)):
        observed.append(materials.mu_r_wire is not None)
        return owner(materials, periodic)

    monkeypatch.setattr(yee, "component_h_materials", check)
    yee.update_h.clear_cache()
    shape = sim._build_realized_grid().shape
    solve = jax.jit(lambda eps: sim.forward(eps_override=jnp.full(shape, eps), n_steps=2,
                                          skip_preflight=True).time_series)
    jax.block_until_ready(solve(jnp.array(1.4)))
    assert observed and all(observed)


def test_traced_mesh_cannot_silently_reclassify_a_radius_filament():
    from rfx.geometry.rasterize_grid import GridCoords, classify_pec_entry
    nodes = jnp.arange(8)*.001
    sizes = (np.full(8, .001),)*3
    wire = PolylineWire(((.003, .003, .002), (.003, .003, .005)), radius=.0001)

    def classify(x):
        coords = GridCoords(x=x, y=nodes, z=nodes, shape=(8,)*3)
        return classify_pec_entry(wire, coords, coords, sizes)

    with pytest.raises(NotImplementedError, match="static lattice metrics"):
        jax.jit(classify)(nodes)
