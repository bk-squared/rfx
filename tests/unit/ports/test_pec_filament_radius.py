"""Positive subcell PEC radii refuse; legacy filaments and volumes retain ownership.

The declared second witness used main 4725b748 at a/d=0.200 as its independent
reference. At a=0.0375 mm, d=0.75 mm the attempted correction missed the 2%
8--14 GHz gate (4.9321% uniform), and the 0.75->0.375 mm X change at 11.3 GHz
was 2.5833 ohms against 1 ohm. No filament field correction is shipped.
The separate full-height wire-port oracle remains in test_wire_port_radius.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import PolylineWire
from tests.unit.ports.test_wire_port_radius import _plates, RADIUS


REFUSAL = r"resolve the wire as a volume.*a >= 0\.5\*d"


def _filament_plates(dx, lane, *, radius=RADIUS, port_radius=None):
    sim = _plates(dx, lane, radius=port_radius)
    sim._ports[0] = replace(sim._ports[0], extent=dx)
    sim.add(PolylineWire(((.004, .004, dx), (.004, .004, .0015)),
                         radius=radius), material="pec")
    return sim


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("ratio", [1e-9, .05, .1, .2, .2001, .3, .499999])
def test_every_positive_subcell_radius_refuses_before_scan(lane, entry, ratio, monkeypatch):
    sim = _filament_plates(.0005, lane, radius=ratio*.0005)
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped"))
    with pytest.raises(ValueError, match=REFUSAL):
        getattr(sim, entry)(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("ratio", [1e-9, .05, .2, .3, np.nextafter(.5, 0.)])
def test_classifier_refuses_the_whole_positive_filament_interval(ratio):
    from rfx.geometry.rasterize_grid import wire_filament_nodes
    nodes = tuple(np.arange(10)*.001 for _ in range(3))
    sizes = tuple(np.full(10, .001) for _ in range(3))
    with pytest.raises(ValueError, match=REFUSAL):
        wire_filament_nodes(((.004, .004, .002), (.004, .004, .007)),
                            ratio*.001, nodes, sizes)


@pytest.mark.parametrize("ratio,kind", [(0., "filament"), (.5, "volume"), (1., "volume")])
def test_zero_radius_and_volume_endpoints_keep_ownership(ratio, kind):
    from rfx.geometry.rasterize_grid import wire_filament_nodes
    nodes = tuple(np.arange(10)*.001 for _ in range(3))
    sizes = tuple(np.full(10, .001) for _ in range(3))
    result = wire_filament_nodes(((.004, .004, .002), (.004, .004, .007)),
                                 ratio*.001, nodes, sizes)
    assert (result is None) == (kind == "volume")
    if kind == "filament":
        assert result == [(4, 4, 2), (4, 4, 7)]


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("port_radius", [None, RADIUS])
def test_legacy_zero_radius_filament_runs_with_either_port(lane, entry, port_radius):
    sim = _filament_plates(.0005, lane, radius=0., port_radius=port_radius)
    result = getattr(sim, entry)(n_steps=100, skip_preflight=True)
    fields = np.asarray(result.time_series)
    assert np.isfinite(fields).all() and np.max(abs(fields)) > 0


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_filament_without_radius_port_refuses_on_dispersive_board(lane, kind, monkeypatch):
    from rfx import Box, DebyePole, LorentzPole
    sim = _filament_plates(.0005, lane)
    poles = (dict(debye_poles=[DebyePole(delta_eps=1., tau=1e-11)]) if kind == "debye"
             else dict(lorentz_poles=[LorentzPole(omega_0=2*np.pi*20e9,
                                                delta=1e9, kappa=(2*np.pi*20e9)**2)]))
    sim.add_material("dispersive", eps_r=2., **poles)
    sim.add(Box((.001, .001, .0005), (.002, .002, .001)), material="dispersive")
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped"))
    with pytest.raises(ValueError, match=REFUSAL):
        sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("entry", ["adi_run", "adi_forward", "subgridded",
                                   "distributed_v2", "distributed_nu", "upml"])
def test_other_filament_paths_refuse_before_scan(entry, monkeypatch):
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
def test_outer_jit_cannot_silently_drop_a_declared_filament_radius(lane):
    sim = _filament_plates(.0005, lane)
    shape = sim._build_realized_grid().shape
    solve = jax.jit(lambda eps: sim.forward(eps_override=jnp.full(shape, eps), n_steps=2,
                                          skip_preflight=True).time_series)
    with pytest.raises(ValueError, match=REFUSAL):
        solve(jnp.array(1.4))


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
