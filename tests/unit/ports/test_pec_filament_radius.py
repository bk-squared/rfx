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


@pytest.mark.parametrize("graded", [False, True])
def test_traced_mesh_cannot_silently_reclassify_a_radius_filament(graded):
    from rfx.geometry.rasterize_grid import GridCoords, classify_pec_entry
    nodes = np.arange(8)*.001
    y_sizes = np.full(8, .001)
    if graded:
        y_sizes[0] = .0001  # Remote minimum must not replace the LOCAL cell.
    y_nodes = np.r_[0., np.cumsum(y_sizes[:-1])]
    sizes = (np.full(8, .001), y_sizes, np.full(8, .001))
    wire = PolylineWire(((.003, .003, .002), (.003, .003, .005)), radius=.0003)

    def classify(x):
        coords = GridCoords(x=x, y=y_nodes, z=nodes, shape=(8,)*3)
        return classify_pec_entry(wire, coords, coords, sizes)

    with pytest.raises(ValueError, match=REFUSAL):
        jax.jit(classify)(nodes)


@pytest.mark.parametrize("ratio,cells", [(.5, 4), (.6, 4), (1.2, 22)])
def test_traced_mesh_resolved_wire_keeps_volume_mask(ratio, cells):
    from rfx.geometry.rasterize_grid import (
        GridCoords, classify_pec_entry, pec_volume_cell_mask,
    )
    nodes = np.arange(8)*.001
    sizes = (np.full(8, .001),)*3
    wire = PolylineWire(((.003, .003, .002), (.003, .003, .005)), radius=ratio*.001)

    def masks(x):
        coords = GridCoords(x=x, y=nodes, z=nodes, shape=(8,)*3)
        mask, sheet, filament = classify_pec_entry(wire, coords, coords, sizes)
        assert sheet is None and filament is None
        # origin/main sends a traced wire straight to this volume sampler.
        return mask, pec_volume_cell_mask(wire, coords, sizes)

    for displacement in (0., .00015):
        x = nodes + displacement*np.sin(np.linspace(0., np.pi, 8))
        mask, main_volume = jax.jit(masks)(x)
        assert np.asarray(mask).tobytes() == np.asarray(main_volume).tobytes()
        assert int(np.sum(mask)) == cells


@pytest.mark.parametrize("ratio", [.3, .6, 1.2])
def test_traced_profile_without_nominal_metrics_refuses(ratio):
    from rfx.geometry.rasterize_grid import (
        cell_sizes_from_nonuniform_grid, centres_from_nonuniform_grid,
        classify_pec_entry, coords_from_nonuniform_grid,
    )
    from rfx.nonuniform import make_nonuniform_grid
    sizes = np.full(7, .001)
    wire = PolylineWire(((.003, .003, .002), (.003, .003, .005)), radius=ratio*.001)

    def classify(profile):
        grid = make_nonuniform_grid((.007, .007), sizes, .001, 0,
                                    dx_profile=profile)
        assert grid.dx_arr_f64 is None
        coords = coords_from_nonuniform_grid(grid)
        return classify_pec_entry(wire, coords, centres_from_nonuniform_grid(grid, coords),
                                  cell_sizes_from_nonuniform_grid(grid))

    with pytest.raises(NotImplementedError, match="x cell sizes are traced.*no nominal untraced"):
        jax.jit(classify)(sizes)


def test_traced_nodes_without_static_cell_sizes_refuse():
    from rfx.geometry.rasterize_grid import GridCoords, classify_pec_entry
    nodes = np.arange(8)*.001
    wire = PolylineWire(((.003, .003, .002), (.003, .003, .005)), radius=.0006)

    def classify(x):
        coords = GridCoords(x=x, y=nodes, z=nodes, shape=(8,)*3)
        return classify_pec_entry(wire, coords, coords)

    with pytest.raises(NotImplementedError, match="x cell sizes are traced.*no nominal untraced"):
        jax.jit(classify)(nodes)


def test_static_graded_profile_without_static_nodes_refuses():
    from rfx.geometry.rasterize_grid import GridCoords, classify_pec_entry
    nodes = np.arange(8)*.001
    x_sizes = np.full(8, .001)
    x_sizes[0] = .0001
    sizes = (x_sizes, np.full(8, .001), np.full(8, .001))
    wire = PolylineWire(((.003, .003, .002), (.003, .003, .005)), radius=.0006)

    def classify(x):
        coords = GridCoords(x=x, y=nodes, z=nodes, shape=(8,)*3)
        return classify_pec_entry(wire, coords, coords, sizes)

    with pytest.raises(NotImplementedError, match="graded x profile has no static node axis"):
        jax.jit(classify)(nodes)


@pytest.mark.parametrize("traced_mesh", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_traced_radius_still_refuses_with_static_metrics(traced_mesh, periodic):
    from rfx.geometry.rasterize_grid import GridCoords, classify_pec_entry
    from rfx.grid import Grid
    nodes = np.arange(8)*.001
    sizes = (np.full(8, .001),)*3
    grid = (Grid(10e9, (.008,)*3, dx=.001, cpml_layers=0, periodic_axes="xyz")
            if periodic else None)

    def classify(radius, x):
        coords = GridCoords(x=x if traced_mesh else nodes, y=nodes, z=nodes, shape=(8,)*3)
        wire = PolylineWire(((.003, .003, .002), (.003, .003, .005)), radius=radius)
        return classify_pec_entry(wire, coords, coords, sizes, grid=grid)

    with pytest.raises(NotImplementedError, match="a traced radius is not supported"):
        jax.jit(classify)(.0006, nodes)
