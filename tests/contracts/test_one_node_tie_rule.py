"""Declared coordinates share lower-node ties across their realization sites."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation
from rfx.geometry.rasterize_grid import (
    _nearest_plane, _static_wire_min_cell, cell_sizes_from_nonuniform_grid,
    cell_sizes_from_uniform_grid, coords_from_nonuniform_grid,
    coords_from_uniform_grid, wire_vertex_nodes,
)


def model(lane):
    kw = dict(freq_max=10e9, domain=(.02, .02, .02), dx=.001, boundary="pec")
    if lane == "graded":
        # Truly graded, with the declared witnesses in the 1 mm portion.
        d = np.array([.001] * 12 + [.0009, .0011] * 3 + [.001] * 2)
        kw.update(dx_profile=d, dy_profile=d, dz_profile=d)
    sim = Simulation(**kw)
    grid = sim._build_realized_grid()
    coords = (coords_from_nonuniform_grid(grid) if lane == "graded"
              else coords_from_uniform_grid(grid))
    sizes = (cell_sizes_from_nonuniform_grid(grid) if lane == "graded"
             else cell_sizes_from_uniform_grid(grid))
    return sim, grid, coords, sizes


KINDS = ["point", "interval", "wire_vertex", "wire_radius", "sheet",
         "traced_sheet", "csg_thin_box", "aperture", "flux", "current_window", "thin_wire",
         "port_wire", "source_wire", "lumped_wire", "port_sheet", "port_lossy", "port_face",
         "probe_sheet", "dft_sheet", "subgrid_bound", "disjoint_bound",
         "disjoint_point", "preflight_node"]


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("kind", KINDS)
def test_declared_half_node_is_lower_and_colocated(kind, lane, monkeypatch):
    sim, grid, coords, sizes = model(lane)
    from rfx.nonuniform import position_to_index
    lookup = (lambda p: position_to_index(grid, p)) if lane == "graded" else grid.position_to_index
    nodes = (coords.x, coords.y, coords.z)
    # Node 10 is a useful argmin witness: decimal 10.5 mm is a few ULP
    # nearer node 11. Node 3 witnesses Python's old half-to-even rounding.
    if kind == "point":
        assert lookup((.0035, .0035, .0035)) == (3, 3, 3)
        assert tuple(grid.index_of(a, .0035) for a in range(3)) == (3, 3, 3)
    elif kind == "interval":
        if lane == "uniform":
            actual = grid.interval_to_indices((.0035,) * 3, (.0095,) * 3)
        else:
            actual = (lookup((.0035,) * 3), lookup((.0095,) * 3))
        assert actual == ((3, 3, 3), (9, 9, 9))
    elif kind == "wire_vertex":
        indices, _ = wire_vertex_nodes([(.0105, .0105, .0105)], nodes, sizes)
        assert indices == [(10, 10, 10)]
    elif kind == "wire_radius":
        # A varying forward cell makes a reverted nearest-node site visible.
        line = np.arange(15) * .001
        d = np.full(15, .001)
        d[10] = .0009
        assert _static_wire_min_cell([(.0105,) * 3], (line,) * 3, (d,) * 3) == .0009
    elif kind == "sheet":
        assert _nearest_plane(coords.z, .0105, .001) == 10
    elif kind == "traced_sheet":
        from jax import enable_x64
        from rfx.geometry.rasterize_grid import sheet_footprint_traced
        with enable_x64():
            mask = sheet_footprint_traced(Box((.003, .003, .0105), (.009, .009, .0105)), coords, 2)
        assert set(np.nonzero(np.asarray(mask))[2]) == {10}
    elif kind == "csg_thin_box":
        # One ULP above this lane's computed midpoint is inside the tie
        # band, but a reverted argmin chooses the upper node on both lanes.
        mid = np.nextafter(.5 * (coords.z[10] + coords.z[11]), np.inf)
        shape = Box((.003, .003, mid - .0001), (.009, .009, mid + .0001))
        assert (shape.corner_lo[2] + shape.corner_hi[2]) / 2 == mid
        assert int(np.argmin(abs(np.asarray(coords.z) - mid))) == 11
        mask = shape.mask_on_coords(coords.x, coords.y, coords.z)
        assert set(np.nonzero(np.asarray(mask))[2]) == {10}
    elif kind == "aperture":
        if lane == "graded":
            from rfx.runners.nonuniform import _build_waveguide_port_config_nu
            freqs = jnp.linspace(40e9, 50e9, 3)
            declaration = Simulation(freq_max=50e9, domain=(.02,) * 3, dx=.001, boundary="cpml")
            declaration.add_waveguide_port(.005, y_range=(.0035, .0095),
                                           z_range=(0., .009), direction="+x", freqs=freqs)
            cfg = _build_waveguide_port_config_nu(sim, declaration._waveguide_ports[0], grid, freqs, 10)
            span = (cfg.u_lo, cfg.u_hi + 1)
        else:
            span, _ = sim._range_to_slice((.0035, .0095), .02, .001, grid.ny, 0)
        assert span == (3, 10)
    elif kind == "flux":
        from rfx.probes.flux_region import resolve_flux_region
        entry = SimpleNamespace(size=(.006, .006), center=(.0065, .0065),
                                axis="z", coordinate=.0035, name="tie")
        record = resolve_flux_region(grid, entry, sim._domain, warn=False)
        assert record["normal_index"] == 3
        assert record["cell_slices"] == [[3, 9], [3, 9]]
    elif kind == "current_window":
        from rfx.current_moments import _index_window
        assert _index_window(coords.x, .0035, .0105, grid.shape[0]) == (3, 10)
    elif kind == "thin_wire":
        if lane == "graded":
            pytest.skip("legacy Holland correction accepts uniform Grid only")
        from rfx.geometry.thin_wire import ThinWire, compute_thin_wire_correction
        _, sigma = compute_thin_wire_correction(grid, ThinWire((.0035,) * 3, (.0035, .0035, .0095), .0001))
        occupied = np.nonzero(np.asarray(sigma))
        assert set(occupied[0]) == set(occupied[1]) == {3}
        assert (min(occupied[2]), max(occupied[2])) == (3, 9)
    elif kind in ("port_wire", "source_wire", "lumped_wire"):
        pos = (.010, .0095, .008)
        sim.add(PolylineWire(((.010, .0095, .003), pos), radius=0), material="pec")
        sim.add(PolylineWire(((.010, .0095, .009), (.010, .0095, .014)), radius=0), material="pec")
        if kind == "port_wire":
            sim.add_port(pos, component="ez", extent=.001)
        elif kind == "source_wire":
            sim.add_source(pos, component="ez", amplitude_kind="current")
        else:
            sim.add_lumped_rlc(pos, component="ez", R=50., topology="parallel")
            from rfx.lumped import _resolve_position_to_index
            assert _resolve_position_to_index(grid, pos) == (10, 9, 8)
        wires = []
        assemble = sim._assemble_materials_nu if lane == "graded" else sim._assemble_materials
        materials = assemble(grid, pec_wires=wires, pec_sheets=[], sheet_specs=[])[0]
        assert {int(j) for w in wires for j in np.nonzero(np.asarray(w.edges[2]))[1]} == {9}
        assert lookup(pos) == (10, 9, 8)
        if kind == "lumped_wire":
            from rfx.lumped import setup_rlc_materials
            stamped = setup_rlc_materials(grid, sim._lumped_rlc[0], materials)
            assert list(zip(*np.nonzero(np.asarray(stamped.sigma_lumped[2])))) == [(10, 9, 8)]
    elif kind in ("port_sheet", "port_lossy", "probe_sheet", "dft_sheet", "port_face"):
        if kind == "port_face":
            sim.add(Box((0., 0., 0.), (.02, .02, .0035)), material="pec")
            sim.add_port((.010, .009, .0035), component="ez", extent=.002)
        else:
            shape = Box((.005, .005, .0035), (.015, .014, .0035))
            if kind == "port_lossy":
                sim.add_thin_conductor(shape, sigma_bulk=5.8e7, surface_impedance_f0=5e9)
            else:
                sim.add(shape, material="pec")
            if kind in ("port_sheet", "port_lossy"):
                sim.add_port((.010, .009, 0.), component="ez", extent=.0035)
            elif kind == "probe_sheet":
                sim.add_probe((.010, .009, .0035), component="ex")
            else:
                sim.add_dft_plane_probe(axis="z", coordinate=.0035, component="ex", n_freqs=2)
        assemble = sim._assemble_materials_nu if lane == "graded" else sim._assemble_materials
        sheets, lossy = [], []
        result = assemble(grid, pec_wires=[], pec_sheets=sheets, sheet_specs=lossy)
        if kind == "port_face":
            assert int(np.nonzero(np.asarray(result[3])[10, 9])[0].max()) + 1 == 3
        elif kind == "port_lossy":
            assert set(np.nonzero(np.asarray(lossy[0].mask))[2]) == {3}
        else:
            assert [int(s.plane) for s in sheets] == [3]
        assert lookup((.010, .009, .0035))[2] == 3
        if kind == "dft_sheet":
            from rfx.probes import probes
            from rfx.runners import uniform
            sim.add_source((.010, .009, .001), component="ez", amplitude_kind="current")
            captured = []

            class Captured(Exception):
                pass

            def capture(*args, **kwargs):
                captured.append(kwargs["index"])
                raise Captured

            monkeypatch.setattr(probes, "init_dft_plane_probe", capture)
            monkeypatch.setattr(uniform, "init_dft_plane_probe", capture)
            with pytest.raises(Captured):
                sim.run(n_steps=1, skip_preflight=True)
            assert captured == [3]
    elif kind in ("subgrid_bound", "disjoint_bound", "disjoint_point"):
        if lane == "graded":
            pytest.skip("subgrid runners require a uniform coarse grid")
        if kind == "disjoint_point":
            from rfx.subgridding.disjoint_runner_contract import _map_position
            mapping = _map_position(name="tie", component="ez", position=(.0035,) * 3,
                                    origin=(0.,) * 3, shape_f=(20,) * 3, dx_f=.001)
            assert mapping.fine_index == (3, 3, 3)
        else:
            from rfx.subgridding.validation import build_subgrid_region, build_stage2_disjoint_region
            sim = Simulation(freq_max=10e9, domain=(.021, .021, .02), dx=.001, boundary="pec")
            grid = sim._build_grid()
            sim._refinement = dict(ratio=3, z_range=(.0035, .0095), xy_margin=.0035)
            region = (build_subgrid_region if kind == "subgrid_bound" else build_stage2_disjoint_region)(sim, grid)
            assert (region.fi_lo, region.fj_lo, region.fk_lo, region.fk_hi) == (3, 3, 3, 10)
            assert (region.fi_hi, region.fj_hi) == (18, 18)
    elif kind == "preflight_node":
        from rfx.preflight._common import profile_node_at
        assert profile_node_at(.001, sizes[0] if lane == "graded" else None, .0035) == pytest.approx(.003)


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("extent", [None, .003])
@pytest.mark.parametrize("lo,hi,x,node,footprint,refused", [
    (.003, .0115, .0115, 11, (3, 11), False),
    (.003, .0113, .0113, 11, (3, 11), False),
    (.003, .0117, .0117, 12, (3, 11), True),
    (.0035, .011, .0035, 3, (4, 11), True),
    (.0033, .011, .0033, 3, (4, 11), True),
    # declared one float step below the trace start (e.g. through a float sum)
    (.0035, .011, float(np.nextafter(.0035, 0.)), 3, (4, 11), True),
], ids=["A", "B", "C", "D", "D2", "D-eps"])
def test_trace_port_realized_footprint(lane, extent, lo, hi, x, node, footprint, refused):
    from rfx.preflight.ports import trace_far_end_findings
    from rfx.geometry.rasterize_grid import sheet_spec_from_shape
    sim, grid, coords, sizes = model(lane)
    shape = Box((lo, .007, .003), (hi, .011, .003))
    sim.add(shape, material="pec")
    sim.add_port((x, .009, .002 if extent is None else 0.), component="ez", extent=extent)
    occupied = np.nonzero(np.asarray(sheet_spec_from_shape(shape, coords, sizes, grid=grid).footprint))[0]
    assert (int(occupied.min()), int(occupied.max())) == footprint
    assert grid.index_of(0, x) == node
    findings = trace_far_end_findings(sim)
    assert bool(findings) is refused
    report = sim.preflight(strict=False)
    reported = [i for i in report if i.code == "trace_port_footprint"]
    assert [str(i) for i in reported] == findings
    assert all(i.severity == "error" for i in reported)
    if refused:
        with pytest.raises(ValueError, match="Move the port.*#1342"):
            sim.run(n_steps=1, skip_preflight=True)
    else:
        assert np.isfinite(sim.run(n_steps=1, skip_preflight=True).time_series).all()


def test_off_ties_keep_rounding_and_input_dtype():
    from rfx._grid_metric import nearest_uniform_index
    for dtype in (np.float32, np.float64):
        for ratio in np.linspace(-20, 20, 401):
            ratio = dtype(ratio)
            if abs(float(ratio) % 1 - .5) > 1e-9:
                assert nearest_uniform_index(ratio) == int(round(ratio))
    from rfx.grid import Grid
    grid = Grid(60e9, (.009, .006, .0042), dx=.0003, cpml_layers=0)
    assert grid.position_to_index((np.float32(.00075), 0., 0.))[0] == 2
    assert float(np.float32(.00075)) / grid.dx > 2.5 + 1e-9
    assert grid.index_of(0, np.float32(.00075)) == 3


def test_subgrid_runner_bound_and_point_sites(monkeypatch):
    """Observe the runner's own setup, before its first fine/coarse step."""
    from rfx.runners.subgridded import _run_subgridded_once
    from rfx.subgridding import jit_runner
    dx = 3 / 1024  # binary-exact coarse and fine cells (ratio 3)
    sim = Simulation(freq_max=10e9, domain=(21 * dx, 21 * dx, 20 * dx), dx=dx, boundary="pec")
    grid = sim._build_grid()
    sim._refinement = dict(ratio=3, z_range=(3.5 * dx, 9.5 * dx), xy_margin=3.5 * dx,
                           validation="off")
    pos = (3 * dx + 3.5 * (dx / 3),) * 3
    assert (pos[0] - 3 * dx) / (dx / 3) == 3.5
    assert round((pos[0] - 3 * dx) / (dx / 3)) == 4
    sim.add_source(pos, component="ez", amplitude_kind="field")
    sim.add_probe(pos, component="ez")
    mats = sim._assemble_materials(grid)[0]
    observed = {}

    class Captured(Exception):
        pass

    def capture(_grid, _coarse, _fine, config, _steps, *, opts):
        observed.update(config=config, opts=opts)
        raise Captured

    monkeypatch.setattr(jit_runner, "run_subgridded_jit", capture)
    with pytest.raises(Captured):
        _run_subgridded_once(sim, grid, mats, None, 1)
    cfg, opts = observed["config"], observed["opts"]
    assert (cfg.fi_lo, cfg.fj_lo, cfg.fk_lo, cfg.fk_hi) == (3, 3, 3, 10)
    assert (cfg.fi_hi, cfg.fj_hi) == (18, 18)
    assert tuple(opts.sources_f[0][:3]) == (3, 3, 3)
    assert tuple(opts.probe_indices_f[0]) == (3, 3, 3)


def test_flux_window_uses_the_band_at_both_ends():
    from rfx.probes.flux_region import resolve_flux_axis
    edges = np.arange(20) * .001
    result = resolve_flux_axis(edges, 0, .0065 + 2e-13, .006)
    assert result["cell_slice"] == (3, 9)
