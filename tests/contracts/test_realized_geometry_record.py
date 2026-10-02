"""Build-only contract for the public realized geometry and all its readers."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def _model(nonuniform=False):
    d = 1e-3
    kwargs = {"dz_profile": np.full(9, d)} if nonuniform else {}
    sim = Simulation(freq_max=10e9, domain=(12*d, 11*d, 9*d), dx=d,
                     boundary=BoundarySpec(x=Boundary(lo="pec", hi="cpml"),
                                           y=Boundary(lo="cpml", hi="pec"), z="pec"),
                     cpml_layers=2, **kwargs)
    sim.add_material("slab", eps_r=3.0)
    sim.add_material("other", eps_r=2.0)
    sim.add(Box((d, 2*d, np.nextafter(2*d, np.inf)), (4*d, 7*d, 4*d)), material="slab")
    sim.add(Box((2*d+2e-6, 3*d+2e-6, 5*d), (8*d-2e-6, 9*d-2e-6, 5*d)), material="pec")
    sim.add(Box((8*d, d, d), (10*d, 3*d, 3*d)), material="pec")
    sim.add(Box((6*d, 2*d, 6*d), (9*d, 4*d, 8*d)), material="other")
    from rfx import GaussianPulse
    sim.add_port(position=(5*d, 5*d, 2*d), component="ez", impedance=50,
                 waveform=GaussianPulse(f0=5e9))
    return sim


def _compare(sim, record, report):
    """Comparison under mutation (a); negative controls below must reject drift."""
    rows = {row["entity"].split(" ")[0]: row for row in report if "axes" in row}
    for entity in record.entities:
        row = rows[entity.label]
        assert row["n_cells"] == entity.n_cells
        for axis, emitted in zip(entity.axes, row["axes"], strict=True):
            np.testing.assert_array_equal(emitted["realized_um"], np.array(axis.bounds_m)*1e6)
            assert emitted["realized_extent_um"] == axis.extent_m*1e6
            # The report displays absolute residuals and suppresses float ties.
            expected = [0.0 if abs(v) < 1e-9*axis.cell_size_m else abs(v)*1e6
                        for v in axis.face_residual_m]
            np.testing.assert_array_equal(emitted["face_residual_um"], expected)
    ctx = sim._campaign_ctx()
    census, _ = ctx.congruence_entries()
    by_label = {e.label: e for e in record.entities}
    for entry, _, _, _ in census:
        row = by_label[entry.label]
        assert row.kind == entry.kind
        if entry.cells is not None:
            assert row.n_cells == int(entry.cells.sum())
            for a in range(3):
                assert row.wall_planes[a] == tuple(entry.wall_planes(a, ctx.periodic, ctx.grid.shape))
    from rfx.preflight.pec_geometry import _sheet_solved_spans
    for entry, a, span in _sheet_solved_spans(ctx, [e for e, _, _, _ in census]):
        assert by_label[entry.label].axes[a].bounds_m == (span.lo, span.hi)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_one_record_many_readers(nonuniform):
    sim = _model(nonuniform)
    record = sim.realized_geometry()
    assert sim.realized_geometry() is record
    _compare(sim, record, sim.fidelity_report(print_report=False))
    assert [e.kind for e in record.entities] == ["material", "sheet", "volume", "material"]
    # The off-centre sheet has node x bounds 3..7 mm, and solved bounds 2.65..7.35 mm.
    sheet = record.entities[1]
    np.testing.assert_allclose(sheet.axes[0].bounds_m, (0.00265, 0.00735), atol=1e-17, rtol=0)
    assert sheet.axes[0].extent_m != 0.004
    assert sheet.plane == (2, 5, 0.005)
    assert record.domain[0].padding == (0, 2)
    assert record.domain[1].padding == (2, 0)
    assert tuple(a.n_cells for a in record.domain) == (12, 11, 9)
    assert record.ports[0].edges == ((5, 7, 2),)
    from tests._realized_geometry import realized
    wrapper = realized(sim)
    assert wrapper.edge_masks is record.edge_masks
    assert wrapper.pec_mask is record.pec_mask
    # Collection skips both PEC rows between the two materials. Keys, not zip.
    masks = sim._campaign_ctx().realized().geometry_masks
    assert set(masks) == {id(sim._geometry[0]), id(sim._geometry[3])}
    for i in (0, 3):
        np.testing.assert_array_equal(record.entities[i].mask, masks[id(sim._geometry[i])])
    assert record.entities[0].n_cells != record.entities[3].n_cells
    with pytest.raises(FrozenInstanceError):
        record.entities = ()
    with pytest.raises(ValueError):
        record.edge_masks[0].setflags(write=True)


@pytest.mark.parametrize("mutation", ["declared", "node_span", "residual", "comparison_disabled"])
def test_comparison_rejects_corrupted_reader(mutation):
    import copy
    sim = _model()
    record = sim.realized_geometry()
    report = copy.deepcopy(sim.fidelity_report(print_report=False))
    row = next(row for row in report if row["entity"].startswith("geometry[1]"))
    axis = row["axes"][0]
    if mutation == "declared":
        axis["realized_um"] = axis["declared_um"]
    elif mutation == "node_span":
        axis["realized_extent_um"] = 4000.0
    elif mutation == "residual":
        axis["face_residual_um"] = (0.0, 0.0)
    else:
        # If _compare is disabled, this expected exception disappears: red.
        axis["realized_extent_um"] += 100.0
    with pytest.raises(AssertionError):
        _compare(sim, record, report)


def test_result_keeps_its_compact_run_record_without_stepping(monkeypatch):
    from rfx import Result
    import rfx.runners.uniform
    sim = _model()
    record = sim.realized_geometry()
    monkeypatch.setattr(rfx.runners.uniform, "run_uniform", lambda *a, **k: Result(None, np.zeros((0, 0)), None, None))
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **k: result)
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    assert result.realized_geometry is not record
    assert result.realized_geometry.entities[1].axes == record.entities[1].axes
    assert result.realized_geometry.entities[1].mask is None
    sim.add(Box((0.001, 0.001, 0.006), (0.002, 0.002, 0.007)), material="slab")
    assert sim.realized_geometry() is not record
    assert len(result.realized_geometry.entities) == 4


def test_cached_record_builds_once_across_readers(monkeypatch):
    import rfx.realized_geometry as records
    from tests._realized_geometry import realized
    sim = _model()
    calls = []
    build = records._build_record

    def counted(*args):
        calls.append(args[0])
        return build(*args)

    monkeypatch.setattr(records, "_build_record", counted)
    record = sim.realized_geometry()
    sim.fidelity_report(print_report=False)
    assert realized(sim).edge_masks is record.edge_masks
    assert sim.realized_geometry() is record
    assert calls == [sim]
    sim.add_material("slab", eps_r=4.0)
    assert sim.realized_geometry() is not record
    assert calls == [sim, sim]


def test_refused_occupancy_is_diagnostic_and_cached():
    sim = _model()
    sim.add(Box((0.001, 0.001, 0.001), (0.0011, 0.0011, 0.002)), material="pec")
    record = sim.realized_geometry()
    refused = record.entities[-1]
    assert refused.kind == "refused"
    assert refused.occupancy_role == "diagnostic"
    assert refused.mask.any()
    assert all(not edge.any() for edge in refused.edge_masks)
    row = next(r for r in sim.fidelity_report(print_report=False)
               if r["entity"].startswith(refused.label))
    assert "refused-by-contract" in [f["kind"] for f in row["findings"]]
    assert sim.realized_geometry() is record


@pytest.mark.parametrize("entry", ["realized", "run", "forward"])
@pytest.mark.parametrize("diagnostic_first", [False, True])
def test_frozen_unresolved_volume_still_refuses_execution(entry, diagnostic_first):
    from tests._realized_geometry import realized

    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), boundary="pec")
    grid = sim.freeze_mesh()
    d = grid.cells("x")[0]
    sim.add(Box((2*d, d, d), (2.2*d, 4*d, 4*d)), material="pec")
    if diagnostic_first:
        record = sim.realized_geometry()
        assert record.entities[0].kind == "refused"
        report = sim.fidelity_report(print_report=False)
        assert any(f["kind"] == "refused-by-contract"
                   for row in report for f in row.get("findings", ()))
        assert sim.realized_geometry() is record
    with pytest.raises(ValueError, match="volume|sub.cell|thickness"):
        if entry == "realized":
            realized(sim)
        else:
            getattr(sim, entry)(n_steps=1, skip_preflight=True)
    later = sim._build_realized_grid()
    assert later.shape == grid.shape
    for axis in "xyz":
        np.testing.assert_array_equal(later.cells(axis), grid.cells(axis))


def _terminated_trace(nonuniform):
    # Reviewer's synth.py: prime the record BEFORE registering the MSL ports.
    d = 127e-6
    kw = {"dz_profile": np.full(10, d)} if nonuniform else {}
    sim = Simulation(freq_max=20e9, domain=(60.4*d, 30*d, 10*d), dx=d,
                     boundary="cpml", cpml_layers=8, **kw)
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, 0), (60.4*d, 30*d, 2*d)), material="sub")
    sim.add(Box((0, 13*d, 2*d), (60.4*d, 17*d, 2*d)), material="pec")
    sim.add(Box((0, 0, 0), (60.4*d, 30*d, 0)), material="pec")
    before = sim.realized_geometry()
    for x, direction in ((10, "+x"), (50, "-x")):
        sim.add_msl_port(position=(x*d, 15*d, 0), width=4*d, height=2*d,
                         direction=direction, impedance=50)
    return sim, before


def _assert_runner_sheet(record, captured, grid):
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.geometry.rasterize_grid import interior_lattice_mask
    from dataclasses import replace
    sheet = captured["pec_sheets"][0]
    sheet = replace(sheet, footprint=interior_lattice_mask(sheet.footprint, grid))
    fp = np.asarray(sheet.footprint)
    row = record.entities[1]
    indices = np.where(fp)
    assert tuple(a.node_range for a in row.axes) == tuple((int(i.min()), int(i.max())) for i in indices)
    assert row.node_count == int(fp.sum())
    edges = realized_pec_edge_masks(None, sheets=[sheet], wires=(), periodic=(False,)*3)
    assert row.edge_counts == tuple(int(np.asarray(m).sum()) for m in edges)
    ranges = []
    for edge in edges:
        indices = np.where(np.asarray(edge))
        ranges.append(tuple((int(i.min()), int(i.max())) for i in indices) if indices[0].size else ())
    assert row.edge_ranges == tuple(ranges)


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("poison_context", [False, True])
def test_run_assembly_witness_after_port_mutation(monkeypatch, nonuniform, poison_context):
    from rfx import Result
    import rfx.runners.uniform as uniform
    import rfx.runners.nonuniform as nu
    sim, before = _terminated_trace(nonuniform)
    before_ctx = sim._pf_campaign_ctx
    after = sim.realized_geometry()
    assert after is not before
    assert sim._pf_campaign_ctx[1] is not before_ctx[1]
    # The unheld line extends into the pad; the port holds the declaration.
    assert int(before.sheets[0].footprint.sum()) > int(after.sheets[0].footprint.sum())
    assembly_calls = []
    owner, name = (nu, "assemble_materials_nu") if nonuniform else (sim, "_assemble_materials")
    assemble = getattr(owner, name)
    def counted_assembly(*args, **kwargs):
        assembly_calls.append(True)
        return assemble(*args, **kwargs)
    monkeypatch.setattr(owner, name, counted_assembly)
    cap = {}
    if nonuniform:
        def fake(grid, materials, n_steps, **kwargs):
            cap.update(kwargs, grid=grid)
            return {"state": None, "time_series": np.zeros((0, 0))}
        monkeypatch.setattr(nu, "run_nonuniform", fake)
    else:
        def fake(*args, **kwargs):
            cap.update(kwargs)
            return Result(None, np.zeros((0, 0)), None, None)
        monkeypatch.setattr(uniform, "run_uniform", fake)
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **k: result)
    context_reads = []
    if poison_context:
        # A run must not read ANY preflight context, even if one is stale.
        def stale_context():
            context_reads.append(True)
            return before_ctx[1]
        monkeypatch.setattr(sim, "_campaign_ctx", stale_context)
        monkeypatch.setattr(sim, "realized_geometry", lambda: before)
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    record = result.realized_geometry
    assert record.lane == ("run_nonuniform" if nonuniform else "run_uniform")
    _assert_runner_sheet(record, cap, cap["grid"])
    _assert_runner_sheet(after, cap, cap["grid"])
    assert context_reads == []
    assert assembly_calls == [True]
    assert record.pec_mask is None and record.edge_masks == () and record.materials is None
    assert record.sheets == () and record.wires == ()
    assert all(e.mask is None and not e.edge_masks and e.sheet is None for e in record.entities)


def test_material_collection_order_is_not_declaration_order(monkeypatch):
    sim = _model()
    original = sim._assemble_materials
    def reordered(*args, **kwargs):
        result = original(*args, **kwargs)
        masks = kwargs.get("geometry_masks")
        if masks is not None:
            masks.reverse()
        return result
    monkeypatch.setattr(sim, "_assemble_materials", reordered)
    record = sim.realized_geometry()
    # Independent known boxes: 3*5*2 and 3*2*2 cells, separated in z.
    assert record.entities[0].n_cells == 30
    assert record.entities[3].n_cells == 12
    np.testing.assert_allclose(record.entities[0].axes[2].bounds_m, (.002, .004))
    np.testing.assert_allclose(record.entities[3].axes[2].bounds_m, (.006, .008))


def test_subgrid_record_names_unrepresented_refinement(monkeypatch):
    from rfx import Result
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .012), dx=.001, boundary="pec")
    sim.add(Box((.002, .002, .002), (.004, .004, .004)), material="pec")
    sim.add_refinement(z_range=(.004, .008), ratio=2)
    monkeypatch.setattr(sim, "_run_subgridded", lambda *a, **k: Result(None, np.zeros((0, 0)), None, None))
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **k: result)
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    assert result.realized_geometry.lane == "run_subgridded"
    assert result.realized_geometry.limitations == ("refined region not represented",)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_asymmetric_entities_against_runner_witness(monkeypatch, nonuniform):
    from rfx import Result
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.geometry.rasterize_grid import interior_lattice_mask
    from dataclasses import replace
    import rfx.runners.uniform as uniform
    import rfx.runners.nonuniform as nu
    sim = _model(nonuniform)
    from rfx import PolylineWire
    sim.add(PolylineWire(((.003, .005, .007), (.005, .005, .007)), radius=0.), material="pec")
    sim.add_pinned_sheet(plane_index=8, i_range=(2, 4), j_range=(2, 4))
    cap = {}
    if nonuniform:
        def fake(grid, materials, n_steps, **kwargs):
            cap.update(kwargs, grid=grid)
            return {"state": None, "time_series": np.zeros((0, 0))}
        monkeypatch.setattr(nu, "run_nonuniform", fake)
    else:
        def fake(*args, **kwargs):
            cap.update(kwargs)
            return Result(None, np.zeros((0, 0)), None, None)
        monkeypatch.setattr(uniform, "run_uniform", fake)
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **k: result)
    def no_preflight():
        raise AssertionError("run must use its own assembly, not preflight")
    monkeypatch.setattr(sim, "_campaign_ctx", no_preflight)
    record = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False).realized_geometry
    grid = cap["grid"]
    assert record.entities[0].n_cells == 30
    assert record.entities[3].n_cells == 12
    for index, cells, sheets in (
        (1, None, [replace(cap["pec_sheets"][0], footprint=interior_lattice_mask(cap["pec_sheets"][0].footprint, grid))]),
        (2, interior_lattice_mask(cap["pec_mask"], grid, cell_axes=(True,)*3), []),
    ):
        row = record.entities[index]
        edges = realized_pec_edge_masks(cells, sheets=sheets, wires=(), periodic=(False,)*3)
        assert row.edge_counts == tuple(int(np.asarray(e).sum()) for e in edges)
        assert row.wall_planes == tuple(tuple(realized_wall_planes(edges, a)) for a in range(3))
        occupancy = cells if cells is not None else sheets[0].footprint
        indices = np.where(occupancy)
        assert tuple(a.node_range for a in row.axes) == tuple(
            (int(i.min()), int(i.max()) + (cells is not None)) for i in indices)
        assert (row.n_cells if cells is not None else row.node_count) == int(np.asarray(occupancy).sum())


    wire = record.entities[4]
    edges = realized_pec_edge_masks(None, sheets=(), wires=cap["pec_wires"], periodic=(False,)*3)
    assert wire.kind == "wire"
    assert wire.edge_counts == tuple(int(np.asarray(e).sum()) for e in edges)
    ranges = []
    for edge in edges:
        indices = np.where(np.asarray(edge))
        ranges.append(tuple((int(i.min()), int(i.max())) for i in indices) if indices[0].size else ())
    assert wire.edge_ranges == tuple(ranges)
    pinned = record.entities[5]
    edges = realized_pec_edge_masks(None, sheets=[cap["pec_sheets"][-1]], wires=(), periodic=(False,)*3)
    assert pinned.label == "pinned_sheet[0]"
    assert pinned.node_count == 9
    assert pinned.edge_counts == tuple(int(np.asarray(e).sum()) for e in edges)
    assert pinned.wall_planes == tuple(tuple(realized_wall_planes(edges, a)) for a in range(3))


@pytest.mark.parametrize("reader", ["preflight", "fidelity_report", "realized_geometry"])
def test_unextendable_conductor_warning_is_present(reader, capsys):
    import warnings
    from rfx import Sphere
    sim = Simulation(freq_max=1e9, domain=(.008,)*3, dx=.001, boundary="cpml", cpml_layers=4)
    sim.add(Sphere((.001, .004, .004), .001), material="pec")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(sim, reader)()
    assert (any("Conducting geometry reaches" in str(w.message) for w in caught)
            or "Conducting geometry reaches" in capsys.readouterr().out)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_concrete_record_failure_propagates(monkeypatch, nonuniform):
    import rfx.realized_geometry as records
    import rfx.runners.uniform as uniform
    import rfx.runners.nonuniform as nu
    from rfx import Result
    sim = _model(nonuniform)
    calls = []

    def fail_record(*args, **kwargs):
        calls.append("record")
        raise ValueError("diagnostic aperture unavailable")

    def runner(*args, **kwargs):
        calls.append("runner")
        if nonuniform:
            return {"state": None, "time_series": np.zeros((0, 0))}
        return Result(None, np.zeros((0, 0)), None, None)

    monkeypatch.setattr(records, "_record_from_assembly", fail_record)
    monkeypatch.setattr(nu if nonuniform else uniform,
                        "run_nonuniform" if nonuniform else "run_uniform", runner)
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **k: result)
    with pytest.raises(ValueError, match="diagnostic aperture unavailable"):
        sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    assert calls == ["record"]


@pytest.mark.parametrize("nonuniform", [False, True])
def test_lossy_run_record_retains_assembly_continuation(monkeypatch, nonuniform):
    from rfx import Result
    import rfx.runners.uniform as uniform
    import rfx.runners.nonuniform as nu
    kwargs = {"dz_profile": np.full(8, .001)} if nonuniform else {}
    sim = Simulation(freq_max=10e9, domain=(.0086, .0086, .008), dx=.001,
                     boundary="cpml", cpml_layers=4, **kwargs)
    sim.add_thin_conductor(Box((0., 0., .004), (.0086, .0086, .004)),
                           sigma_bulk=1e4, surface_impedance_f0=10e9)
    monkeypatch.setattr(uniform, "run_uniform", lambda *a, **k: Result(None, np.zeros((0, 0)), None, None))
    monkeypatch.setattr(nu, "run_nonuniform", lambda *a, **k: {"state": None, "time_series": np.zeros((0, 0))})
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **k: result)
    record = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False).realized_geometry
    assert record.entities[0].kind == "lossy"
    assert record.entities[0].continued_faces == ("x-lo", "x-hi", "y-lo", "y-hi")


@pytest.mark.parametrize("normal", ["x", "y", "z"])
@pytest.mark.parametrize("ranged", [False, True])
def test_graded_waveguide_record_matches_runner_aperture(monkeypatch, normal, ranged):
    import rfx.runners.nonuniform as nu

    graded_axis = "y" if normal == "z" else "z"
    ramp = np.linspace(.0004, .0008, 20)
    profile = np.concatenate((ramp, ramp[::-1]))
    domain = [.03, .02286, .02286]
    domain["xyz".index(graded_axis)] = float(profile.sum())
    sim = Simulation(freq_max=20e9, domain=tuple(domain), dx=.001,
                     boundary=BoundarySpec(**{axis: "cpml" if axis == normal else "pec"
                                              for axis in "xyz"}),
                     cpml_layers=2, **{f"d{graded_axis}_profile": profile})
    value_range = (.002, .0072) if ranged else None
    sim.add_waveguide_port(.01, direction=f"+{normal}", mode=(1, 0),
                           f0=15e9, freqs=np.array([15e9]),
                           **{f"{graded_axis}_range": value_range})
    before = sim.realized_geometry()
    captured = {}

    def runner(grid, materials, n_steps, **kwargs):
        captured.update(kwargs, grid=grid)
        return {"state": None, "time_series": np.zeros((0, 0))}

    monkeypatch.setattr(nu, "run_nonuniform", runner)
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **k: result)
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    cfg, = captured["waveguide_ports"]
    # The runner consumes half-open CELL slices; their endpoints are the
    # inclusive NODE endpoints recorded for the aperture.
    transverse = iter(((cfg.u_lo, cfg.u_hi), (cfg.v_lo, cfg.v_hi)))
    expected = tuple((cfg.x_index, cfg.x_index) if axis == normal else next(transverse)
                     for axis in "xyz")
    assert before.ports[0].aperture == expected
    assert result.realized_geometry.ports[0].aperture == expected
    if ranged:
        # Independent nearest-node witness on the declared float64 spine.
        nodes = np.concatenate(([0.], np.cumsum(profile)))
        endpoints = tuple(int(np.argmin(abs(nodes - value))) for value in value_range)
        assert expected["xyz".index(graded_axis)] == endpoints


@pytest.mark.parametrize("axis", ["dx_arr", "dy_arr", "dz"])
def test_record_skips_only_traced_mesh_coordinates(monkeypatch, axis):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    import rfx.realized_geometry as records

    def fail_record(*args, **kwargs):
        raise ValueError("concrete record defect")

    monkeypatch.setattr(records, "_record_from_assembly", fail_record)

    def traced(profile):
        grid = SimpleNamespace(dx_arr=np.ones(3), dy_arr=np.ones(3), dz=np.ones(3))
        setattr(grid, axis, profile)
        assert records.record_from_assembly(None, grid) is None
        # A tracer in materials is not permission to swallow record errors.
        setattr(grid, axis, np.ones(3))
        with pytest.raises(ValueError, match="concrete record defect"):
            records.record_from_assembly(None, grid, materials=profile)
        return profile.sum()

    jax.make_jaxpr(traced)(jnp.ones(3))
