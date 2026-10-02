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


def test_result_keeps_the_pre_run_object_without_stepping(monkeypatch):
    from rfx import Result
    import rfx.runners.uniform
    sim = _model()
    record = sim.realized_geometry()
    monkeypatch.setattr(rfx.runners.uniform, "run_uniform", lambda *a, **k: Result(None, np.zeros((0, 0)), None, None))
    monkeypatch.setattr(sim, "_attach_run_settling_witness", lambda result, **k: result)
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    assert result.realized_geometry is record
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
