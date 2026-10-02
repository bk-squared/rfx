"""Build-only #1138 contract: solved edges decide acceptance, not node span."""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.mesh_edges import edge_aware_profiles
from rfx.preflight.pec_geometry import _sheet_solved_spans
from rfx.preflight._common import _fmt_len


CODE = "sheet_effective_size"
DOMAIN = (0.02, 0.02, 0.006)
DX = 0.001


def _build(*, inset=0.0, aligned=False, snap="strict"):
    sheet = Box((0.005 + inset, 0.005 + inset, 0.003),
                (0.015 - inset, 0.015 - inset, 0.003))
    profiles = edge_aware_profiles(DOMAIN, DX, sheets=[sheet]) if aligned else {}
    sim = Simulation(10e9, DOMAIN, dx=DX, boundary="pec", snap=snap, **profiles)
    sim.add(sheet, material="pec")
    sim.add_source((0.002, 0.002, 0.002), component="ez")
    return sim


class _ReachedRunner(Exception):
    pass


def _stub_runners(monkeypatch):
    calls = []

    def runner(*args, **kwargs):
        calls.append(True)
        raise _ReachedRunner

    monkeypatch.setattr("rfx.runners.uniform.run_uniform", runner)
    monkeypatch.setattr(Simulation, "_run_nonuniform", runner)
    monkeypatch.setattr(Simulation, "_forward_from_materials", runner)
    monkeypatch.setattr(Simulation, "_forward_nonuniform_from_materials", runner)
    return calls


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("inset", [0.0, 2e-6])
def test_strict_refuses_inset_and_on_node_sheets(monkeypatch, inset, entry):
    calls = _stub_runners(monkeypatch)
    sim = _build(inset=inset)
    report = sim.preflight()
    row, = report.by_code(CODE)
    assert row.severity == "error"
    assert "rfx.mesh_edges.edge_aware_profiles" in str(row)
    assert 'Simulation(..., snap="declared")' in str(row)
    assert not report.ok
    with pytest.raises(ValueError, match="conductor sheet dimension"):
        kwargs = {"compute_s_params": False} if entry == "run" else {}
        getattr(sim, entry)(n_steps=1, **kwargs)
    assert not calls


def test_edge_aware_sheet_proceeds(monkeypatch):
    calls = _stub_runners(monkeypatch)
    sim = _build(inset=2e-6, aligned=True)
    assert not sim.preflight().by_code(CODE)
    with pytest.raises(_ReachedRunner):
        sim.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


def test_declared_acceptance_is_recorded_and_changes_no_solved_numbers(monkeypatch):
    calls = _stub_runners(monkeypatch)
    strict, accepted = _build(), _build(snap="declared")
    row, = accepted.preflight().by_code(CODE)
    assert row.severity == "warning"
    before, after = strict.realized_geometry(), accepted.realized_geometry()
    assert before.snap == "strict"
    assert after.snap == "declared"
    for a, b in zip(before.entities, after.entities, strict=True):
        assert a.axes == b.axes
        np.testing.assert_array_equal(a.mask, b.mask)
    for a, b in zip(before.nodes, after.nodes, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(_ReachedRunner):
        accepted.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


@pytest.mark.parametrize("inset,aligned", [(0.0, False), (2e-6, False), (2e-6, True)])
def test_verdict_spans_equal_public_record_per_entity_axis(inset, aligned):
    sim = _build(inset=inset, aligned=aligned)
    # A second declaration tests identity and both declaration routes.
    sim.add_thin_conductor(Box((0.004, 0.004, 0.004), (0.016, 0.016, 0.004)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = sim.realized_geometry()
    ctx = sim._campaign_ctx()
    actual = {(e.label, "xyz"[a]): (span.lo, span.hi)
              for e, a, span in _sheet_solved_spans(ctx, ctx.interior_pec_entries())}
    expected = {(e.label, a.axis): a.bounds_m for e in record.entities
                if e.kind == "sheet" for a in e.axes if a.axis != "xyz"[e.plane[0]]}
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == expected[key], key
    findings = sim.preflight().by_code(CODE)
    offenders = []
    for e in record.entities:
        for a in e.axes:
            bounds = a.comparison_bounds_m or a.declared_bounds_m
            drawn = bounds[1] - bounds[0]
            if drawn <= 0 or a.axis == "xyz"[e.plane[0]]:
                continue
            residual = (a.extent_m - drawn) / drawn
            if abs(residual) > 0.01:
                offenders.append((e, a, drawn, residual))
    assert len(findings) == bool(offenders)
    if offenders:
        message = str(findings[0])
        assert message.startswith(f"{len(offenders)} conductor sheet dimension(s)")
        for e, a, drawn, residual in offenders:
            assert f"{e.label} '{e.name}' {a.axis}: drawn {_fmt_len(drawn)}" in message
            assert f"solved as {_fmt_len(a.extent_m)} ({residual:+.2%})" in message
    # Independently pin the on-node +0.7 cell and inset loss of 1.3 cells.
    if not aligned:
        axis = record.entities[0].axes[0]
        assert axis.extent_m == pytest.approx(0.0107 if inset == 0 else 0.0087)


@pytest.mark.parametrize("value", [None, "", "warn", "STRICT", 1])
def test_snap_validation(value):
    with pytest.raises(ValueError, match="snap must be"):
        Simulation(10e9, DOMAIN, snap=value)


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
@pytest.mark.parametrize("side", ["lo", "hi"])
@pytest.mark.parametrize("outside", [0.0, 0.001])
def test_domain_face_ends_compare_only_the_drawing_inside_domain(monkeypatch, boundary, side, outside):
    calls = _stub_runners(monkeypatch)
    lo, hi = ((-outside, 0.015) if side == "lo" else (0.005, DOMAIN[0] + outside))
    sheet = Box((lo, 0.005, 0.003), (hi, 0.015, 0.003))
    profiles = edge_aware_profiles(DOMAIN, DX, sheets=[sheet])
    sim = Simulation(10e9, DOMAIN, dx=DX, boundary=boundary, cpml_layers=2, **profiles)
    sim.add(sheet, material="pec")
    sim.add_source((0.002, 0.002, 0.002), component="ez")
    record = sim.realized_geometry()
    axis = record.entities[0].axes[0]
    assert axis.declared_bounds_m == (lo, hi)
    assert axis.comparison_bounds_m == (max(lo, 0), min(hi, DOMAIN[0]))
    assert axis.free_ends == ((False, True) if side == "lo" else (True, False))
    drawn = axis.comparison_bounds_m[1] - axis.comparison_bounds_m[0]
    assert axis.extent_m == pytest.approx(drawn, abs=1e-15)
    ctx = sim._campaign_ctx()
    for e, a, span in _sheet_solved_spans(ctx, ctx.interior_pec_entries()):
        public = record.entities[0].axes[a]
        assert span.comparison_bounds(e.lo[a], e.hi[a], sim._domain[a]) == public.comparison_bounds_m
        assert (span.free_lo, span.free_hi) == public.free_ends
    assert not sim.preflight().by_code(CODE)
    with pytest.raises(_ReachedRunner):
        sim.run(n_steps=1, compute_s_params=False)
    assert calls == [True]


def test_five_line_patch_reports_only_in_domain_free_edge_residuals():
    sim = Simulation(4e9, (0.08, 0.06, 0.02), boundary="cpml", cpml_layers=8, dx=0.005)
    sim.add(Box((-0.019, -0.0145, 0.0008), (0.019, 0.0145, 0.0008)), material="pec")
    row, = sim.preflight().by_code(CODE)
    assert row.severity == "error"
    assert f"drawn {_fmt_len(0.019)}" in str(row)
    assert f"drawn {_fmt_len(0.0145)}" in str(row)
    assert "-11.84%" in str(row)
    assert "-18.97%" in str(row)
    axes = sim.realized_geometry().entities[0].axes
    assert axes[0].extent_m == pytest.approx(0.01675)
    assert axes[1].extent_m == pytest.approx(0.01175)
