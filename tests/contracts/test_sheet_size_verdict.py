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
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-15)
    findings = sim.preflight().by_code(CODE)
    offenders = []
    for e in record.entities:
        for a in e.axes:
            drawn = a.declared_bounds_m[1] - a.declared_bounds_m[0]
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
