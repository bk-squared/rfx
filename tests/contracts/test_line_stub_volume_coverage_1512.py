"""Build-only checks for the #1512 volume diagnostic and transition fixtures."""
import numpy as np
import pytest

from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.preflight.line_stub import line_stub_findings
from rfx.probes.msl_wave_decomp import realized_trace_planes_on_column
from rfx.geometry.port_termination import default_msl_terminates
from rfx.sources.msl_port import (
    msl_cross_section_span, msl_port_from_entry, validate_msl_port_geometry,
)


def _build(case):
    if case.startswith("beta-"):
        from types import SimpleNamespace
        from scripts.diagnostics.msl_beta_rail_e2e import _build as build
        wide = "wide" in case
        args = SimpleNamespace(domain_x=.04 if wide else .02, feed_x=.005 if wide else .0025)
        return build(args, 9.8 if "misdeclared" in case else 2.2)
    if case.startswith('ladder-'):
        from scripts.diagnostics.patch_sheet_realization_ladder import build
        return build(case[7:])
    if case == 'ab-volume':
        from scripts.diagnostics.sheet_vs_volume_patch_radiation_ab import build
        return build('volume')
    if case == 'production':
        from scripts.diagnostics.patch_edgefed_production_verify import build, H_SUB
        return build(H_SUB/4, 18)
    if case == 'witness':
        from scripts.diagnostics.patch_edgefed_match_vs_resonance_witness import build
        return build()
    if case.startswith('inset'):
        from scripts.diagnostics.patch_inset_match_vs_pull import build
        return build(.002 if case.endswith('fixed') else 0., tip_fixed=case.endswith('fixed'))[0]
    raise AssertionError(case)


@pytest.mark.parametrize('case', ['ladder-vol1', 'ladder-vol2', 'ab-volume',
                                  'production', 'witness', 'inset-0', 'inset-fixed',
                                  'beta-wide-correct', 'beta-wide-misdeclared'])
def test_volume_has_zero_realized_stub_and_passes_port_preflight(case):
    sim = _build(case)
    grid = sim._build_realized_grid()
    sheets, wires, lossy = [], [], []
    result = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires, sheet_specs=lossy)
    edges = realized_pec_edge_masks(result[3], sheets, wires, periodic=sim._periodic_flags())
    for pe in sim._msl_ports:
        port = msl_port_from_entry(pe)
        span = msl_cross_section_span(grid, port)
        validate_msl_port_geometry(grid, port, pec_edge_masks=edges, sheet_specs=lossy,
                                  pec_faces=sim._boundary_spec.pec_faces(),
                                  periodic=sim._periodic_flags(), name=pe.name)
        assert default_msl_terminates(sim, grid, position=pe.position, width=pe.width,
                                      height=pe.height, direction=pe.direction)
        assert realized_trace_planes_on_column(
            edges, 2, (span["i_feed"], span["w_centre"]), span["n_hi"],
            periodic=sim._periodic_flags()) != (None, None)
        # Read the actual longitudinal PEC edges handed to the stepper.
        occupied = np.flatnonzero(np.asarray(edges[0])[:, span['w_centre'], span['n_hi']])
        assert occupied[0] == span['i_feed']
    assert line_stub_findings(sim, grid) == []


def test_beta_default_outside_band_drawing_stays_unchanged():
    from rfx.preflight.line_stub import resonant_odd_orders
    sim = _build("beta-default-correct")
    assert sim._geometry[1].shape.corner_lo[0] == .001
    finding, = line_stub_findings(sim)
    assert resonant_odd_orders(finding, (4e9, 18e9)) is None


@pytest.mark.parametrize('wide', [False, True])
def test_beta_arms_share_strip_geometry_and_stub_decision(wide):
    from types import SimpleNamespace
    from scripts.diagnostics.msl_beta_rail_e2e import _build, FREQS
    from rfx.preflight.line_stub import require_no_resonant_line_stub
    args = SimpleNamespace(domain_x=.04 if wide else .02, feed_x=.005 if wide else .0025)
    arms = [_build(args, eps, spacing) for eps, spacing in [(2.2, None), (2.2, 2), (9.8, None)]]
    assert all(sim._geometry == arms[0]._geometry for sim in arms)
    for sim in arms:
        require_no_resonant_line_stub(sim, FREQS)
        findings = line_stub_findings(sim)
        if wide:
            assert findings == []
        else:
            assert len(findings) == 1
            assert findings[0].frequency_hz == pytest.approx(line_stub_findings(arms[0])[0].frequency_hz)
            assert sim._geometry[1].shape.corner_lo[0] == .001
