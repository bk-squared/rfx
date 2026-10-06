"""Notch branches stay intact when the through-line ends at its port nodes."""
import numpy as np
import pytest

from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.preflight.line_stub import line_stub_findings
from rfx.probes.msl_wave_decomp import realized_trace_planes_on_column
from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry


@pytest.mark.parametrize('case', ['graded', 'tuning'])
def test_notch_builder_has_no_port_tail_and_covers_both_ports(case):
    if case == 'graded':
        from validation.research.multiband_nu.msl_notch_graded import build_graded
        sim = build_graded('A', 'offset')[0]
        branch = sim._geometry[2].shape
        assert branch.corner_lo == pytest.approx((.0117, .002759, .000254))
        assert branch.corner_hi == pytest.approx((.0123, .014759, .000254))
    else:
        from validation.tmtt_paper.msl_stub_notch_tuning import build_sim, F_TARGET
        sim = build_sim(np.asarray([F_TARGET], dtype=np.float32))[0]
    report = sim.preflight()
    assert not report.errors, [str(e) for e in report.errors]
    assert not line_stub_findings(sim)
    assert not any(f.code == 'line_stub_behind_port' for f in report)
    grid = sim._build_realized_grid()
    sheets, wires, lossy = [], [], []
    assemble = sim._assemble_materials_nu if hasattr(grid, 'dx_arr') else sim._assemble_materials
    result = assemble(grid, pec_sheets=sheets, pec_wires=wires, sheet_specs=lossy)
    edges = realized_pec_edge_masks(result[3], sheets, wires, periodic=sim._periodic_flags())
    for pe in sim._msl_ports:
        span = msl_cross_section_span(grid, msl_port_from_entry(pe))
        occupied = np.flatnonzero(np.asarray(edges[0])[:, span['w_centre'], span['n_hi']])
        endpoint = occupied[0] if pe.direction == '+x' else occupied[-1] + 1
        assert endpoint == span['i_feed']
        assert realized_trace_planes_on_column(
            edges, 2, (span['i_feed'], span['w_centre']), span['n_hi'],
            periodic=sim._periodic_flags()) != (None, None)
