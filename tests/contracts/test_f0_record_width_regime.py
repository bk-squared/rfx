"""D:c1d171: only PEC-like f0 sheets claim the measured free-edge span."""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.core.yee import EPS_0, MU_0


@pytest.mark.parametrize('ratio', [0.999, 1.001], ids=['below', 'above'])
def test_f0_record_width_regime(ratio):
    dx, f0 = 1e-3, 9e9
    resistance = ratio * 1e-2 * np.sqrt(MU_0 / EPS_0)
    sim = Simulation(freq_max=10e9, domain=(10 * dx,) * 3, dx=dx,
                     boundary='pec', snap='declared')
    sim.add_thin_conductor(Box((2 * dx, 3 * dx, 5 * dx), (7 * dx, 8 * dx, 5 * dx)),
                           sigma_bulk=np.pi * f0 * MU_0 / resistance**2,
                           thickness=35e-6, surface_impedance_f0=f0)
    grid = sim._build_grid()
    # Expected endpoints come from the grid nodes, independently of report helpers.
    nodes = np.arange(grid.ny, dtype=float) * float(grid.dx)
    offset = 0.35 * float(grid.dx) if ratio < 1 else 0.0
    expected = (nodes[3] - offset, nodes[8] + offset)
    record = sim.realized_geometry()
    entity = record.entities[0]
    assert entity.axes[1].bounds_m == pytest.approx(expected, rel=1e-9)
    assert entity.axes[1].extent_m == pytest.approx(expected[1] - expected[0], rel=1e-9)
    note = ('edge measured at σ 5.8e7 S/m; PEC-like regime up to 1e-2·η0 by rule, not measured between'
            if ratio < 1 else 'edge not measured')
    assert entity.edge_note == note
    row = record.realization_table.row('thin_conductor[0]')
    assert row.edge_note == note
    assert row.model == ('solved sheet edges' if ratio < 1 else 'node span')


@pytest.mark.parametrize('snap', ['strict', 'declared'])
@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('path', ['preflight', 'run', 'skip_preflight'])
def test_above_threshold_extent_warns_without_refusal(snap, graded, path):
    import warnings

    from rfx.model.realization_verdict import judge_realization

    dx, f0 = 1e-3, 9e9
    resistance = 1.001 * 1e-2 * np.sqrt(MU_0 / EPS_0)
    profiles = {'dz_profile': np.full(10, dx)} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(10 * dx,) * 3, dx=dx,
                     boundary='pec', snap=snap, **profiles)
    sim.add_thin_conductor(Box((2.15 * dx, 3.15 * dx, 5 * dx),
                               (7.85 * dx, 8.85 * dx, 5 * dx)),
                           sigma_bulk=np.pi * f0 * MU_0 / resistance**2,
                           thickness=35e-6, surface_impedance_f0=f0)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    cells = np.asarray(grid.cells(1), dtype=float)
    nodes = np.r_[0., np.cumsum(cells)]
    occupied = nodes[(nodes >= 3.15 * dx) & (nodes <= 8.85 * dx)]
    expected = occupied[-1] - occupied[0]
    # The node span misses the drawing by >1%; D:8ea1e2 still forbids refusal.
    assert abs(expected / ((8.85 - 3.15) * dx) - 1) > .01
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if path == 'preflight':
            report = sim.preflight()
            assert not report.errors
            table = report.realization_table
            findings = [f for f in report if f.code == 'realization']
        else:
            result = sim.run(n_steps=1, compute_s_params=False,
                             skip_preflight=path == 'skip_preflight')
            table = result.realized_geometry.realization_table
            findings = [w.message for w in caught
                        if getattr(w.message, 'code', None) == 'realization']
    row = table.row('thin_conductor[0]')
    assert row.axes[1].realized_span_m == pytest.approx(expected)
    assert row.status == 'not judged: edge not measured'
    assert len(findings) == 1
    assert findings[0].severity == 'warning'
    assert 'edge not measured' in str(findings[0])
    assert not any(f.severity == 'error' for f in table.findings)
    assert judge_realization(table, snap) == ()  # Admission/finalization delivers once.
