"""PEC and f0 rows report and admit one solved free-edge model."""
import warnings

import pytest

from rfx import Box, Simulation


@pytest.mark.parametrize("sigma", [5.8e7, 1e4])  # 1.405 ohm/sq at 5 GHz, below 1e-2 eta0
@pytest.mark.parametrize("aligned", [False, True])
def test_f0_and_pec_share_solved_spans_and_one_verdict(sigma, aligned):
    dx = .001
    offset = .35 * dx if aligned else 0.
    shape = Box((4*dx-offset, 4*dx-offset, 6*dx),
                (10*dx+offset, 12*dx+offset, 6*dx))
    rows = []
    for f0 in (True, False):
        sim = Simulation(10e9, (.020, .020, .020), dx=dx,
                         boundary="pec", snap="strict")
        sim.add_source((.002, .002, .002), component="ez", amplitude_kind="current")
        if f0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim.add_thin_conductor(shape, sigma_bulk=sigma,
                                       surface_impedance_f0=5e9)
        else:
            sim.add(shape, material="pec")
        label = "thin_conductor[0]" if f0 else "geometry[0]"
        report = sim.preflight(check_ntff=False, check_resolution=False)
        findings = [i for i in report if i.code == "sheet_effective_size"
                    or (i.code == "realization" and i.loc == label)]
        if aligned:
            assert findings == []
            sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
        else:
            with pytest.raises(ValueError, match="conductor sheet dimension"):
                sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
            assert len(findings) == 1
            assert findings[0].code == "sheet_effective_size"
            assert findings[0].severity == "error"
        row = sim.realized_geometry().realization_table.row(label)
        # Covered nodes span 6 and 8 cells; each of two free ends adds .35 cell.
        for axis, covered in zip(row.axes[:2], (6, 8), strict=True):
            assert axis.cell_size_m == pytest.approx(dx)
            assert axis.realized_span_m == pytest.approx((covered + 2*.35)*dx)
            assert axis.declared_span_m == pytest.approx(covered*dx + 2*offset)
            assert axis.miss_cells == pytest.approx(0. if aligned else .7, abs=1e-12)
            assert axis.miss_percent == pytest.approx(
                0. if aligned else 100*.7/covered, abs=1e-12)
        assert row.model == "solved sheet edges"
        rows.append(row)
    assert tuple(a.bounds_m for a in rows[0].axes) == tuple(a.bounds_m for a in rows[1].axes)
    assert tuple(a.realized_span_m for a in rows[0].axes) == tuple(
        a.realized_span_m for a in rows[1].axes)
