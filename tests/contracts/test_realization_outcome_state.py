"""Derived realization outcomes have a consumer and need no design input."""
from types import SimpleNamespace

from rfx import Simulation
from rfx._declarations import NOT_DECLARATIONS, declaration_key
from rfx.interop._design import EXCLUDED_SIMULATION_ATTRS
from rfx.model.realization_integrals import quantities, read_rows
from rfx.model.realization_table import RealizationRow


def test_outcomes_are_consumed_and_stand_in_has_no_drawn_reference():
    sim = Simulation(10e9, (.0103, .010, .010), dx=.001, boundary='pec', snap='declared')
    key = declaration_key(sim)
    report = sim.preflight(check_ntff=False, check_resolution=False)
    assert '_realization_early_outcomes' in dict(NOT_DECLARATIONS)
    assert '_realization_early_outcomes' in EXCLUDED_SIMULATION_ATTRS
    assert declaration_key(sim) == key
    table = sim._campaign_ctx()._realized.realization_table
    wall = [f for f in table.findings if f.code == 'wall_axis_off_node']
    assert len(wall) == 1 and wall[0].loc == 'x'
    assert any(str(i) == str(wall[0]) for i in report)
    assert 'legacy:wall:x' in table.judged

    grid = sim._campaign_ctx().grid
    quantities(grid, create=True)['filled'] = {0: {3: 1e-9}}
    row = RealizationRow('geometry[0]', 'material')
    escaped = None
    try:
        result, = read_rows((row,), grid, SimpleNamespace())
    except AttributeError as exc:
        escaped = exc
    assert escaped is None, f'stand-in geometry was required: {escaped}'
    assert result.status == 'not judged: no reference'
    assert result.filled_volume_m3 == 1e-9
    assert result.declared_volume_m3 is None
    assert result.integral_comparison is None
