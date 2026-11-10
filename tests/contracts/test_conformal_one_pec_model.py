"""The public field and S entries hand the time step exactly one PEC model."""
import numpy as np
import pytest
from pathlib import Path

from tests.contracts.conformal_scenes import capture_setups, scene


@pytest.mark.parametrize('name', ['S', 'Q', 'P'])
def test_one_rule_and_no_volume_staircase(monkeypatch, name):
    rows = capture_setups(monkeypatch)
    scene(name).run(n_steps=4, compute_s_params=False)
    scene(name).compute_waveguide_s_matrix(n_steps=4, normalize=False)
    assert len(rows) == 2
    for entry, row in zip(('run', 'waveguide'), rows):
        assert row['weights'] is not None and row['areas'] is not None
    for key in ('masks', 'weights', 'areas', 'materials'):
        assert len(rows[0][key]) == len(rows[1][key])
        for component, (a, b) in enumerate(zip(rows[0][key], rows[1][key])):
            np.testing.assert_array_equal(a, b, err_msg=f'{name}: {key}[{component}] differs between entries')


@pytest.mark.parametrize('name', ['S', 'Q', 'P'])
@pytest.mark.parametrize('entry', ['run', 'waveguide'])
def test_no_volume_staircase(monkeypatch, name, entry):
    rows = capture_setups(monkeypatch)
    sim = scene(name)
    if entry == 'run':
        sim.run(n_steps=4, compute_s_params=False)
    else:
        sim.compute_waveguide_s_matrix(n_steps=4, normalize=False)
    assert len(rows) == 1
    counts = tuple(np.count_nonzero(a) for a in rows[0]['masks'])
    assert counts == (0, 0, 0), (name, entry, 'volume staircase', counts)


@pytest.mark.parametrize('name', ['S', 'Q', 'P'])
def test_forward_refuses_conformal_pec(name):
    with pytest.raises(NotImplementedError, match=r'^the differentiable forward solve .* refuses') as error:
        scene(name).forward(n_steps=4, skip_preflight=True)
    assert 'conformal_pec=True' in str(error.value)


def test_reference_short_refuses_by_name():
    with pytest.raises(NotImplementedError, match=r'conformal PEC.*port_reference_sims.*/issues/1373'):
        scene('P').compute_waveguide_s_matrix(n_steps=4, normalize='flux',
                                            port_reference_sims=[scene('P')])


def test_strict_lane_admission_reads_conformal_open_faces():
    field = scene('S', snap='strict')
    lane = scene('S', ports=2, snap='strict')
    a = field.run(n_steps=4, compute_s_params=False)
    b = lane.compute_waveguide_s_matrix(n_steps=4, normalize=False)
    rows = []
    for result in (a, b):
        row = next(r for r in result.realized_geometry.realization_table.rows if r.label == 'geometry[0]')
        assert row.model.startswith('conformal open faces')
        rows.append(row)
    assert rows[0].open_area_m2 == rows[1].open_area_m2


def test_sphere_changes_only_domain_wall_rows(monkeypatch, record_property):
    rows = capture_setups(monkeypatch)
    scene('S').run(n_steps=4, compute_s_params=False)
    # Captured through run() on b969e2c40, with the same C4 observation wrapper.
    baseline = np.load(Path(__file__).parents[1] / 'data/conformal_node_plane/S-b969e2c40.npz')
    counts = {}
    for key in ('weights', 'areas'):
        for component, actual in enumerate(rows[0][key]):
            touches = np.zeros(actual.shape, dtype=bool)
            spans_y = component == 1 if key == 'weights' else component != 1
            touches[:, [0, 17, 18] if spans_y else [0, 18], :] = True
            old = baseline[f'{key}{component}']
            np.testing.assert_array_equal(actual[~touches], old[~touches],
                                          err_msg=f'S: {key}[{component}] changed away from domain walls')
            counts[f'{key}{component}'] = int(np.count_nonzero(actual != old))
    record_property('changed_wall_rows', counts)
