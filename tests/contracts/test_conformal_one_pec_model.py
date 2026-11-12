"""The public field and S entries hand the time step exactly one PEC model."""
import numpy as np
import pytest

from tests.contracts.conformal_scenes import assert_same_setup, capture_setups, scene


@pytest.mark.parametrize('name', ['S', 'Q', 'P'])
def test_one_rule_and_no_volume_staircase(monkeypatch, name):
    rows = capture_setups(monkeypatch)
    scene(name).run(n_steps=4, compute_s_params=False)
    scene(name, ports=2).compute_waveguide_s_matrix(n_steps=4, normalize=False)
    assert len(rows) == 3  # run(), then one device setup per lane excitation.
    for setup_index, device in enumerate(rows[1:]):
        assert_same_setup(rows[0], device, label=f'{name}: lane device {setup_index}')


def test_reference_short_refuses_by_name():
    with pytest.raises(NotImplementedError, match=r'conformal PEC.*port_reference_sims.*/issues/1373'):
        scene('P', ports=2).compute_waveguide_s_matrix(n_steps=4, normalize='flux',
                                            port_reference_sims=[scene('P'), scene('P')])


def test_strict_lane_admission_reads_conformal_open_faces(monkeypatch):
    setups = capture_setups(monkeypatch)
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
    assert len(setups) == 3
    for setup_index, device in enumerate(setups[1:]):
        assert_same_setup(setups[0], device, label=f'S strict: lane device {setup_index}')


@pytest.mark.parametrize('entry', ['run', 'waveguide'])
def test_conformal_sphere_keeps_only_hand_counted_sheet_edges(monkeypatch, entry):
    """The sheet/wire branch drops the sphere staircase and retains the sheet."""
    from rfx import Box

    rows = capture_setups(monkeypatch)
    sim = scene('S', ports=1 if entry == 'run' else 2)
    # dx = 0.00127 m. Corners / dx = (40, 4, 2), (40, 8, 6).
    # The yz sheet is away from the sphere (x ~55), ports (x=24,88),
    # and domain walls. Ex: none; Ey: 4*5=20; Ez: 5*4=20.
    sim.add(Box((.05080, .00508, .00254), (.05080, .01016, .00762)), material='pec')
    if entry == 'run':
        sim.run(n_steps=4, compute_s_params=False)
    else:
        sim.compute_waveguide_s_matrix(n_steps=4, normalize=False)
    assert len(rows) == (1 if entry == 'run' else 2)
    for setup_index, row in enumerate(rows):
        expected = tuple(np.zeros(row['grid'].shape, dtype=bool) for _ in range(3))
        # The 16 low-x CPML cells shift physical node 40 to array index 56.
        assert row['grid'].pad_x_lo == 16
        expected[1][56, 4:8, 2:7] = True
        expected[2][56, 4:9, 2:6] = True
        # No port release intersects these edges; no indices are excluded.
        assert tuple(np.count_nonzero(a) for a in expected) == (0, 20, 20)
        for component, (actual, sheet) in enumerate(zip(row['masks'], expected)):
            np.testing.assert_array_equal(
                actual, sheet, err_msg=f'{entry} setup {setup_index}: sheet-only E{"xyz"[component]}')
