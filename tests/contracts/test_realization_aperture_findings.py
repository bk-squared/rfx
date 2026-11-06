"""Off-grid WR-90 diagnostics retain every aperture axis."""
import pytest

from tests.unit.ports.test_port_aperture_rasterization import _snap_sim, _build


@pytest.mark.parametrize('profile', ['discrete', 'analytic', 'roundup'])
def test_wr90_snap_reports_each_axis_and_keeps_user_wall(profile):
    sim = (_build(dx=.001, domain=(.10, .030, .010), y_range=(0., .02286),
                  freqs=[9e9], f0=9e9) if profile == 'roundup' else _snap_sim(profile))
    report = sim.preflight(check_ntff=False, check_resolution=False)
    snaps = report.by_code('port_aperture_snap')
    axes = ['y'] if profile == 'roundup' else ['y', 'z']
    assert len(snaps) == len(axes)
    assert all(sum(f'declared {axis}-width' in str(i) for i in snaps) == 1 for axis in axes)
    assert all(i.source == '_check_waveguide_port_aperture_snap' for i in snaps)
    assert not sim._geometry  # These walls come from the declared domain.
    walls = report.by_code('wall_axis_off_node')
    assert [i.loc for i in walls] == ([] if profile == 'roundup' else ['z'])
