"""Manual same-process comparison on the existing calculator lock fixtures."""
import json
from unittest.mock import patch

import numpy as np
import pytest


def old_voltage(grid, ex_dft, ey_dft, *, center_xy, pin_radius, outer_radius, **kwargs):
    from rfx.measurement.coax_records import coaxial_tem_reference_plane_vi_from_cartesian_plane
    u = (np.arange(grid.nx, dtype=np.float64) - grid.pad_x_lo) * grid.dx
    v = (np.arange(grid.ny, dtype=np.float64) - grid.pad_y_lo) * grid.dx
    ex = np.asarray(ex_dft, dtype=np.complex128)
    ey = np.asarray(ey_dft, dtype=np.complex128)
    return np.asarray(coaxial_tem_reference_plane_vi_from_cartesian_plane(
        u, v, ex, ey, ex, ey, center_u_m=center_xy[0], center_v_m=center_xy[1],
        inner_radius=pin_radius, outer_radius=outer_radius, eps_r=2.1).vi.voltage,
        dtype=np.complex128)


@pytest.mark.parametrize('name', ['line', 'two_port', 'msl'])
def test_same_process_calculator_comparison(name):
    from tests.locks import test_sparams_split_bit_identity as fixtures
    from rfx.sources import coaxial_port
    build = getattr(fixtures, f'_coax_{name}_result')
    with patch.object(coaxial_port, 'coaxial_line_plane_voltage', old_voltage):
        before = build()
    after = build()
    field = 's11' if name == 'line' else 's_params'
    old, new = (np.asarray(getattr(result, field)) for result in (before, after))
    row = dict(calculator=name, max_magnitude_change=float(np.max(np.abs(np.abs(new)-np.abs(old)))),
               max_phase_change_deg=float(np.max(np.abs(np.angle(new*np.conj(old), deg=True)))),
               old_magnitude=np.abs(old).tolist(), new_magnitude=np.abs(new).tolist())
    if name == 'msl':
        row.update(old_column_power=np.sum(np.abs(old[:, 0])**2, axis=0).tolist(),
                   new_column_power=np.sum(np.abs(new[:, 0])**2, axis=0).tolist())
    print('S6B_PAIRED ' + json.dumps(row), flush=True)
    assert np.all(np.isfinite(old))
    assert np.all(np.isfinite(new))
