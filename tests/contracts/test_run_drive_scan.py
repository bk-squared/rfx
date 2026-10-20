"""run()'s S drive scan on the uniform and the graded mesh: one driver, every column.

Two 50 ohm lumped ports, port 0 driven and port 1 passive, beside an eps_r = 4.3
block whose faces sit between cell faces; port 1 is 0.3 mm from the block. run()
drives each port in turn (compute_lumped_wire_s_matrix_via_scan), so the passive
port's column is measured, and the scanned matrix of this passive structure is
reciprocal. Subpixel smoothing reaches every drive: it moves S22 most (the port
next to the block face), and run()'s driven column equals forward()'s with the
same smoothing, bit for bit.
"""
import warnings

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation

FREQS = np.linspace(3e9, 9e9, 7)


def _box(mesh):
    kw = (dict(dx_profile=np.r_[np.full(6, 1e-3), np.full(4, 0.6e-3), np.full(6, 1e-3)],
               domain=(0, 12e-3, 10e-3)) if mesh == 'graded' else dict(domain=(12.4e-3, 12e-3, 10e-3)))
    sim = Simulation(freq_max=12e9, dx=1e-3, boundary='cpml', cpml_layers=6, **kw)
    for n, x in enumerate((4e-3, 8e-3)):
        sim.add_port(position=(x, 6e-3, 4e-3), component='ez', impedance=50.0, excite=n == 0,
                     waveform=GaussianPulse(f0=6e9, bandwidth=0.8))
    sim.add_material('d', eps_r=4.3)
    sim.add(Box((6.3e-3, 2.3e-3, 1.3e-3), (7.7e-3, 9.7e-3, 8.7e-3)), material='d')
    return sim


def _run(mesh, smoothing):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(_box(mesh).run(n_steps=300, skip_preflight=True, compute_s_params=True,
                                          s_param_freqs=FREQS, subpixel_smoothing=smoothing).s_params)


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_scanned_matrix_is_reciprocal(mesh):
    s = _run(mesh, False)
    assert s.shape == (2, 2, FREQS.size) and np.isfinite(s).all()
    # Measured 4e-6 (graded) and 6e-6 (uniform) of a peak near 1 with smoothing on.
    assert np.abs(s[0, 1] - s[1, 0]).max() <= 1e-4 * np.abs(s).max()


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_smoothing_reaches_every_drive(mesh):
    on, off = _run(mesh, True), _run(mesh, False)
    # Port 1 is 0.3 mm from the block face: smoothing moves its own reflection
    # (measured 0.06 graded, 0.09 uniform); a drive that drops it moves nothing.
    assert np.abs(on[1, 1] - off[1, 1]).max() > 1e-2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forward = np.asarray(_box(mesh).forward(n_steps=300, skip_preflight=True, port_s11_freqs=FREQS,
                                                subpixel_smoothing=True).s_params)
    assert on[:, 0].tobytes() == forward[:, 0].tobytes()
