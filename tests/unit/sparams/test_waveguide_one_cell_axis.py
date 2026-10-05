"""Discrete TE10 on a 22 mm guide with one transverse height cell."""
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.core.yee import EPS_0, MU_0
from rfx.sources.waveguide_port import C0_LOCAL, _compute_mode_impedance

FREQS = jnp.array([9e9, 10e9, 11e9])
DX = .001


def _guide(height):
    sim = Simulation(freq_max=11e9, domain=(.12, .022, height*DX), dx=DX,
                     boundary=BoundarySpec(x='cpml', y='pec', z='pec'), cpml_layers=40)
    for position, direction, name in [(.015, '+x', 'left'), (.097, '-x', 'right')]:
        sim.add_waveguide_port(position, direction=direction, name=name, mode=(1, 0),
                               mode_type='TE', mode_profile='discrete', freqs=FREQS,
                               f0=10e9, bandwidth=.4, ref_offset=3, probe_offset=10)
    return sim


def test_te10_cutoff_impedance_and_height_witnesses():
    cutoffs, impedances = [], []
    for height in (1, 2, 4):
        sim = _guide(height)
        grid = sim._build_grid()
        for port in sim._waveguide_ports:
            cfg = sim._build_waveguide_port_config(port, grid, FREQS,
                                                   int(grid.num_timesteps(4.)))
            widths = np.asarray(cfg.u_widths)
            assert widths.size == 22
            assert len(cfg.v_widths) == height
            np.testing.assert_allclose(widths.sum(), .022, rtol=1e-6)
            fc = (2 / widths[0]) * np.sin(np.pi / (2 * len(widths))) * C0_LOCAL / (2*np.pi)
            np.testing.assert_allclose(cfg.f_cutoff, fc, rtol=1e-6)
            # Continuum longitudinal limit, using the discrete transverse cutoff.
            expected = np.sqrt(MU_0/EPS_0) / np.sqrt(1-(fc/np.asarray(FREQS))**2)
            z = _compute_mode_impedance(FREQS, cfg.f_cutoff, 'TE')
            np.testing.assert_allclose(z, expected, rtol=1e-6)
            # Actual Yee time/longitudinal stencil: replace omega by its sine symbol.
            omega_d = 2*np.sin(np.pi*np.asarray(FREQS)*cfg.dt)/cfg.dt
            expected_yee = MU_0*omega_d / np.sqrt((omega_d/C0_LOCAL)**2-(2*np.pi*fc/C0_LOCAL)**2)
            z_yee = _compute_mode_impedance(FREQS, cfg.f_cutoff, 'TE', dt=cfg.dt, dx=cfg.dx)
            np.testing.assert_allclose(z_yee, expected_yee, rtol=1e-6)
            cutoffs.append(cfg.f_cutoff)
            impedances.append(z_yee)
    np.testing.assert_allclose(cutoffs, cutoffs[0], rtol=1e-6)
    np.testing.assert_allclose(impedances, np.broadcast_to(impedances[0], (6, 3)), rtol=1e-6)


def test_one_cell_twoport_empty_guide():
    # No reference-run normalization: with normalize='flux' an empty guide's
    # S11 is the device run minus itself (exactly 0) and cannot see a
    # reflection. Raw S11 must stay below -40 dB and |S21| near 1.
    result = _guide(1).compute_waveguide_s_matrix(num_periods=60, normalize=False)
    s = np.asarray(result.s_params)
    assert np.isfinite(s).all()
    assert np.max(np.abs(s[0, 0])) < .01
    np.testing.assert_allclose(np.abs(s[1, 0]), 1., atol=.005)
