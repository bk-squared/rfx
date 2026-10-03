"""Thevenin voltage oracle using solved E line integrals, not port readouts.

At 1 MHz a millimetre vacuum gap is approximately open (no external R).
The loaded cases add 50 ohms across the whole gap. Independently drive
that environment with a known 1 A current (current moment I*d on each
edge), measure Z_in=integral(E dl)/I, then compare the port with
V_gap = w*Z_in/(50+Z_in). For a three-edge wire each external R is 50/3.
The source's internal 50 ohms remains present in open-circuit cases.
"""
from functools import lru_cache
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

FREQUENCY = 1e6
RESISTANCE = 50.
VOLTAGE_RTOL = 1e-3  # Pre-declared complex voltage-ratio bar.
CASES = ('uniform', 'offgrid', 'equal', 'transverse', 'wire')
PULSE = GaussianPulse(f0=8e9, bandwidth=.9)


def _geometry(case):
    domain = (.012, .010, .012)
    profiles = {}
    if case not in ('uniform', 'offgrid'):
        dx = .0005 if case in ('transverse', 'wire') else .001
        dy = .0004 if case in ('transverse', 'wire') else .001
        dz = np.full(12, .001)
        if case == 'wire':
            dz[3:6] = [.0008, .001, .0012]
        profiles = dict(dx_profile=np.r_[.001, np.full(round((domain[0]-.002)/dx), dx), .001],
                        dy_profile=np.r_[.001, np.full(round((domain[1]-.002)/dy), dy), .001], dz_profile=dz)
    sim = Simulation(freq_max=20e9, domain=domain, dx=.001,
                     cpml_layers=3, boundary='cpml', **profiles)
    positions = [(.004, .005, .003)]
    lengths = [.001]
    if case == 'wire':
        positions = [(.004, .005, z) for z in (.003, .0038, .0048)]
        lengths = [.0008, .001, .0012]
    return sim, positions, np.asarray(lengths)


@lru_cache(maxsize=None)
def response(case, loaded, independent_current=False, mutation=False):
    sim, positions, lengths = _geometry(case)
    n = len(positions)
    if independent_current:
        # Current-moment source is in A*m: I(t)*edge length gives 1 A.
        for pos, length in zip(positions, lengths):
            sim.add_source(pos, 'ez', amplitude_kind='current',
                           waveform=lambda t, d=float(length): PULSE(t) * d)
    else:
        # Off-grid declaration snaps to (4,5,3) mm on the 1 mm grid.
        declared = (.0043, .0052, .0034) if case == 'offgrid' else positions[0]
        sim.add_port(declared, 'ez', impedance=RESISTANCE,
                     waveform=PULSE, extent=float(sum(lengths)) if n > 1 else None)
    if loaded:
        for pos in positions:
            sim.add_lumped_rlc(pos, 'ez', R=RESISTANCE/n, topology='parallel')
    for pos in positions:
        sim.add_probe(pos, 'ez')
    grid = sim._build_grid() if case in ('uniform', 'offgrid') else sim._build_nonuniform_grid()
    if case == 'offgrid' and not independent_current:
        idx = tuple(grid.position_to_index(declared))
        realized = tuple((idx[a]-getattr(grid, f'pad_{axis}_lo'))*.001
                         for a, axis in enumerate('xyz'))
        print(f'offgrid declared={declared}, indices={idx}, realized={realized}')
        assert realized == pytest.approx(positions[0], abs=1e-12)
    dt = float(grid.dt)
    steps = int(round(1.2e-9/dt))
    if mutation:
        from rfx.runners import nonuniform as runner
        from rfx.core.yee import cell_component_e_coeffs

        def old_drive(g, cell, component, excitation, count, materials,
                      *, sigma_port, unit_field, time=None):
            cb = cell_component_e_coeffs(materials, cell, component, g.dt)[1]
            samples = jax.vmap(excitation)(jnp.arange(count, dtype=jnp.float32)*g.dt)
            return cb * samples * unit_field

        with mock.patch.object(runner, 'port_drive_waveform', old_drive):
            result = sim.run(n_steps=steps, compute_s_params=False, skip_preflight=True)
    else:
        result = sim.run(n_steps=steps, compute_s_params=False, skip_preflight=True)
    # Independently integrate the solved field on the declared, node-aligned
    # physical edges. Do not call port voltage/current/S-parameter helpers.
    gap = np.asarray(result.time_series, dtype=np.float64) @ lengths
    t = np.arange(steps)*dt
    phase = np.exp(-2j*np.pi*FREQUENCY*t)
    w = np.asarray(jax.vmap(PULSE)(jnp.arange(steps, dtype=jnp.float32)*dt))
    return np.sum(gap*phase) / np.sum(w*phase)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('loaded', [False, True], ids=['open', 'matched'])
def test_source_voltage_from_solved_gap_field(case, loaded):
    ratio = response(case, loaded)
    z_in = response(case, True, independent_current=True) if loaded else None
    if loaded:
        assert abs(z_in/RESISTANCE - 1) <= VOLTAGE_RTOL
    expected = z_in/(RESISTANCE+z_in) if loaded else 1.
    normalized = ratio/expected
    print(f'{case=} {loaded=} f={FREQUENCY:g} V/W={ratio!r} '
          f'Z_in={z_in!r} normalized={normalized!r} error={abs(normalized-1):.12g}')
    assert abs(normalized - 1) <= VOLTAGE_RTOL


def test_old_uniform_drive_fails_unequal_transverse_voltage_oracle():
    ratio = response('transverse', False, mutation=True)
    print(f'old Cb*w/d_parallel: V/W={ratio!r}, error={abs(ratio-1):.12g}')
    old_expected = RESISTANCE * .0005 * .0004 / .001
    assert abs(ratio/old_expected - 1) <= VOLTAGE_RTOL
    assert abs(ratio - 1) > VOLTAGE_RTOL


def msl_response(lane, mode, *, mutation=False):
    """An open two-plate feed; solved centre-line E integral at 1 MHz."""
    profiles = {} if lane == 'uniform' else dict(
        dx_profile=np.full(16, .001), dy_profile=np.full(14, .001),
        dz_profile=np.full(14, .001))
    sim = Simulation(freq_max=20e9, domain=(.016, .014, .014), dx=.001,
                     cpml_layers=3, boundary='cpml', **profiles)
    for z in (.004, .006):
        sim.add(Box((.004, .005, z), (.012, .009, z)), material='pec')
    sim.add_msl_port(position=(.006, .007, .004), width=.004, height=.002,
                     direction='+x', impedance=RESISTANCE, waveform=PULSE, mode=mode)
    for z in (.004, .005):
        sim.add_probe((.006, .007, z), 'ez')
    grid = sim._build_grid() if lane == 'uniform' else sim._build_nonuniform_grid()
    dt = float(grid.dt)
    steps = round(1.2e-9/dt)
    if mutation:
        from rfx.sources import port_drive
        original = port_drive.port_drive_waveform

        def old_legacy(*args, **kwargs):
            # Keep the shared builder; remove the port's load factor.
            kwargs['sigma_port'] = 1.
            return original(*args, **kwargs)

        with mock.patch.object(port_drive, 'port_drive_waveform', old_legacy):
            result = sim.run(n_steps=steps, compute_s_params=False, skip_preflight=True)
    else:
        result = sim.run(n_steps=steps, compute_s_params=False, skip_preflight=True)
    gap = np.sum(np.asarray(result.time_series, dtype=np.float64), axis=1)*.001
    phase = np.exp(-2j*np.pi*FREQUENCY*np.arange(steps)*dt)
    w = np.asarray(jax.vmap(PULSE)(jnp.arange(steps, dtype=jnp.float32)*dt))
    return np.sum(gap*phase)/np.sum(w*phase)


@pytest.mark.parametrize('lane,mode', [
    ('uniform', 'uniform'), ('uniform', 'laplace'), ('graded', 'laplace')])
def test_msl_source_voltage_from_solved_gap_field(lane, mode):
    ratio = msl_response(lane, mode)
    print(f'MSL {lane=} {mode=} f={FREQUENCY:g} V/W={ratio!r}, error={abs(ratio-1):.12g}')
    assert abs(ratio-1) <= VOLTAGE_RTOL


def test_old_msl_legacy_drive_fails_voltage_oracle():
    ratio = msl_response('uniform', 'uniform', mutation=True)
    print(f'old MSL legacy V/W={ratio!r}, error={abs(ratio-1):.12g}')
    assert abs(ratio-1) > VOLTAGE_RTOL
