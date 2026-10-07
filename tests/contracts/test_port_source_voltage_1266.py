"""Thevenin voltage oracle using solved E line integrals, not port readouts.

At 1 MHz a millimetre vacuum gap is approximately open (no external R).
The loaded cases add 50 ohms across the whole gap. Independently drive
that environment with a known 1 A current (current moment I*d on each
edge), measure Z_in=integral(E dl)/I, then compare the port with
V_gap = w*Z_in/(50+Z_in). For a three-edge wire each external R is 50/3.
The source's internal 50 ohms remains present in open-circuit cases.

The 1 MHz readout over 1.2 ns is effectively the DC ratio sum(gap)/sum(w),
not a resolved sinusoidal steady-state measurement. sum(w) is the remnant
of the differentiated Gaussian cut at t=0. The response must decay before
truncation: reviewer B2's MSL record-length check measured errors 4.2e-3
at 0.9 ns, 1.8e-4 at 1.2 ns, and 5e-5 at 3 ns. We retain the 1.2 ns
record and 1e-3 bar; a short record is not an independent voltage oracle.
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
CASES = ('uniform', 'offgrid', 'equal', 'transverse', 'wire', 'wire_uniform')
PULSE = GaussianPulse(f0=8e9, bandwidth=.9)


def _geometry(case):
    domain = (.012, .010, .012)
    profiles = {}
    if case not in ('uniform', 'offgrid', 'wire_uniform'):
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
    if case == 'wire_uniform':
        positions = [(.004, .005, z) for z in (.003, .004, .005)]
        lengths = [.001] * 3
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
    grid = sim._build_grid() if case in ('uniform', 'offgrid', 'wire_uniform') else sim._build_nonuniform_grid()
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
        from rfx.sources import port_drive
        original = port_drive.port_drive_waveform

        def old_drive(*args, **kwargs):
            # Mutate the shared builder, including the graded runner's
            # imported alias. Uniform lumped/wire callers import it locally.
            kwargs['sigma_port'] = 1.
            return original(*args, **kwargs)

        with (
            mock.patch.object(port_drive, 'port_drive_waveform', old_drive),
            mock.patch.object(runner, 'port_drive_waveform', old_drive),
        ):
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


@pytest.mark.parametrize('case', ['uniform', 'wire_uniform'])
def test_shared_builder_without_sigma_fails_uniform_voltage_oracle(case):
    ratio = response(case, False, mutation=True)
    n = 3 if case == 'wire_uniform' else 1
    expected_old = RESISTANCE * .001**2 / (n * .001)
    print(f'{case}: shared builder without sigma V/W={ratio!r}')
    assert abs(ratio/expected_old - 1) <= VOLTAGE_RTOL
    assert abs(ratio - 1) > VOLTAGE_RTOL


def msl_response(lane, mode, *, mutation=False):
    """Open two-plate feed; near-DC solved-field ratio after 1.2 ns decay."""
    assert lane in ('uniform', 'x_graded', 'y_graded', 'z_graded')
    profiles = {} if lane == 'uniform' else dict(
        dx_profile=np.full(16, .001), dy_profile=np.full(14, .001),
        dz_profile=np.full(14, .001))
    if lane == 'x_graded':
        # Unequal propagation cells at the feed; y/z cross-section stays uniform.
        profiles['dx_profile'][5:8] = [.0008, .0009, .0013]
    elif lane == 'y_graded':
        profiles['dy_profile'][5:9] = [.0007, .0013, .0006, .0014]
    elif lane == 'z_graded':
        profiles['dz_profile'][4:6] = [.0008, .0012]
    sim = Simulation(freq_max=20e9, domain=(.016, .014, .014), dx=.001,
                     cpml_layers=3, boundary='cpml', **profiles)
    # #1512: the upper plate starts at the port's grid node. Drawn from 4 mm it left a
    # 2 mm open stub behind the port whose effective quarter wave (23.7 GHz) is within
    # 1.5x of the 0..20 GHz default read. The pinned V/w ratios are unchanged by this.
    from rfx.preflight.line_port_coverage import port_node_coordinate
    sim.add(Box((.004, .005, .004), (.012, .009, .004)), material='pec')
    sim.add(Box((port_node_coordinate(sim, (.006, .007, .004)), .005, .006),
                (.012, .009, .006)), material='pec')
    sim.add_msl_port(position=(.006, .007, .004), width=.004, height=.002,
                     direction='+x', impedance=RESISTANCE, waveform=PULSE, mode=mode)
    probe_z = (.004, .0048) if lane == 'z_graded' else (.004, .005)
    lengths = np.diff(np.r_[probe_z, .006])
    for z in probe_z:
        sim.add_probe((.006, .007, z), 'ez')
    grid = sim._build_grid() if lane == 'uniform' else sim._build_nonuniform_grid()
    if lane == 'x_graded':
        from rfx.nonuniform import position_to_index
        feed = position_to_index(grid, (.006, .007, .004))
        dx = np.asarray(grid.cells('x'))
        assert dx[feed[0]-1] != dx[feed[0]]
        np.testing.assert_allclose(grid.cells('y'), .001)
        np.testing.assert_allclose(grid.cells('z'), .001)
        print(f'x-graded feed index={feed}, x={grid.node_of("x", feed[0])}, '
              f'adjacent dx={dx[feed[0]-1:feed[0]+1]}')
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
    gap = np.asarray(result.time_series, dtype=np.float64) @ lengths
    phase = np.exp(-2j*np.pi*FREQUENCY*np.arange(steps)*dt)
    w = np.asarray(jax.vmap(PULSE)(jnp.arange(steps, dtype=jnp.float32)*dt))
    return np.sum(gap*phase)/np.sum(w*phase)


@pytest.mark.parametrize('lane,mode', [
    ('uniform', 'uniform'), ('uniform', 'laplace'), ('x_graded', 'laplace')])
def test_msl_source_voltage_from_solved_gap_field(lane, mode):
    ratio = msl_response(lane, mode)
    print(f'MSL {lane=} {mode=} f={FREQUENCY:g} V/W={ratio!r}, error={abs(ratio-1):.12g}')
    assert abs(ratio-1) <= VOLTAGE_RTOL


def test_old_msl_legacy_drive_fails_voltage_oracle():
    ratio = msl_response('uniform', 'uniform', mutation=True)
    print(f'old MSL legacy V/W={ratio!r}, error={abs(ratio-1):.12g}')
    assert abs(ratio-1) > VOLTAGE_RTOL


@pytest.mark.parametrize('lane', ['y_graded', 'z_graded'])
@pytest.mark.xfail(strict=True, reason="MSL feed profile normalized on the centre column only; graded feed cross-section gives V/w 0.98-1.01 (#1373 note)")
def test_msl_graded_cross_section_source_voltage(lane):
    """Measure the limitation against the desired voltage, so a fix XPASSes.

    Do not assert proximity to the known wrong value: that would keep this
    test failing after the profile normalization is fixed.
    """
    ratio = msl_response(lane, 'laplace')
    deviation = abs(ratio - 1)
    print(f'MSL {lane}: V/W={ratio!r}, deviation={deviation:.12g}')
    assert deviation <= VOLTAGE_RTOL
