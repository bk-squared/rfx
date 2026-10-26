"""D5: both entries retain witness rows; one excited lumped port needs one scan."""
import numpy as np
import pytest

from tests.contracts.run_forward_census import BUILDERS, invoke, port_drive_witness_leaves, port_s_pair
from tests.contracts.test_run_forward_one_pipeline import check_fields, test_both_refuse as check_refusal


@pytest.mark.parametrize('freqs', [None, np.array([2e9, 7e9, 20e9])], ids=['default', 'explicit'])
def test_single_lumped_s_uses_only_main_scan(monkeypatch, freqs):
    from rfx import simulation
    calls = []
    step = simulation.run

    def count(*args, **kwargs):
        calls.append(1)
        return step(*args, **kwargs)

    monkeypatch.setattr(simulation, 'run', count)
    run = invoke(BUILDERS['P-lumped']('uniform'), 'run', 'uniform', 150,
                 {} if freqs is None else dict(compute_s_params=True, s_param_freqs=freqs))
    assert len(calls) == 1, 'one declared excited lumped port must use its main record'
    forward = invoke(BUILDERS['P-lumped']('uniform'), 'forward', 'uniform', 150,
                     dict(port_s11_freqs=np.asarray(run.freqs)))
    np.testing.assert_array_equal(run.freqs, forward.freqs)
    np.testing.assert_allclose(run.s_params, forward.s_params, rtol=0,
                               atol=1e-4*np.max(np.abs(forward.s_params)))
    assert run.settling_witness['scored_port_record_ids'] == ['port:0:mode:0']
    assert 'per_drive' not in run.settling_witness
    assert len(run.settling_witness['runs']) == len(forward.settling_witness['runs']) == 2
    for key in ('share_per_bin', 'error_per_bin'):
        np.testing.assert_array_equal(run.settling_witness[key], forward.settling_witness[key])


@pytest.mark.parametrize('path', ['uniform', 'graded', 'multi-device-uniform'])
def test_forward_keeps_waveguide_witness_rows(path):
    if path == 'multi-device-uniform':
        for entry in ('run', 'forward'):
            with pytest.raises(NotImplementedError) as caught:
                invoke(BUILDERS['P-waveguide'](path), entry, path, 150)
            assert 'waveguide ports (add_waveguide_port) are not supported on' in str(caught.value)
        return
    run = invoke(BUILDERS['P-waveguide'](path), 'run', path, 150)
    forward = invoke(BUILDERS['P-waveguide'](path), 'forward', path, 150)
    actual, expected = run.settling_witness, forward.settling_witness
    assert len(expected['runs']) == len(actual['runs']) == 2
    for left, right in zip(actual['runs'], expected['runs']):
        for key in ('share_per_bin', 'error_per_bin'):
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize('declaration,path', [
    ('P-lumped', 'uniform'), ('P-lumped2', 'uniform'),
    ('P-waveguide', 'uniform'), ('P-waveguide', 'graded'),
    ('P-waveguide', 'multi-device-uniform'), ('P-lumped', 'multi-device-uniform'),
])
@pytest.mark.parametrize('direction', ('a', 'b'))  # equal S requests (#1373)
def test_d5_census_disposition(declaration, path, direction):
    if (declaration, path) == ('P-waveguide', 'multi-device-uniform'):
        check_refusal(declaration, path, direction)
        return
    check_fields(declaration, path, direction)


@pytest.fixture(scope='module')
def two_port_s_results():
    return port_s_pair('P-lumped2', 'uniform')


@pytest.mark.parametrize('reverse_rows', [False, True])
def test_two_port_drive_witnesses_use_port_record_kind(two_port_s_results, reverse_rows):
    from copy import deepcopy
    run, forward = (two_port_s_results[entry]['result'] for entry in ('run', 'forward'))
    if reverse_rows:
        # Probe and port rows may move independently on either entry.
        witness = deepcopy(forward.settling_witness)
        witness['runs'] = type(witness['runs'])(reversed(witness['runs']))
        # ForwardResult derives its witness; hand the helper a view with the moved rows.
        from types import SimpleNamespace
        forward = SimpleNamespace(sparam_time_records=forward.sparam_time_records,
                                  settling_witness=witness, s_params=forward.s_params)
    leaves = port_drive_witness_leaves(BUILDERS['P-lumped2']('uniform'), run, forward)
    assert set(leaves) == {
        f'{prefix}/per_drive/{column}/{key}'
        for prefix in ('settling_witness', 'settling_witness/runs/0')
        for column in (0, 1) for key in ('error_per_bin', 'share_per_bin')
    }


def test_two_port_drive_witnesses_follow_run_row_location(two_port_s_results):
    from copy import deepcopy
    run, forward = (two_port_s_results[entry]['result'] for entry in ('run', 'forward'))
    witness = deepcopy(run.settling_witness)
    witness['runs'] = type(witness['runs'])(reversed(witness['runs']))
    run = run._replace(settling_witness=witness)
    leaves = port_drive_witness_leaves(BUILDERS['P-lumped2']('uniform'), run, forward)
    assert set(leaves) == {
        f'{prefix}/per_drive/{column}/{key}'
        for prefix in ('settling_witness', 'settling_witness/runs/1')
        for column in (0, 1) for key in ('error_per_bin', 'share_per_bin')
    }


def test_two_port_drive_witnesses_tell_driven_from_passive():
    # 600 steps: both drive witnesses are determined and differ, so the
    # driven entry's equality with forward's port row is not a constant.
    from copy import deepcopy
    pair = port_s_pair('P-lumped2', 'uniform', n_steps=600)
    run, forward = (pair[entry]['result'] for entry in ('run', 'forward'))
    sim = BUILDERS['P-lumped2']('uniform')
    port_drive_witness_leaves(sim, run, forward)
    row = next(r for r in run.settling_witness['runs'] if r.get('route') == 's_channels')
    driven, passive = row['per_drive']
    assert driven['status'] == passive['status'] == 'pass'
    assert not np.array_equal(driven['error_per_bin'], passive['error_per_bin'])
    # Each entry judges the S it returns: run's S row (and its top level when
    # that repeats per_drive) is the worst drive, the passive one here;
    # forward's S row is its driven record (decision C).
    worst = max((driven, passive), key=lambda drive: drive['db'])
    assert worst is passive
    forward_row = next(r for r in forward.settling_witness['runs'] if r.get('route') == 's_channels')
    for key in ('db', 'error_per_bin', 'share_per_bin'):
        np.testing.assert_array_equal(row[key], worst[key])
        np.testing.assert_array_equal(forward_row[key], driven[key])
        if 'per_drive' in run.settling_witness:
            np.testing.assert_array_equal(run.settling_witness[key], worst[key])
    assert row['db'] != forward_row['db']
    witness = deepcopy(run.settling_witness)
    for holder in [witness] + [r for r in witness['runs'] if r.get('route') == 's_channels']:
        if 'per_drive' in holder:
            holder['per_drive'] = type(holder['per_drive'])(reversed(holder['per_drive']))
    with pytest.raises(AssertionError):
        port_drive_witness_leaves(sim, run._replace(settling_witness=witness), forward)


@pytest.mark.parametrize('mode,drives', [('passive', 1), ('two_ports', 2), ('no_s', 0),
                                         ('other_length', 1)])
def test_other_lumped_s_requests_keep_their_drive_scans(monkeypatch, mode, drives):
    from rfx import simulation
    step, calls = simulation.run, []

    def count(*args, **kwargs):
        calls.append(1)
        return step(*args, **kwargs)

    monkeypatch.setattr(simulation, 'run', count)
    if mode == 'passive':
        from tests.contracts.run_forward_census import base_scene, SOURCE
        sim = base_scene('uniform', source=False).add_port(SOURCE, 'ez', impedance=50., excite=False)
    else:
        sim = BUILDERS['P-lumped2' if mode == 'two_ports' else 'P-lumped']('uniform')
    # A single lumped port asking S for another record length keeps its drive scan.
    options = {'no_s': {'compute_s_params': False}, 'other_length': {'s_param_n_steps': 200}}
    result = sim.run(n_steps=150, skip_preflight=True, **options.get(mode, {}))
    assert len(calls) == 1 + drives
    assert (result.s_params is None) == (mode == 'no_s')


@pytest.mark.parametrize('n_steps', [150, 900])
def test_run_and_forward_judge_each_row_alike(n_steps):
    # Each row's tail starts after the drive plus transit to the records it
    # judges (probes for the probe row, ports for the port row) on both entries.
    # 150 steps: the rows stop at different gates if the starts differ;
    # 900 steps: every row reaches a verdict.
    run = invoke(BUILDERS['P-lumped']('uniform'), 'run', 'uniform', n_steps)
    forward = invoke(BUILDERS['P-lumped']('uniform'), 'forward', 'uniform', n_steps,
                     dict(port_s11_freqs=np.asarray(run.freqs)))
    a, b = run.settling_witness, forward.settling_witness
    assert a['status'] == b['status']
    if n_steps == 900:
        assert a['status'] != 'undetermined'
    assert len(a['runs']) == len(b['runs']) >= 2
    for row_a, row_b in zip(a['runs'], b['runs']):
        assert (row_a['status'], row_a['reason']) == (row_b['status'], row_b['reason'])
        for key in ('share_per_bin', 'error_per_bin'):
            np.testing.assert_array_equal(np.asarray(row_a[key]), np.asarray(row_b[key]))
